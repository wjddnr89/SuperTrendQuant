import time
import asyncio
import pandas as pd
import numpy as np
import json
import os
import requests
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# =====================================================================
# ⚙️ [글로벌 환경 설정]
# =====================================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TOSS_CLIENT_ID = os.getenv("TOSS_CLIENT_ID")
TOSS_CLIENT_SECRET = os.getenv("TOSS_CLIENT_SECRET")

# openapi.json 공식 규격 Base URL
TOSS_BASE_URL = "https://openapi.tossinvest.com" 

KR_MAX_SLOTS = 1              
US_MAX_SLOTS = 1              

HURDLE_ATR_MULT = 1.25      
RS_PERIOD = 130             

FEE_HALF = 0.00225          
SLIPPAGE = 0.0005           

UNIVERSE_FILE = "universe.json"
HOLDING_FILE = "holding.json"

# =====================================================================
# 📦 [Stateful 메모리 관리 레이어]
# =====================================================================
cached_data_30m = {}      # 종목별 야후파이낸스 오리지널 30분봉 데이터프레임
cached_bench_30m = {}     # 벤치마크 오리지널 30분봉 데이터프레임
cached_bench_1h = {}      # 벤치마크 오리지널 1시간봉 데이터프레임
last_candle_base_time = None  # 마지막으로 야후 데이터를 백그라운드 동기화한 30분봉 기준 시간선

# =====================================================================
# 📂 [장부 파일 관리 레이어]
# =====================================================================
def load_holdings():
    if not os.path.exists(HOLDING_FILE):
        return {"KR": {}, "US": {}}
    try:
        with open(HOLDING_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content: return {"KR": {}, "US": {}}
            return json.loads(content)
    except Exception as e:
        print(f"❌ 장부 파일 읽기 오류: {e}")
        return {"KR": {}, "US": {}}

def save_holdings(holdings_data):
    try:
        with open(HOLDING_FILE, "w", encoding="utf-8") as f:
            json.dump(holdings_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ 장부 파일 저장 오류: {e}")

# =====================================================================
# 📢 [통신 및 토스 계좌 관리 레이어]
# =====================================================================
def send_telegram(message):
    print(f"📢 [Telegram Send] {message.replace('*', '')}")
    if not TELEGRAM_TOKEN or "YOUR_" in TELEGRAM_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=5)
    except Exception as e: print(f"❌ 텔레그램 알림 전송 실패: {e}")

class TossAccountManager:
    def __init__(self):
        self.token = None
        self.token_expiry = 0
        self.account_seq = os.getenv("TOSS_ACCOUNT_SEQ", "1")  

    def get_access_token(self):
        """🎯 OAuth2 토큰 발급"""
        if self.token and time.time() < self.token_expiry - 60:
            return self.token

        url = f"{TOSS_BASE_URL}/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        payload = {"grant_type": "client_credentials"}
        
        try:
            res = requests.post(
                url, 
                headers=headers, 
                data=payload, 
                auth=(TOSS_CLIENT_ID, TOSS_CLIENT_SECRET), 
                timeout=5
            )
            if res.status_code == 200:
                rd = res.json()
                self.token = rd.get("access_token")
                expires_in = rd.get("expires_in", 3600)
                self.token_expiry = time.time() + int(expires_in)
                print("🟩 [토큰 발급 성공] 토스 Open API 게이트웨이 인증 수립 완료.")
                return self.token
            else:
                print(f"❌ [토큰 발급 실패] STATUS: {res.status_code} | RESPONSE: {res.text}")
                raise Exception(f"토스 토큰 발급 실패: {res.text}")
        except Exception as e:
            print(f"🚨 인증 레이어 통신 예외 발생: {e}")
            raise e

    def fetch_real_assets(self, target_market="KR"):
        """🎯 실시간 예수금 및 잔고 조회 연동"""
        try:
            token = self.get_access_token()
            if not token: return {}
                
            headers = {
                "Authorization": f"Bearer {token}",
                "X-Tossinvest-Account": str(self.account_seq),
                "Content-Type": "application/json"
            }
            
            # 1. 실제 매수 가능 금액 (예수금) 조회
            buying_power_url = f"{TOSS_BASE_URL}/api/v1/buying-power"
            currency_param = "KRW" if target_market == "KR" else "USD"
            
            bp_res = requests.get(buying_power_url, headers=headers, params={"currency": currency_param}, timeout=5)
            if bp_res.status_code != 200:
                print(f"❌ 예수금 조회 실패 (HTTP {bp_res.status_code}): {bp_res.text}")
                return {}
                
            bp_data = bp_res.json().get("result", {})
            fresh_cash = float(bp_data.get("cashBuyingPower", 0))

            # 2. 실제 주식 잔고 조회
            holdings_url = f"{TOSS_BASE_URL}/api/v1/holdings"
            holdings_res = requests.get(holdings_url, headers=headers, timeout=5)
            if holdings_res.status_code != 200:
                print(f"❌ 주식 잔고 조회 실패 (HTTP {holdings_res.status_code}): {holdings_res.text}")
                return {}
            
            holdings_data = holdings_res.json().get("result", {})
            items_list = holdings_data.get("items", [])
            
            # 3. 봇의 데이터 스펙에 맞게 가공 및 총 자산 추정 계산
            mock_account_data = {
                "summary": {
                    "totalAssetValue": fresh_cash,  
                    "totalCashBalance": fresh_cash
                },
                "stocks": []
            }
            
            total_stock_value = 0
            for item in items_list:
                currency = item.get("currency", "KRW")
                is_kr_stock = (currency == "KRW")
                
                if (target_market == "KR" and is_kr_stock) or (target_market == "US" and not is_kr_stock):
                    qty = int(float(item.get("quantity", 0)))
                    purchase_price = float(item.get("purchasePrice", 0))
                    
                    mock_account_data["stocks"].append({
                        "symbol": item.get("symbol"), 
                        "holdingQuantity": qty,
                        "averagePurchasePrice": purchase_price
                    })
                    total_stock_value += (qty * purchase_price)
            
            mock_account_data["summary"]["totalAssetValue"] += total_stock_value
            
            summary = mock_account_data["summary"]
            fmt = ",.0f" if target_market == "KR" else ",.2f"
            unit = "원" if target_market == "KR" else "달러"
            print(f"💰 [실계좌 동기화 완료] 가용예수금: {summary['totalCashBalance']:{fmt}} {unit} | 총 자산(추정): {summary['totalAssetValue']:{fmt}} {unit}")
            
            return mock_account_data
            
        except Exception as e: 
            print(f"❌ 자산 연동 모듈 파싱 예외 발생: {e}")
            return {}

    def fetch_current_prices(self, symbols_str):
        """🎯 토스 Open API 다건 현재가 일괄 조회 (/api/v1/prices)"""
        try:
            token = self.get_access_token()
            if not token: return None
                
            url = f"{TOSS_BASE_URL}/api/v1/prices"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            params = {"symbols": symbols_str}
            
            res = requests.get(url, headers=headers, params=params, timeout=5)
            if res.status_code == 200:
                return res.json()
            else:
                print(f"⚠️ [토스 현재가 조회 실패] HTTP {res.status_code}: {res.text}")
                return None
        except Exception as e:
            print(f"⚠️ 현재가 API 에러 연동 실패: {e}")
            return None

    def fetch_open_orders(self):
        """🎯 [추적 매수] 토스 Open API 규격에 맞게 결제 대기/미체결 상태인 매수/매도 주문 리스트 조회"""
        try:
            token = self.get_access_token()
            if not token: return []
            
            url = f"{TOSS_BASE_URL}/api/v1/orders"
            headers = {
                "Authorization": f"Bearer {token}",
                "X-Tossinvest-Account": str(self.account_seq),
                "Content-Type": "application/json"
            }
            # 미체결 주문 필터링을 위해 status 파라미터 전달
            params = {"status": "OPEN"}
            
            res = requests.get(url, headers=headers, params=params, timeout=5)
            if res.status_code == 200:
                return res.json().get("result", {}).get("items", [])
            else:
                print(f"⚠️ 미체결 주문 조회 실패 (HTTP {res.status_code}): {res.text}")
                return []
        except Exception as e:
            print(f"❌ 미체결 주문 조회 중 예외 발생: {e}")
            return []

    def cancel_order(self, order_id):
        """🎯 [추적 매수] 특정 미체결 주문을 ID 기반으로 철회/취소"""
        try:
            token = self.get_access_token()
            if not token: return False
            
            url = f"{TOSS_BASE_URL}/api/v1/orders/{order_id}"
            headers = {
                "Authorization": f"Bearer {token}",
                "X-Tossinvest-Account": str(self.account_seq),
                "Content-Type": "application/json"
            }
            
            res = requests.delete(url, headers=headers, timeout=5)
            if res.status_code in [200, 204]:
                print(f"🗑️ [추적 매수] 구형 미체결 주문 취소 완료. ID: {order_id}")
                return True
            else:
                print(f"⚠️ 주문 취소 실패 (HTTP {res.status_code}): {res.text}")
                return False
        except Exception as e:
            print(f"❌ 주문 취소 연동 중 예외 발생: {e}")
            return False

    def send_order(self, symbol, side, qty, order_type="market", price=None):
        """🎯 주문 생성 API"""
        try:
            token = self.get_access_token()
            url = f"{TOSS_BASE_URL}/api/v1/orders"
            
            payload = {
                "symbol": str(symbol),
                "side": "BUY" if side.lower() == "buy" else "SELL",
                "orderType": "MARKET" if order_type.lower() == "market" else "LIMIT",
                "quantity": str(int(qty))
            }
            
            if order_type.lower() == "limit" and price is not None:
                payload["price"] = str(int(price))

            headers = {
                "Authorization": f"Bearer {token}", 
                "X-Tossinvest-Account": str(self.account_seq),
                "Content-Type": "application/json"
            }
            
            res = requests.post(url, headers=headers, json=payload, timeout=5)
            if res.status_code in [200, 201]:
                print(f"✅ 토스 공식 주문 전송 성공: {symbol} | {side} | {qty}주")
                return True
            else:
                print(f"❌ 주문 실패 응답 코드: {res.status_code} | 내용: {res.text}")
                return False
        except Exception as e:
            print(f"❌ 주문 모듈 실행 중 예외: {e}")
            return False

# =====================================================================
# 🛠️ [핵심 알고리즘 계산기]
# =====================================================================
def calculate_supertrend(df, period=7, multiplier=3.0):
    if df.empty or len(df) < period: 
        df['Trend'] = 1
        df['ATR_pct'] = 0.02
        return df
    
    high, low, close = df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze()
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    df['ATR_pct'] = atr / close
    hl2 = (high + low) / 2
    basic_ub, basic_lb = hl2 + (multiplier * atr), hl2 - (multiplier * atr)
    final_ub, final_lb = basic_ub.copy(), basic_lb.copy()
    
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        final_ub.iloc[i] = basic_ub.iloc[i] if basic_ub.iloc[i] < final_ub.iloc[i-1] or close.iloc[i-1] > final_ub.iloc[i-1] else final_ub.iloc[i-1]
        final_lb.iloc[i] = basic_lb.iloc[i] if basic_lb.iloc[i] > final_lb.iloc[i-1] or close.iloc[i-1] < final_lb.iloc[i-1] else final_lb.iloc[i-1]
            
    for i in range(1, len(df)):
        if trend.iloc[i-1] == 1:
            trend.iloc[i] = -1 if close.iloc[i] < final_lb.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if close.iloc[i] > final_ub.iloc[i] else -1
            
    df['Trend'] = trend
    return df

def load_universe():
    default_kr = {"005930": "KOSPI", "091990": "KOSDAQ", "403870": "KOSDAQ"} 
    default_us = ["QQQ", "SOXL"]
    if not os.path.exists(UNIVERSE_FILE): return default_kr, default_us
    try:
        with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("KR_UNIVERSE_MAP", default_kr), data.get("US_UNIVERSE_LIST", default_us)
    except: return default_kr, default_us

# =====================================================================
# 🤖 [실전 하이브리드 자동매매 봇 클래스]
# =====================================================================
class HybridTradingBot:
    def __init__(self, account_mgr):
        self.account_mgr = account_mgr
        self.current_market = None
        self.last_briefing_date = {"KR": None, "US": None}

    def check_market_schedule(self):
        now = datetime.now()
        weekday = now.weekday()
        if weekday >= 5: return "SLEEP"
        time_str = now.strftime("%H:%M:%S")
        if time_str[:5] == "15:30": return "KR_CLOSE"
        if time_str[:5] == "05:00": return "US_CLOSE"
        
        if "09:00:00" <= time_str < "15:30:00": return "KR"
        elif "22:30:00" <= time_str or time_str < "05:00:00": return "US"
        else: return "SLEEP"

    def convert_to_yf_ticker(self, symbol, market_type=None):
        if market_type == "KOSPI": return f"{symbol}.KS"
        if market_type == "KOSDAQ": return f"{symbol}.KQ"
        return symbol

    def sync_yahoo_finance_state(self, yf_tickers, benchmarks):
        """🎯 30분에 단 1번만 백그라운드 호출되어 전 종목 과거 데이터를 메모리에 통째로 새로고침(Sync)합니다."""
        global cached_data_30m, cached_bench_30m, cached_bench_1h
        print("📥 [Stateful 동기화] 30분봉 경계 진입 - 야후 파이낸스 기본 축적 데이터 일괄 갱신 시작...")
        try:
            raw_30m = yf.download(yf_tickers + benchmarks, period="30d", interval="30m", progress=False)
            raw_1h = yf.download(benchmarks, period="30d", interval="1h", progress=False)
            
            # 1. 벤치마크 30분봉/1시간봉 메모리 적재
            for b in benchmarks:
                if b in raw_30m['Close']:
                    cached_bench_30m[b] = raw_30m['Close'][b].dropna().copy()
                if b in raw_1h['Open']:
                    cached_bench_1h[b] = pd.DataFrame({
                        'Open': raw_1h['Open'][b], 'High': raw_1h['High'][b], 
                        'Low': raw_1h['Low'][b], 'Close': raw_1h['Close'][b]
                    }).dropna().copy()
            
            # 2. 유니버스 종목별 30분봉 메모리 적재
            for yf_tk in yf_tickers:
                if yf_tk in raw_30m['Close']:
                    df_ticker = pd.DataFrame({
                        'Open': raw_30m['Open'][yf_tk], 'High': raw_30m['High'][yf_tk],
                        'Low': raw_30m['Low'][yf_tk], 'Close': raw_30m['Close'][yf_tk]
                    }).dropna()
                    cached_data_30m[yf_tk] = df_ticker.copy()
            
            print("🟩 [Stateful 동기화 완료] 야후 파이낸스 캐시 갱신이 안전하게 마감되었습니다.")
            return True
        except Exception as e:
            print(f"❌ [Stateful 동기화 실패] 야후 파이낸스 통신 오류: {e}")
            return False

    async def execute_trade_cycle(self):
        global cached_data_30m, cached_bench_30m, cached_bench_1h, last_candle_base_time
        
        kr_universe_map, us_universe_list = load_universe()
        self.current_market = self.check_market_schedule()
        now = datetime.now()
        
        print(f"\n🔄 [통합 스캔 시작] {now.strftime('%Y-%m-%d %H:%M:%S')} | 마켓컨텍스트: {self.current_market}")
        if self.current_market == "SLEEP": 
            print("💤 장외 시간입니다. 휴식 중...")
            return

        is_close_briefing = "CLOSE" in self.current_market
        target_market = self.current_market.replace("_CLOSE", "")

        # 계좌 잔고 파싱 레이어
        account_data = self.account_mgr.fetch_real_assets(target_market=target_market)
        if not account_data:
            print("⚠️ 계좌 데이터를 가공하지 못해 다음 스캔 사이클로 롤오버합니다.")
            return

        summary = account_data.get("summary", {})
        total_account_value = float(summary.get("totalAssetValue", 0))
        fresh_cash = float(summary.get("totalCashBalance", 0))
        
        real_holding_stocks = {}
        for s in account_data.get("stocks", []):
            sym = s.get("symbol")
            if sym:
                real_holding_stocks[sym] = {
                    "qty": int(s.get("holdingQuantity", 0)),
                    "entry_price": float(s.get("averagePurchasePrice", 0))
                }

        holdings_master = load_holdings()
        market_holdings = holdings_master[target_market]

        for t in list(market_holdings.keys()):
            if t not in real_holding_stocks or real_holding_stocks[t]['qty'] == 0:
                del market_holdings[t]
        
        current_universe_keys = list(kr_universe_map.keys()) if target_market == "KR" else us_universe_list
        for t in list(real_holding_stocks.keys()):
            if real_holding_stocks[t]['qty'] > 0 and t in current_universe_keys:
                market_holdings[t] = {
                    "qty": real_holding_stocks[t]['qty'],
                    "entry_price": real_holding_stocks[t]['entry_price']
                }
                
        holdings_master[target_market] = market_holdings
        save_holdings(holdings_master)
        bot_holding_stocks = market_holdings

        benchmarks = ["^KS11", "^KQ11", "QQQ"]
        if target_market == "KR":
            yf_tickers = [self.convert_to_yf_ticker(sym, mkt) for sym, mkt in kr_universe_map.items()]
            active_max_slots = KR_MAX_SLOTS
        else:
            yf_tickers = us_universe_list
            active_max_slots = US_MAX_SLOTS

        # -----------------------------------------------------------------
        # ⏱️ [구조 고도화 제어 1] 30분 타임프레임 흐름 감시 및 야후 동기화 트리거
        # -----------------------------------------------------------------
        current_candle_base = now.replace(minute=(now.minute // 30) * 30, second=0, microsecond=0)
        
        if last_candle_base_time is None or last_candle_base_time != current_candle_base:
            success = self.sync_yahoo_finance_state(yf_tickers, benchmarks)
            if success:
                last_candle_base_time = current_candle_base
            else:
                if not cached_data_30m:
                    print("⚠️ 기존 로컬 캐시가 완전히 비어있어 이번 사이클을 진행할 수 없습니다.")
                    return
                print("⚠️ 야후 동기화에 실패하여 메모리에 백업된 직전 캐시 데이터로 감시를 계속 진행합니다.")

        # -----------------------------------------------------------------
        # 🌟 토스 Open API 다건 현재가 일괄 동기화 (Batch)
        # -----------------------------------------------------------------
        realtime_prices = {}
        if current_universe_keys:
            symbols_str = ",".join(current_universe_keys)
            try:
                price_res = self.account_mgr.fetch_current_prices(symbols_str)
                if price_res and 'result' in price_res:
                    for item in price_res['result']:
                        sym = item.get('symbol')
                        if sym and item.get('lastPrice'):
                            realtime_prices[sym] = float(item['lastPrice'])
                    print(f"💰 [토스 실시간 시세 조회] {len(realtime_prices)}개 종목 연동 완료.")
            except Exception as e:
                print(f"⚠️ 토스 현재가 일괄 조회 실패 (지연 데이터로 자동 대체됩니다): {e}")

        # -----------------------------------------------------------------
        # 🛡️ [추적 매수 제어 파이프라인] 미체결 대기 주문 모니터링 및 실시간 정정 취소
        # -----------------------------------------------------------------
        existing_order_symbols = []  # 아래 매매 연산에서 제외시킬 대기용 컨테이너
        if not is_close_briefing:
            open_orders = self.account_mgr.fetch_open_orders()
            for order in open_orders:
                sym = order.get("symbol")
                order_id = order.get("orderId")
                side = order.get("side", "").lower()

                # 매도 미체결 주문은 이미 거래소 자산 잠금이 발생했으므로 봇에서 스킵처리
                if side == "sell":
                    existing_order_symbols.append(sym)
                    continue

                # 매수 미체결 주문 추적 처리
                if side == "buy" and sym in realtime_prices:
                    current_price = realtime_prices[sym]
                    order_price = float(order.get("price", 0))

                    # 실시간 시장가격이 이전 지정가 주문 대비 0.5% 초과하여 멀어졌다면 취소 처리
                    if order_price > 0 and (current_price - order_price) / order_price > 0.005:
                        print(f"🔄 [추적 매수] 미체결 매수 발견 ({sym}): 기존 지정가 {order_price} -> 현재 실시간가 {current_price}. 기존 주문 취소 후 다음 사이클 재진입을 대기합니다.")
                        self.account_mgr.cancel_order(order_id)
                        # 주문을 취소했으므로 이번 루프의 매수 룰 레이어에서 새롭게 잡힐 수 있습니다.
                    else:
                        # 아직 현재가와 괴리가 적은 주문은 유효하므로 유지하며, 중복 매수 주문 투하를 방지하기 위해 락 처리
                        existing_order_symbols.append(sym)

# -----------------------------------------------------------------
        # ⏱️ [구조 고도화 제어 2] 메모리 캐시로부터 복사본 추출 및 지표 연산
        # -----------------------------------------------------------------
        bench_ret_30m = {}
        upper_trends_1h = {}
        for b in benchmarks:
            if b not in cached_bench_30m: continue
            b_close = cached_bench_30m[b]
            if len(b_close) >= RS_PERIOD:
                bench_ret_30m[b] = b_close.pct_change(RS_PERIOD)
            
            if b not in cached_bench_1h: continue
            df_b = cached_bench_1h[b].copy()  
            if not df_b.empty:
                upper_trends_1h[b] = calculate_supertrend(df_b, period=10, multiplier=3.0)

        all_candidates = []
        processed_data = {}

        scan_source = kr_universe_map.items() if target_market == "KR" else [(u, "US") for u in us_universe_list]
        for symbol, mkt_type in scan_source:
            yf_tk = self.convert_to_yf_ticker(symbol, mkt_type)
            if yf_tk not in cached_data_30m: continue
            
            df = cached_data_30m[yf_tk].copy()
            if len(df) < (RS_PERIOD + 5): continue

            # 🌟 하이브리드 데이터 실시간 스티칭 (Data Stitching) 레이어
            if symbol in realtime_prices:
                toss_price = realtime_prices[symbol]
                df.iloc[-1, df.columns.get_loc('Close')] = toss_price
                if toss_price > df.iloc[-1, df.columns.get_loc('High')]:
                    df.iloc[-1, df.columns.get_loc('High')] = toss_price
                if toss_price < df.iloc[-1, df.columns.get_loc('Low')]:
                    df.iloc[-1, df.columns.get_loc('Low')] = toss_price

            bench_symbol = "^KS11" if mkt_type == "KOSPI" else ("^KQ11" if mkt_type == "KOSDAQ" else "QQQ")
            market_signal = upper_trends_1h[bench_symbol]['Trend'].iloc[-1] if bench_symbol in upper_trends_1h else -1
            
            df = calculate_supertrend(df, period=7, multiplier=4.5 if symbol in ["SOXL", "SOXS"] else 3.0)
            if bench_symbol in bench_ret_30m:
                df['RS'] = df['Close'].pct_change(RS_PERIOD) - bench_ret_30m[bench_symbol]
            else: continue
            
            df = df.dropna()
            processed_data[symbol] = df

            if df['Trend'].iloc[-1] == 1 and market_signal == 1:
                all_candidates.append({
                    'ticker': symbol, 
                    'rs': df['RS'].iloc[-1], 
                    'price': df['Close'].iloc[-1], 
                    'atr_pct': df['ATR_pct'].iloc[-1]
                })

        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)

        if is_close_briefing:
            today_str = now.strftime("%Y-%m-%d")
            if self.last_briefing_date[target_market] != today_str:
                market_name = "국내 주식" if target_market == "KR" else "해외 주식"
                pos_info = "\n• ".join([f"{tk} ({pos['qty']}주)" for tk, pos in bot_holding_stocks.items()]) if bot_holding_stocks else "보유 없음"
                briefing_msg = f"🏁 *[{market_name} 마감]*\n• 총자산: {total_account_value:,.0f}\n• 예수금: {fresh_cash:,.0f}\n• 포지션:\n• {pos_info}"
                send_telegram(briefing_msg)
                self.last_briefing_date[target_market] = today_str
            return

        # =====================================================================
        # ⚡ [실전 매매 제어부]
        # =====================================================================
        has_action = False

        # 매도 룰 (이미 수동/자동 매도 접수 중인 종목은 제외)
        for t in list(bot_holding_stocks.keys()):
            if t not in processed_data or t in existing_order_symbols: continue
            if processed_data[t]['Trend'].iloc[-1] == -1:
                order_qty = bot_holding_stocks[t]['qty']
                success = self.account_mgr.send_order(symbol=t, side="sell", qty=order_qty, order_type="market")
                if success:
                    send_telegram(f"🚨 *[추세 이탈 매도]*\n• 종목: {t} | {order_qty}주 전량 청산")
                    del market_holdings[t]
                    holdings_master[target_market] = market_holdings
                    save_holdings(holdings_master)
                    has_action = True

        # 순환매 룰
        if all_candidates and bot_holding_stocks:
            for t in list(bot_holding_stocks.keys()):
                if t not in processed_data or t in existing_order_symbols: continue
                current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
                available_news = [c for c in all_candidates if c['ticker'] not in bot_holding_stocks and c['ticker'] not in existing_order_symbols]
                if not available_news: continue
                best_new = available_news[0]
                
                if best_new['rs'] - current_rs > (best_new['atr_pct'] * HURDLE_ATR_MULT):
                    old_qty = bot_holding_stocks[t]['qty']
                    sell_success = self.account_mgr.send_order(symbol=t, side="sell", qty=old_qty, order_type="market")
                    if sell_success:
                        send_telegram(f"🔄 *[순환매 교체]*\n• 주도주 교체로 {t} 매도 완료")
                        del market_holdings[t]
                        holdings_master[target_market] = market_holdings
                        save_holdings(holdings_master)
                        has_action = True

        # 매수 룰 (미체결 상태로 정상 대기 중인 종목은 existing_order_symbols 필터에 의해 진입 차단)
        for candidate in all_candidates:
            if len(bot_holding_stocks) >= active_max_slots: break
            t = candidate['ticker']
            
            if t not in bot_holding_stocks and t not in existing_order_symbols:
                price = candidate['price']
                target_allocation = total_account_value * 0.90  
                alloc_money = min(fresh_cash, target_allocation)
                qty = int(alloc_money // (price * (1 + SLIPPAGE) * (1 + FEE_HALF)))
                
                if qty > 0:
                    if target_market == "KR":
                        success = self.account_mgr.send_order(symbol=t, side="buy", qty=qty, order_type="limit", price=price)
                    else:
                        success = self.account_mgr.send_order(symbol=t, side="buy", qty=qty, order_type="market")
                    
                    if success:
                        send_telegram(f"🟩 *[추세 주도주 진입]*\n• 종목: {t} | 수량: {qty}주 매수 집행")
                        market_holdings[t] = {"qty": qty, "entry_price": price}
                        holdings_master[target_market] = market_holdings
                        save_holdings(holdings_master)
                        has_action = True
                    break

        if not has_action:
            print("🔍 [대기] 현재 포지션을 유지하며 추세를 관망합니다.")

    async def start_trading_bot(self):
        send_telegram("🤖 *Ver 23.0 추적 매수 모듈 전체 병합 완료!*")
        while True:
            try: 
                await self.execute_trade_cycle()
            except Exception as e: 
                print(f"🚨 엔진 내부 예외 발생: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    mgr = TossAccountManager()
    bot = HybridTradingBot(mgr)
    asyncio.run(bot.start_trading_bot())