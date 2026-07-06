import time
import asyncio
import pandas as pd
import numpy as np
import json
import os
import requests
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
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

TOSS_BASE_URL = "https://openapi.tossinvest.com" 

KR_MAX_SLOTS = 1              
US_MAX_SLOTS = 1              

HURDLE_ATR_MULT = 1.25       
KR_RS_PERIOD = 100          
US_RS_PERIOD = 130          

FEE_HALF = 0.00225          
SLIPPAGE = 0.0005           

UNIVERSE_FILE = "universe.json"
HOLDING_FILE = "holding.json"

# =====================================================================
# 📦 [Stateful 메모리 관리 레이어]
# =====================================================================
cached_data_30m = {}      
cached_bench_30m = {}     
cached_bench_1h = {}      
last_candle_base_time = None  
missing_data_30m = {}

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
            data = json.loads(content)
            if not isinstance(data, dict):
                return {"KR": {}, "US": {}}
            return {
                "KR": data.get("KR") if isinstance(data.get("KR"), dict) else {},
                "US": data.get("US") if isinstance(data.get("US"), dict) else {}
            }
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
        if self.token and time.time() < self.token_expiry - 60:
            return self.token
        url = f"{TOSS_BASE_URL}/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        payload = {"grant_type": "client_credentials"}
        try:
            res = requests.post(url, headers=headers, data=payload, auth=(TOSS_CLIENT_ID, TOSS_CLIENT_SECRET), timeout=5)
            if res.status_code == 200:
                rd = res.json()
                self.token = rd.get("access_token")
                self.token_expiry = time.time() + int(rd.get("expires_in", 3600))
                return self.token
            else:
                raise Exception(f"토스 토큰 발급 실패: {res.text}")
        except Exception as e:
            raise e

    def fetch_real_assets(self, target_market="KR"):
        try:
            token = self.get_access_token()
            if not token: return {}
            headers = {"Authorization": f"Bearer {token}", "X-Tossinvest-Account": str(self.account_seq), "Content-Type": "application/json"}
            buying_power_url = f"{TOSS_BASE_URL}/api/v1/buying-power"
            currency_param = "KRW" if target_market == "KR" else "USD"
            bp_res = requests.get(buying_power_url, headers=headers, params={"currency": currency_param}, timeout=5)
            if bp_res.status_code != 200:
                print(f"❌ 매수가능금액 조회 실패: {bp_res.status_code} | {bp_res.text}")
                return {}
            fresh_cash = float(bp_res.json().get("result", {}).get("cashBuyingPower", 0))

            holdings_url = f"{TOSS_BASE_URL}/api/v1/holdings"
            holdings_res = requests.get(holdings_url, headers=headers, timeout=5)
            if holdings_res.status_code != 200:
                print(f"❌ 보유종목 조회 실패: {holdings_res.status_code} | {holdings_res.text}")
                return {}
            items_list = holdings_res.json().get("result", {}).get("items", [])
            
            # 🎯 깔끔한 일원화: 토스 공식 명세서 규격 필드명(quantity, purchasePrice) 그대로 전달하도록 고정
            mock_account_data = {"summary": {"totalAssetValue": fresh_cash, "totalCashBalance": fresh_cash}, "stocks": []}
            total_stock_value = 0
            for item in items_list:
                currency = item.get("currency", "KRW")
                if (target_market == "KR" and currency == "KRW") or (target_market == "US" and currency != "KRW"):
                    qty = int(float(item.get("quantity", 0)))
                    purchase_price = float(item.get("purchasePrice", 0))
                    
                    mock_account_data["stocks"].append({
                        "symbol": item.get("symbol"), 
                        "quantity": qty, 
                        "purchasePrice": purchase_price
                    })
                    # TODO: totalAssetValue는 현재가/평가금액 필드가 확인되면 매수단가가 아니라 평가금액 기준으로 바꿔야 합니다.
                    total_stock_value += (qty * purchase_price)
            mock_account_data["summary"]["totalAssetValue"] += total_stock_value
            return mock_account_data
        except Exception as e: 
            print(f"❌ 실계좌 조회 예외: {e}")
            return {}

    def fetch_current_prices(self, symbols_str):
        try:
            token = self.get_access_token()
            if not token: return None
            res = requests.get(f"{TOSS_BASE_URL}/api/v1/prices", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, params={"symbols": symbols_str}, timeout=5)
            if res.status_code == 200:
                return res.json()
            print(f"❌ 현재가 조회 실패: {res.status_code} | {res.text}")
            return None
        except Exception as e:
            print(f"❌ 현재가 조회 예외: {e}")
            return None

    def fetch_open_orders(self):
        try:
            token = self.get_access_token()
            if not token: return []
            res = requests.get(f"{TOSS_BASE_URL}/api/v1/orders", headers={"Authorization": f"Bearer {token}", "X-Tossinvest-Account": str(self.account_seq), "Content-Type": "application/json"}, params={"status": "OPEN"}, timeout=5)
            if res.status_code == 200:
                return res.json().get("result", {}).get("items", [])
            print(f"❌ 미체결 주문 조회 실패: {res.status_code} | {res.text}")
            return []
        except Exception as e:
            print(f"❌ 미체결 주문 조회 예외: {e}")
            return []

    def cancel_order(self, order_id):
        try:
            token = self.get_access_token()
            if not token: return False
            res = requests.delete(f"{TOSS_BASE_URL}/api/v1/orders/{order_id}", headers={"Authorization": f"Bearer {token}", "X-Tossinvest-Account": str(self.account_seq), "Content-Type": "application/json"}, timeout=5)
            if res.status_code in [200, 204]:
                return True
            print(f"❌ 주문 취소 실패: {order_id} | {res.status_code} | {res.text}")
            return False
        except Exception as e:
            print(f"❌ 주문 취소 예외: {order_id} | {e}")
            return False

    def send_order(self, symbol, side, qty, order_type="market", price=None):
        try:
            token = self.get_access_token()
            payload = {"symbol": str(symbol), "side": "BUY" if side.lower() == "buy" else "SELL", "orderType": "MARKET" if order_type.lower() == "market" else "LIMIT", "quantity": str(int(qty))}
            if order_type.lower() == "limit" and price is not None: payload["price"] = str(int(price))
            res = requests.post(f"{TOSS_BASE_URL}/api/v1/orders", headers={"Authorization": f"Bearer {token}", "X-Tossinvest-Account": str(self.account_seq), "Content-Type": "application/json"}, json=payload, timeout=5)
            if res.status_code in [200, 201]:
                return True
            print(f"❌ 주문 실패: {symbol} {side} {qty}주 | {res.status_code} | {res.text}")
            return False
        except Exception as e:
            print(f"❌ 주문 예외: {symbol} {side} {qty}주 | {e}")
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
        if trend.iloc[i-1] == 1: trend.iloc[i] = -1 if close.iloc[i] < final_lb.iloc[i] else 1
        else: trend.iloc[i] = 1 if close.iloc[i] > final_ub.iloc[i] else -1
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
        kr_now = datetime.now(ZoneInfo("Asia/Seoul"))
        us_now = datetime.now(ZoneInfo("America/New_York"))

        kr_time = kr_now.strftime("%H:%M:%S")
        us_time = us_now.strftime("%H:%M:%S")

        if kr_now.weekday() <= 4 and kr_time[:5] == "15:30":
            return "KR_CLOSE"
        if us_now.weekday() <= 4 and us_time[:5] == "16:00":
            return "US_CLOSE"
        if kr_now.weekday() <= 4 and "09:00:00" <= kr_time < "15:30:00":
            return "KR"
        if us_now.weekday() <= 4 and "09:30:00" <= us_time < "16:00:00":
            return "US"
        return "SLEEP"

    def convert_to_yf_ticker(self, symbol, market_type=None):
        if market_type == "KOSPI": return f"{symbol}.KS"
        if market_type == "KOSDAQ": return f"{symbol}.KQ"
        return symbol

    def sync_yahoo_finance_state(self, yf_tickers, benchmarks, current_candle_base=None):
        global cached_data_30m, cached_bench_30m, cached_bench_1h
        print("📥 [Stateful 동기화] 30분봉 데이터프레임 일괄 갱신 시작...")
        try:
            raw_30m = yf.download(yf_tickers + benchmarks, period="30d", interval="30m", progress=False)
            raw_1h = yf.download(benchmarks, period="30d", interval="1h", progress=False)
            for b in benchmarks:
                if b in raw_30m['Close']: cached_bench_30m[b] = raw_30m['Close'][b].dropna().copy()
                if b in raw_1h['Open']: cached_bench_1h[b] = pd.DataFrame({'Open': raw_1h['Open'][b], 'High': raw_1h['High'][b], 'Low': raw_1h['Low'][b], 'Close': raw_1h['Close'][b]}).dropna().copy()
            for yf_tk in yf_tickers:
                if yf_tk in raw_30m['Close']:
                    df_raw = pd.DataFrame({'Open': raw_30m['Open'][yf_tk], 'High': raw_30m['High'][yf_tk], 'Low': raw_30m['Low'][yf_tk], 'Close': raw_30m['Close'][yf_tk]}).dropna().copy()
                    if df_raw.empty:
                        continue

                    bench_key = "^KS11" if yf_tk.endswith(".KS") else ("^KQ11" if yf_tk.endswith(".KQ") else "QQQ")
                    if current_candle_base is not None and bench_key in raw_30m['Close']:
                        bench_index = raw_30m['Close'][bench_key].dropna().index
                        ffill_until = pd.Timestamp(current_candle_base) - pd.Timedelta(minutes=30)
                        historical_index = bench_index[bench_index <= ffill_until]
                        actual_index = df_raw.index[df_raw.index > ffill_until]
                        combined_index = historical_index.union(actual_index)
                        if not historical_index.empty:
                            # TODO: 필요하면 has_real_bar 컬럼을 추가해 ffill로 채운 봉과 실제 봉을 구분합니다.
                            df_raw = df_raw.reindex(combined_index).ffill().dropna().copy()

                    cached_data_30m[yf_tk] = df_raw
            print("🟩 [Stateful 동기화 완료] 캐시 갱신 마감.")
            return True
        except Exception as e:
            print(f"❌ 야후 파이낸스 통신 오류: {e}")
            return False

    def sync_missing_yahoo_tickers(self, retry_tickers, market_tz, current_candle_base):
        global cached_data_30m
        if not retry_tickers:
            return []

        print(f"🔁 [개별 재시도] 최신 30분봉 누락 종목 {len(retry_tickers)}개 재다운로드: {', '.join(retry_tickers)}")
        refreshed = []
        try:
            raw_30m = yf.download(retry_tickers, period="30d", interval="30m", progress=False)
            for yf_tk in retry_tickers:
                try:
                    if isinstance(raw_30m.columns, pd.MultiIndex):
                        if yf_tk not in raw_30m['Close']:
                            continue
                        df = pd.DataFrame({
                            'Open': raw_30m['Open'][yf_tk],
                            'High': raw_30m['High'][yf_tk],
                            'Low': raw_30m['Low'][yf_tk],
                            'Close': raw_30m['Close'][yf_tk]
                        }).dropna().copy()
                    else:
                        if 'Close' not in raw_30m:
                            continue
                        df = pd.DataFrame({
                            'Open': raw_30m['Open'],
                            'High': raw_30m['High'],
                            'Low': raw_30m['Low'],
                            'Close': raw_30m['Close']
                        }).dropna().copy()

                    if df.empty:
                        continue

                    latest_ts = pd.Timestamp(df.index[-1])
                    if latest_ts.tzinfo is not None:
                        latest_ts = latest_ts.tz_convert(market_tz)
                    else:
                        latest_ts = latest_ts.tz_localize(market_tz)
                    latest_ts = latest_ts.replace(second=0, microsecond=0)

                    if latest_ts >= pd.Timestamp(current_candle_base):
                        cached_data_30m[yf_tk] = df
                        refreshed.append(yf_tk)
                except Exception as e:
                    print(f"⚠️ {yf_tk} 개별 30분봉 재시도 처리 실패: {e}")
            return refreshed
        except Exception as e:
            print(f"❌ 개별 종목 30분봉 재시도 실패: {e}")
            return []

    async def execute_trade_cycle(self):
        global cached_data_30m, cached_bench_30m, cached_bench_1h, last_candle_base_time, missing_data_30m
        
        kr_universe_map, us_universe_list = load_universe()
        self.current_market = self.check_market_schedule()
        now = datetime.now()
        
        print(f"\n🔄 [통합 스캔 시작] {now.strftime('%Y-%m-%d %H:%M:%S')} | 마켓컨텍스트: {self.current_market}")
        if self.current_market == "SLEEP": return

        is_close_briefing = "CLOSE" in self.current_market
        target_market = self.current_market.replace("_CLOSE", "")
        current_rs_period = KR_RS_PERIOD if target_market == "KR" else US_RS_PERIOD

        account_data = self.account_mgr.fetch_real_assets(target_market=target_market)
        if not account_data: return

        summary = account_data.get("summary", {})
        total_account_value = float(summary.get("totalAssetValue", 0))
        fresh_cash = float(summary.get("totalCashBalance", 0))
        
        # 🎯 정석 필드명(quantity, purchasePrice) 조전 동기화 완료
        real_holding_stocks = {}
        for s in account_data.get("stocks", []):
            sym = s.get("symbol")
            if sym: 
                real_holding_stocks[sym] = {
                    "qty": int(s.get("quantity", 0)),
                    "buy_price": float(s.get("purchasePrice", 0))
                }

        holdings_master = load_holdings()
        market_holdings = holdings_master[target_market]

        for t in list(market_holdings.keys()):
            if t not in real_holding_stocks or real_holding_stocks[t]['qty'] == 0: del market_holdings[t]
        
        # 🛡️ 장부 보존 레이어
        current_universe_keys = list(kr_universe_map.keys()) if target_market == "KR" else us_universe_list
        for t in list(real_holding_stocks.keys()):
            if real_holding_stocks[t]['qty'] > 0 and t in current_universe_keys:
                api_price = real_holding_stocks[t]['buy_price']
                existing_price = market_holdings.get(t, {}).get('buy_price', 0)
                
                if api_price <= 0 and existing_price > 0:
                    final_buy_price = existing_price
                else:
                    final_buy_price = api_price if api_price > 0 else existing_price
                    
                market_holdings[t] = {"qty": real_holding_stocks[t]['qty'], "buy_price": final_buy_price}
                
        holdings_master[target_market] = market_holdings
        save_holdings(holdings_master)
        bot_holding_stocks = market_holdings

        def sync_holdings_after_order(reason):
            refreshed_account = self.account_mgr.fetch_real_assets(target_market=target_market)
            if not refreshed_account:
                print(f"⚠️ {reason}: 실계좌 재조회 실패로 장부를 변경하지 않습니다.")
                return False

            refreshed_real_holding_stocks = {}
            for s in refreshed_account.get("stocks", []):
                sym = s.get("symbol")
                if sym:
                    refreshed_real_holding_stocks[sym] = {
                        "qty": int(s.get("quantity", 0)),
                        "buy_price": float(s.get("purchasePrice", 0))
                    }

            synced_market_holdings = {}
            for t, pos in refreshed_real_holding_stocks.items():
                if pos["qty"] > 0 and t in current_universe_keys:
                    api_price = pos["buy_price"]
                    existing_price = market_holdings.get(t, {}).get("buy_price", 0)
                    final_buy_price = existing_price if api_price <= 0 and existing_price > 0 else (api_price if api_price > 0 else existing_price)
                    synced_market_holdings[t] = {"qty": pos["qty"], "buy_price": final_buy_price}

            market_holdings.clear()
            market_holdings.update(synced_market_holdings)
            holdings_master[target_market] = market_holdings
            save_holdings(holdings_master)
            print(f"✅ {reason}: 실계좌 기준으로 장부를 동기화했습니다.")
            return True

        benchmarks = ["^KS11", "^KQ11", "QQQ"]
        yf_tickers = [self.convert_to_yf_ticker(sym, mkt) for sym, mkt in kr_universe_map.items()] if target_market == "KR" else us_universe_list
        active_max_slots = KR_MAX_SLOTS if target_market == "KR" else US_MAX_SLOTS

        market_tz = ZoneInfo("Asia/Seoul") if target_market == "KR" else ZoneInfo("America/New_York")
        market_now = datetime.now(market_tz)
        current_candle_base = market_now.replace(minute=(market_now.minute // 30) * 30, second=0, microsecond=0)
        fresh_benchmarks = ["^KS11", "^KQ11"] if target_market == "KR" else ["QQQ"]

        if last_candle_base_time is None or last_candle_base_time != current_candle_base:
            if self.sync_yahoo_finance_state(yf_tickers, benchmarks, current_candle_base):
                has_fresh_data = True
                for bench in fresh_benchmarks:
                    if bench not in cached_bench_30m or cached_bench_30m[bench].empty:
                        has_fresh_data = False
                        break
                    latest_ts = pd.Timestamp(cached_bench_30m[bench].index[-1])
                    if latest_ts.tzinfo is not None:
                        latest_ts = latest_ts.tz_convert(market_tz)
                    else:
                        latest_ts = latest_ts.tz_localize(market_tz)
                    latest_ts = latest_ts.replace(second=0, microsecond=0)
                    if latest_ts < pd.Timestamp(current_candle_base):
                        has_fresh_data = False
                        break

                if has_fresh_data:
                    last_candle_base_time = current_candle_base
                else:
                    print("⏳ 야후 최신 30분봉 지연 감지: 다음 사이클에서 재시도합니다.")

        for yf_tk, target_base in list(missing_data_30m.items()):
            if target_base != current_candle_base:
                del missing_data_30m[yf_tk]

        retry_tickers = list(missing_data_30m.keys())
        if retry_tickers:
            refreshed_tickers = self.sync_missing_yahoo_tickers(retry_tickers, market_tz, current_candle_base)
            for yf_tk in refreshed_tickers:
                missing_data_30m.pop(yf_tk, None)

        realtime_prices = {}
        if current_universe_keys:
            symbols_str = ",".join(current_universe_keys)
            price_res = self.account_mgr.fetch_current_prices(symbols_str)
            if price_res and 'result' in price_res:
                for item in price_res['result']:
                    sym = item.get('symbol')
                    if sym and item.get('lastPrice'): realtime_prices[sym] = float(item['lastPrice'])

        existing_order_symbols = []  
        if not is_close_briefing:
            for order in self.account_mgr.fetch_open_orders():
                sym = order.get("symbol")
                if order.get("side", "").lower() == "sell": existing_order_symbols.append(sym)
                elif sym in realtime_prices: existing_order_symbols.append(sym)

        bench_ret_30m = {}
        upper_trends_1h = {}
        for b in benchmarks:
            if b not in cached_bench_30m: continue
            if len(cached_bench_30m[b]) >= current_rs_period: bench_ret_30m[b] = cached_bench_30m[b].pct_change(current_rs_period)
            if b in cached_bench_1h and not cached_bench_1h[b].empty:
                latest_1h_ts = pd.Timestamp(cached_bench_1h[b].index[-1])
                if latest_1h_ts.tzinfo is not None:
                    latest_1h_ts = latest_1h_ts.tz_convert(market_tz)
                else:
                    latest_1h_ts = latest_1h_ts.tz_localize(market_tz)
                current_1h_base = market_now.replace(minute=0, second=0, microsecond=0)
                if latest_1h_ts.replace(minute=0, second=0, microsecond=0) >= pd.Timestamp(current_1h_base):
                    upper_trends_1h[b] = calculate_supertrend(cached_bench_1h[b].copy(), period=10, multiplier=3.0)

        all_candidates = []
        processed_data = {}
        skipped_stale_tickers = []
        scan_source = kr_universe_map.items() if target_market == "KR" else [(u, "US") for u in us_universe_list]
        
        for symbol, mkt_type in scan_source:
            yf_tk = self.convert_to_yf_ticker(symbol, mkt_type)
            if yf_tk not in cached_data_30m:
                missing_data_30m[yf_tk] = current_candle_base
                skipped_stale_tickers.append(symbol)
                continue
            df_orig = cached_data_30m[yf_tk].copy()
            if len(df_orig) < (current_rs_period + 5): continue

            latest_stock_ts = pd.Timestamp(df_orig.index[-1])
            if latest_stock_ts.tzinfo is not None:
                latest_stock_ts = latest_stock_ts.tz_convert(market_tz)
            else:
                latest_stock_ts = latest_stock_ts.tz_localize(market_tz)
            latest_stock_ts = latest_stock_ts.replace(second=0, microsecond=0)
            if latest_stock_ts < pd.Timestamp(current_candle_base):
                missing_data_30m[yf_tk] = current_candle_base
                skipped_stale_tickers.append(symbol)
                continue
            missing_data_30m.pop(yf_tk, None)

            # 실시간 현재가를 30분봉 데이터에 섞지 않습니다.
            # 아래 방식은 +1초짜리 임시 행을 만들어 RS 계산 후 dropna에서 제거될 가능성이 큽니다.
            # if symbol in realtime_prices:
            #     toss_price = realtime_prices[symbol]
            #     df_orig.loc[df_orig.index[-1] + pd.Timedelta(seconds=1)] = [toss_price, toss_price, toss_price, toss_price]
            
            df = df_orig.copy()
            bench_symbol = "^KS11" if mkt_type == "KOSPI" else ("^KQ11" if mkt_type == "KOSDAQ" else "QQQ")
            market_signal = upper_trends_1h[bench_symbol]['Trend'].iloc[-1] if bench_symbol in upper_trends_1h else -1
            df = calculate_supertrend(df, period=7, multiplier=4.5 if symbol in ["SOXL", "SOXS"] else 3.0)
            
            if bench_symbol in bench_ret_30m: df['RS'] = df['Close'].pct_change(current_rs_period) - bench_ret_30m[bench_symbol]
            else: continue
            
            df = df.dropna()
            processed_data[symbol] = df
            if df['Trend'].iloc[-1] == 1 and market_signal == 1:
                all_candidates.append({'ticker': symbol, 'rs': df['RS'].iloc[-1], 'price': df['Close'].iloc[-1], 'atr_pct': df['ATR_pct'].iloc[-1]})

        if skipped_stale_tickers:
            print(f"⏳ 최신 30분봉 누락으로 이번 사이클 제외: {', '.join(skipped_stale_tickers)}")

        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)

        if is_close_briefing:
            today_str = now.strftime("%Y-%m-%d")
            if self.last_briefing_date[target_market] != today_str:
                pos_info = "\n• ".join([f"{tk} ({pos['qty']}주)" for tk, pos in bot_holding_stocks.items()]) if bot_holding_stocks else "보유 없음"
                send_telegram(f"🏁 *[{'국내 주식' if target_market == 'KR' else '해외 주식'} 마감]*\n• 총자산: {total_account_value:,.0f}\n• 포지션:\n• {pos_info}")
                self.last_briefing_date[target_market] = today_str
            return

        has_action = False
        should_refresh_cash = False  

        # [매도 룰 1단계: Supertrend 추세 이탈]
        for t in list(bot_holding_stocks.keys()):
            if t not in processed_data or t in existing_order_symbols: continue
            if processed_data[t]['Trend'].iloc[-1] == -1:
                order_qty = bot_holding_stocks[t]['qty']
                if self.account_mgr.send_order(symbol=t, side="sell", qty=order_qty, order_type="market"):
                    send_telegram(f"🚨 *[추세 이탈 매도 주문 전송]*\n• 종목: {t} | {order_qty}주 매도 주문 전송 (추세 하락 전환)")
                    sync_holdings_after_order("추세 이탈 매도 주문 후")
                    has_action = True
                    should_refresh_cash = True

        # [매도 룰 2단계: 순환매 주도주 교체 - 철통 하드 브레이크]
        if all_candidates and bot_holding_stocks:
            for t in list(bot_holding_stocks.keys()):
                if t not in processed_data or t in existing_order_symbols: continue
                
                current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), None)
                if current_rs is None and 'RS' in processed_data[t].columns: current_rs = processed_data[t]['RS'].iloc[-1]
                if current_rs is None: current_rs = -999.0

                available_news = [c for c in all_candidates if c['ticker'] not in bot_holding_stocks and c['ticker'] not in existing_order_symbols]
                if not available_news: continue
                best_new = available_news[0]
                
                current_p = realtime_prices.get(t, processed_data[t]['Close'].iloc[-1])
                buy_p = bot_holding_stocks[t]['buy_price']
                
                if buy_p <= 0:
                    print(f"⚠️ {t} 종목의 정확한 매수 단가가 확인되지 않아 이번 사이클은 순환매를 대기합니다.")
                    continue

                profit_pct = (current_p - buy_p) / buy_p

                # 🛑 하드 브레이크 적용 (수익률 1% 미만인 상태라면 홀딩)
                if profit_pct < 0.01: 
                    continue 

                if current_rs != -999.0 and (best_new['rs'] - current_rs > (best_new['atr_pct'] * HURDLE_ATR_MULT)):
                    old_qty = bot_holding_stocks[t]['qty']
                    if self.account_mgr.send_order(symbol=t, side="sell", qty=old_qty, order_type="market"):
                        send_telegram(f"🔄 *[순환매 교체 매도 주문 전송]*\n• 주도주 교체: {t} 매도 주문 전송 ➡️ {best_new['ticker']} 진입 대기\n• 직전 수익률: {profit_pct*100:.2f}%")
                        sync_holdings_after_order("순환매 교체 매도 주문 후")
                        has_action = True
                        should_refresh_cash = True

        # 🔥 매도 발생 시 실시간 예수금 재조회
        if should_refresh_cash:
            print("🔄 매도 대금 반영을 위해 실시간 예수금을 재조회합니다.")
            refreshed_account = self.account_mgr.fetch_real_assets(target_market=target_market)
            if refreshed_account:
                fresh_cash = float(refreshed_account.get("summary", {}).get("totalCashBalance", fresh_cash))
                print(f"💰 동기화된 새 예수금 잔액: {fresh_cash:,.0f}원/달러")

        # [매수 룰: 압도적 대장주 슬롯 집중 진입]
        # TODO: 매도 주문 직후 체결/계좌 반영 지연 중 같은 사이클에서 신규 매수가 열릴 수 있어, 분할매수 정책 확정 후 주문 intent 단위 pending 관리가 필요합니다.
        for candidate in all_candidates:
            if len(bot_holding_stocks) >= active_max_slots: break
            t = candidate['ticker']
            
            # ⭐ 이제 실계좌-장부 매핑이 완전히 동기화되어, 보유 중인 종목은 절대로 이 블록(신규 진입)에 침범하지 못합니다.
            if t not in bot_holding_stocks and t not in existing_order_symbols:
                price = realtime_prices.get(t, candidate['price'])
                # 🎯 가용 자산의 90%를 깨끗하게 집중 배팅합니다 (기존 50% 분할 코드에서 원안이었던 90% 집중으로 복구 완료)
                target_allocation = fresh_cash * 0.90
                alloc_money = min(fresh_cash, target_allocation)
                qty = int(alloc_money // (price * (1 + SLIPPAGE) * (1 + FEE_HALF)))
                if qty > 0:
                    if self.account_mgr.send_order(symbol=t, side="buy", qty=qty, order_type="market"):
                        send_telegram(f"🟩 *[추세 주도주 매수 주문 전송]*\n• 종목: {t} | 수량: {qty}주 매수 주문 전송 (예수금 비중 90% 적용)")
                        sync_holdings_after_order("추세 주도주 매수 주문 후")
                        has_action = True
                    break

        if not has_action: print("🔍 [대기] 현재 포지션을 유지하며 추세를 관망합니다.")

    async def start_trading_bot(self):
        send_telegram("🤖 *Ver 24.6 토스 스키마 일원화 및 보유종목 오판 버그 100% 완전 수정 완료!*")
        while True:
            try: await self.execute_trade_cycle()
            except Exception as e: print(f"🚨 엔진 내부 예외 발생: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    mgr = TossAccountManager()
    bot = HybridTradingBot(mgr)
    asyncio.run(bot.start_trading_bot())
