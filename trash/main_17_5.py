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

# .env 파일로부터 안전하게 보안 키들을 메모리에 로드
load_dotenv()

# =====================================================================
# ⚙️ [글로벌 환경 설정] 보안 키 분리 및 자산 배분 세팅
# =====================================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TOSS_CLIENT_ID = os.getenv("TOSS_CLIENT_ID")
TOSS_CLIENT_SECRET = os.getenv("TOSS_CLIENT_SECRET")
TOSS_BASE_URL = "https://openapi.tossinvest.com"

# 통화별 최대 보유 슬롯 (한 번에 1개 종목만 집중 투자)
KR_MAX_SLOTS = 1              
US_MAX_SLOTS = 1              

HURDLE_ATR_MULT = 1.25      
RS_PERIOD = 130             

FEE_HALF = 0.00225          
SLIPPAGE = 0.0005           

UNIVERSE_FILE = "universe.json"

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
        self.account_seq = None

    def get_access_token(self):
        if self.token and time.time() < self.token_expiry - 60:
            return self.token
        url = f"{TOSS_BASE_URL}/oauth2/token"
        data = {"grant_type": "client_credentials", "client_id": TOSS_CLIENT_ID, "client_secret": TOSS_CLIENT_SECRET}
        res = requests.post(url, headers={"Content-Type": "application/x-www-form-urlencoded"}, data=data)
        if res.status_code == 200:
            rd = res.json()
            self.token = rd["access_token"]
            self.token_expiry = time.time() + rd["expires_in"]
            return self.token
        raise Exception(f"토스 토큰 발급 실패: {res.text}")

    def get_account_headers(self):
        token = self.get_access_token()
        if not self.account_seq:
            url = f"{TOSS_BASE_URL}/api/v1/accounts"
            res = requests.get(url, headers={"Authorization": f"Bearer {token}"})
            if res.status_code == 200:
                self.account_seq = res.json().get("result", [])[0]["accountSeq"]
            else: raise Exception("계좌 번호 획득 실패")
        return {"Authorization": f"Bearer {token}", "X-Tossinvest-Account": str(self.account_seq), "Content-Type": "application/json"}

    def fetch_real_assets(self):
        try:
            url = f"{TOSS_BASE_URL}/api/v1/assets"
            res = requests.get(url, headers=self.get_account_headers())
            if res.status_code == 200: return res.json().get("result", {})
        except Exception as e: print(f"❌ 토스 자산 조회 실패: {e}")
        return {}

    def send_order(self, symbol, side, qty, order_type="market", price=None):
        try:
            url = f"{TOSS_BASE_URL}/api/v1/orders"
            payload = {
                "symbol": str(symbol),
                "side": str(side),           
                "orderType": str(order_type),  
                "quantity": str(qty)          
            }
            if order_type == "limit" and price is not None:
                payload["price"] = str(int(price))

            res = requests.post(url, headers=self.get_account_headers(), json=payload)
            if res.status_code == 200:
                print(f"✅ 토스 주문 성공: {symbol} | {side} | {qty}주")
                return True
            else:
                print(f"❌ 토스 주문 실패: {res.text}")
                return False
        except Exception as e:
            print(f"❌ 토스 주문 예외 발생: {e}")
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
    default_kr = {"005930": "KOSPI", "091990": "KOSDAQ", "235980": "KOSDAQ"} 
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

    async def execute_trade_cycle(self):
        kr_universe_map, us_universe_list = load_universe()
        self.current_market = self.check_market_schedule()
        now = datetime.now()
        
        print(f"\n🔄 [하이브리드 엔진 통합 스캔] {now.strftime('%Y-%m-%d %H:%M:%S')} | 마켓: {self.current_market}")
        if self.current_market == "SLEEP": return

        # 1. 토스 Open API 계좌 실물 연동
        assets = self.account_mgr.fetch_real_assets()
        summary = assets.get("summary", {})
        total_account_value = float(summary.get("totalAssetValue", 0)) if summary else 0.0
        fresh_cash = float(summary.get("totalCashBalance", 0)) if summary else 0.0
        holding_stocks = {s.get("symbol"): {"qty": int(s.get("holdingQuantity", 0)), "entry_price": float(s.get("averagePurchasePrice", 0))} for s in assets.get("stocks", [])} if assets.get("stocks") else {}

        is_close_briefing = "CLOSE" in self.current_market
        target_market = self.current_market.replace("_CLOSE", "")

        benchmarks = ["^KS11", "^KQ11", "QQQ"]
        if target_market == "KR":
            yf_tickers = [self.convert_to_yf_ticker(sym, mkt) for sym, mkt in kr_universe_map.items()]
            currency_unit, fmt, active_max_slots = "원", ",.0f", KR_MAX_SLOTS
        else:
            yf_tickers = us_universe_list
            currency_unit, fmt, active_max_slots = "달러", ",.2f", US_MAX_SLOTS

        # 2. 야후 파이낸스 우회 조회 데이터 다운로드
        try:
            raw_30m = yf.download(yf_tickers + benchmarks, period="30d", interval="30m", progress=False)
            raw_1h = yf.download(benchmarks, period="30d", interval="1h", progress=False)
        except Exception as e:
            print(f"❌ 야후 파이낸스 다운로드 에러: {e}")
            return

        # 3. 지수 멀티 타임프레임 필터 빌드
        bench_ret_30m = {}
        upper_trends_1h = {}
        for b in benchmarks:
            b_close = raw_30m['Close'][b].dropna()
            if len(b_close) >= RS_PERIOD:
                bench_ret_30m[b] = b_close.pct_change(RS_PERIOD)
            df_b = pd.DataFrame({'Open': raw_1h['Open'][b], 'High': raw_1h['High'][b], 'Low': raw_1h['Low'][b], 'Close': raw_1h['Close'][b]}).dropna()
            if not df_b.empty:
                upper_trends_1h[b] = calculate_supertrend(df_b, period=10, multiplier=3.0)

        all_candidates = []
        processed_data = {}

        # 4. 개별 종목 상대강도(RS) 연산 및 필터링
        scan_source = kr_universe_map.items() if target_market == "KR" else [(u, "US") for u in us_universe_list]
        for symbol, mkt_type in scan_source:
            yf_tk = self.convert_to_yf_ticker(symbol, mkt_type)
            if yf_tk not in raw_30m['Close']: continue
            df = pd.DataFrame({
                'Open': raw_30m['Open'][yf_tk], 'High': raw_30m['High'][yf_tk],
                'Low': raw_30m['Low'][yf_tk], 'Close': raw_30m['Close'][yf_tk]
            }).dropna()
            if len(df) < (RS_PERIOD + 5): continue

            bench_symbol = "^KS11" if mkt_type == "KOSPI" else ("^KQ11" if mkt_type == "KOSDAQ" else "QQQ")
            market_signal = upper_trends_1h[bench_symbol]['Trend'].iloc[-1] if bench_symbol in upper_trends_1h else -1
            
            df = calculate_supertrend(df, period=7, multiplier=4.5 if symbol in ["SOXL", "SOXS"] else 3.0)
            if bench_symbol in bench_ret_30m:
                df['RS'] = df['Close'].pct_change(RS_PERIOD) - bench_ret_30m[bench_symbol]
            else: continue
            
            df = df.dropna()
            processed_data[symbol] = df

            if df['Trend'].iloc[-1] == 1 and market_signal == 1:
                all_candidates.append({'ticker': symbol, 'rs': df['RS'].iloc[-1], 'price': df['Close'].iloc[-1], 'atr_pct': df['ATR_pct'].iloc[-1]})

        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)

        # 5. 장 마감 요약 브리핑
        if is_close_briefing:
            today_str = now.strftime("%Y-%m-%d")
            if self.last_briefing_date[target_market] != today_str:
                market_name = "🇰🇷 국장(보안형)" if target_market == "KR" else "🇺🇸 미장(보안형)"
                pos_info = "\n• ".join([f"{tk} ({pos['qty']}주) / 평단: {pos['entry_price']}" for tk, pos in holding_stocks.items()]) if holding_stocks else "보유 무"
                leader_info = f"*{all_candidates[0]['ticker']}* (RS: {all_candidates[0]['rs']:+.4f})" if all_candidates else "선별 실패"

                send_telegram(f"🏁 *[{market_name} 마감 리포트]*\n• 총 자 산: {total_account_value:{fmt}} {currency_unit}\n• 예수금: {fresh_cash:{fmt}} {currency_unit}\n• 포지션 상황:\n• {pos_info}\n👑 RS 1위 대장주: {leader_info}")
                self.last_briefing_date[target_market] = today_str
            return

        # =====================================================================
        # ⚡ [실전 매매 집행 레이어] (★자산 보안 및 50% 통제 규칙 집행)
        # =====================================================================
        # [Rule 1] 실시간 추세 이탈 청산 (Sell)
        for t in list(holding_stocks.keys()):
            if t not in processed_data: continue
            if processed_data[t]['Trend'].iloc[-1] == -1:
                order_qty = holding_stocks[t]['qty']
                success = self.account_mgr.send_order(symbol=t, side="sell", qty=order_qty, order_type="market")
                if success:
                    send_telegram(f"🚨 *[실전 매도 완료]* 추세 이탈 청산\n• 종목: {t}\n• 수량: {order_qty}주\n• 시장가 전량 매도 완료")

        # [Rule 2] 실시간 동적 순환매 (Rotation)
        if all_candidates and holding_stocks:
            for t in list(holding_stocks.keys()):
                if t not in processed_data: continue
                current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
                available_news = [c for c in all_candidates if c['ticker'] not in holding_stocks]
                if not available_news: continue
                best_new = available_news[0]
                
                dynamic_hurdle = best_new['atr_pct'] * HURDLE_ATR_MULT
                if best_new['rs'] - current_rs > dynamic_hurdle:
                    old_qty = holding_stocks[t]['qty']
                    sell_success = self.account_mgr.send_order(symbol=t, side="sell", qty=old_qty, order_type="market")
                    if sell_success:
                        send_telegram(f"🔄 *[순환매 집행]* 구 주도주 탈락 ➡️ 매도\n• 종목: {t} 매도 완료")

        # [Rule 3] 실시간 50% 분할 투자 매수 (Buy)
        for candidate in all_candidates:
            if len(holding_stocks) >= active_max_slots: break
            t = candidate['ticker']
            
            if t not in holding_stocks:
                price = candidate['price']
                
                # 자산 배분 규칙: 통화별 총자산 가치의 최대 50%만 진입 제한
                target_allocation = total_account_value * 0.50
                alloc_money = min(fresh_cash, target_allocation)
                
                # 수수료, 슬리피지 감안 보수적 수량 연산
                qty = int(alloc_money // (price * (1 + SLIPPAGE) * (1 + FEE_HALF)))
                
                if qty > 0:
                    if target_market == "KR":
                        # 국장: 지정가 정수 매수
                        success = self.account_mgr.send_order(symbol=t, side="buy", qty=qty, order_type="limit", price=price)
                        order_mode = f"지정가 ({int(price):,d}원)"
                    else:
                        # 미장: 시장가 정수 매수
                        success = self.account_mgr.send_order(symbol=t, side="buy", qty=qty, order_type="market")
                        order_mode = "시장가"
                    
                    if success:
                        send_telegram(f"🟩 *[50% 비중 실전 매수 완료]*\n• 종목: {t}\n• 수량: {qty}주\n• 방식: {order_mode}\n🛡️ 리스크 관리용 안전 현금 버퍼 확보 유지")
                    break

    async def start_trading_bot(self):
        send_telegram("🤖 *Ver 17.5 50% 자산 분할 및 환경변수 보안형 최종 엔진 가동!*")
        while True:
            try: await self.execute_trade_cycle()
            except Exception as e: print(f"🚨 메인 루프 예외: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    mgr = TossAccountManager()
    bot = HybridTradingBot(mgr)
    asyncio.run(bot.start_trading_bot())