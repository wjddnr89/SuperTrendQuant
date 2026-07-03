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

KR_MAX_SLOTS = 1              
US_MAX_SLOTS = 1              

HURDLE_ATR_MULT = 1.25       
KR_RS_PERIOD = 100          
US_RS_PERIOD = 130          

FEE_HALF = 0.00225          
SLIPPAGE = 0.0005           

UNIVERSE_FILE = "universe.json"
HOLDING_FILE = "holding.json"
MOCK_ACCOUNT_FILE = "mock_account.json"

# =====================================================================
# 📦 [Stateful 메모리 관리 및 유니버스 레이어]
# =====================================================================
def load_universe():
    default_kr = {"005930": "KOSPI", "091990": "KOSDAQ", "403870": "KOSDAQ"} 
    default_us = ["QQQ", "SOXL"]
    if not os.path.exists(UNIVERSE_FILE): return default_kr, default_us
    try:
        with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("KR_UNIVERSE_MAP", default_kr), data.get("US_UNIVERSE_LIST", default_us)
    except: return default_kr, default_us

def load_holdings():
    if not os.path.exists(HOLDING_FILE):
        with open(HOLDING_FILE, "w") as f: json.dump({"KR": {}, "US": {}}, f, indent=4)
    with open(HOLDING_FILE, "r") as f: return json.load(f)

def save_holdings(data):
    with open(HOLDING_FILE, "w") as f: json.dump(data, f, indent=4)

# =====================================================================
# 💰 [가상 계좌 관리 시스템]
# =====================================================================
def load_mock_account():
    if not os.path.exists(MOCK_ACCOUNT_FILE):
        initial_account = {
            "KRW_balance": 10000000.0,
            "USD_balance": 6500.0
        }
        with open(MOCK_ACCOUNT_FILE, "w") as f: json.dump(initial_account, f, indent=4)
    with open(MOCK_ACCOUNT_FILE, "r") as f: return json.load(f)

def save_mock_account(data):
    with open(MOCK_ACCOUNT_FILE, "w") as f: json.dump(data, f, indent=4)

# =====================================================================
# 📢 [텔레그램 알림 레이어]
# =====================================================================
def send_telegram(msg):
    print(f"📢 [TELEGRAM] {msg}")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        print(f"⚠️ 텔레그램 발송 실패: {e}")

# =====================================================================
# 📊 [수학적 알고리즘 계산기 레이어 (멀티인덱스 방어 완비)]
# =====================================================================
def calculate_supertrend(df, period=7, multiplier=3.0):
    if df.empty: return df
    
    # 🌟 yfinance의 MultiIndex 혹은 Series 혼선 방지를 위한 고정 처리
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    close = df['Close'].squeeze()
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    hl2 = (high + low) / 2
    final_ub = hl2 + (multiplier * atr)
    final_lb = hl2 - (multiplier * atr)
    
    # 판다스 내부의 고유 이름표(Label) 충돌을 방지하기 위해 numpy 값 배열로 연산 제어
    final_ub_val = final_ub.values
    final_lb_val = final_lb.values
    close_val = close.values
    
    for i in range(1, len(df)):
        if close_val[i-1] > final_ub_val[i-1]: 
            final_ub_val[i] = min(final_ub_val[i], final_ub_val[i-1])
        if close_val[i-1] < final_lb_val[i-1]: 
            final_lb_val[i] = max(final_lb_val[i], final_lb_val[i-1])
        
    trend = np.zeros(len(df))
    for i in range(1, len(df)):
        if close_val[i] > final_ub_val[i-1]: trend[i] = 1
        elif close_val[i] < final_lb_val[i-1]: trend[i] = -1
        else: trend[i] = trend[i-1] if trend[i-1] != 0 else 1
        
    df = df.copy()
    df['Trend'] = trend
    df['ATR_pct'] = (atr / close) * 100
    return df

# =====================================================================
# 🔄 [가상 계좌 매핑 인터페이스]
# =====================================================================
class TossAccountManager:
    def fetch_real_assets(self, target_market="KR"):
        account = load_mock_account()
        holdings = load_holdings()
        market_holdings = holdings.get(target_market, {})
        
        holding_stocks = {}
        for symbol, info in market_holdings.items():
            holding_stocks[symbol] = {
                "quantity": float(info["qty"]),
                "purchasePrice": float(info["buy_price"])
            }
            
        balance = account["KRW_balance"] if target_market == "KR" else account["USD_balance"]
        return {
            "summary": {"totalCashBalance": balance},
            "holding_stocks": holding_stocks
        }

    def fetch_open_orders(self, target_market="KR"):
        return []

    def send_order(self, symbol, side, qty, order_type="market"):
        account = load_mock_account()
        is_kr = not any(c.isalpha() for c in symbol)
        balance_key = "KRW_balance" if is_kr else "USD_balance"
        
        try:
            ticker_str = symbol
            if is_kr:
                kr_map, _ = load_universe()
                m_type = kr_map.get(symbol, "KOSPI")
                ticker_str += ".KS" if m_type == "KOSPI" else ".KQ"
            df = yf.download(ticker_str, period="1d", interval="1m", progress=False)
            price = float(df['Close'].squeeze().iloc[-1])
        except Exception:
            price = 100000.0 if is_kr else 50.0
                
        if side == "buy":
            total_cost = qty * (price * (1 + SLIPPAGE) * (1 + FEE_HALF))
            if account[balance_key] < total_cost:
                print(f"❌ [가상 매수 실패] 잔고 부족 (필요: {total_cost:,.2f} / 잔고: {account[balance_key]:,.2f})")
                return False
            account[balance_key] -= total_cost
            save_mock_account(account)
            print(f"📦 [가상 체결 완료] {symbol} {qty}주 매수 / 체결단가: {price:,.2f}")
            return True
            
        elif side == "sell":
            total_revenue = qty * (price * (1 - SLIPPAGE) * (1 - FEE_HALF))
            account[balance_key] += total_revenue
            save_mock_account(account)
            print(f"📦 [가상 체결 완료] {symbol} {qty}주 매도 / 체결단가: {price:,.2f}")
            return True
            
        return False

# =====================================================================
# 🧠 [실시간 매매 코어 엔진 루프]
# =====================================================================
class CoreTradingEngine:
    def __init__(self):
        self.account_mgr = TossAccountManager()

    def execute_trade_cycle(self, target_market="KR"):
        print(f"\n🔄 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {target_market} 시장 매매 연산 가동")
        
        kr_universe_map, us_universe_list = load_universe()
        universe = list(kr_universe_map.keys()) if target_market == "KR" else us_universe_list
                
        holdings_master = load_holdings()
        market_holdings = holdings_master.get(target_market, {})
        
        account_data = self.account_mgr.fetch_real_assets(target_market=target_market)
        if not account_data: return
        
        fresh_cash = float(account_data.get("summary", {}).get("totalCashBalance", 0))
        real_holding_stocks = account_data.get("holding_stocks", {})
        
        # 장부와 가상 계좌 동기화
        for t in list(market_holdings.keys()):
            if t not in real_holding_stocks: del market_holdings[t]
        for t in real_holding_stocks:
            if t not in market_holdings:
                market_holdings[t] = {"qty": real_holding_stocks[t]["quantity"], "buy_price": real_holding_stocks[t]["purchasePrice"]}
        holdings_master[target_market] = market_holdings
        save_holdings(holdings_master)
        bot_holding_stocks = list(market_holdings.keys())

        open_orders = self.account_mgr.fetch_open_orders(target_market=target_market)
        existing_order_symbols = [o.get("stockCode") for o in open_orders if o.get("stockCode")]

        all_candidates = []
        upper_trends_1h = {}
        bench_ret_30m = {}
        
        bench_list = ["^KS11", "^KQ11", "QQQ"]
        for b in bench_list:
            try:
                b_df = yf.download(b, period="5d", interval="1h", progress=False)
                if not b_df.empty and len(b_df) >= 7:
                    upper_trends_1h[b] = calculate_supertrend(b_df, period=7, multiplier=3.0)
                
                b_df_30m = yf.download(b, period="5d", interval="30m", progress=False)
                current_rs_period = KR_RS_PERIOD if target_market == "KR" else US_RS_PERIOD
                
                if not b_df_30m.empty and len(b_df_30m) > current_rs_period:
                    # 🌟 지수의 30분 변동률을 순수한 float 단일값으로 강제 추출하여 딕셔너리에 저장
                    val = b_df_30m['Close'].squeeze().pct_change(current_rs_period).iloc[-1]
                    bench_ret_30m[b] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                else:
                    print(f"⏳ 지수 {b} 현재 데이터 축적 중...")
            except Exception as e:
                print(f"❌ 지수 {b} 파싱 실패: {e}")

        # 개별 종목 분석
        for symbol in universe:
            try:
                ticker_str = symbol
                if target_market == "KR":
                    m_type = kr_universe_map.get(symbol, "KOSPI")
                    ticker_str = symbol + ".KS" if m_type == "KOSPI" else symbol + ".KQ"
                    
                df_1h = yf.download(ticker_str, period="7d", interval="1h", progress=False)
                df_30m = yf.download(ticker_str, period="5d", interval="30m", progress=False)
                
                if df_1h.empty or df_30m.empty: continue
                
                df = calculate_supertrend(df_1h, period=7, multiplier=4.5 if symbol in ["SOXL", "SOXS"] else 3.0)
                
                mkt_type = "KOSPI" if ticker_str.endswith(".KS") else ("KOSDAQ" if ticker_str.endswith(".KQ") else "US")
                bench_symbol = "^KS11" if mkt_type == "KOSPI" else ("^KQ11" if mkt_type == "KOSDAQ" else "QQQ")
                
                if bench_symbol not in upper_trends_1h or bench_symbol not in bench_ret_30m:
                    continue
                
                # 🌟 트렌드 시그널 추출 안전 가드
                market_signal = upper_trends_1h[bench_symbol]['Trend'].squeeze().iloc[-1]
                
                current_rs_period = KR_RS_PERIOD if target_market == "KR" else US_RS_PERIOD
                
                # 🌟 [오류 해결의 핵심] 순수한 값끼리 뺄셈 연산 처리하여 판다스 라벨 오류 원천 차단
                stock_ret_series = df_30m['Close'].squeeze().pct_change(current_rs_period)
                stock_ret_val = stock_ret_series.iloc[-1] if not isinstance(stock_ret_series, pd.Series) else stock_ret_series.squeeze().iloc[-1]
                
                df['RS'] = float(stock_ret_val) - bench_ret_30m[bench_symbol]
                    
                df = df.dropna()
                if df.empty: continue
                
                last_trend = df['Trend'].squeeze().iloc[-1]
                last_price = df['Close'].squeeze().iloc[-1]
                last_atr_pct = df['ATR_pct'].squeeze().iloc[-1]
                last_rs = df['RS'].squeeze().iloc[-1]
                
                if last_trend == 1 and market_signal == 1:
                    all_candidates.append({
                        'ticker': symbol, 
                        'rs': float(last_rs), 
                        'price': float(last_price), 
                        'atr_pct': float(last_atr_pct)
                    })
            except Exception as e:
                print(f"⚠️ 종목 {symbol} 연산 중 예외 스킵: {e}")

        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)
        print(f"🔥 [대장주 스캔] 진입 후보 리스트: {all_candidates}")

        # 2. 매도 룰 파이프라인
        available_news = [c for c in all_candidates if c['ticker'] not in bot_holding_stocks]
        has_action = False
        should_refresh_cash = False

        for t in list(market_holdings.keys()):
            if t in existing_order_symbols: continue
            try:
                m_type = kr_universe_map.get(t, "KOSPI") if target_market == "KR" else "US"
                ticker_str = t + ".KS" if target_market == "KR" and m_type == "KOSPI" else (t + ".KQ" if target_market == "KR" else t)
                
                current_df = yf.download(ticker_str, period="5d", interval="1h", progress=False)
                current_df = calculate_supertrend(current_df, period=7, multiplier=4.5 if t in ["SOXL", "SOXS"] else 3.0)
                
                if current_df.empty: continue
                current_price = float(current_df['Close'].squeeze().iloc[-1])
                current_trend = current_df['Trend'].squeeze().iloc[-1]
                
                buy_price = float(market_holdings[t]["buy_price"])
                qty = int(market_holdings[t]["qty"])
                current_return = (current_price - buy_price) / buy_price
                
                # 규칙 1: 추세 이탈 매도
                if current_trend == -1:
                    if self.account_mgr.send_order(symbol=t, side="sell", qty=qty, order_type="market"):
                        send_telegram(f"🚨 *[추세 이탈 매도]*\n• 종목: {t}\n• 수익률: {current_return*100:.2f}%\n• 마감 기준 추세 변곡 전량 청산")
                        del market_holdings[t]
                        holdings_master[target_market] = market_holdings
                        save_holdings(holdings_master)
                        has_action = True
                        should_refresh_cash = True
                    continue

                # 규칙 2: 주도주 교체 매도
                if current_return >= 0.01:
                    if available_news:
                        top_news = available_news[0]
                        if top_news['rs'] > market_holdings[t].get('rs', -999):
                            if self.account_mgr.send_order(symbol=t, side="sell", qty=qty, order_type="market"):
                                send_telegram(f"🔄 *[주도주 교체 매도]*\n• 기존: {t} (수익률: {current_return*100:.2f}%)\n• 사유: 마감 1분 전 마진 확보 및 순환매 교체")
                                del market_holdings[t]
                                holdings_master[target_market] = market_holdings
                                save_holdings(holdings_master)
                                has_action = True
                                should_refresh_cash = True
            except Exception as e:
                print(f"매도 연산 에러 패스: {e}")

        if should_refresh_cash:
            refreshed_account = self.account_mgr.fetch_real_assets(target_market=target_market)
            if refreshed_account:
                fresh_cash = float(refreshed_account.get("summary", {}).get("totalCashBalance", fresh_cash))

        # 3. 매수 룰 파이프라인
        active_max_slots = KR_MAX_SLOTS if target_market == "KR" else US_MAX_SLOTS
        
        for candidate in all_candidates:
            if len(market_holdings) >= active_max_slots: 
                break
                
            t = candidate['ticker']
            
            if t not in bot_holding_stocks and t not in existing_order_symbols:
                price = candidate['price']
                target_allocation = fresh_cash * 0.90
                alloc_money = min(fresh_cash, target_allocation)
                    
                qty = int(alloc_money // (price * (1 + SLIPPAGE) * (1 + FEE_HALF)))
                
                if qty > 0:
                    if self.account_mgr.send_order(symbol=t, side="buy", qty=qty, order_type="market"):
                        send_telegram(f"🟩 *[추세 주도주 진입]*\n• 종목: {t} | 수량: {qty}주 매수 집행 (예수금 비중 90% 적용)")
                        market_holdings[t] = {"qty": qty, "buy_price": price, "rs": candidate['rs']}
                        holdings_master[target_market] = market_holdings
                        save_holdings(holdings_master)
                        has_action = True
                    break

        if not has_action: 
            print("🔍 [대기] 현재 포지션을 유지하며 추세를 관망합니다.")

    async def start_trading_pipeline(self):
        kr_acc = load_mock_account()
        send_telegram(f"🤖 *[가상 모의투자 봇 가동 - yfinance 인덱스 에러 긴급 수정판]*\n• 원화: {kr_acc['KRW_balance']:,} 원\n• 달러: {kr_acc['USD_balance']:,} 달러\n• ⏱️ 매시 29분, 59분(봉 마감 1분 전)에만 매매를 판단합니다.")
        
        while True:
            try:
                now = datetime.now()
                if now.minute == 29 or now.minute == 59:
                    self.execute_trade_cycle(target_market="KR")
                    self.execute_trade_cycle(target_market="US")
                    
                    print("⏰ 해당 봉 판단 완료. 중복 연산 방지를 위해 61초간 대기 상태로 진입합니다.")
                    await asyncio.sleep(61)
                else:
                    if now.second == 0:  
                        print(f"⏳ [대기 중] 현재 {now.strftime('%H:%M')} -> 다음 봉 마감(29분/59분)을 기다리는 중...")
                    await asyncio.sleep(10)
                    
            except Exception as e:
                print(f"🚨 메인 루프 예외 발생: {e}")
                await asyncio.sleep(10)

# =====================================================================
# 🚀 메인 엔트리 포인트
# =====================================================================
if __name__ == "__main__":
    engine = CoreTradingEngine()
    asyncio.run(engine.start_trading_pipeline())