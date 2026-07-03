import os
import sys
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import io

# 1. 윈도우 터미널 버퍼링을 완전히 끄고 실시간 출력 강제 활성화
sys.stdout.reconfigure(line_buffering=True)

# 2. 윈도우 환경 이모지 및 utf-8 인코딩 에러 방지
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==========================================
# ⚙️ 사용자가 설정해야 하는 구역
# ==========================================
TELEGRAM_TOKEN = "8580742809:AAGpnuw6pZNpWwf5dWxMEIrrl2gbasDwa6M"
TELEGRAM_CHAT_ID = "8323058445"

# 가상 계좌 상태 파일 경로 (프로그램이 꺼졌다가 켜져도 잔고 유지)
ACCOUNT_FILE = "virtual_account.csv"
INITIAL_CASH = 10000.0
MAX_SLOTS = 2
HURDLE_RATE = 0.015
# ==========================================

def send_telegram(message):
    """텔레그램으로 실시간 알림을 쏩니다."""
    print(f"📢 [Telegram] {message}")
    if "여기에" in TELEGRAM_TOKEN: return # 토큰 설정 안 되어 있으면 패스
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=5)
    except Exception as e: print(f"❌ 알림 전송 실패: {e}")

def calculate_supertrend(df, period=7, multiplier=3.0):
    high, low, close = df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze()
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
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
    return trend

def load_account():
    """가상 계좌 정보(잔고, 보유 종목)를 불러옵니다."""
    if os.path.exists(ACCOUNT_FILE):
        df = pd.read_csv(ACCOUNT_FILE)
        cash = float(df.loc[df['ticker'] == 'CASH', 'entry_price'].values[0])
        positions = {}
        for _, row in df.iterrows():
            if row['ticker'] == 'CASH': continue
            positions[row['ticker']] = {
                'qty': int(row['qty']),
                'entry_price': float(row['entry_price']),
                'highest_price': float(row['highest_price'])
            }
        return cash, positions
    return INITIAL_CASH, {}

def save_account(cash, positions):
    """가상 계좌 정보를 파일에 저장합니다."""
    data = [{'ticker': 'CASH', 'qty': 0, 'entry_price': cash, 'highest_price': cash}]
    for t, p in positions.items():
        data.append({'ticker': t, 'qty': p['qty'], 'entry_price': p['entry_price'], 'highest_price': p['highest_price']})
    pd.DataFrame(data).to_csv(ACCOUNT_FILE, index=False)

def check_market_and_trade():
    """30분마다 실행되며 실시간 데이터를 분석해 모의 매매를 집행합니다."""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n🔄 [{now_str}] 실시간 모니터링 사이클 가동...")
    
    full_universe = ["SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON", "SOXS"]
    
    # 1. 실시간 데이터 다운로드 (최근 5일치 30분봉 데이터)
    try:
        raw_data = yf.download(full_universe + ["QQQ"], period="5d", interval="30m", progress=False)
        qqq_daily = yf.download("QQQ", period="1y", interval="1d", progress=False)
    except Exception as e:
        print(f"❌ 데이터 로드 에러: {e}")
        return

    # QQQ EMA200 확인
    qqq_daily['EMA200'] = qqq_daily['Close'].ewm(span=200, adjust=False).mean()
    current_qqq_ema200 = qqq_daily['EMA200'].iloc[-1].squeeze()
    current_qqq_close = raw_data['Close']['QQQ'].dropna().iloc[-1].squeeze()
    
    is_bull_market = current_qqq_close > current_qqq_ema200
    target_pool = ["SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON"] if is_bull_market else ["SOXS"]

    cash, positions = load_account()
    data_dict = {}
    
    # 각 종목 스펙 연산
    qqq_ret_5d = raw_data['Close']['QQQ'].dropna().pct_change(65).iloc[-1].squeeze()
    
    all_candidates = []
    for ticker in full_universe:
        df = pd.DataFrame({
            'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
            'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker]
        }).dropna()
        if df.empty: continue
        
        mult = 4.5 if ticker in ["SOXL", "SOXS"] else 3.0
        df['Trend'] = calculate_supertrend(df, period=7, multiplier=mult)
        df['Return_5d'] = df['Close'].pct_change(65)
        
        data_dict[ticker] = df
        
        # 마지막 확정 봉의 시그널 수집
        curr_trend = df['Trend'].iloc[-1]
        prev_trend = df['Trend'].iloc[-2]
        rs_score = df['Return_5d'].iloc[-1] - qqq_ret_5d
        
        if ticker in target_pool and curr_trend == 1:
            all_candidates.append({
                'ticker': ticker, 'rs': rs_score, 
                'signal_buy': (prev_trend == -1), 'price': df['Close'].iloc[-1]
            })

    # [1] 트레일링 익절 및 SuperTrend 매도 실시간 감시
    for t in list(positions.keys()):
        curr_price = data_dict[t]['Close'].iloc[-1]
        curr_trend = data_dict[t]['Trend'].iloc[-1]
        
        # 최고가 업데이트
        if curr_price > positions[t]['highest_price']:
            positions[t]['highest_price'] = curr_price
            
        profit = (curr_price - positions[t]['entry_price']) / positions[t]['entry_price']
        drop = (positions[t]['highest_price'] - curr_price) / positions[t]['highest_price']
        
        # 청산 조건 체크
        sell_reason = ""
        if profit >= 0.30 and drop >= 0.10:
            sell_reason = f"🎯 트레일링 익절 완료 (수익률: {profit*100:+.2f}%)"
        elif curr_trend == -1:
            sell_reason = f"📉 SuperTrend 매도 시그널 데드크로스 발생"
            
        if sell_reason:
            qty = positions[t]['qty']
            recv_cash = qty * curr_price * (1 - 0.00225 - 0.0005) # 수수료/슬리피지 차감
            cash += recv_cash
            send_telegram(f"🚨 *[가상 매도 알림]*\n종목: {t}\n사유: {sell_reason}\n매도가: ${curr_price:.2f}\n정산 금액: ${recv_cash:.2f}")
            del positions[t]

    # [2] 순환매 허들 체크
    if all_candidates:
        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)
        top_2_tickers = [c['ticker'] for c in all_candidates[:2]]
        
        for t in list(positions.keys()):
            if t in top_2_tickers: continue
            
            available_news = [c for c in all_candidates[:2] if c['ticker'] not in positions]
            if not available_news: continue
            best_new = available_news[0]
            
            current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
            
            if best_new['rs'] - current_rs > HURDLE_RATE:
                curr_price = data_dict[t]['Close'].iloc[-1]
                qty = positions[t]['qty']
                recv_cash = qty * curr_price * (1 - 0.00225 - 0.0005)
                cash += recv_cash
                send_telegram(f"🔄 *[가상 순환매 매도]*\n종목: {t}\n사유: 더 강한 주도주({best_new['ticker']}) 발견으로 교체\n매도가: ${curr_price:.2f}")
                del positions[t]

        # [3] 신규 매수 집행
        for candidate in all_candidates:
            if len(positions) >= MAX_SLOTS: break
            t = candidate['ticker']
            
            if t not in positions and candidate['signal_buy']:
                curr_price = candidate['price']
                current_assets = cash + sum([p['qty'] * data_dict[pos_t]['Close'].iloc[-1] for pos_t, p in positions.items()])
                target_unit = current_assets * (1 / MAX_SLOTS) * 0.995
                alloc = min(cash, target_unit)
                
                qty = int(alloc // (curr_price * 1.0005))
                cost = qty * curr_price * (1 + 0.00225 + 0.0005)
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t] = {'qty': qty, 'entry_price': curr_price, 'highest_price': curr_price}
                    send_telegram(f"🟩 *[가상 매수 알림]*\n종목: {t}\n매수평균가: ${curr_price:.2f}\n수량: {qty}주\n투입 자금: ${cost:.2f}")

    save_account(cash, positions)
    
    # 현재 계좌 요약 브리핑
    total_val = cash + sum([p['qty'] * data_dict[pos_t]['Close'].iloc[-1] for pos_t, p in positions.items()])
    print(f"💰 현재 가상 자산 총액: ${total_val:.2f} | 현금: ${cash:.2f} | 보유 종목 수: {len(positions)}개")

def is_market_open():
    """현재 시간이 미국 본장 운영 시간인지 판별합니다. (한국시간 22:30 ~ 05:00)"""
    now = datetime.now()
    # 주말(토, 일)에는 가동 중지
    if now.weekday() >= 5: return False
    
    current_time = now.strftime("%H:%M")
    # 서머타임 기준 미국 본장 (22:30 ~ 익일 05:00)
    if "22:30" <= current_time or current_time <= "05:00":
        return True
    return False

if __name__ == "__main__":
    print("🚀 Ver 5.9.4-Live 포워드 모의투자 엔진이 시작되었습니다.")
    print("📱 설정된 텔레그램으로 시스템 가동 확인 메시지를 전송합니다.")
    send_telegram("🤖 *Ver 5.9.4 '2슬롯 불나방' 라이브 모의투자 엔진 가동 시작!*")
    
    while True:
        try:
            if is_market_open():
                check_market_and_trade()
                # 30분(1800초) 대기 후 다음 봉 확인
                time.sleep(1800)
            else:
                # 장이 아닐 때는 5분마다 체크하며 휴식
                print(f"💤 현재 시간 {datetime.now().strftime('%H:%M')}, 미국 본장 시간이 아니므로 대기 중...")
                time.sleep(300)
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            sys.exit()
        except Exception as e:
            print(f"💥 루프 에러 발생: {e}")
            time.append(60)