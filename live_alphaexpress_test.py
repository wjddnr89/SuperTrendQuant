import os
import sys
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import io

# 1. 윈도우 터미널 버퍼링 완전히 끄고 실시간 출력 강제 활성화
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
ACCOUNT_FILE = "virtual_account_v6.csv"
INITIAL_CASH = 10000.0
MAX_SLOTS = 2          # 🌟 백테스트 알파 달성의 핵심: 2슬롯 유지
HURDLE_RATE = 0.015    # 순환매 교체 허들
FEE_HALF = 0.00225     # 편도 수수료
SLIPPAGE = 0.0005      # 편도 슬리피지
# ==========================================

def send_telegram(message):
    """텔레그램으로 실시간 알림을 쏩니다."""
    print(f"📢 [Telegram] {message}")
    if "여기에" in TELEGRAM_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=5)
    except Exception as e: print(f"❌ 알림 전송 실패: {e}")

def calculate_supertrend(df, period=7, multiplier=3.0):
    """백테스트 엔진과 100% 일치하는 정밀 슈퍼트렌드 연산"""
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
                'entry_price': float(row['entry_price'])
            }
        return cash, positions
    return INITIAL_CASH, {}

def save_account(cash, positions):
    """가상 계좌 정보를 파일에 저장합니다."""
    data = [{'ticker': 'CASH', 'qty': 0, 'entry_price': cash}]
    for t, p in positions.items():
        data.append({'ticker': t, 'qty': p['qty'], 'entry_price': p['entry_price']})
    pd.DataFrame(data).to_csv(ACCOUNT_FILE, index=False)

def check_market_and_trade():
    """30분마다 실시간 데이터를 분석하여 백테스트와 동일한 메커니즘으로 모의 매매를 집행합니다."""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n🔄 [{now_str}] 실시간 모니터링 사이클 가동...")
    
    # 생존 편향이 완벽히 배제된 14개 오리지널 유니버스
    full_universe = ["SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON", "SOXS"]
    
    try:
        # 30분봉 실시간 데이터 수집 (안정적인 지표 계산을 위해 15일치 수집)
        raw_data = yf.download(full_universe + ["QQQ"], period="15d", interval="30m", progress=False)
    except Exception as e:
        print(f"❌ 데이터 로드 에러: {e}")
        return

    cash, positions = load_account()
    data_dict = {}
    
    # QQQ 5일 상대강도 기준점 연산 (30분봉 65개 = 약 5영업일)
    qqq_ret_5d = raw_data['Close']['QQQ'].dropna().pct_change(65).iloc[-1].squeeze()
    
    all_candidates = []
    
    # 1. 전 종목 기술지표 및 상대강도(RS) 연산
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
        
        # 최신 확정봉 데이터 스캔
        curr_trend = df['Trend'].iloc[-1]
        prev_trend = df['Trend'].iloc[-2]
        rs_score = df['Return_5d'].iloc[-1] - qqq_ret_5d
        
        # 매수 타점 스캔 pool 구성 (Trend가 1인 모든 종목 타겟)
        if curr_trend == 1:
            all_candidates.append({
                'ticker': ticker, 
                'rs': rs_score, 
                'signal_buy': (prev_trend == -1), # 직전 봉에서 골든크로스가 터졌는가?
                'price': df['Close'].iloc[-1]
            })

    # 2. [STEP 1] 기존 포지션 SuperTrend 매도 실시간 감시 (당일 종가 즉시 체결)
    for t in list(positions.keys()):
        if t not in data_dict: continue
        curr_price = data_dict[t]['Close'].iloc[-1]
        curr_trend = data_dict[t]['Trend'].iloc[-1]
        
        if curr_trend == -1:
            qty = positions[t]['qty']
            # 백테스트와 동일한 슬리피지 및 수수료 실시간 반영
            real_sell_price = curr_price * (1 - SLIPPAGE)
            recv_cash = qty * real_sell_price * (1 - FEE_HALF)
            cash += recv_cash
            
            pnl = (real_sell_price - positions[t]['entry_price']) / positions[t]['entry_price']
            
            send_telegram(f"🚨 *[가상 매도 알림]*\n종목: {t}\n사유: 📉 SuperTrend 데드크로스 발생 (즉시 청산)\n매도가: ${real_sell_price:.2f}\n수익률: {pnl*100:+.2f}%")
            del positions[t]

    # 3. [STEP 2] 순환매 허들 연산 및 교체 매도 집행
    if all_candidates:
        # RS(상대강도) 스코어 기준 내림차순 정렬
        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)
        
        for t in list(positions.keys()):
            if t not in data_dict: continue
            
            # 현재 보유중인 종목의 RS 점수 계산
            current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
            
            # 현재 유니버스 전체 대장주(1위) 후보 스캔
            available_news = [c for c in all_candidates if c['ticker'] not in positions]
            if not available_news: continue
            best_new = available_news[0]
            
            # 🌟 백테스트 알파의 핵심 공식: 대장주와의 RS 격차가 HURDLE_RATE를 넘으면 기존주 과감히 당일 청산
            if best_new['rs'] - current_rs > HURDLE_RATE:
                curr_price = data_dict[t]['Close'].iloc[-1]
                qty = positions[t]['qty']
                
                real_sell_price = curr_price * (1 - SLIPPAGE)
                recv_cash = qty * real_sell_price * (1 - FEE_HALF)
                cash += recv_cash
                
                pnl = (real_sell_price - positions[t]['entry_price']) / positions[t]['entry_price']
                
                send_telegram(f"🔄 *[가상 순환매 매도]*\n종목: {t}\n사유: 🔥 더 강한 주도주 발견 ({best_new['ticker']}와 RS 격차 초과)\n매도가: ${real_sell_price:.2f}\n기존 수익률: {pnl*100:+.2f}%")
                del positions[t]

        # 4. [STEP 3] 신규 대장주 진입 (당일 종가 즉시 매수 체결)
        for candidate in all_candidates:
            if len(positions) >= MAX_SLOTS: break # 2슬롯 제약 철저히 유지
            t = candidate['ticker']
            
            if t not in positions and candidate['signal_buy']:
                curr_price = candidate['price']
                
                # 가치 산정 및 2분할 자금 계산
                current_assets = cash + sum([p['qty'] * data_dict[pos_t]['Close'].iloc[-1] for pos_t, p in positions.items() if pos_t in data_dict])
                target_unit = current_assets * (1 / MAX_SLOTS) * 0.995
                alloc = min(cash, target_unit)
                
                real_buy_price = curr_price * (1 + SLIPPAGE)
                qty = int(alloc // (real_buy_price * 1.0005))
                cost = qty * real_buy_price * (1 + FEE_HALF)
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t] = {'qty': qty, 'entry_price': real_buy_price}
                    send_telegram(f"🟩 *[가상 매수 알림]*\n종목: {t}\n매수평균가(슬리피지 포함): ${real_buy_price:.2f}\n수량: {qty}주\n투입 자금: ${cost:.2f}")

    save_account(cash, positions)
    
    # 5. 실시간 자산현황 브리핑
    total_val = cash + sum([p['qty'] * data_dict[pos_t]['Close'].iloc[-1] for pos_t, p in positions.items() if pos_t in data_dict])
    print(f"💰 현재 가상 자산 총액: ${total_val:.2f} | 현금: ${cash:.2f} | 보유 슬롯: {len(positions)}/{MAX_SLOTS}")

def is_market_open():
    """현재 시간이 미국 본장 운영 시간인지 판별합니다. (한국시간 22:30 ~ 06:00 / 섬머타임 유연 대응)"""
    now = datetime.now()
    if now.weekday() >= 5: return False # 주말 차단
    
    current_time = now.strftime("%H:%M")
    # 🌟 윈터타임 마감 연장 유연성 확보를 위해 06:00까지 안전 마진 확보
    if "22:30" <= current_time or current_time <= "06:00":
        return True
    return False

if __name__ == "__main__":
    print("🚀 Ver 6.0 '알파 익스프레스' 실시간 순환매 모의투자 엔진이 시작되었습니다.")
    print("📱 설정된 텔레그램으로 엔진 가동 확인 메시지를 전송합니다.")
    send_telegram("🤖 *Ver 6.0 '알파 익스프레스' 라이브 모의투자 엔진 가동 시작!*\n(조건: 2슬롯 제한 + 30m 종가 실시간 체결 고도화)")
    
    while True:
        try:
            if is_market_open():
                check_market_and_trade()
                time.sleep(1800) # 30분 대기
            else:
                print(f"💤 현재 시간 {datetime.now().strftime('%H:%M')}, 미국 본장 시간이 아니므로 대기 중...")
                time.sleep(300)  # 장외 시간엔 5분간 휴식
        except KeyboardInterrupt:
            print("\n👋 프로그램을 안전하게 종료합니다.")
            sys.exit()
        except Exception as e:
            print(f"💥 루프 에러 발생: {e}")
            time.sleep(60) # 🌟 오타 수정: time.append(60) -> time.sleep(60)으로 정상 대기 조치