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
# ⚙️ Ver 7.0 'The One' 핵심 설정 구역
# ==========================================
TELEGRAM_TOKEN = "8580742809:AAGpnuw6pZNpWwf5dWxMEIrrl2gbasDwa6M"
TELEGRAM_CHAT_ID = "8323058445"

# 가상 계좌 상태 파일 경로 (프로그램이 꺼졌다가 켜져도 잔고 유지)
ACCOUNT_FILE = "virtual_account_v7.csv"
INITIAL_CASH = 10000.0

# 🥇 Ver 7.0 핵심 파라미터 고정 (그리드 서치 압도적 1위 설정)
MAX_SLOTS = 1           # 극강의 원탑 주도주 집중 투자
HURDLE_ATR_MULT = 1.25  # 장중 노이즈 원천 차단 동적 배수
ALLOW_LATE_CHASE = True # 실시간 대장주 머리꼭대기 추격 매수 허용
RS_PERIOD = 130         # 10영업일 진성 모멘텀 필터 (30분봉 130개)

# 비용 함수
FEE_HALF = 0.00225     # 편도 수수료 (0.225%)
SLIPPAGE = 0.0005      # 편도 슬리피지 (0.05%)

FULL_UNIVERSE = [
    "SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "MRVL",
    "TSLA", "AEHR", "AXON", "SOXS", "LLY", "UNH", "MDT", "RZLV", "FN", "AMD", "COHR", "MP", "TSM"
]
# ==========================================

def send_telegram(message):
    """텔레그램으로 실시간 알림을 쏩니다."""
    print(f"📢 [Telegram] {message}")
    if "YOUR_" in TELEGRAM_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=5)
    except Exception as e: print(f"❌ 알림 전송 실패: {e}")

def calculate_supertrend(df, period=7, multiplier=3.0):
    high, low, close = df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze()
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    df['ATR_pct'] = atr / close  # 동적 허들용 변동성 비율 계산
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
    """미국 본장 중에 30분마다 돌며 Ver 7.0 핵심 연산 및 가상 매매를 집행합니다."""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n🔄 [{now_str}] Ver 7.0 실시간 모니터링 사이클 가동...")
    
    # 1. 데이터 수집 (야후 파이낸스 30분봉)
    try:
        raw_data = yf.download(FULL_UNIVERSE + ["QQQ"], period="15d", interval="30m", progress=False)
        qqq_close = raw_data['Close']['QQQ'].dropna()
        qqq_ret = qqq_close.pct_change(RS_PERIOD)
    except Exception as e:
        print(f"❌ 데이터 로드 에러: {e}")
        return

    cash, positions = load_account()
    data_dict = {}
    all_candidates = []
    
    # 2. 유니버스 데이터 파싱 및 지표 계산 (SuperTrend, RS, ATR)
    for ticker in FULL_UNIVERSE:
        df = pd.DataFrame({
            'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
            'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker]
        }).dropna()
        if df.empty: continue
        
        mult = 4.5 if ticker in ["SOXL", "SOXS"] else 3.0
        df = calculate_supertrend(df, period=7, multiplier=mult)
        df['RS'] = df['Close'].pct_change(RS_PERIOD) - qqq_ret
        data_dict[ticker] = df
        
        # 최신 확정봉 시그널 추출
        curr_trend = df['Trend'].iloc[-1]
        prev_trend = df['Trend'].iloc[-2]
        rs_score = df['RS'].iloc[-1]
        atr_pct = df['ATR_pct'].iloc[-1]
        price = df['Close'].iloc[-1]
        
        if curr_trend == 1:
            # ALLOW_LATE_CHASE = True 이므로 상시 True, 아닐 경우 골든크로스 당일에만 진입
            is_buy_signal = True if ALLOW_LATE_CHASE else (prev_trend == -1)
            all_candidates.append({
                'ticker': ticker, 'rs': rs_score, 'signal_buy': is_buy_signal, 'price': price, 'atr_pct': atr_pct
            })

    # RS 등수대로 내림차순 정렬 (리더보드 빌드)
    all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)

    # 실시간 계좌 총 평가액 갱신
    current_total_assets = cash
    for pos_t, p in positions.items():
        current_total_assets += p['qty'] * data_dict[pos_t]['Close'].iloc[-1]

    # --------------------------------------------------
    # [청산 로직 1] 보유 종목 추세 이탈 (SuperTrend 데드크로스)
    # --------------------------------------------------
    for t in list(positions.keys()):
        curr_trend = data_dict[t]['Trend'].iloc[-1]
        if curr_trend == -1:
            curr_price = data_dict[t]['Close'].iloc[-1]
            qty = positions[t]['qty']
            
            real_sell_price = curr_price * (1 - SLIPPAGE)
            recv_cash = qty * real_sell_price * (1 - FEE_HALF)
            cash += recv_cash
            
            send_telegram(f"🚨 *[추세 이탈 전량 매도]*\n종목: {t}\n사유: SuperTrend 하락 전환\n매도가: ${curr_price:.2f}\n정산금액: ${recv_cash:.2f}")
            del positions[t]

    # --------------------------------------------------
    # [청산 로직 2] ★ Ver 7.0 핵심: 1.25 ATR 동적 허들 순환매 교체 매도
    # --------------------------------------------------
    if positions and all_candidates:
        for t in list(positions.keys()):
            current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
            
            # 내 종목을 제외하고 새로 치고 올라오는 원탑 후보 선별
            available_news = [c for c in all_candidates if c['ticker'] != t]
            if not available_news: continue
            best_new = available_news[0]
            
            # 동적 허들 장벽 연산: 1등의 변동성% * 1.25
            dynamic_hurdle = best_new['atr_pct'] * HURDLE_ATR_MULT
            
            # 왕좌 탈환 조건 확정 시 교체 매도
            if best_new['rs'] - current_rs > dynamic_hurdle:
                curr_price = data_dict[t]['Close'].iloc[-1]
                qty = positions[t]['qty']
                
                real_sell_price = curr_price * (1 - SLIPPAGE)
                recv_cash = qty * real_sell_price * (1 - FEE_HALF)
                cash += recv_cash
                
                send_telegram(f"🔄 *[동적 허들 순환매 매도]*\n구 종목: {t}\n신규 대장주: {best_new['ticker']}\n사유: 상대강도 격차 임계점({dynamic_hurdle*100:.2f}%) 돌파\n매도가: ${curr_price:.2f}")
                del positions[t]

    # --------------------------------------------------
    # [진입 로직 3] 신규 대장주 올인 (MAX_SLOTS = 1)
    # --------------------------------------------------
    if not positions and all_candidates:
        candidate = all_candidates[0] # 리더보드 원탑 1등 고정
        t = candidate['ticker']
        
        if candidate['signal_buy']:
            curr_price = candidate['price']
            # 계좌 가치 전체를 한 종목에 집중 투자 (수수료용 0.5% 버퍼 제외)
            target_unit = current_total_assets * 0.995
            alloc = min(cash, target_unit)
            
            real_buy_price = curr_price * (1 + SLIPPAGE)
            qty = int(alloc // (real_buy_price * 1.0005))
            cost = qty * real_buy_price * (1 + FEE_HALF)
            
            if qty > 0 and cash >= cost:
                cash -= cost
                positions[t] = {'qty': qty, 'entry_price': real_buy_price}
                send_telegram(f"🟩 *[신규 대장주 원탑 진입]*\n종목: {t}\n매수평균가: ${real_buy_price:.2f}\n수량: {qty}주\n투입자금: ${cost:.2f}")

    # 계좌 정보 파일 세이브
    save_account(cash, positions)
    
    # --------------------------------------------------
    # [주기 브리핑] 매 30분 사이클 실시간 현황 요약
    # --------------------------------------------------
    total_val = cash
    pos_info = "보유 종목 없음"
    if positions:
        t = list(positions.keys())[0]
        c_price = data_dict[t]['Close'].iloc[-1]
        total_val += positions[t]['qty'] * c_price
        pnl_pct = (c_price - positions[t]['entry_price']) / positions[t]['entry_price'] * 100
        pos_info = f"{t} ({positions[t]['qty']}주) | 수익률: {pnl_pct:+.2f}%"

    print(f"💰 [계좌 브리핑] 총자산: ${total_val:.2f} | 현금: ${cash:.2f} | 포지션: {pos_info}")

def is_market_open():
    """현재 시간이 미국 본장 운영 시간인지 판별합니다. (한국시간 서머타임 기준 22:30 ~ 05:00)"""
    now = datetime.now()
    if now.weekday() >= 5: return False # 토, 일 거래 제외
    
    current_time = now.strftime("%H:%M")
    if "22:30" <= current_time or current_time <= "05:00":
        return True
    return False

if __name__ == "__main__":
    print("🚀 Ver 7.0 'The One' 라이브 모의투자 엔진 로드 완료.")
    print("💤 미국 본장 개장 전까지 시스템은 대기 상태를 유지하며 텔레그램은 침묵합니다.")
    
    # 장이 열렸을 때 최초 1회만 시스템 기동 알림 문자를 쏘기 위한 추적 변수
    has_notified_start = False 
    
    while True:
        try:
            if is_market_open():
                # 미국 본장이 시작된 직후, 가동 알림 문자를 보낸 적이 없다면 이때 최초 발송
                if not has_notified_start:
                    send_telegram("🤖 *Ver 7.0 'The One' 집중 투자 라이브 모의엔진 가동 시작!*\n(MAX_SLOTS=1 / HURDLE=1.25 / 130 RS 필터)")
                    has_notified_start = True
                
                # 30분 주기 모니터링 및 트레이딩 연산 엔진 가동
                check_market_and_trade()
                time.sleep(1800)
            else:
                # 장외 시간일 때는 터미널 화면에만 로그를 찍고 텔레그램은 완벽히 침묵
                print(f"💤 현재 시간 {datetime.now().strftime('%H:%M')}, 미국 본장 시간이 아니므로 대기 중...")
                time.sleep(300) # 5분 간격 체크
                
        except KeyboardInterrupt:
            print("\n👋 프로그램을 안전하게 종료합니다.")
            sys.exit()
        except Exception as e:
            print(f"💥 시스템 루프 내 예외 에러 발생: {e}")
            time.sleep(60)