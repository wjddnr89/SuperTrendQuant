import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# ==========================================
# ⚙️ 국장 백테스트 표준 고정 조건 설정 구역
# ==========================================
INITIAL_CASH = 10000000.0  # 시작 자산 (원화 1,000만 원)
MAX_SLOTS = 1              # 1종목 몰빵 복리 구조

HURDLE_ATR_MULT = 1.25     # 동적 허들 배수
ALLOW_LATE_CHASE = True    # 추격 매수 허용

RS_PERIOD = 130            # 30분봉 130개 기준 RS 연산
FEE_HALF = 0.00225         # 편도 수수료 (0.225%)
SLIPPAGE = 0.0005          # 편도 슬리피지 (0.05%)

# 🔬 실험군 3개 모드 동시 가동
EXPERIMENTAL_MODES = ['None', 'KOSPI_Single_Filter', 'Dual_Filter']

# 🎯 [정밀 하드코딩] 유저님 지정 30종목 실제 소속 시장 매핑 완벽 반영
FILTER_MAP = {
    # 코스피 소속 우량주 (14종목) -> 실제 .KS로 데이터 수집
    "066570.KS": "KOSPI",  # LG전자
    "009150.KS": "KOSPI",  # 삼성전기
    "000660.KS": "KOSPI",  # SK하이닉스
    "011070.KS": "KOSPI",  # LG이노텍
    "005380.KS": "KOSPI",  # 현대차
    "005930.KS": "KOSPI",  # 삼성전자
    "007660.KS": "KOSPI",  # 이구산업
    "017670.KS": "KOSPI",  # SK텔레콤
    "034020.KS": "KOSPI",  # 두산에너빌리티
    "006800.KS": "KOSPI",  # 미래에셋증권
    "010140.KS": "KOSPI",  # 삼성중공업
    "012450.KS": "KOSPI",  # 한화에어로스페이스
    "267260.KS": "KOSPI",  # HD현대일렉트릭
    "010120.KS": "KOSPI",  # LS
    
    # 코스닥 소속 주도주 (16종목) -> 실제 .KQ로 데이터 수집
    "010170.KQ": "KOSDAQ", # 대한항공 (KQ 임시 우회)
    "058610.KQ": "KOSDAQ", # 에스피지
    "222800.KQ": "KOSDAQ", # 심텍
    "189300.KQ": "KOSDAQ", # 인텔리안테크
    "032820.KQ": "KOSDAQ", # 우리기술
    "478340.KQ": "KOSDAQ", # 삼현
    "086520.KQ": "KOSDAQ", # 에코프로머티
    "240810.KQ": "KOSDAQ", # 원텍
    "042700.KQ": "KOSDAQ", # 한미반도체
    "336260.KQ": "KOSDAQ", # 두산퓨어셀
    "056190.KQ": "KOSDAQ", # 에스에프에이
    "052020.KQ": "KOSDAQ", # 에스티큐브
    "064290.KQ": "KOSDAQ", # 로보스타
    "489790.KQ": "KOSDAQ", # 신규상장주
    "347700.KQ": "KOSDAQ", # 이오플로우
    "031980.KQ": "KOSDAQ"  # 피에스케이
}
# ==========================================

print("📥 야후 파이낸스 국장 [최근 1개월] 정밀 하드코딩 데이터 통합 다운로드 중...")

universe = list(FILTER_MAP.keys())
benchmarks = ["^KS11", "^KQ11"]  # 코스피 지수, 코스닥 지수 모두 수집

raw_30m = yf.download(universe + ["^KS11"], period="30d", interval="30m", progress=False)
raw_1h = yf.download(benchmarks, period="30d", interval="1h", progress=False)

ks_close_30m = raw_30m['Close']['^KS11'].dropna()
ks_ret = ks_close_30m.pct_change(RS_PERIOD)

def calculate_supertrend(df, period=7, multiplier=3.0):
    if df.empty or len(df) < period: return df
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

# 3. 종목별 데이터 가공 및 검증
data_dict = {}
valid_ticker_indices = []

for ticker in universe:
    df = pd.DataFrame({
        'Open': raw_30m['Open'][ticker], 'High': raw_30m['High'][ticker],
        'Low': raw_30m['Low'][ticker], 'Close': raw_30m['Close'][ticker]
    }).dropna()
    
    if df.empty or len(df) < (RS_PERIOD + 5): 
        print(f"⚠️ [제외] {ticker} 종목은 한 달 데이터가 불완전하여 제외됩니다.")
        continue
        
    df = calculate_supertrend(df, period=7, multiplier=3.0)
    df['RS'] = df['Close'].pct_change(RS_PERIOD) - ks_ret
    df = df.dropna()
    
    if not df.empty:
        data_dict[ticker] = df
        valid_ticker_indices.append(df.index)

all_timestamps = sorted(list(set(ks_close_30m.index).intersection(*valid_ticker_indices)))
print(f"📊 최종 동기화된 테스트 봉 개수: {len(all_timestamps)}개 (최근 1개월장)")

final_summary = {}

# 🔄 백테스트 엔진 가동
for mode in EXPERIMENTAL_MODES:
    print(f"⚙️ 국장 모드 [{mode}] 시뮬레이션 연산 중...")
    
    # 1시간봉 지수 추세 사전 연산
    upper_trends = {}
    for bench in benchmarks:
        df_bench = pd.DataFrame({'Open': raw_1h['Open'][bench], 'High': raw_1h['High'][bench], 'Low': raw_1h['Low'][bench], 'Close': raw_1h['Close'][bench]}).dropna()
        upper_trends[bench] = calculate_supertrend(df_bench, period=10, multiplier=3.0)['Trend']

    cash = INITIAL_CASH
    positions = {}
    history = []
    total_trades_count = 0
    holding_durations = []

    for idx, ts in enumerate(all_timestamps):
        if idx < 5: continue 
        
        current_portfolio_value = cash
        for t, pos in positions.items():
            current_portfolio_value += pos['qty'] * data_dict[t]['Close'].loc[ts]

        all_candidates = []
        for t, df in data_dict.items():
            if ts not in df.index: continue
            
            curr_trend = df['Trend'].loc[ts]
            loc_idx = df.index.get_loc(ts)
            if loc_idx < 1: continue
            prev_trend = df['Trend'].iloc[loc_idx - 1]
            
            rs_score = df['RS'].loc[ts]
            atr_pct = df['ATR_pct'].loc[ts]
            price = df['Close'].loc[ts]
            
            # 💡 [필터링 분기 핵심 구현]
            if mode == 'None':
                market_signal = 1  # 필터 없음 (무조건 패스)
            elif mode == 'KOSPI_Single_Filter':
                past_sigs = upper_trends["^KS11"][upper_trends["^KS11"].index <= ts]
                market_signal = past_sigs.iloc[-1] if not past_sigs.empty else 1  # 무조건 코스피 지수만 추종
            elif mode == 'Dual_Filter':
                belong_bench = "^KS11" if FILTER_MAP[t] == "KOSPI" else "^KQ11"
                past_sigs = upper_trends[belong_bench][upper_trends[belong_bench].index <= ts]
                market_signal = past_sigs.iloc[-1] if not past_sigs.empty else 1  # 소속 시장 지수 추종
            
            if curr_trend == 1 and market_signal == 1:
                is_buy_signal = True if ALLOW_LATE_CHASE else (prev_trend == -1)
                all_candidates.append({
                    'ticker': t, 'rs': rs_score, 'signal_buy': is_buy_signal, 'price': price, 'atr_pct': atr_pct
                })
                    
        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)
        
        # STEP 1: 추세 이탈 청산
        for t in list(positions.keys()):
            if ts not in data_dict[t].index: continue
            curr_trend = data_dict[t]['Trend'].loc[ts]
            if curr_trend == -1:
                price = data_dict[t]['Close'].loc[ts]
                real_sell_price = price * (1 - SLIPPAGE)
                cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                holding_durations.append(idx - positions[t]['entry_idx'])
                del positions[t]

        # STEP 2: 순환매 교체 매도
        if all_candidates:
            for t in list(positions.keys()):
                current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
                available_news = [c for c in all_candidates if c['ticker'] not in positions]
                if not available_news: continue
                best_new = available_news[0]
                dynamic_hurdle = best_new['atr_pct'] * HURDLE_ATR_MULT
                
                if best_new['rs'] - current_rs > dynamic_hurdle:
                    price = data_dict[t]['Close'].loc[ts]
                    real_sell_price = price * (1 - SLIPPAGE)
                    cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                    holding_durations.append(idx - positions[t]['entry_idx'])
                    del positions[t]

            # STEP 3: 매수 진입
            for candidate in all_candidates:
                if len(positions) >= MAX_SLOTS: break
                t = candidate['ticker']
                if t not in positions and candidate['signal_buy']:
                    price = candidate['price']
                    target_unit = current_portfolio_value * 0.995
                    alloc = min(cash, target_unit)
                    real_buy_price = price * (1 + SLIPPAGE)
                    qty = int(alloc // (real_buy_price * 1.0005))
                    cost = qty * real_buy_price * (1 + FEE_HALF)
                    
                    if qty > 0 and cash >= cost:
                        cash -= cost
                        positions[t] = {'qty': qty, 'entry_price': real_buy_price, 'entry_idx': idx}
                        total_trades_count += 1

        history.append({'timestamp': ts, 'total_value': current_portfolio_value})

    for t, pos in positions.items():
        holding_durations.append(len(all_timestamps) - 1 - pos['entry_idx'])

    res_df = pd.DataFrame(history).set_index('timestamp')
    res_df['Peak'] = res_df['total_value'].cummax()
    res_df['Drawdown'] = (res_df['total_value'] - res_df['Peak']) / res_df['Peak']

    total_return = (res_df['total_value'].iloc[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    mdd = res_df['Drawdown'].min() * 100
    
    start_ts, end_ts = res_df.index[0], res_df.index[-1]
    universe_returns = []
    for ticker in data_dict.keys():
        universe_returns.append((data_dict[ticker]['Close'].loc[end_ts] - data_dict[ticker]['Close'].loc[start_ts]) / data_dict[ticker]['Close'].loc[start_ts] * 100)
    universe_bh_avg_return = np.mean(universe_returns)
    avg_holding_bars = np.mean(holding_durations) if holding_durations else 0

    final_summary[mode] = {
        '수익률': f"{total_return:+.2f}%",
        'MDD': f"{mdd:.2f}%",
        '진짜알파': f"{total_return - universe_bh_avg_return:+.2f}%",
        '매수횟수': f"{total_trades_count}회",
        '평균보유봉': f"{avg_holding_bars:.1f}개"
    }

# ==================================================
# 📊 [최종 출력] 백테스트 결과표
# ==================================================
print("\n" + "="*70)
print("🏆 국장 고정 30종목 [정밀 매핑 기반 3대 필터 비교] 최종 성적표")
print("="*70)
summary_df = pd.DataFrame(final_summary).T
print(summary_df.to_string())
print("="*70)
print("💡 [None]: 상위 필터 없음 (30분봉 단일 몰빵 순환매)")
print("💡 [KOSPI_Single_Filter]: 시장 불문 무조건 코스피 지수 1시간봉 필터")
print("💡 [Dual_Filter]: 정밀 매핑된 소속 지수(코스피/코스닥) 1시간봉 분리 필터")
print("="*70)