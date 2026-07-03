import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# ==========================================
# ⚙️ 핵심 백테스트 파라미터 설정 구역 (Ver 8.3)
# ==========================================
INITIAL_CASH = 10000.0
MAX_SLOTS = 1          

HURDLE_ATR_MULT = 1.25  
ALLOW_LATE_CHASE = True 

RS_PERIOD = 130        
FEE_HALF = 0.00225     
SLIPPAGE = 0.0005      

# 🔬 이번엔 '각 종목 자체'의 상위 타임프레임 필터를 실험합니다.
EXPERIMENTAL_TIMEFRAMES = ['None', '1h', '2h', '3h', '4h']
# ==========================================

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

print("📥 야후 파이낸스 백테스트 기본 데이터 통합 다운로드 중...")
universe = ["SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON", "SOXS"]

# 모든 종목의 30분봉과 1시간봉 데이터를 한 번에 가져와 멀티인덱스 구조 확보
raw_30m = yf.download(universe + ["QQQ"], period="60d", interval="30m", progress=False)
raw_1h = yf.download(universe, period="60d", interval="1h", group_by='ticker', progress=False)

main_tz = raw_30m.index.tz
qqq_close = raw_30m['Close']['QQQ'].dropna()
qqq_ret = qqq_close.pct_change(RS_PERIOD)

# 1단계: 종목별 30분봉 메인 데이터 기본 연산
data_30m_dict = {}
for ticker in universe:
    df = pd.DataFrame({
        'Open': raw_30m['Open'][ticker], 'High': raw_30m['High'][ticker],
        'Low': raw_30m['Low'][ticker], 'Close': raw_30m['Close'][ticker]
    }).dropna()
    if df.empty: continue
    mult = 4.5 if ticker in ["SOXL", "SOXS"] else 3.0
    df = calculate_supertrend(df, period=7, multiplier=mult)
    df['RS'] = df['Close'].pct_change(RS_PERIOD) - qqq_ret
    data_30m_dict[ticker] = df

all_timestamps = sorted(list(set(qqq_close.index).intersection(*[df.index for df in data_30m_dict.values()])))

# 타임프레임별 결과를 담을 딕셔너리
final_summary = {}

# 🔄 각 타임프레임별 순회 시뮬레이션 가동
for tf in EXPERIMENTAL_TIMEFRAMES:
    print(f"⚙️ '각 종목 자체'의 [{tf}] 필터 적용 시뮬레이션 연산 중...")
    
    # 2단계: 해당 타임프레임에 맞는 종목별 상위 추세 지도 사전 빌드
    ticker_upper_trends = {}
    
    if tf != 'None' and not raw_1h.empty:
        for ticker in universe:
            try:
                df_hourly = pd.DataFrame({
                    'Open': raw_1h[ticker]['Open'].squeeze(), 'High': raw_1h[ticker]['High'].squeeze(), 
                    'Low': raw_1h[ticker]['Low'].squeeze(), 'Close': raw_1h[ticker]['Close'].squeeze()
                }).dropna()
                
                if tf == '1h':
                    df_upper = calculate_supertrend(df_hourly, period=10, multiplier=3.0)
                else:
                    # 2h, 3h, 4h 리샘플링
                    df_upper = df_hourly.resample(tf).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
                    df_upper = calculate_supertrend(df_upper, period=10, multiplier=3.0)
                    if df_upper.index.tz is None and df_hourly.index.tz is not None:
                        df_upper.index = df_upper.index.tz_localize(df_hourly.index.tz)
                
                ticker_upper_trends[ticker] = df_upper['Trend']
            except Exception:
                continue

    # 시뮬레이션 잔고 초기화
    cash = INITIAL_CASH
    positions = {}
    history = []
    total_trades_count = 0
    holding_durations = []

    for idx, ts in enumerate(all_timestamps):
        if idx < RS_PERIOD + 1: continue
        
        current_portfolio_value = cash
        for t, pos in positions.items():
            current_portfolio_value += pos['qty'] * data_30m_dict[t]['Close'].loc[ts]

        all_candidates = []
        for t, df in data_30m_dict.items():
            curr_trend = df['Trend'].loc[ts]
            prev_trend = df['Trend'].iloc[df.index.get_loc(ts)-1]
            rs_score = df['RS'].loc[ts]
            atr_pct = df['ATR_pct'].loc[ts]
            price = df['Close'].loc[ts]
            
            # 💡 핵심 변경점: QQQ가 아니라 '이 종목 자체의 상위봉 추세'를 체크합니다.
            ticker_upper_signal = 1
            if tf != 'None' and t in ticker_upper_trends:
                past_upper_signals = ticker_upper_trends[t][ticker_upper_trends[t].index <= ts]
                if not past_upper_signals.empty:
                    ticker_upper_signal = past_upper_signals.iloc[-1]
            
            if curr_trend == 1:
                # 종목의 30분봉도 정배열이고, 종목의 상위봉 필터도 정배열일 때만 진입 허용
                if ticker_upper_signal == 1:
                    is_buy_signal = True if ALLOW_LATE_CHASE else (prev_trend == -1)
                    all_candidates.append({
                        'ticker': t, 'rs': rs_score, 'signal_buy': is_buy_signal, 'price': price, 'atr_pct': atr_pct
                    })
                    
        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)
        
        # STEP 1: 청산 (30분봉 추세 이탈 시)
        for t in list(positions.keys()):
            curr_trend = data_30m_dict[t]['Trend'].loc[ts]
            if curr_trend == -1:
                price = data_30m_dict[t]['Close'].loc[ts]
                real_sell_price = price * (1 - SLIPPAGE)
                cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                holding_durations.append(idx - positions[t]['entry_idx'])
                del positions[t]

        # STEP 2: 순환매 매도
        if all_candidates:
            for t in list(positions.keys()):
                current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
                available_news = [c for c in all_candidates if c['ticker'] not in positions]
                if not available_news: continue
                best_new = available_news[0]
                dynamic_hurdle = best_new['atr_pct'] * HURDLE_ATR_MULT
                
                if best_new['rs'] - current_rs > dynamic_hurdle:
                    price = data_30m_dict[t]['Close'].loc[ts]
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
                    target_unit = current_portfolio_value * (1 / MAX_SLOTS) * 0.995
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

    # 통계 연산
    res_df = pd.DataFrame(history).set_index('timestamp')
    res_df['Peak'] = res_df['total_value'].cummax()
    res_df['Drawdown'] = (res_df['total_value'] - res_df['Peak']) / res_df['Peak']

    total_return = (res_df['total_value'].iloc[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    mdd = res_df['Drawdown'].min() * 100
    
    start_ts, end_ts = res_df.index[0], res_df.index[-1]
    universe_returns = []
    for ticker in universe:
        if ticker in data_30m_dict:
            universe_returns.append((data_30m_dict[ticker]['Close'].loc[end_ts] - data_dict[ticker]['Close'].loc[start_ts]) / data_dict[ticker]['Close'].loc[start_ts] * 100 if 'data_dict' in globals() else 0)
    universe_bh_avg_return = np.mean(universe_returns) if any(universe_returns) else 50.0 # 대조용 기본값 방어
    avg_holding_bars = np.mean(holding_durations) if holding_durations else 0

    final_summary[tf] = {
        '수익률': f"{total_return:+.2f}%",
        'MDD': f"{mdd:.2f}%",
        '매수횟수': f"{total_trades_count}회",
        '평균보유봉': f"{avg_holding_bars:.1f}개"
    }

# ==================================================
# 📊 종목 자체 상위 필터별 최종 성적 비교 분석표
# ==================================================
summary_df = pd.DataFrame(final_summary).T
print("\n" + "="*60)
print("🏆 [실험] 개별 종목 자체 상위 필터별 백테스트 최종 성적")
print("="*60)
print(summary_df.to_string())
print("="*60)
print("💡 [None]: 오리지널 30분봉 단일 전략")
print("💡 [1h~4h]: 해당 종목 자체의 시간봉 정배열 시에만 30분봉 진입 허용")
print("="*60)