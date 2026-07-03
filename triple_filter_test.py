import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# ==========================================
# ⚙️ 핵심 백테스트 파라미터 설정 구역 (완벽 통일)
# ==========================================
INITIAL_CASH = 10000.0
MAX_SLOTS = 3          

HURDLE_ATR_MULT = 1.25  
ALLOW_LATE_CHASE = True 

RS_PERIOD = 130        
FEE_HALF = 0.00225     
SLIPPAGE = 0.0005      

# 🔬 실험군에 'Triple' 필터를 추가하여 1:1 진검승부를 봅니다.
EXPERIMENTAL_TIMEFRAMES = ['None', '30m', '1h', '2h', '3h', '4h', 'Triple']

# 🎯 트리플 필터 전용 종목-지수 매핑 맵
FILTER_MAP = {
    "TQQQ": "QQQ", "SOXL": "QQQ", "SOXS": "QQQ", "NVDA": "QQQ", 
    "AMD": "QQQ", "MU": "QQQ", "MRVL": "QQQ", "TSLA": "QQQ", "TSM": "QQQ",
    "GEV": "SPY", "VRT": "SPY", "LLY": "SPY", "UNH": "SPY", "MDT": "SPY",
    "RKLB": "IWM", "RDW": "IWM", "PL": "IWM", "OUST": "IWM", 
    "AEHR": "IWM", "MP": "IWM",
    "FN": "SPY", "COHR": "SPY", "AXON": "SPY", "RZLV": "IWM"
}
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
universe = list(FILTER_MAP.keys())
benchmarks = ["QQQ", "SPY", "IWM"]

# 메인 30분봉 데이터 다운로드
raw_30m = yf.download(universe + benchmarks, period="60d", interval="30m", progress=False)
# 상위 필터용 1시간봉 데이터 다운로드 (벤치마크 3개 전부 다운)
raw_1h = yf.download(benchmarks, period="60d", interval="1h", progress=False)

main_tz = raw_30m.index.tz
qqq_close = raw_30m['Close']['QQQ'].dropna()
qqq_ret = qqq_close.pct_change(RS_PERIOD)

# 종목별 30분봉 오리지널 SuperTrend(EMA 기반) 및 RS 연산 적재
data_dict = {}
for ticker in universe:
    df = pd.DataFrame({
        'Open': raw_30m['Open'][ticker], 'High': raw_30m['High'][ticker],
        'Low': raw_30m['Low'][ticker], 'Close': raw_30m['Close'][ticker]
    }).dropna()
    if df.empty: continue
    mult = 4.5 if ticker in ["SOXL", "SOXS"] else 3.0
    df = calculate_supertrend(df, period=7, multiplier=mult)
    df['RS'] = df['Close'].pct_change(RS_PERIOD) - qqq_ret
    data_dict[ticker] = df

all_timestamps = sorted(list(set(qqq_close.index).intersection(*[df.index for df in data_dict.values()])))
final_summary = {}

# 🔄 각 타임프레임 및 트리플 필터 순회 시뮬레이션
for tf in EXPERIMENTAL_TIMEFRAMES:
    print(f"⚙️ 필터 [{tf}] 적용 시뮬레이션 엔진 연산 중...")
    
    # 지수별 시그널 저장용 딕셔너리
    upper_trends = {}
    
    # [케이스 1] 30m 단일 QQQ 필터
    if tf == '30m' and not raw_30m.empty:
        df_qqq = pd.DataFrame({'Open': raw_30m['Open']['QQQ'], 'High': raw_30m['High']['QQQ'], 'Low': raw_30m['Low']['QQQ'], 'Close': raw_30m['Close']['QQQ']}).dropna()
        upper_trends['QQQ'] = calculate_supertrend(df_qqq, period=10, multiplier=3.0)['Trend']
        
    # [케이스 2] 1h 단일 QQQ 필터
    elif tf == '1h' and not raw_1h.empty:
        df_qqq = pd.DataFrame({'Open': raw_1h['Open']['QQQ'], 'High': raw_1h['High']['QQQ'], 'Low': raw_1h['Low']['QQQ'], 'Close': raw_1h['Close']['QQQ']}).dropna()
        upper_trends['QQQ'] = calculate_supertrend(df_qqq, period=10, multiplier=3.0)['Trend']
        
    # [케이스 3] 2h, 3h, 4h 리샘플링 QQQ 필터
    elif tf in ['2h', '3h', '4h'] and not raw_1h.empty:
        df_hourly = pd.DataFrame({'Open': raw_1h['Open']['QQQ'], 'High': raw_1h['High']['QQQ'], 'Low': raw_1h['Low']['QQQ'], 'Close': raw_1h['Close']['QQQ']}).dropna()
        df_upper = df_hourly.resample(tf).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        df_upper = calculate_supertrend(df_upper, period=10, multiplier=3.0)
        if df_upper.index.tz is None and df_hourly.index.tz is not None:
            df_upper.index = df_upper.index.tz_localize(df_hourly.index.tz)
        upper_trends['QQQ'] = df_upper['Trend']
        
    # 🔥 [케이스 4] 유저님이 지시하신 오리지널 베이스에 트리플 필터 이식
    elif tf == 'Triple' and not raw_1h.empty:
        for bench in benchmarks:
            df_bench = pd.DataFrame({'Open': raw_1h['Open'][bench], 'High': raw_1h['High'][bench], 'Low': raw_1h['Low'][bench], 'Close': raw_1h['Close'][bench]}).dropna()
            upper_trends[bench] = calculate_supertrend(df_bench, period=10, multiplier=3.0)['Trend']

    # 자산 및 변수 초기화
    cash = INITIAL_CASH
    positions = {}
    history = []
    total_trades_count = 0
    holding_durations = []

    for idx, ts in enumerate(all_timestamps):
        if idx < RS_PERIOD + 1: continue
        
        current_portfolio_value = cash
        for t, pos in positions.items():
            current_portfolio_value += pos['qty'] * data_dict[t]['Close'].loc[ts]
            
        # STEP 1: 종목별 상위 필터 시그널 동적 매칭
        # 기본값은 무조건 프리패스(None), 필터가 켜지면 매칭된 지수 추적
        upper_market_signal = 1 
        
        if tf != 'None' and upper_trends:
            if tf == 'Triple':
                # 트리풀 모드일 때는 종목마다 매칭된 지수를 찾아가서 시그널을 확인해야 하므로 아래 후보 루프에서 실시간 처리합니다.
                pass 
            else:
                # QQQ 단일 지수 필터 모드일 때
                past_sigs = upper_trends['QQQ'][upper_trends['QQQ'].index <= ts]
                if not past_sigs.empty: upper_market_signal = past_sigs.iloc[-1]

        # 진입 후보 탐색
        all_candidates = []
        for t, df in data_dict.items():
            curr_trend = df['Trend'].loc[ts]
            prev_trend = df['Trend'].iloc[df.index.get_loc(ts)-1]
            rs_score = df['RS'].loc[ts]
            atr_pct = df['ATR_pct'].loc[ts]
            price = df['Close'].loc[ts]
            
            # 🔥 트리플 필터 모드일 때만 종목별 1시간봉 지수 시그널 실시간 할당
            if tf == 'Triple':
                belong_bench = FILTER_MAP[t]
                past_sigs = upper_trends[belong_bench][upper_trends[belong_bench].index <= ts]
                final_sig = past_sigs.iloc[-1] if not past_sigs.empty else 1
            else:
                final_sig = upper_market_signal
            
            if curr_trend == 1 and final_sig == 1:
                is_buy_signal = True if ALLOW_LATE_CHASE else (prev_trend == -1)
                all_candidates.append({
                    'ticker': t, 'rs': rs_score, 'signal_buy': is_buy_signal, 'price': price, 'atr_pct': atr_pct
                })
                    
        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)
        
        # STEP 2: 청산 (30분봉 추세 무너지면 전량 컷)
        for t in list(positions.keys()):
            curr_trend = data_dict[t]['Trend'].loc[ts]
            if curr_trend == -1:
                price = data_dict[t]['Close'].loc[ts]
                real_sell_price = price * (1 - SLIPPAGE)
                cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                holding_durations.append(idx - positions[t]['entry_idx'])
                del positions[t]

        # STEP 3: 순환매 강제 스왑 매도 (대장주 교체 로직)
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

            # STEP 4: 순환매 빈자리 새로 채우기 매수
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

    # 통계 처리
    res_df = pd.DataFrame(history).set_index('timestamp')
    res_df['Peak'] = res_df['total_value'].cummax()
    res_df['Drawdown'] = (res_df['total_value'] - res_df['Peak']) / res_df['Peak']

    total_return = (res_df['total_value'].iloc[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    mdd = res_df['Drawdown'].min() * 100
    
    start_ts, end_ts = res_df.index[0], res_df.index[-1]
    universe_returns = []
    for ticker in universe:
        if ticker in data_dict:
            universe_returns.append((data_dict[ticker]['Close'].loc[end_ts] - data_dict[ticker]['Close'].loc[start_ts]) / data_dict[ticker]['Close'].loc[start_ts] * 100)
    universe_bh_avg_return = np.mean(universe_returns)
    avg_holding_bars = np.mean(holding_durations) if holding_durations else 0

    final_summary[tf] = {
        '수익률': f"{total_return:+.2f}%",
        'MDD': f"{mdd:.2f}%",
        '진짜알파': f"{total_return - universe_bh_avg_return:+.2f}%",
        '매수횟수': f"{total_trades_count}회",
        '평균보유봉': f"{avg_holding_bars:.1f}개"
    }

# ==================================================
# 📊 [최종 출력] 트리플 필터가 결합된 종합 분석표
# ==================================================
summary_df = pd.DataFrame(final_summary).T
print("\n" + "="*60)
print("🏆 완벽 조건 통일: 트리플 필터 이식 백테스트 최종 성적표")
print("="*60)
print(summary_df.to_string())
print("="*60)
print("💡 [None]: 오리지널 30분봉 단일 전략")
print("💡 [30m~4h]: 해당 시간봉 QQQ 정배열 시에만 30분봉 진입 허용 필터")
print("💡 [Triple]: 각 종목 고유 지수(QQQ, SPY, IWM) 1시간봉 정배열 진입 필터")
print("="*60)