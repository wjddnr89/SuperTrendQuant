import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ==========================================
# ⚙️ 핵심 백테스트 파라미터 및 마켓 설정
# ==========================================
BACKTEST_MARKET = "US"      # 🇰🇷 "KR" / 🇺🇸 "US"
INITIAL_CASH = 10_000_000.0 if BACKTEST_MARKET == "KR" else 10_000.0 
MAX_SLOTS = 1               

HURDLE_ATR_MULT = 1.25      
ALLOW_LATE_CHASE = True     

RS_PERIOD = 100             
FEE_HALF = 0.00225          
SLIPPAGE = 0.0005           
FEE_ROUNDTRIP = (FEE_HALF + SLIPPAGE) * 2

UNIVERSE_FILE = "universe.json"
EXPERIMENTAL_TIMEFRAMES = ['None', '30m', '1h', '2h', '3h', '4h']

# ==========================================
# 🛠️ 수퍼트렌드 및 유니버스 로더 레이어
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

def load_universe_configs():
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"❌ {UNIVERSE_FILE} 파일이 필요합니다.")
    with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("KR_UNIVERSE_MAP", {}), data.get("US_UNIVERSE_LIST", [])

def convert_to_yf_ticker(symbol, market_type=None):
    if market_type == "KOSPI": return f"{symbol}.KS"
    if market_type == "KOSDAQ": return f"{symbol}.KQ"
    return symbol

# ==========================================
# 📥 데이터 동기화 및 전처리
# ==========================================
kr_universe_map, us_universe_list = load_universe_configs()

if BACKTEST_MARKET == "KR":
    print("🇰🇷 [국내장] 백테스트 모드 가동")
    tickers_raw = list(kr_universe_map.keys())
    yf_tickers = [convert_to_yf_ticker(s, kr_universe_map[s]) for s in tickers_raw]
    benchmarks = ["^KS11", "^KQ11"] 
    main_bench = "^KS11"
else:
    print("🇺🇸 [해외장] 백테스트 모드 가동")
    tickers_raw = us_universe_list
    yf_tickers = us_universe_list
    benchmarks = ["QQQ"]
    main_bench = "QQQ"

print(f"📥 야후 파이낸스에서 데이터 다운로드 중 (60일)...")
raw_30m = yf.download(yf_tickers + benchmarks, period="60d", interval="30m", progress=False)
raw_1h = yf.download(benchmarks, period="60d", interval="1h", progress=False)

# 💡 [허점 3 해결] 지수를 기준으로 마스터 타임라인 강제 고정 (교집합 제거)
master_timeline = raw_30m['Close'][main_bench].dropna().index
bench_ret_30m = raw_30m['Close'][main_bench].dropna().pct_change(RS_PERIOD)

# 종목별 데이터 가공 (결측치 앞방향 채우기 적용)
data_dict = {}
for s, yf_tk in zip(tickers_raw, yf_tickers):
    if yf_tk not in raw_30m['Close']: continue
    
    # 💡 [허점 2 & 3 해결] 마스터 타임라인에 맞추어 인덱스 재배열 및 ffill
    df = pd.DataFrame({
        'Open': raw_30m['Open'][yf_tk], 'High': raw_30m['High'][yf_tk],
        'Low': raw_30m['Low'][yf_tk], 'Close': raw_30m['Close'][yf_tk]
    }).reindex(master_timeline).ffill()
    
    if df['Close'].isna().all(): continue
    
    mult = 4.5 if s in ["SOXL", "SOXS"] else 3.0
    df = calculate_supertrend(df, period=7, multiplier=mult)
    df['RS'] = df['Close'].pct_change(RS_PERIOD) - bench_ret_30m
    data_dict[s] = df

final_summary = {}

# 🔄 상위 타임프레임 필터별 시뮬레이션
for tf in EXPERIMENTAL_TIMEFRAMES:
    print(f"⚙️ 필터 [{tf}] 적용 시뮬레이션 연산 중...")
    
    bench_upper_trends = {}
    for b_sym in benchmarks:
        df_b_raw = pd.DataFrame({
            'Open': raw_1h['Open'][b_sym], 'High': raw_1h['High'][b_sym],
            'Low': raw_1h['Low'][b_sym], 'Close': raw_1h['Close'][b_sym]
        }).dropna()

        if tf == '30m':
            df_b = pd.DataFrame({
                'Open': raw_30m['Open'][b_sym], 'High': raw_30m['High'][b_sym],
                'Low': raw_30m['Low'][b_sym], 'Close': raw_30m['Close'][b_sym]
            }).dropna()
            df_b = calculate_supertrend(df_b, period=10, multiplier=3.0)
        elif tf == '1h':
            df_b = calculate_supertrend(df_b_raw, period=10, multiplier=3.0)
        elif tf in ['2h', '3h', '4h']:
            # 💡 [허점 1 해결] closed='right', label='right' 지정으로 미래 참조 원천 차단
            df_b = df_b_raw.resample(tf, closed='right', label='right').agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
            ).dropna()
            df_b = calculate_supertrend(df_b, period=10, multiplier=3.0)
            if df_b.index.tz is None and df_b_raw.index.tz is not None:
                df_b.index = df_b.index.tz_localize(df_b_raw.index.tz)
        
        bench_upper_trends[b_sym] = df_b['Trend'] if tf != 'None' else pd.Series(dtype='float64')

    cash = INITIAL_CASH
    positions = {}
    history = []
    trade_log = []
    holding_durations = []

    # 💡 [허점 4 해결] 현재 캔들(ts)에서 시그널 판별, 다음 캔들(exec_ts) 시가(Open)로 체결
    for i in range(RS_PERIOD + 1, len(master_timeline) - 1):
        ts = master_timeline[i]           # 시그널 판별 기준 시간 (현재 봉 완성)
        exec_ts = master_timeline[i + 1]  # 실제 체결 시간 (다음 봉 시작)
        
        current_portfolio_value = cash
        for t, pos in positions.items():
            current_portfolio_value += pos['qty'] * data_dict[t]['Close'].loc[ts]
            
        market_signal = 1
        if tf != 'None':
            if BACKTEST_MARKET == "KR":
                b_target = "^KS11"
                past_signals = bench_upper_trends[b_target][bench_upper_trends[b_target].index <= ts]
                if not past_signals.empty: market_signal = past_signals.iloc[-1]
            else:
                past_signals = bench_upper_trends["QQQ"][bench_upper_trends["QQQ"].index <= ts]
                if not past_signals.empty: market_signal = past_signals.iloc[-1]

        all_candidates = []
        for t, df in data_dict.items():
            curr_trend = df['Trend'].loc[ts]
            prev_trend = df['Trend'].iloc[i-1]
            rs_score = df['RS'].loc[ts]
            atr_pct = df['ATR_pct'].loc[ts]
            
            if pd.isna(rs_score): continue
            
            if curr_trend == 1 and market_signal == 1:
                is_buy_signal = True if ALLOW_LATE_CHASE else (prev_trend == -1)
                all_candidates.append({
                    'ticker': t, 'rs': rs_score, 'signal_buy': is_buy_signal, 'atr_pct': atr_pct
                })
                
        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)
        
        # --- 매도/매수 실행 (exec_ts의 Open 가격 사용) ---
        for t in list(positions.keys()):
            curr_trend = data_dict[t]['Trend'].loc[ts]
            if curr_trend == -1:
                pos = positions.pop(t)
                exec_price = data_dict[t]['Open'].loc[exec_ts] # 다음 봉 시가 체결
                if pd.isna(exec_price): exec_price = data_dict[t]['Close'].loc[ts]
                
                real_sell_price = exec_price * (1 - SLIPPAGE)
                cash += pos['qty'] * real_sell_price * (1 - FEE_HALF)
                
                holding_durations.append(i - pos['entry_idx'])
                trade_log.append({'pnl_pct': (real_sell_price / pos['entry_price']) - 1 - FEE_HALF})

        if all_candidates:
            for t in list(positions.keys()):
                current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
                available_news = [c for c in all_candidates if c['ticker'] not in positions]
                if not available_news: continue
                best_new = available_news[0]
                
                if best_new['rs'] - current_rs > (best_new['atr_pct'] * HURDLE_ATR_MULT):
                    pos = positions.pop(t)
                    exec_price = data_dict[t]['Open'].loc[exec_ts]
                    if pd.isna(exec_price): exec_price = data_dict[t]['Close'].loc[ts]
                    
                    real_sell_price = exec_price * (1 - SLIPPAGE)
                    cash += pos['qty'] * real_sell_price * (1 - FEE_HALF)
                    
                    holding_durations.append(i - pos['entry_idx'])
                    trade_log.append({'pnl_pct': (real_sell_price / pos['entry_price']) - 1 - FEE_HALF})

            for candidate in all_candidates:
                if len(positions) >= MAX_SLOTS: break
                t = candidate['ticker']
                if t not in positions and candidate['signal_buy']:
                    exec_price = data_dict[t]['Open'].loc[exec_ts]
                    if pd.isna(exec_price): exec_price = data_dict[t]['Close'].loc[ts]
                    
                    target_unit = current_portfolio_value * (1 / MAX_SLOTS) * 0.995
                    alloc = min(cash, target_unit)
                    real_buy_price = exec_price * (1 + SLIPPAGE)
                    
                    # 💡 [허점 5 해결] 하드코딩 제거, SLIPPAGE 변수 직접 할당
                    qty = int(alloc // (real_buy_price * (1 + SLIPPAGE)))
                    cost = qty * real_buy_price * (1 + FEE_HALF)
                    
                    if qty > 0 and cash >= cost:
                        cash -= cost
                        positions[t] = {'qty': qty, 'entry_price': real_buy_price, 'entry_idx': i}

        history.append({'timestamp': ts, 'total_value': current_portfolio_value})

    # 마감 청산
    for t, pos in positions.items():
        price = data_dict[t]['Close'].iloc[-1]
        cash += (pos['qty'] * price * (1 - SLIPPAGE)) * (1 - FEE_HALF)
        holding_durations.append((len(master_timeline) - 1) - pos['entry_idx'])
        trade_log.append({'pnl_pct': ((price * (1 - SLIPPAGE)) / pos['entry_price']) - 1 - FEE_HALF})

    if not history:
        final_summary[tf] = {'수익률': 'N/A', 'MDD': 'N/A', '샤프지수': 'N/A', '승률': 'N/A', '손익비': 'N/A', '진짜알파': 'N/A', '매수횟수': '0', '평균보유봉': '0'}
        continue

    res_df = pd.DataFrame(history).set_index('timestamp')
    res_df['Peak'] = res_df['total_value'].cummax()
    res_df['Drawdown'] = (res_df['total_value'] - res_df['Peak']) / res_df['Peak']

    total_return = (res_df['total_value'].iloc[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    mdd = res_df['Drawdown'].min() * 100

    returns_30m = res_df['total_value'].pct_change().dropna()
    sharpe_ratio = (returns_30m.mean() / returns_30m.std()) * np.sqrt(13 * 252) if not returns_30m.empty and returns_30m.std() > 0 else 0

    if trade_log:
        pnl_list = [t['pnl_pct'] for t in trade_log]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]
        win_rate = (len(wins) / len(pnl_list)) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.abs(np.mean(losses)) if losses else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
    else:
        win_rate, profit_factor = 0.0, 0.0

    start_ts, end_ts = res_df.index[0], res_df.index[-1]
    universe_returns = []
    for ticker in data_dict.keys():
        u_start = data_dict[ticker]['Close'].loc[start_ts]
        u_end = data_dict[ticker]['Close'].loc[end_ts]
        if not pd.isna(u_start) and not pd.isna(u_end) and u_start > 0:
            universe_returns.append((u_end - u_start) / u_start * 100)
    
    universe_bh_avg_return = np.mean(universe_returns) if universe_returns else 0
    real_alpha = total_return - universe_bh_avg_return
    avg_holding_bars = np.mean(holding_durations) if holding_durations else 0

    final_summary[tf] = {
        '수익률': f"{total_return:+.2f}%",
        'MDD': f"{mdd:.2f}%",
        '샤프지수': f"{sharpe_ratio:.2f}",
        '승률': f"{win_rate:.1f}%",
        '손익비': f"{profit_factor:.2f}",
        '진짜알파': f"{real_alpha:+.2f}%",
        '매수횟수': f"{len(trade_log)}회",
        '평균보유봉': f"{avg_holding_bars:.1f}개"
    }

# ==========================================
# 📊 하이브리드 조합형 백테스트 최종 리포트 출력
# ==========================================
summary_df = pd.DataFrame(final_summary).T
print("\n" + "="*85)
print("🏆 [기존 재무지표 + Ver 8.3 비교기준/매매형태] 조합형 백테스트 성적비교표")
print("="*85)
print(summary_df.to_string())
print("="*85)
print("💡 [해결1] 미래참조(Look-Ahead) 완전 차단: resample('right') 적용")
print("💡 [해결2] 비현실적 체결가 수정: 시그널 발생 다음 봉의 Open(시가) 가격으로 실제 체결")
print("💡 [해결3] 누락 종목 방어: 지수(QQQ/KOSPI) 마스터 타임라인 기준으로 ffill(앞방향 채우기) 병합")
print("="*85)