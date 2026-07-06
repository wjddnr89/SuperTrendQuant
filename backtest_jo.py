import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ==========================================
# ⚙️ 핵심 백테스트 파라미터 및 마켓 설정
# ==========================================
BACKTEST_MARKET = "KR"      # 🇰🇷 "KR" / 🇺🇸 "US"
INITIAL_CASH = 10_000_000.0 if BACKTEST_MARKET == "KR" else 10_000.0 
MAX_SLOTS = 1               

HURDLE_ATR_MULT = 1.25      
ALLOW_LATE_CHASE = True     

FEE_HALF = 0.00225          
SLIPPAGE = 0.0005           
FEE_ROUNDTRIP = (FEE_HALF + SLIPPAGE) * 2

UNIVERSE_FILE = "universe.json"

# 🔥 [변경 포인트 1] 시장 필터는 가장 우수한 '1h'로 고정하고, RS_PERIOD 후보군을 설정합니다.
FIXED_TIMEFRAME = '1h'
EXPERIMENTAL_RS_PERIODS = [20, 60, 100, 130, 180, 240] 
max_rs_period = max(EXPERIMENTAL_RS_PERIODS)

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
    print("🇰🇷 [국내장] RS_PERIOD 실험 모드 가동")
    tickers_raw = list(kr_universe_map.keys())
    yf_tickers = [convert_to_yf_ticker(s, kr_universe_map[s]) for s in tickers_raw]
    ticker_bench_map = {s: ("^KS11" if kr_universe_map[s] == "KOSPI" else "^KQ11") for s in tickers_raw}
    benchmarks = ["^KS11", "^KQ11"] 
    main_bench = "^KS11"
else:
    print("🇺🇸 [해외장] RS_PERIOD 실험 모드 가동")
    tickers_raw = us_universe_list
    yf_tickers = us_universe_list
    ticker_bench_map = {s: "QQQ" for s in tickers_raw}
    benchmarks = ["QQQ"]
    main_bench = "QQQ"

print(f"📥 야후 파이낸스에서 데이터 다운로드 중 (60일)...")
raw_30m = yf.download(yf_tickers + benchmarks, period="60d", interval="30m", progress=False)
raw_1h = yf.download(benchmarks, period="60d", interval="1h", progress=False)

# 마스터 타임라인 강제 고정
master_timeline = raw_30m['Close'][main_bench].dropna().index

# 상위 타임프레임(1h) 필터 미리 연산
bench_upper_trends = {}
for b_sym in benchmarks:
    df_b_raw = pd.DataFrame({
        'Open': raw_1h['Open'][b_sym], 'High': raw_1h['High'][b_sym],
        'Low': raw_1h['Low'][b_sym], 'Close': raw_1h['Close'][b_sym]
    }).dropna()
    df_b = calculate_supertrend(df_b_raw, period=10, multiplier=3.0)
    bench_upper_trends[b_sym] = df_b['Trend']

# 기본 가격 및 수퍼트렌드 데이터 매핑 (RS는 루프 내부에서 동적 계산)
base_data_dict = {}
for s, yf_tk in zip(tickers_raw, yf_tickers):
    if yf_tk not in raw_30m['Close']: continue
    
    df = pd.DataFrame({
        'Open': raw_30m['Open'][yf_tk], 'High': raw_30m['High'][yf_tk],
        'Low': raw_30m['Low'][yf_tk], 'Close': raw_30m['Close'][yf_tk]
    }).reindex(master_timeline)
    df['HasRealBar'] = df['Close'].notna()
    df = df.ffill()
    
    if df['Close'].isna().all(): continue
    
    mult = 4.5 if s in ["SOXL", "SOXS"] else 3.0
    df = calculate_supertrend(df, period=7, multiplier=mult)
    base_data_dict[s] = df

final_summary = {}

# 🔄 [변경 포인트 2] RS 주도주 주기별 시뮬레이션 루프
for rs_p in EXPERIMENTAL_RS_PERIODS:
    print(f"⚙️ 상대강도 주기 RS_PERIOD [{rs_p}봉] 적용 시뮬레이션 연산 중...")
    
    # 해당 rs_p에 맞추어 종목별 소속 벤치마크 수익률 계산
    bench_ret_30m = {b: raw_30m['Close'][b].dropna().pct_change(rs_p) for b in benchmarks}
    
    # 종목별 딕셔너리에 동적으로 RS 스코어 주입
    data_dict = {}
    for s, df in base_data_dict.items():
        df_copy = df.copy()
        bench_symbol = ticker_bench_map[s]
        df_copy['RS'] = df_copy['Close'].pct_change(rs_p) - bench_ret_30m[bench_symbol]
        data_dict[s] = df_copy

    cash = INITIAL_CASH
    positions = {}
    history = []
    trade_log = []
    holding_durations = []

    # 실험의 공정성을 위해 후보군 중 가장 큰 max_rs_period 이후부터 시뮬레이션 시작
    for i in range(max_rs_period + 1, len(master_timeline) - 1):
        ts = master_timeline[i]           
        exec_ts = master_timeline[i + 1]  
        
        current_portfolio_value = cash
        for t, pos in positions.items():
            current_portfolio_value += pos['qty'] * data_dict[t]['Close'].loc[ts]
            
        all_candidates = []
        for t, df in data_dict.items():
            if not df['HasRealBar'].loc[ts]:
                continue

            bench_symbol = ticker_bench_map[t]
            market_signal = -1
            past_signals = bench_upper_trends[bench_symbol][bench_upper_trends[bench_symbol].index <= ts]
            if not past_signals.empty: market_signal = past_signals.iloc[-1]

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
            if not data_dict[t]['HasRealBar'].loc[ts]:
                continue

            curr_trend = data_dict[t]['Trend'].loc[ts]
            if curr_trend == -1:
                pos = positions.pop(t)
                exec_price = data_dict[t]['Open'].loc[exec_ts] if data_dict[t]['HasRealBar'].loc[exec_ts] else data_dict[t]['Close'].loc[ts]
                if pd.isna(exec_price): exec_price = data_dict[t]['Close'].loc[ts]
                
                real_sell_price = exec_price * (1 - SLIPPAGE)
                cash += pos['qty'] * real_sell_price * (1 - FEE_HALF)
                
                holding_durations.append(i - pos['entry_idx'])
                trade_log.append({'pnl_pct': (real_sell_price / pos['entry_price']) - 1 - FEE_HALF})

        if all_candidates:
            for t in list(positions.keys()):
                if not data_dict[t]['HasRealBar'].loc[ts]:
                    continue

                current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
                available_news = [c for c in all_candidates if c['ticker'] not in positions]
                if not available_news: continue
                best_new = available_news[0]
                
                if best_new['rs'] - current_rs > (best_new['atr_pct'] * HURDLE_ATR_MULT):
                    pos = positions.pop(t)
                    exec_price = data_dict[t]['Open'].loc[exec_ts] if data_dict[t]['HasRealBar'].loc[exec_ts] else data_dict[t]['Close'].loc[ts]
                    if pd.isna(exec_price): exec_price = data_dict[t]['Close'].loc[ts]
                    
                    real_sell_price = exec_price * (1 - SLIPPAGE)
                    cash += pos['qty'] * real_sell_price * (1 - FEE_HALF)
                    
                    holding_durations.append(i - pos['entry_idx'])
                    trade_log.append({'pnl_pct': (real_sell_price / pos['entry_price']) - 1 - FEE_HALF})

            for candidate in all_candidates:
                if len(positions) >= MAX_SLOTS: break
                t = candidate['ticker']
                if t not in positions and candidate['signal_buy']:
                    exec_price = data_dict[t]['Open'].loc[exec_ts] if data_dict[t]['HasRealBar'].loc[exec_ts] else data_dict[t]['Close'].loc[ts]
                    if pd.isna(exec_price): exec_price = data_dict[t]['Close'].loc[ts]
                    
                    target_unit = cash * 0.90
                    alloc = min(cash, target_unit)
                    real_buy_price = exec_price * (1 + SLIPPAGE)
                    
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
        final_summary[f"RS_{rs_p}"] = {'수익률': 'N/A', 'MDD': 'N/A', '샤프지수': 'N/A', '승률': 'N/A', '손익비': 'N/A', '진짜알파': 'N/A', '매수횟수': '0', '평균보유봉': '0'}
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

    final_summary[f"RS_Period {rs_p}"] = {
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
# 📊 RS_PERIOD 비교 백테스트 최종 리포트 출력
# ==========================================
summary_df = pd.DataFrame(final_summary).T
print("\n" + "="*85)
print(f"🏆 [시장필터 1h 고정] 상대강도 주기(RS_PERIOD) 변수별 성적비교표 ({BACKTEST_MARKET})")
print("="*85)
print(summary_df.to_string())
print("="*85)
print("💡 모든 실험군은 동일한 시작 지점(max_rs_period 이후)부터 연산되어 공정하게 비교됩니다.")
print("="*85)
