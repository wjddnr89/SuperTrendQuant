import gc
import sys
import numpy as np
import pandas as pd
import yfinance as yf

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

def calculate_supertrend(df, period=7, multiplier=3.0):
    high, low, close = df['High'], df['Low'], df['Close']
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
    return trend, atr

def run_custom_universe_backtest():
    print("==========================================================")
    print("🛰️ [Ver 5.7] 질문자님 지정 독자 유니버스 3슬롯 엔진 가동")
    print("✅ 레버리지: SOXL, TQQQ / 숏: SOXS")
    print("✅ 주도섹터: NVDA, MU, GEV, VRT, RKLB, PL, RDW, OUST, TSLA, AEHR, AXON")
    print("✅ 조건: 3개 슬롯 제한(비중 각 33.3%) + 30분봉 초정밀 대응")
    print("==========================================================")
    
    # 💡 질문자님이 명시해주신 커스텀 롱/숏 분리 유니버스
    long_universe = [
        "SOXL", "TQQQ",          # 레버리지 롱
        "NVDA", "MU",            # 메모리
        "GEV", "VRT",            # 전력인프라
        "RKLB", "PL", "RDW",     # 우주 (로켓랩, 플래닛랩스, 레드와이어)
        "OUST", "TSLA",          # 로봇/자율주행 (아우스터, 테슬라)
        "AEHR", "AXON"           # 기타 (에흐르, 엑손)
    ]
    short_universe = ["SOXS"]    # 레버리지 숏
    full_universe = long_universe + short_universe
    
    # 1. 시장 필터용 QQQ 일봉 데이터 다운로드
    print(" -> 매크로 필터용 QQQ 일봉 데이터 구축 중...")
    qqq_daily = yf.download("QQQ", period="2y", interval="1d", progress=False)
    qqq_daily['EMA200'] = qqq_daily['Close'].ewm(span=200, adjust=False).mean()
    qqq_daily_map = qqq_daily['EMA200'].dropna()
    qqq_daily_map.index = qqq_daily_map.index.strftime('%Y-%m-%d')
    
    # 2. 메인 30분봉 데이터 다운로드 (최근 60일)
    print(" -> 커스텀 유니버스 30분봉 데이터 다운로드 중...")
    raw_data = yf.download(full_universe + ["QQQ"], period="60d", interval="30m", progress=False)
    
    data_dict = {}
    all_dates = None
    ticker_stats = {t: {'trades': 0, 'wins': 0, 'pnl_list': [], 'holding_bars': []} for t in full_universe}
    
    for ticker in full_universe:
        try:
            df = pd.DataFrame({
                'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
                'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker]
            }).dropna()
            if df.empty: continue
            
            # 레버리지(SOXL, SOXS)에 가혹한 필터 지표 승수 적용
            mult = 4.5 if ticker in ["SOXL", "SOXS"] else 3.0
            df['Trend'], df['ATR'] = calculate_supertrend(df, period=7, multiplier=mult)
            df['Return_5d'] = df['Close'].pct_change(65) # 30분봉 기준 약 5영업일
            
            data_dict[ticker] = df
            all_dates = df.index if all_dates is None else all_dates.intersection(df.index)
        except Exception:
            continue

    all_dates = sorted(list(all_dates))
    qqq_close_30m = raw_data['Close']['QQQ'].dropna()
    qqq_ret_5d = qqq_close_30m.pct_change(65)
    
    initial_cash = 10000.0
    cash = initial_cash
    positions = {} 
    
    FEE_HALF = 0.00225  
    SLIPPAGE = 0.0005  
    MAX_SLOTS = 3  # 합의된 3슬롯 자산 배분 규칙
    
    equity_history = []
    pending_orders = [] 

    for idx, date in enumerate(all_dates):
        date_str = date.strftime('%Y-%m-%d')
        current_qqq_ema200 = qqq_daily_map.loc[date_str] if date_str in qqq_daily_map.index else qqq_daily_map[qqq_daily_map.index < date_str].iloc[-1]

        # [체결 단계 - 매도]
        for order in [o for o in pending_orders if o['type'] == 'SELL']:
            t = order['ticker']
            if t in positions:
                o_open = data_dict[t].loc[date, 'Open']
                real_sell_price = o_open * (1 - SLIPPAGE)
                pnl = (real_sell_price - positions[t]['entry_price']) / positions[t]['entry_price']
                cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                
                ticker_stats[t]['trades'] += 1
                if pnl > 0: ticker_stats[t]['wins'] += 1
                ticker_stats[t]['pnl_list'].append(pnl)
                ticker_stats[t]['holding_bars'].append(idx - positions[t]['entry_idx'])
                del positions[t]
                
        # [체결 단계 - 매수]
        buy_orders = [o for o in pending_orders if o['type'] == 'BUY']
        for order in buy_orders:
            t = order['ticker']
            if t not in positions and len(positions) < MAX_SLOTS:
                o_open = data_dict[t].loc[date, 'Open']
                real_buy_price = o_open * (1 + SLIPPAGE)
                
                current_assets = cash + sum([p['qty'] * data_dict[pos_t].loc[date, 'Open'] for pos_t, p in positions.items()])
                # 정확히 자산의 33.3%씩 진입
                target_unit_size = current_assets * (1 / MAX_SLOTS) * 0.995
                actual_alloc = min(cash, target_unit_size)
                
                qty = int(actual_alloc // real_buy_price)
                cost = qty * real_buy_price * (1 + FEE_HALF)
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t] = {'qty': qty, 'entry_price': real_buy_price, 'entry_idx': idx}
                    
        pending_orders = [] 
        
        # [자산 가치 트래킹]
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        equity_history.append(cash + current_pos_val)
        
        # 만기일 자동 강제 청산
        if idx == len(all_dates) - 1:
            for t in list(positions.keys()):
                c_close = data_dict[t].loc[date, 'Close']
                real_sell_price = c_close * (1 - SLIPPAGE)
                pnl = (real_sell_price - positions[t]['entry_price']) / positions[t]['entry_price']
                cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                
                ticker_stats[t]['trades'] += 1
                if pnl > 0: ticker_stats[t]['wins'] += 1
                ticker_stats[t]['pnl_list'].append(pnl)
                ticker_stats[t]['holding_bars'].append(idx - positions[t]['entry_idx'])
                del positions[t]
            equity_history[-1] = cash
            break

        # [신호 발생 엔진]
        for t in list(positions.keys()):
            if data_dict[t].loc[date, 'Trend'] == -1:
                pending_orders.append({'ticker': t, 'type': 'SELL'})
                
        is_bull_market = qqq_close_30m.loc[date] > current_qqq_ema200
        target_pool = long_universe if is_bull_market else short_universe
        
        raw_buy_candidates = []
        for ticker in target_pool:
            if ticker in positions: continue
            df = data_dict[ticker]
            b_idx = df.index.get_loc(date)
            if b_idx < 1 or pd.isna(df['Return_5d'].iloc[b_idx]) or pd.isna(qqq_ret_5d.loc[date]): continue
                
            if df['Trend'].iloc[b_idx] == 1 and df['Trend'].iloc[b_idx-1] == -1:
                rs_score = df['Return_5d'].iloc[b_idx] - qqq_ret_5d.loc[date]
                raw_buy_candidates.append({'ticker': ticker, 'rs': rs_score})
                
        if raw_buy_candidates:
            raw_buy_candidates = sorted(raw_buy_candidates, key=lambda x: x['rs'], reverse=True)
            for candidate in raw_buy_candidates:
                pending_orders.append({'ticker': candidate['ticker'], 'type': 'BUY'})

    # ────────────── 📊 결과 분석 및 통계 ──────────────
    equity_series = pd.Series(equity_history, index=all_dates)
    final_return = ((equity_series.iloc[-1] - initial_cash) / initial_cash) * 100
    roll_max = equity_series.cummax()
    mdd = ((equity_series - roll_max) / roll_max).min() * 100
    daily_returns = equity_series.resample('D').last().dropna().pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0
        
    all_pnl_flat = []
    for t in full_universe:
        all_pnl_flat.extend(ticker_stats[t]['pnl_list'])

    print("\n==========================================================")
    print("      🎯 [최종 성적표] 지정 커스텀 유니버스 백테스트 결과")
    print("==========================================================")
    print(f" 🟩 [누적 수익률]      : {final_return:+.2f}%")
    print(f" 🟥 [최대 낙폭 (MDD)]   : {mdd:.2f}%")
    print(f" 📊 [샤프 지수 (Sharpe)]: {sharpe_ratio:.2f}")
    print(f" ⚖️ [알고리즘 승률]    : {(len([r for r in all_pnl_flat if r > 0])/len(all_pnl_flat)*100 if all_pnl_flat else 0):.2f}% (총 {len(all_pnl_flat)}회 거래)")
    print("==========================================================\n")

    print("==========================================================")
    print("      🔍 [진단 결과] 섹터별 지정 종목 세부 기여도")
    print("==========================================================")
    print(f" {'Ticker':<8} | {'Trades':<6} | {'Win Rate':<8} | {'Avg Return':<10} | {'Total PnL':<10}")
    print("-" * 58)
    for t in full_universe:
        stats = ticker_stats[t]
        if stats['trades'] == 0: continue
        w_rate = (stats['wins'] / stats['trades']) * 100
        avg_ret = np.mean(stats['pnl_list']) * 100
        total_pnl = sum(stats['pnl_list']) * 100
        print(f" {t:<8} | {stats['trades']:<6} | {w_rate:6.1f}% | {avg_ret:+9.2f}% | {total_pnl:+9.2f}%")
    print("==========================================================\n")

if __name__ == "__main__":
    run_custom_universe_backtest()
    gc.collect()