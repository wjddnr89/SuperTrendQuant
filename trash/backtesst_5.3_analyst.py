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

def run_analytics_backtest():
    print("==========================================================")
    print("📊 [Ver 5.4] 퀀트 심층 진단 및 성과 분리 백테스트 엔진 가동")
    print("==========================================================")
    
    long_universe = ["SOXL", "TQQQ", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AVGO", "LLY", "AMD", "QLD", "USO"]
    short_universe = ["SOXS", "SQQQ"]
    full_universe = long_universe + short_universe
    
    # 1. 매크로 필터용 QQQ 일봉 연동
    qqq_daily = yf.download("QQQ", period="2y", interval="1d", progress=False)
    qqq_daily['EMA200'] = qqq_daily['Close'].ewm(span=200, adjust=False).mean()
    qqq_daily_map = qqq_daily['EMA200'].dropna()
    qqq_daily_map.index = qqq_daily_map.index.strftime('%Y-%m-%d')
    
    # 2. 메인 30분봉 데이터 로드
    raw_data = yf.download(full_universe + ["QQQ"], period="60d", interval="30m", progress=False)
    
    data_dict = {}
    all_dates = None
    
    # 종목별 통계 저장소 초기화
    ticker_stats = {t: {'trades': 0, 'wins': 0, 'pnl_list': [], 'holding_bars': []} for t in full_universe}
    
    for ticker in full_universe:
        try:
            df = pd.DataFrame({
                'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
                'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker]
            }).dropna()
            if df.empty: continue
            
            mult = 4.5 if ticker in ["SOXL", "SOXS", "TQQQ", "SQQQ"] else 3.0
            df['Trend'], df['ATR'] = calculate_supertrend(df, period=7, multiplier=mult)
            df['Return_5d'] = df['Close'].pct_change(65)
            
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
                
                # 💡 종목 통계 기록
                ticker_stats[t]['trades'] += 1
                if pnl > 0: ticker_stats[t]['wins'] += 1
                ticker_stats[t]['pnl_list'].append(pnl)
                ticker_stats[t]['holding_bars'].append(idx - positions[t]['entry_idx'])
                
                del positions[t]
                
        # [체결 단계 - 매수]
        buy_orders = [o for o in pending_orders if o['type'] == 'BUY']
        for order in buy_orders:
            t = order['ticker']
            if t not in positions and len(positions) < 4:
                o_open = data_dict[t].loc[date, 'Open']
                real_buy_price = o_open * (1 + SLIPPAGE)
                
                current_assets = cash + sum([p['qty'] * data_dict[pos_t].loc[date, 'Open'] for pos_t, p in positions.items()])
                target_unit_size = current_assets * 0.25 * 0.995
                actual_alloc = min(cash, target_unit_size)
                
                qty = int(actual_alloc // real_buy_price)
                cost = qty * real_buy_price * (1 + FEE_HALF)
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t] = {'qty': qty, 'entry_price': real_buy_price, 'entry_idx': idx}
                    
        pending_orders = [] 
        
        # [자산 가치 기록]
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        equity_history.append(cash + current_pos_val)
        
        # 마지막 날 강제 청산
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

        # [신호 스캔]
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

    # ────────────── 📊 심층 리포트 연산 ──────────────
    print("\n" + "="*58)
    print("      🔍 [진단 결과 1] 롱(Long) vs 숏(Short) 성과 분리")
    print("="*58)
    
    long_trades_pnl = []
    short_trades_pnl = []
    all_pnl_flat = []
    all_holdings = []
    
    for t in full_universe:
        pnl_list = ticker_stats[t]['pnl_list']
        all_pnl_flat.extend(pnl_list)
        all_holdings.extend(ticker_stats[t]['holding_bars'])
        if t in long_universe:
            long_trades_pnl.extend(pnl_list)
        else:
            short_trades_pnl.extend(pnl_list)
            
    print(f" 🟢 롱  전략 총 수익률 합계 : {sum(long_trades_pnl)*100:+.2f}% (총 {len(long_trades_pnl)}회 거래)")
    print(f" 🔴 숏  전략 총 수익률 합계 : {sum(short_trades_pnl)*100:+.2f}% (총 {len(short_trades_pnl)}회 거래)")
    
    print("\n" + "="*58)
    print("      🔍 [진단 결과 2] 종목별 세부 기여도 매트릭스")
    print("="*58)
    print(f" {'Ticker':<8} | {'Trades':<6} | {'Win Rate':<8} | {'Avg Return':<10} | {'Total PnL':<10}")
    print("-" * 58)
    
    for t in full_universe:
        stats = ticker_stats[t]
        if stats['trades'] == 0: continue
        w_rate = (stats['wins'] / stats['trades']) * 100
        avg_ret = np.mean(stats['pnl_list']) * 100
        total_pnl = sum(stats['pnl_list']) * 100
        print(f" {t:<8} | {stats['trades']:<6} | {w_rate:6.1f}% | {avg_ret:+9.2f}% | {total_pnl:+9.2f}%")
        
    print("\n" + "="*58)
    print("      🔍 [진단 결과 3] 마이크로 구조 및 변동성 분포")
    print("="*58)
    avg_hold_bars = np.mean(all_holdings) if all_holdings else 0
    # 30분봉 개수를 시간으로 환산 (미 증시 하루 정규장 = 13개 봉 = 6.5시간)
    avg_hold_hours = (avg_hold_bars * 30) / 60
    
    print(f" ⏱️ 평균 포지션 보유 기간   : {avg_hold_bars:.1f}개 봉 (약 {avg_hold_hours:.1f} 영업시간)")
    print(f" 🚀 시스템 최대 익절 거래   : {max(all_pnl_flat)*100:+.2f}%" if all_pnl_flat else " N/A")
    print(f" 📉 시스템 최대 손절 거래   : {min(all_pnl_flat)*100:+.2f}%" if all_pnl_flat else " N/A")
    print("="*58 + "\n")

if __name__ == "__main__":
    run_analytics_backtest()
    gc.collect()
    

# 기존 엔진에 추가할 벤치마크 계산 코드
benchmark_returns = {}
for t in full_universe:
    # 60일간의 단순 수익률: (마지막 종가 - 시작가) / 시작가
    start_price = data_dict[t]['Close'].iloc[0]
    end_price = data_dict[t]['Close'].iloc[-1]
    benchmark_returns[t] = (end_price - start_price) / start_price * 100

avg_benchmark_return = np.mean(list(benchmark_returns.values()))

print("\n" + "="*58)
print("      🔍 [진단 결과 4] 벤치마크 비교 (단순 보유 vs 전략)")
print("="*58)
print(f" 📈 15개 종목 단순 보유 평균 수익률 : {avg_benchmark_return:+.2f}%")
print(f" 🤖 우리 전략의 최종 누적 수익률   : {final_return:+.2f}%")
print(f" 🏆 알파(Alpha, 초과 수익)        : {final_return - avg_benchmark_return:+.2f}%")
print("="*58 + "\n")