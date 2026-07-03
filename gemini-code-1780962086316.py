import os
import sys
import gc
import numpy as np
import pandas as pd
import yfinance as yf

# Windows 환경에서 한글 깨짐 방지
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

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
    return trend, atr

def run_analytics_backtest():
    print("==========================================================")
    print("📊 [Ver 5.5] 2슬롯 확장 및 당일 종가 즉시 체결 알파 엔진 가동")
    print("==========================================================")
    
    # [생존 편향 제거] 원래 14개 유니버스 100% 그대로 유지
    long_universe = ["SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON"]
    short_universe = ["SOXS"]
    full_universe = long_universe + short_universe
    
    START_CASH = 10000.0
    MAX_SLOTS = 2      
    HURDLE_RATE = 0.015
    FEE_HALF = 0.00225  
    SLIPPAGE = 0.0005  
    
    print("⏳ 야후 파이낸스로부터 30분봉 데이터 수집 중...")
    raw_data = yf.download(full_universe + ["QQQ"], period="59d", interval="30m", progress=False)
    
    data_dict = {}
    all_dates = None
    ticker_stats = {t: {'trades': 0, 'wins': 0, 'pnl_list': [], 'holding_bars': []} for t in full_universe}
    
    for ticker in full_universe + ["QQQ"]:
        try:
            df = pd.DataFrame({
                'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
                'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker]
            }).dropna()
            if df.empty: continue
            
            if ticker != "QQQ":
                mult = 4.5 if ticker in ["SOXL", "SOXS"] else 3.0
                df['Trend'], df['ATR'] = calculate_supertrend(df, period=7, multiplier=mult)
            
            df['Return_5d'] = df['Close'].pct_change(65)
            data_dict[ticker] = df
            
            if ticker != "QQQ":
                all_dates = df.index if all_dates is None else all_dates.intersection(df.index)
        except Exception:
            continue

    all_dates = sorted(list(all_dates))
    qqq_ret_5d = data_dict['QQQ']['Return_5d']
    
    cash = START_CASH
    positions = {} 
    portfolio_log = []

    # 🌟 [2번 적용] 타임 오염을 막기 위해 당일 30분봉 마감 시점(Close)에 즉시 체결 구조로 개편
    for idx, date in enumerate(all_dates):
        if date not in data_dict['QQQ'].index: continue
        
        # 1. 매도 시그널 스캔 및 [당일 종가 즉시 매도 체결]
        for t in list(positions.keys()):
            if date not in data_dict[t].index: continue
            curr_trend = data_dict[t].loc[date, 'Trend']
            
            if curr_trend == -1:
                c_close = data_dict[t].loc[date, 'Close']
                real_sell_price = c_close * (1 - SLIPPAGE) # 갭 리스크 없는 당일 종가 체결
                pnl = (real_sell_price - positions[t]['entry_price']) / positions[t]['entry_price']
                cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                
                ticker_stats[t]['trades'] += 1
                if pnl > 0: ticker_stats[t]['wins'] += 1
                ticker_stats[t]['pnl_list'].append(pnl)
                ticker_stats[t]['holding_bars'].append(idx - positions[t]['entry_idx'])
                
                del positions[t]

        # 2. 강제 순환매 교체 매도 연산 및 [즉시 청산]
        raw_buy_candidates = []
        for ticker in full_universe:
            if date not in data_dict[ticker].index or pd.isna(data_dict[ticker]['Return_5d'].loc[date]) or pd.isna(qqq_ret_5d.loc[date]): continue
            if data_dict[ticker]['Trend'].loc[date] == 1:
                loc_idx = data_dict[ticker].index.get_loc(date)
                if loc_idx >= 1 and data_dict[ticker]['Trend'].iloc[loc_idx-1] == -1:
                    rs_score = data_dict[ticker]['Return_5d'].loc[date] - qqq_ret_5d.loc[date]
                    raw_buy_candidates.append({'ticker': ticker, 'rs': rs_score})
                    
        if raw_buy_candidates:
            raw_buy_candidates = sorted(raw_buy_candidates, key=lambda x: x['rs'], reverse=True)
            
            for t in list(positions.keys()):
                if date not in data_dict[t].index: continue
                current_rs = data_dict[t].loc[date, 'Return_5d'] - qqq_ret_5d.loc[date]
                if raw_buy_candidates[0]['rs'] - current_rs > HURDLE_RATE:
                    c_close = data_dict[t].loc[date, 'Close']
                    real_sell_price = c_close * (1 - SLIPPAGE)
                    pnl = (real_sell_price - positions[t]['entry_price']) / positions[t]['entry_price']
                    cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                    
                    ticker_stats[t]['trades'] += 1
                    if pnl > 0: ticker_stats[t]['wins'] += 1
                    ticker_stats[t]['pnl_list'].append(pnl)
                    ticker_stats[t]['holding_bars'].append(idx - positions[t]['entry_idx'])
                    
                    del positions[t]
            
            # 3. 신규 매수 시그널 발생 종목 [당일 종가 즉시 매수 체결]
            for candidate in raw_buy_candidates:
                t = candidate['ticker']
                if t in positions: continue
                if len(positions) >= MAX_SLOTS: break # 4슬롯 제한 체크
                
                c_close = data_dict[t].loc[date, 'Close']
                real_buy_price = c_close * (1 + SLIPPAGE)
                
                # 4슬롯 균등 자산 배분 (25% 씩 진입)
                current_assets = cash + sum([p['qty'] * data_dict[pos_t].loc[date, 'Close'] for pos_t, p in positions.items() if date in data_dict[pos_t].index])
                target_unit_size = current_assets * (1 / MAX_SLOTS) * 0.995
                actual_alloc = min(cash, target_unit_size)
                
                qty = int(actual_alloc // (real_buy_price * 1.0005))
                cost = qty * real_buy_price * (1 + FEE_HALF)
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t] = {'qty': qty, 'entry_price': real_buy_price, 'entry_idx': idx}

        # 마지막 영업일 마감 시점 강제 청산
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
            portfolio_log.append({"timestamp": date, "total_value": cash})
            break

        # 당일 자산 가치 기록
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items() if date in data_dict[t].index])
        portfolio_log.append({"timestamp": date, "total_value": cash + current_pos_val})

    # ────────────── 📊 심층 리포트 연산 ──────────────
    final_return = ((cash - START_CASH) / START_CASH) * 100
    
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
    avg_hold_hours = (avg_hold_bars * 30) / 60
    
    print(f" ⏱️ 평균 포지션 보유 기간   : {avg_hold_bars:.1f}개 봉 (약 {avg_hold_hours:.1f} 영업시간)")
    print(f" 🚀 시스템 최대 익절 거래   : {max(all_pnl_flat)*100:+.2f}%" if all_pnl_flat else " N/A")
    print(f" 📉 시스템 최대 손절 거래   : {min(all_pnl_flat)*100:+.2f}%" if all_pnl_flat else " N/A")

    # 4. 벤치마크 계산 바인딩
    benchmark_returns = {}
    for t in full_universe:
        if t in data_dict and not data_dict[t].empty:
            start_price = data_dict[t]['Close'].iloc[0]
            end_price = data_dict[t]['Close'].iloc[-1]
            benchmark_returns[t] = (end_price - start_price) / start_price * 100

    avg_benchmark_return = np.mean(list(benchmark_returns.values())) if benchmark_returns else 0.0

    print("\n" + "="*58)
    print("      🔍 [진단 결과 4] 벤치마크 비교 (단순 보유 vs 전략)")
    print("="*58)
    print(f" 📈 14개 종목 단순 보유 평균 수익률 : {avg_benchmark_return:+.2f}%")
    print(f" 🤖 우리 전략의 최종 누적 수익률   : {final_return:+.2f}%")
    print(f" 🏆 알파(Alpha, 초과 수익)        : {final_return - avg_benchmark_return:+.2f}%")
    print("="*58 + "\n")

if __name__ == "__main__":
    run_analytics_backtest()
    gc.collect()