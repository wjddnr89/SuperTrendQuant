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
    if df.empty or len(df) < period: 
        return pd.Series(1, index=df.index), pd.Series(0, index=df.index), pd.Series(0, index=df.index)
    high, low, close = df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze()
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    atr_pct = atr / close  # 동적 허들에 사용할 ATR 비율 연산
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
    return trend, atr, atr_pct

def run_analytics_backtest_v96():
    print("==========================================================")
    print("🏆 [Ver 9.6] 1슬롯 집중 투자 + 동적 허들(1.25*ATR) 알파 엔진 가동")
    print("==========================================================")
    
    long_universe = ["SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON"]
    short_universe = ["SOXS"]
    full_universe = long_universe + short_universe
    
    START_CASH = 10000.0
    MAX_SLOTS = 1          # ⭐ 1슬롯 올인 구조 롤백
    RS_PERIOD = 130        # 130개봉 주도주 추세 주기 고정
    HURDLE_ATR_MULT = 1.25 # ⭐ 황금 최적화 허들 배수
    FEE_HALF = 0.00225  
    SLIPPAGE = 0.0005  
    
    print("⏳ 야후 파이낸스로부터 30분봉 및 1시간봉 데이터 동시 수집 중...")
    raw_30m = yf.download(full_universe + ["QQQ"], period="59d", interval="30m", progress=False)
    raw_1h = yf.download(["QQQ"], period="59d", interval="1h", progress=False)
    
    # 1. QQQ 1시간봉 SuperTrend 필터 맵 빌드
    df_qqq_1h = pd.DataFrame({
        'Open': raw_1h['Open']['QQQ'].squeeze(), 'High': raw_1h['High']['QQQ'].squeeze(),
        'Low': raw_1h['Low']['QQQ'].squeeze(), 'Close': raw_1h['Close']['QQQ'].squeeze()
    }).dropna()
    qqq_1h_trend, _, _ = calculate_supertrend(df_qqq_1h, period=10, multiplier=3.0)
    
    # 2. 메인 30분봉 데이터 가공 및 동기화 설정
    data_dict = {}
    all_dates = None
    ticker_stats = {t: {'trades': 0, 'wins': 0, 'pnl_list': [], 'holding_bars': []} for t in full_universe}
    
    qqq_close_30m = raw_30m['Close']['QQQ'].dropna()
    qqq_ret_30m = qqq_close_30m.pct_change(RS_PERIOD)
    
    for ticker in full_universe:
        try:
            df = pd.DataFrame({
                'Open': raw_30m['Open'][ticker], 'High': raw_30m['High'][ticker],
                'Low': raw_30m['Low'][ticker], 'Close': raw_30m['Close'][ticker]
            }).dropna()
            if df.empty: continue
            
            mult = 4.5 if ticker in ["SOXL", "SOXS"] else 3.0
            df['Trend'], df['ATR'], df['ATR_pct'] = calculate_supertrend(df, period=7, multiplier=mult)
            df['RS'] = df['Close'].pct_change(RS_PERIOD) - qqq_ret_30m
            
            data_dict[ticker] = df
            all_dates = df.index if all_dates is None else all_dates.intersection(df.index)
        except Exception:
            continue

    all_dates = sorted(list(all_dates))
    
    cash = START_CASH
    positions = {} 
    portfolio_log = []

    # 🔄 타임라인 스캔 시작 (30분 주기 동기화 연산)
    for idx, date in enumerate(all_dates):
        if idx < RS_PERIOD + 1: continue
        
        # 30분 현재 시점에서 가장 최신의 QQQ 1시간봉 '날씨' 연동
        past_qqq_1h_signals = qqq_1h_trend[qqq_1h_trend.index <= date]
        upper_market_signal = past_qqq_1h_signals.iloc[-1] if not past_qqq_1h_signals.empty else 1
        
        # 1. 매도 시그널 스캔 (보유 종목 30분봉 추세가 꺾이면 즉시 청산)
        for t in list(positions.keys()):
            if date not in data_dict[t].index: continue
            curr_trend = data_dict[t].loc[date, 'Trend']
            
            if curr_trend == -1:
                c_close = data_dict[t].loc[date, 'Close']
                real_sell_price = c_close * (1 - SLIPPAGE)
                pnl = (real_sell_price - positions[t]['entry_price']) / positions[t]['entry_price']
                cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                
                ticker_stats[t]['trades'] += 1
                if pnl > 0: ticker_stats[t]['wins'] += 1
                ticker_stats[t]['pnl_list'].append(pnl)
                ticker_stats[t]['holding_bars'].append(idx - positions[t]['entry_idx'])
                
                del positions[t]

        # 2. 강제 동적 순환매 교체 매도 연산 및 신규 진입 후보군 추출
        raw_buy_candidates = []
        for ticker in full_universe:
            if date not in data_dict[ticker].index or pd.isna(data_dict[ticker]['RS'].loc[date]): continue
            
            if data_dict[ticker]['Trend'].loc[date] == 1:
                rs_score = data_dict[ticker]['RS'].loc[date]
                atr_pct_val = data_dict[ticker]['ATR_pct'].loc[date]
                raw_buy_candidates.append({'ticker': ticker, 'rs': rs_score, 'atr_pct': atr_pct_val})
                    
        if raw_buy_candidates:
            raw_buy_candidates = sorted(raw_buy_candidates, key=lambda x: x['rs'], reverse=True)
            best_candidate = raw_buy_candidates[0]
            
            # ⭐ [핵심 개조] 고정 허들 제거 -> 새로운 1등 종목의 ATR 기반 동적 허들 적용
            dynamic_hurdle = best_candidate['atr_pct'] * HURDLE_ATR_MULT
            
            for t in list(positions.keys()):
                if date not in data_dict[t].index: continue
                current_rs = data_dict[t].loc[date, 'RS']
                
                # 새로운 1등과의 RS 격차가 동적 허들 돌파 시에만 포지션 스왑
                if best_candidate['rs'] - current_rs > dynamic_hurdle:
                    c_close = data_dict[t].loc[date, 'Close']
                    real_sell_price = c_close * (1 - SLIPPAGE)
                    pnl = (real_sell_price - positions[t]['entry_price']) / positions[t]['entry_price']
                    cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                    
                    ticker_stats[t]['trades'] += 1
                    if pnl > 0: ticker_stats[t]['wins'] += 1
                    ticker_stats[t]['pnl_list'].append(pnl)
                    ticker_stats[t]['holding_bars'].append(idx - positions[t]['entry_idx'])
                    
                    del positions[t]
            
            # 3. 신규 매수 집행 (QQQ 1시간 필터가 정배열일 때만 문을 열어줌)
            if upper_market_signal == 1:
                for candidate in raw_buy_candidates:
                    t = candidate['ticker']
                    if t in positions: continue
                    if len(positions) >= MAX_SLOTS: break # 1슬롯 제한
                    
                    c_close = data_dict[t].loc[date, 'Close']
                    real_buy_price = c_close * (1 + SLIPPAGE)
                    
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
    res_df = pd.DataFrame(portfolio_log).set_index('timestamp')
    res_df['Peak'] = res_df['total_value'].cummax()
    res_df['Drawdown'] = (res_df['total_value'] - res_df['Peak']) / res_df['Peak']
    mdd = res_df['Drawdown'].min() * 100
    final_return = ((res_df['total_value'].iloc[-1] - START_CASH) / START_CASH) * 100
    
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
    print(f" 📉 계좌 최고 MDD (최대낙폭) : {mdd:.2f}%")

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
    run_analytics_backtest_v96()
    gc.collect()