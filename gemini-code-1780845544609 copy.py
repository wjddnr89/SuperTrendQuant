import gc
import sys
import numpy as np
import pandas as pd
import yfinance as yf

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

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
    return trend

def run_v591_backtest():
    print("==========================================================")
    print("🚀 [Ver 5.9.1] 변동성 완화형 초정밀 수익 극대화 엔진 가동")
    print("✅ 강화 포인트 1: 트레일링 스탑 조건 완화 (+30% 도달 시 발동 / 고점 대비 -10% 청산)")
    print("✅ 강화 포인트 2: 순환매 허들 레이트 (+1.5% 점수 격차 시에만 교체 고수)")
    print("==========================================================")
    
    full_universe = ["SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON", "SOXS"]
    
    qqq_daily = yf.download("QQQ", period="2y", interval="1d", progress=False)
    qqq_daily['EMA200'] = qqq_daily['Close'].ewm(span=200, adjust=False).mean()
    qqq_ema_map = qqq_daily['EMA200'].dropna()
    qqq_ema_map.index = qqq_ema_map.index.strftime('%Y-%m-%d')
    
    raw_data = yf.download(full_universe + ["QQQ"], period="60d", interval="30m", progress=False)
    
    data_dict = {}
    all_dates = None
    ticker_stats = {t: {'trades': 0, 'wins': 0, 'pnl_list': []} for t in full_universe}
    
    for ticker in full_universe:
        try:
            df = pd.DataFrame({
                'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
                'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker]
            }).dropna()
            if df.empty: continue
            
            mult = 4.5 if ticker in ["SOXL", "SOXS"] else 3.0
            df['Trend'] = calculate_supertrend(df, period=7, multiplier=mult)
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
    MAX_SLOTS = 2
    FEE_HALF = 0.00225
    SLIPPAGE = 0.0005
    HURDLE_RATE = 0.015  # 1.5% 점수 격차 허들 고수
    
    equity_history = []
    pending_orders = []

    for idx, date in enumerate(all_dates):
        date_str = date.strftime('%Y-%m-%d')
        current_qqq_ema200 = qqq_ema_map.loc[date_str] if date_str in qqq_ema_map.index else qqq_ema_map.iloc[-1]

        # [1. 매도 주문 체결]
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
                del positions[t]
                
        # [2. 매수 주문 체결]
        for order in [o for o in pending_orders if o['type'] == 'BUY']:
            t = order['ticker']
            if t not in positions and len(positions) < MAX_SLOTS:
                o_open = data_dict[t].loc[date, 'Open']
                real_buy_price = o_open * (1 + SLIPPAGE)
                current_assets = cash + sum([p['qty'] * data_dict[pos_t].loc[date, 'Open'] for pos_t, p in positions.items()])
                target_unit_size = current_assets * (1 / MAX_SLOTS) * 0.995
                actual_alloc = min(cash, target_unit_size)
                qty = int(actual_alloc // real_buy_price)
                cost = qty * real_buy_price * (1 + FEE_HALF)
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t] = {
                        'qty': qty, 
                        'entry_price': real_buy_price,
                        'highest_price': real_buy_price
                    }
                    
        pending_orders = []
        
        # [3. 완화된 실시간 트레일링 익절 감시]
        for t in list(positions.keys()):
            current_close = data_dict[t].loc[date, 'Close']
            if current_close > positions[t]['highest_price']:
                positions[t]['highest_price'] = current_close
                
            profit_from_entry = (current_close - positions[t]['entry_price']) / positions[t]['entry_price']
            drop_from_peak = (positions[t]['highest_price'] - current_close) / positions[t]['highest_price']
            
            # 💡 [질문자님 요청 반영] 발동은 +30% 이상부터, 컷 라인은 고점 대비 -10%로 유연하게 완화
            if profit_from_entry >= 0.30 and drop_from_peak >= 0.10:
                pending_orders.append({'ticker': t, 'type': 'SELL'})

        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        equity_history.append(cash + current_pos_val)
        
        if idx == len(all_dates) - 1:
            for t in list(positions.keys()):
                c_close = data_dict[t].loc[date, 'Close']
                pnl = (c_close * (1 - SLIPPAGE) - positions[t]['entry_price']) / positions[t]['entry_price']
                cash += positions[t]['qty'] * c_close * (1 - SLIPPAGE) * (1 - FEE_HALF)
                ticker_stats[t]['trades'] += 1
                if pnl > 0: ticker_stats[t]['wins'] += 1
                ticker_stats[t]['pnl_list'].append(pnl)
                del positions[t]
            equity_history[-1] = cash
            break

        # [4. SuperTrend 시그널 매도 감시]
        for t in list(positions.keys()):
            if data_dict[t].loc[date, 'Trend'] == -1 and t not in [o['ticker'] for o in pending_orders]:
                pending_orders.append({'ticker': t, 'type': 'SELL'})
                
        # [5. 상대강도(RS) 연산 및 허들 레이트 기반 순환매 판단]
        is_bull_market = qqq_close_30m.loc[date] > current_qqq_ema200
        target_pool = ["SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON"] if is_bull_market else ["SOXS"]
        
        all_candidates = []
        for ticker in target_pool:
            df = data_dict[ticker]
            b_idx = df.index.get_loc(date)
            if b_idx < 1 or pd.isna(df['Return_5d'].iloc[b_idx]) or pd.isna(qqq_ret_5d.loc[date]): continue
            
            rs_score = df['Return_5d'].iloc[b_idx] - qqq_ret_5d.loc[date]
            if df['Trend'].iloc[b_idx] == 1:
                all_candidates.append({'ticker': ticker, 'rs': rs_score, 'signal_buy': (df['Trend'].iloc[b_idx-1] == -1)})

        if all_candidates:
            all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)
            top_3_tickers = [c['ticker'] for c in all_candidates[:3]]
            
            for t in list(positions.keys()):
                if t in top_3_tickers or t in [o['ticker'] for o in pending_orders]: continue
                
                available_news = [c for c in all_candidates[:3] if c['ticker'] not in positions]
                if not available_news: continue
                best_new = available_news[0]
                
                current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
                
                if best_new['rs'] - current_rs > HURDLE_RATE:
                    pending_orders.append({'ticker': t, 'type': 'SELL'})
            
            for candidate in all_candidates:
                if len(positions) >= MAX_SLOTS: break
                if candidate['ticker'] not in positions and candidate['signal_buy']:
                    pending_orders.append({'ticker': candidate['ticker'], 'type': 'BUY'})

    equity_series = pd.Series(equity_history, index=all_dates)
    final_return = ((equity_series.iloc[-1] - initial_cash) / initial_cash) * 100
    mdd = ((equity_series - equity_series.cummax()) / equity_series.cummax()).min() * 100
    
    print("\n==========================================================")
    print("      🎯 [최종 성적표] Ver 5.9.1 변동성 완화 엔진 결과")
    print("==========================================================")
    print(f" 🟩 [누적 수익률]      : {final_return:+.2f}%")
    print(f" 🟥 [최대 낙폭 (MDD)]   : {mdd:.2f}%")
    print("==========================================================\n")
    
    print("==========================================================")
    print("      🔍 [진단 결과] 종목별 세부 누적 PnL")
    print("==========================================================")
    for t in full_universe:
        stats = ticker_stats[t]
        if stats['trades'] == 0: continue
        print(f" * {t:<6} : Trades {stats['trades']}회 | Total PnL {sum(stats['pnl_list'])*100:+.2f}%")
    print("==========================================================\n")

if __name__ == "__main__":
    run_v591_backtest()
    gc.collect()