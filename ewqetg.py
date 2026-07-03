import os
import numpy as np
import pandas as pd
import yfinance as yf
import datetime

FULL_UNIVERSE = ["SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON", "SOXS"]
START_CASH = 10000.0
MAX_SLOTS = 2
HURDLE_RATE = 0.015

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

print("⏳ [버전 A] 야후 파이낸스로부터 59일 데이터 수집 중 (마켓 필터 제외)...")
raw_data = yf.download(FULL_UNIVERSE + ["QQQ"], period="59d", interval="30m", progress=False)

data_dict = {}
all_indices = sorted(list(raw_data['Close']['QQQ'].dropna().index))

for ticker in FULL_UNIVERSE + ["QQQ"]:
    df = pd.DataFrame({
        'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
        'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker]
    }).dropna()
    if df.empty: continue
    
    mult = 4.5 if ticker in ["SOXL", "SOXS"] else 3.0
    df['Trend'] = calculate_supertrend(df, period=7, multiplier=mult)
    df['Return_5d'] = df['Close'].pct_change(65)
    data_dict[ticker] = df

cash = START_CASH
positions = {}
trade_history = []
portfolio_log = []

for idx in all_indices:
    if idx not in data_dict['QQQ'].index: continue
    qqq_ret_5d = data_dict['QQQ'].loc[idx, 'Return_5d']
    
    all_candidates = []
    for ticker in FULL_UNIVERSE:
        if ticker not in data_dict or idx not in data_dict[ticker].index: continue
        df = data_dict[ticker]
        loc_idx = df.index.get_loc(idx)
        if loc_idx < 1: continue
        
        curr_trend = df['Trend'].iloc[loc_idx]
        prev_trend = df['Trend'].iloc[loc_idx - 1]
        rs_score = df['Return_5d'].iloc[loc_idx] - qqq_ret_5d
        price = df['Close'].iloc[loc_idx]
        
        if curr_trend == 1:
            all_candidates.append({
                'ticker': ticker, 'rs': rs_score,
                'signal_buy': (prev_trend == -1), 'price': price
            })
            
    for t in list(positions.keys()):
        if t not in data_dict or idx not in data_dict[t].index: continue
        curr_price = data_dict[t].loc[idx, 'Close']
        curr_trend = data_dict[t].loc[idx, 'Trend']
        
        if curr_price > positions[t]['highest_price']:
            positions[t]['highest_price'] = curr_price
            
        profit = (curr_price - positions[t]['entry_price']) / positions[t]['entry_price']
        drop = (positions[t]['highest_price'] - curr_price) / positions[t]['highest_price']
        
        sell_reason = ""
        if profit >= 0.30 and drop >= 0.10:
            sell_reason = "Trailing"
        elif curr_trend == -1:
            sell_reason = "DeadCross"
            
        if sell_reason:
            qty = positions[t]['qty']
            recv_cash = qty * curr_price * (1 - 0.00225 - 0.0005)
            cash += recv_cash
            trade_history.append({"pnl_pct": profit * 100, "success": profit > 0})
            del positions[t]

    if all_candidates:
        all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)
        top_2_tickers = [c['ticker'] for c in all_candidates[:2]]
        
        for t in list(positions.keys()):
            if t in top_2_tickers: continue
            available_news = [c for c in all_candidates[:2] if c['ticker'] not in positions]
            if not available_news: continue
            best_new = available_news[0]
            current_rs = next((c['rs'] for c in all_candidates if c['ticker'] == t), -999.0)
            
            if best_new['rs'] - current_rs > HURDLE_RATE:
                curr_price = data_dict[t].loc[idx, 'Close']
                qty = positions[t]['qty']
                recv_cash = qty * curr_price * (1 - 0.00225 - 0.0005)
                cash += recv_cash
                pnl_pct = (curr_price - positions[t]['entry_price']) / positions[t]['entry_price'] * 100
                trade_history.append({"pnl_pct": pnl_pct, "success": pnl_pct > 0})
                del positions[t]

        for candidate in all_candidates:
            if len(positions) >= MAX_SLOTS: break
            t = candidate['ticker']
            if t not in positions and candidate['signal_buy']:
                curr_price = candidate['price']
                current_assets = cash + sum([p['qty'] * data_dict[pos_t].loc[idx, 'Close'] for pos_t, p in positions.items() if idx in data_dict[pos_t].index])
                target_unit = current_assets * (1 / MAX_SLOTS) * 0.995
                alloc = min(cash, target_unit)
                qty = int(alloc // (curr_price * 1.0005))
                cost = qty * curr_price * (1 + 0.00225 + 0.0005)
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t] = {'qty': qty, 'entry_price': curr_price, 'highest_price': curr_price}

    total_val = cash + sum([p['qty'] * data_dict[pos_t].loc[idx, 'Close'] for pos_t, p in positions.items() if idx in data_dict[pos_t].index])
    portfolio_log.append({"timestamp": idx, "total_value": total_val})

port_df = pd.DataFrame(portfolio_log).set_index("timestamp")
port_df['returns'] = port_df['total_value'].pct_change().fillna(0)
final_value = port_df['total_value'].iloc[-1]
total_return = ((final_value - START_CASH) / START_CASH) * 100
sharpe_ratio = (port_df['returns'].mean() / port_df['returns'].std()) * np.sqrt(13 * 252) if port_df['returns'].std() != 0 else 0
port_df['peak'] = port_df['total_value'].cummax()
mdd = ((port_df['total_value'] - port_df['peak']) / port_df['peak']).min() * 100
th_df = pd.DataFrame(trade_history)
win_rate = (th_df['success'].sum() / len(th_df)) * 100 if not th_df.empty else 0
profit_factor = (th_df[th_df['pnl_pct'] > 0]['pnl_pct'].mean() / abs(th_df[th_df['pnl_pct'] < 0]['pnl_pct'].mean())) if not th_df.empty and th_df['pnl_pct'].min() < 0 else 0

print("\n=============================================")
print(f" 📈 [버전 A] 마켓 필터 제외 59일 결과 보고서")
print("=============================================")
print(f"🔥 1. 누적 수익률   : {total_return:.2f}%")
print(f"🔥 2. 샤프 비율     : {sharpe_ratio:.2f}")
print(f"🔥 3. 승률          : {win_rate:.2f}% (총 {len(trade_history)}회 청산)")
print(f"🔥 4. 손익비        : {profit_factor:.2f} : 1")
print(f"🔥 5. 최대 낙폭(MDD): {mdd:.2f}%")
print("=============================================")