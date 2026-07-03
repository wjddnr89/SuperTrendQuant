import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# ==========================================
# ⚙️ 30분봉 백테스트 설정 구역 (최적점 Ver 6.7)
# ==========================================
INITIAL_CASH = 10000.0
MAX_SLOTS = 1          

HURDLE_ATR_MULT = 1.25  # 🥇 유저님의 데이터 검증 최적값 적용
ALLOW_LATE_CHASE = True # 실시간 대장주 추격 매수 허용

RS_PERIOD = 130        
FEE_HALF = 0.00225     
SLIPPAGE = 0.0005      
# ==========================================

def calculate_supertrend(df, period=7, multiplier=3.0):
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

print(f"📥 야후 파이낸스 최장 한계치 데이터 다운로드 중... (허들 배수: {HURDLE_ATR_MULT})")
universe = ["SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON", "SOXS"]

raw = yf.download(universe + ["QQQ"], period="60d", interval="30m", progress=False)

data_dict = {}
qqq_close = raw['Close']['QQQ'].dropna()
qqq_ret = qqq_close.pct_change(RS_PERIOD)

for ticker in universe:
    df = pd.DataFrame({
        'Open': raw['Open'][ticker], 'High': raw['High'][ticker],
        'Low': raw['Low'][ticker], 'Close': raw['Close'][ticker]
    }).dropna()
    if df.empty: continue
    
    mult = 4.5 if ticker in ["SOXL", "SOXS"] else 3.0
    df = calculate_supertrend(df, period=7, multiplier=mult)
    df['RS'] = df['Close'].pct_change(RS_PERIOD) - qqq_ret
    data_dict[ticker] = df

all_timestamps = sorted(list(set(qqq_close.index).intersection(*[df.index for df in data_dict.values()])))

cash = INITIAL_CASH
positions = {} 
history = []

total_trades_count = 0
holding_durations = []

print(f"⚙️ 총 {len(all_timestamps)}개의 30분봉 데이터 시뮬레이션 가동...")
for idx, ts in enumerate(all_timestamps):
    if idx < RS_PERIOD + 1: continue 
    
    current_portfolio_value = cash
    for t, pos in positions.items():
        current_portfolio_value += pos['qty'] * data_dict[t]['Close'].loc[ts]
    
    all_candidates = []
    for t, df in data_dict.items():
        curr_trend = df['Trend'].loc[ts]
        prev_trend = df['Trend'].iloc[df.index.get_loc(ts)-1]
        rs_score = df['RS'].loc[ts]
        atr_pct = df['ATR_pct'].loc[ts]
        price = df['Close'].loc[ts]
        
        if curr_trend == 1:
            is_buy_signal = True if ALLOW_LATE_CHASE else (prev_trend == -1)
            all_candidates.append({
                'ticker': t, 'rs': rs_score, 'signal_buy': is_buy_signal, 'price': price, 'atr_pct': atr_pct
            })
            
    all_candidates = sorted(all_candidates, key=lambda x: x['rs'], reverse=True)
    
    # STEP 1: 데드크로스 청산
    for t in list(positions.keys()):
        curr_trend = data_dict[t]['Trend'].loc[ts]
        if curr_trend == -1:
            price = data_dict[t]['Close'].loc[ts]
            real_sell_price = price * (1 - SLIPPAGE)
            cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
            holding_durations.append(idx - positions[t]['entry_idx'])
            del positions[t]

    # STEP 2: 동적 허들 순환매 교체 매도
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

        # STEP 3: 신규 진입
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

# 지표 연산
res_df = pd.DataFrame(history).set_index('timestamp')
res_df['Peak'] = res_df['total_value'].cummax()
res_df['Drawdown'] = (res_df['total_value'] - res_df['Peak']) / res_df['Peak']

total_return = (res_df['total_value'].iloc[-1] - INITIAL_CASH) / INITIAL_CASH * 100
mdd = res_df['Drawdown'].min() * 100
qqq_return = (qqq_close.loc[res_df.index[-1]] - qqq_close.loc[res_df.index[0]]) / qqq_close.loc[res_df.index[0]] * 100

# 🌟 [유저님 피드백 반영] 유니버스 동일 비중 단순 보유(Buy & Hold) 수익률 연산 구역
start_ts = res_df.index[0]
end_ts = res_df.index[-1]
universe_returns = []

for ticker in universe:
    if ticker in data_dict:
        p_start = data_dict[ticker]['Close'].loc[start_ts]
        p_end = data_dict[ticker]['Close'].loc[end_ts]
        ret = (p_end - p_start) / p_start * 100
        universe_returns.append(ret)

universe_bh_avg_return = np.mean(universe_returns) # 유니버스 단순 보유 평균 수익률

avg_holding_bars = np.mean(holding_durations) if holding_durations else 0
avg_holding_hours = (avg_holding_bars * 30) / 60

# ==================================================
# 🏆 업데이트된 백테스트 보고서 출력 (진짜 알파 검증)
# ==================================================
print("\n" + "="*50)
print(f"🏆 오리지널 30분봉 백테스트 보고서 (HURDLE MULT: {HURDLE_ATR_MULT})")
print("="*50)
print(f"🔥 30분봉 전략 누적 수익률: {total_return:+.2f}%")
print(f"📉 30분봉 전략 최악 낙폭 (MDD): {mdd:.2f}%")
print("-" * 50)
print(f"📊 [기존 BM] QQQ 동기간 단순 보유 수익률: {qqq_return:+.2f}%")
print(f"💎 [진짜 BM] 유니버스 동일 비중 단순 보유 수익률: {universe_bh_avg_return:+.2f}%")
print("-" * 50)
print(f"✨ 진짜 알파 (전략 수익률 - 유니버스 보유): {total_return - universe_bh_avg_return:+.2f}%")
print(f"🔄 [회전율 매트릭스] 총 매수 진입 횟수   : {total_trades_count}회")
print(f"⏱️ [포지션 생명주기] 평균 보유 기간      : {avg_holding_bars:.1f}개 봉 (약 {avg_holding_hours:.1f} 영업시간)")
print("="*50)