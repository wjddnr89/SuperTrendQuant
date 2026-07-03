import gc
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

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
    
    for i in range(1, len(df)):
        final_ub.iloc[i] = basic_ub.iloc[i] if basic_ub.iloc[i] < final_ub.iloc[i-1] or close.iloc[i-1] > final_ub.iloc[i-1] else final_ub.iloc[i-1]
        final_lb.iloc[i] = basic_lb.iloc[i] if basic_lb.iloc[i] > final_lb.iloc[i-1] or close.iloc[i-1] < final_lb.iloc[i-1] else final_lb.iloc[i-1]
            
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if trend.iloc[i-1] == 1:
            trend.iloc[i] = -1 if close.iloc[i] < final_lb.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if close.iloc[i] > final_ub.iloc[i] else -1
    return trend, atr

def calculate_adx(df, window=14):
    high, low, close = df['High'], df['Low'], df['Close']
    upmove = high - high.shift(1)
    downmove = low.shift(1) - low
    
    plus_dm = np.where((upmove > downmove) & (upmove > 0), upmove, 0.0)
    minus_dm = np.where((downmove > upmove) & (downmove > 0), downmove, 0.0)
    
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=window).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=window).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(window=window).mean()

def run_hell_backtest_v4():
    print("==========================================================")
    print("🔥 [조율 D 반영] 2022 대폭락장 관통 복리 백테스트 (Ver 4.0)")
    print("🎯 수정 사항: 1차 익절 달성 후 청산 마디 가변 우상향 락인(Lock-in)")
    print("==========================================================")
    
    universe = ["SOXL", "SOXS", "TQQQ", "SQQQ", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AVGO", "LLY", "AMD", "QLD", "USO"]
    leverage_tickers = ["SOXL", "SOXS", "TQQQ", "SQQQ"]
    
    start_date = "2022-01-01"
    end_date = "2026-06-01"
    
    raw_data = yf.download(universe, start=start_date, end=end_date, progress=False)
    
    data_dict = {}
    all_dates = None
    
    for ticker in universe:
        try:
            df = pd.DataFrame({
                'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
                'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker], 'Volume': raw_data['Volume'][ticker]
            }).dropna()
            if df.empty: continue
            
            mult = 4.5 if ticker in leverage_tickers else 3.0
            df['Trend'], df['ATR'] = calculate_supertrend(df, period=7, multiplier=mult)
            df['ADX'] = calculate_adx(df, window=14)
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            
            df = df.dropna()
            data_dict[ticker] = df
            all_dates = df.index if all_dates is None else all_dates.intersection(df.index)
        except Exception:
            continue

    all_dates = sorted(list(all_dates))
    
    initial_cash = 10000.0
    cash = initial_cash
    positions = {}
    FEE_PENALTY = 0.0045
    
    # 파라미터 고정
    ATR_TARGET_MULTIPLIER = 2.5
    ADX_ENTRY_BARRIER = 30       
    TRAILING_BUFFER = 0.015       
    
    equity_history = []
    trade_logs = []
    peak_assets = initial_cash
    emergency_mode = False
    
    for date in all_dates:
        # 1. 실시간 가격 업데이트 및 청산/리스크 관리 감시
        for t in list(positions.keys()):
            c_price = data_dict[t].loc[date, 'Close']
            pos = positions[t]
            entry_p = pos['entry_price']
            atr = pos['atr']
            trend = data_dict[t].loc[date, 'Trend']
            
            # 진입 후 해당 종목의 최고점 갱신 트래킹
            if c_price > pos['highest_price']:
                positions[t]['highest_price'] = c_price
            
            # 가변 익절라인 검사
            target_tp = entry_p + (atr * ATR_TARGET_MULTIPLIER)
            if c_price >= target_tp and not pos['half_exit']:
                half_qty = pos['qty'] // 2
                if half_qty > 0:
                    cash += half_qty * c_price * (1 - FEE_PENALTY)
                    positions[t]['qty'] -= half_qty
                    positions[t]['half_exit'] = True
                    # 🚨 조율 D 핵심: 1차 익절을 달성한 순간, 청산 마디 기준점을 본전에서 '익절가'로 락인(Lock-in)
                    positions[t]['stop_loss_line'] = entry_p * 1.05 # 최소 본전+5% 확보
                    trade_logs.append((c_price - entry_p) / entry_p)
            
            # 🚨 조율 D 반영: 익절 전에는 기존 버퍼(-1.5%) 적용, 익절 후에는 고정 락인선 혹은 최고점 대비 -2*ATR 추적 적용
            if pos['half_exit']:
                # 익절 후에는 수익 보존을 위해 타이트하게 최고점 대비 -2.5*ATR 지점까지 밀리면 잔여물량 전량 매도
                atr_trailing = positions[t]['highest_price'] - (atr * 2.5)
                final_stop_line = max(pos['stop_loss_line'], atr_trailing)
            else:
                final_stop_line = entry_p * (1 - TRAILING_BUFFER)
            
            if c_price < final_stop_line or trend == -1:
                cash += positions[t]['qty'] * c_price * (1 - FEE_PENALTY)
                trade_logs.append((c_price - entry_p) / entry_p)
                del positions[t]
                
        # 2. 총 자산 평가 및 하방 가드 브레이크
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        total_assets = cash + current_pos_val
        equity_history.append(total_assets)
        
        if total_assets > peak_assets: peak_assets = total_assets
        drawdown = (total_assets - peak_assets) / peak_assets
        if drawdown <= -0.15: emergency_mode = True
        if emergency_mode and drawdown >= -0.05: emergency_mode = False
            
        # 3. 신규 매수 시그널 탐색 및 진입 후보 정렬
        buy_candidates = []
        for ticker in universe:
            if ticker in positions or date not in data_dict[ticker].index: continue
            df = data_dict[ticker]
            idx = df.index.get_loc(date)
            if idx < 2: continue
            
            if df['Trend'].iloc[idx] == 1 and df['Trend'].iloc[idx-1] == 1 and (df['Volume'].iloc[idx] > df['Vol_MA20'].iloc[idx-1]) and (df['ADX'].iloc[idx] >= ADX_ENTRY_BARRIER):
                buy_candidates.append({
                    'ticker': ticker, 'price': df['Close'].iloc[idx], 'atr': df['ATR'].iloc[idx], 'adx': df['ADX'].iloc[idx], 'is_leverage': ticker in leverage_tickers
                })
                
        if buy_candidates:
            best = sorted(buy_candidates, key=lambda x: x['adx'], reverse=True)[0]
            t_ticker = best['ticker']
            is_lev = best['is_leverage']
            
            base_pct = 0.125 if is_lev else 0.25
            if emergency_mode: base_pct *= 0.5
            
            target_amount = total_assets * base_pct * 0.995
            qty = int(target_amount // best['price'])
            cost = qty * best['price'] * (1 + FEE_PENALTY)
            
            lev_slots = sum([1 for t, d in positions.items() if d['is_leverage']])
            norm_slots = sum([1 for t, d in positions.items() if not d['is_leverage']])
            
            if (is_lev and lev_slots < 8) or (not is_lev and norm_slots < 4):
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t_ticker] = {'qty': qty, 'entry_price': best['price'], 'atr': best['atr'], 'half_exit': False, 'is_leverage': is_lev, 'adx': best['adx'], 'highest_price': best['price'], 'stop_loss_line': best['price'] * (1 - TRAILING_BUFFER)}
            else:
                same_group = {t: d for t, d in positions.items() if d['is_leverage'] == is_lev}
                if same_group:
                    weakest = min(same_group, key=lambda x: same_group[x]['adx'])
                    if best['adx'] > (positions[weakest]['adx'] + 5.0):
                        cash += positions[weakest]['qty'] * data_dict[weakest].loc[date, 'Close'] * (1 - FEE_PENALTY)
                        del positions[weakest]
                        if cash >= cost and qty > 0:
                            cash -= cost
                            positions[t_ticker] = {'qty': qty, 'entry_price': best['price'], 'atr': best['atr'], 'half_exit': False, 'is_leverage': is_lev, 'adx': best['adx'], 'highest_price': best['price'], 'stop_loss_line': best['price'] * (1 - TRAILING_BUFFER)}

    # 지표 연산
    equity_series = pd.Series(equity_history, index=all_dates)
    final_return = ((equity_series.iloc[-1] - initial_cash) / initial_cash) * 100
    mdd = ((equity_series - equity_series.cummax()) / equity_series.cummax()).min() * 100
    daily_returns = equity_series.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    wins = [r for r in trade_logs if r > 0]
    losses = [r for r in trade_logs if r <= 0]
    win_rate = (len(wins) / len(trade_logs) * 100) if trade_logs else 0
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_loss_ratio = (avg_win / avg_loss) if avg_loss != 0 else 0

    print("\n==========================================================")
    print("      🎯 [Ver 4.0] 가변 트레일링 스톱 적용 완료 스코어보드")
    print("==========================================================")
    print(f" 🟩 [누적 수익률]      : {final_return:+.2f}%")
    print(f" 🟥 [최대 낙폭 (MDD)]   : {mdd:.2f}%")
    print(f" 📊 [샤프 지수 (Sharpe)]: {sharpe_ratio:.2f}")
    print(f" ⚖️ [알고리즘 승률]    : {win_rate:.2f}% (총 {len(trade_logs)}회 거래)")
    print(f" 💰 [평균 손익비]      : {profit_loss_ratio:.2f} (평균익절: {avg_win*100:+.2f}% / 평균손절: {avg_loss*100:.2f}%)")
    print("==========================================================\n")

if __name__ == "__main__":
    run_hell_backtest_v4()
    gc.collect()