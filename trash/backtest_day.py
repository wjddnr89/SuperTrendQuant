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

def run_final_daily_backtest():
    print("==========================================================")
    print("🔥 [일봉 복귀 완결판] 대시세 락인(Lock-in) 추적 백테스트 가동")
    print("🎯 조건: 레버리지 / 우량주 무조건 진입 시 비중 25% 고정 (최대 4슬롯)")
    print("🎯 청산: 최고점 대비 -12% 붕괴 시 혹은 SuperTrend 매도 전환 시 100% 청산")
    print("==========================================================")
    
    universe = ["SOXL", "SOXS", "TQQQ", "SQQQ", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AVGO", "LLY", "AMD", "QLD", "USO"]
    leverage_tickers = ["SOXL", "SOXS", "TQQQ", "SQQQ"]
    
    start_date = "2022-01-01"
    end_date = "2026-06-01"
    
    print(f" -> 15종목 일봉 4년치 데이터 다운로드 중...")
    raw_data = yf.download(universe, start=start_date, end=end_date, progress=False)
    
    data_dict = {}
    all_dates = None
    
    for ticker in universe:
        try:
            df = pd.DataFrame({
                'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
                'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker]
            }).dropna()
            
            if df.empty: continue
            
            mult = 4.5 if ticker in leverage_tickers else 3.0
            df['Trend'], df['ATR'] = calculate_supertrend(df, period=7, multiplier=mult)
            
            data_dict[ticker] = df
            all_dates = df.index if all_dates is None else all_dates.intersection(df.index)
        except Exception:
            continue

    all_dates = sorted(list(all_dates))
    
    initial_cash = 10000.0
    cash = initial_cash
    positions = {} # {ticker: {'qty': 수량, 'entry_price': 진입가, 'highest_price': 진입이후최고가}}
    
    FEE_PENALTY = 0.0045 
    DROP_THRESHOLD = 0.12 # 최고점 대비 -12% 밀리면 익절 락인 익스프레스 스위치
    
    equity_history = []
    trade_logs = []
    
    for date in all_dates:
        # 1. 보유 포지션 최고가 갱신 및 가변 트레일링 스톱 감시
        for t in list(positions.keys()):
            c_price = data_dict[t].loc[date, 'Close']
            trend = data_dict[t].loc[date, 'Trend']
            pos = positions[t]
            
            # 진입 이후 주가의 최고점 실시간 갱신 트래킹
            if c_price > pos['highest_price']:
                positions[t]['highest_price'] = c_price
            
            # 🚨 [Ver 6.0 핵심]: 최고점 대비 지정된 비율(-12%) 이상 하락했는지 감시
            trailing_stop_line = positions[t]['highest_price'] * (1 - DROP_THRESHOLD)
            
            if c_price < trailing_stop_line or trend == -1:
                cash += positions[t]['qty'] * c_price * (1 - FEE_PENALTY)
                trade_logs.append((c_price - pos['entry_price']) / pos['entry_price'])
                del positions[t]
                
        # 2. 총 자산 평가액 기록
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        total_assets = cash + current_pos_val
        equity_history.append(total_assets)
        
        # 3. 신규 매수 시그널 탐색 (순수 턴어라운드 저격)
        buy_candidates = []
        for ticker in universe:
            if ticker in positions or date not in data_dict[ticker].index: continue
            
            df = data_dict[ticker]
            idx = df.index.get_loc(date)
            if idx < 1: continue
            
            if df['Trend'].iloc[idx] == 1 and df['Trend'].iloc[idx-1] == -1:
                buy_candidates.append({
                    'ticker': ticker, 'price': df['Close'].iloc[idx], 'is_leverage': ticker in leverage_tickers
                })
                
        # 진입 시뮬레이션
        for candidate in buy_candidates:
            t_ticker = candidate['ticker']
            
            if len(positions) >= 4:
                continue 
                
            base_pct = 0.25
            target_amount = total_assets * base_pct * 0.995 
            
            qty = int(target_amount // candidate['price'])
            cost = qty * candidate['price'] * (1 + FEE_PENALTY)
            
            if qty > 0 and cash >= cost:
                cash -= cost
                positions[t_ticker] = {
                    'qty': qty,
                    'entry_price': candidate['price'],
                    'highest_price': candidate['price'], # 진입가가 초기 최고가
                    'is_leverage': candidate['is_leverage']
                }

    # 지표 연산
    equity_series = pd.Series(equity_history, index=all_dates)
    final_return = ((equity_series.iloc[-1] - initial_cash) / initial_cash) * 100
    
    roll_max = equity_series.cummax()
    drawdowns = (equity_series - roll_max) / roll_max
    mdd = drawdowns.min() * 100
    
    daily_returns = equity_series.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0
        
    wins = [r for r in trade_logs if r > 0]
    losses = [r for r in trade_logs if r <= 0]
    win_rate = (len(wins) / len(trade_logs) * 100) if trade_logs else 0.0
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    profit_loss_ratio = (avg_win / avg_loss) if avg_loss != 0 else 0.0

    print("\n==========================================================")
    print("      🎯 [완료] 일봉 완결판 트레일링 버전 최종 스코어보드")
    print("==========================================================")
    print(f" 🟩 [누적 수익률]      : {final_return:+.2f}%")
    print(f" 🟥 [최대 낙폭 (MDD)]   : {mdd:.2f}%")
    print(f" 📊 [샤프 지수 (Sharpe)]: {sharpe_ratio:.2f}")
    print(f" ⚖️ [알고리즘 승률]    : {win_rate:.2f}% (총 {len(trade_logs)}회 거래)")
    print(f" 💰 [평균 손익비]      : {profit_loss_ratio:.2f} (평균익절: {avg_win*100:+.2f}% / 평균손절: {avg_loss*100:.2f}%)")
    print("==========================================================\n")

if __name__ == "__main__":
    run_final_daily_backtest()
    gc.collect()