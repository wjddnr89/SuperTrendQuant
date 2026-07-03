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

def run_30m_trend_backtest():
    print("==========================================================")
    print("⏱️ [30분봉 전면 복귀] 고해상도 실시간 동조형 백테스트 가동")
    print("🎯 조건: 레버리지/우량주 무조건 한 종목당 비중 25% 고정 (최대 4슬롯)")
    print("🎯 청산: 30분봉 상 SuperTrend 매도(-1) 전환 시 즉시 100% 청산")
    print("==========================================================")
    
    # RKLB 및 주도주 유니버스
    universe = ["SOXL", "SOXS", "TQQQ", "SQQQ", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AVGO", "LLY", "AMD", "QLD", "USO"]
    leverage_tickers = ["SOXL", "SOXS", "TQQQ", "SQQQ"]
    
    # yfinance 30분봉 최대 허용 범위 (최근 60일 고정)
    print(" -> 15종목 30분봉 초정밀 데이터 다운로드 중 (최근 60일)...")
    raw_data = yf.download(universe, period="60d", interval="30m", progress=False)
    
    data_dict = {}
    all_dates = None
    
    for ticker in universe:
        try:
            df = pd.DataFrame({
                'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
                'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker]
            }).dropna()
            
            if df.empty: continue
            
            # 30분봉 민감도를 맞추기 위한 트레이딩뷰 표준 멀티플라이어 세팅
            mult = 4.5 if ticker in leverage_tickers else 3.0
            df['Trend'], df['ATR'] = calculate_supertrend(df, period=7, multiplier=mult)
            
            data_dict[ticker] = df
            all_dates = df.index if all_dates is None else all_dates.intersection(df.index)
        except Exception:
            continue

    all_dates = sorted(list(all_dates))
    
    initial_cash = 10000.0
    cash = initial_cash
    positions = {} 
    
    FEE_PENALTY = 0.0045 # 왕복 패널티 0.45%
    equity_history = []
    trade_logs = []
    
    for date in all_dates:
        # 1. 30분봉 단위 실시간 청산 감시
        for t in list(positions.keys()):
            c_price = data_dict[t].loc[date, 'Close']
            trend = data_dict[t].loc[date, 'Trend']
            pos = positions[t]
            
            # 30분봉에서 매도 신호(-1) 뜨면 시차 없이 그 즉시 100% 전량 탈출
            if trend == -1:
                cash += positions[t]['qty'] * c_price * (1 - FEE_PENALTY)
                trade_logs.append((c_price - pos['entry_price']) / pos['entry_price'])
                del positions[t]
                
        # 2. 총 자산 평가액 기록
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        total_assets = cash + current_pos_val
        equity_history.append(total_assets)
        
        # 3. 30분봉 단위 신규 매수 시그널 탐색
        buy_candidates = []
        for ticker in universe:
            if ticker in positions or date not in data_dict[ticker].index: continue
            
            df = data_dict[ticker]
            idx = df.index.get_loc(date)
            if idx < 1: continue
            
            # SuperTrend 매수 전환 최초 30분봉 확인 즉시 진입
            if df['Trend'].iloc[idx] == 1 and df['Trend'].iloc[idx-1] == -1:
                buy_candidates.append({
                    'ticker': ticker, 'price': df['Close'].iloc[idx], 'atr': df['ATR'].iloc[idx], 'is_leverage': ticker in leverage_tickers
                })
                
        # 진입 시뮬레이션
        for candidate in buy_candidates:
            t_ticker = candidate['ticker']
            
            if len(positions) >= 4:
                continue 
                
            # 지정하신 대로 우량주/레버리지 불문하고 과감하게 자산의 25%씩 베팅
            base_pct = 0.25
            target_amount = total_assets * base_pct * 0.995 
            
            qty = int(target_amount // candidate['price'])
            cost = qty * candidate['price'] * (1 + FEE_PENALTY)
            
            if qty > 0 and cash >= cost:
                cash -= cost
                positions[t_ticker] = {
                    'qty': qty,
                    'entry_price': candidate['price'],
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
    print("      🎯 [완료] 30분봉 하이퍼 버전 최종 성적표")
    print("==========================================================")
    print(f" 🟩 [누적 수익률]      : {final_return:+.2f}%")
    print(f" 🟥 [최대 낙폭 (MDD)]   : {mdd:.2f}%")
    print(f" 📊 [샤프 지수 (Sharpe)]: {sharpe_ratio:.2f}")
    print(f" ⚖️ [알고리즘 승률]    : {win_rate:.2f}% (총 {len(trade_logs)}회 거래)")
    print(f" 💰 [평균 손익비]      : {profit_loss_ratio:.2f} (평균익절: {avg_win*100:+.2f}% / 평균손절: {avg_loss*100:.2f}%)")
    print("==========================================================\n")

if __name__ == "__main__":
    run_30m_trend_backtest()
    gc.collect()