import gc
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# ==========================================================
# 1. 2022년 대폭락장 변동성 매트릭스 30분봉 시뮬레이터 엔진
# ==========================================================
def generate_2022_crash_30m_data(ticker, seed_val):
    """
    2022년 실제 역사적 하락 궤적(SOXL -90%, TQQQ -80%, NVDA -50%)을
    30분봉 단위(하루 13개 봉, 1년 약 3,200개 봉)로 완벽하게 재현하는 합성 엔진
    """
    np.random.seed(seed_val)
    # 2022년 영업일 기준 30분봉 타임라인 생성
    dr = pd.date_range(start="2022-01-03 09:30", end="2022-12-30 16:00", freq="30min")
    # 정규장 시간(09:30 ~ 16:00)만 필터링
    dr = dr[(dr.hour > 9) | ((dr.hour == 9) & (dr.minute >= 30))]
    dr = dr[dr.hour < 16]
    
    n_bars = len(dr)
    
    # 종목별 2022년 실제 연간 하락 타겟 수익률 및 변동성 설정
    if ticker in ["SOXL", "SOXS"]:
        target_trend = -2.3 / n_bars  # 연간 약 -90% 폭락 궤적
        volatility = 0.025            # 3배 레버리지 특유의 극심한 장중 흔들림
    elif ticker in ["TQQQ", "SQQQ"]:
        target_trend = -1.6 / n_bars  # 연간 약 -80% 폭락 궤적
        volatility = 0.018
    else:
        target_trend = -0.7 / n_bars  # 일반 우량주 연간 약 -50% 조정 궤적
        volatility = 0.012

    # 헤지 종목(인버스)은 반대 궤적 부여
    if ticker in ["SQQQ", "SOXS"]:
        target_trend = -target_trend * 0.4 # 인버스도 변동성 잠식으로 생각보다 못 버팀 반영

    # 30분봉 캔들 생성
    returns = np.random.normal(loc=target_trend, scale=volatility, size=n_bars)
    price_path = 100.0 * np.exp(np.cumsum(returns))
    
    # 고가, 저가, 시가 노이즈 반영
    high_noise = np.abs(np.random.normal(0, volatility * 0.5, n_bars))
    low_noise = np.abs(np.random.normal(0, volatility * 0.5, n_bars))
    
    df = pd.DataFrame(index=dr)
    df['Close'] = price_path
    df['High'] = price_path * (1 + high_noise)
    df['Low'] = price_path * (1 - low_noise)
    df['Open'] = df['Close'].shift(1).fillna(100.0)
    
    return df

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

# ==========================================================
# 2. 30분봉 폭락장 시뮬레이션 가동
# ==========================================================
def run_crash_30m_backtest():
    print("==========================================================")
    print("💀 [스트레스 테스트] 2022 대폭락장 관통 30분봉 백테스트 가동")
    print("🎯 조건: 레버리지/우량주 무조건 진입 시 비중 25% 고정 (최대 4슬롯)")
    print("🎯 환경: SOXL -90% / TQQQ -80% 지옥의 일방향 하락장 시뮬레이션")
    print("==========================================================")
    
    universe = ["SOXL", "SOXS", "TQQQ", "SQQQ", "NVDA", "AAPL", "MSFT", "AMZN"]
    leverage_tickers = ["SOXL", "SOXS", "TQQQ", "SQQQ"]
    
    data_dict = {}
    all_dates = None
    
    print(" -> 2022년 전 종목 고해상도 30분봉 역사적 궤적 생성 중...")
    for seed_idx, ticker in enumerate(universe):
        df = generate_2022_crash_30m_data(ticker, seed_val=seed_idx + 42)
        mult = 4.5 if ticker in leverage_tickers else 3.0
        df['Trend'], df['ATR'] = calculate_supertrend(df, period=7, multiplier=mult)
        data_dict[ticker] = df
        all_dates = df.index if all_dates is None else all_dates.intersection(df.index)
        
    all_dates = sorted(list(all_dates))
    
    initial_cash = 10000.0
    cash = initial_cash
    positions = {}
    
    FEE_PENALTY = 0.0045 # 왕복 패널티 0.45% 고정
    equity_history = []
    trade_logs = []
    
    for date in all_dates:
        # 1. 30분봉 단위 실시간 청산 감시
        for t in list(positions.keys()):
            c_price = data_dict[t].loc[date, 'Close']
            trend = data_dict[t].loc[date, 'Trend']
            pos = positions[t]
            
            if trend == -1: # 매도 신호 즉시 전량 손절 탈출
                cash += positions[t]['qty'] * c_price * (1 - FEE_PENALTY)
                trade_logs.append((c_price - pos['entry_price']) / pos['entry_price'])
                del positions[t]
                
        # 2. 자산 평가
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        total_assets = cash + current_pos_val
        equity_history.append(total_assets)
        
        # 3. 신규 매수 시그널 탐색 (하락장 속 데드캣 바운스 저격)
        buy_candidates = []
        for ticker in universe:
            if ticker in positions: continue
            
            df = data_dict[ticker]
            idx = df.index.get_loc(date)
            if idx < 1: continue
            
            if df['Trend'].iloc[idx] == 1 and df['Trend'].iloc[idx-1] == -1:
                buy_candidates.append({'ticker': ticker, 'price': df['Close'].iloc[idx]})
                
        for candidate in buy_candidates:
            t_ticker = candidate['ticker']
            if len(positions) >= 4: break
                
            base_pct = 0.25
            target_amount = total_assets * base_pct * 0.995
            qty = int(target_amount // candidate['price'])
            cost = qty * candidate['price'] * (1 + FEE_PENALTY)
            
            if qty > 0 and cash >= cost:
                cash -= cost
                positions[t_ticker] = {'qty': qty, 'entry_price': candidate['price']}

    # 결과 분석
    equity_series = pd.Series(equity_history, index=all_dates)
    final_return = ((equity_series.iloc[-1] - initial_cash) / initial_cash) * 100
    
    roll_max = equity_series.cummax()
    drawdowns = (equity_series - roll_max) / roll_max
    mdd = drawdowns.min() * 100
    
    daily_returns = equity_series.resample('D').last().pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0
    
    wins = [r for r in trade_logs if r > 0]
    losses = [r for r in trade_logs if r <= 0]
    win_rate = (len(wins) / len(trade_logs) * 100) if trade_logs else 0.0
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    profit_loss_ratio = (avg_win / avg_loss) if avg_loss != 0 else 0.0

    print("\n==========================================================")
    print("      🎯 [결과] 2022 지옥의 폭락장 30분봉 최종 성적표")
    print("==========================================================")
    print(f" 🟩 [누적 수익률]      : {final_return:+.2f}%")
    print(f" 🟥 [최대 낙폭 (MDD)]   : {mdd:.2f}%")
    print(f" 📊 [샤프 지수 (Sharpe)]: {sharpe_ratio:.2f}")
    print(f" ⚖️ [알고리즘 승률]    : {win_rate:.2f}% (총 {len(trade_logs)}회 거래)")
    print(f" 💰 [평균 손익비]      : {profit_loss_ratio:.2f} (평균익절: {avg_win*100:+.2f}% / 평균손절: {avg_loss*100:.2f}%)")
    print("==========================================================\n")

if __name__ == "__main__":
    run_crash_30m_backtest()
    gc.collect()