import gc
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# 터미널 인코딩 깨짐 방지
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# ==========================================
# 1. 오리지널 기술 지표 연산 엔진 (SuperTrend)
# ==========================================
def calculate_supertrend(df, period=7, multiplier=3.0):
    high, low, close = df['High'], df['Low'], df['Close']
    # True Range 계산
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

# ==========================================
# 2. 순수 추세 추종 시뮬레이터
# ==========================================
def run_pure_trend_backtest():
    print("==========================================================")
    print("🔥 [트레이딩뷰 차트 동조형] 순수 추세 추종 백테스트 가동")
    print("🔥 매매 패널티(수수료+슬리피지): 왕복 0.45% 강제 차감")
    print("==========================================================")
    
    # 분석 유니버스 (레버리지 및 우량주 일체)
    universe = ["SOXL", "SOXS", "TQQQ", "SQQQ", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AVGO", "LLY", "AMD", "QLD", "USO"]
    leverage_tickers = ["SOXL", "SOXS", "TQQQ", "SQQQ"]
    
    start_date = "2022-01-01"
    end_date = "2026-06-01"
    
    print(f" -> 15종목 유니버스 데이터 다운로드 중 ({start_date} ~ {end_date})...")
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
            
            # 레버리지 종목은 변동성을 감안해 multiplier 4.5, 일반주는 차트 표준인 3.0 적용
            mult = 4.5 if ticker in leverage_tickers else 3.0
            df['Trend'], df['ATR'] = calculate_supertrend(df, period=7, multiplier=mult)
            
            data_dict[ticker] = df
            all_dates = df.index if all_dates is None else all_dates.intersection(df.index)
        except Exception:
            continue

    all_dates = sorted(list(all_dates))
    
    # 자금 변수 초기화
    initial_cash = 10000.0
    cash = initial_cash
    positions = {} # {ticker: {'qty': 수량, 'entry_price': 진입가, 'atr': 진입시ATR, 'half_exit': 플래그}}
    
    FEE_PENALTY = 0.0045 # 왕복 수수료 패널티 0.45%
    equity_history = []
    trade_logs = []
    
    # --- 타임 레이스 시작 ---
    for date in all_dates:
        # 1. 포지션 청산 및 익절 관리 (보유 종목 전수 조사)
        for t in list(positions.keys()):
            c_price = data_dict[t].loc[date, 'Close']
            trend = data_dict[t].loc[date, 'Trend']
            pos = positions[t]
            
            # 💡 [익절 전략]: 대시세 도중 최소한의 수익 잠금을 위한 ATR 가변 분할 익절 (+3.5배수)
            target_tp = pos['entry_price'] + (pos['atr'] * 3.5)
            if c_price >= target_tp and not pos['half_exit']:
                half_qty = pos['qty'] // 2
                if half_qty > 0:
                    cash += half_qty * c_price * (1 - FEE_PENALTY)
                    positions[t]['qty'] -= half_qty
                    positions[t]['half_exit'] = True
                    trade_logs.append((c_price - pos['entry_price']) / pos['entry_price'])
            
            # 💡 [청산 교정]: 잔파동 손절 제거! 오직 차트처럼 SuperTrend 매도 신호(-1) 전환 시에만 전량 청산
            if trend == -1:
                cash += positions[t]['qty'] * c_price * (1 - FEE_PENALTY)
                trade_logs.append((c_price - pos['entry_price']) / pos['entry_price'])
                del positions[t]
                
        # 2. 총 자산 평가액 기록
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        total_assets = cash + current_pos_val
        equity_history.append(total_assets)
        
        # 3. 신규 매수 시그널 탐색 (차트 신호와 동조화)
        buy_candidates = []
        for ticker in universe:
            if ticker in positions or date not in data_dict[ticker].index: continue
            
            df = data_dict[ticker]
            idx = df.index.get_loc(date)
            if idx < 1: continue
            
            # 💡 [진입 교정]: 횡보 필터(ADX)를 걷어내고, SuperTrend가 매도(-1)에서 매수(1)로 바뀐 '최초의 봉'을 저격
            if df['Trend'].iloc[idx] == 1 and df['Trend'].iloc[idx-1] == -1:
                buy_candidates.append({
                    'ticker': ticker, 'price': df['Close'].iloc[idx], 'atr': df['ATR'].iloc[idx], 'is_leverage': ticker in leverage_tickers
                })
                
        # 진입 후보가 있으면 자금 배분 후 즉시 매수
        for candidate in buy_candidates:
            t_ticker = candidate['ticker']
            is_lev = candidate['is_leverage']
            
            # 슬롯 계산 (레버리지 최대 8개, 일반주 최대 4개 한도 수용)
            lev_slots = sum([1 for t, d in positions.items() if d['is_leverage']])
            norm_slots = sum([1 for t, d in positions.items() if not d['is_leverage']])
            
            if (is_lev and lev_slots >= 8) or (not is_lev and norm_slots >= 4):
                continue # 슬롯 풀이면 패스
                
            base_pct = 0.125 if is_lev else 0.25
            target_amount = total_assets * base_pct * 0.995 # 안전마진 포함 비중 설정
            
            qty = int(target_amount // candidate['price'])
            cost = qty * candidate['price'] * (1 + FEE_PENALTY)
            
            if qty > 0 and cash >= cost:
                cash -= cost
                positions[t_ticker] = {
                    'qty': qty,
                    'entry_price': candidate['price'],
                    'atr': candidate['atr'],
                    'half_exit': False,
                    'is_leverage': is_lev
                }

    # ==========================================
    # 3. 5대 핵심 성과 지표 계산
    # ==========================================
    equity_series = pd.Series(equity_history, index=all_dates)
    
    # 1) 누적 수익률
    final_return = ((equity_series.iloc[-1] - initial_cash) / initial_cash) * 100
    
    # 2) MDD
    roll_max = equity_series.cummax()
    drawdowns = (equity_series - roll_max) / roll_max
    mdd = drawdowns.min() * 100
    
    # 3) 샤프 지수 (연율화)
    daily_returns = equity_series.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0
        
    # 4) 승률
    wins = [r for r in trade_logs if r > 0]
    losses = [r for r in trade_logs if r <= 0]
    win_rate = (len(wins) / len(trade_logs) * 100) if trade_logs else 0.0
    
    # 5) 손익비
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    profit_loss_ratio = (avg_win / avg_loss) if avg_loss != 0 else 0.0

    print("\n==========================================================")
    print("      🎯 [완결판] 트레이딩뷰 동조형 백테스트 스코어보드")
    print("==========================================================")
    print(f" 🟩 [누적 수익률]      : {final_return:+.2f}%")
    print(f" 🟥 [최대 낙폭 (MDD)]   : {mdd:.2f}%")
    print(f" 📊 [샤프 지수 (Sharpe)]: {sharpe_ratio:.2f}")
    print(f" ⚖️ [알고리즘 승률]    : {win_rate:.2f}% (총 {len(trade_logs)}회 거래)")
    print(f" 💰 [평균 손익비]      : {profit_loss_ratio:.2f} (평균익절: {avg_win*100:+.2f}% / 평균손절: {avg_loss*100:.2f}%)")
    print("==========================================================\n")

if __name__ == "__main__":
    run_pure_trend_backtest()
    gc.collect()