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

def run_perfect_30m_backtest():
    print("==========================================================")
    print("🎯 [Ver 5.1] 왜곡 차단 + QQQ 필터 탑재 초정밀 30분봉 백테스트")
    print("✅ 피드백 반영: 신호 발생 후 [다음 봉 시가(Open)] 체결 적용")
    print("✅ 피드백 반영: QQQ 200EMA 상회 시에만 신규 롱 진입 허용")
    print("==========================================================")
    
    universe = ["SOXL", "SOXS", "TQQQ", "SQQQ", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AVGO", "LLY", "AMD", "QLD", "USO"]
    leverage_tickers = ["SOXL", "SOXS", "TQQQ", "SQQQ"]
    
    # 시장 필터용 QQQ 포함 다운로드
    download_list = universe + ["QQQ"]
    print(" -> 실제 시장 30분봉 데이터 다운로드 중 (최근 60일)...")
    raw_data = yf.download(download_list, period="60d", interval="30m", progress=False)
    
    # 1. QQQ 시장 필터 계산 (200 EMA)
    qqq_close = raw_data['Close']['QQQ'].dropna()
    qqq_ema200 = qqq_close.ewm(span=200, adjust=False).mean()
    
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
    positions = {} 
    
    FEE_PENALTY = 0.0045 
    equity_history = []
    trade_logs = []
    
    # 시차 체결을 위한 변수 (이전 봉에서 발생한 주문을 기억)
    pending_orders = [] # [{'ticker': t, 'type': 'BUY'/'SELL', 'qty': q}]

    for idx, date in enumerate(all_dates):
        # ────────────── [현실 세계 주문 체결 단계] ──────────────
        # 대기 중인 주문이 있다면, 현재 봉의 '시가(Open)'로 체결시킴 (Look-Ahead Bias 제거)
        current_total_assets = cash + sum([p['qty'] * data_dict[t].loc[date, 'Open'] for t, p in positions.items()])
        
        for order in pending_orders:
            t = order['ticker']
            o_open = data_dict[t].loc[date, 'Open']
            
            if order['type'] == 'SELL' and t in positions:
                cash += positions[t]['qty'] * o_open * (1 - FEE_PENALTY)
                trade_logs.append((o_open - positions[t]['entry_price']) / positions[t]['entry_price'])
                del positions[t]
                
            elif order['type'] == 'BUY' and t not in positions and len(positions) < 4:
                target_amount = current_total_assets * 0.25 * 0.995
                qty = int(target_amount // o_open)
                cost = qty * o_open * (1 + FEE_PENALTY)
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t] = {'qty': qty, 'entry_price': o_open}
                    
        pending_orders = [] # 체결 완료 후 대기열 비우기
        
        # ────────────── [자산 평가액 기록] ──────────────
        # 현재 봉의 종가 기준으로 자산 가치 기록
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        equity_history.append(cash + current_pos_val)
        
        # 만약 마지막 봉이라면 미청산 포지션 강제 청산 처리 후 종료
        if idx == len(all_dates) - 1:
            for t in list(positions.keys()):
                c_close = data_dict[t].loc[date, 'Close']
                cash += positions[t]['qty'] * c_close * (1 - FEE_PENALTY)
                trade_logs.append((c_close - positions[t]['entry_price']) / positions[t]['entry_price'])
                del positions[t]
            equity_history[-1] = cash
            break

        # ────────────── [현재 봉 종가 기준 신호 탐색 단계] ──────────────
        # 1. 청산 신호 탐색
        for t in list(positions.keys()):
            if data_dict[t].loc[date, 'Trend'] == -1:
                pending_orders.append({'ticker': t, 'type': 'SELL'})
                
        # 2. 진입 신호 탐색 (시장 필터 결합)
        # 현재 시점 QQQ 종가가 QQQ 200EMA 위에 있을 때만 신규 롱 진입 주문 승인
        is_market_bull = qqq_close.loc[date] > qqq_ema200.loc[date]
        
        if is_market_bull:
            for ticker in universe:
                if ticker in positions or any(o['ticker'] == ticker for o in pending_orders): continue
                
                df = data_dict[ticker]
                b_idx = df.index.get_loc(date)
                if b_idx < 1: continue
                
                if df['Trend'].iloc[b_idx] == 1 and df['Trend'].iloc[b_idx-1] == -1:
                    pending_orders.append({'ticker': ticker, 'type': 'BUY'})

    # ────────────── [지표 연산 (교정 완료)] ──────────────
    equity_series = pd.Series(equity_history, index=all_dates)
    final_return = ((equity_series.iloc[-1] - initial_cash) / initial_cash) * 100
    
    roll_max = equity_series.cummax()
    drawdowns = (equity_series - roll_max) / roll_max
    mdd = drawdowns.min() * 100
    
    # 샤프 지수 계산 교정: 일별 데이터로 리샘플링 후 계산
    daily_equity = equity_series.resample('D').last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0
        
    wins = [r for r in trade_logs if r > 0]
    losses = [r for r in trade_logs if r <= 0]
    win_rate = (len(wins) / len(trade_logs) * 100) if trade_logs else 0.0
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    profit_loss_ratio = (avg_win / avg_loss) if avg_loss != 0 else 0.0

    print("\n==========================================================")
    print("      🎯 [완료] 초정밀 실전형 Ver 5.1 최종 성적표")
    print("==========================================================")
    print(f" 🟩 [누적 수익률]      : {final_return:+.2f}%")
    print(f" 🟥 [최대 낙폭 (MDD)]   : {mdd:.2f}%")
    print(f" 📊 [샤프 지수 (Sharpe)]: {sharpe_ratio:.2f}")
    print(f" ⚖️ [알고리즘 승률]    : {win_rate:.2f}% (총 {len(trade_logs)}회 거래)")
    print(f" 💰 [평균 손익비]      : {profit_loss_ratio:.2f} (평균익절: {avg_win*100:+.2f}% / 평균손절: {avg_loss*100:.2f}%)")
    print("==========================================================\n")

if __name__ == "__main__":
    run_perfect_30m_backtest()
    gc.collect()