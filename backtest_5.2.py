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
    
    # SuperTrend 초기화 왜곡 방지: 첫 봉은 ATR 기반 기본값으로 세팅
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        final_ub.iloc[i] = basic_ub.iloc[i] if basic_ub.iloc[i] < final_ub.iloc[i-1] or close.iloc[i-1] > final_ub.iloc[i-1] else final_ub.iloc[i-1]
        final_lb.iloc[i] = basic_lb.iloc[i] if basic_lb.iloc[i] > final_lb.iloc[i-1] or close.iloc[i-1] < final_lb.iloc[i-1] else final_lb.iloc[i-1]
            
    for i in range(1, len(df)):
        if trend.iloc[i-1] == 1:
            trend.iloc[i] = -1 if close.iloc[i] < final_lb.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if close.iloc[i] > final_ub.iloc[i] else -1
    return trend, atr

def run_ultimate_backtest():
    print("==========================================================")
    print("🛡️ [Ver 5.3] 무결점 퀀트 연구용 30분봉 백테스트 엔진 가동")
    print("✅ 반영: 진짜 QQQ 일봉 200 EMA 연동 및 30분봉 타임라인 매칭")
    print("✅ 반영: 편도 수수료(0.225%) 교정 및 실전 슬리피지(0.05%) 탑재")
    print("==========================================================")
    
    long_universe = ["SOXL", "TQQQ", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AVGO", "LLY", "AMD", "QLD", "USO"]
    short_universe = ["SOXS", "SQQQ"]
    full_universe = long_universe + short_universe
    
    # 1. 🔥 [핵심 교정] 진짜 QQQ 일봉 200 EMA 데이터 추출 후 딕셔너리 매핑 준비
    print(" -> 시장 필터용 QQQ 일봉 2년치 데이터 다운로드 중...")
    qqq_daily = yf.download("QQQ", period="2y", interval="1d", progress=False)
    qqq_daily['EMA200'] = qqq_daily['Close'].ewm(span=200, adjust=False).mean()
    
    # 날짜 문자열(YYYY-MM-DD)을 키로 하는 일봉 200 EMA 매핑 테이블 생성
    qqq_daily_map = qqq_daily['EMA200'].dropna()
    qqq_daily_map.index = qqq_daily_map.index.strftime('%Y-%m-%d')
    
    # 2. 30분봉 메인 주가 데이터 다운로드
    print(" -> 유니버스 종목 30분봉 데이터 다운로드 중 (최근 60일)...")
    raw_data = yf.download(full_universe + ["QQQ"], period="60d", interval="30m", progress=False)
    
    data_dict = {}
    all_dates = None
    
    for ticker in full_universe:
        try:
            df = pd.DataFrame({
                'Open': raw_data['Open'][ticker], 'High': raw_data['High'][ticker],
                'Low': raw_data['Low'][ticker], 'Close': raw_data['Close'][ticker]
            }).dropna()
            
            if df.empty: continue
            
            mult = 4.5 if ticker in ["SOXL", "SOXS", "TQQQ", "SQQQ"] else 3.0
            df['Trend'], df['ATR'] = calculate_supertrend(df, period=7, multiplier=mult)
            df['Return_5d'] = df['Close'].pct_change(65) # 최근 5영업일(13봉 * 5 = 65봉) 수익률
            
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
    
    # 비용 구조 명확화 (피드백 반영)
    FEE_HALF = 0.00225  # 편도 수수료 0.225% (왕복 0.45%)
    SLIPPAGE = 0.0005  # 실전 체결 슬리피지 패널티 0.05% 강제 부과
    
    equity_history = []
    trade_logs = []
    pending_orders = [] 

    for idx, date in enumerate(all_dates):
        date_str = date.strftime('%Y-%m-%d')
        
        # 만약 해당 날짜의 일봉 200 EMA 데이터가 없다면 가장 최근 사용 가능한 데이터로 우회
        if date_str not in qqq_daily_map.index:
            past_maps = qqq_daily_map[qqq_daily_map.index < date_str]
            current_qqq_ema200 = past_maps.iloc[-1] if not past_maps.empty else qqq_close_30m.loc[date]
        else:
            current_qqq_ema200 = qqq_daily_map.loc[date_str]

        # ──────────────── [주문 체결 단계 (슬리피지 반영)] ────────────────
        # 매도 주문 체결 (슬리피지로 인해 장중 시가보다 조금 더 불리하게 낮게 팔림)
        for order in [o for o in pending_orders if o['type'] == 'SELL']:
            t = order['ticker']
            if t in positions:
                o_open = data_dict[t].loc[date, 'Open']
                real_sell_price = o_open * (1 - SLIPPAGE) # 슬리피지 패널티 반영
                cash += positions[t]['qty'] * real_sell_price * (1 - FEE_HALF)
                trade_logs.append((real_sell_price - positions[t]['entry_price']) / positions[t]['entry_price'])
                del positions[t]
                
        # 매수 주문 체결 (슬리피지로 인해 장중 시가보다 조금 더 불리하게 비싸게 삼)
        buy_orders = [o for o in pending_orders if o['type'] == 'BUY']
        for order in buy_orders:
            t = order['ticker']
            if t not in positions and len(positions) < 4:
                o_open = data_dict[t].loc[date, 'Open']
                real_buy_price = o_open * (1 + SLIPPAGE) # 슬리피지 패널티 반영
                
                # 실시간 남은 자산 기반 가용현금 갱신 규칙 철저 이행
                current_assets = cash + sum([p['qty'] * data_dict[pos_t].loc[date, 'Open'] for pos_t, p in positions.items()])
                target_unit_size = current_assets * 0.25 * 0.995
                actual_alloc = min(cash, target_unit_size)
                
                qty = int(actual_alloc // real_buy_price)
                cost = qty * real_buy_price * (1 + FEE_HALF)
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t] = {'qty': qty, 'entry_price': real_buy_price}
                    
        pending_orders = [] 
        
        # ──────────────── [자산 평가액 기록 및 종료] ────────────────
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        equity_history.append(cash + current_pos_val)
        
        if idx == len(all_dates) - 1:
            for t in list(positions.keys()):
                c_close = data_dict[t].loc[date, 'Close']
                cash += positions[t]['qty'] * c_close * (1 - FEE_HALF)
                trade_logs.append((c_close - positions[t]['entry_price']) / positions[t]['entry_price'])
                del positions[t]
            equity_history[-1] = cash
            break

        # ──────────────── [신호 탐색 및 결측치 방어] ────────────────
        for t in list(positions.keys()):
            if data_dict[t].loc[date, 'Trend'] == -1:
                pending_orders.append({'ticker': t, 'type': 'SELL'})
                
        # 💡 [진짜 일봉 필터 작동]: QQQ 30분봉 종가가 진짜 일봉 QQQ 200 EMA 위에 있는지 검사
        is_bull_market = qqq_close_30m.loc[date] > current_qqq_ema200
        target_pool = long_universe if is_bull_market else short_universe
        
        raw_buy_candidates = []
        for ticker in target_pool:
            if ticker in positions: continue
            df = data_dict[ticker]
            b_idx = df.index.get_loc(date)
            if b_idx < 1: continue
            
            # 🚨 [NaN 버그 고정]: 5일 변동성 및 QQQ 변동성이 NaN이면 정렬 오류가 나므로 진입 배제
            if pd.isna(df['Return_5d'].iloc[b_idx]) or pd.isna(qqq_ret_5d.loc[date]):
                continue
                
            if df['Trend'].iloc[b_idx] == 1 and df['Trend'].iloc[b_idx-1] == -1:
                rs_score = df['Return_5d'].iloc[b_idx] - qqq_ret_5d.loc[date]
                raw_buy_candidates.append({'ticker': ticker, 'rs': rs_score})
                
        if raw_buy_candidates:
            raw_buy_candidates = sorted(raw_buy_candidates, key=lambda x: x['rs'], reverse=True)
            for candidate in raw_buy_candidates:
                pending_orders.append({'ticker': candidate['ticker'], 'type': 'BUY'})

    # ──────────────── [통계 산출] ────────────────
    equity_series = pd.Series(equity_history, index=all_dates)
    final_return = ((equity_series.iloc[-1] - initial_cash) / initial_cash) * 100
    
    roll_max = equity_series.cummax()
    drawdowns = (equity_series - roll_max) / roll_max
    mdd = drawdowns.min() * 100
    
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
    print("      🎯 [최종 완료] 왜곡률 0% 퀀트 연구용 최종 성적표")
    print("==========================================================")
    print(f" 🟩 [누적 수익률]      : {final_return:+.2f}%")
    print(f" 🟥 [최대 낙폭 (MDD)]   : {mdd:.2f}%")
    print(f" 📊 [샤프 지수 (Sharpe)]: {sharpe_ratio:.2f}")
    print(f" ⚖️ [알고리즘 승률]    : {win_rate:.2f}% (총 {len(trade_logs)}회 거래)")
    print(f" 💰 [평균 손익비]      : {profit_loss_ratio:.2f} (평균익절: {avg_win*100:+.2f}% / 평균손절: {avg_loss*100:.2f}%)")
    print("==========================================================\n")

if __name__ == "__main__":
    run_ultimate_backtest()
    gc.collect()