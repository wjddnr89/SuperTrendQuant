import gc
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# 터미널 인코딩 깨짐 방지
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# ==========================================
# 1. 기술 지표 연산 엔진
# ==========================================
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
    adx = dx.rolling(window=window).mean()
    return adx

# ==========================================
# 2. 지옥의 백테스트 시뮬레이터 Engine
# ==========================================
def run_hell_backtest():
    print("==========================================================")
    print("🔥 [명세서 V1.1 규격] 2022 대폭락장 관통 복리 백테스트 가동")
    print("🔥 매매 패널티(수수료+슬리피지): 왕복 0.45% 강제 차감")
    print("==========================================================")
    
    universe = ["SOXL", "SOXS", "TQQQ", "SQQQ", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AVGO", "LLY", "AMD", "QLD", "USO"]
    leverage_tickers = ["SOXL", "SOXS", "TQQQ", "SQQQ"]
    
    # 데이터 다운로드 (2022년 폭락장 전체 포함 필수 조건 충족)
    start_date = "2022-01-01"
    end_date = "2026-06-01"
    
    print(f" -> 15종목 유니버스 데이터 다운로드 중 ({start_date} ~ {end_date})...")
    raw_data = yf.download(universe, start=start_date, end=end_date, progress=False)
    
    # 개별 종목 지표 가공
    data_dict = {}
    all_dates = None
    
    for ticker in universe:
        try:
            df = pd.DataFrame({
                'Open': raw_data['Open'][ticker],
                'High': raw_data['High'][ticker],
                'Low': raw_data['Low'][ticker],
                'Close': raw_data['Close'][ticker],
                'Volume': raw_data['Volume'][ticker]
            }).dropna()
            
            if df.empty: continue
            
            mult = 4.5 if ticker in leverage_tickers else 3.0
            df['Trend'], df['ATR'] = calculate_supertrend(df, period=7, multiplier=mult)
            df['ADX'] = calculate_adx(df, window=14)
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            
            df = df.dropna()
            data_dict[ticker] = df
            
            if all_dates is None:
                all_dates = df.index
            else:
                all_dates = all_dates.intersection(df.index)
        except Exception as e:
            continue

    all_dates = sorted(list(all_dates))
    
    # 자금 및 포지션 변수 초기화
    initial_cash = 10000.0
    cash = initial_cash
    positions = {} # {ticker: {'qty': 수량, 'entry_price': 진입가, 'atr': 진입시ATR, 'half_exit': 플래그, 'is_leverage': 여부, 'adx': 진입시ADX}}
    
    # 리스크 관리 파라미터
    FEE_PENALTY = 0.0045 # 왕복 0.45%
    
    # 성과 지표 기록용 변수
    equity_history = []
    trade_logs = [] # 각 매매의 손익률 기록용
    
    peak_assets = initial_cash
    emergency_mode = False
    
    # --- 타임 시뮬레이션 루프 (시간 여행 시작) ---
    for date in all_dates:
        # 1. 실시간 가격 업데이트 및 청산/리스크 관리 감시
        current_prices = {}
        for t in list(positions.keys()):
            current_prices[t] = data_dict[t].loc[date, 'Close']
            
            pos = positions[t]
            c_price = current_prices[t]
            entry_p = pos['entry_price']
            atr = pos['atr']
            trend = data_dict[t].loc[date, 'Trend']
            
            # 리스크 1: ATR 가변 6배수 1차 분할 익절 (+50% 물량)
            target_tp = entry_p + (atr * 6)
            if c_price >= target_tp and not pos['half_exit']:
                half_qty = pos['qty'] // 2
                if half_qty > 0:
                    realized_revenue = half_qty * c_price * (1 - FEE_PENALTY) # 패널티 차감
                    cash += realized_revenue
                    positions[t]['qty'] -= half_qty
                    positions[t]['half_exit'] = True
                    trade_logs.append((c_price - entry_p) / entry_p) # 수익률 기록
            
            # 리스크 2: Trailing Stop (본전 하향 돌파) 혹은 SuperTrend 매도 신호시 전량 청산
            if c_price < entry_p or trend == -1:
                final_revenue = positions[t]['qty'] * c_price * (1 - FEE_PENALTY) # 패널티 차감
                cash += final_revenue
                p_rtn = (c_price - entry_p) / entry_p
                trade_logs.append(p_rtn)
                del positions[t]
                
        # 2. 총 자산 평가 및 하방 가드 브레이크 연산
        current_pos_val = sum([d['qty'] * data_dict[t].loc[date, 'Close'] for t, d in positions.items()])
        total_assets = cash + current_pos_val
        equity_history.append(total_assets)
        
        if total_assets > peak_assets:
            peak_assets = total_assets
            
        drawdown = (total_assets - peak_assets) / peak_assets
        if drawdown <= -0.15:
            emergency_mode = True
        if emergency_mode and drawdown >= -0.05:
            emergency_mode = False
            
        # 3. 신규 매수 시그널 탐색 및 진입 후보 정렬
        buy_candidates = []
        for ticker in universe:
            if ticker in positions: continue
            if date not in data_dict[ticker].index: continue
            
            df = data_dict[ticker]
            idx = df.index.get_loc(date)
            if idx < 2: continue
            
            # 지표 추출
            c_trend = df['Trend'].iloc[idx]
            # 명세서 2번: '직전 최종 마감된 확정 봉' 기준 동조화 모사
            p_trend = df['Trend'].iloc[idx-1] 
            
            c_vol = df['Volume'].iloc[idx]
            p_vol_ma20 = df['Vol_MA20'].iloc[idx-1]
            c_adx = df['ADX'].iloc[idx]
            c_close = df['Close'].iloc[idx]
            c_atr = df['ATR'].iloc[idx]
            
            # 3대 필터 결합 (SuperTrend 동조 & 거래량 이평 돌파 & ADX 20 이상)
            if c_trend == 1 and p_trend == 1 and (c_vol > p_vol_ma20) and (c_adx >= 20):
                buy_candidates.append({
                    'ticker': ticker, 'price': c_close, 'atr': c_atr, 'adx': c_adx, 'is_leverage': ticker in leverage_tickers
                })
                
        # 복수 신호 발생 시 ADX 최고점 1개 종목만 최종 선택 (명세서 3번 조항)
        if buy_candidates:
            buy_candidates = sorted(buy_candidates, key=lambda x: x['adx'], reverse=True)
            best = buy_candidates[0]
            t_ticker = best['ticker']
            
            # 슬롯 계산 및 가변 비중
            is_lev = best['is_leverage']
            base_pct = 0.125 if is_lev else 0.25
            if emergency_mode: base_pct *= 0.5
            
            target_amount = total_assets * base_pct * 0.995 # 99.5% 안전마진
            qty = int(target_amount // best['price'])
            cost = qty * best['price'] * (1 + FEE_PENALTY) # 진입 시 수수료 가산
            
            # 슬롯 한도 계산
            lev_slots = sum([1 for t, d in positions.items() if d['is_leverage']])
            norm_slots = sum([1 for t, d in positions.items() if not d['is_leverage']])
            
            # 슬롯 여유가 있을 때 신규 진입
            if (is_lev and lev_slots < 8) or (not is_lev and norm_slots < 4):
                if qty > 0 and cash >= cost:
                    cash -= cost
                    positions[t_ticker] = {
                        'qty': qty, 'entry_price': best['price'], 'atr': best['atr'], 'half_exit': False, 'is_leverage': is_lev, 'adx': best['adx']
                    }
            # 슬롯 한도가 찼을 때 [ADX 5.0 버퍼 적자생존 교체매매] 발동
            else:
                same_group = {t: d for t, d in positions.items() if d['is_leverage'] == is_lev}
                if same_group:
                    weakest = min(same_group, key=lambda x: same_group[x]['adx'])
                    if best['adx'] > (positions[weakest]['adx'] + 5.0):
                        # 최약체 강제 청산
                        w_qty = positions[weakest]['qty']
                        w_close = data_dict[weakest].loc[date, 'Close']
                        cash += w_qty * w_close * (1 - FEE_PENALTY)
                        del positions[weakest]
                        
                        # 신규 유망주 진입
                        if cash >= cost and qty > 0:
                            cash -= cost
                            positions[t_ticker] = {
                                'qty': qty, 'entry_price': best['price'], 'atr': best['atr'], 'half_exit': False, 'is_leverage': is_lev, 'adx': best['adx']
                            }

    # ==========================================
    # 3. 5대 핵심 성과 지표(Metric) 정밀 연산
    # ==========================================
    equity_series = pd.Series(equity_history, index=all_dates)
    
    # 1) 누적 수익률 (Cumulative Return)
    final_return = ((equity_series.iloc[-1] - initial_cash) / initial_cash) * 100
    
    # 2) MDD (Maximum Drawdown)
    roll_max = equity_series.cummax()
    drawdowns = (equity_series - roll_max) / roll_max
    mdd = drawdowns.min() * 100
    
    # 3) 샤프 지수 (Sharpe Ratio - 일일 변동성 기준 연율화, 무위험수익률=0 가정)
    daily_returns = equity_series.pct_change().dropna()
    if daily_returns.std() != 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
        
    # 4) 승률 (Win Rate)
    wins = [r for r in trade_logs if r > 0]
    win_rate = (len(wins) / len(trade_logs) * 100) if trade_logs else 0.0
    
    # 5) 손익비 (Profit Factor / Profit to Loss Ratio)
    losses = [r for r in trade_logs if r <= 0]
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    profit_loss_ratio = (avg_win / avg_loss) if avg_loss != 0 else 0.0

    # 결과 표출
    print("\n==========================================================")
    print("      🎯 지옥의 대폭락장(2022~현재) 최종 백테스트 스코어보드")
    print("==========================================================")
    print(f" 🟩 [누적 수익률]      : {final_return:+.2f}%")
    print(f" 🟥 [최대 낙폭 (MDD)]   : {mdd:.2f}%")
    print(f" 📊 [샤프 지수 (Sharpe)]: {sharpe_ratio:.2f}")
    print(f" ⚖️ [알고리즘 승률]    : {win_rate:.2f}% (총 {len(trade_logs)}회 거래)")
    print(f" 💰 [평균 손익비]      : {profit_loss_ratio:.2f} (평균익절: {avg_win*100:+.2f}% / 평균손절: {avg_loss*100:.2f}%)")
    print("==========================================================\n")

if __name__ == "__main__":
    run_hell_backtest()
    gc.collect()