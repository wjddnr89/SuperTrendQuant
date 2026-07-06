import gc
import sys
import numpy as np
import pandas as pd
import yfinance as yf

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

def run_buy_and_hold_benchmark():
    print("==========================================================")
    print("📊 [벤치마크] 질문자님 지정 유니버스 2달 단순 보유(Buy & Hold) 엔진 가동")
    print("✅ 조건: 시작일 시가(Open)에 14개 종목 균등 분할 매수 (각 비중 7.14%)")
    print("✅ 수수료: 진입 시 편도 0.225% / 슬리피지 없음 (단순 보유 표준)")
    print("==========================================================")
    
    # 질문자님의 14개 지정 종목
    full_universe = [
        "SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", 
        "RKLB", "PL", "RDW", "OUST", "TSLA", "AEHR", "AXON", "SOXS"
    ]
    
    print(" -> 유니버스 30분봉 데이터 다운로드 및 정렬 중...")
    raw_data = yf.download(full_universe, period="60d", interval="30m", progress=False)
    
    data_dict = {}
    all_dates = None
    
    for ticker in full_universe:
        try:
            df = pd.DataFrame({
                'Open': raw_data['Open'][ticker],
                'Close': raw_data['Close'][ticker]
            }).dropna()
            if df.empty: continue
            data_dict[ticker] = df
            all_dates = df.index if all_dates is None else all_dates.intersection(df.index)
        except Exception:
            continue
            
    all_dates = sorted(list(all_dates))
    
    initial_cash = 10000.0
    num_assets = len(data_dict)
    cash_per_asset = initial_cash / num_assets
    FEE_HALF = 0.00225
    
    # 1. 첫 번째 봉의 시가(Open)에 전 종목 매입 후 고정
    first_date = all_dates[0]
    shares = {}
    total_initial_cost = 0.0
    
    individual_returns = {}
    
    for ticker, df in data_dict.items():
        open_price = df.loc[first_date, 'Open']
        # 수수료를 감안한 구매 가능 수량 계산
        qty = (cash_per_asset / (1 + FEE_HALF)) / open_price
        shares[ticker] = qty
        total_initial_cost += qty * open_price * (1 + FEE_HALF)
        
    # 남은 잔돈 현금
    remaining_cash = initial_cash - total_initial_cost
    equity_history = []
    
    # 2. 매 30분봉마다 포트폴리오 가치 합산 (실시간 MDD 측정용)
    for date in all_dates:
        portfolio_value = remaining_cash
        for ticker, qty in shares.items():
            portfolio_value += qty * data_dict[ticker].loc[date, 'Close']
        equity_history.append(portfolio_value)
        
    # 3. 마지막 봉의 종가 기준으로 개별 종목 최종 수익률 확정
    last_date = all_dates[-1]
    for ticker, qty in shares.items():
        init_p = data_dict[ticker].loc[first_date, 'Open']
        last_p = data_dict[ticker].loc[last_date, 'Close']
        # 단순 보유 수익률 (수수료 미반영 순수 가격 변동률)
        individual_returns[ticker] = ((last_p - init_p) / init_p) * 100

    # ────────────── 📊 단순 보유 통계 연산 ──────────────
    equity_series = pd.Series(equity_history, index=all_dates)
    final_return = ((equity_series.iloc[-1] - initial_cash) / initial_cash) * 100
    roll_max = equity_series.cummax()
    mdd = ((equity_series - roll_max) / roll_max).min() * 100
    
    return_values = list(individual_returns.values())
    avg_return = np.mean(return_values)
    median_return = np.median(return_values)
    
    print("\n==========================================================")
    print(" 📊 [최종 성적표] 14개 주도주 단순 보유(Buy & Hold) 결과")
    print("==========================================================")
    print(f" 🟩 [포트폴리오 누적 수익률]: {final_return:+.2f}%")
    print(f" 🟥 [포트폴리오 최대 낙폭]  : {mdd:.2f}%")
    print("-" * 58)
    print(f" 📈 [개별 종목 단순 평균 수익률] : {avg_return:+.2f}%")
    print(f" 📉 [개별 종목 중앙값 수익률]   : {median_return:+.2f}%")
    print("==========================================================\n")
    
    print("==========================================================")
    print(" 🔍 [상세 내역] 각 종목별 2달간 순수 상승률 (시가->종가)")
    print("==========================================================")
    for ticker in sorted(individual_returns, key=individual_returns.get, reverse=True):
        print(f"  * {ticker:<6} : {individual_returns[ticker]:+.2f}%")
    print("==========================================================\n")

if __name__ == "__main__":
    run_buy_and_hold_benchmark()
    gc.collect()