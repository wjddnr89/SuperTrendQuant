import gc
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# ... [calculate_supertrend 함수는 동일하므로 생략] ...

def run_analytics_backtest():
    # ... [데이터 준비 및 로직 동일] ...
    # (위에서 작성하신 로직을 그대로 사용하되, 마지막에 아래 코드를 함수 안에 포함하세요)

    # ────────────── 📊 심층 리포트 및 벤치마크 연산 ──────────────
    # ... [진단 결과 1, 2, 3 로직] ...
    
    # 💡 [진단 결과 4] 벤치마크 계산을 함수 안으로 이동!
    benchmark_returns = {}
    for t in full_universe:
        start_price = data_dict[t]['Close'].iloc[0]
        end_price = data_dict[t]['Close'].iloc[-1]
        benchmark_returns[t] = (end_price - start_price) / start_price * 100

    avg_benchmark_return = np.mean(list(benchmark_returns.values()))
    
    # 여기서 final_return은 앞선 루프에서 계산된 수익률 변수입니다.
    final_return = ((equity_history[-1] - initial_cash) / initial_cash) * 100

    print("\n" + "="*58)
    print("      🔍 [진단 결과 4] 벤치마크 비교 (단순 보유 vs 전략)")
    print("="*58)
    print(f" 📈 15개 종목 단순 보유 평균 수익률 : {avg_benchmark_return:+.2f}%")
    print(f" 🤖 우리 전략의 최종 누적 수익률   : {final_return:+.2f}%")
    print(f" 🏆 알파(Alpha, 초과 수익)        : {final_return - avg_benchmark_return:+.2f}%")
    print("="*58 + "\n")

if __name__ == "__main__":
    run_analytics_backtest()
    gc.collect()