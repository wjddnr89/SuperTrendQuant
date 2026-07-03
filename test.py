import yfinance as yf
import pandas as pd
import numpy as np

# 비교실험 대상 유니버스 (총 24개 종목)
FULL_UNIVERSE = [
    "SOXL", "TQQQ", "NVDA", "MU", "GEV", "VRT", "RKLB", "PL", "RDW", "OUST", "MRVL",
    "TSLA", "AEHR", "AXON", "SOXS", "LLY", "UNH", "MDT", "RZLV", "FN", "AMD", "COHR", "MP", "TSM"
]

def run_universe_comparison():
    print(f"🚀 {len(FULL_UNIVERSE)}개 종목 30분봉 데이터 다운로드 중 (최근 60일)...")
    
    # 30분봉으로 뽑을 수 있는 최대 기간인 60일치를 통째로 긁어옵니다.
    # group_by='ticker'를 주면 각 종목별로 데이터를 이쁘게 쪼개서 가져올 수 있습니다.
    try:
        raw_data = yf.download(FULL_UNIVERSE, period="60d", interval="30m", group_by='ticker', progress=False)
    except Exception as e:
        print(f"❌ 데이터 다운로드 실패: {e}")
        return

    results = []

    for ticker in FULL_UNIVERSE:
        try:
            # 해당 종목의 데이터만 추출 후 결측치 제거
            if ticker not in raw_data.columns.levels[0]:
                continue
                
            df_ticker = raw_data[ticker].dropna()
            if df_ticker.empty or len(df_ticker) < 10: 
                continue
            
            # 1차원 Series로 확정하기 위해 squeeze 적용
            close_series = df_ticker['Close'].squeeze()
            
            # 1. 누적 수익률 (Buy & Hold)
            start_price = float(close_series.iloc[0])
            end_price = float(close_series.iloc[-1])
            total_return = ((end_price / start_price) - 1) * 100
            
            # 2. MDD (최대 낙폭) 계산
            rolling_max = close_series.cummax()
            drawdown = (close_series - rolling_max) / rolling_max
            mdd = drawdown.min() * 100
            
            results.append({
                'Ticker': ticker,
                'Start_Price': f"${start_price:.2f}",
                'End_Price': f"${end_price:.2f}",
                'Return(%)': round(total_return, 2),
                'MDD(%)': round(mdd, 2)
            })
            
        except Exception as ticker_error:
            print(f"⚠️ {ticker} 연산 중 오류 발생 (데이터 부족 등): {ticker_error}")
            continue

    # 데이터프레임 변환 및 수익률 순 정렬
    if results:
        results_df = pd.DataFrame(results).sort_values(by='Return(%)', ascending=False)
        print("\n" + "="*60)
        print("🏆 유니버스 최근 2달(60영업일) 30분봉 단순 보유 실험 결과")
        print("="*60)
        print(results_df.to_string(index=False))
        print("="*60)
    else:
        print("❌ 분석 가능한 데이터가 존재하지 않습니다.")

if __name__ == "__main__":
    run_universe_comparison()