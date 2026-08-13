# Portfolio risk experiment

현재 A 또는 C의 신호는 유지하고 투자 비중, 보유 종목 수, ATR 위험 비중과
계좌 Drawdown 브레이크만 비교합니다. C는 A와 같은 전략에서 QQQ 시장필터만
제거한 계좌입니다. `unified_quant`는 수정하지 않습니다.

- D0: 현재 A 또는 C
- D1: 75% 투자
- D2: 최대 2종목 동일 비중
- D3: 최대 3종목 동일 비중
- D4A/D4/D4B: 최대 1종목, 포트폴리오 ATR 위험 예산 2.0%/2.5%/3.0%
- D5: 최대 2종목, 포트폴리오 ATR 위험 예산 2.5%
- D6: D5 + 계좌 DD -15% 전량청산 + 20세션 휴식

```powershell
.\.venv\Scripts\python.exe playground\portfolio_risk_experiment\test_risk_overlay.py
.\.venv\Scripts\python.exe playground\portfolio_risk_experiment\run_experiment.py
.\.venv\Scripts\python.exe playground\portfolio_risk_experiment\run_experiment.py --account C --results-dir playground\portfolio_risk_experiment\results\account_c_no_market_filter_2015_2026
```
