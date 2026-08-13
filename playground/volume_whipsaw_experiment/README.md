# Volume confirmation experiment

`unified_quant`를 수정하지 않고 canonical Nasdaq-100 best 전략의 신규 진입 후보에만
거래량 확인 조건을 추가하는 연구용 오버레이입니다.

비교 변형:

- `BASE`: canonical best (playground A 가상계좌와 같은 전략 파라미터)
- `RVOL20_GE_1.0`, `RVOL20_GE_1.2`, `RVOL20_GE_1.5`
- `CMF20_GT_0`
- `OBV_SLOPE10_GT_0`
- Calmar 기준 최선 RVOL + CMF20
- Calmar 기준 최선 RVOL + OBV slope

지표는 완성된 일봉까지만 사용하며 주문은 canonical runner처럼 다음 세션 시가에
체결됩니다. RVOL20의 분모는 현재 거래량을 제외한 직전 20세션 평균입니다.
과거 구성종목의 단일 세션 가격 누락은 직전 종가로 평가만 하며, 실제 주문 체결은
여전히 해당 세션의 정확한 가격 바가 있어야 가능합니다.

```powershell
.\.venv\Scripts\python.exe playground\volume_whipsaw_experiment\test_volume_overlay.py
.\.venv\Scripts\python.exe playground\volume_whipsaw_experiment\run_experiment.py
.\.venv\Scripts\python.exe playground\volume_whipsaw_experiment\run_experiment.py `
  --market-filter none `
  --results-dir playground\volume_whipsaw_experiment\results\account_c_no_market_filter
.\.venv\Scripts\python.exe playground\volume_whipsaw_experiment\run_experiment.py `
  --confirmation-window 3 `
  --results-dir playground\volume_whipsaw_experiment\results\account_a_sticky3
```

결과는 기본적으로 `results/canonical_best_2015_2026/` 아래에 저장됩니다.
