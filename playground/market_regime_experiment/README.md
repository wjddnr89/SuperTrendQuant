# Market regime experiment

현재 A 계좌의 QQQ 진입 허용 필터를 보유 포지션 위험관리로 확장하는 독립
playground 실험입니다. `unified_quant`는 수정하지 않습니다.

- M0: 현재 A
- M1: QQQ Supertrend 하락 즉시 전량 청산
- M2: 하락 3봉 확인 후 전량 청산, 상승 2봉 확인 후 재진입
- M3: 하락 전환 시 50% 리밸런싱, 상승 전환 시 100% 복원
- M4: QQQ Supertrend + EMA200 강세/중립/약세 3단계
- M5: M4에 약세 2봉, 강세 3봉 확인 추가

```powershell
.\.venv\Scripts\python.exe playground\market_regime_experiment\test_regime_overlay.py
.\.venv\Scripts\python.exe playground\market_regime_experiment\run_experiment.py
```

