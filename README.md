# SuperTrendQuant

SuperTrend 기반 전략 연구, 백테스트, 모의투자와 실거래를 하나의 전략
엔진으로 실행하는 주식 자동매매 프로젝트입니다.

현재 기준 구현은 [`unified_quant`](unified_quant/README.md)입니다. 기존
`jo_factory/`와 `module/`은 통합 전 구현을 비교하기 위한 레거시 소스로만
남아 있으며, 패키징과 CLI는 `unified_quant/src/supertrend_quant`를 사용합니다.

```bash
uv sync
uv run quant-backtest --help
uv run quant-search --help
uv run quant-optimize --help
uv run quant-paper --help
uv run quant-live --help
```

설정 예시와 안전한 실행 순서는
[`unified_quant/README.md`](unified_quant/README.md)를 참고하세요.
