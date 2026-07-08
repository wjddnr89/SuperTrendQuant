# Split Config Guide

새 전략은 이 폴더에 추가한다. 전략 파일은 매매 로직만 담고, 시장/유니버스/실행환경은 runtime에서 고른다.

```bash
uv run quant-backtest \
  --strategy configs/strategies/leader_rotation.yaml \
  --runtime configs/runtimes/simulation.yaml

uv run quant-paper \
  --strategy configs/strategies/leader_rotation.yaml \
  --runtime configs/runtimes/simulation.yaml \
  --state state/leader_rotation.paper.json

uv run quant-live \
  --strategy configs/strategies/main_jo_leader_rotation.yaml \
  --runtime configs/runtimes/live_toss.yaml
```

`quant-paper`와 `quant-live`는 기본이 loop다. 장 시간에 계속 돌리며, 한 번만 점검하려면 `--once`를 붙인다.

## 역할 분리

- `strategies/`: Supertrend, RS, 필터, exit 확인봉, rotation 규칙
- `runtimes/`: market, universe 파일, 명시 종목 목록, 실행 기간, 수수료, 주문 브로커, loop 설정

## 새 전략 추가 규칙

1. `configs/strategies/leader_rotation.yaml` 또는 `simple_supertrend.yaml`을 복사한다.
2. `name`만 새 이름으로 바꾼다.
3. `signals.entries`, `signals.filters`, `signals.exits`에서 지원되는 키만 조정한다. 지원되지 않는 키는 로딩 단계에서 에러가 난다.
4. 같은 전략을 backtest, paper, live로 옮길 때는 전략 파일은 그대로 두고 `--runtime`만 바꾼다.

simulation runtime의 `market`을 `US` 또는 `KR`로 바꾸면 해당 시장 universe를 사용한다. live runtime은 `AUTO`가 기본이며 열린 장에 맞는 universe만 사용한다.

runtime 파일을 수정하지 않고 임시로 바꿀 수도 있다.

```bash
uv run quant-backtest \
  --strategy configs/strategies/leader_rotation.yaml \
  --runtime configs/runtimes/simulation.yaml \
  --market KR

uv run quant-backtest \
  --strategy configs/strategies/leader_rotation.yaml \
  --runtime configs/runtimes/simulation.yaml \
  --market US \
  --symbols SOXL,TQQQ,NVDA
```

benchmark는 config에서 고르지 않는다. 엔진이 종목별로 자동 매핑한다.

- US: `QQQ`
- KR KOSPI: `^KS11`
- KR KOSDAQ: `^KQ11`

현재 엔진이 실제로 해석하는 component는 아래다.

- `supertrend`: `enabled`, `period`, `multiplier`, `atr_method`, `symbol_multipliers`
- `benchmark_trend`: `enabled`, `timeframe`
- `relative_strength`: `lookback_bars` 또는 시장별 `lookback_bars.US`, `lookback_bars.KR`, `lookback_bars.default`
- `supertrend_flip`: `confirm_bars`

config에 적힌 값은 무시되지 않는다. 아직 엔진이 지원하지 않는 component나 key는 조용히 넘어가지 않고 `ValueError`로 실패한다.

## backtest, paper 결과 저장

`quant-backtest`는 기본적으로 `results/backtests/<run_id>/`에 저장한다.

- `summary.json`: 설정, 지표, skipped symbols
- `equity.csv`: 시점별 equity

`quant-paper`는 `results/paper/<run_id>/`에 저장하고, 계좌 상태는 별도 state 파일에 이어서 저장한다.

- `metadata.json`: 실행 설정
- `cycles.jsonl`: cycle별 주문계획, 체결, 계좌 snapshot
- `equity.csv`: cycle별 equity/cash/positions value
- `state/*.json`: paper 계좌 현금/보유종목/마지막 처리 봉

비교:

```bash
uv run quant-compare \
  --paper-dir results/paper/<paper_run_id> \
  --backtest-dir results/backtests/<backtest_run_id>
```
