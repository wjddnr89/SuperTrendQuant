# Strategy and Runtime Guide

Commands in this guide run from the repository root.

## Modes

```bash
uv run quant-backtest \
  --strategy unified_quant/configs/strategies/leader_rotation.yaml \
  --runtime unified_quant/configs/runtimes/simulation.yaml

uv run quant-search \
  --strategy unified_quant/configs/strategies/triple_filters.yaml \
  --runtime unified_quant/configs/runtimes/research_us.yaml

uv run quant-optimize \
  --strategy unified_quant/configs/strategies/triple_filters.yaml \
  --runtime unified_quant/configs/runtimes/research_us.yaml \
  --n-trials 100

uv run quant-paper \
  --strategy unified_quant/configs/strategies/leader_rotation.yaml \
  --runtime unified_quant/configs/runtimes/simulation.yaml \
  --once

uv run quant-live \
  --strategy unified_quant/configs/strategies/leader_rotation.yaml \
  --runtime unified_quant/configs/runtimes/live_toss.yaml
```

`quant-paper` and `quant-live` loop unless `--once` is provided. Paper uses a
persistent JSON account and avoids processing the same candle twice. Live uses
the Toss broker, synchronizes holdings, rejects stale data and duplicate open
orders, sends Telegram notifications, and asks for confirmation when
`execution.live_confirm_required` is true.

## Split configuration

Strategy files define:

- strategy type and portfolio allocation;
- entry indicators;
- asset and benchmark filters;
- exit confirmation;
- leader-rotation rules.

Runtime files define:

- US, KR, or AUTO market selection;
- root-relative `universe.json` and optional symbol overrides;
- timeframe and download period;
- capital, costs, and broker;
- paper/live state and result locations.

Benchmarks are mapped automatically: US symbols use `QQQ`, KOSPI symbols use
`^KS11`, and KOSDAQ symbols use `^KQ11`.

## Research promotion

Grid search and Optuna load the same YAML pair as the normal backtest. They
evaluate train/validation/test segments and report stable market and
equal-weight benchmarks. The selected strategy YAML should be re-run with
`quant-backtest`, then with `quant-paper`, before changing only the runtime to
`live_toss.yaml`.

The `triple_filters.yaml` example combines three SuperTrend settings, an
Ichimoku cloud filter, an EMA trend filter, benchmark trend, relative strength,
and a confirmed Triple SuperTrend exit.

## Results

Backtests write beneath their runtime's `backtest.results_dir`:

- `summary.json`: metrics, configuration, trades, and skipped symbols;
- `equity.csv`: historical account equity.

Paper runs write beneath `paper.results_dir`:

- `metadata.json`: immutable configuration snapshot;
- `cycles.jsonl`: order plans, fills, prices, and account snapshots;
- `equity.csv`: cycle-level equity and cash.

Compare saved runs with:

```bash
uv run quant-compare \
  --paper-dir results/paper/<paper_run_id> \
  --backtest-dir results/backtests/<backtest_run_id>
```

## Live profile compatibility

`leader_rotation.yaml` plus `live_toss.yaml` preserves the operational profile:

- AUTO routing between KR and US market sessions;
- Toss execution with interactive confirmation enabled;
- `holding.json` synchronization and a 60-second loop;
- 30-minute stock data, benchmark trend filtering, and stale-symbol retries;
- SOXL/SOXS multiplier `4.5`;
- one-position, 90% allocation leader rotation;
- 1% minimum-profit brake before a rotation sale;
- sell-then-buy ordering with refreshed live cash.

## Verification

```bash
uv run python -m unittest discover -s unified_quant/tests -v
uv run quant-backtest --help
uv run quant-paper --help
uv run quant-live --help
uv run quant-search --help
uv run quant-optimize --help
```
