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

uv run quant-compare-strategies \
  --runtime unified_quant/configs/runtimes/simulation.yaml \
  --rank-by calmar

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
- an index-profile union, a manual `universe.json`, or explicit symbols;
- timeframe and download period;
- capital, costs, and broker;
- paper/live state and result locations.

Available index profiles are `nasdaq100`, `sp500`, `dow30`, `kospi200`, and
`kosdaq150`. A single US profile maps RS to `QQQ`, `SPY`, or `DIA`; a multi-profile
US union uses `SPY`. KOSPI and KOSDAQ symbols use `^KS11` and `^KQ11`.

Profile membership and filter results are frozen in a daily JSON snapshot. The
balanced default requires US/KR prices of `$5`/`1,000원`, 20-day average turnover
of `$10M`/`10억원`, and 120 completed daily bars. Management, suspension,
delisting, ETF/ETN, SPAC, and preferred-share checks are independently editable
under `universe.filters`.

## Research promotion

Grid search and Optuna load the same YAML pair as the normal backtest. They
evaluate train/validation/test segments and report stable market and
equal-weight benchmarks. The selected strategy YAML should be re-run with
`quant-backtest`, then with `quant-paper`, before changing only the runtime to
`live_toss.yaml`.

Every strategy YAML has a required, separate scoring section:

```yaml
scoring:
  type: relative_strength
  params:
    lookback_bars: 100
```

The scorer ranks eligible entry candidates; it is not a signal filter. The
`triple_filters.yaml` example combines three SuperTrend settings, an Ichimoku
cloud filter, an EMA trend filter, benchmark trend, this relative-strength
scorer, and a confirmed Triple SuperTrend exit. `triple_filters_standalone.yaml`
keeps the same signal and trend filters but supports multiple positions and
never sells solely to rotate leaders.

## Results

Backtests write beneath their runtime's `backtest.results_dir`:

- `summary.json`: metrics, configuration, trades, and skipped symbols;
- `equity.csv`: historical account equity.
- `universe_snapshot.json`: membership, filters, exclusions, and as-of hash.

Strategy comparisons write beneath `results/research/comparisons/<run_id>`:

- `comparison.csv`: ranked metrics for every successful strategy YAML;
- `summary.json`: winner, comparison settings, common date range, and failures;
- `strategies/`: the normal backtest summary and equity files for each strategy.

Paper runs write beneath `paper.results_dir`:

- `metadata.json`: immutable configuration snapshot;
- `cycles.jsonl`: order plans, fills, prices, and account snapshots;
- `equity.csv`: cycle-level equity and cash.
- `universe_snapshot.json`: the exact daily universe used by the run.

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
uv run quant-compare-strategies --help
uv run quant-paper --help
uv run quant-live --help
uv run quant-search --help
uv run quant-optimize --help
```
