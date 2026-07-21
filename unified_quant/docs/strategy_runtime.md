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
the Toss broker, synchronizes holdings, rejects incomplete historical coverage
and duplicate open orders, sends Telegram notifications, and asks for confirmation when
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
- point-in-time index events, a compatibility profile union, a manual
  `universe.json`, or explicit symbols;
- timeframe and download period;
- capital, costs, and broker;
- paper/live state and result locations.

The shared `configs/data.yaml` defines:

- Parquet provider and local cache;
- price adjustment and corporate-action policy;
- validation/source policy;
- optional R2 bucket and prefix. Publication is restricted to the dedicated
  strict publisher.

`quant-data` reads this file directly and therefore does not require a strategy
or runtime:

```bash
uv run quant-data status
uv run quant-data sync
uv run quant-data --data-config path/to/data.yaml validate

# Offline publication readiness; this never accesses R2 or EODHD.
PYTHONPATH=unified_quant/src .venv/bin/python \
  unified_quant/scripts/publish_and_verify_r2.py --preflight-only
```

The same data YAML keeps a `market_overrides.KR` Yahoo compatibility setting
until the versioned Parquet contracts are implemented for Korea.

Available index profiles are `nasdaq100`, `sp500`, `dow30`, `kospi200`, and
`kosdaq150`, plus `russell3000`. US profiles map their ETF benchmark to `QQQ`,
`SPY`, `DIA`, or `IWV`; a multi-profile US union uses `SPY`. KOSPI and KOSDAQ
symbols use `^KS11` and `^KQ11`.

For `universe.source: index_events`, a stored anchor plus every effective-date
event reconstructs the exact member set for the requested date. Stable
`security_id` values survive ticker changes; custom overlays are applied last.
Compatibility profile membership and optional filters can still be frozen in a
daily JSON snapshot for KR or legacy runs.

## Market-data contract

V1 stores US completed daily sessions in immutable, Zstandard-compressed
Parquet versions. DuckDB reads only the requested securities and dates directly
from those files. A release records one mutually consistent version for raw
prices, actions, factors, identifiers, and any imported index datasets.

- signals use `total_return_adjusted` OHLC by default;
- fills and account valuation use raw OHLC;
- dividends, splits, mergers, ticker changes, and delistings are applied by an
  exactly-once portfolio ledger;
- paper/live block the entire order plan when historical coverage is incomplete;
- US live signal history comes from the same daily release as research and
  backtest, while Toss quotes remain the execution boundary;
- KR uses the legacy Yahoo path until the same Parquet contracts are implemented.

Run `quant-data sync`, `validate`, and `status` before research or execution as
needed on a personal machine. See [market_data.md](market_data.md) for source,
index-import, R2/CAS, and compaction workflows.

## Research promotion

Grid search and Optuna load the same strategy/runtime pair and shared data YAML
as the normal backtest. They
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

- `summary.json`: metrics, configuration, trades, skipped symbols, data release,
  completed session, quality/warnings, and processed corporate-action IDs;
- `equity.csv`: historical account equity.
- `universe_snapshot.json`: membership, filters, exclusions, and as-of hash.

Strategy comparisons write beneath `results/research/comparisons/<run_id>`:

- `comparison.csv`: ranked metrics for every successful strategy YAML;
- `summary.json`: winner, comparison settings, common date range, and failures;
- `strategies/`: the normal backtest summary and equity files for each strategy.

Paper runs write beneath `paper.results_dir`:

- `metadata.json`: immutable configuration snapshot;
- `cycles.jsonl`: order plans, fills, raw execution prices, and account snapshots;
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
- completed daily US Parquet signals and raw Toss execution quotes;
- Yahoo compatibility data for KR;
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
uv run quant-data --help
```
