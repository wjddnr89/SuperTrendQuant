# SuperTrendQuant Unified

`unified_quant` is the authoritative package that combines the extensible
strategy and paper/live runtime from `jo_factory` with the research workflow
from `module`. Backtest, grid search, Optuna optimization, paper trading, and
live trading all load the same strategy/runtime YAML pair.

Run every command below from the repository root so paths such as
`universe.json`, `state/`, and `results/` resolve consistently.

## Setup

```bash
uv sync
```

## Common commands

```bash
# Backtest
uv run quant-backtest \
  --strategy unified_quant/configs/strategies/leader_rotation.yaml \
  --runtime unified_quant/configs/runtimes/simulation.yaml

# US grid search
uv run quant-search \
  --strategy unified_quant/configs/strategies/triple_filters.yaml \
  --runtime unified_quant/configs/runtimes/research_us.yaml \
  --timeframes 30m,1h,2h,4h,1d \
  --show-best-config

# Compare every strategy YAML on one shared simulation runtime
uv run quant-compare-strategies

# Compare with the US research runtime and composite ranking
uv run quant-compare-strategies \
  --runtime unified_quant/configs/runtimes/research_us.yaml \
  --rank-by composite

# KR Optuna optimization
uv run quant-optimize \
  --strategy unified_quant/configs/strategies/triple_filters.yaml \
  --runtime unified_quant/configs/runtimes/research_kr.yaml \
  --n-trials 100 \
  --save-best-dir results/research/kr/best

# One paper cycle
uv run quant-paper \
  --strategy unified_quant/configs/strategies/leader_rotation.yaml \
  --runtime unified_quant/configs/runtimes/simulation.yaml \
  --once

# Live Toss runtime (loops and asks before sending orders)
uv run quant-live \
  --strategy unified_quant/configs/strategies/leader_rotation.yaml \
  --runtime unified_quant/configs/runtimes/live_toss.yaml
```

Do not add `--yes` to the live command until the generated order plan has been
validated. `live_toss.yaml` retains the existing Toss broker, confirmation,
holdings-file, and 60-second loop settings.

Live execution consumes completed candles only and fails closed when quotes,
cost basis, or open-order checks are unavailable. It also blocks strategy
execution when the account contains an unknown position outside the configured
universe. A position recorded as previously managed remains exit-only after an
index removal or filter rejection; it cannot be bought again.
Post-sell buys are submitted only after the broker confirms the prerequisite
position has disappeared from the refreshed account.

## Configuration

- `configs/strategies/leader_rotation.yaml`: current operational leader strategy.
- `configs/strategies/triple_filters.yaml`: leader rotation with Triple SuperTrend, Ichimoku, and EMA filters.
- `configs/strategies/triple_filters_standalone.yaml`: independent multi-position version that ranks new entries by relative strength without rotating existing holdings.
- `configs/runtimes/simulation.yaml`: backtest and paper defaults.
- `configs/runtimes/research_us.yaml`: US research defaults.
- `configs/runtimes/research_kr.yaml`: KR research defaults.
- `configs/runtimes/live_toss.yaml`: live Toss execution profile.

`quant-compare-strategies` recursively discovers every YAML beneath
`configs/strategies`, aligns all candidates to the same post-warmup date range,
and selects one winner by Calmar ratio (default) or an equal-weight composite
percentile score. Results are saved beneath `results/research/comparisons` with
the comparison table, summary, and each strategy's normal backtest artifacts.

Strategy YAML owns signal and portfolio behavior. Runtime YAML owns market,
universe, data period, capital, costs, broker, and output paths. Research uses
the `sp500` profile in the US and the `kospi200` + `kosdaq150` union in Korea.
The live runtime deliberately keeps the existing manual `universe.json` list.

Index universes use the nested runtime section below. Available profiles are
`nasdaq100`, `sp500`, `dow30`, `kospi200`, and `kosdaq150`; profiles from the
same market are unioned and deduplicated.

```yaml
universe:
  source: profiles
  profiles:
    US: [sp500]
  refresh: daily
  snapshot_dir: state/universes
  filters:
    enabled: true
    min_price: {US: 5, KR: 1000}
    avg_turnover_window: 20
    min_avg_turnover: {US: 10000000, KR: 1000000000}
    min_history_daily_bars: 120
```

The remaining status and security-type filters are shown in the supplied
runtime files. Every value is editable. `--symbols` remains the highest-priority
unfiltered override, while `--universe-file` selects a manual JSON universe and
`--universe-profiles` overrides a single-market profile list.

Each strategy YAML also requires a top-level `scoring` section. The supplied
profiles select `relative_strength` with `lookback_bars: 100`; strategies apply
their own signal/filter rules first, then use the common scorer registry to fill
available positions in deterministic score order.

See [architecture.md](docs/architecture.md) for the shared-engine design and
[strategy_runtime.md](docs/strategy_runtime.md) for runtime behavior and output
files.

## Verification

```bash
uv run python -m unittest discover -s unified_quant/tests -v
uv run quant-backtest --help
uv run quant-compare-strategies --help
uv run quant-search --help
uv run quant-optimize --help
uv run quant-paper --help
uv run quant-live --help
```
