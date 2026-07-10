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
execution when the account contains a position outside the configured
universe; this prevents the strategy from liquidating manually managed assets.
Post-sell buys are submitted only after the broker confirms the prerequisite
position has disappeared from the refreshed account.

## Configuration

- `configs/strategies/leader_rotation.yaml`: current operational leader strategy.
- `configs/strategies/triple_filters.yaml`: Triple SuperTrend with Ichimoku and EMA filters.
- `configs/runtimes/simulation.yaml`: backtest and paper defaults.
- `configs/runtimes/research_us.yaml`: US research defaults.
- `configs/runtimes/research_kr.yaml`: KR research defaults.
- `configs/runtimes/live_toss.yaml`: live Toss execution profile.

Strategy YAML owns signal and portfolio behavior. Runtime YAML owns market,
universe, data period, capital, costs, broker, and output paths. All supplied
runtimes use `universe_file: universe.json` because commands are executed from
the repository root. Relative paths also fall back to the unified project and
repository roots, so imports and tests do not depend on a hidden working
directory.

See [architecture.md](docs/architecture.md) for the shared-engine design and
[strategy_runtime.md](docs/strategy_runtime.md) for runtime behavior and output
files.

## Verification

```bash
uv run python -m unittest discover -s unified_quant/tests -v
uv run quant-backtest --help
uv run quant-search --help
uv run quant-optimize --help
uv run quant-paper --help
uv run quant-live --help
```
