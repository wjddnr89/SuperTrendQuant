# Strategy Runtime

This is the split runtime for reusable strategy definitions. Existing scripts
such as `main.py`, `main_jo.py`, and `backtest/*.py` are left in place.

## Modes

```bash
uv run quant-backtest \
  --strategy configs/strategies/leader_rotation.yaml \
  --runtime configs/runtimes/simulation.yaml

uv run quant-paper \
  --strategy configs/strategies/leader_rotation.yaml \
  --runtime configs/runtimes/simulation.yaml \
  --state state/paper.json

uv run quant-live \
  --strategy configs/strategies/leader_rotation.yaml \
  --runtime configs/runtimes/live_toss.yaml
```

`quant-live` loops by default. It prints the order plan and asks for `yes` before sending orders
when `execution.live_confirm_required` is true.

`quant-paper` loops by default and uses the live-style market loop, but executes against
`PaperBroker`. It checks market hours, runs only for the open market, avoids
re-processing the same 30m candle, persists paper account state, and writes
cycle/equity logs for later comparison. Use `--once` for a single diagnostic
cycle.

## Config Layout

- `configs/strategies/*.yaml`: strategy logic only.
- `configs/runtimes/simulation.yaml`: backtest and paper settings.
- `configs/runtimes/live_toss.yaml`: live Toss execution settings.

Strategies define entries, filters, exits, and rotation rules. Runtimes define
market, universe source, symbols, period, capital, costs, broker, and live loop
settings.

Benchmark symbols are not configured manually. They are mapped per symbol:

- US -> `QQQ`
- KR KOSPI -> `^KS11`
- KR KOSDAQ -> `^KQ11`

## Result Storage

Backtest runs are saved under `results/backtests/<run_id>/` unless
`--no-save` is passed.

- `summary.json`: metrics, config, trade returns, skipped symbols
- `equity.csv`: historical strategy equity

Paper runs are saved under `results/paper/<run_id>/`.

- `metadata.json`: config snapshot
- `cycles.jsonl`: every paper cycle with order plan, fills, prices, and account snapshots
- `equity.csv`: timestamp, market, candle base, equity, cash, positions value

The paper account itself is stored separately in `paper.state_file`, for example
`state/paper.json`, so paper trading can continue from the last virtual account.

Compare saved runs:

```bash
uv run quant-compare \
  --paper-dir results/paper/<paper_run_id> \
  --backtest-dir results/backtests/<backtest_run_id>
```

## Architecture

- `supertrend_quant.config`: loads and composes YAML/TOML settings.
- `supertrend_quant.indicators`: indicator calculations such as Supertrend.
- `supertrend_quant.strategies`: strategy interface, registry, and strategy modules
  that create shared order plans from market data.
- `supertrend_quant.brokers`: paper JSON broker and Toss live broker.
- `supertrend_quant.live_runtime`: migrated `main_jo.py` live loop, schedule,
  holdings sync, open-order guard, Telegram, and Toss order dispatch.
- `supertrend_quant.runners`: mode-specific execution flow.
- `supertrend_quant.cli`: `uv run` entrypoints.

## Live Runtime Notes

`configs/strategies/leader_rotation.yaml` with `configs/runtimes/live_toss.yaml`
is the current live profile. The live runtime keeps these execution semantics:

- KR/US market schedule and close briefing windows.
- Toss account cash/holding sync into `holding.json`.
- Telegram notifications for startup, close briefing, and sent orders.
- 30m yfinance stock/benchmark cache plus stale-symbol exclusion.
- Individual stale-symbol retry during live cycles.
- KR benchmark routing: KOSPI -> `^KS11`, KOSDAQ -> `^KQ11`; US -> `QQQ`.
- Benchmark Supertrend buy filter when enabled by the strategy config.
- Relative-strength lookback from `lookback_bars` in the strategy config.
- SOXL/SOXS Supertrend multiplier of 4.5 from `symbol_multipliers`.
- Open-order guard and 1% minimum-profit hard brake before leader-rotation sells.
- Same-cycle post-sell leader buy intent, with buy quantity recalculated after
  refreshed live cash is read.
- 90% available-cash allocation for new live buys.

## Verification

```bash
uv run python -m unittest discover -s tests
uv run quant-backtest --strategy configs/strategies/leader_rotation.yaml --runtime configs/runtimes/simulation.yaml
uv run quant-paper --strategy configs/strategies/leader_rotation.yaml --runtime configs/runtimes/simulation.yaml --state /tmp/supertrend-paper.json --once --ignore-schedule
```
