# main_jo_leader_rotation manual 10-run comparison

Generated after manually repeating: edit YAML -> run backtest -> compare -> edit YAML.

Base command:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/qra-uv-venv uv run quant-backtest --strategy configs/strategies/main_jo_leader_rotation.yaml --runtime configs/runtimes/simulation.yaml --run-id <run-id>
```

## Result table

| Run | Key change | Timeframe | Period | Return | MDD | Sharpe | Win rate | Trades | Read |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | multiplier 2.5, benchmark 1h | 30m | 60d | +18.99% | -8.05% | 9.80 | 100.00% | 2 | Best before this manual pass. |
| 01 | multiplier 2.3 | 30m | 60d | +16.48% | -8.34% | 8.55 | 75.00% | 4 | More trades, worse return and drawdown. |
| 02 | multiplier 2.7 | 30m | 60d | +17.29% | -8.09% | 8.94 | 100.00% | 2 | Too slow versus 2.5. |
| 03 | multiplier 2.4 | 30m | 60d | +18.58% | -8.05% | 9.53 | 100.00% | 2 | Close, but still below 2.5. |
| 04 | daily candles, benchmark 1d | 1d | 2y | +0.00% | +0.00% | 0.00 | 0.00% | 0 | No trades; not useful for this strategy as configured. |
| 05 | hourly candles, benchmark 1h | 1h | 1y | +0.00% | +0.00% | 0.00 | 0.00% | 0 | No trades. |
| 06 | hourly candles, benchmark off | 1h | 1y | +0.00% | +0.00% | 0.00 | 0.00% | 0 | No trades even without benchmark filter. |
| 07 | supertrend period 5 | 30m | 60d | +18.54% | -8.05% | 9.51 | 100.00% | 2 | More sensitive period did not improve return. |
| 08 | supertrend period 10 | 30m | 60d | +18.59% | -8.05% | 9.54 | 100.00% | 2 | Slower period also did not improve return. |
| 09 | benchmark filter off | 30m | 60d | +19.32% | -8.04% | 9.88 | 100.00% | 2 | Best manual result; filter was slightly limiting upside. |
| 10 | benchmark off, multiplier 2.3 | 30m | 60d | +9.77% | -13.90% | 5.15 | 50.00% | 6 | Too aggressive; clearly worse. |

## Decision

Keep the original 30m/60d runtime for this strategy. The 1d/2y and 1h/1y tests produced no trades, so they are not comparable candidates without changing the entry/rotation logic.

Final chosen settings:

- `configs/runtimes/simulation.yaml`: `data.timeframe: 30m`, `data.period: 60d`
- `configs/strategies/main_jo_leader_rotation.yaml`: `supertrend.period: 7`, `supertrend.multiplier: 2.5`
- `configs/strategies/main_jo_leader_rotation.yaml`: `benchmark_trend.enabled: false`

The most meaningful finding is that benchmark filtering hurt this specific 60-day 30-minute test slightly. Lowering the multiplier opened more trades but materially worsened drawdown and return, so the safer improvement is only removing the benchmark filter while keeping multiplier 2.5.
