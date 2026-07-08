# main_jo_leader_rotation timeframe/period 10-run comparison

Strategy was fixed while only `configs/runtimes/simulation.yaml` `data.timeframe` and `data.period` were changed.

Fixed strategy settings:

- `supertrend.period: 7`
- `supertrend.multiplier: 2.5`
- `benchmark_trend.enabled: false`

Base command:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/qra-uv-venv uv run quant-backtest --strategy configs/strategies/main_jo_leader_rotation.yaml --runtime configs/runtimes/simulation.yaml --run-id <run-id>
```

## Completed runs

| Run | Timeframe | Period | Return | MDD | Sharpe | Win rate | Trades | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 01 | 5m | 5d | -13.80% | -15.21% | -15.99 | 21.43% | 14 | Too noisy, too many bad trades. |
| 02 | 15m | 5d | +0.00% | +0.00% | 0.00 | 0.00% | 0 | No trades. |
| 03 | 15m | 10d | -20.34% | -23.72% | -16.02 | 41.67% | 12 | Very poor; still too noisy. |
| 04 | 30m | 5d | +0.00% | +0.00% | 0.00 | 0.00% | 0 | Period too short. |
| 05 | 30m | 10d | +0.00% | +0.00% | 0.00 | 0.00% | 0 | Period still too short. |
| 06 | 30m | 30d | +20.33% | -8.04% | 10.32 | 100.00% | 2 | Best result; enough history without extra stale regime. |
| 07 | 30m | 60d | +20.27% | -8.04% | 10.30 | 100.00% | 2 | Nearly identical to 30d, slightly lower. |
| 08 | 1h | 60d | +0.00% | +0.00% | 0.00 | 0.00% | 0 | No trades. |
| 09 | 1h | 1y | +0.00% | +0.00% | 0.00 | 0.00% | 0 | No trades even with more history. |
| 10 | 1d | 2y | +0.00% | +0.00% | 0.00 | 0.00% | 0 | Current strategy does not trigger on daily candles. |

## Aborted runs

These were not counted in the 10 completed runs:

- `5m/30d`: interrupted after excessive runtime.
- `5m/10d`: interrupted after excessive runtime.

The slow path happens because high-resolution intraday data greatly increases candle count and this backtest repeatedly recalculates indicators over expanding histories.

## Decision

Set `configs/runtimes/simulation.yaml` to:

```yaml
data:
  timeframe: 30m
  period: 30d
```

Why:

- `30m/30d` had the best return: `+20.33%`.
- `30m/60d` was nearly tied at `+20.27%`, so 30 days is preferable because it is faster and avoids carrying older regime data.
- `5m` and `15m` produced too many bad trades.
- `1h` and `1d` produced no trades with the current strategy logic.

The meaningful zone for this strategy is currently `30m` with at least `30d` of history.
