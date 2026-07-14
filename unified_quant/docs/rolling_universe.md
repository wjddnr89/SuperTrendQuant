# Rolling Universe Backtests

This note documents the rolling-universe backtest mode added for
survivorship-aware Nasdaq-100 research.  Static profile universes use today's
constituents across the whole historical period; rolling universes instead read
dated constituent snapshots and allow new buys only in the active snapshot for
each bar.

## Runtime Config

Use `universe.source: history_file` in a runtime YAML:

```yaml
market: US
universe:
  source: history_file
  history_file: unified_quant/data/universes/nasdaq100_quarterly_history.json
data:
  timeframe: 1d
  period: 3y
```

`history_file` mode is intended for backtests and research.  Current live/paper
universe refresh still uses the normal `file` or `profiles` sources.

## History File Format

The file can be a mapping with `snapshots`:

```json
{
  "market": "US",
  "profile": "nasdaq100",
  "snapshots": [
    {
      "effective_date": "2023-07-03",
      "symbols": ["AAPL", "MSFT", "NVDA"]
    },
    {
      "effective_date": "2023-10-02",
      "symbols": ["AAPL", "MSFT", "NVDA", "ARM"]
    }
  ]
}
```

Each snapshot applies from `effective_date` until the next snapshot.  Ticker
strings are enough for US stocks; member mappings with `symbol`, `exchange`,
`yfinance_symbol`, or `benchmark` are also accepted when a ticker needs custom
metadata.

## Behavior

The data downloader fetches the union of all symbols across all snapshots.  The
backtest timeline then uses bars from the symbols active in each snapshot, not
the common intersection of every historical constituent.  Strategies keep held
symbols visible for exit logic after removal, but removed symbols are excluded
from new-buy candidate lists.

Because Nasdaq's public API only exposes the current Nasdaq-100 list, the
historical `history_file` must come from a separate trusted constituent-history
source such as archived index/ETF holdings data.
