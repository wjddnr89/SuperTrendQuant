# Point-in-Time Index Universes

The authoritative US path is `universe.source: index_events`. It replaces the
old quarterly snapshot approximation with a stable-security replay:

```yaml
market: US
universe:
  source: index_events
  profiles:
    US: [nasdaq100]
  filters:
    enabled: false
data:
  timeframe: 1d
  period: max
```

The shared Parquet provider and `index_source_mode` now live in
`configs/data.yaml`, not in each US runtime.

For each profile, the engine selects the latest anchor on or before the target
date, applies every actual-effective-date `ADD` and `REMOVE`, then applies active
custom overlays. Membership carries `security_id`; `symbol_history` supplies the
ticker that was active on each date. The resulting schedule gates new entries,
while already-held removals remain exit-only.

`history_file` remains a compatibility source for existing JSON snapshots, but
it is not used by the supplied US runtimes. A snapshot applies today's or a
manually approximated member set between dates and therefore has weaker audit
and effective-date guarantees.

This reconstructs constituent membership, not an official vendor index level.
See [market_data.md](market_data.md) for imports, source policy, validation, and
the distinction between SPY/QQQ/IWV ETF benchmarks and vendor index formulas.
