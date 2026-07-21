# Versioned Market Data

## V1 scope

V1 stores US daily data. Backtest, research, paper, and live signal generation
all use completed `1d` sessions. Intraday history is an interface-only V2 seam;
the current in-memory candle overlay has no disk or R2 persistence method.

The system reconstructs point-in-time **constituent membership** for `sp500`,
`nasdaq100`, and `russell3000`. It does not reproduce an index vendor's divisor,
float adjustment, weighting, or official index level. Strategy benchmarks are
the total-return-adjusted ETF series `SPY`, `QQQ`, and `IWV` respectively.

## Data flow

```text
SEC CIK/ticker/exchange + Nasdaq Trader symbol directories
                         |
                         v
          security_master + symbol_history
                         |
Yahoo raw daily OHLCV + corporate actions + archived source payloads
                         |
                         v
         immutable Parquet delta versions (Zstandard)
                         |
          validation + adjustment-factor rebuild
                         |
                         v
        atomic release pointer (all dataset versions together)
                         |
              DuckDB read_parquet(...)
                         |
          one pandas conversion at the engine boundary
```

Parquet remains the source of truth. DuckDB queries the files directly; it
does not create a persistent SQL database or write Parquet back after every
query. Daily files are appended as parent-linked deltas. `compact` periodically
collapses a long delta chain without changing its logical rows.

## Datasets

- `security_master`: stable `security_id`, current descriptive fields, active dates.
- `symbol_history`: ticker intervals keyed by `security_id`.
- `daily_price_raw`: unadjusted OHLCV by security and completed session.
- `corporate_actions`: dividends, splits, reductions, stock dividends, spin-offs,
  mergers, ticker changes, and delistings.
- `adjustment_factors`: split-only and total-return backward factors.
- `index_constituent_anchors`: a known constituent set on an anchor date.
- `index_membership_events`: actual effective-date `ADD`/`REMOVE` events.
- `custom_universe_overlays`: user additions/removals applied after index replay.
- `source_archive`: hashes and paths for the exact downloaded payloads.

Every curated row carries `source`, `retrieved_at`, and `source_hash`; adapters
also retain `source_url` when available. SEC CIK is preferred as the stable ID.
A listed instrument without a CIK gets a persistent internal UUID and a warning,
then remains linkable through explicit symbol-history events.

The SEC describes `company_tickers_exchange.json` as its CIK/name/ticker/exchange
association file, but explicitly does not guarantee its accuracy or scope. The
Nasdaq Trader directories complement it with Nasdaq- and other-exchange-listed
symbols, ETF flags, test-issue flags, and a file-generation timestamp:

- <https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data>
- <https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs>

Set a truthful `SEC_USER_AGENT` in `.env`; SEC asks automated clients to declare
a name and contact email.

## First sync and daily operation

Copy `.env.example` to `.env`, fill `SEC_USER_AGENT`, then run:

```bash
# Initial local backfill. Default start is 2000-01-01.
uv run quant-data \
  sync --backfill-start 2000-01-01

uv run quant-data status

uv run quant-data validate
```

Before a run, preflight calculates the latest completed XNYS session and waits
90 minutes after its close. It attempts automatic R2 synchronization at most
once for that expected session. `quant-data sync --force` is the explicit retry.
For a personal machine, run sync once after the US close or immediately before
backtest/research/paper/live. There is no server daemon requirement.

Source sync overlaps the last seven calendar days to catch late corrections,
appends raw-price/action deltas, rebuilds all adjustment factors, runs hard
validation, and only then commits a cross-dataset release. An interrupted or
invalid update can leave unpublished candidate versions, but consumers keep
reading the preceding release.

## Price semantics and corporate actions

`daily_price_raw` never changes old prices merely because a dividend or split
was announced. Corporate actions are separate facts. Adjustment factors can be
rebuilt and versioned when those facts are corrected.

- Signal default: `total_return_adjusted` OHLC.
- Optional `signal_price_mode`: `split_adjusted` or `raw`.
- Execution and account valuation: raw OHLC plus the corporate-action ledger.
- Dividend tax default: `0`; configurable as `dividend_tax_rate`.

The portfolio ledger processes each event ID exactly once. A dated cash
distribution becomes a receivable on its ex-date and moves to cash on its
payment date; both the receivable and entitlement ID survive a paper restart.
It also handles splits, capital reductions, stock dividends, spin-offs,
cash/stock mergers, ticker changes, delistings, and exact cash-in-lieu terms.
Backtest trade returns include distributions accrued while the trade was open.
Incomplete action terms are left unapplied and become `degraded` warnings under
the default `warn` policy, so ranking may continue. `block` turns them into hard
errors.

## Point-in-time universes

An anchor plus all events through a requested date yields that day's membership:

```text
anchor members
+ ADD events with effective_date <= requested date
- REMOVE events with effective_date <= requested date
+/- active custom overlays
```

This is done for every event date, not at artificial quarter boundaries. Import
vendor announcements or licensed extracts without translating them into custom
selection rules:

```bash
# Anchor file: security_id or historical symbol column
uv run quant-data import-index \
  --kind anchor --index-id sp500 --effective-date 2020-01-02 \
  --input anchor.csv --source-name sp_global --official

# Event file: effective_date, operation, and security_id or symbol
uv run quant-data import-index \
  --kind events --index-id sp500 --input events.csv \
  --source-name sp_global --official
```

Each import gzip-archives the original bytes by SHA-256, validates the combined
repository, and commits a new cross-dataset release. `best_effort` lets an
official event override contradictory lower-grade evidence, while an unresolved
same-grade conflict blocks replay. `official_only` also requires an official
anchor and a continuous official event path through the requested date; it
fails instead of silently dropping a non-official-only gap.

## R2 team cache

R2 objects are immutable. Each dataset has version manifests and a small
`current.json`; a release points to one mutually consistent version of every
dataset. Publishing uses conditional `PutObject` with `If-Match`/`If-None-Match`
ETags. If another teammate advances the pointer first, the publisher reloads
both lineages. Disjoint primary-key changes are rebased into a validated child
version; equal business rows are deduplicated. A same-key different-value change
is placed under `conflicts/` and never silently overwrites the winner.

Cloudflare documents R2's S3-compatible conditional `PutObject` operations:
<https://developers.cloudflare.com/r2/api/s3/api/>.

Configure the non-secret bucket/prefix in `configs/data.yaml`. The endpoint and
credentials are referenced by environment-variable name only:

```yaml
data_store:
  local_cache_dir: data/cache
  publish_enabled: false
  r2:
    enabled: true
    endpoint_env: R2_ENDPOINT_URL
    bucket: supertrend-quant-data
    prefix: supertrend-quant
    access_key_env: R2_ACCESS_KEY_ID
    secret_key_env: R2_SECRET_ACCESS_KEY
```

Use `sync --remote-only` to pull and `conflicts` to inspect quarantined
candidates. Direct publication through `quant-data sync --publish`,
`quant-data bootstrap-us --publish`, or `publish_enabled: true` is blocked
fail-closed because those paths cannot prove the complete lifecycle,
cross-validation, private-archive acknowledgement, and cold-download gates.

Run the dedicated operator path instead. `--preflight-only` executes every
local gate, reports all independent blockers in one JSON result, and never
constructs an R2 client or uses the network:

```bash
PYTHONPATH=unified_quant/src .venv/bin/python \
  unified_quant/scripts/publish_and_verify_r2.py --preflight-only

# Required when the preflight identifies private/internal-only source archives.
PYTHONPATH=unified_quant/src .venv/bin/python \
  unified_quant/scripts/publish_and_verify_r2.py \
  --ack-private-internal-only-source-archives
```

The second command verifies Cloudflare's private bucket state before its first
write, publishes with conflict-aware conditional writes, then redownloads the
release into a cold cache and verifies hashes and local gates again.

## Validation and compaction

Hard gates include required schemas, provenance, primary-key uniqueness,
positive/consistent OHLC, non-negative volume, XNYS trading dates, the expected
completed session, positive factors, raw-price/factor key coverage, known
security IDs, and possible index-event transitions. A changed overlapping price
row is quarantined instead of replacing the current version. Incomplete
corporate-action terms are warnings unless configured to block.

`status` reports release quality, total logical Parquet bytes, chain depth,
publisher, unresolved-action counts, and quarantined conflicts.

Run `validate` after any manual import and before publish. Compact when status
shows a long chain (for example monthly or after roughly 30 daily deltas):

```bash
uv run quant-data compact --dataset daily_price_raw
```

DuckDB supports direct multi-file Parquet scans plus filter/projection pushdown,
so this path avoids a Parquet-to-database-to-Parquet bottleneck:
<https://duckdb.org/docs/stable/data/parquet/overview>.

The provider first prunes immutable files from each manifest's minimum/maximum
session metadata, then applies symbol, session, and column projection in the
DuckDB scan. Tests capture the DuckDB execution plan and verify that an older
year partition is not opened for a short-period query. A real 500-symbol,
20-year load should be timed after the initial backfill on the target PC; the
three-second target is recorded as a hardware-dependent operational benchmark,
not a CI pass/fail threshold.
