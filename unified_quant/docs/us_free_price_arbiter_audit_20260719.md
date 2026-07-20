# US free-price arbiter audit — 2026-07-19

## Decision

The seven selected legacy US price mismatches remain **fail-closed**. No
cross-validation pass, generic exception, dataset edit, release apply, R2 call,
or EODHD call was made. APC has a candidate one-row EODHD raw-close anomaly,
but the free evidence is not issuer-bound or independently sourced enough to
confirm an EODHD defect or authorize a repair.

## Pinned scope and controls

- Symbols: `APC`, `HOT`, `IR`, `LB`, `PCL`, `POM`, `SPLS`.
- Release: `20260715-20260718T230255094849Z`.
- Baseline cross-validation report SHA-256:
  `42a5ce96f79a0c8adfc7e49d90f3abac833ff6dca0ac849604abb1db2b6546f0`.
- Frozen WIKI ZIP SHA-256:
  `36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae`.
- Acquisition was preflight-capped at 7 Stooq plus 7 Boris/Kaggle URLs. The
  bounded run recorded exactly 14 HTTP attempts and zero retries. Offline
  artifacts independently preserve 13 raw responses plus one uncached HOT
  Boris 404 failure outcome; those artifacts do not by themselves reconstruct
  transport retry history. The lost HOT body was not retried.
- Every replayed audit call is cache-only: Yahoo 0, Stooq 0, Boris 0, EODHD 0,
  and R2 0.

## Provider evidence

| Provider | Evidence and basis | Result |
| --- | --- | --- |
| EODHD | Archived raw JSON; raw OHLCV; `adjusted_close` ignored | All retained Parquet rows exactly reproduce the archived provider bytes. HOT correctly excludes a 122-row zero-volume carry tail after its merger. |
| Yahoo | Previously cached `indicators.quote` raw OHLCV | Rejected for every symbol. Six tickers now resolve to another issuer/ETF; HOT has retired/incomplete metadata and 55 incomplete bars. |
| Frozen WIKI | Exact ZIP/member/per-symbol hashes; separate raw and adjusted OHLCV | Valid private audit evidence, but license is unknown, so redistribution and public publication remain blocked. |
| Stooq | One URL per symbol | All seven responses were Cloudflare HTML challenge pages, not price data. |
| Boris/Kaggle v3 | Exact per-URL response hashes; CC0; inferred adjusted basis | Valid CSV for APC, IR, and LB. PCL, POM, and SPLS returned cached 404 JSON; HOT is the recorded uncached 404. Files have no issuer metadata and upstream independence is unproven. |

## Backtest-sensitive reproduction

The audit recomputes Triple Supertrend with `(10, 1)`, `(11, 2)`, `(12, 3)`,
Wilder ATR, and exit-down-count 2. It pins both baseline signal hashes and the
exact changed sessions for all seven fields. The table gives the change counts
after substituting frozen WIKI prices in the actual
`total_return_adjusted` mode, in this order:

`ST1, ST2, ST3, AllUp, DownCount, Buy, Sell`.

| Symbol | Changed-field counts | Free third-arbiter result | Final disposition |
| --- | --- | --- | --- |
| APC | `0, 1, 0, 0, 1, 0, 0` | Boris adjusted history passes strict long scale/return stability against WIKI | Fail-closed; obtain issuer-bound raw 2015-11-10 close |
| HOT | `21, 41, 29, 20, 55, 9, 7` | No valid third price payload | Fail-closed |
| IR | `1, 0, 3, 0, 4, 0, 0` | Boris CSV usable diagnostically, but strict scale stability fails | Fail-closed |
| LB | `5, 0, 0, 5, 5, 1, 0` | Boris CSV usable diagnostically, but strict scale stability fails | Fail-closed |
| PCL | `0, 3, 0, 0, 3, 0, 1` | No valid third price payload | Fail-closed |
| POM | `0, 1, 1, 1, 1, 1, 0` | No valid third price payload | Fail-closed |
| SPLS | `3, 0, 0, 0, 3, 0, 0` | No valid third price payload | Fail-closed |

For APC, the largest EODHD/WIKI raw-close disagreement is 2015-11-10:
EODHD `65.54` versus WIKI `63.42` (3.3428%). Across 721 reviewed sessions,
Boris adjusted close versus WIKI adjusted close has return correlation
`0.9999993` and maximum one-scale deviation `0.0394%`. This is consistent with
a one-row EODHD raw anomaly, but Boris/WIKI upstream independence is unproven;
it is not independent confirmation and does not establish a safe replacement.

## Reproduction

```bash
.venv/bin/python unified_quant/scripts/audit_us_free_price_arbiters.py --no-write
.venv/bin/python -m pytest -q unified_quant/tests/test_audit_us_free_price_arbiters.py
```

The pinned JSON report is
`results/data_quality/us_cross_validation/free_price_arbiter_audit_20260719.json`
with SHA-256
`2db2a0dce3dde3e096f7686c69c15bd7048322cc8087510440fae689f1416bc7`.
