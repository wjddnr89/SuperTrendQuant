# Frozen WIKI price arbitration: BBBY, BBT, legacy DD

Reviewed at: 2026-07-19
Base release: `20260715-20260718T221334562619Z`
Operation: `repair_us_wiki_price_arbitration`

## Decision

| Target | Raw price | Identity | Actions / factors | Write decision |
|---|---|---|---|---|
| BBBY | Passed, price-only | Passed on `BBBY_old.US` and the existing SEC-bound identity | WIKI action coverage is incomplete; unchanged | Archive evidence only |
| BBT | Passed, price-only | Passed on `BBT_old.US`; the official BBT -> TFC boundary remains pinned | Four WIKI dividends are absent from the current ledger; unchanged | Archive evidence only |
| legacy DD | Raw rows already equal the frozen WIKI segment | Existing legacy DD -> DWDP merger identity remains pinned | Blocked: the 2015 Chemours distribution cannot safely be represented yet | No repair |

Yahoo symbol-only comparisons are not an acceptable identity basis for BBBY or BBT because those symbols can resolve to later/reused issuers. The accepted comparison is the exact frozen WIKI ticker history joined to the already identity-bound security ID.

## BBBY result

- Security ID: `US:EODHD:dbd287da-b35f-5aaa-873c-76f941b4d93b`
- Compared sessions: 650, from 2015-01-02 through 2018-03-07
- Exact relation fingerprint: `fecd59e84360bd2173ab0e0d30d190731cfb9f97d1a3ef3dae81292783e0ab1a`
- Close: 637 exact sessions; maximum absolute difference `0.01`; return correlation `0.9999990453003861`
- Replacing the reviewed sessions with WIKI OHLCV while retaining the current factors changed no Triple Supertrend state, entry, or exit field.
- Triple Supertrend fingerprint before and after substitution: `2d964d020e6707aff59cd96ee362b131f99a68ef4ccdf2758f5e369b0f744ec2`
- WIKI lists five dividends; the current ledger also contains 2017-09-14 and 2017-12-14 dividends. This proves WIKI is not a complete action source, so no action or factor is replaced.

## BBT result

- Security ID: `US:EODHD:aadcce22-62c7-522f-bbeb-861933af1d99`
- Compared sessions: 813, from 2015-01-02 through 2018-03-27
- Exact relation fingerprint: `ed5375f9a9e4e5e83db239bc4c5e7d70af86cfd75ed703331d55efabb3c5770c`
- Close: 811 exact sessions; maximum absolute difference `0.015`; return correlation `0.9999989350223387`
- Replacing the reviewed sessions with WIKI OHLCV while retaining the current factors changed no Triple Supertrend state, entry, or exit field.
- Triple Supertrend fingerprint before and after substitution: `d55ecc680990a769157863997388ba4f88b71a297157f5070ae23e1241301853`
- The current action ledger is missing WIKI dividends on 2015-05-13, 2015-08-12, 2015-11-10, and 2016-02-10. WIKI in turn omits current events on 2018-02-08 and 2018-03-05. BBT is therefore passed for raw price only; action/factor validation remains incomplete.

## Legacy DD block

The frozen WIKI row on 2015-07-01 reports `3.2`, but this is not authorized as a cash dividend. Official discovery material describes a Chemours share distribution of one CC share per five legacy DD shares. The issuer tax material suggests a child value of `$3.242` per legacy DD share and a 94.915% / 5.085% parent-child basis allocation.

The current local store lacks all three items required for an identity-bound repair:

1. Hash-pinned official 2015 Chemours spin-off bytes in `source_archive`.
2. A canonical Chemours child security ID.
3. A complete identity-bound Chemours price path suitable for valuation and exit.

Accordingly, legacy DD remains fail-closed. No `3.2` cash or special dividend is created, no WIKI proxy factor is applied, and no identity is changed. A proxy-only sensitivity run would change four first-Supertrend states and two sell-signal dates, which makes an unsupported approximation material to the requested strategy.

Discovery-only official URLs, not yet local hash-pinned:

- [DuPont completion announcement](https://www.sec.gov/Archives/edgar/data/30554/000003055415000065/exhibit991pressrelease.htm)
- [Issuer Form 8937 and tax-basis material](https://s23.q4cdn.com/116192123/files/doc_downloads/Tax-Cost-Basis-Allocation.pdf)

## Atomic write boundary

The default command is a read-only plan. Apply writes only:

- one header + BBBY + BBT frozen WIKI CSV extract;
- one canonical JSON provenance object;
- one new `source_archive` dataset version;
- one new release pointer preserving every prior warning and adding:
  `licenseName=Unknown; private/internal-only; redistribution/public publication blocked`.

It does not write raw prices, actions, factors, identities, index rows, R2, or any remote service. Local apply requires `--ack-private-internal-only-local-repair`. A private R2 publisher acknowledgement remains separate.

## Verification

- Read-only plan against the base release: passed.
- Unit and tamper suite: 6 passed.
- Tamper cases cover the ZIP hash, formal license metadata, partial evidence archives, decompressed payload hashes, and the local private-only apply acknowledgement.
