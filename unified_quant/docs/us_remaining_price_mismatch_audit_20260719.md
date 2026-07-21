# US remaining non-no-data price mismatch audit (2026-07-19)

## Scope and controls

- Release: `20260715-20260718T230255094849Z`
- Baseline report: `/tmp/crossval-current-baseline.json`
- Baseline SHA-256: `83e141b08c4f84e5b466044a78ad922910740f5a6a5cb42e54c15b68301cd7bd`
- Audited set: 26 non-no-data mismatches after excluding the 34 terminal no-data cases and GPN.
- No network, EODHD, R2, release apply, or dataset mutation was used.
- Frozen WIKI ZIP SHA-256: `36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae`.
- WIKI remains `Unknown` license and is private/internal-only. A local `/tmp` ZIP is evidence for this audit, not publication-gate authority. Any accepted case still needs a repository `source_archive` extract plus a canonical provenance audit.

The 26 original failures are exactly 10 incomplete-history/ticker-reuse responses, 7 ETF ticker collisions, 4 retired `NONE/MUTUALFUND/YHD` metadata responses, 4 numeric/adjustment-basis conflicts, and 1 invalid-OHLCV response.

## Outcome

- Confirmed current raw-price defects: **0**.
- Previously confirmed and already repaired raw-price defect: **COL pre-boundary decimal scale**; current COL raw prices agree with frozen WIKI for the reviewed overlap and Triple Supertrend is unchanged.
- Confirmed current economic-model gap: **1 (legacy DD / Chemours distribution factor)**. DD raw prices themselves exactly equal WIKI for all 672 sessions.
- Frozen-WIKI price-only candidates with identical full-series Triple Supertrend after substitution: **14**.
- Cases that must remain fail-closed because the available alternate price series changes Triple Supertrend or lacks an exact third-provider arbiter: **12**.

No generic Yahoo ticker-reuse or missing-currency exception should be added. The only safe follow-up is an exact, identity-bound, private-only source-archive cohort for the 14 signal-identical cases, with action/factor coverage explicitly kept separate.

## Exact disposition by provider failure class

### Yahoo coverage/ticker reuse (10)

| Symbol | Yahoo result identifies | Frozen WIKI overlap | TripleST substitution | Disposition | Index relevance |
|---|---|---:|---|---|---|
| ADT | Current ADT Inc. beginning 2018, not old ADT | 334 | all 0 | WIKI price-only candidate | S&P 500 through 2016-05-02 |
| TE | T1 Energy beginning 2020, not TECO Energy | 377 | all 0 | WIKI price-only candidate | S&P 500 through 2016-07-01 |
| STI | Solidion beginning 2022, not SunTrust | 813 | all 0 | WIKI price-only candidate | S&P 500 through 2019-12-09 |
| SNDK | 2025 Sandisk listing, not legacy SanDisk | 342 | all 0 | WIKI price-only candidate | S&P 500 and Nasdaq-100 |
| FOXA | New Fox Corp. prelisting stub, not old 21CF history | 813 | all 0 | WIKI price-only candidate | S&P 500 and Nasdaq-100 |
| FOX | New Fox Corp. prelisting stub, not old 21CF history | 813 | all 0 | WIKI price-only candidate | S&P 500 and Nasdaq-100 |
| APC | ARKO Petroleum beginning 2026, not Anadarko | 813 | ST2 1, down-count 1; buy/sell 0 | fail-closed | S&P 500 through 2019-08-09 |
| IR | Current reused IR identity, not the requested old IR interval | 813 | ST1 1, ST3 3, down-count 4; buy/sell 0 | fail-closed | S&P 500 lineage continues through TT |
| LB | LandBridge beginning 2024, not L Brands | 813 | ST1/all-up/down-count 5; buy 1 | fail-closed, high priority | S&P 500 through 2021-08-03 |
| POM | Pomdoctor beginning 2025, not Pepco | 308 | ST2 1, ST3 1, all-up/down-count 1; buy 1 | fail-closed, high priority | S&P 500 through 2016-03-24 |

### Yahoo ETF ticker collisions (7)

| Symbol | Yahoo ETF collision | Frozen WIKI overlap | TripleST substitution | Disposition | Index relevance |
|---|---|---:|---|---|---|
| CAM | AB California Intermediate Municipal ETF | 314 | all 0 | WIKI price-only candidate | S&P 500 through 2016-04-04 |
| FB | ProShares S&P 500 Dynamic Daily Buffer ETF | 813 | all 0 | WIKI price-only candidate | S&P 500 and Nasdaq-100 through META transition |
| NFX | Corgi NFLX 2x Daily ETF | 813 | all 0 | WIKI price-only candidate | S&P 500 through 2019-02-15 |
| INFO | Harbor PanAgora Dynamic Large Cap Core ETF | 205 | all 0 | WIKI price-only candidate | S&P 500, 2017-06-02 through 2022-03-02 |
| EMC | Global X Emerging Markets Great Consumer ETF | 423 | all 0 | WIKI price-only candidate | S&P 500 through 2016-09-07 |
| SPLS | PIMCO US Stocks PLUS Active Bond ETF | 679 | ST1/down-count 3; buy/sell 0 | fail-closed | S&P 500 and Nasdaq-100 |
| PCL | PGIM Corporate Bond 10+ Year ETF | 285 | ST2/down-count 3; sell 1 | fail-closed, high priority | S&P 500 through 2016-02-22 |

### Retired Yahoo metadata (`NONE/MUTUALFUND/YHD`) (4)

| Symbol | Yahoo numeric result | Frozen WIKI result | TripleST substitution | Disposition |
|---|---|---|---|---|
| COL | Padded retired series; unstable relative to repaired raw basis | 813-session stable overlap | all 0 | WIKI price-only candidate; current raw defect already repaired |
| SCG | Starts 2015-07-16 and covers only 86.68% of local history | 813-session stable overlap | all 0 | WIKI price-only candidate |
| EVHC | Pre-2016-12 adjustment basis is unstable | 813-session stable two-regime overlap | all 0 | WIKI price-only candidate |
| HOT | 55 all-null rows; remaining 380 sessions match local exactly | Full 435-session WIKI relation conflicts with local | ST1 21, ST2 41, ST3 29; buy 9, sell 7 | fail-closed, highest price-conflict priority |

### Numeric/adjustment-basis conflicts and invalid OHLCV (5)

| Symbol | Exact finding | Normalized-provider TripleST sensitivity | Disposition / priority |
|---|---|---|---|
| FRCB | Yahoo has one invalid bar on 2023-12-11 and 10 sub-penny close deviations above 0.5%; relation SHA `878b511b04122e9d9fb09a16888079dd0ace330585e3e838a461162fc928bc6c` | ST2/down-count 3; sell 1 | fail-closed; low index impact because deviations occur after S&P removal |
| LILA | Two later split regimes are stable; the first regime is broken by one 2018-01-03 outlier with 22.5169% scale deviation | ST1 4, ST2 5, ST3 33; buy 1, sell 5 | fail-closed; no tracked-index membership |
| FWONA | Stable before the 2023-07-20 distribution; 12 post-event sessions deviate, maximum 2.5341% | ST1 2, ST2 3, ST3 2; sell 1 | fail-closed; disagreement is after Nasdaq-100 removal |
| FWONK | Stable before the 2023-07-20 distribution; 11 post-event sessions deviate, maximum 1.6716% | ST1 1, ST2 3; buy/sell 0 | fail-closed; disagreement is after Nasdaq-100 removal |
| DD | Current raw equals frozen WIKI exactly for 672/672 sessions; Yahoo is on a different historical adjustment basis. The missing Chemours stock-distribution factor remains material. | Existing factor proxy changes 4 ST1 states and 2 sell dates | raw price validated, action/factor remains fail-closed; highest economic-model priority |

## Frozen WIKI candidate pins

These 14 cases have identical complete Triple Supertrend signal hashes after replacing every WIKI-overlap raw OHLCV row and applying the current factors. They are candidates only; the release does not yet contain their exact extract/provenance records.

| Symbol | Overlap | Raw-line extract SHA-256 | Exact EOD/WIKI relation SHA-256 |
|---|---:|---|---|
| ADT | 334 | `fb9c6cecafa8b3fab1346787cc5e5fc45364664ec64b12ad1725d1dc629cea51` | `b1f34098371f11b11532060d7989a8be4b04822e9a9f634e0d9c4d517bd86422` |
| CAM | 314 | `fbc31dcb9550fa7e956d72d45fb00bbb64fd2794008817a8985c7324d4e28bc2` | `f64e6e8460ce962cf9727b90eb8080e961ad094041dfa946fcd7ab995c7c8251` |
| COL | 813 | `bef8afd45e986a70c32d43aaed2b43593e5e152bf60d509f9ec224e019d11ed0` | `438021cc67e0b737a83b35c495f7a8ae9a04d75abd69e2abc9121cc856e5b665` |
| EMC | 423 | `96bc80c99cbcb07d40fb9d4aba03463c8761894466ddab4924f2cc392f43ab8f` | `2fb01f84dae24ee64dcb5daebdf7ea3b9e0b3379543507f6a40dda00be3e30b5` |
| EVHC | 813 | `7d1bbdeeb3e355ea351f7c3aabc3c3896e52e035b1e3daeee8befc63d6f78ae1` | `6f0fc3fef42f1a37d81e1bcf619c56d95e647d344fede76d7fede4fdaf6414d5` |
| FB | 813 | `aa838c12a7c9c2cea588c1d2597da679d64fedf026de4d8f9ba6f779d7270ade` | `acc3fca09da894c3970e67f42c64214cdd5eefb56022bba30c7824dbb90718a3` |
| FOX | 813 | `beedbf58ea004ff5acb317e90e0ad2293ee9432f9a8724412bcb826cad21b6de` | `1fb94274eeba511ed567a4f56be97fa9c88475deee09a7280623b18ed5f96d9c` |
| FOXA | 813 | `2573b3e10ba24ce69e7b5acd20ec232f3701d7c78a9f70a2e9c38aec433b639c` | `1331326ad9827d7f5c82c004fbe1f69a9425dee166a3d6921633a3fcbbcb4da9` |
| INFO | 205 | `65270c6ad23368c35fe2df4e1602bc086584ea53ab2a781c7178a6b82672ef58` | `8ff2717ad7ce6b626a91c05f786a487b0885940f01432876d6a1662379954581` |
| NFX | 813 | `21e2717cbcd3c2f1bb3c8a29a2abeccb1772d1e64c9ac1e3f79dab872f51a80d` | `b9a0a75317d404fee885801c0da9752fcb44fc08a593adcb7974d25aabd44477` |
| SCG | 813 | `5d4b568c6b937720fe9c5b0bb1cba9a227b9a166cd21564d79902f572a5dfb18` | `1e67e6951a66e566e2e5e905c62cfe51da284c24bb7963a49ebc32cc52a27427` |
| SNDK | 342 | `7b38012925dae32ceb93032bc59208f778d8a87680be3b13cdd0ea5ec74374f6` | `b3fa31f0b31892e29126616d10e261bc22d4934fe8df0cf126bfb9dd339f8bcf` |
| STI | 813 | `6de4a417c833280fc944881596394b585b6dec9183c534e354596d6bc6ca50b0` | `f9f3aad2af50bbe3569a7161705a66f4a37081d02eef2fa48f47486462385d14` |
| TE | 377 | `63eb5ae21f69709e3a3e920cf02470b0638fc26988734951282ffc73e14bcc6d` | `dd9cd4244d79eaf29288f2cff1e5a79f2fb4f8930ace32d08c951e5ccae373e2` |

## Fail-closed WIKI relation pins

These exact WIKI substitutions change strategy state and therefore must not be promoted to reviewed price-only passes.

| Symbol | Overlap | Raw-line extract SHA-256 | Relation SHA-256 |
|---|---:|---|---|
| APC | 813 | `fb41890836a1802e55cb36fa4700c83cd7ccfce1a28744006015687cd9f39ef9` | `63cacc3b2bcc7e71dfc35f510bbff86140fcad6a6476094c720a8194653d72c7` |
| HOT | 435 | `5fe823edd493eccefa09a218890a1e3d1a4f89c919e6b25d5c2252508038263e` | `4861a53bda386a5c2c6db45817adb5ce429403ba0aa643fcda097652e6c4fa70` |
| IR | 813 | `ee91581e935aa62f86a960717916a9995cdf8fdb7cc6245eb7c283e33f02ee18` | `8dcb73e5e3e56499c820aab9e4dc7101da224ab092ec9b67c4acf394838fa1a6` |
| LB | 813 | `7c32382df55f786f9c43f216f34b177158c2272ec32b7c488aba2f980694bd75` | `030c2b63f62928a561b586cc65dd55ae992d709cb04bfe5e87a6ef57747b9890` |
| PCL | 285 | `55c6d5a97129ea244a97650b1c3a93bd8809ae732fe3bfdc46253a31957f0410` | `83fdf628a2544c7fd32b69d50f2a52f0f9b038a587405803430e88435392a4ea` |
| POM | 308 | `b07f1667e887ecc16b02d39e32b9845a6b69e13273506475fe3e67676237a23d` | `38025bf218f84db84a40bb7c920e578f1af9141803a827c78a8e6f39d5623084` |
| SPLS | 679 | `1d9938cef66e9a93b918408bcc3164be04f599c40bddafabe80d39a717b7d72a` | `c6959488b034ab7156f49b960ab0c2a46a1d7b06514b37299cdc79ab865978bd` |

## Backtest-impact order

1. **DD**: confirmed missing stock-distribution economics and S&P 500 membership; repair requires exact official child/factor modeling, not a raw-price substitution.
2. **HOT, LB, POM, PCL**: alternate evidence changes buy/sell dates while the securities are index members.
3. **APC, IR, SPLS**: alternate evidence changes trend/down-count state but not buy/sell in the measured overlap.
4. **FRCB, LILA, FWONA, FWONK**: provider disagreement changes strategy state, but the disputed periods are outside the tracked-index membership window or the security is not in the tracked indices.
5. **The 14 WIKI candidates**: raw substitution changes no Triple Supertrend field; their remaining work is reproducible archive/provenance integration and explicit action/factor-gap reporting.
