# US 13 special price mismatch audit (2026-07-19)

## Decision

This is an offline, read-only audit of the 13 remaining special non-WIKI price
targets after the seven Yahoo caches were filled. No network, EODHD API, R2,
release apply, validator/configuration change, or dataset mutation was used.

There are **zero unconditional data-accuracy passes** in this cohort:

| Disposition | Count | Targets |
|---|---:|---|
| Conditional evidence-only closure; retain current prices | 1 | GPN |
| Confirmed economic action/factor repair, but local evidence is incomplete | 1 | legacy DD |
| Possible 2023 action-date/adjustment-basis repair, but not locally provable | 2 | FWONA, FWONK |
| Price arbitration must remain fail-closed | 9 | LILA, FRCB, APC, HOT, IR, LB, PCL, POM, SPLS |

The earlier working hypothesis that LILA was an action/factor candidate was
rejected. Its failure is one anomalous Yahoo bar on 2018-01-03. FWONA and FWONK
do have an exact intermediate adjustment regime around their 2023 actions, but
the local archive does not contain official 2023 terms that would justify
changing an action date, action type, or factor.

## Immutable audit pins

- Strict report: `/tmp/us_cross_validation_after_7_fetches.json`
  - SHA-256: `aa783468d4e58d975eb355a34eafb1b386c0a3477a7e2b7b94a58410a5538bcb`
  - release: `20260715-20260719T051324634358Z`
  - event mismatches: `0`; price mismatches: `64`; unresolved: `0`
  - candidate input: `32cf8a701a37041584b4a8117064c858d122d3fa50b6f76f19f3e05bd4060c64`
  - lifecycle evidence report: `7760962880440d35d900305061c611041d449494cf17273a5502817892860554`
  - lifecycle resolutions: `c955a3fdc6bec69a99a763f43e4de5d989d2520bec0a6a2833588ae07f158d1`
- Release manifest SHA-256:
  `127a97a567e10eb89086fb2b7f732119c0171bb343a67d82f1b0d4b3af651736`
- Supporting audit:
  `unified_quant/docs/us_remaining_price_mismatch_audit_20260719.md`, SHA-256
  `dc4418a8638165e62a0eb468b034ed5e740d1daadb87d91b06bc43a4090da3dd`.
- Free-provider audit:
  `results/data_quality/us_cross_validation/free_price_arbiter_audit_20260719.json`,
  SHA-256 `2db2a0dce3dde3e096f7686c69c15bd7048322cc8087510440fae689f1416bc7`.
- GPN offline arbitration: `/tmp/gpn_price_audit_latest.json`, file SHA-256
  `49613891544308f31d51b9f68523de0a76da3a8d48c973642e2a6cd716e3aed0`,
  canonical-body SHA-256
  `dbc3c948b87c5877059c9b806d20bf13244f51a7f2823e5961fa9c427adbd0e7`.

Any repair plan must abort if these audited dataset manifests have drifted:

| Dataset | Version | Manifest SHA-256 |
|---|---|---|
| `daily_price_raw` | `lifecycle-metadata-20260715-cc5c8c1a37f34563a77188240741bcef-daily_price_raw` | `187e20fe5718699362d6571efd5b21e717214088d8060a3d8dbfa4802c431b04` |
| `corporate_actions` | `arnc-hwm-ticker-20260715-77b58f7a838f410e8c297dc1427fd76e-corporate_actions` | `fc6cb7be2eabf653ac23410f67bdce496839864cb0f02ee3eb7da3f1caf10c54` |
| `adjustment_factors` | `lifecycle-metadata-20260715-cc5c8c1a37f34563a77188240741bcef-adjustment_factors` | `1e2d68e93dcd7fbc4296cdbfa408b9d4b6eddbbfba197a1053850aee2d0243ea` |
| `security_master` | `lifecycle-metadata-20260715-cc5c8c1a37f34563a77188240741bcef-security_master` | `1aac36c77074da11bd971e0edcfe532528d3a3202e10dca599b48aad7782fec9` |
| `symbol_history` | `lifecycle-metadata-20260715-cc5c8c1a37f34563a77188240741bcef-symbol_history` | `c05fce0d1a359df2f39f6a14f67a911bd07cc768589a0f3ba11abc4fff5b7f74` |
| `source_archive` | `lifecycle-metadata-20260715-cc5c8c1a37f34563a77188240741bcef-source_archive` | `627a7b0f7d7fed1eec253a6120f0ca724a6a5cfe5a9c17c1012509c534df1841` |

## Exact target audit

### GPN: retain current values, conditional evidence-only closure

- Target: `3e611a634291d14b524dfcd8ff1e33d920c15d9dd859b4065ff5f8adafba2661`
- Security: `US:EODHD:d3e52f8f-ead7-581c-adc2-af968904d1a8`
- EODHD raw: `39235d1f822263c250a69557ff9d9cd8310a0b7487ad68cc9a5dfec39fb2a46c`
- Yahoo source/wrapper:
  `be071582774ed528eff679d4e5d5630f191ef3f0519635ebc5627e8789ecda11` /
  `fce7693dcd1bad37d8fcc1f8d6825851651831eba90df0b5ac7e6dc69170ff98`
- Eikon third source:
  `5004f8b8c7c76eafde90bb0002669a460f70de8967bce7462e6287d92825f90d`
- Frozen WIKI GPN extract/ZIP:
  `9b60c9c5bdb6b8de302828807b42103441cb8adf8d73e1c3f3d74d27be8fc839` /
  `36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae`
- Disputed fields: 2015-08-24 high, EODHD `109.99` versus normalized Yahoo
  `106.6900028`; 2019-08-19 low, EODHD `158.54` versus Yahoo `154.00`.
- WIKI and Eikon support the 2015 EODHD value. Eikon supports the 2019 EODHD
  value. Accepted values equal the current release, so prices, actions, factors,
  and the backtest must not change.
- GPN is S&P 500 relevant. The rejected Yahoo 2015 high would change nine trend
  sessions and two buy-signal dates. The rejected Yahoo 2019 low changes no
  Triple Supertrend state.
- Full closure requires privately archiving the exact Eikon CSV and GPN-only
  WIKI extract, then adding a code-pinned third-source arbitration path. WIKI
  alone cannot arbitrate 2019 because it ends in 2018. Eikon has no repository
  license grant and is private/internal-validation-only; public redistribution
  remains prohibited.

### Legacy DD: raw price is validated; Chemours economics are missing

- Target: `dc6a71ee5440b11b34d08eb014b46d30e20b225111211ca24b8673f1cc781513`
- Security: `US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1`
- Yahoo source/wrapper:
  `2df771bc77ee599e13b1210294a2d957c3160313b1f1ba2048aafe31c17f6b22` /
  `bf7d8dee599335606bf192f13b4720116f6abffe2a723fed06c4b94b5411ad09`
- Frozen-WIKI-backed current raw segment:
  `36bc4a610a64882f576cecff4f73fb9022e19083664e8c3383d75b25e40bc77a`
- Official 2017 merger/identity evidence:
  `098828aa2714df3fdd52a18b1fffb91d6a72865ff8dd4e94e84f7bc079cf0e64`
- Current DD raw OHLC equals WIKI exactly on all `672/672` sessions. Do not
  rewrite any DD OHLC row.
- Current topology: one merger action, zero cash-distribution actions, and 672
  factor rows with both factors equal to `1.0`.
- WIKI contains eleven ordinary dividends plus a rounded `3.2` entry on
  2015-07-01. The `3.2` entry is a proxy for the Chemours stock distribution,
  not a cash dividend. Booking it as cash would double-count economics once the
  child position is created.
- A WIKI factor proxy changes ST1/down-count on four sessions and sell signals
  on two sessions. Current/proxy signal hashes are
  `38c920a8e9efbd9efce4ee4600c30df4316c64f8dfd9f4eb8f96c3000fef63be` /
  `d0276d5bbb381e809a67c9174e985ae902581f02b0d2d77714d0fa026f80fc78`.
- DD was in the S&P 500 from 2015-01-07 through 2017-09-01, making this the
  highest-priority economic-model gap.

Local feasibility is **blocked**. The current source archive has no official
2015 Chemours terms bytes, no CC security, and no identity-bound CC price path.
It does contain a current EODHD exchange-list row for `CC` (Chemours Co, NYSE,
USD, ISIN `US1638511089`) inside
`2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99`,
which is enough to precompute, but not yet publish, the expected security ID:

`US:EODHD:a710ce0e-a20e-5558-ba2b-f2719e2981cf`

The ID must be re-derived as UUIDv5 of `eodhd:US:CC:symbol:CC` and matched to
the fetched identity before apply.

Official external evidence identified but **not locally hash-pinned**:

1. SEC distribution announcement:
   `https://www.sec.gov/Archives/edgar/data/1627223/000119312515215110/d832629dex991.htm`
2. Chemours 2015 Q2 10-Q:
   `https://www.sec.gov/Archives/edgar/data/1627223/000162722315000023/cc-2015630x10q.htm`
3. Issuer basis allocation, required for the ledger's `cost_basis_fraction`:
   `https://s23.q4cdn.com/116192123/files/doc_downloads/Tax-Cost-Basis-Allocation.pdf`

The first two establish one CC share per five DD shares and regular-way CC
trading on 2015-07-01. The third is still required to promote the discovery-only
`0.05085` child / `0.94915` parent basis fractions to exact reviewed values.

#### Executable DD repair plan after evidence intake

1. Archive exact bytes and SHA-256 for all three official documents. Reject the
   repair if the normalized text does not independently prove distribution
   date, ratio, regular-way date, and basis allocation.
2. Minimum EODHD identity/path requests, each attempted once and archived as
   exact bytes: `fundamentals/CC.US`, bounded `eod/CC.US` from 2015-07-01,
   and bounded `splits/CC.US`. A complete action/factor path additionally needs
   bounded `div/CC.US`; without it the child path remains price-only.
3. Expected minimum full-repair archive delta: three official documents plus
   four EODHD responses = **7 `source_archive` rows**, followed by one reviewed
   canonical evidence report row. Every archive ID must equal the exact payload
   SHA-256.
4. Add exactly one `security_master` CC row and one `symbol_history` row,
   effective 2015-07-01, after symbol/exchange/currency/ISIN and expected SID
   all match.
5. Add `N` complete, identity-bound CC `daily_price_raw` rows. Require first
   regular-way session 2015-07-01, no duplicate sessions, valid OHLC, and exact
   reproduction of retained provider bytes. Do not infer `N` before reading the
   archived response.
6. Add one official legacy-DD `spinoff` action: effective/ex-date 2015-07-01,
   ratio `0.2`, new security/SID `CC` /
   `US:EODHD:a710ce0e-a20e-5558-ba2b-f2719e2981cf`, and reviewed basis metadata.
   CC dividend/split rows are separate provider actions.
7. Recompute all 672 DD factor rows and all `N` CC factor rows in a new version.
   The spin-off-only factor changes the 124 DD sessions before 2015-07-01;
   ordinary DD dividends still require exact event evidence before the legacy
   DD total-return series can be called complete.
8. A `special_dividend` surrogate is forbidden. The ledger must create one CC
   share for five DD shares without also crediting cash. Signal continuity
   requires explicit spin-off handling in factor construction; current generic
   factor construction ignores `spinoff`, so this future repair needs a reviewed
   code change rather than a fabricated cash action.

Required DD tests:

- exact-byte/hash and phrase pins for all official and provider artifacts;
- SID derivation, ISIN, currency, exchange, active-date, and raw-session coverage;
- primary-key uniqueness and full repository snapshot validation;
- ledger invariant: five DD shares create one CC share, DD quantity is retained,
  parent plus child basis equals prior basis, and no cash is double-counted;
- factor invariant: DD `split_factor` remains `1.0`; only reviewed economic
  distribution/dividend terms alter `total_return_factor`;
- exact Triple Supertrend difference sessions and hashes for raw and adjusted
  streams, plus S&P portfolio P&L/turnover attribution;
- cache-only cross-validation replay with event mismatches still zero;
- plan/apply idempotency, compare-and-swap conflict, injected-failure rollback,
  and preservation of every unrelated dataset row/version.

### LILA: one Yahoo bar, not a factor repair

- Target: `650c704f17e2f0802949992f52d1d1e25140b9a57b4230619f857aca23207222`
- Security: `US:EODHD:1b6b9beb-42b0-5a06-81f3-23a49627565f`
- EODHD raw: `2d83bb3bc51253905dab9cb15156797f3097fb41e905bf3fff0d60d34aa19bc4`
- Yahoo source/wrapper:
  `c5476ac51d1622259314907e5e12d8499912de921dc799a9efa70bb9c0b7c84f` /
  `dd91a6e535f15042c750fdd0a33f8bc05423a27a17a96d3cbb0a4ed6dcb5569d`
- Official identity:
  `0efad7b02b77a0daefab021c58fdbbb40f03955f069f42eac3e24d403f2813e4`
- Provider split bytes:
  `79d96f165f3c75cc67310e8954cc0829f7266a779bb05a6e1583d384680843a8`
- 2,144 complete overlap sessions. The first regime median Yahoo/EODHD close
  scale is `0.6037199418026842`; the sole maximum deviation is `22.5169683433%`
  on 2018-01-03.
- On that date EODHD O/H/L/C is
  `22.7493/23.5024/22.1819/23.1735`; Yahoo reports the identical value
  `10.84011173248291` for all four OHLC fields. Excluding only this date leaves
  677 sessions with median scale `0.6037199472467316` and maximum relative
  scale deviation `1.1026624227e-5`.
- The other regimes are independently stable: 2020-09-11 through 2026-06-16
  median `0.6796199638444141`, maximum deviation `1.0280522319e-5`; from
  2026-06-17 median `0.9999999949407198`, maximum deviation `4.4795710474e-8`.
- Available Boris, frozen WIKI, and old Yahoo LILA histories end in 2017 and
  cannot arbitrate 2018-01-03. The suspicious shape is strong diagnostic
  evidence against Yahoo, but it is not independent price proof.
- No tracked-index membership. Yahoo-normalized substitution changes ST1 `4`,
  ST2 `5`, ST3 `33`, buy `1`, and sell `5` states.

Safe plan: obtain and archive an issuer-bound third-provider raw response for
2018-01-02 through 2018-01-04. If it supports EODHD, add a target-specific
evidence-only closure with zero data changes. If it supports Yahoo, replace
exactly the one 2018-01-03 raw row and republish all 2,144 factor rows for
lineage even though the split-derived factor values should remain unchanged.
Tests must pin the three-day raw response hash, all four OHLC values and volume,
prove all three scale regimes stable, replay the exact signal sensitivity, and
keep the target fail-closed if the third source is absent or disagrees with both.
The 2020/2026 provider split ratios are a separate action-quality audit and must
not be changed to make this price target pass.

### FWONA and FWONK: exact intermediate regime, ambiguous 2023 economics

Common official identity evidence is the 2017 filing
`6a4fe3ee6fea801819f375c2c4426cfb3b619e659dbe93ae5cfdcfe6d4cc45ce`.
It proves the LMCA/LMCK to FWONA/FWONK identity transition, but contains no 2023
action terms.

FWONA pins:

- Target/security:
  `30abc4bfe0b6fdbb2fe9a13f01b37cd32f03d04bc6fe2793a827417da503c2d7` /
  `US:EODHD:6c98b8f3-f222-5def-92e5-a0633c3f0775`
- Current/LMCA EODHD:
  `7b74e79f8111f3db8e6ea69a1a1118217a4b0e516192b8b61e7b78421a7b3732` /
  `a24b16dbdab994f25eb215c2214d242d487680633e9c08e7e9ad770b1d1edaf4`
- Yahoo source/wrapper:
  `fe5403a7287e7c4d306c66283227aa67308ab0c1596da3ba1c82e82068c1dc1a` /
  `b731f3770212b77d60433b664cfa6642b9538c33eea89e89601c0016ec31b702`
- EODHD dividend/split bytes:
  `ae30b498255ddd27655e57ae035b42c23332baff49fae74d11f03507b26ff015` /
  `07679b58f34de9b42532b453ad6f92886fdcb90fb56a7615e2f34e822996fcfe`
- Current action IDs: cash
  `6da7560019c447e9eb8bbb48bd4d5a269d2dd50f28c2ea7a1bfa9caeb4e79822`
  (`1.25496`, 2023-07-20); split
  `77f291b68de1297e0071cba83718c28a81150b012c9e600318f51c96dedd1f65`
  (`1.018`, 2023-07-20).

FWONK pins:

- Target/security:
  `c3ecd457f56d417fd19a19f650191131efeaba5383784b1468382e4497a1216c` /
  `US:EODHD:8e7e0713-31d7-55a7-8878-74ba653d9090`
- Current/LMCK EODHD:
  `7cf53de85db781770b75931da1e2951dd5ea29d2653cc882963d0f6e4844a1ea` /
  `b4340db7ba9299755d99a7fbe24f8de1b1f22d7505ff4b90c2eb1a7fe9815c68`
- Yahoo source/wrapper:
  `24ab19f9caeba2a564d74f2dc2b485b0003e39d244bf824bea34cd146ec8824a` /
  `4e9b8e33508d4f2216710694b9911cfd7de1d546cdd6c8a6af3124507dda941d`
- EODHD dividend/split bytes:
  `190626fabf8f566e006d097520ff7ebaccc98965908ca654c7899c7b216efea6` /
  `37272e5c65049da58db2ba0335ec8d939e2650c9beb7c6cd17245c5c9ec81eda`
- Current action IDs: cash
  `93b13cc502ff63b47b6e0b4acf1d5f54cbc8af1da58aad5587fc3d9685df3ec2`
  (`1.25373`, 2023-07-20); split
  `f01c3c8810afa874177091d9d4869290005aebbc1a464400388dd132bb5eead3`
  (`1.017`, 2023-07-20).

Exact observed scale phases:

| Security | Dates | Rows | Median Yahoo/EODHD close scale | Maximum deviation |
|---|---|---:|---:|---:|
| FWONA | 2017-01-25..2023-07-19 | 1,631 | `0.9574252159216095` | `3.5956327966e-6` |
| FWONA | 2023-07-20..2023-08-03 | 11 | `0.9746588684899797` | `3.2831206198e-8` |
| FWONA | 2023-08-04..2026-07-15 | 738 | `1.0` | `0.0080048284976` on 2023-08-04 |
| FWONK | 2017-01-25..2023-07-19 | 1,631 | `0.9668477556283082` | `3.3991345023e-6` |
| FWONK | 2023-07-20..2023-08-03 | 11 | `0.9832841744013218` | `5.7542196341e-8` |
| FWONK | 2023-08-04..2026-07-15 | 738 | `1.0` | `0.0004047150395` on 2023-08-04 |

The middle/pre-regime step is `1.0179999986231627` for FWONA and
`1.0170000071648637` for FWONK, numerically matching the stored split ratios.
This proves a staged provider adjustment, not the legal/economic type or date
of a second event. In particular, it is unsafe to manufacture a 2023-08-04
action solely to create another validator regime.

A diagnostic-only in-memory move of each cash row from 2023-07-20 to
2023-08-04 changes 2,161 of 2,899 factor values per security. It changes FWONA
ST2/ST3/down/sell by `3/2/5/1` and FWONK ST1/ST2/down by `1/3/4`, with no FWONK
buy/sell change. These are counterfactual fingerprints, not repair authority.
Both disputes are after Nasdaq-100 removal on 2016-06-10, so portfolio exposure
is zero in the tracked membership window, but data quality must remain
fail-closed.

Safe plan:

1. Archive official 2023 issuer/SEC bytes proving action type, ex/effective,
   record/payment dates, ratio, value, and any basis treatment for both classes.
2. Archive an issuer-bound third raw provider for 2023-07-17 through 2023-08-07.
   FWONA especially requires arbitration of the 2023-08-04 close: EODHD
   `66.21`, Yahoo `65.68000030517578`, while the other Yahoo OHLC fields match
   EODHD within rounding.
3. If official terms confirm the current 2023-07-20 rows, change no action or
   factor; close only through exact third-source/adjustment-basis evidence.
4. If official terms prove a distinct later event, replace/add only the exact
   affected action row per class and recompute all 2,899 factor rows per class.
   Never infer the action type from a Yahoo scale step.
5. Tests must pin official and raw hashes, reproduce all three scale phases,
   arbitrate every disputed OHLC cell, compare class-specific factor regimes,
   pin Triple Supertrend session differences, and prove no change to historical
   Nasdaq-100 portfolio exposure.

## Remaining eight price holds

These have no safe local repair. Every target stays fail-closed until an
independent, issuer-bound raw source is archived.

| Symbol | Target / security | Exact core evidence | Index/backtest relevance and next action |
|---|---|---|---|
| FRCB | `066961f4d693ffa5663979906ca5170b424a3202920f36a37593574d573342be` / `US:EODHD:3453b450-03d1-52ee-9100-cf80856b06ef` | EODHD `3c96be232fb5f567e77a94fe315b67aa61e520d1611842620c04680fb5df6ab3`; Yahoo `0771483cbe4a2533617968f930eb8f536af038f0d64b71636bb3e9ea840148d1`; wrapper `3553f9153dd408c5addb9e7a2d075704ce8395d555e0096be07c96945cdef19a`; reviewed correction `53eed10c5d6a7ccc262215b7848d30efa606a1621d2e793ca21b6002f8a5c298`; relation `878b511b04122e9d9fb09a16888079dd0ace330585e3e838a461162fc928bc6c` | Disputed dates follow S&P removal on 2023-05-04; alternate changes ST2/down 3 and sell 1. Existing EODHD self-correction is not independent arbitration. |
| APC | `a042abc3b784d48e2c5674ee182357e4fb0dc70053d948cb476b10e9856091b9` / `US:EODHD:f485cff6-47f0-5c3f-85ec-1c54895aae21` | EODHD `90eef3bbeb48107348add4e0b4ea648001870b52d33692d7c444cbe82de5e0ab`; Yahoo `e6ee01ca7f8ea98dbc594300f430b1e946020c05b7973e3481570e1ee613b12c`; WIKI extract `2fdbc9bf39054dd0811571ed825417a4742f69a41361a789af64a585ba562e55`; relation `63cacc3b2bcc7e71dfc35f510bbff86140fcad6a6476094c720a8194653d72c7`; Boris `e44acd2e028850f9085977a4994e216ecc426c6735cb77746d28dd4537453f0b` | S&P through 2019-08-09; WIKI changes ST2/down 1. Obtain issuer-bound 2015-11-10 raw close; Boris is adjusted and identity-weak. |
| HOT | `118704d83243ecae0cfb838423558950ae729eceba8503b8ae07cece7fa2758f` / `US:EODHD:3073ffd2-9115-5bf6-8bec-fddcd41749e5` | EODHD `8612a211ec6514a093f6d62a9448c82eb75f45778f244f5a67d6c93ee820a40a`; Yahoo `ac478ed4db8bd8b768a70da774a64e8ac43c1ff6e2bc6e22a5ae79f1ffeaf642`; WIKI extract `9e85b82f0c6fe1138fed54eea99bca884c69b110d5c206270b60d7f28a3f3b81`; relation `4861a53bda386a5c2c6db45817adb5ce429403ba0aa643fcda097652e6c4fa70` | S&P through Marriott acquisition; alternate changes ST1/2/3 `21/41/29`, buy 9, sell 7. Highest third-raw priority. |
| IR | `34467b4f4e7c4d0a0b3b41d1b38bea89a369cdda27c3eaae0acdd401f1b7bb18` / `US:EODHD:cb64587f-5f98-5931-adbf-9804aff1bcf0` | EODHD `2aa4f00c6ca844b4df25f21bb236125129a4528a71e352be33b5e8dbe835b6b5`; Yahoo `00f45d7a4617a4a8f96574bd6c5d729ed510261e44a160b155ae0a288011d729`; WIKI extract `b1d8cdb334fb5156d02adfdf31d6ad1ac9375f6cf08d09fac942397b872f7a7d`; relation `8dcb73e5e3e56499c820aab9e4dc7101da224ab092ec9b67c4acf394838fa1a6`; Boris `ba2356b248e53f993b80d8192d268a9af8522a6eeb3c1f0d1f8864e4ac817600` | Legacy S&P lineage continues as TT; alternate changes ST1 1, ST3 3, down 4. Obtain issuer-labelled raw history. |
| LB | `65f58d24688230cd3a598ee9cc92a2f483db68ace1c27fa6fdc116cdf397097a` / `US:EODHD:e144ef86-76af-5fee-9041-4effc6d321bc` | EODHD `c3e4ce3289d31ee04bd0f97359925bda66a69eb1c0b656644bc57696244b7901`; Yahoo `83bc794550b34bc93c6fd11385881678447d54ba49585a12a38288e257e0226d`; WIKI extract `4e392160964bb55a4585115db2660039f7621a32b8ba67acd8dd30e477de45ed`; relation `030c2b63f62928a561b586cc65dd55ae992d709cb04bfe5e87a6ef57747b9890`; Boris `195c53db12a30976c7c051d8a73b667a2d67c76c61a17d452d94d4e8278bd3f5` | S&P through 2021-08-03; alternate changes ST1/all-up/down 5 and buy 1. High priority legacy L Brands raw history. |
| PCL | `6641ad2fd50e5a4028b06df94f31600c98c43d08e19315eb938907f4abaaa87f` / `US:EODHD:bd9648b7-1b95-5f55-a777-1c7d660cd2db` | EODHD `8c886049acfdd0bf097225bce70de4512e1fb5dabfee6ead3d3d2bd35babbcea`; Yahoo `b0fa62a2b50f09a7009ec3f9af94af637a4cbc3b13120a35ab9e17358378aa52`; WIKI extract `cd716ef088a30b3d3cbdfa36f3eed92287a7825dd87b0e931d8092429db6a466`; relation `83fdf628a2544c7fd32b69d50f2a52f0f9b038a587405803430e88435392a4ea` | S&P through 2016-02-22; alternate changes ST2/down 3 and sell 1. High-priority third raw. |
| POM | `e1762ec53fe21e95d39adf0d225656ab39d22857daaebee109e3de74cfa7aad7` / `US:EODHD:9f2cfe0f-b5b2-5b9e-8685-6fc38307afd3` | EODHD `a3d79c7ffe208cbe3be44ee3f7e7502b39ef54cae4afddb6088f10c4bba695e1`; Yahoo `ab59b1c1328c15026286e7a03b116830395b545d40f79d8fd21549ccea4097d6`; WIKI extract `78ad1a696e69210e0bdbd2850e9c2628b710b7ec9c8cb7a5e8bd4b35cf2edf81`; relation `38025bf218f84db84a40bb7c920e578f1af9141803a827c78a8e6f39d5623084` | S&P through 2016-03-24; alternate changes ST2/ST3/all-up/down 1 and buy 1. High-priority third raw. |
| SPLS | `bdd64d073e1941ac21b8edc31bd542204a1adf85d561408ebac51a068fd17728` / `US:EODHD:591b1e97-ff78-5a6f-806d-0bb7885d2231` | EODHD `add2e21b817013126c8af70207310e579c849987bebee77f9f065d492bb2a649`; Yahoo `eddbafaee6f591e8abf58819a429856efcb14c175148b8cce173e16cf2cf2786`; WIKI extract `9e2161067b3863db1d3e71cbd3ecb59776833d7327361287fa18e4ae68b83ee9`; relation `c6959488b034ab7156f49b960ab0c2a46a1d7b06514b37299cdc79ab865978bd` | S&P and Nasdaq-100 legacy; alternate changes ST1/down 3. Obtain issuer-bound raw history; the current Yahoo ticker is an ETF collision. |

## Work order

1. Acquire and hash-pin DD/CC official evidence and the CC identity-bound path;
   then implement the explicit spin-off ledger/factor model.
2. Archive GPN Eikon plus WIKI evidence and close GPN without changing data.
3. Acquire third raw histories for HOT, LB, POM, and PCL because alternate
   evidence changes buy/sell state during tracked membership.
4. Acquire official 2023 terms and bounded third raw data for FWONA/FWONK;
   do not infer action rows from scale regimes.
5. Acquire the three-day LILA arbiter and issuer-bound APC/IR/SPLS/FRCB raw
   evidence. Lower portfolio priority does not relax the data-accuracy gate.
