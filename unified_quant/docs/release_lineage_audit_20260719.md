# Release lineage and source-archive audit

- Audited release: `20260715-20260718T183729772858Z`
- Audit mode: local read-only inspection; no provider, EODHD, R2, or network access
- Audit date: 2026-07-19 KST

## Executive result

1. The release pointer and all nine dataset pointers are internally consistent. The immutable release bytes equal `releases/current.json`; all nine pointer versions, manifest paths, manifest hashes, manifest identities, and latest manifest files validate.
2. The release is **not release-exact for adjustment lineage**. `adjustment_factors` still names the ECA/QVCAQ corporate-action version, while the release now names the later FRC OCC corporate-action version.
3. The exact SIVB and FRC raw OCC PDF bindings are intact. Each action row, source-archive row, gzip object, raw byte count, and raw SHA-256 agrees with the reviewed PDF.
4. Full source-archive verification is blocked by 14 rows whose raw digest equals `source_hash` but whose `archive_id` is a different composite identity hash.
5. All 203 non-provider action rows with SHA-256 provenance have an archive binding. Provider action provenance is incomplete: 426 rows, covering 19 unique provider hashes, have neither a source-archive row nor a payload object.
6. The lifecycle finalizer contains the correct post-merge factor rebuild, but cannot currently reach it: the CELG/BMYRT preservation precheck rejects the already-known stale input lineage before the rebuild runs.

## Pointer and manifest inventory

| Dataset | Current/release pointer | Latest files valid | Logical rows |
|---|---:|---:|---:|
| `adjustment_factors` | yes | yes | 2,097,679 |
| `corporate_actions` | yes | yes | 24,087 |
| `daily_price_raw` | yes | yes | 2,097,679 |
| `index_constituent_anchors` | yes | yes | 605 |
| `index_membership_events` | yes | yes | 795 |
| `lifecycle_resolutions` | yes | yes | 177 |
| `security_master` | yes | yes | 877 |
| `source_archive` | yes | yes | 2,048 |
| `symbol_history` | yes | yes | 902 |

Every latest manifest reports zero unresolved actions and zero conflicts. This proves physical pointer/file consistency, not the semantic lineage and archive invariants discussed below.

## P0: stale adjustment-factor lineage

Release inputs:

- `daily_price_raw`: `eca-qvcaq-transition-20260715-e7d95a0a245744b59a357169156ad32c-daily_price_raw`
- `corporate_actions`: `frc-occ-52352-20260715-9c669b9d1b944d9196b12323080f7f32-corporate_actions`
- Therefore expected factor lineage:
  `eca-qvcaq-transition-20260715-e7d95a0a245744b59a357169156ad32c-daily_price_raw+frc-occ-52352-20260715-9c669b9d1b944d9196b12323080f7f32-corporate_actions`

Observed factor manifest and every factor row instead use:

`eca-qvcaq-transition-20260715-e7d95a0a245744b59a357169156ad32c-daily_price_raw+eca-qvcaq-transition-20260715-e7d95a0a245744b59a357169156ad32c-corporate_actions`

Reproduced failures:

- `repair_us_eca_qvcaq_transitions._assert_release_factor_lineage` raises `RuntimeError: Already-repaired adjustment-factor manifest is not release-exact.` The check is at `unified_quant/scripts/repair_us_eca_qvcaq_transitions.py:2008-2036`.
- The cross-validation release gate independently enforces the same manifest and row-level invariant at `unified_quant/src/supertrend_quant/market_store/cross_validation.py:2999-3017`.

This is a hard pre-publication blocker even though the two OCC repairs changed only provenance fields and not corporate-action economics.

## P0: finalizer cannot yet repair the stale lineage

The finalizer's intended output is correct: after merging corporate actions, it rebuilds every factor against the planned corporate-action version at `unified_quant/scripts/finalize_us_lifecycle_coverage.py:4773-4783`.

The actual call order blocks that repair:

1. The candidate loop invokes `_preserved_exact_repair_resolution` before the rebuild at `finalize_us_lifecycle_coverage.py:4627-4643`.
2. A CELG or BMYRT candidate routes to `_preserve_exact_celg_resolution` at `finalize_us_lifecycle_coverage.py:3905-3934`.
3. That function requires the *input* factor manifest and all factor rows to already name the current corporate-action version at `finalize_us_lifecycle_coverage.py:2949-2983`.
4. On the audited release, the direct read-only call fails with `RuntimeError: Exact BMYRT adjustment-factor manifest lineage is stale.`
5. The SIVB-specific preservation function itself succeeds on the audited release; the blocking precheck is the shared CELG/BMYRT preservation path.

Required sequencing decision before finalization:

- either perform a safe factor-only full rebuild first; or
- make the preservation precheck verify exact economic rows/keys while allowing this precisely identified provenance-only input staleness, then require the newly built output lineage to be release-exact.

Simply skipping output lineage validation would be unsafe.

## Exact 14-row archive mismatch inventory

The existing full verifier at `unified_quant/scripts/publish_and_verify_r2.py:420-458` reads every object, decompresses it, hashes the raw bytes, and requires `digest == archive_id == source_hash`. It stops on the first row below. A complete read-only scan found 2,034 valid rows and these 14 mismatches. There were no missing or unreadable objects, and in every mismatch the raw SHA-256 exactly equals `source_hash`.

| Dataset | `archive_id` | `source_hash` / actual raw SHA-256 | Bytes | URL |
|---|---|---|---:|---|
| `eodhd_div` | `62b1df178614e3b8c5a9bc0f4cac8438946cc8c297c985ecd7be0e6ea14bc423` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | 2 | `https://eodhd.com/api/div/FRCB.US?from=2023-05-03&to=2026-07-15` |
| `eodhd_eod` | `8e3667ac5f9a407a63bca8e9ac134f11523530051111571d5a47ba9e2e1a78d1` | `3c96be232fb5f567e77a94fe315b67aa61e520d1611842620c04680fb5df6ab3` | 91,529 | `https://eodhd.com/api/eod/FRCB.US?from=2023-05-03&to=2026-07-15` |
| `eodhd_ovv_div` | `84811457cb719ddcf2756efd41922aace51c682a0a9abc52a997e1cada86fdaf` | `63d125e117f9eeb8dcfb65833216553b46010f91abec2992ccc5e28c290f7fa6` | 4,665 | `https://eodhd.com/api/div/OVV.US?from=2020-01-27&to=2026-07-15` |
| `eodhd_ovv_eod` | `89d477dee0ed285cb326e4737b42fe4b685e875ac1bac4019c2df22bea9f3c8c` | `2911e9b1eb3e59f3649f1a7ccef3b3a62b6b2667ed910aca8e335001afceafca` | 187,316 | `https://eodhd.com/api/eod/OVV.US?from=2020-01-27&to=2026-07-15` |
| `eodhd_ovv_splits` | `b40ae5d1ee136ea2c15dbdcfeb1a443ea28286916aaedb6d3a4c7d1960b303de` | `195def5749f8d07f7311576b9470a2cf2c22a8b866bb2e263763311e828793b5` | 52 | `https://eodhd.com/api/splits/OVV.US?from=2020-01-27&to=2026-07-15` |
| `eodhd_qvcaq_div` | `985e539ff612e98c26f81b0dd4ac05387cfcaa91191de5eb4544a24e7dcfd22d` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | 2 | `https://eodhd.com/api/div/QVCAQ.US?from=2026-04-24&to=2026-07-15` |
| `eodhd_qvcaq_eod` | `cdb868a6b94fd620ac01a66f5bd44ca56ecb357a3f9e3d47851a6898d388453d` | `66a03be49bab3e158b6133fb2e49897008e90acbd4629ff11c812d6ee46f76aa` | 5,998 | `https://eodhd.com/api/eod/QVCAQ.US?from=2026-04-24&to=2026-07-15` |
| `eodhd_qvcaq_splits` | `954f03350a7b7f7342286c80c058b171e2776ac9b4dadca8de842008debf0e4e` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | 2 | `https://eodhd.com/api/splits/QVCAQ.US?from=2026-04-24&to=2026-07-15` |
| `eodhd_splits` | `c38253725941d3e78dec01244016d2769d048e5d544c739869fc434eb2bbad29` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | 2 | `https://eodhd.com/api/splits/FRCB.US?from=2023-05-03&to=2026-07-15` |
| `frcb_reviewed_ohlcv_envelope_correction` | `5de7d46d9fd1d7a1f2674a41826f4592d6786a59bb0e9891143aadc6974588c1` | `53eed10c5d6a7ccc262215b7848d30efa606a1621d2e793ca21b6002f8a5c298` | 684 | `https://eodhd.com/api/eod/FRCB.US?from=2023-05-03&to=2026-07-15` |
| `occ_reviewed_memo_extraction` | `c568a6ac21ddc05d3c5821c228b94b7bd7e52a602a96b1cfb2f5f08ee24af658` | `377bcc0663eb9666b9f639edaf541b0ec729fd4b84ac876e345baaa9bf413668` | 516 | `https://infomemo.theocc.com/infomemos?number=52352` |
| `ovintiv_issuer_reorganization` | `5550528f1a63b94d35e6880c9c046756a64390f6675a65e2bf728dc560166474` | `cb6cdb670b3a30d38f0529d242f4ea470052c04204e3101537627f7df3955bef` | 236,338 | `https://investor.ovintiv.com/2020-01-24-Encana-Completes-Reorganization-and-Establishes-Corporate-Domicile-in-the-U-S` |
| `qvc_issuer_stock_cost_basis` | `57dd1d48a3580b65a68633138740f0dc261654bfd0fea4b25509dcf29ca7397b` | `55829c9064eee534b6f79027648172494a507f8b9be16e9598dc57cdd58c165b` | 65,806 | `https://investors.qvcgrp.com/investors/stock-cost-basis` |
| `sec_rule_provision_notice` | `3958dc0304e1449a9fd3e33d538877eddd70053c897b61e0fb6666555e05967c` | `58d199861b620211b63c846e3184baf1ff7982adb124e085c5f726e2fd06af59` | 1,474 | `https://www.sec.gov/Archives/edgar/data/876661/000087666120000056/ruleprovisionnotice.htm` |

The first five FRC-family rows were produced by the FRC/PARA repair; the remaining nine OVV/QVCAQ/official rows were produced by the ECA/QVCAQ repair. Four request records intentionally share the two-byte `[]` payload hash `4f53...` and the same object path.

## Is `archive_id == source_hash` the real invariant?

**Judgment: yes for the current publishable repository contract.** The schema declaration is under-specified, but the system-level read/write contract is consistently content-addressed:

- Core ingestion sets `archive_id = artifact.source_hash` and names the object with that hash at `unified_quant/src/supertrend_quant/market_store/ingest.py:801-825`.
- Direct SEC archive replay requires `archive_id == source_hash == SHA-256(raw bytes)` at `unified_quant/scripts/collect_us_lifecycle_actions.py:147-190`.
- Cross-validation evidence lookup treats the evidence SHA as `archive_id` and requires `source_hash == archive_id` at `cross_validation.py:2829-2852`.
- R2 preflight verifies every archive object with the same invariant at `publish_and_verify_r2.py:420-458`.
- Numerous repair/finalization paths also look up an archive row by the evidence content hash.

`schemas.py:213-224` only declares `archive_id` as the primary key and does not express its digest relationship, so the schema should eventually make the contract explicit. That omission does not make the verifier a stray legacy assumption.

The divergent writers are:

- ECA/QVCAQ: `archive_id = SHA256(source | source_url | source_hash)` at `repair_us_eca_qvcaq_transitions.py:1422-1438`.
- FRC/PARA: `archive_id = SHA256(source | source_hash)` at `repair_us_frc_para_lifecycle.py:1375-1395`.

Those writers solve a real modeling problem: identical payload bytes such as `[]` can come from several endpoints, while a content hash cannot be both a unique object ID and a one-row-per-request ID. However, they changed one writer's semantics without migrating readers, validators, tests, or the schema. Relaxing only the R2 verifier would leave cross-validation and other hash lookups inconsistent.

Safe choices are therefore:

1. Preserve the current content-addressed contract: canonicalize `archive_id` to `source_hash`, deduplicate identical objects, and store per-request endpoint provenance in a separate request/binding relation or explicitly structured metadata; or
2. Perform a deliberate schema migration in which `archive_id` becomes record identity and `source_hash` remains the content digest, then update every reader, evidence lookup, integrity validator, and publication fingerprint together.

For the current release, option 1 is the lower-risk path. Do not merely weaken `_verify_archive_payloads`.

## Exact SIVB/FRC OCC status

These two raw PDF bindings are not among the 14 mismatches:

- SIVB OCC 52179: action state `raw_pdf_bound`, archive state `present`, 566,940 raw bytes, SHA-256 `28f25f761b61e7f898fa7d1c237520a1a4fed97e88af0b788ef52256608c6035`, two pages.
- FRC OCC 52352: action state `raw_pdf_bound`, archive state `present`, 566,923 raw bytes, SHA-256 `0ab8f0c49076f97049dcde80078affda258108ce7c904241bf1eba46b44aec66`, two pages.

The FRC *legacy reviewed-extraction JSON* is one of the 14 composite-ID rows. The raw FRC PDF itself is exact.

## Additional provider-action provenance gap

Comparing each corporate-action SHA-256 against both `source_archive.archive_id` and `source_archive.source_hash` found:

- non-provider rows: 203 bound, 0 unbound;
- provider rows: 23,458 bound, 426 unbound;
- missing provider payload identities: 19 unique hashes;
- no matching payload filenames anywhere below `data/cache/archives` for those 19 hashes.

The missing hashes cover 405 EODHD dividend rows, four EODHD split rows, and 17 rows labelled `derived_ticker_identity`. This is not the immediate cause of the existing R2 archive verifier failure, because the verifier only validates rows that are already in `source_archive`; it is nevertheless a provenance-completeness blocker if every provider-derived action is required to be replayable from immutable raw bytes.

## Required pre-publication order

1. Resolve the 14-row archive-ID semantic conflict without weakening only one validator.
2. Decide and enforce the provider-action raw archive completeness policy for the 19 missing hashes.
3. Repair the finalizer sequencing or perform a factor-only rebuild so the lifecycle finalizer can run.
4. Rebuild all adjustment factors and require manifest plus row lineage to name the final `daily_price_raw` and `corporate_actions` versions.
5. Regenerate lifecycle and cross-validation reports against that exact release.
6. Run the local release validator and full source-archive verifier again before any R2 access.
