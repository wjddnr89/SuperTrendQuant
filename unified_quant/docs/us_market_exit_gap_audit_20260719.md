# WIN / CHK / FTR / ENDP market-exit gap audit (2026-07-19)

Scope: local release `20260715-20260718T230255094849Z`, local SEC cache and
release source archive only. No HTTP, EODHD, R2, or dataset write was performed.
The reproducible report is produced by
`unified_quant/scripts/audit_us_market_exit_gaps.py`; its current canonical JSON
SHA-256 is
`404217e79a2db65ad44a7cfef787a376b51d343a1b02f44b4f53abdab2649eb0`.

## Findings

| Legacy symbol | Stored tail | Official market exit and legal cancellation | Reorganized/security relationship | Current conflict | S&P 500 / Nasdaq-100 Triple Supertrend impact | Safe disposition |
|---|---|---|---|---|---|---|
| WIN | 1,118 rows; 2015-01-02..2020-07-10. Last positive volume is 2019-06-28; the last six rows are sparse, flat, zero-volume values. | A 2020-07-30 filing proves that the common was already on OTC Pink. The local official bytes do **not** prove the first OTC date or ticker. Legacy common was cancelled for zero distribution on 2020-09-21. | New Windstream Holdings II LLC units went to creditor classes; they are not a public-price successor for legacy common holders. | `WIN/NASDAQ` history remains open and the terminal date is driven by six stale rows, not a proved trading boundary. | Removed from S&P 500 on 2015-04-07; no later ADD/anchor. Any repair limited to the questioned post-2019 tail changes zero modeled trades/equity. | Keep fail-closed/degraded. Do not invent a ticker/date. Archive official transition evidence plus an independent OTC tail before changing identity or prices. |
| CHK | 1,381 rows; 2015-01-02..2020-06-26; no rows at or after the OTC start. | NYSE suspended trading before market on 2020-06-29; CHKAQ began OTC Pink trading on 2020-06-30; NYSE removal was 2020-07-31. Legacy equity was cancelled for zero distribution on 2021-02-09. | CHKAQ is the same legacy equity. Reorganized CHK, stored under `US:EODHD:97548dea-74f0-55a8-b906-47d5c2a072e1`, is distinct and begins 2021-02-10. | Legacy master/history says `NASDAQ`, stays open, omits CHKAQ/OTC, and overlaps a distinct reorganized CHK symbol interval. The legal zero-distribution action itself is correct. | Removed from S&P 500 on 2018-03-19; no later ADD/anchor. Adding only the CHKAQ tail changes zero modeled trades/equity. | Close legacy CHK/NYSE, add CHKAQ/OTC on the same old SID, load independently archived CHKAQ prices, and never bridge returns to reorganized CHK. |
| FTR | 1,340 rows; 2015-01-02..2020-04-29. Last positive volume is 2020-04-23; 2020-04-24, 27, 28 and 29 are flat 0.26 with zero volume. | Nasdaq suspended at open on 2020-04-24. The 2020-04-17 filing anticipated `FTRQ`; by 2020-05-01 an official filing confirmed actual OTC trading as `FTRCQ`. Nasdaq removal was 2020-05-08. Legacy common was cancelled for zero distribution on 2021-04-30. | FTRCQ is the same legacy common. Reorganized FYBR is distinct new common, approved to begin Nasdaq trading around 2021-05-04. | Master/history says `NYSE`, remains open, lacks the OTC interval, and treats four post-suspension placeholders as prices. FYBR is absent from this release, which is acceptable for legacy-holder economics but incomplete globally. | Removed from S&P 500 on 2017-03-20; no later ADD/anchor. Deleting/replacing only post-2020 rows changes zero modeled trades/equity. | Delete the four placeholders, model Nasdaq-to-OTC on the same old SID, and load an independently archived FTRCQ tail. Do not treat anticipated `FTRQ` as an observed price identity or join FYBR returns. |
| ENDP | 1,970 rows; 2015-01-02..2022-10-27. Exactly 44 stored rows begin on the official OTC transition date. | Nasdaq suspended at open and ENDPQ began OTC trading on 2022-08-26. Form 25-NSE filing on 2022-09-14 is mentioned, but the exact exchange-removal effective date is not bound by the local transition object. Legacy equity cancellation is 2024-04-23. | ENDPQ is the same legacy Endo International plc equity. Endo, Inc. was newly formed without the predecessor's participation and is not a shareholder-return successor. | All 44 post-2022-08-26 rows remain labeled `ENDP/NASDAQ`; the ENDPQ tail after 2022-10-27 is absent and the symbol interval stays open. | Removed from Nasdaq-100 on 2016-07-18 and S&P 500 on 2017-03-02; no later ADD/anchor. Rebinding/supplementing the later OTC tail changes zero modeled trades/equity. | Rebind the 44 rows to ENDPQ/OTC on the same old SID and independently supplement through cancellation. Never join new Endo, Inc. prices. |

## Data-quality conclusion

All four remain `dataset_repair_required` for the strict provider/price gate.
Their current zero-distribution cancellation actions and applied lifecycle
resolutions are legally plausible, but those events occur long after the stored
terminal price because the same-security OTC continuation is missing or
mislabelled. The existing exact terminal-readiness exceptions are safe only as
degraded, index-scoped exceptions: all four were removed from the modeled
indices years before the disputed tails and have no later membership re-entry.
They must not be generalized into a date tolerance or used to declare the price
histories provider-validated.

## Pinned local official evidence

- WIN OTC-status filing cache SHA-256:
  `4709354a7e186519638633ade9cc3652cf88c9608cbdd287acfb7b8123cc0bf9`;
  cancellation archive SHA-256:
  `656d5eebc149b51f53a0bb48bc3de4547f5b53986f97f84e4cdc679fc4bca125`.
- CHK suspension/CHKAQ filing cache SHA-256:
  `f1dc291d2ba3b9e420f9c3e973c3bb622ef54f74597a474dbd1c7515039f263d`;
  NYSE Form 25-NSE SHA-256:
  `4aee77582bc25e6707f20b4a903f2e95f9d2ec15b7436926ce458153140bf2d8`;
  cancellation archive SHA-256:
  `80f610bb05f197ef740bae2b23c03af96786118ff08dc23a4c78038a577c4842`.
- FTR suspension/anticipated-label filing cache SHA-256:
  `0f48e56a3f066e6800e4b5e9940d7b03353cad96f870cd8554e7661ce8cca08e`;
  confirmed FTRCQ filing cache SHA-256:
  `32de36ada64faa871090e95248f871016b1c70d1b36016730c285e2b5b2ed7d9`;
  Nasdaq Form 25-NSE SHA-256:
  `87b825a5c34d16eb07423b26d258060acc01d463d11c50f4ccb3cdbf3ab110d8`;
  cancellation archive SHA-256:
  `b581e1ff7cb90e9abf699236ca40f6ccc4b7233fbbb587129bd355d5f23f76cc`.
- ENDP suspension/ENDPQ filing cache SHA-256:
  `1c6adf64a35e70602ccca19cfbb957f5abcc8808d7def6ba45eb8f76c49ba971`;
  cancellation/successor archive SHA-256:
  `5669e33821e77bf97bd722bf909f649ad10caebb91973f00a2b053515a5c1377`.
