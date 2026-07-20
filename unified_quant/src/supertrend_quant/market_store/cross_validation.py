"""Fail-closed validation of archived market-data cross-checks.

The collector is intentionally separate from this module.  Publication only
needs deterministic local reads: one immutable report row, its exact archived
JSON payload, and the archived official/provider evidence named by that payload.
The pinned OLD LILA/LILAK overlap is explicitly not claimed to have a disclosed
independent upstream; that limitation is itself part of the validated contract.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import math
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse

import exchange_calendars as xcals
import pandas as pd
import yaml

from .lifecycle import canonical_lifecycle_event_id
from .lifecycle_coverage import lifecycle_candidate_id
from .manifest import DataRelease, sha256_bytes
from .official_lifecycle_evidence import (
    OfficialLifecycleExceptionEvidenceSpec,
    load_official_lifecycle_exception_evidence,
)
from .reviewed_price_evidence import (
    REVIEWED_PRICE_EVIDENCE_BASIS,
    TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256,
    TRUSTED_REVIEWED_PRICE_EVIDENCE_TARGET_IDS,
    build_reviewed_price_projection,
    reviewed_price_evidence_inventory_sha256,
    reviewed_price_evidence_registry,
    reviewed_price_evidence_sha256,
    verify_reviewed_price_projection,
)
from .reviewed_remaining_price_exceptions import (
    REVIEWED_REMAINING_PRICE_EXCEPTION_BASIS,
    TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_INVENTORY_SHA256,
    reviewed_remaining_price_exception_inventory,
    validate_reviewed_remaining_price_exception,
)
from .source_archive_price_evidence import (
    REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS,
    TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256,
    TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS,
    WIKI_DOWNLOAD_URL,
    WIKI_EXTRACT_SHA256,
    WIKI_PROVENANCE_SHA256,
    source_archive_price_only_inventory_sha256,
    source_archive_price_only_registry,
    source_archive_price_only_spec_sha256,
    verify_source_archive_price_only_evidence,
)
from .wiki14_price_evidence import (
    REVIEWED_WIKI14_PRICE_ONLY_BASIS,
    TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256,
    TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS,
    WIKI14_DOWNLOAD_URL,
    WIKI14_PROVENANCE_SHA256,
    verify_wiki14_price_only_evidence,
    wiki14_price_only_inventory_sha256,
    wiki14_price_only_registry,
    wiki14_price_only_spec_sha256,
)
from .terminal_policy_exceptions import (
    TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTION_EVENT_IDS,
    TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTIONS_SHA256,
    reviewed_terminal_policy_action_mismatches,
    reviewed_terminal_policy_exception_inventory_sha256,
    reviewed_terminal_policy_exception_sha256,
    reviewed_terminal_policy_exceptions,
    reviewed_terminal_policy_release_warning_mismatches,
    reviewed_terminal_policy_report_mismatches,
)
from .validation import index_member_identity_gap_fingerprint
from .yahoo_chart import (
    ALLOWED_US_EXCHANGE_NAMES,
    US_EXCHANGE_TIMEZONE,
    normalize_yahoo_symbol,
    parse_yahoo_chart_json,
    parse_yahoo_chart_no_data_evidence,
)


CROSS_VALIDATION_SCHEMA = "us_lifecycle_cross_validation/v7"
CROSS_VALIDATION_DATASET = "cross_validation_reports"
INDEPENDENT_PRICE_PROVIDER = "yahoo_chart"
INDEPENDENT_PRICE_HOST = "query1.finance.yahoo.com"
INDEPENDENT_PRICE_ENDPOINT = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
)
YAHOO_NO_DATA_TERMINAL_ACTION_TYPES = frozenset(
    {"delisting", "cash_merger", "stock_merger", "ticker_change"}
)
YAHOO_NO_DATA_TERMINAL_DATE_RELATIONS = frozenset(
    {
        "event_on_terminal_session",
        "event_on_next_xnys_session",
        "legal_event_in_sessionless_gap_before_next_xnys",
        "stored_non_session_boundary_then_next_xnys_event",
        "event_session_after_last_trade",
    }
)
YAHOO_NO_DATA_SUCCESSOR_VALIDATION_BASIS = (
    "exact_official_successor_identity_interval_with_passed_price_overlap"
)
REVIEWED_NO_DATA_SUCCESSOR_CHAIN_BASIS = (
    "code_pinned_official_finite_chain_to_passed_price/v1"
)
REVIEWED_PERMANENT_EXCEPTION_NO_DATA_BASIS = (
    "code_pinned_permanent_lifecycle_exception_no_data/v1"
)
REVIEWED_NO_DATA_UNSUPPORTED_PATH_BASIS = (
    "code_pinned_official_unsupported_trading_path_no_data/v1"
)
REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_BASIS = (
    "code_pinned_reviewed_nonterminal_same_sid_ticker_no_data/v1"
)
# Permanent lifecycle exceptions are already identity/date/source pinned by the
# separately fingerprinted official-evidence registry.  The price gate may
# recognize that exact reviewed result as an explicit no-data limitation; it
# must never reinterpret it as a passed price comparison.
PERMANENT_EXCEPTION_NO_DATA_CODE = "permanent_lifecycle_exception_no_data"
UNSUPPORTED_PATH_NO_DATA_CODE = "official_unsupported_trading_path_no_data"
TRUSTED_REVIEWED_PERMANENT_EXCEPTION_NO_DATA_TARGET_IDS = frozenset(
    {
        "28ecd1a7f4224e6d0bbe1db6b77030131d16edd952c182f2afd36cf32ddcb2af",
        "7a86baed1c01c95f89ac3235093e04947d5af4c362692237ecc51da57fcfc046",
        "8e4019d9e30a1697a498ab984b1dfc65cac75f8023e7fd30e5fb91a2fff8865d",
        "932963b415c64528e31076c5b39807c1799c3835cc9b8d000999fce4df7b8f67",
        "a1a391246a313a058341bfae9a17b5cabb6c94d2d930672e8a06414c869f3ec5",
    }
)
# BMYRT is the sole reviewed unsupported-path case.  It contains one official
# retrospective valuation mark, not a provider OHLCV path, and is outside the
# index universes.  Its exact YAML projection is independently pinned here so a
# generic date/calendar bypass cannot be introduced by configuration alone.
TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_TARGET_IDS = frozenset(
    {"abfebadbc76b0c17e2e76e4190c3a45a75b71a02dcf47d7c8a39d42d5f0f465d"}
)
TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_SHA256 = (
    "be4947649c79882ab9c0f3ce482373455634943223ee3ad2cce8a0baa1f6158f"
)
# This inventory is intentionally narrower than the reviewed nonterminal
# action registry.  It authorizes Yahoo no-data handling only for the closed
# same-security ticker intervals reviewed below.  It does not create, or stand
# in for, a terminal lifecycle resolution.
TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SPECS = {
    "fb3d264732079815004e26780f47e9c816133970ad35ab903054fa5c97406a48": {
        "event_id": "fb3d264732079815004e26780f47e9c816133970ad35ab903054fa5c97406a48",
        "source_target_id": "ef89f10e83177128247e7b62c97338dba1c62fdce831e5840631075781afa79d",
        "successor_target_id": "5554726d8670f59fd104028a7ef71ab5ea3d2abbd89b2c77c0354d3a3090dc8d",
        "security_id": "US:EODHD:f5daeed5-d1a2-5279-aa49-8c06c902b97f",
        "old_symbol": "ARNC",
        "successor_symbol": "HWM",
        "old_active_from": "2016-11-01",
        "old_active_to": "2020-03-31",
        "successor_active_from": "2020-04-01",
        "effective_date": "2020-04-01",
        "official_source_hash": "99bf852ed888154abdd1754398c1cc33dba31ddc6aee6bb5912c56df22ff24ee",
        "reviewed_extraction_sha256": "0121bd4918ff07fbab92be65b4ca12bd5546e83e3804aeb39266b573d2cb0ec5",
    },
    "fc556f24050c3205150b7934f431b72d6348ab5fbfad3e85bfbb149c7b9781bd": {
        "event_id": "fc556f24050c3205150b7934f431b72d6348ab5fbfad3e85bfbb149c7b9781bd",
        "source_target_id": "76cfddc97b878414119dfd9db08e356216cffc4ddc2839188451df534e11296f",
        "successor_target_id": "9648613a55f30697b2d3bb6893a3526b7d582169fc2dca21b6e2e4c9e481e1b6",
        "security_id": "US:EODHD:e9eea478-61d8-5762-9f5b-fbdfd69a02a3",
        "old_symbol": "SYMC",
        "successor_symbol": "NLOK",
        "old_active_from": "2015-01-01",
        "old_active_to": "2019-11-01",
        "successor_active_from": "2019-11-04",
        "effective_date": "2019-11-04",
        "official_source_hash": "87a584813a438f76e5cee9ae800678771bc2df5ed6f3e50461273c1849026e18",
        "reviewed_extraction_sha256": "2f2cf8effa9cc40fdac0f51edad9f3857af1892ba232b05f6132f8228749715e",
    },
    "2df5c4c0298e5ff531aaa785146a20cba98d22080c970eabbd841b802ec60e7e": {
        "event_id": "2df5c4c0298e5ff531aaa785146a20cba98d22080c970eabbd841b802ec60e7e",
        "source_target_id": "1529c06b9d2cc60317b5ba93bb28edc60665e08f0b47759d9fd7587cd1d10b16",
        "successor_target_id": "417f2715f9774e20fa90f071b452f7c40d2fd3aaa5e5b0c3a0debef8dc0b212e",
        "security_id": "US:EODHD:30662d16-c6e4-5187-9721-2b23ac10e4d0",
        "old_symbol": "FI",
        "successor_symbol": "FISV",
        "old_active_from": "2023-06-07",
        "old_active_to": "2025-11-10",
        "successor_active_from": "2025-11-11",
        "effective_date": "2025-11-11",
        "official_source_hash": "d4cd0c2f981bfd0be14d2ebccfc8e852a94177e5fba86abe2c027c5510fc07d3",
        "reviewed_extraction_sha256": "09a9b72d468c9ef25f0bce0291b8eb1493657dde22d69c242d832d14a20d6554",
    },
    "8f6dd7b99d5cc344bb60449f3536979a54dd1737f35e9b16b99f795b1d271dc5": {
        "event_id": "8f6dd7b99d5cc344bb60449f3536979a54dd1737f35e9b16b99f795b1d271dc5",
        "source_target_id": "eceadd8f599c26f54b6bc40f0147cedd912a06a976e422fcd744b9ed66982de7",
        "successor_target_id": "4fe4dc952f4791e86d4251ffecb2dbe99f31d7990f71713afa96c2c06ed3b48d",
        "security_id": "US:EODHD:3234e888-8ab8-5985-b09c-b4bb40a3ddc2",
        "old_symbol": "ABC",
        "successor_symbol": "COR",
        "old_active_from": "2015-01-01",
        "old_active_to": "2023-08-29",
        "successor_active_from": "2023-08-30",
        "effective_date": "2023-08-30",
        "official_source_hash": "87aba717e7a0808699b57c6e8bccaea1933a5f645ceb45760ea6b71fe4ecb4cf",
        "reviewed_extraction_sha256": "2fbaf5a6fa59773c0b6ac3adbe7010226349d4527a09cbf9b8343cec44874ea3",
    },
    "958ed869cc179ffda932c0012af35439ea22b21b07691fb6c5e221844cb0a0ed": {
        "event_id": "958ed869cc179ffda932c0012af35439ea22b21b07691fb6c5e221844cb0a0ed",
        "source_target_id": "e8a1c89244a6f605eb465c4ab4d49c0e7ff96d792d6696eaafbadfcc68f9a4fe",
        "successor_target_id": "2b025b50241eff6b6d3fa48f9072b04fd8297605710302562157d08f7d36a4b8",
        "security_id": "US:EODHD:5eca6dac-4c4c-50af-9fc2-17c5839a4efc",
        "old_symbol": "QVCGA",
        "successor_symbol": "QVCAQ",
        "old_active_from": "2025-02-24",
        "old_active_to": "2026-04-23",
        "successor_active_from": "2026-04-24",
        "effective_date": "2026-04-24",
        "official_source_hash": "55829c9064eee534b6f79027648172494a507f8b9be16e9598dc57cdd58c165b",
        "reviewed_extraction_sha256": "4f7c0f7f64f269f62f7cb93f282ffb6cded8541139cd9500f7183d2e2afb5864",
    },
    "47235ed0f22108df208fefcab63d0bb9118c5ecb58345387b1a50431e7bc388c": {
        "event_id": "47235ed0f22108df208fefcab63d0bb9118c5ecb58345387b1a50431e7bc388c",
        "source_target_id": "5fc4ea842f47f1919b4c1c390e69f858953c1c7f4416f029de98d6010af8c31b",
        "successor_target_id": "1421e436f15f5c4d8832bb6feebf78eca4a4d58561afc73a93f9acbeecfd08fb",
        "security_id": "US:EODHD:6a76982a-782c-5b73-abd3-8c86f47d3a1f",
        "old_symbol": "GDI",
        "successor_symbol": "IR",
        "old_active_from": "2017-05-12",
        "old_active_to": "2020-02-28",
        "successor_active_from": "2020-03-02",
        "effective_date": "2020-03-02",
        "official_source_hash": "6a7fac2b87f5f445343e95545e33fee1f528b493c2dcc6467e868b7a107af074",
        "reviewed_extraction_sha256": "9ae8080af0d847add0585a271ecab7e74c7a05d825d61a5dbb4f566a9b3d7462",
    },
    "4a662e7caca7ed147c918e5907187b6890397ca72b9b8a2e06e7ee411cedbd7c": {
        "event_id": "4a662e7caca7ed147c918e5907187b6890397ca72b9b8a2e06e7ee411cedbd7c",
        "source_target_id": "f74256440c631c6cc35cc1fccbc5927a4c3aa9bc6bbc0731f0920f7cc5ff6bef",
        "successor_target_id": "ad7bd4aed99d6b50ab40eeff15e793c380ff11af02e7e63bfd93765cb715605c",
        "security_id": "US:EODHD:8102c8e2-e5d1-5331-a987-4692d29da477",
        "old_symbol": "VIP",
        "successor_symbol": "VEON",
        "old_active_from": "2015-01-01",
        "old_active_to": "2017-03-30",
        "successor_active_from": "2017-03-31",
        "effective_date": "2017-03-31",
        "official_source_hash": "cb257624d286b531e891aed1a9c21f0a2c1fef92023331433cdbb8b0434416aa",
        "reviewed_extraction_sha256": "77dd2c1b5ab80ef18c095ab69978e64f13e6c5e8e9d4999d124e509d70306db5",
    },
    "5c67b30c00cf201d6248706eabb50a89f50312ff11b445e4a612af76168d4cbf": {
        "event_id": "5c67b30c00cf201d6248706eabb50a89f50312ff11b445e4a612af76168d4cbf",
        "source_target_id": "746efe7a4e27638baad5fc57bc17e09b78f2471bbe1e54cba2dd97761be2ba5e",
        "successor_target_id": "d254e908c6f7b7a67854b03022115ed4ba931b5e4207f120af41f66051bdac3b",
        "security_id": "US:EODHD:89fe6d28-737c-5b16-82e6-c1207561311c",
        "old_symbol": "FBHS",
        "successor_symbol": "FBIN",
        "old_active_from": "2015-01-01",
        "old_active_to": "2022-12-14",
        "successor_active_from": "2022-12-15",
        "effective_date": "2022-12-15",
        "official_source_hash": "2c2703ed8949f1d72ceea49e655005cd39165a8020b1750c71894d185d987135",
        "reviewed_extraction_sha256": "ead6a571ef8399af9446760da8d60d83117037da52e84b56c8196f9ada56bf86",
    },
    "3df08f0e3e4593c773a5cddf9d7ff1abc46b017110516d1fb1dc65b3d89dbd43": {
        "event_id": "3df08f0e3e4593c773a5cddf9d7ff1abc46b017110516d1fb1dc65b3d89dbd43",
        "source_target_id": "e8a6309d5fd8f74308c83b2bbeabdcd493e1bc3284f40c6d5ce5916570b12937",
        "successor_target_id": "af7a5f93e9f258403ef18cfa423f48711071a3e6b7979f4e68ccc023daa42a13",
        "security_id": "US:EODHD:97548dea-74f0-55a8-b906-47d5c2a072e1",
        "old_symbol": "CHK",
        "successor_symbol": "EXE",
        "old_active_from": "2021-02-10",
        "old_active_to": "2024-10-01",
        "successor_active_from": "2024-10-02",
        "effective_date": "2024-10-02",
        "official_source_hash": "5112367c6043776743c2532071d2d857d77faae96c8317f77af0aa0c8e9259b1",
        "reviewed_extraction_sha256": "aac50db97258c6a6cb5194517eeb0afae83b8b37f3528cf1201186f02a1ab068",
    },
    "a066f9db433eb3bce0365744b09de62e7c10a64d9d89eabed22b3ec359963718": {
        "event_id": "a066f9db433eb3bce0365744b09de62e7c10a64d9d89eabed22b3ec359963718",
        "source_target_id": "2584232860aff99d29bd3553d9f8f0872fd49f7359410d01fbac3095476ae5b2",
        "successor_target_id": "5e5949741c0186142ffd243804ef3acf50e08f6a79e9fd7f4e96e1736d41ef3c",
        "security_id": "US:EODHD:a1542ac4-30f6-57dc-bf2f-79c0ea6aefd2",
        "old_symbol": "BHGE",
        "successor_symbol": "BKR",
        "old_active_from": "2017-07-05",
        "old_active_to": "2019-10-17",
        "successor_active_from": "2019-10-18",
        "effective_date": "2019-10-18",
        "official_source_hash": "b69b9c819e8ec2b2188a32aeb7edb54b98228d97a150fb692789d2f5dd5b5421",
        "reviewed_extraction_sha256": "700b7dc884627c18e220bf55af60f9d93d9ca2963ce0c92e08af72033e3b0001",
    },
    "350960af29b81ec304e10cc318837f2e24c70ce2f89983bad95df38ad7f66cda": {
        "event_id": "350960af29b81ec304e10cc318837f2e24c70ce2f89983bad95df38ad7f66cda",
        "source_target_id": "41c49f30d3fc35262e0ae9e880f48fd2270495880e1186431675148084d82ce1",
        "successor_target_id": "89ae740db02af5d4361a11c27fb5658dcfa042ffd173b610abb9ab9f0b8c022b",
        "security_id": "US:EODHD:dc3f4283-a3cc-5bc7-916c-9ffdd71c9874",
        "old_symbol": "WYN",
        "successor_symbol": "WYND",
        "old_active_from": "2015-01-01",
        "old_active_to": "2018-05-31",
        "successor_active_from": "2018-06-01",
        "effective_date": "2018-06-01",
        "official_source_hash": "cd00a425ef8881c2683632e5443ee4f5d4d59dab46332d506a6ea1f2fdba26ec",
        "reviewed_extraction_sha256": "6a49cbf442aaac6ae0b11a9f2cd6874ce7e7105e9c30cea57386ac82cc18be06",
    },
    "31281f82fe09566d70782ba37514ea57e1da1a6915b68ed14c56a9569832a53e": {
        "event_id": "31281f82fe09566d70782ba37514ea57e1da1a6915b68ed14c56a9569832a53e",
        "source_target_id": "89ae740db02af5d4361a11c27fb5658dcfa042ffd173b610abb9ab9f0b8c022b",
        "successor_target_id": "04dc5c10e0331a10edbaba8c47dc224af90b3cbe9f571dfdb7cadd968f45a5e7",
        "security_id": "US:EODHD:dc3f4283-a3cc-5bc7-916c-9ffdd71c9874",
        "old_symbol": "WYND",
        "successor_symbol": "TNL",
        "old_active_from": "2018-06-01",
        "old_active_to": "2021-02-16",
        "successor_active_from": "2021-02-17",
        "effective_date": "2021-02-17",
        "official_source_hash": "71c464267f0f6d51eece7a2b7d55cc9b422dd689d82f758c7475d563f99ac0f2",
        "reviewed_extraction_sha256": "5ae8aee782a9af6aebd66b319cc999aca961df6770383548af47d5954f3fc79d",
    },
    "6cdae488bbfad53d85b79752afcb2c54c5b19b41b9f55800cb8ab4db51901d50": {
        "event_id": "6cdae488bbfad53d85b79752afcb2c54c5b19b41b9f55800cb8ab4db51901d50",
        "source_target_id": "3e5c46bd8f6290b79c0bffec55ab97345577e371d2e3e990184b1be8aae20c98",
        "successor_target_id": "30bf8cdc6139b20bdbdc76bbbb8d7a7ebf2b5ae40d1464fbd0c600db3c388cbf",
        "security_id": "US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3",
        "old_symbol": "DWDP",
        "successor_symbol": "DD",
        "old_active_from": "2017-09-01",
        "old_active_to": "2019-06-02",
        "successor_active_from": "2019-06-03",
        "effective_date": "2019-06-03",
        "official_source_hash": "ae9343609e64dcd8421f11462b8782cc8db38a130e03c983714f3c10ba8db311",
        "reviewed_extraction_sha256": "c3f61acc3697b854cab1080b023465ad8706010e88b43a7ebd04d4bba65c962e",
    },
}
TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SHA256 = (
    "a52154daf044535aec7d159f9629c0fa99f4a0570adbaae671de66a675a3ce82"
)
# A no-data target may not inherit another no-data exception generically.  These
# are the only reviewed roots for which every intermediate official transition,
# immutable Yahoo response and final passed price target is pinned below in the
# policy and independently pinned here in code.
TRUSTED_REVIEWED_NO_DATA_SUCCESSOR_CHAIN_ROOT_TARGET_IDS = frozenset(
    {
        "1f01bc0c30fbdc4487eb9c59da93009ae7011f87c795ebb200995ee9a870d759",
        "216524a2eee30741484210db94a57a4b14ab7f93707136b78df6267875ab66ff",
        "25e55d92b46ef655ff5c58668dc96c51bf54f4e4587b4a147cf0312b5d2af26e",
        "2d362833156ca2bb1051ec6e3bdd2b31d53efe60fb899aebf207d37ae8b8845b",
        "41c49f30d3fc35262e0ae9e880f48fd2270495880e1186431675148084d82ce1",
        "68b3dae375d615c0b2055f7ac4c991f6793ca7f1fbee107d7127c4c0ad04f2d4",
        "72aa68de77973dc79bc38a9d9a8dd94bf3a18fd40ac6adb5f5b485f1ffd0c9bb",
        "76cfddc97b878414119dfd9db08e356216cffc4ddc2839188451df534e11296f",
        "81e5768be6adec5ee6d4c86e4c735bb60177b53f9ad74c8ba56c74dd7f7db5e4",
        "ed476fc5c82bb4e87e2a13df3da9585e04d7e4a4d702265966e962fcf60f0d9d",
        "f067891eacb63f1f608117430f7ce70179deed2532f92616f1e88de04044a492",
        "f9d91f6df227097a90e57cebbbb08e3c7c5e083dd8ecc1642ec3bf5e2a0625f8",
        "0c7ccbea602b6ae66d806f0f13edfc3034b14fd7ab49b98bf8e7667b6d0be110",
        "4ab52d92c2c23f0103bd7b20979c943223107b3ff72d5ca3d42e5717ccf5bb10",
        "cd1e97410c98f59bfb065a2f3642cb602b77241b1bc5dd0428631f0f0ff80e31",
        "db0b71658e5be84e59ce757b46b9c150d8d8af4e768b3fbb37e2d2f7191d3204",
    }
)
TRUSTED_REVIEWED_NO_DATA_SUCCESSOR_CHAINS_SHA256 = (
    "c80314e3f28d2c99ba0fce7b83088130ed324c6c4970d098e5b12e8b3e987a87"
)
# These observed no-data targets are deliberately outside the finite-chain
# allowlist. They contain a cycle, terminal dead end, or an unrepaired identity
# boundary and cannot gain chain inheritance from configuration alone.
BLOCKED_NO_DATA_SUCCESSOR_CHAIN_TARGET_IDS = frozenset(
    {
        "0559abf3b5c53f9ef1d97049535ac26e3dc8012816741a27d9aad6d4986e1943",
        "1acf5880447d9823963a27deae4b38608c9637a14f2079901f58962dc5413bac",
        "1ea053f6a05564df7b2828325d0a6c1447a8d869dd5e0e2b42197f6328974b5a",
        "530c856446f4bdcb109d0419f58a9c6d6c026d68c4e053dd136c73e095c94551",
        "69cb8012e50ecce9963a0664c1623a306a30b9b33e4115b68216e2b4694adb2d",
        "8e4019d9e30a1697a498ab984b1dfc65cac75f8023e7fd30e5fb91a2fff8865d",
        "a9a0d3192c39eba70cb0f216fc057875c1a00f9383c35f7ef88ddfba223c46a0",
        "c33f3c60ba68a59b84e740131e3b748315bfc2f1c637efa323e46242f04a77e6",
        "d06aa00a80b62169e36e130e01bf61dd8686a44b3b6e8fcf7be064be1abc910a",
        "da9bcd14a22df24a61f7a198866a23f5d15c021c7d52d235fa9433866ef15200",
    }
)
TERMINAL_EVENT_VALIDATION = "terminal_resolution_report"
NONTERMINAL_EVENT_VALIDATION = "nonterminal_official_provenance"
PINNED_EXTERNAL_OVERLAP_VALIDATION = "pinned_external_overlap"
PERMANENT_EXCEPTION_VALIDATION = (
    "permanent_lifecycle_exception_official_provenance"
)
PERMANENT_EXCEPTION_CODES = frozenset(
    {"unsupported_consideration", "recovery_uncertain"}
)
DEFAULT_OFFICIAL_LIFECYCLE_HINTS = (
    Path(__file__).resolve().parents[3] / "configs/us_lifecycle_hints.yaml"
)
# Publication must not trust a changed YAML registry merely because the
# generated report and release were rehashed together.  This is the reviewed
# fingerprint of every exact permanent-exception identity/date/claim/URL/SHA
# binding in ``us_lifecycle_hints.yaml``.  Adding, pinning, or changing any
# exception requires an explicit code update here as well.
TRUSTED_PERMANENT_EXCEPTION_REGISTRY_SHA256 = (
    "6eb5c0af209d73d893872661ae1e9215212a9095a30ec1a160661c0aced268da"
)
REVIEWED_NONTERMINAL_EXTRACTION_FIELDS = (
    "event_id",
    "security_id",
    "action_type",
    "effective_date",
    "new_security_id",
    "new_symbol",
    "ratio",
    "cash_amount",
    "currency",
    "source_kind",
    "source_url",
    "source_hash",
)
# The R2 gate has no authority to trust a manifest merely because it was
# embedded in a newly rehashed report.  This fingerprint is the code-reviewed
# v7 inventory in configs/us_cross_validation.yaml; adding or changing an
# extraction therefore requires an explicit code review on both sides.
TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256 = (
    "eaa4c8f8f791f153f90b9fbbd47c9bbae5148039429c70fece689fcdb0543b54"
)
# A terminal override is deliberately narrower than a normal reviewed event:
# it may only replace the collector's price-level eligibility heuristic after
# the corporate action, parsed lifecycle report, and archived official source
# already agree exactly.  The two-event inventory is code pinned so a report or
# policy rewrite cannot authorize another terminal event.
TRUSTED_REVIEWED_TERMINAL_OVERRIDE_EVENT_IDS = frozenset(
    {
        "52e8663611264e84d2b91d4c2eb5fd8346f001086987649d097950a420e66c05",
        "94b7da742aa70fd546532862fdab23dd9bcc15b0c48efb7efdbde1f66d378630",
    }
)
TRUSTED_REVIEWED_TERMINAL_OVERRIDES_SHA256 = (
    "bf023f5c600046da8cda780a266a585c625029595c7195dbccff4d5822668bce"
)
REVIEWED_TERMINAL_MARKET_DATE_CORRECTION_FIELDS = (
    "event_id",
    "superseded_event_id",
    "candidate_id",
    "security_id",
    "symbol",
    "action_type",
    "report_effective_date",
    "official_completion_date",
    "effective_date",
    "ex_date",
    "announcement_date",
    "payment_date",
    "last_price_date",
    "date_relation",
    "allowed_report_mismatches",
    "new_security_id",
    "new_symbol",
    "ratio",
    "cash_amount",
    "currency",
    "source_kind",
    "source_url",
    "source_hash",
    "report_source_url",
    "report_source_hash",
    "lifecycle_evidence_report_sha256",
    "filing_accession_number",
    "filing_date",
)
# These are exact semantic market-transition corrections, not relaxations of
# the LILA/LILAK price heuristic.  Each new action ID, superseded collector ID,
# legal/market dates, action/report SEC URL+SHA pairs, terms and candidate are
# code pinned independently.  This shape also supports a future reviewed
# terminal-tail correction without weakening the one-session market rule.
TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTION_EVENT_IDS = frozenset(
    {
        "39caca00fbedb7b23e8b5294cdf3bdc5aba9014f98bf72b6a6da341602651192",
        "951960822646a8f972002508d162bd37a3ab3aaec42349fbff37caa1919b7b51",
        "ea791beeaa569aaf19b1df04e59df3710d45382cd70e5d82078d7247286272c6",
    }
)
TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTIONS_SHA256 = (
    "9ef9740b7377db9f89fab1aa0ea97455357c5faf6374fb64fcea644bd1890b6a"
)
# SIVB/SIVBQ is a dedicated, fail-closed provenance path.  It is intentionally
# not added to the broad official-host allowlists: only these two exact actions,
# their immutable SEC/EODHD/OCC objects, and the independently reviewed raw OCC
# 52179 bytes may pass.  The legacy JSON extraction remains archived for audit
# history but has no authority to replace the raw PDF.
TRUSTED_SIVB_EVIDENCE_BINDING_EVENT_IDS = frozenset(
    {
        "01419d978e03e608512e4e898e695fdb39953278b08dc8138d97e0d0e21e4caa",
        "f8cbecc851776e882ac85872972ac5c1680a672138276c1a9301d532c2d8ad3f",
    }
)
TRUSTED_SIVB_EVIDENCE_BINDINGS_SHA256 = (
    "9f783e6cbc1a80cf54cbe692774a7fe575a8b3caeacc6ac52a82e0539b195902"
)
# FRC/FRCB has the same narrow trust requirement for OCC memo 52352.  The
# ticker action is authoritative only when it points at the independently
# reviewed raw PDF; the older deterministic JSON extraction is retained solely
# as non-authoritative audit history.  This is deliberately not a broad OCC
# host allowlist.
TRUSTED_FRC_EVIDENCE_BINDING_EVENT_IDS = frozenset(
    {
        "e351f774b133eae45d49e0fbe60215e5bbceec540c3386076f4c3f2b6c57d9ea",
    }
)
TRUSTED_FRC_EVIDENCE_BINDINGS_SHA256 = (
    "0f5c2d3a656c9f99a3c1391dc54c68e8b7f4c631fe77e89f9bfb1fdca4e4c39e"
)
# NTCO/NTCOY is also a dedicated, fail-closed provenance path.  Neither OCC,
# Cboe nor BNY is added to a broad official-host allowlist: only the exact
# same-security ticker action, terminal cash action and immutable reviewed
# derived/raw objects below can pass.  The lifecycle report remains mandatory
# for the terminal event; this binding cannot manufacture or replace it.
TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS = frozenset(
    {
        "0bd231d216016d33c8d6eb1ec9bb85450a21b9298c4abda34a3399727e9b2c00",
        "d1eeadf8f4779212ff4b3162af1731bf7bfc804c10756cc4bb8074c307e1f746",
    }
)
TRUSTED_NTCO_EVIDENCE_BINDINGS_SHA256 = (
    "7e93dc8229f459a64154a61d843315e1b7b3550057ab27ab70326f59a99b4eef"
)
REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_FIELDS = (
    "symbol",
    "security_id",
    "old_candidate_id",
    "candidate_id",
    "old_event_id",
    "event_id",
    "action_type",
    "report_candidate_active_to",
    "report_candidate_last_price_date",
    "report_crosscheck_old_price_session",
    "report_effective_date",
    "official_completion_date",
    "last_real_session",
    "market_transition_session",
    "date_relation",
    "new_security_id",
    "new_symbol",
    "ratio",
    "cash_amount",
    "raw_source_url",
    "raw_source_hash",
    "raw_source_bytes",
    "removed_tail_start",
    "removed_tail_end",
    "removed_tail_count",
    "removed_tail_sha256",
    "official_source_url",
    "official_source_hash",
    "official_source_bytes",
    "filing_accession_number",
    "filing_acceptance_datetime",
    "successor_source_hash",
    "index_removals_observed",
    "lifecycle_evidence_report_sha256",
    "registry_item_sha256",
)
# The two exact repair planners emit these five reviewed terminal boundaries.
# Publication trusts neither the YAML nor those manifests alone: the combined
# policy inventory and each planner-specific embedded registry are code-pinned,
# and every archived SEC/EOD payload, removed tail and release row is
# re-attested.
TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS = frozenset(
    {
        "dd616ed974225aa84c3b537b9d21ce2a27e13a1f5af8c7e2488b765a007cac31",
        "5607a776f99741c085a54e45eddc90282d7d7fe5fe86a3b8cab2350bb7188ca7",
        "162dee2832998354021e79efcafa8a915d3c2f0744b939c5f570c93deb2f1a55",
        "2aa7c18ca6ac8f0e4680a7e5456a04ba2f401fabfb2cc7dc0a5326e298f71176",
        "d82975bc819ca47d10c7b2e2ca963422629980682933a4ee13b355fe564e6344",
    }
)
TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256 = (
    "6e90ae5c50fe4428d2afa3fbac1efb31812388b247b818abac8e7f8a9b9333ce"
)
_TRUSTED_LEGACY_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS = frozenset(
    {
        "dd616ed974225aa84c3b537b9d21ce2a27e13a1f5af8c7e2488b765a007cac31",
        "5607a776f99741c085a54e45eddc90282d7d7fe5fe86a3b8cab2350bb7188ca7",
        "162dee2832998354021e79efcafa8a915d3c2f0744b939c5f570c93deb2f1a55",
    }
)
_TRUSTED_LEGACY_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256 = (
    "119a19f588748671ca3ac2a344fd2863bc79b1c7a1680a6d88513bbe35ccb734"
)
_TRUSTED_SHORT_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS = frozenset(
    {
        "2aa7c18ca6ac8f0e4680a7e5456a04ba2f401fabfb2cc7dc0a5326e298f71176",
        "d82975bc819ca47d10c7b2e2ca963422629980682933a4ee13b355fe564e6344",
    }
)
_TRUSTED_SHORT_TERMINAL_PRICE_TAIL_REGISTRY_SHA256 = (
    "df25fae0e153ea722a326638764f8a7ceeba6d080ee68307fe75b15f599030bc"
)
# These gates are an overlay, not another broad exception family.  They freeze
# the complete semantic projection of the 18 terminal rows audited after the
# lifecycle finalizer: action, applied resolution, candidate/report semantics,
# exact archive rows and (for SIVBQ) its identity-bound verified hint.  The
# aggregate lifecycle-report hash remains release binding, but can never pass
# one of these events without all event-local hashes also matching.
REVIEWED_TERMINAL_EVENT_GATE_FIELDS = (
    "event_id",
    "candidate_id",
    "security_id",
    "symbol",
    "policy_code",
    "action_sha256",
    "resolution_sha256",
    "report_semantic_sha256",
    "archive_ids",
    "archive_binding_sha256",
    "hint_key",
    "hint_sha256",
    "lifecycle_evidence_report_sha256",
)
TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_POLICY_CODES = {
    "162dee2832998354021e79efcafa8a915d3c2f0744b939c5f570c93deb2f1a55": (
        "provider_tail_market_transition/v1"
    ),
    "2093c4a169a10534ac01ff370ec37aaf240cf6a26bc32c9c0746b89cbe8281d9": (
        "terminal_close_before_legal_completion/v1"
    ),
    "25bce725b19ce21cebac0fa09351a30e5b89479256f7d1ed25f9218b557754c4": "direct_report_exact/v1",
    "2aa7c18ca6ac8f0e4680a7e5456a04ba2f401fabfb2cc7dc0a5326e298f71176": (
        "provider_tail_market_transition/v1"
    ),
    "350bb85a7395ef9272e5f2867afdd4e523c99c258752120977ec1f35e36a2c8a": (
        "abmd_nontradeable_cvr_lower_bound/v1"
    ),
    "39caca00fbedb7b23e8b5294cdf3bdc5aba9014f98bf72b6a6da341602651192": (
        "legal_completion_before_market_transition/v1"
    ),
    "42934ff153b211e42af214f866f5bc4e9e6b9a020168b5a6488030a9e929b8af": (
        "terminal_close_before_legal_completion/v1"
    ),
    "50f8bfb2bb620c136dd9f3ce8699d049d5c394f3b2a447cb316019f55f6351f7": (
        "terminal_close_before_legal_completion/v1"
    ),
    "5607a776f99741c085a54e45eddc90282d7d7fe5fe86a3b8cab2350bb7188ca7": (
        "provider_tail_market_transition/v1"
    ),
    "7b987d3143d81db664d272eabe464941bcf22c05b0cae715662741ea0a304636": "eca_dual_provenance/v1",
    "825ea0640b20da42dcfa1c516ff921f272b0fd0a0fd4020de509674832391806": (
        "legal_completion_before_market_transition_missing_report_successor/v1"
    ),
    "8bf08a94b34187174e451f5f9ec549ba9a4f15f9d658005e9a3bc10d45820131": (
        "bmyrt_official_exit_mark/v1"
    ),
    "cb355a88e767bd5f557350ddf9c13f1b324da6e8f96c622a2e1f8eeea01fa36a": (
        "celg_next_session_cvr_delivery/v1"
    ),
    "d82975bc819ca47d10c7b2e2ca963422629980682933a4ee13b355fe564e6344": (
        "provider_tail_market_transition/v1"
    ),
    "dd616ed974225aa84c3b537b9d21ce2a27e13a1f5af8c7e2488b765a007cac31": (
        "provider_tail_market_transition/v1"
    ),
    "ea791beeaa569aaf19b1df04e59df3710d45382cd70e5d82078d7247286272c6": (
        "legal_completion_before_market_transition/v1"
    ),
    "f553d393e8bda37561276fec20d5b9bce5f722609e466e96bc9e199c624891c1": (
        "para_no_election_default_stock/v1"
    ),
    "f8cbecc851776e882ac85872972ac5c1680a672138276c1a9301d532c2d8ad3f": (
        "sivbq_verified_legal_cancellation/v1"
    ),
}
TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_EVENT_IDS = frozenset(
    TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_POLICY_CODES
)
# Filled only after reviewing the normalized 18-row YAML registry.
TRUSTED_REVIEWED_TERMINAL_EVENT_GATES_SHA256 = (
    "0e4e2d0c0129d4b85a1dbe9b36b833afabb482ed98d0d5ec63dc5220a5fc9536"
)
TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS = {
    "dd616ed974225aa84c3b537b9d21ce2a27e13a1f5af8c7e2488b765a007cac31": {
        "index_id": "sp500",
        "replay_date": "2020-10-07",
        "security_id": "US:EODHD:3dd6d6ce-e7a1-5078-b258-df5b18404c9d",
        "next_remove_event_id": (
            "59d17bfad7dceb1c4903d45cc083841209982df638ec0f18006f2d3a7987d12d"
        ),
        "next_remove_effective_date": "2020-10-12",
        "next_remove_source": "community_sp500_history",
        "next_remove_source_hash": (
            "39a9202c9ef69a74c0ff07e2113ad41fb6da7c8c5b6cd9541f0185fb4391e717"
        ),
        "fingerprint": (
            "989c5d44ef1b8cf8a682d807b63a62ebe3c3f38eb6f57e6314b3fe381d5c7d04"
        ),
    },
}
TRUSTED_PINNED_EXTERNAL_OVERLAPS: dict[str, dict[str, Any]] = {
    "LILA": {
        "active_from": "2015-07-02",
        "active_to": "2017-12-29",
        "primary_source": "yahoo_chart_adjusted_basis_primary",
        "primary_source_url": (
            "https://query1.finance.yahoo.com/v8/finance/chart/LILA"
            "?period1=1434931200&period2=1514764800&interval=1d&events=history"
        ),
        "external_source": "boris_kaggle_cc0_v3",
        "external_source_url": (
            "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/"
            "price-volume-data-for-all-us-stocks-etfs/Data%2FStocks%2Flila.us.txt"
            "?datasetVersionNumber=3"
        ),
        "external_source_sha256": (
            "9885111c20ca809ce8791c429cd8eb66a62470b53ab71f7c2ac6a573d576f73c"
        ),
        "raw_rows": 599,
        "overlap_start": "2015-07-02",
        "overlap_end": "2017-11-10",
        "overlap_sessions": 597,
        "primary_sessions": 630,
        "uncrosschecked_tail_sessions": 33,
        "minimum_return_correlation": 0.995,
        "maximum_p99_scaled_close_error": 0.05,
        "upstream_provider_disclosed": False,
        "independent_provider_claimed": False,
        "license": "CC0: Public Domain",
        "license_url": (
            "https://www.kaggle.com/datasets/borismarjanovic/"
            "price-volume-data-for-all-us-stocks-etfs/versions/3"
        ),
    },
    "LILAK": {
        "active_from": "2015-07-02",
        "active_to": "2017-12-29",
        "primary_source": "yahoo_chart_adjusted_basis_primary",
        "primary_source_url": (
            "https://query1.finance.yahoo.com/v8/finance/chart/LILAK"
            "?period1=1434931200&period2=1514764800&interval=1d&events=history"
        ),
        "external_source": "boris_kaggle_cc0_v3",
        "external_source_url": (
            "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/"
            "price-volume-data-for-all-us-stocks-etfs/Data%2FStocks%2Flilak.us.txt"
            "?datasetVersionNumber=3"
        ),
        "external_source_sha256": (
            "b5a56cc0c1b5a478354d85149c2370ccde6146f7f43d94566dcc76382db610e4"
        ),
        "raw_rows": 599,
        "overlap_start": "2015-07-02",
        "overlap_end": "2017-11-10",
        "overlap_sessions": 597,
        "primary_sessions": 630,
        "uncrosschecked_tail_sessions": 33,
        "minimum_return_correlation": 0.995,
        "maximum_p99_scaled_close_error": 0.05,
        "upstream_provider_disclosed": False,
        "independent_provider_claimed": False,
        "license": "CC0: Public Domain",
        "license_url": (
            "https://www.kaggle.com/datasets/borismarjanovic/"
            "price-volume-data-for-all-us-stocks-etfs/versions/3"
        ),
    },
}
VALIDATED_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "lifecycle_resolutions",
    "adjustment_factors",
)


def independent_provider_source_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify Yahoo-derived rows even when one provenance field is incomplete."""

    mask = pd.Series(False, index=frame.index, dtype=bool)
    if "source" in frame.columns:
        mask |= (
            frame["source"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.startswith(INDEPENDENT_PRICE_PROVIDER)
        )
    if "source_url" in frame.columns:
        hosts = frame["source_url"].fillna("").astype(str).map(
            lambda value: (urlparse(value).hostname or "").lower()
        )
        mask |= hosts.eq(INDEPENDENT_PRICE_HOST)
    return mask


def provider_affected_identity_ids(
    master: pd.DataFrame,
    prices: pd.DataFrame,
) -> set[str]:
    """Return Yahoo-backed identities and every identity reusing their ticker."""

    required_master = {"security_id", "primary_symbol"}
    required_prices = {"security_id"}
    _require(
        required_master.issubset(master.columns),
        "security_master lacks identity columns required for provider scope.",
    )
    _require(
        required_prices.issubset(prices.columns),
        "daily_price_raw lacks security_id required for provider scope.",
    )
    direct_ids = {
        str(value).strip()
        for value in prices.loc[independent_provider_source_mask(prices), "security_id"]
        if pd.notna(value) and str(value).strip()
    }
    if not direct_ids:
        return set()
    symbol_by_id = {
        str(row["security_id"]).strip(): str(row["primary_symbol"]).strip().upper()
        for row in master.to_dict(orient="records")
        if str(row.get("security_id", "")).strip()
    }
    reused_symbols = {
        symbol_by_id[security_id]
        for security_id in direct_ids
        if symbol_by_id.get(security_id)
    }
    return {
        str(row["security_id"]).strip()
        for row in master.to_dict(orient="records")
        if str(row.get("security_id", "")).strip()
        and str(row.get("primary_symbol", "")).strip().upper() in reused_symbols
    } | direct_ids


def canonical_json_bytes(value: Any) -> bytes:
    """Stable bytes used for policy/report hashes and source archiving."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def deterministic_source_archive_id(
    source: str,
    source_url: str,
    source_hash: str,
) -> str:
    """Return the reviewed source-envelope archive ID used by newer repairs."""

    return sha256_bytes(
        f"{_text(source)}|{_text(source_url)}|{_text(source_hash).lower()}".encode()
    )


def source_archive_binding_matches(
    row: Mapping[str, Any],
    *,
    source: str,
    source_url: str,
    source_hash: str,
) -> bool:
    """Match exact provenance and either supported deterministic ID scheme."""

    digest = _text(source_hash).lower()
    expected_ids = {
        digest,
        deterministic_source_archive_id(source, source_url, digest),
    }
    return bool(
        _text(row.get("source")) == _text(source)
        and _text(row.get("source_url")) == _text(source_url)
        and _text(row.get("source_hash")).lower() == digest
        and _text(row.get("archive_id")).lower() in expected_ids
    )


def _trusted_sivb_evidence_binding_rows() -> list[dict[str, Any]]:
    """Return the complete reviewed SIVB action/evidence inventory."""

    common_evidence = [
        {
            "role": "sec_market_raw",
            "dataset": "sec_edgar_filing",
            "source": "sec_edgar_filing",
            "content_type": "text/html",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/719739/"
                "000119312523073665/d485308d8k.htm"
            ),
            "source_hash": (
                "69f3b20dfab4c9c43641a3c38a99f288129665af40e5ae3e6993ec36ccf4fcef"
            ),
            "object_path": (
                "archives/2026-07-15/"
                "69f3b20dfab4c9c43641a3c38a99f288129665af40e5ae3e6993ec36ccf4fcef.html.gz"
            ),
            "effective_date": "2026-07-15",
            "content_bytes": 33250,
            "raw_payload": True,
        },
        {
            "role": "eodhd_otc_prices_raw",
            "dataset": "eodhd_eod",
            "source": "eodhd_eod",
            "content_type": "application/json",
            "source_url": (
                "https://eodhd.com/api/eod/SIVBQ.US?from=2023-03-28&to=2024-11-08"
            ),
            "source_hash": (
                "038c5a1ab7a5b439835a12507ebacc8bd8342ba73005479a0c57acc60ff04a1f"
            ),
            "object_path": (
                "archives/2026-07-15/"
                "038c5a1ab7a5b439835a12507ebacc8bd8342ba73005479a0c57acc60ff04a1f.json.gz"
            ),
            "effective_date": "2026-07-15",
            "content_bytes": 44932,
            "raw_payload": True,
        },
        {
            "role": "occ_memo_52179_raw_pdf",
            "dataset": "occ_information_memo",
            "source": "occ_information_memo",
            "content_type": "application/pdf",
            "source_url": "https://infomemo.theocc.com/infomemos?number=52179",
            "source_hash": (
                "28f25f761b61e7f898fa7d1c237520a1a4fed97e88af0b788ef52256608c6035"
            ),
            "object_path": (
                "archives/2026-07-15/"
                "28f25f761b61e7f898fa7d1c237520a1a4fed97e88af0b788ef52256608c6035.pdf.gz"
            ),
            "effective_date": "2026-07-15",
            "content_bytes": 566940,
            "raw_payload": True,
        },
        {
            "role": "occ_memo_52179_legacy_reviewed_extraction",
            "dataset": "occ_reviewed_memo_extraction",
            "source": "occ_reviewed_memo_extraction",
            "content_type": "application/json",
            "source_url": "https://infomemo.theocc.com/infomemos?number=52179",
            "source_hash": (
                "11e986df7ea010021dd343353e24da8009673053b555b32cce95d4aae1598c6f"
            ),
            "object_path": (
                "archives/2026-07-15/"
                "11e986df7ea010021dd343353e24da8009673053b555b32cce95d4aae1598c6f.json.gz"
            ),
            "effective_date": "2026-07-15",
            "content_bytes": 659,
            "raw_payload": False,
        },
    ]
    rows = [
        {
            "event_id": (
                "01419d978e03e608512e4e898e695fdb39953278b08dc8138d97e0d0e21e4caa"
            ),
            "security_id": "US:EODHD:181d58e2-2b82-5f5d-8c33-311c5d7b5129",
            "action_type": "ticker_change",
            "effective_date": "2023-03-28",
            "ex_date": "2023-03-28",
            "announcement_date": "2023-03-27",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "US:EODHD:181d58e2-2b82-5f5d-8c33-311c5d7b5129",
            "new_symbol": "SIVBQ",
            "official": True,
            "action_source_kind": "official_crosscheck",
            "action_source": "occ_information_memo",
            "action_source_url": "https://infomemo.theocc.com/infomemos?number=52179",
            "action_source_hash": (
                "28f25f761b61e7f898fa7d1c237520a1a4fed97e88af0b788ef52256608c6035"
            ),
            "action_metadata_sha256": (
                "086360be7d6b0642b95121e8b78f3b23beff00ed5b1ee6adf2a6da3840607b81"
            ),
            "evidence": [dict(item) for item in common_evidence],
        },
        {
            "event_id": (
                "f8cbecc851776e882ac85872972ac5c1680a672138276c1a9301d532c2d8ad3f"
            ),
            "security_id": "US:EODHD:181d58e2-2b82-5f5d-8c33-311c5d7b5129",
            "action_type": "delisting",
            "effective_date": "2024-11-08",
            "ex_date": "2024-11-08",
            "announcement_date": "2024-11-08",
            "payment_date": "",
            "cash_amount": 0.0,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "official": True,
            "action_source_kind": "official_crosscheck",
            "action_source": "sec_edgar+stored_price_crosscheck",
            "action_source_url": (
                "https://www.sec.gov/Archives/edgar/data/719739/"
                "000119312524254186/d904756d8k.htm"
            ),
            "action_source_hash": (
                "14371aef1566bfdcda9ca3171b1ced46d095adf34d899a8bde6d8e038d68e231"
            ),
            "action_metadata_sha256": (
                "28c562187b5e25bfca2b767dbc12d17519c98ebb86bdd861afe27a0916881c08"
            ),
            "report_candidate": {
                "active_to": "2023-03-09",
                "exchange": "NASDAQ",
                "index_remove_dates": ["2023-03-15"],
                "last_price_date": "2023-03-09",
                "name": "SVB Financial Group",
                "security_id": "US:EODHD:181d58e2-2b82-5f5d-8c33-311c5d7b5129",
                "symbol": "SIVB",
            },
            "evidence": [
                {
                    "role": "sec_cancellation_raw",
                    "dataset": "sec_edgar_filing",
                    "source": "sec_edgar_filing",
                    "content_type": "text/html",
                    "source_url": (
                        "https://www.sec.gov/Archives/edgar/data/719739/"
                        "000119312524254186/d904756d8k.htm"
                    ),
                    "source_hash": (
                        "14371aef1566bfdcda9ca3171b1ced46d095adf34d899a8bde6d8e038d68e231"
                    ),
                    "object_path": (
                        "archives/2026-07-15/"
                        "14371aef1566bfdcda9ca3171b1ced46d095adf34d899a8bde6d8e038d68e231.html.gz"
                    ),
                    "effective_date": "2026-07-15",
                    "content_bytes": 54478,
                    "raw_payload": True,
                },
                *[dict(item) for item in common_evidence],
            ],
        },
    ]
    return rows


def trusted_sivb_evidence_bindings() -> dict[str, dict[str, Any]]:
    """Return the exact code-pinned SIVB trust inventory."""

    rows = _trusted_sivb_evidence_binding_rows()
    event_ids = {_text(item.get("event_id")) for item in rows}
    _require(
        event_ids == set(TRUSTED_SIVB_EVIDENCE_BINDING_EVENT_IDS)
        and trusted_sivb_evidence_binding_inventory_sha256(rows)
        == TRUSTED_SIVB_EVIDENCE_BINDINGS_SHA256,
        "Trusted SIVB evidence inventory is not code-pinned.",
    )
    return {str(item["event_id"]): item for item in rows}


def trusted_sivb_evidence_binding_inventory_sha256(
    rows: list[Mapping[str, Any]] | None = None,
) -> str:
    """Fingerprint the complete fail-closed diagnostic inventory."""

    values = rows if rows is not None else _trusted_sivb_evidence_binding_rows()
    ordered = sorted(
        (dict(item) for item in values), key=lambda item: _text(item.get("event_id"))
    )
    return canonical_json_sha256(ordered)


def _action_metadata_sha256(value: Any) -> str:
    try:
        parsed = value if isinstance(value, Mapping) else json.loads(_text(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(parsed, Mapping):
        return ""
    encoded = json.dumps(
        dict(parsed), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256_bytes(encoded)


def trusted_sivb_evidence_binding_diagnostic(
    action: Mapping[str, Any],
    archive: pd.DataFrame,
) -> dict[str, Any] | None:
    """Attest one exact SIVB action and every immutable supporting object row."""

    event_id = _text(action.get("event_id"))
    spec = trusted_sivb_evidence_bindings().get(event_id)
    if spec is None:
        return None
    action_binding_exact = bool(
        _text(action.get("security_id")) == spec["security_id"]
        and _text(action.get("action_type")).lower() == spec["action_type"]
        and _date(action.get("effective_date")) == spec["effective_date"]
        and _date(action.get("ex_date")) == spec["ex_date"]
        and _date(action.get("announcement_date")) == spec["announcement_date"]
        and _date(action.get("payment_date")) == spec["payment_date"]
        and _exact_number_text(action.get("cash_amount"), "cash_amount")
        == _exact_number_text(spec["cash_amount"], "cash_amount")
        and _exact_number_text(action.get("ratio"), "ratio")
        == _exact_number_text(spec["ratio"], "ratio")
        and _text(action.get("currency")).upper() == spec["currency"]
        and _text(action.get("new_security_id")) == spec["new_security_id"]
        and _text(action.get("new_symbol")).upper() == spec["new_symbol"]
        and _text(action.get("official")).lower() == "true"
        and _text(action.get("source_kind")) == spec["action_source_kind"]
        and _text(action.get("source")) == spec["action_source"]
        and _text(action.get("source_url")) == spec["action_source_url"]
        and _text(action.get("source_hash")).lower() == spec["action_source_hash"]
        and _action_metadata_sha256(action.get("metadata"))
        == spec["action_metadata_sha256"]
    )
    evidence_bindings: dict[str, bool] = {}
    for evidence in spec["evidence"]:
        digest = _text(evidence.get("source_hash")).lower()
        matches = archive.loc[archive["archive_id"].map(_text).eq(digest)]
        evidence_bindings[str(evidence["role"])] = bool(
            len(matches) == 1
            and _text(matches.iloc[0].get("source_hash")).lower() == digest
            and _text(matches.iloc[0].get("source_url"))
            == evidence["source_url"]
            and _text(matches.iloc[0].get("dataset")) == evidence["dataset"]
            and _text(matches.iloc[0].get("source")) == evidence["source"]
            and _text(matches.iloc[0].get("content_type"))
            == evidence["content_type"]
            and _text(matches.iloc[0].get("object_path"))
            == evidence["object_path"]
            and _date(matches.iloc[0].get("effective_date"))
            == evidence["effective_date"]
        )
    trusted = action_binding_exact and all(evidence_bindings.values())
    return {
        "inventory_sha256": (
            TRUSTED_SIVB_EVIDENCE_BINDINGS_SHA256
        ),
        "action_binding_exact": action_binding_exact,
        "evidence_archive_bindings": evidence_bindings,
        "sec_raw_archived": bool(
            evidence_bindings.get("sec_market_raw")
            and (
                spec["action_type"] != "delisting"
                or evidence_bindings.get("sec_cancellation_raw")
            )
        ),
        "eodhd_raw_archived": bool(evidence_bindings.get("eodhd_otc_prices_raw")),
        "occ_raw_pdf_archived": bool(
            evidence_bindings.get("occ_memo_52179_raw_pdf")
        ),
        "legacy_extraction_archived": bool(
            evidence_bindings.get("occ_memo_52179_legacy_reviewed_extraction")
        ),
        "legacy_extraction_authoritative": False,
        "evidence_hashes": [
            str(item["source_hash"]) for item in spec["evidence"]
        ],
        "status": "trusted" if trusted else "blocked",
    }


def trusted_sivb_report_diagnostic_passed(
    event: Mapping[str, Any],
) -> bool:
    """Recognize only the complete code-pinned SIVB report diagnostic."""

    event_id = _text(event.get("event_id"))
    spec = trusted_sivb_evidence_bindings().get(event_id)
    diagnostic = event.get("trusted_sivb_evidence_binding")
    if spec is None or not isinstance(diagnostic, Mapping):
        return False
    expected_roles = {
        str(item["role"]): True for item in spec["evidence"]
    }
    expected_hashes = [
        str(item["source_hash"]) for item in spec["evidence"]
    ]
    return bool(
        diagnostic.get("inventory_sha256")
        == TRUSTED_SIVB_EVIDENCE_BINDINGS_SHA256
        and diagnostic.get("action_binding_exact") is True
        and diagnostic.get("evidence_archive_bindings") == expected_roles
        and diagnostic.get("sec_raw_archived") is True
        and diagnostic.get("eodhd_raw_archived") is True
        and diagnostic.get("occ_raw_pdf_archived") is True
        and diagnostic.get("legacy_extraction_archived") is True
        and diagnostic.get("legacy_extraction_authoritative") is False
        and diagnostic.get("evidence_hashes") == expected_hashes
        and diagnostic.get("status") == "trusted"
    )


def _trusted_frc_evidence_binding_rows() -> list[dict[str, Any]]:
    """Return the one exact FRC ticker action and its two OCC artifacts."""

    return [
        {
            "event_id": (
                "e351f774b133eae45d49e0fbe60215e5bbceec540c3386076f4c3f2b6c57d9ea"
            ),
            "security_id": "US:EODHD:3453b450-03d1-52ee-9100-cf80856b06ef",
            "action_type": "ticker_change",
            "effective_date": "2023-05-03",
            "ex_date": "2023-05-03",
            "announcement_date": "2023-05-02",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "US:EODHD:3453b450-03d1-52ee-9100-cf80856b06ef",
            "new_symbol": "FRCB",
            "official": True,
            "action_source_kind": "official_crosscheck",
            "action_source": "occ_information_memo",
            "action_source_url": "https://infomemo.theocc.com/infomemos?number=52352",
            "action_source_hash": (
                "0ab8f0c49076f97049dcde80078affda258108ce7c904241bf1eba46b44aec66"
            ),
            "action_retrieved_at": "2026-07-18T18:35:42Z",
            "action_metadata_sha256": (
                "6cd2c29ee9b870a4fbffadaf984aac9a211d579106e72b5bbf43548fdc2cfbb2"
            ),
            "evidence": [
                {
                    "role": "occ_memo_52352_raw_pdf",
                    "archive_id": (
                        "0ab8f0c49076f97049dcde80078affda258108ce7c904241bf1eba46b44aec66"
                    ),
                    "dataset": "occ_information_memo",
                    "source": "occ_information_memo",
                    "content_type": "application/pdf",
                    "source_url": "https://infomemo.theocc.com/infomemos?number=52352",
                    "source_hash": (
                        "0ab8f0c49076f97049dcde80078affda258108ce7c904241bf1eba46b44aec66"
                    ),
                    "object_path": (
                        "archives/2026-07-15/"
                        "0ab8f0c49076f97049dcde80078affda258108ce7c904241bf1eba46b44aec66.pdf.gz"
                    ),
                    "effective_date": "2026-07-15",
                    "retrieved_at": "2026-07-18T18:35:42Z",
                    "content_bytes": 566923,
                    "raw_payload": True,
                },
                {
                    "role": "occ_memo_52352_legacy_reviewed_extraction",
                    "archive_id": (
                        "c568a6ac21ddc05d3c5821c228b94b7bd7e52a602a96b1cfb2f5f08ee24af658"
                    ),
                    "dataset": "occ_reviewed_memo_extraction",
                    "source": "occ_reviewed_memo_extraction",
                    "content_type": "application/json",
                    "source_url": "https://infomemo.theocc.com/infomemos?number=52352",
                    "source_hash": (
                        "377bcc0663eb9666b9f639edaf541b0ec729fd4b84ac876e345baaa9bf413668"
                    ),
                    "object_path": (
                        "archives/2026-07-15/"
                        "377bcc0663eb9666b9f639edaf541b0ec729fd4b84ac876e345baaa9bf413668.json.gz"
                    ),
                    "effective_date": "2026-07-15",
                    "retrieved_at": "2026-07-18T00:00:00Z",
                    "content_bytes": 516,
                    "raw_payload": False,
                },
            ],
        }
    ]


def trusted_frc_evidence_binding_inventory_sha256(
    rows: list[Mapping[str, Any]] | None = None,
) -> str:
    """Fingerprint the complete FRC raw-PDF/legacy audit inventory."""

    values = rows if rows is not None else _trusted_frc_evidence_binding_rows()
    ordered = sorted(
        (dict(item) for item in values), key=lambda item: _text(item.get("event_id"))
    )
    return canonical_json_sha256(ordered)


def trusted_frc_evidence_bindings() -> dict[str, dict[str, Any]]:
    """Return the exact code-pinned FRC trust inventory."""

    rows = _trusted_frc_evidence_binding_rows()
    event_ids = {_text(item.get("event_id")) for item in rows}
    _require(
        event_ids == set(TRUSTED_FRC_EVIDENCE_BINDING_EVENT_IDS)
        and trusted_frc_evidence_binding_inventory_sha256(rows)
        == TRUSTED_FRC_EVIDENCE_BINDINGS_SHA256,
        "Trusted FRC evidence inventory is not code-pinned.",
    )
    return {str(item["event_id"]): item for item in rows}


def trusted_frc_evidence_binding_diagnostic(
    action: Mapping[str, Any],
    archive: pd.DataFrame,
) -> dict[str, Any] | None:
    """Attest the exact FRC action and both immutable OCC archive rows."""

    event_id = _text(action.get("event_id"))
    spec = trusted_frc_evidence_bindings().get(event_id)
    if spec is None:
        return None
    action_binding_exact = bool(
        _text(action.get("security_id")) == spec["security_id"]
        and _text(action.get("action_type")).lower() == spec["action_type"]
        and _date(action.get("effective_date")) == spec["effective_date"]
        and _date(action.get("ex_date")) == spec["ex_date"]
        and _date(action.get("announcement_date")) == spec["announcement_date"]
        and _date(action.get("payment_date")) == spec["payment_date"]
        and _exact_number_text(action.get("cash_amount"), "cash_amount")
        == _exact_number_text(spec["cash_amount"], "cash_amount")
        and _exact_number_text(action.get("ratio"), "ratio")
        == _exact_number_text(spec["ratio"], "ratio")
        and _text(action.get("currency")).upper() == spec["currency"]
        and _text(action.get("new_security_id")) == spec["new_security_id"]
        and _text(action.get("new_symbol")).upper() == spec["new_symbol"]
        and _text(action.get("official")).lower() == "true"
        and _text(action.get("source_kind")) == spec["action_source_kind"]
        and _text(action.get("source")) == spec["action_source"]
        and _text(action.get("source_url")) == spec["action_source_url"]
        and _text(action.get("source_hash")).lower() == spec["action_source_hash"]
        and _text(action.get("retrieved_at")) == spec["action_retrieved_at"]
        and _action_metadata_sha256(action.get("metadata"))
        == spec["action_metadata_sha256"]
    )
    evidence_bindings: dict[str, bool] = {}
    for evidence in spec["evidence"]:
        archive_id = str(evidence["archive_id"])
        matches = archive.loc[archive["archive_id"].map(_text).eq(archive_id)]
        evidence_bindings[str(evidence["role"])] = bool(
            len(matches) == 1
            and _text(matches.iloc[0].get("source_hash")).lower()
            == evidence["source_hash"]
            and _text(matches.iloc[0].get("source_url")) == evidence["source_url"]
            and _text(matches.iloc[0].get("dataset")) == evidence["dataset"]
            and _text(matches.iloc[0].get("source")) == evidence["source"]
            and _text(matches.iloc[0].get("content_type"))
            == evidence["content_type"]
            and _text(matches.iloc[0].get("object_path"))
            == evidence["object_path"]
            and _date(matches.iloc[0].get("effective_date"))
            == evidence["effective_date"]
            and _text(matches.iloc[0].get("retrieved_at"))
            == evidence["retrieved_at"]
        )
    trusted = action_binding_exact and all(evidence_bindings.values())
    return {
        "inventory_sha256": TRUSTED_FRC_EVIDENCE_BINDINGS_SHA256,
        "action_binding_exact": action_binding_exact,
        "evidence_archive_bindings": evidence_bindings,
        "occ_raw_pdf_archived": bool(
            evidence_bindings.get("occ_memo_52352_raw_pdf")
        ),
        "legacy_extraction_archived": bool(
            evidence_bindings.get("occ_memo_52352_legacy_reviewed_extraction")
        ),
        "legacy_extraction_authoritative": False,
        "evidence_hashes": [
            str(item["source_hash"]) for item in spec["evidence"]
        ],
        "status": "trusted" if trusted else "blocked",
    }


def trusted_frc_report_diagnostic_passed(event: Mapping[str, Any]) -> bool:
    """Recognize only the complete code-pinned FRC report diagnostic."""

    event_id = _text(event.get("event_id"))
    spec = trusted_frc_evidence_bindings().get(event_id)
    diagnostic = event.get("trusted_frc_evidence_binding")
    if spec is None or not isinstance(diagnostic, Mapping):
        return False
    expected_roles = {str(item["role"]): True for item in spec["evidence"]}
    expected_hashes = [str(item["source_hash"]) for item in spec["evidence"]]
    return bool(
        diagnostic.get("inventory_sha256")
        == TRUSTED_FRC_EVIDENCE_BINDINGS_SHA256
        and diagnostic.get("action_binding_exact") is True
        and diagnostic.get("evidence_archive_bindings") == expected_roles
        and diagnostic.get("occ_raw_pdf_archived") is True
        and diagnostic.get("legacy_extraction_archived") is True
        and diagnostic.get("legacy_extraction_authoritative") is False
        and diagnostic.get("evidence_hashes") == expected_hashes
        and diagnostic.get("status") == "trusted"
    )


def _trusted_ntco_evidence_binding_rows() -> list[dict[str, Any]]:
    """Return the two exact NTCO/NTCOY actions and reviewed evidence rows."""

    identity_url = "https://infomemo.theocc.com/infomemos?number=54105"
    cash_url = (
        "https://www.adrbny.com/content/dam/adr/documents/"
        "corporate-actions-dr/files/ad1145447.pdf"
    )
    common_security_id = "US:EODHD:7e3cea59-409f-5cf0-8429-9d4245013622"
    return [
        {
            "event_id": (
                "0bd231d216016d33c8d6eb1ec9bb85450a21b9298c4abda34a3399727e9b2c00"
            ),
            "terminal": False,
            "security_id": common_security_id,
            "action_type": "ticker_change",
            "effective_date": "2024-02-12",
            "ex_date": "2024-02-12",
            "announcement_date": "2024-02-09",
            "record_date": "",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "new_security_id": common_security_id,
            "new_symbol": "NTCOY",
            "official": True,
            "action_source_kind": "clearing_and_exchange_notices",
            "action_source": "official_ntco_ntcoy_identity",
            "action_source_url": identity_url,
            "action_source_hash": (
                "8c9312d2079c238a4fa47b701d24b8e707c040080cb8a5ce0d62f6bd82fd54cb"
            ),
            "action_retrieved_at": "2026-07-18T18:47:16.808110Z",
            "action_metadata_sha256": (
                "1509c1aafa2c28c94fc9541d28dfe4135a3d9391430d5bdb1a4953b347cf75d4"
            ),
            "evidence": [
                {
                    "role": "reviewed_identity_extraction",
                    "archive_id": (
                        "8c9312d2079c238a4fa47b701d24b8e707c040080cb8a5ce0d62f6bd82fd54cb"
                    ),
                    "dataset": "official_ntco_ntcoy_identity",
                    "source": "official_ntco_ntcoy_identity",
                    "content_type": "application/json",
                    "source_url": identity_url,
                    "source_hash": (
                        "8c9312d2079c238a4fa47b701d24b8e707c040080cb8a5ce0d62f6bd82fd54cb"
                    ),
                    "object_path": (
                        "archives/2026-07-15/"
                        "8c9312d2079c238a4fa47b701d24b8e707c040080cb8a5ce0d62f6bd82fd54cb.json.gz"
                    ),
                    "effective_date": "2026-07-15",
                    "retrieved_at": "2026-07-18T18:47:16.808110Z",
                    "content_bytes": 482,
                    "raw_payload": False,
                },
                {
                    "role": "occ_memo_54105_raw_pdf",
                    "archive_id": (
                        "5eb3c0058195d6d6c22361e362f1623c4eae7dd7262e9497ef7b53f8a564e913"
                    ),
                    "dataset": "official_occ",
                    "source": "official_occ",
                    # The immutable archive row inherited the collector's
                    # response header, but its hash-pinned bytes are a PDF.
                    "content_type": "text/html",
                    "source_url": identity_url,
                    "source_hash": (
                        "5eb3c0058195d6d6c22361e362f1623c4eae7dd7262e9497ef7b53f8a564e913"
                    ),
                    "object_path": (
                        "archives/2026-07-15/"
                        "5eb3c0058195d6d6c22361e362f1623c4eae7dd7262e9497ef7b53f8a564e913.bin.gz"
                    ),
                    "effective_date": "2026-07-15",
                    "retrieved_at": "2026-07-18T17:41:32.027461Z",
                    "content_bytes": 567172,
                    "raw_payload": True,
                },
                {
                    "role": "cboe_ntco_restriction_raw_pdf",
                    "archive_id": (
                        "e67d8046a4c532daa11dac1faacb9e573238812c90d15be9792b5d6a46e67928"
                    ),
                    "dataset": "official_cboe",
                    "source": "official_cboe",
                    "content_type": "application/pdf",
                    "source_url": (
                        "https://cdn.cboe.com/resources/product_restriction/2024/"
                        "Cboe-Options-Exchanges-Restrictions-on-Transactions-in-"
                        "Options-on-Natura-Co-Holding-S-A.pdf"
                    ),
                    "source_hash": (
                        "e67d8046a4c532daa11dac1faacb9e573238812c90d15be9792b5d6a46e67928"
                    ),
                    "object_path": (
                        "archives/2026-07-15/"
                        "e67d8046a4c532daa11dac1faacb9e573238812c90d15be9792b5d6a46e67928.bin.gz"
                    ),
                    "effective_date": "2026-07-15",
                    "retrieved_at": "2026-07-18T17:41:15.684878Z",
                    "content_bytes": 126194,
                    "raw_payload": True,
                },
            ],
        },
        {
            "event_id": (
                "d1eeadf8f4779212ff4b3162af1731bf7bfc804c10756cc4bb8074c307e1f746"
            ),
            "terminal": True,
            "security_id": common_security_id,
            "action_type": "delisting",
            "effective_date": "2024-09-04",
            "ex_date": "2024-09-04",
            "announcement_date": "2024-08-26",
            "record_date": "",
            "payment_date": "2024-09-04",
            "cash_amount": 5.043659,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "official": True,
            "action_source_kind": "depositary_corporate_action_notice",
            "action_source": "official_ntcoy_cash_termination",
            "action_source_url": cash_url,
            "action_source_hash": (
                "3be830009ee6942d4ca604aafdc19730a693f13e62b046641a638e9a1ee112c1"
            ),
            "action_retrieved_at": "2026-07-18T18:47:16.808110Z",
            "action_metadata_sha256": (
                "a61d317ca5abf8c438d2b4379ab1eacb624478b2469cc048a68a69f8cf342924"
            ),
            "evidence": [
                {
                    "role": "reviewed_cash_termination_extraction",
                    "archive_id": (
                        "3be830009ee6942d4ca604aafdc19730a693f13e62b046641a638e9a1ee112c1"
                    ),
                    "dataset": "official_ntcoy_cash_termination",
                    "source": "official_ntcoy_cash_termination",
                    "content_type": "application/json",
                    "source_url": cash_url,
                    "source_hash": (
                        "3be830009ee6942d4ca604aafdc19730a693f13e62b046641a638e9a1ee112c1"
                    ),
                    "object_path": (
                        "archives/2026-07-15/"
                        "3be830009ee6942d4ca604aafdc19730a693f13e62b046641a638e9a1ee112c1.json.gz"
                    ),
                    "effective_date": "2026-07-15",
                    "retrieved_at": "2026-07-18T18:47:16.808110Z",
                    "content_bytes": 341,
                    "raw_payload": False,
                },
                {
                    "role": "bny_cash_notice_raw_pdf",
                    "archive_id": (
                        "057a95798bd3b9ff4f0f0be6fd34d7ed42d6ad1d27a12ad1c41e30700c8da28b"
                    ),
                    "dataset": "official_bny",
                    "source": "official_bny",
                    "content_type": "application/pdf",
                    "source_url": cash_url,
                    "source_hash": (
                        "057a95798bd3b9ff4f0f0be6fd34d7ed42d6ad1d27a12ad1c41e30700c8da28b"
                    ),
                    "object_path": (
                        "archives/2026-07-15/"
                        "057a95798bd3b9ff4f0f0be6fd34d7ed42d6ad1d27a12ad1c41e30700c8da28b.bin.gz"
                    ),
                    "effective_date": "2026-07-15",
                    "retrieved_at": "2026-07-18T17:42:06.789266Z",
                    "content_bytes": 173916,
                    "raw_payload": True,
                },
                {
                    "role": "bny_termination_notice_raw_pdf",
                    "archive_id": (
                        "abc23f86378f82f5efb475fc23e0a82e398db4201dc7e52fd4064d088eb47b83"
                    ),
                    "dataset": "official_bny_termination",
                    "source": "official_bny_termination",
                    "content_type": "application/pdf",
                    "source_url": (
                        "https://www.adrbny.com/content/dam/adr/documents/"
                        "corporate-actions-dr/files/ad1140774.pdf"
                    ),
                    "source_hash": (
                        "abc23f86378f82f5efb475fc23e0a82e398db4201dc7e52fd4064d088eb47b83"
                    ),
                    "object_path": (
                        "archives/2026-07-15/"
                        "abc23f86378f82f5efb475fc23e0a82e398db4201dc7e52fd4064d088eb47b83.bin.gz"
                    ),
                    "effective_date": "2026-07-15",
                    "retrieved_at": "2026-07-18T18:42:57.215876Z",
                    "content_bytes": 162474,
                    "raw_payload": True,
                },
                {
                    "role": "bny_books_closed_notice_raw_pdf",
                    "archive_id": (
                        "b5d449fcad19cd38fb461c8f9bae4c1856c0fb7ccd4903e062f87840315ba675"
                    ),
                    "dataset": "official_bny_books_closed",
                    "source": "official_bny_books_closed",
                    "content_type": "application/pdf",
                    "source_url": (
                        "https://www.adrbny.com/content/dam/adr/documents/"
                        "books-closed/files/bc1141635.pdf"
                    ),
                    "source_hash": (
                        "b5d449fcad19cd38fb461c8f9bae4c1856c0fb7ccd4903e062f87840315ba675"
                    ),
                    "object_path": (
                        "archives/2026-07-15/"
                        "b5d449fcad19cd38fb461c8f9bae4c1856c0fb7ccd4903e062f87840315ba675.bin.gz"
                    ),
                    "effective_date": "2026-07-15",
                    "retrieved_at": "2026-07-18T18:47:16.808110Z",
                    "content_bytes": 271127,
                    "raw_payload": True,
                },
            ],
        },
    ]


def trusted_ntco_evidence_binding_inventory_sha256(
    rows: list[Mapping[str, Any]] | None = None,
) -> str:
    """Fingerprint the complete exact NTCO transition trust inventory."""

    values = rows if rows is not None else _trusted_ntco_evidence_binding_rows()
    ordered = sorted(
        (dict(item) for item in values), key=lambda item: _text(item.get("event_id"))
    )
    return canonical_json_sha256(ordered)


def trusted_ntco_evidence_bindings() -> dict[str, dict[str, Any]]:
    """Return the exact code-pinned NTCO transition trust inventory."""

    rows = _trusted_ntco_evidence_binding_rows()
    event_ids = {_text(item.get("event_id")) for item in rows}
    _require(
        event_ids == set(TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS)
        and trusted_ntco_evidence_binding_inventory_sha256(rows)
        == TRUSTED_NTCO_EVIDENCE_BINDINGS_SHA256,
        "Trusted NTCO evidence inventory is not code-pinned.",
    )
    return {str(item["event_id"]): item for item in rows}


def trusted_ntco_evidence_binding_diagnostic(
    action: Mapping[str, Any],
    archive: pd.DataFrame,
) -> dict[str, Any] | None:
    """Attest one exact NTCO action and every reviewed archive row."""

    event_id = _text(action.get("event_id"))
    spec = trusted_ntco_evidence_bindings().get(event_id)
    if spec is None:
        return None
    action_binding_exact = bool(
        _text(action.get("security_id")) == spec["security_id"]
        and _text(action.get("action_type")).lower() == spec["action_type"]
        and _date(action.get("effective_date")) == spec["effective_date"]
        and _date(action.get("ex_date")) == spec["ex_date"]
        and _date(action.get("announcement_date")) == spec["announcement_date"]
        and _date(action.get("record_date")) == spec["record_date"]
        and _date(action.get("payment_date")) == spec["payment_date"]
        and _exact_number_text(action.get("cash_amount"), "cash_amount")
        == _exact_number_text(spec["cash_amount"], "cash_amount")
        and _exact_number_text(action.get("ratio"), "ratio")
        == _exact_number_text(spec["ratio"], "ratio")
        and _text(action.get("currency")).upper() == spec["currency"]
        and _text(action.get("new_security_id")) == spec["new_security_id"]
        and _text(action.get("new_symbol")).upper() == spec["new_symbol"]
        and _text(action.get("official")).lower() == "true"
        and _text(action.get("source_kind")) == spec["action_source_kind"]
        and _text(action.get("source")) == spec["action_source"]
        and _text(action.get("source_url")) == spec["action_source_url"]
        and _text(action.get("source_hash")).lower() == spec["action_source_hash"]
        and _text(action.get("retrieved_at")) == spec["action_retrieved_at"]
        and _action_metadata_sha256(action.get("metadata"))
        == spec["action_metadata_sha256"]
    )
    evidence_bindings: dict[str, bool] = {}
    for evidence in spec["evidence"]:
        archive_id = str(evidence["archive_id"])
        matches = archive.loc[archive["archive_id"].map(_text).eq(archive_id)]
        evidence_bindings[str(evidence["role"])] = bool(
            len(matches) == 1
            and _text(matches.iloc[0].get("source_hash")).lower()
            == evidence["source_hash"]
            and _text(matches.iloc[0].get("source_url")) == evidence["source_url"]
            and _text(matches.iloc[0].get("dataset")) == evidence["dataset"]
            and _text(matches.iloc[0].get("source")) == evidence["source"]
            and _text(matches.iloc[0].get("content_type"))
            == evidence["content_type"]
            and _text(matches.iloc[0].get("object_path"))
            == evidence["object_path"]
            and _date(matches.iloc[0].get("effective_date"))
            == evidence["effective_date"]
            and _text(matches.iloc[0].get("retrieved_at"))
            == evidence["retrieved_at"]
        )
    trusted = action_binding_exact and all(evidence_bindings.values())
    raw_roles = {
        str(item["role"])
        for item in spec["evidence"]
        if item.get("raw_payload") is True
    }
    return {
        "inventory_sha256": TRUSTED_NTCO_EVIDENCE_BINDINGS_SHA256,
        "expected_terminal": spec["terminal"],
        "action_binding_exact": action_binding_exact,
        "evidence_archive_bindings": evidence_bindings,
        "derived_action_evidence_archived": bool(
            evidence_bindings.get("reviewed_identity_extraction")
            or evidence_bindings.get("reviewed_cash_termination_extraction")
        ),
        "raw_official_evidence_archived": bool(
            raw_roles and all(evidence_bindings.get(role) for role in raw_roles)
        ),
        "evidence_hashes": [
            str(item["source_hash"]) for item in spec["evidence"]
        ],
        "status": "trusted" if trusted else "blocked",
    }


def trusted_ntco_report_diagnostic_passed(event: Mapping[str, Any]) -> bool:
    """Recognize only a complete code-pinned NTCO report diagnostic."""

    event_id = _text(event.get("event_id"))
    spec = trusted_ntco_evidence_bindings().get(event_id)
    diagnostic = event.get("trusted_ntco_evidence_binding")
    if spec is None or not isinstance(diagnostic, Mapping):
        return False
    expected_roles = {str(item["role"]): True for item in spec["evidence"]}
    expected_hashes = [str(item["source_hash"]) for item in spec["evidence"]]
    return bool(
        diagnostic.get("inventory_sha256")
        == TRUSTED_NTCO_EVIDENCE_BINDINGS_SHA256
        and diagnostic.get("expected_terminal") is spec["terminal"]
        and diagnostic.get("action_binding_exact") is True
        and diagnostic.get("evidence_archive_bindings") == expected_roles
        and diagnostic.get("derived_action_evidence_archived") is True
        and diagnostic.get("raw_official_evidence_archived") is True
        and diagnostic.get("evidence_hashes") == expected_hashes
        and diagnostic.get("status") == "trusted"
    )


def _permanent_exception_registry_payload(
    specs: Mapping[str, OfficialLifecycleExceptionEvidenceSpec],
) -> dict[str, dict[str, Any]]:
    return {
        evidence_id: asdict(spec)
        for evidence_id, spec in sorted(specs.items())
        if spec.resolution_kind == "exception"
    }


def trusted_permanent_exception_specs() -> dict[
    str, OfficialLifecycleExceptionEvidenceSpec
]:
    """Load the code-fingerprinted permanent exception registry."""

    specs = load_official_lifecycle_exception_evidence(
        DEFAULT_OFFICIAL_LIFECYCLE_HINTS
    )
    permanent = {
        evidence_id: spec
        for evidence_id, spec in specs.items()
        if spec.resolution_kind == "exception"
    }
    observed = canonical_json_sha256(
        _permanent_exception_registry_payload(permanent)
    )
    _require(
        observed == TRUSTED_PERMANENT_EXCEPTION_REGISTRY_SHA256,
        "Permanent lifecycle exception registry is not the code-pinned inventory.",
    )
    return permanent


def permanent_exception_spec_for_resolution(
    resolution: Mapping[str, Any],
    specs: Mapping[str, OfficialLifecycleExceptionEvidenceSpec],
) -> OfficialLifecycleExceptionEvidenceSpec | None:
    """Return one exact identity/date-bound spec, never a symbol-only match."""

    security_id = _text(resolution.get("security_id"))
    symbol = _text(resolution.get("symbol")).upper()
    last_price_date = _date(resolution.get("last_price_date"))
    matches = [
        spec
        for spec in specs.values()
        if spec.resolution_kind == "exception"
        and security_id in spec.candidate_security_ids
        and symbol in spec.candidate_symbols
        and last_price_date in spec.candidate_last_price_dates
    ]
    _require(
        len(matches) <= 1,
        "Permanent lifecycle exception has ambiguous official registry bindings: "
        + _text(resolution.get("candidate_id")),
    )
    return matches[0] if matches else None


def dataframe_sha256(frame: pd.DataFrame, primary_key: tuple[str, ...]) -> str:
    """Canonical logical-frame hash independent of Parquet byte encoding."""

    columns = sorted(str(column) for column in frame.columns)
    ordered = frame.loc[:, columns].copy()
    if primary_key and not ordered.empty:
        ordered = ordered.sort_values(list(primary_key), kind="stable")

    def normalize(value: Any) -> Any:
        if value is None:
            return None
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return value

    records = [
        {column: normalize(row[column]) for column in columns}
        for row in ordered.to_dict(orient="records")
    ]
    return canonical_json_sha256(records)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _integer(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Cross-validation {name} is not an integer.") from exc
    _require(result >= 0, f"Cross-validation {name} cannot be negative.")
    return result


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else pd.Timestamp(parsed).date().isoformat()


def _number(value: Any) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def _exact_number_text(value: Any, field: str) -> str | None:
    """Canonical decimal text for equality without a numeric tolerance."""

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeError(
            f"Reviewed nonterminal extraction {field} is not numeric."
        ) from exc
    _require(
        parsed.is_finite(),
        f"Reviewed nonterminal extraction {field} must be finite.",
    )
    if parsed == 0:
        return "0"
    return format(parsed.normalize(), "f")


def _event_gate_digest(value: Any, field: str, *, empty_allowed: bool = False) -> str:
    output = _text(value).lower()
    _require(
        (empty_allowed and not output)
        or (
            len(output) == 64
            and all(character in "0123456789abcdef" for character in output)
        ),
        f"Reviewed terminal event gate {field} must be lowercase SHA-256.",
    )
    return output


def _normalize_event_gate_value(value: Any) -> Any:
    """Return JSON-safe values for stable event-local semantic fingerprints."""

    if isinstance(value, Mapping):
        return {
            str(key): _normalize_event_gate_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_event_gate_value(item) for item in value]
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def terminal_event_gate_action_semantics(
    action: Mapping[str, Any],
) -> dict[str, Any]:
    """Project every decision-bearing terminal action field deterministically."""

    return {
        "event_id": _text(action.get("event_id")).lower(),
        "security_id": _text(action.get("security_id")),
        "action_type": _text(action.get("action_type")).lower(),
        "effective_date": _date(action.get("effective_date")),
        "ex_date": _date(action.get("ex_date")),
        "announcement_date": _date(action.get("announcement_date")),
        "record_date": _date(action.get("record_date")),
        "payment_date": _date(action.get("payment_date")),
        "new_security_id": _text(action.get("new_security_id")),
        "new_symbol": _text(action.get("new_symbol")).upper(),
        "ratio": _exact_number_text(action.get("ratio"), "ratio"),
        "cash_amount": _exact_number_text(
            action.get("cash_amount"), "cash_amount"
        ),
        "currency": _text(action.get("currency")).upper(),
        "official": _text(action.get("official")).lower() == "true",
        "source_kind": _text(action.get("source_kind")),
        "source": _text(action.get("source")),
        "source_url": _text(action.get("source_url")),
        "source_hash": _text(action.get("source_hash")).lower(),
        "retrieved_at": _text(action.get("retrieved_at")),
        "metadata_sha256": _action_metadata_sha256(action.get("metadata")),
    }


def terminal_event_gate_resolution_semantics(
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the complete reviewed resolution decision and successor binding."""

    fields = (
        "candidate_id",
        "security_id",
        "symbol",
        "resolution",
        "event_id",
        "exception_code",
        "exception_reason",
        "reviewed_by",
        "reviewed_at",
        "recheck_after",
        "successor_security_id",
        "successor_symbol",
        "source_url",
        "source",
        "retrieved_at",
        "source_hash",
    )
    output = {field: _text(resolution.get(field)) for field in fields}
    output["candidate_id"] = output["candidate_id"].lower()
    output["event_id"] = output["event_id"].lower()
    output["symbol"] = output["symbol"].upper()
    output["resolution"] = output["resolution"].lower()
    output["successor_symbol"] = output["successor_symbol"].upper()
    output["source_hash"] = output["source_hash"].lower()
    output["last_price_date"] = _date(resolution.get("last_price_date"))
    return output


def terminal_event_gate_report_semantics(
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project one report row without volatile search-artifact retrieval times."""

    if not isinstance(record, Mapping):
        return {"record_missing": True}
    verified = record.get("verified_event")
    parsed = record.get("parsed")
    event_state = (
        "verified"
        if isinstance(verified, Mapping)
        else "parsed"
        if isinstance(parsed, Mapping)
        else "missing"
    )
    output = {
        "event_state": event_state,
        "candidate": record.get("candidate"),
        "event": verified if isinstance(verified, Mapping) else parsed,
        "source_url": _text(record.get("source_url")),
        "source_hash": _text(record.get("source_hash")).lower(),
        "successor_security_id": _text(record.get("successor_security_id")),
        "eligible_for_apply": record.get("eligible_for_apply"),
        "manual_review": record.get("manual_review"),
        "manual_review_reason": _text(record.get("manual_review_reason")),
        "crosscheck": record.get("crosscheck"),
        "filing": record.get("filing"),
        "error": _text(record.get("error")),
    }
    return _normalize_event_gate_value(output)


def terminal_event_gate_archive_semantics(
    archive: pd.DataFrame,
    archive_ids: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Project every exact archive row named by one event gate."""

    _require(
        "archive_id" in archive.columns,
        "Reviewed terminal event gate source archive lacks archive_id.",
    )
    rows: list[dict[str, Any]] = []
    for archive_id in sorted(str(value).lower() for value in archive_ids):
        matches = archive.loc[archive["archive_id"].map(_text).str.lower().eq(archive_id)]
        _require(
            len(matches) == 1,
            "Reviewed terminal event gate archive_id must bind exactly once: "
            + archive_id,
        )
        row = matches.iloc[0]
        rows.append(
            {
                "archive_id": _text(row.get("archive_id")).lower(),
                "dataset": _text(row.get("dataset")),
                "object_path": _text(row.get("object_path")),
                "content_type": _text(row.get("content_type")),
                "effective_date": _date(row.get("effective_date")),
                "source": _text(row.get("source")),
                "retrieved_at": _text(row.get("retrieved_at")),
                "source_hash": _text(row.get("source_hash")).lower(),
                "source_url": _text(row.get("source_url")),
            }
        )
    return rows


def reviewed_identity_bound_hint_sha256(
    hint_key: str,
    *,
    hints_path: Path = DEFAULT_OFFICIAL_LIFECYCLE_HINTS,
) -> str:
    """Hash one exact identity-bound hint rather than the mutable whole file."""

    payload = yaml.safe_load(hints_path.read_text(encoding="utf-8")) or {}
    hints = payload.get("identity_bound_hints") or {}
    _require(
        isinstance(hints, Mapping) and hint_key in hints,
        "Reviewed terminal event gate identity-bound hint is missing: " + hint_key,
    )
    return canonical_json_sha256(_normalize_event_gate_value(hints[hint_key]))


def _canonical_reviewed_terminal_event_gate(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping),
        "Reviewed terminal event gate must be an object.",
    )
    _require(
        set(value) == set(REVIEWED_TERMINAL_EVENT_GATE_FIELDS),
        "Reviewed terminal event gate fields are not exact.",
    )
    event_id = _event_gate_digest(value.get("event_id"), "event_id")
    archive_ids = value.get("archive_ids")
    _require(
        isinstance(archive_ids, list)
        and bool(archive_ids)
        and len(archive_ids) == len(set(archive_ids)),
        "Reviewed terminal event gate archive_ids are invalid.",
    )
    normalized_archive_ids = sorted(
        _event_gate_digest(item, "archive_id") for item in archive_ids
    )
    hint_key = _text(value.get("hint_key"))
    hint_sha256 = _event_gate_digest(
        value.get("hint_sha256"), "hint_sha256", empty_allowed=True
    )
    _require(
        bool(hint_key) == bool(hint_sha256),
        "Reviewed terminal event gate hint key/hash binding is incomplete.",
    )
    output = {
        "event_id": event_id,
        "candidate_id": _event_gate_digest(
            value.get("candidate_id"), "candidate_id"
        ),
        "security_id": _text(value.get("security_id")),
        "symbol": _text(value.get("symbol")).upper(),
        "policy_code": _text(value.get("policy_code")),
        "action_sha256": _event_gate_digest(
            value.get("action_sha256"), "action_sha256"
        ),
        "resolution_sha256": _event_gate_digest(
            value.get("resolution_sha256"), "resolution_sha256"
        ),
        "report_semantic_sha256": _event_gate_digest(
            value.get("report_semantic_sha256"), "report_semantic_sha256"
        ),
        "archive_ids": normalized_archive_ids,
        "archive_binding_sha256": _event_gate_digest(
            value.get("archive_binding_sha256"), "archive_binding_sha256"
        ),
        "hint_key": hint_key,
        "hint_sha256": hint_sha256,
        "lifecycle_evidence_report_sha256": _event_gate_digest(
            value.get("lifecycle_evidence_report_sha256"),
            "lifecycle_evidence_report_sha256",
        ),
    }
    _require(
        bool(output["security_id"] and output["symbol"])
        and TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_POLICY_CODES.get(event_id)
        == output["policy_code"],
        "Reviewed terminal event gate identity/policy code is not approved.",
    )
    return output


def reviewed_terminal_event_gates(
    events_policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = events_policy.get("reviewed_terminal_event_gates")
    _require(isinstance(raw, list), "Reviewed terminal event gates must be a list.")
    output: dict[str, dict[str, Any]] = {}
    for value in raw:
        normalized = _canonical_reviewed_terminal_event_gate(value)
        event_id = str(normalized["event_id"])
        _require(
            event_id not in output,
            "Duplicate reviewed terminal event gate event_id: " + event_id,
        )
        output[event_id] = normalized
    return output


def reviewed_terminal_event_gate_sha256(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256(_canonical_reviewed_terminal_event_gate(value))


def reviewed_terminal_event_gate_inventory_sha256(
    events_policy: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(reviewed_terminal_event_gates(events_policy))


def reviewed_terminal_event_gate_mismatches(
    action: Mapping[str, Any],
    resolution: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    archive: pd.DataFrame,
    gate: Mapping[str, Any],
    lifecycle_report_sha256: str,
) -> tuple[str, ...]:
    """Attest every event-local semantic hash and exact supporting object row."""

    expected = _canonical_reviewed_terminal_event_gate(gate)
    mismatches: list[str] = []
    action_semantics = terminal_event_gate_action_semantics(action)
    resolution_semantics = terminal_event_gate_resolution_semantics(resolution)
    report_semantics = terminal_event_gate_report_semantics(record)
    hashes = {
        "action_sha256": canonical_json_sha256(action_semantics),
        "resolution_sha256": canonical_json_sha256(resolution_semantics),
        "report_semantic_sha256": canonical_json_sha256(report_semantics),
    }
    mismatches.extend(
        field for field, digest in hashes.items() if digest != expected[field]
    )
    if _text(lifecycle_report_sha256).lower() != expected[
        "lifecycle_evidence_report_sha256"
    ]:
        mismatches.append("lifecycle_evidence_report_sha256")
    identity_pairs = {
        "event_id": (action_semantics["event_id"], expected["event_id"]),
        "security_id": (action_semantics["security_id"], expected["security_id"]),
        "candidate_id": (
            resolution_semantics["candidate_id"],
            expected["candidate_id"],
        ),
        "resolution_symbol": (
            resolution_semantics["symbol"],
            expected["symbol"],
        ),
    }
    mismatches.extend(
        field for field, (actual, wanted) in identity_pairs.items() if actual != wanted
    )
    if resolution_semantics["resolution"] != "applied":
        mismatches.append("resolution_kind")
    if not action_semantics["official"]:
        mismatches.append("official")
    if canonical_lifecycle_event_id(
        action_semantics["security_id"],
        action_semantics["action_type"],
        action_semantics["effective_date"],
    ) != expected["event_id"]:
        mismatches.append("canonical_event_id")
    if lifecycle_candidate_id(
        action_semantics["security_id"], resolution_semantics["last_price_date"]
    ) != expected["candidate_id"]:
        mismatches.append("canonical_candidate_id")

    try:
        archive_semantics = terminal_event_gate_archive_semantics(
            archive, expected["archive_ids"]
        )
    except RuntimeError:
        archive_semantics = []
        mismatches.append("archive_ids")
    if canonical_json_sha256(archive_semantics) != expected[
        "archive_binding_sha256"
    ]:
        mismatches.append("archive_binding_sha256")
    pair_ids: set[str] = set()
    if {"archive_id", "source_url", "source_hash"}.issubset(archive.columns):
        pair_mask = (
            archive["source_url"].map(_text).eq(action_semantics["source_url"])
            & archive["source_hash"]
            .map(_text)
            .str.lower()
            .eq(action_semantics["source_hash"])
        )
        pair_ids = {
            _text(value).lower()
            for value in archive.loc[pair_mask, "archive_id"]
        }
    if not pair_ids or not pair_ids <= set(expected["archive_ids"]):
        mismatches.append("action_archive_pair")

    hint_key = str(expected["hint_key"])
    if hint_key:
        try:
            hint_sha256 = reviewed_identity_bound_hint_sha256(hint_key)
        except RuntimeError:
            hint_sha256 = ""
        if hint_sha256 != expected["hint_sha256"]:
            mismatches.append("identity_bound_hint_sha256")
    return tuple(dict.fromkeys(mismatches))


def _canonical_reviewed_nonterminal_extraction(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one manually reviewed exact extraction."""

    _require(
        isinstance(value, Mapping),
        "Reviewed nonterminal extraction must be an object.",
    )
    _require(
        set(value) == set(REVIEWED_NONTERMINAL_EXTRACTION_FIELDS),
        "Reviewed nonterminal extraction fields are not exact.",
    )
    effective_date = _date(value.get("effective_date"))
    _require(
        bool(effective_date)
        and _text(value.get("effective_date")) == effective_date,
        "Reviewed nonterminal extraction effective_date must be exact ISO date.",
    )
    output = {
        "event_id": _text(value.get("event_id")),
        "security_id": _text(value.get("security_id")),
        "action_type": _text(value.get("action_type")).lower(),
        "effective_date": effective_date,
        "new_security_id": _text(value.get("new_security_id")),
        "new_symbol": _text(value.get("new_symbol")).upper(),
        "ratio": _exact_number_text(value.get("ratio"), "ratio"),
        "cash_amount": _exact_number_text(
            value.get("cash_amount"), "cash_amount"
        ),
        "currency": _text(value.get("currency")).upper(),
        "source_kind": _text(value.get("source_kind")),
        "source_url": _text(value.get("source_url")),
        "source_hash": _text(value.get("source_hash")).lower(),
    }
    _require(
        bool(output["event_id"] and output["security_id"]),
        "Reviewed nonterminal extraction identity is incomplete.",
    )
    _require(
        output["action_type"]
        in {"cash_merger", "stock_merger", "spinoff", "ticker_change", "delisting"},
        "Reviewed nonterminal extraction action_type is invalid.",
    )
    _require(
        output["currency"] == "USD",
        "Reviewed nonterminal extraction currency must be USD.",
    )
    source_hash = str(output["source_hash"])
    _require(
        len(source_hash) == 64
        and source_hash == source_hash.lower()
        and all(character in "0123456789abcdef" for character in source_hash),
        "Reviewed nonterminal extraction source_hash must be lowercase SHA-256.",
    )
    action_type = str(output["action_type"])
    ratio = output["ratio"]
    cash = output["cash_amount"]
    successor = bool(output["new_security_id"] and output["new_symbol"])
    if action_type == "stock_merger":
        _require(
            successor and ratio is not None and Decimal(str(ratio)) > 0,
            "Reviewed stock-merger extraction lacks exact successor/ratio terms.",
        )
        _require(
            cash is None or Decimal(str(cash)) >= 0,
            "Reviewed stock-merger extraction cash cannot be negative.",
        )
    elif action_type == "spinoff":
        _require(
            successor
            and ratio is not None
            and Decimal(str(ratio)) > 0
            and cash is None,
            "Reviewed spin-off extraction lacks exact successor/ratio terms.",
        )
    elif action_type == "ticker_change":
        _require(
            successor and ratio is None and cash is None,
            "Reviewed ticker-change extraction has invalid economic terms.",
        )
    elif action_type == "cash_merger":
        _require(
            not output["new_security_id"]
            and not output["new_symbol"]
            and ratio is None
            and cash is not None
            and Decimal(str(cash)) > 0,
            "Reviewed cash-merger extraction has invalid economic terms.",
        )
    else:
        _require(
            not output["new_security_id"]
            and not output["new_symbol"]
            and ratio is None
            and cash is not None
            and Decimal(str(cash)) >= 0,
            "Reviewed delisting extraction has invalid economic terms.",
        )
    return output


def reviewed_nonterminal_extractions(
    events_policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the unique, normalized reviewed-extraction inventory."""

    raw = events_policy.get("reviewed_nonterminal_extractions")
    _require(
        isinstance(raw, list),
        "Policy reviewed_nonterminal_extractions must be a list.",
    )
    output: dict[str, dict[str, Any]] = {}
    for value in raw:
        normalized = _canonical_reviewed_nonterminal_extraction(value)
        event_id = str(normalized["event_id"])
        _require(
            event_id not in output,
            f"Duplicate reviewed nonterminal extraction event_id: {event_id}",
        )
        output[event_id] = normalized
    return output


def reviewed_nonterminal_extraction_mismatches(
    action: Mapping[str, Any],
    extraction: Mapping[str, Any],
) -> tuple[str, ...]:
    """Compare every reviewed field against one corporate-action row."""

    expected = _canonical_reviewed_nonterminal_extraction(extraction)
    actual = {
        "event_id": _text(action.get("event_id")),
        "security_id": _text(action.get("security_id")),
        "action_type": _text(action.get("action_type")).lower(),
        "effective_date": _date(action.get("effective_date")),
        "new_security_id": _text(action.get("new_security_id")),
        "new_symbol": _text(action.get("new_symbol")).upper(),
        "ratio": _exact_number_text(action.get("ratio"), "ratio"),
        "cash_amount": _exact_number_text(
            action.get("cash_amount"), "cash_amount"
        ),
        "currency": _text(action.get("currency")).upper(),
        "source_kind": _text(action.get("source_kind")),
        "source_url": _text(action.get("source_url")),
        "source_hash": _text(action.get("source_hash")).lower(),
    }
    return tuple(
        field
        for field in REVIEWED_NONTERMINAL_EXTRACTION_FIELDS
        if actual[field] != expected[field]
    )


def reviewed_nonterminal_extraction_sha256(
    extraction: Mapping[str, Any],
) -> str:
    """Fingerprint the normalized reviewed extraction stored in the report."""

    return canonical_json_sha256(
        _canonical_reviewed_nonterminal_extraction(extraction)
    )


def reviewed_nonterminal_inventory_sha256(
    events_policy: Mapping[str, Any],
) -> str:
    """Fingerprint the complete reviewed inventory independent of list order."""

    return canonical_json_sha256(reviewed_nonterminal_extractions(events_policy))


def reviewed_nonterminal_same_sid_no_data_binding(
    target: Mapping[str, Any],
    event: Mapping[str, Any],
    reviewed_extractions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Recognize only a code-pinned same-SID nonterminal ticker interval.

    The generic terminal-provider exception normally requires one applied
    lifecycle resolution.  That would be false for a same-security ticker
    continuation, so this binding substitutes an exact event/identity/target
    proof for that one reviewed interval and nothing else.
    """

    if (
        canonical_json_sha256(
            TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SPECS
        )
        != TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SHA256
    ):
        raise RuntimeError(
            "Reviewed nonterminal same-SID no-data inventory is not code-pinned."
        )
    event_id = _text(event.get("event_id")).lower()
    spec = TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SPECS.get(event_id)
    if spec is None:
        return None
    extraction = reviewed_extractions.get(event_id)
    if extraction is None:
        return None
    extraction_hash = reviewed_nonterminal_extraction_sha256(extraction)
    target_id = _text(target.get("target_id")).lower()
    security_id = _text(target.get("security_id"))
    symbol = _text(target.get("provider_symbol") or target.get("symbol")).upper()
    active_from = _date(
        target.get("active_from") or target.get("identity_active_from")
    )
    active_to = _date(
        target.get("active_to") or target.get("identity_active_to")
    )
    exact_target = (
        target_id == spec["source_target_id"]
        and security_id == spec["security_id"]
        and symbol == spec["old_symbol"]
        and active_from == spec["old_active_from"]
        and active_to == spec["old_active_to"]
        and _text(target.get("terminal_event_id")).lower() == event_id
        and _text(target.get("successor_security_id")) == security_id
    )
    exact_event = (
        event.get("status") == "passed"
        and _text(event.get("validation_kind")) == NONTERMINAL_EVENT_VALIDATION
        and not _text(event.get("candidate_id"))
        and event.get("lifecycle_report_extraction_approved") is False
        and event.get("reviewed_extraction_match") is True
        and _text(event.get("reviewed_extraction_sha256")) == extraction_hash
        and _text(event.get("security_id")) == spec["security_id"]
        and _text(event.get("action_type")).lower() == "ticker_change"
        and _date(event.get("effective_date")) == spec["effective_date"]
        and _text(event.get("new_security_id")) == spec["security_id"]
        and _text(event.get("new_symbol")).upper() == spec["successor_symbol"]
        and _text(event.get("evidence_sha256")).lower()
        == spec["official_source_hash"]
        and extraction_hash == spec["reviewed_extraction_sha256"]
    )
    if not exact_target or not exact_event:
        return None
    return {
        "code": "reviewed_nonterminal_same_sid_ticker_no_data",
        "validation_basis": REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_BASIS,
        "event_id": event_id,
        "source_target_id": spec["source_target_id"],
        "successor_target_id": spec["successor_target_id"],
        "security_id": spec["security_id"],
        "old_symbol": spec["old_symbol"],
        "successor_symbol": spec["successor_symbol"],
        "old_active_from": spec["old_active_from"],
        "old_active_to": spec["old_active_to"],
        "successor_active_from": spec["successor_active_from"],
        "effective_date": spec["effective_date"],
        "official_source_hash": spec["official_source_hash"],
        "reviewed_extraction_sha256": extraction_hash,
        "registry_sha256": (
            TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SHA256
        ),
        "same_security_id_continuation": True,
        "terminal_resolution_required": False,
        "terminal_resolution_forbidden": True,
    }


def reviewed_terminal_overrides(
    events_policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the unique exact terminal-heuristic override inventory.

    Terminal overrides intentionally use the same complete action-field shape
    as reviewed nonterminal extractions.  Their separate registry and code pin
    prevent a terminal approval from being inferred from the broader
    nonterminal inventory.
    """

    raw = events_policy.get("reviewed_terminal_overrides")
    _require(
        isinstance(raw, list),
        "Policy reviewed_terminal_overrides must be a list.",
    )
    output: dict[str, dict[str, Any]] = {}
    for value in raw:
        normalized = _canonical_reviewed_nonterminal_extraction(value)
        event_id = str(normalized["event_id"])
        _require(
            event_id not in output,
            f"Duplicate reviewed terminal override event_id: {event_id}",
        )
        output[event_id] = normalized
    return output


def reviewed_terminal_override_mismatches(
    action: Mapping[str, Any],
    override: Mapping[str, Any],
) -> tuple[str, ...]:
    """Compare every corporate-action field with one terminal override."""

    return reviewed_nonterminal_extraction_mismatches(action, override)


def reviewed_terminal_override_sha256(
    override: Mapping[str, Any],
) -> str:
    """Fingerprint one normalized terminal override."""

    return reviewed_nonterminal_extraction_sha256(override)


def reviewed_terminal_override_inventory_sha256(
    events_policy: Mapping[str, Any],
) -> str:
    """Fingerprint the complete terminal override inventory."""

    return canonical_json_sha256(reviewed_terminal_overrides(events_policy))


def _canonical_reviewed_terminal_market_date_correction(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact legal-date to market-session correction."""

    _require(
        isinstance(value, Mapping),
        "Reviewed terminal market-date correction must be an object.",
    )
    _require(
        set(value) == set(REVIEWED_TERMINAL_MARKET_DATE_CORRECTION_FIELDS),
        "Reviewed terminal market-date correction fields are not exact.",
    )

    dates: dict[str, str] = {}
    for field in (
        "report_effective_date",
        "official_completion_date",
        "effective_date",
        "ex_date",
        "announcement_date",
        "last_price_date",
        "filing_date",
    ):
        parsed = _date(value.get(field))
        _require(
            bool(parsed) and _text(value.get(field)) == parsed,
            f"Reviewed terminal market-date correction {field} must be exact ISO date.",
        )
        dates[field] = parsed
    payment_date = _date(value.get("payment_date"))
    _require(
        _text(value.get("payment_date")) in {"", payment_date},
        "Reviewed terminal market-date correction payment_date is invalid.",
    )

    allowed = value.get("allowed_report_mismatches")
    allowed_values = (
        [str(item) for item in allowed] if isinstance(allowed, list) else []
    )
    _require(
        allowed_values
        in (
            ["effective_date"],
            ["effective_date", "source_hash", "source_url"],
        ),
        "Reviewed terminal market-date correction may allow only the exact "
        "date mismatch, or the date plus a reviewed replacement URL/hash.",
    )
    output = {
        "event_id": _text(value.get("event_id")).lower(),
        "superseded_event_id": _text(value.get("superseded_event_id")).lower(),
        "candidate_id": _text(value.get("candidate_id")).lower(),
        "security_id": _text(value.get("security_id")),
        "symbol": _text(value.get("symbol")).upper(),
        "action_type": _text(value.get("action_type")).lower(),
        **dates,
        "payment_date": payment_date,
        "date_relation": _text(value.get("date_relation")),
        "allowed_report_mismatches": allowed_values,
        "new_security_id": _text(value.get("new_security_id")),
        "new_symbol": _text(value.get("new_symbol")).upper(),
        "ratio": _exact_number_text(value.get("ratio"), "ratio"),
        "cash_amount": _exact_number_text(
            value.get("cash_amount"), "cash_amount"
        ),
        "currency": _text(value.get("currency")).upper(),
        "source_kind": _text(value.get("source_kind")),
        "source_url": _text(value.get("source_url")),
        "source_hash": _text(value.get("source_hash")).lower(),
        "report_source_url": _text(value.get("report_source_url")),
        "report_source_hash": _text(value.get("report_source_hash")).lower(),
        "lifecycle_evidence_report_sha256": _text(
            value.get("lifecycle_evidence_report_sha256")
        ).lower(),
        "filing_accession_number": _text(value.get("filing_accession_number")),
    }
    for field in (
        "event_id",
        "superseded_event_id",
        "candidate_id",
        "source_hash",
        "report_source_hash",
        "lifecycle_evidence_report_sha256",
    ):
        digest = str(output[field])
        _require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"Reviewed terminal market-date correction {field} must be SHA-256.",
        )
    _require(
        bool(output["security_id"] and output["symbol"]),
        "Reviewed terminal market-date correction identity is incomplete.",
    )
    action_type = str(output["action_type"])
    terms_valid = False
    if action_type == "cash_merger":
        terms_valid = bool(
            not output["new_security_id"]
            and not output["new_symbol"]
            and output["ratio"] is None
            and output["cash_amount"] is not None
            and Decimal(str(output["cash_amount"])) > 0
        )
    elif action_type == "stock_merger":
        terms_valid = bool(
            output["new_security_id"]
            and output["new_symbol"]
            and output["ratio"] is not None
            and Decimal(str(output["ratio"])) > 0
            and output["cash_amount"] is None
        )
    elif action_type == "ticker_change":
        terms_valid = bool(
            output["new_security_id"]
            and output["new_symbol"]
            and output["ratio"] is None
            and output["cash_amount"] is None
        )
    elif action_type == "delisting":
        terms_valid = bool(
            not output["new_security_id"]
            and not output["new_symbol"]
            and output["ratio"] is None
            and output["cash_amount"] is not None
            and Decimal(str(output["cash_amount"])) >= 0
        )
    _require(
        terms_valid
        and output["currency"] == "USD"
        and output["source_kind"] == "official_crosscheck",
        "Reviewed terminal market-date correction has invalid action terms.",
    )
    _require(
        output["event_id"]
        == canonical_lifecycle_event_id(
            str(output["security_id"]),
            str(output["action_type"]),
            str(output["effective_date"]),
        )
        and output["superseded_event_id"]
        == canonical_lifecycle_event_id(
            str(output["security_id"]),
            str(output["action_type"]),
            str(output["report_effective_date"]),
        )
        and output["candidate_id"]
        == lifecycle_candidate_id(
            str(output["security_id"]), str(output["last_price_date"])
        ),
        "Reviewed terminal market-date correction canonical IDs are invalid.",
    )
    _require(
        output["date_relation"]
        in {
            "next_xnys_session_after_terminal_close",
            "first_successor_trading_session_after_terminal_close",
            "first_xnys_session_after_last_otc_close",
        }
        and output["effective_date"] == output["ex_date"]
        and output["announcement_date"]
        in {output["official_completion_date"], output["effective_date"]}
        and output["official_completion_date"] <= output["effective_date"]
        and output["report_effective_date"] <= output["official_completion_date"]
        and not output["payment_date"]
        and output["report_effective_date"] != output["effective_date"],
        "Reviewed terminal market-date correction date semantics are invalid.",
    )
    terminal = pd.Timestamp(str(output["last_price_date"]))
    effective = pd.Timestamp(str(output["effective_date"]))
    sessions = xcals.get_calendar("XNYS").sessions_in_range(
        terminal + pd.Timedelta(days=1), effective
    )
    normalized = [pd.Timestamp(item).tz_localize(None).normalize() for item in sessions]
    _require(
        normalized == [effective.normalize()],
        "Reviewed terminal market-date correction is not the next XNYS session.",
    )
    _require(
        bool(output["filing_accession_number"]),
        "Reviewed terminal market-date correction filing accession is missing.",
    )
    _require(
        bool(output["source_url"] and output["report_source_url"]),
        "Reviewed terminal market-date correction source URLs are incomplete.",
    )
    replacement_fields = {"source_hash", "source_url"}
    allowed_set = set(output["allowed_report_mismatches"])
    action_report_sources_differ = bool(
        output["source_url"] != output["report_source_url"]
        or output["source_hash"] != output["report_source_hash"]
    )
    _require(
        (action_report_sources_differ and replacement_fields <= allowed_set)
        or (not action_report_sources_differ and not (replacement_fields & allowed_set)),
        "Reviewed terminal market-date replacement provenance is inconsistent.",
    )
    return output


def reviewed_terminal_market_date_corrections(
    events_policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the unique, normalized terminal market-date corrections."""

    raw = events_policy.get("reviewed_terminal_market_date_corrections")
    _require(
        isinstance(raw, list),
        "Policy reviewed_terminal_market_date_corrections must be a list.",
    )
    output: dict[str, dict[str, Any]] = {}
    for value in raw:
        normalized = _canonical_reviewed_terminal_market_date_correction(value)
        event_id = str(normalized["event_id"])
        _require(
            event_id not in output,
            "Duplicate reviewed terminal market-date correction event_id: "
            + event_id,
        )
        output[event_id] = normalized
    return output


def reviewed_terminal_market_date_correction_sha256(
    correction: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(
        _canonical_reviewed_terminal_market_date_correction(correction)
    )


def reviewed_terminal_market_date_correction_inventory_sha256(
    events_policy: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(
        reviewed_terminal_market_date_corrections(events_policy)
    )


def reviewed_terminal_market_date_action_mismatches(
    action: Mapping[str, Any],
    correction: Mapping[str, Any],
) -> tuple[str, ...]:
    """Compare every corrected terminal corporate-action field exactly."""

    expected = _canonical_reviewed_terminal_market_date_correction(correction)
    actual = {
        "event_id": _text(action.get("event_id")).lower(),
        "security_id": _text(action.get("security_id")),
        "action_type": _text(action.get("action_type")).lower(),
        "effective_date": _date(action.get("effective_date")),
        "ex_date": _date(action.get("ex_date")),
        "announcement_date": _date(action.get("announcement_date")),
        "payment_date": _date(action.get("payment_date")),
        "new_security_id": _text(action.get("new_security_id")),
        "new_symbol": _text(action.get("new_symbol")).upper(),
        "ratio": _exact_number_text(action.get("ratio"), "ratio"),
        "cash_amount": _exact_number_text(action.get("cash_amount"), "cash_amount"),
        "currency": _text(action.get("currency")).upper(),
        "source_kind": _text(action.get("source_kind")),
        "source_url": _text(action.get("source_url")),
        "source_hash": _text(action.get("source_hash")).lower(),
    }
    fields = tuple(actual)
    mismatches = [field for field in fields if actual[field] != expected[field]]
    if _text(action.get("official")).lower() != "true":
        mismatches.append("official")
    return tuple(mismatches)


def reviewed_terminal_market_date_report_mismatches(
    action: Mapping[str, Any],
    resolution: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    correction: Mapping[str, Any],
    lifecycle_report_sha256: str,
) -> tuple[str, ...]:
    """Allow only the reviewed parser-date mismatch in one exact report row."""

    expected = _canonical_reviewed_terminal_market_date_correction(correction)
    if not isinstance(record, Mapping):
        return ("lifecycle_report_record",)
    value = record.get("verified_event")
    event = value if isinstance(value, Mapping) else record.get("parsed")
    if not isinstance(event, Mapping):
        return ("lifecycle_report_event",)
    mismatches: list[str] = []
    observed_report_mismatches = set(
        reviewed_terminal_report_mismatches(action, resolution, record)
    )
    sivb_binding = trusted_sivb_evidence_bindings().get(expected["event_id"])
    if sivb_binding is not None and "report_candidate" in sivb_binding:
        if record.get("candidate") == sivb_binding["report_candidate"]:
            observed_report_mismatches -= {
                "candidate_symbol",
                "candidate_last_price_date",
                "candidate_active_to",
            }
        else:
            observed_report_mismatches.add("sivb_report_candidate")
    allowed = set(expected["allowed_report_mismatches"])
    for field in sorted(observed_report_mismatches ^ allowed):
        mismatches.append("lifecycle_report_" + field)
    if _date(event.get("effective_date")) != expected["report_effective_date"]:
        mismatches.append("report_effective_date")
    report_source_url = _text(
        event.get("source_url") or record.get("source_url")
    )
    report_source_hash = _text(
        event.get("source_hash") or record.get("source_hash")
    ).lower()
    if report_source_url != expected["report_source_url"]:
        mismatches.append("report_source_url")
    if report_source_hash != expected["report_source_hash"]:
        mismatches.append("report_source_hash")
    if _text(lifecycle_report_sha256).lower() != expected[
        "lifecycle_evidence_report_sha256"
    ]:
        mismatches.append("lifecycle_evidence_report_sha256")

    resolution_pairs = {
        "candidate_id": (
            _text(resolution.get("candidate_id")).lower(),
            expected["candidate_id"],
        ),
        "resolution_event_id": (
            _text(resolution.get("event_id")).lower(),
            expected["event_id"],
        ),
        "resolution_security_id": (
            _text(resolution.get("security_id")),
            expected["security_id"],
        ),
        "resolution_symbol": (
            _text(resolution.get("symbol")).upper(),
            expected["symbol"],
        ),
        "last_price_date": (
            _date(resolution.get("last_price_date")),
            expected["last_price_date"],
        ),
    }
    mismatches.extend(
        field for field, (actual, wanted) in resolution_pairs.items() if actual != wanted
    )
    candidate = record.get("candidate")
    if not isinstance(candidate, Mapping):
        mismatches.append("lifecycle_report_candidate")
    else:
        candidate_pairs = {
            "candidate_security_id": (
                _text(candidate.get("security_id")),
                expected["security_id"],
            ),
            "candidate_symbol": (
                _text(candidate.get("symbol")).upper(),
                expected["symbol"],
            ),
            "candidate_last_price_date": (
                _date(candidate.get("last_price_date")),
                expected["last_price_date"],
            ),
            "candidate_active_to": (
                _date(candidate.get("active_to")),
                expected["last_price_date"],
            ),
        }
        if sivb_binding is None or "report_candidate" not in sivb_binding:
            mismatches.extend(
                field
                for field, (actual, wanted) in candidate_pairs.items()
                if actual != wanted
            )
    filing = record.get("filing")
    if not isinstance(filing, Mapping):
        mismatches.append("lifecycle_report_filing")
    else:
        if _text(filing.get("accession_number")) != expected[
            "filing_accession_number"
        ]:
            mismatches.append("filing_accession_number")
        if _date(filing.get("filing_date")) != expected["filing_date"]:
            mismatches.append("filing_date")
    return tuple(dict.fromkeys(mismatches))


def _canonical_reviewed_terminal_price_tail_correction(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact provider-tail removal emitted by the repair planner."""

    _require(
        isinstance(value, Mapping),
        "Reviewed terminal price-tail correction must be an object.",
    )
    _require(
        set(value) == set(REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_FIELDS),
        "Reviewed terminal price-tail correction fields are not exact.",
    )
    date_fields = (
        "report_candidate_active_to",
        "report_candidate_last_price_date",
        "report_crosscheck_old_price_session",
        "report_effective_date",
        "official_completion_date",
        "last_real_session",
        "market_transition_session",
        "removed_tail_start",
        "removed_tail_end",
    )
    dates: dict[str, str] = {}
    for field in date_fields:
        parsed = _date(value.get(field))
        _require(
            bool(parsed) and _text(value.get(field)) == parsed,
            f"Reviewed terminal price-tail correction {field} is not exact ISO date.",
        )
        dates[field] = parsed
    index_removals = value.get("index_removals_observed")
    _require(
        isinstance(index_removals, list) and bool(index_removals),
        "Reviewed terminal price-tail correction index removals are missing.",
    )
    normalized_removals: list[dict[str, str]] = []
    for item in index_removals:
        _require(
            isinstance(item, Mapping)
            and set(item) == {"index_id", "effective_date"},
            "Reviewed terminal price-tail correction index removal is not exact.",
        )
        effective_date = _date(item.get("effective_date"))
        index_id = _text(item.get("index_id")).lower()
        _require(
            bool(index_id and effective_date)
            and _text(item.get("effective_date")) == effective_date,
            "Reviewed terminal price-tail correction index removal is invalid.",
        )
        normalized_removals.append(
            {"index_id": index_id, "effective_date": effective_date}
        )
    _require(
        len(normalized_removals)
        == len({(item["index_id"], item["effective_date"]) for item in normalized_removals}),
        "Reviewed terminal price-tail correction index removals are duplicated.",
    )
    ratio = _number(value.get("ratio"))
    cash_amount = _number(value.get("cash_amount"))
    raw_source_bytes = _integer(value.get("raw_source_bytes"), "raw_source_bytes")
    official_source_bytes = _integer(
        value.get("official_source_bytes"), "official_source_bytes"
    )
    removed_tail_count = _integer(
        value.get("removed_tail_count"), "removed_tail_count"
    )
    output: dict[str, Any] = {
        "symbol": _text(value.get("symbol")).upper(),
        "security_id": _text(value.get("security_id")),
        "old_candidate_id": _text(value.get("old_candidate_id")).lower(),
        "candidate_id": _text(value.get("candidate_id")).lower(),
        "old_event_id": _text(value.get("old_event_id")).lower(),
        "event_id": _text(value.get("event_id")).lower(),
        "action_type": _text(value.get("action_type")).lower(),
        **dates,
        "date_relation": _text(value.get("date_relation")),
        "new_security_id": _text(value.get("new_security_id")),
        "new_symbol": _text(value.get("new_symbol")).upper(),
        "ratio": ratio,
        "cash_amount": cash_amount,
        "raw_source_url": _text(value.get("raw_source_url")),
        "raw_source_hash": _text(value.get("raw_source_hash")).lower(),
        "raw_source_bytes": raw_source_bytes,
        "removed_tail_count": removed_tail_count,
        "removed_tail_sha256": _text(value.get("removed_tail_sha256")).lower(),
        "official_source_url": _text(value.get("official_source_url")),
        "official_source_hash": _text(value.get("official_source_hash")).lower(),
        "official_source_bytes": official_source_bytes,
        "filing_accession_number": _text(value.get("filing_accession_number")),
        "filing_acceptance_datetime": _text(
            value.get("filing_acceptance_datetime")
        ),
        "successor_source_hash": _text(value.get("successor_source_hash")).lower(),
        "index_removals_observed": normalized_removals,
        "lifecycle_evidence_report_sha256": _text(
            value.get("lifecycle_evidence_report_sha256")
        ).lower(),
        "registry_item_sha256": _text(value.get("registry_item_sha256")).lower(),
    }
    digest_fields = (
        "old_candidate_id",
        "candidate_id",
        "old_event_id",
        "event_id",
        "raw_source_hash",
        "removed_tail_sha256",
        "official_source_hash",
        "lifecycle_evidence_report_sha256",
        "registry_item_sha256",
    )
    for field in digest_fields:
        digest = str(output[field])
        _require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"Reviewed terminal price-tail correction {field} must be SHA-256.",
        )
    _require(
        bool(output["symbol"] and output["security_id"])
        and output["action_type"]
        in {"stock_merger", "cash_merger", "ticker_change"},
        "Reviewed terminal price-tail correction action identity is invalid.",
    )
    successor_source_hash = str(output["successor_source_hash"])
    successor_hash_valid = len(successor_source_hash) == 64 and all(
        character in "0123456789abcdef" for character in successor_source_hash
    )
    if output["action_type"] == "stock_merger":
        terms_valid = bool(output["new_security_id"] and output["new_symbol"])
        terms_valid = bool(
            terms_valid
            and ratio is not None
            and ratio > 0
            and cash_amount is None
            and successor_hash_valid
        )
    elif output["action_type"] == "cash_merger":
        terms_valid = bool(
            not output["new_security_id"]
            and not output["new_symbol"]
            and ratio is None
            and cash_amount is not None
            and cash_amount > 0
            and not successor_source_hash
        )
    else:
        terms_valid = bool(
            output["new_security_id"]
            and output["new_symbol"]
            and ratio is None
            and cash_amount is None
            and successor_hash_valid
        )
    _require(
        terms_valid,
        "Reviewed terminal price-tail correction action terms are invalid.",
    )
    _require(
        output["event_id"]
        == canonical_lifecycle_event_id(
            str(output["security_id"]),
            str(output["action_type"]),
            str(output["market_transition_session"]),
        )
        and output["old_event_id"]
        == canonical_lifecycle_event_id(
            str(output["security_id"]),
            str(output["action_type"]),
            str(output["report_effective_date"]),
        )
        and output["candidate_id"]
        == lifecycle_candidate_id(
            str(output["security_id"]), str(output["last_real_session"])
        )
        and output["old_candidate_id"]
        == lifecycle_candidate_id(
            str(output["security_id"]),
            str(output["report_candidate_last_price_date"]),
        ),
        "Reviewed terminal price-tail correction canonical IDs are invalid.",
    )
    _require(
        output["date_relation"] == "next_xnys_session_after_terminal_close"
        and output["report_candidate_active_to"]
        == output["report_candidate_last_price_date"]
        == output["removed_tail_end"]
        and output["report_effective_date"] == output["official_completion_date"]
        and output["removed_tail_start"] == output["market_transition_session"]
        and output["last_real_session"] < output["removed_tail_start"]
        and output["removed_tail_start"] <= output["removed_tail_end"]
        and removed_tail_count > 0
        and raw_source_bytes > 0
        and official_source_bytes > 0,
        "Reviewed terminal price-tail correction boundary semantics are invalid.",
    )
    terminal = pd.Timestamp(str(output["last_real_session"]))
    transition = pd.Timestamp(str(output["market_transition_session"]))
    sessions = xcals.get_calendar("XNYS").sessions_in_range(
        terminal + pd.Timedelta(days=1), transition
    )
    normalized_sessions = [
        pd.Timestamp(item).tz_localize(None).normalize() for item in sessions
    ]
    _require(
        normalized_sessions == [transition.normalize()],
        "Reviewed terminal price-tail transition is not the next XNYS session.",
    )
    raw_url = urlparse(str(output["raw_source_url"]))
    official_url = urlparse(str(output["official_source_url"]))
    _require(
        raw_url.scheme == "https"
        and (raw_url.hostname or "").lower() == "eodhd.com"
        and raw_url.path.startswith("/api/eod/")
        and official_url.scheme == "https"
        and (official_url.hostname or "").lower() == "www.sec.gov"
        and official_url.path.startswith("/Archives/edgar/data/"),
        "Reviewed terminal price-tail evidence URLs are invalid.",
    )
    _require(
        bool(output["filing_accession_number"])
        and len(str(output["filing_acceptance_datetime"])) == 14
        and str(output["filing_acceptance_datetime"]).isdigit(),
        "Reviewed terminal price-tail filing identity is invalid.",
    )
    item_without_hash = {
        key: item for key, item in output.items() if key != "registry_item_sha256"
    }
    _require(
        canonical_json_sha256(item_without_hash) == output["registry_item_sha256"],
        "Reviewed terminal price-tail registry item hash is invalid.",
    )
    return output


def reviewed_terminal_price_tail_corrections(
    events_policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = events_policy.get("reviewed_terminal_price_tail_corrections")
    _require(
        isinstance(raw, list),
        "Policy reviewed_terminal_price_tail_corrections must be a list.",
    )
    output: dict[str, dict[str, Any]] = {}
    for value in raw:
        normalized = _canonical_reviewed_terminal_price_tail_correction(value)
        event_id = str(normalized["event_id"])
        _require(
            event_id not in output,
            "Duplicate reviewed terminal price-tail correction event_id: " + event_id,
        )
        output[event_id] = normalized
    return output


def reviewed_terminal_price_tail_correction_sha256(
    correction: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(
        _canonical_reviewed_terminal_price_tail_correction(correction)
    )


def reviewed_terminal_price_tail_correction_inventory_sha256(
    events_policy: Mapping[str, Any],
) -> str:
    raw = events_policy.get("reviewed_terminal_price_tail_corrections")
    _require(isinstance(raw, list), "Terminal price-tail inventory must be a list.")
    normalized = [
        _canonical_reviewed_terminal_price_tail_correction(value) for value in raw
    ]
    return canonical_json_sha256(normalized)


def reviewed_terminal_price_tail_action_mismatches(
    action: Mapping[str, Any],
    correction: Mapping[str, Any],
) -> tuple[str, ...]:
    expected = _canonical_reviewed_terminal_price_tail_correction(correction)
    actual = {
        "event_id": _text(action.get("event_id")).lower(),
        "security_id": _text(action.get("security_id")),
        "action_type": _text(action.get("action_type")).lower(),
        "effective_date": _date(action.get("effective_date")),
        "ex_date": _date(action.get("ex_date")),
        "announcement_date": _date(action.get("announcement_date")),
        "record_date": _date(action.get("record_date")),
        "payment_date": _date(action.get("payment_date")),
        "new_security_id": _text(action.get("new_security_id")),
        "new_symbol": _text(action.get("new_symbol")).upper(),
        "ratio": _number(action.get("ratio")),
        "cash_amount": _number(action.get("cash_amount")),
        "currency": _text(action.get("currency")).upper(),
        "source_kind": _text(action.get("source_kind")),
        "source_url": _text(action.get("source_url")),
        "source_hash": _text(action.get("source_hash")).lower(),
        "official": _text(action.get("official")).lower(),
    }
    wanted = {
        "event_id": expected["event_id"],
        "security_id": expected["security_id"],
        "action_type": expected["action_type"],
        "effective_date": expected["market_transition_session"],
        "ex_date": expected["market_transition_session"],
        "announcement_date": expected["official_completion_date"],
        "record_date": "",
        "payment_date": "",
        "new_security_id": expected["new_security_id"],
        "new_symbol": expected["new_symbol"],
        "ratio": expected["ratio"],
        "cash_amount": expected["cash_amount"],
        "currency": "USD",
        "source_kind": "official_crosscheck",
        "source_url": expected["official_source_url"],
        "source_hash": expected["official_source_hash"],
        "official": "true",
    }
    return tuple(field for field in actual if actual[field] != wanted[field])


def reviewed_terminal_price_tail_report_mismatches(
    action: Mapping[str, Any],
    resolution: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    archive: pd.DataFrame,
    correction: Mapping[str, Any],
    lifecycle_report_sha256: str,
) -> tuple[str, ...]:
    expected = _canonical_reviewed_terminal_price_tail_correction(correction)
    if not isinstance(record, Mapping):
        return ("lifecycle_report_record",)
    event = record.get("verified_event")
    if not isinstance(event, Mapping):
        event = record.get("parsed")
    if not isinstance(event, Mapping):
        return ("lifecycle_report_event",)
    mismatches: list[str] = []
    observed = set(reviewed_terminal_report_mismatches(action, resolution, record))
    allowed = {"effective_date"} if expected["old_event_id"] != expected["event_id"] else set()
    mismatches.extend("lifecycle_report_" + field for field in sorted(observed ^ allowed))
    report_pairs = {
        "report_effective_date": (
            _date(event.get("effective_date")),
            expected["report_effective_date"],
        ),
        "report_source_url": (
            _text(event.get("source_url") or record.get("source_url")),
            expected["official_source_url"],
        ),
        "report_source_hash": (
            _text(event.get("source_hash") or record.get("source_hash")).lower(),
            expected["official_source_hash"],
        ),
        "lifecycle_evidence_report_sha256": (
            _text(lifecycle_report_sha256).lower(),
            expected["lifecycle_evidence_report_sha256"],
        ),
        "resolution_candidate_id": (
            _text(resolution.get("candidate_id")).lower(),
            expected["candidate_id"],
        ),
        "resolution_event_id": (
            _text(resolution.get("event_id")).lower(),
            expected["event_id"],
        ),
        "resolution_security_id": (
            _text(resolution.get("security_id")),
            expected["security_id"],
        ),
        "resolution_symbol": (
            _text(resolution.get("symbol")).upper(),
            expected["symbol"],
        ),
        "resolution_last_price_date": (
            _date(resolution.get("last_price_date")),
            expected["last_real_session"],
        ),
        "resolution_successor_security_id": (
            _text(resolution.get("successor_security_id")),
            expected["new_security_id"],
        ),
        "resolution_successor_symbol": (
            _text(resolution.get("successor_symbol")).upper(),
            expected["new_symbol"],
        ),
        "resolution_source_url": (
            _text(resolution.get("source_url")),
            expected["official_source_url"],
        ),
        "resolution_source_hash": (
            _text(resolution.get("source_hash")).lower(),
            expected["official_source_hash"],
        ),
    }
    mismatches.extend(
        field for field, (actual, wanted) in report_pairs.items() if actual != wanted
    )
    if _text(resolution.get("resolution")).lower() != "applied":
        mismatches.append("resolution_kind")
    candidate = record.get("candidate")
    if not isinstance(candidate, Mapping):
        mismatches.append("lifecycle_report_candidate")
    else:
        candidate_pairs = {
            "candidate_security_id": (
                _text(candidate.get("security_id")), expected["security_id"]
            ),
            "candidate_symbol": (
                _text(candidate.get("symbol")).upper(), expected["symbol"]
            ),
            "candidate_last_price_date": (
                _date(candidate.get("last_price_date")),
                expected["report_candidate_last_price_date"],
            ),
            "candidate_active_to": (
                _date(candidate.get("active_to")),
                expected["report_candidate_active_to"],
            ),
            "candidate_index_remove_dates": (
                sorted(
                    _date(item)
                    for item in candidate.get("index_remove_dates", [])
                    if _date(item)
                )
                if isinstance(candidate.get("index_remove_dates"), list)
                else [],
                sorted(
                    item["effective_date"]
                    for item in expected["index_removals_observed"]
                ),
            ),
        }
        mismatches.extend(
            field for field, (actual, wanted) in candidate_pairs.items() if actual != wanted
        )
        if lifecycle_candidate_id(
            expected["security_id"], _date(candidate.get("last_price_date"))
        ) != expected["old_candidate_id"]:
            mismatches.append("old_candidate_id")
    crosscheck = record.get("crosscheck")
    if not isinstance(crosscheck, Mapping) or _date(
        crosscheck.get("old_price_session")
    ) != expected["report_crosscheck_old_price_session"]:
        mismatches.append("report_crosscheck_old_price_session")
    filing = record.get("filing")
    if not isinstance(filing, Mapping):
        mismatches.append("lifecycle_report_filing")
    else:
        if _text(filing.get("accession_number")) != expected["filing_accession_number"]:
            mismatches.append("filing_accession_number")
        if _date(filing.get("filing_date")) != expected["official_completion_date"]:
            mismatches.append("filing_date")
    archive_expected = [
        (expected["official_source_hash"], expected["official_source_url"]),
        (expected["raw_source_hash"], expected["raw_source_url"]),
    ]
    if expected["successor_source_hash"]:
        archive_expected.append((expected["successor_source_hash"], None))
    for digest, source_url in archive_expected:
        matches = archive.loc[archive["archive_id"].map(_text).eq(digest)]
        if len(matches) != 1 or _text(matches.iloc[0].get("source_hash")).lower() != digest:
            mismatches.append("archive_" + digest[:12])
        elif source_url is not None and _text(matches.iloc[0].get("source_url")) != source_url:
            mismatches.append("archive_url_" + digest[:12])
    return tuple(dict.fromkeys(mismatches))


def reviewed_terminal_report_mismatches(
    action: Mapping[str, Any],
    resolution: Mapping[str, Any],
    record: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Compare exact terminal terms with the collector's parsed report row."""

    if not isinstance(record, Mapping):
        return ("lifecycle_report_record",)
    value = record.get("verified_event")
    event = value if isinstance(value, Mapping) else record.get("parsed")
    if not isinstance(event, Mapping):
        return ("lifecycle_report_event",)

    mismatches: list[str] = []
    exact_pairs = {
        "action_type": (
            _text(action.get("action_type")).lower(),
            _text(event.get("action_type")).lower(),
        ),
        "effective_date": (
            _date(action.get("effective_date")),
            _date(event.get("effective_date")),
        ),
        "new_symbol": (
            _text(action.get("new_symbol")).upper(),
            _text(event.get("new_symbol")).upper(),
        ),
        "ratio": (
            _exact_number_text(action.get("ratio"), "ratio"),
            _exact_number_text(event.get("ratio"), "ratio"),
        ),
        "cash_amount": (
            _exact_number_text(action.get("cash_amount"), "cash_amount"),
            _exact_number_text(event.get("cash_amount"), "cash_amount"),
        ),
        "source_url": (
            _text(action.get("source_url")),
            _text(event.get("source_url") or record.get("source_url")),
        ),
        "source_hash": (
            _text(action.get("source_hash")).lower(),
            _text(event.get("source_hash") or record.get("source_hash")).lower(),
        ),
    }
    mismatches.extend(
        field for field, (actual, expected) in exact_pairs.items() if actual != expected
    )
    successor = _text(resolution.get("successor_security_id"))
    if _text(action.get("new_security_id")) != successor:
        mismatches.append("new_security_id")
    record_successor = _text(record.get("successor_security_id"))
    if record_successor != successor:
        mismatches.append("lifecycle_report_successor_security_id")
    candidate = record.get("candidate")
    if not isinstance(candidate, Mapping) or _text(
        candidate.get("security_id")
    ) != _text(action.get("security_id")):
        mismatches.append("lifecycle_report_security_id")
    if _text(action.get("currency") or "USD").upper() != "USD":
        mismatches.append("currency")
    return tuple(mismatches)


def pinned_external_overlap_spec_is_trusted(spec: Mapping[str, Any]) -> bool:
    """Match a report/config entry to the code-pinned CC0 artifact contract."""

    symbol = _text(spec.get("symbol")).upper()
    expected = TRUSTED_PINNED_EXTERNAL_OVERLAPS.get(symbol)
    if expected is None or set(spec) != {"symbol", *expected}:
        return False
    return all(spec.get(key) == value for key, value in expected.items())


def _nonterminal_terms_complete(action: Mapping[str, Any]) -> bool:
    """Reproduce the collector's stored-terms checks without a terminal report."""

    action_type = _text(action.get("action_type")).lower()
    if not _date(action.get("effective_date")):
        return False
    if _text(action.get("currency") or "USD").upper() != "USD":
        return False
    if action_type in {"stock_merger", "spinoff", "ticker_change"} and not (
        _text(action.get("new_security_id")) and _text(action.get("new_symbol"))
    ):
        return False
    if action_type == "stock_merger":
        value = _number(action.get("ratio"))
        return value is not None and value > 0
    if action_type == "spinoff":
        value = _number(action.get("ratio"))
        return (
            value is not None
            and value > 0
            and _number(action.get("cash_amount")) is None
        )
    if action_type == "cash_merger":
        value = _number(action.get("cash_amount"))
        return value is not None and value > 0
    if action_type == "delisting":
        value = _number(action.get("cash_amount"))
        return value is not None and value >= 0
    return action_type == "ticker_change"


def _official_url(value: Any, allowed_hosts: set[str]) -> bool:
    parsed = urlparse(_text(value))
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and parsed.username is None
        and parsed.password is None
        and any(host == allowed or host.endswith("." + allowed) for allowed in allowed_hosts)
    )


def _price_target_id(
    security_id: str,
    symbol: str,
    active_from: str,
    active_to: str,
) -> str:
    return canonical_json_sha256(
        {
            "provider": INDEPENDENT_PRICE_PROVIDER,
            "security_id": security_id,
            "provider_symbol": normalize_yahoo_symbol(symbol),
            "active_from": active_from,
            "active_to": active_to,
        }
    )


def _expected_price_targets(
    master: pd.DataFrame,
    history: pd.DataFrame,
    security_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Expand each in-scope identity to every stored symbol-history interval."""

    _require(
        {"security_id", "primary_symbol"}.issubset(master.columns),
        "security_master lacks fields required for exact price targets.",
    )
    _require(
        {"security_id", "symbol", "effective_from", "effective_to"}.issubset(
            history.columns
        ),
        "symbol_history lacks fields required for exact price targets.",
    )
    master_rows: dict[str, dict[str, Any]] = {}
    for security_id, group in master.assign(
        _security_id=master["security_id"].map(_text)
    ).groupby("_security_id", sort=False):
        if security_id:
            _require(
                len(group) == 1,
                f"security_master identity is duplicated: {security_id}",
            )
            master_rows[security_id] = group.iloc[0].to_dict()

    output: dict[str, dict[str, str]] = {}
    for security_id in sorted(security_ids):
        intervals = history.loc[history["security_id"].map(_text).eq(security_id)].copy()
        rows: list[dict[str, str]] = []
        for row in intervals.to_dict(orient="records"):
            symbol = _text(row.get("symbol")).upper()
            active_from = _date(row.get("effective_from"))
            active_to = _date(row.get("effective_to"))
            _require(
                bool(symbol and active_from),
                f"symbol_history interval is incomplete for {security_id}.",
            )
            rows.append(
                {
                    "security_id": security_id,
                    "symbol": symbol,
                    "provider_symbol": normalize_yahoo_symbol(symbol),
                    "active_from": active_from,
                    "active_to": active_to,
                }
            )
        if not rows:
            master_row = master_rows.get(security_id)
            _require(
                master_row is not None,
                f"Price target identity is absent from security_master: {security_id}",
            )
            symbol = _text(master_row.get("primary_symbol")).upper()
            _require(bool(symbol), f"Price target symbol is missing for {security_id}.")
            rows = [
                {
                    "security_id": security_id,
                    "symbol": symbol,
                    "provider_symbol": normalize_yahoo_symbol(symbol),
                    "active_from": _date(master_row.get("active_from")),
                    "active_to": _date(master_row.get("active_to")),
                }
            ]
        seen_intervals: set[tuple[str, str, str]] = set()
        for spec in rows:
            interval_key = (
                spec["provider_symbol"],
                spec["active_from"],
                spec["active_to"],
            )
            _require(
                interval_key not in seen_intervals,
                f"Duplicate symbol_history price target for {security_id}.",
            )
            seen_intervals.add(interval_key)
            target_id = _price_target_id(
                security_id,
                spec["provider_symbol"],
                spec["active_from"],
                spec["active_to"],
            )
            _require(target_id not in output, "Duplicate canonical price target_id.")
            output[target_id] = spec
    return output


def _internal_target_sessions(
    prices: pd.DataFrame,
    target: Mapping[str, Any],
) -> pd.DatetimeIndex:
    _require(
        {"security_id", "session"}.issubset(prices.columns),
        "daily_price_raw lacks session fields required for full-history coverage.",
    )
    security_id = _text(target.get("security_id"))
    rows = prices.loc[prices["security_id"].map(_text).eq(security_id)].copy()
    rows = rows.loc[~independent_provider_source_mask(rows)].copy()
    parsed = pd.to_datetime(rows["session"], errors="coerce").dt.normalize()
    _require(
        not bool(parsed.isna().any()),
        f"Internal price sessions are invalid for {security_id}.",
    )
    active_from = _date(target.get("active_from"))
    active_to = _date(target.get("active_to"))
    if active_from:
        rows = rows.loc[parsed.ge(pd.Timestamp(active_from))].copy()
        parsed = parsed.loc[rows.index]
    if active_to:
        rows = rows.loc[parsed.le(pd.Timestamp(active_to))].copy()
        parsed = parsed.loc[rows.index]
    _require(
        not bool(parsed.duplicated().any()),
        f"Internal price sessions are duplicated for {security_id}.",
    )
    return pd.DatetimeIndex(parsed.sort_values())


def _provider_target_sessions(
    bars: pd.DataFrame,
    target: Mapping[str, Any],
) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(bars["session"], errors="coerce").dt.normalize()
    _require(not bool(parsed.isna().any()), "Archived Yahoo sessions are invalid.")
    active_from = _date(target.get("active_from"))
    active_to = _date(target.get("active_to"))
    if active_from:
        parsed = parsed.loc[parsed.ge(pd.Timestamp(active_from))]
    if active_to:
        parsed = parsed.loc[parsed.le(pd.Timestamp(active_to))]
    _require(
        not bool(parsed.duplicated().any()),
        "Archived Yahoo sessions are duplicated inside one identity interval.",
    )
    return pd.DatetimeIndex(parsed.sort_values())


def _all_internal_target_rows(
    prices: pd.DataFrame,
    target: Mapping[str, Any],
) -> pd.DataFrame:
    security_id = _text(target.get("security_id"))
    rows = prices.loc[prices["security_id"].map(_text).eq(security_id)].copy()
    parsed = pd.to_datetime(rows["session"], errors="coerce").dt.normalize()
    _require(
        not bool(parsed.isna().any()),
        f"Internal price sessions are invalid for {security_id}.",
    )
    active_from = _date(target.get("active_from"))
    active_to = _date(target.get("active_to"))
    if active_from:
        rows = rows.loc[parsed.ge(pd.Timestamp(active_from))].copy()
        parsed = parsed.loc[rows.index]
    if active_to:
        rows = rows.loc[parsed.le(pd.Timestamp(active_to))].copy()
        parsed = parsed.loc[rows.index]
    _require(
        not bool(parsed.duplicated().any()),
        f"Internal price sessions are duplicated for {security_id}.",
    )
    return rows.assign(_session=parsed).sort_values("_session", kind="stable")


def _xnys_session_strings(start: str, end: str) -> tuple[str, ...]:
    sessions = xcals.get_calendar("XNYS").sessions_in_range(start, end)
    return tuple(
        pd.Timestamp(value).tz_localize(None).date().isoformat() for value in sessions
    )


def _pinned_overlap_spec(
    policy: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any] | None:
    key = (
        _text(target.get("symbol")).upper(),
        _date(target.get("active_from")),
        _date(target.get("active_to")),
    )
    matches = [
        dict(spec)
        for spec in policy["prices"].get("pinned_external_overlaps") or ()
        if isinstance(spec, dict)
        and (
            _text(spec.get("symbol")).upper(),
            _date(spec.get("active_from")),
            _date(spec.get("active_to")),
        )
        == key
    ]
    _require(len(matches) <= 1, "Pinned external overlap policy is duplicated.")
    return matches[0] if matches else None


def _parse_pinned_external_payload(
    payload: bytes, spec: Mapping[str, Any]
) -> pd.DataFrame:
    symbol = _text(spec.get("symbol")).upper()
    _require(
        not payload.lstrip().startswith((b"<", b"<!")),
        f"Pinned {symbol} external payload is HTML.",
    )
    try:
        raw = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise RuntimeError(f"Pinned {symbol} external CSV is unreadable.") from exc
    columns = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
    _require(list(raw.columns) == columns, f"Pinned {symbol} external schema changed.")
    _require(
        len(raw) == _integer(spec.get("raw_rows"), "raw_rows"),
        f"Pinned {symbol} external raw row count changed.",
    )
    sessions = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="coerce")
    _require(not bool(sessions.isna().any()), f"Pinned {symbol} dates are invalid.")
    raw["session"] = sessions.dt.date.astype(str)
    _require(
        not bool(raw["session"].duplicated().any())
        and bool(raw["session"].is_monotonic_increasing),
        f"Pinned {symbol} sessions are not unique and sorted.",
    )
    for column in ("Open", "High", "Low", "Close", "Volume", "OpenInt"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    numeric = raw[["Open", "High", "Low", "Close", "Volume", "OpenInt"]]
    finite = numeric.apply(lambda values: values.map(math.isfinite)).all().all()
    coherent = (
        numeric[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
        & numeric["Volume"].ge(0)
        & numeric["OpenInt"].ge(0)
        & numeric["High"].ge(numeric[["Open", "Low", "Close"]].max(axis=1))
        & numeric["Low"].le(numeric[["Open", "High", "Close"]].min(axis=1))
    )
    _require(
        bool(finite) and bool(coherent.all()),
        f"Pinned {symbol} external OHLCV is invalid.",
    )
    start = _date(spec.get("overlap_start"))
    end = _date(spec.get("overlap_end"))
    segment = raw.loc[raw["session"].ge(start) & raw["session"].le(end)].copy()
    expected = _xnys_session_strings(start, end)
    _require(
        len(expected) == _integer(spec.get("overlap_sessions"), "overlap_sessions")
        and tuple(segment["session"].astype(str)) == expected,
        f"Pinned {symbol} external overlap coverage changed.",
    )
    return segment.rename(columns={"Close": "close"}).loc[:, ["session", "close"]]


def _recompute_pinned_overlap(
    internal_rows: pd.DataFrame,
    external_rows: pd.DataFrame,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    expected_primary = _xnys_session_strings(
        _date(spec.get("active_from")), _date(spec.get("active_to"))
    )
    internal_sessions = internal_rows["_session"].dt.date.astype(str)
    _require(
        len(expected_primary)
        == _integer(spec.get("primary_sessions"), "primary_sessions")
        and tuple(internal_sessions) == expected_primary,
        "Pinned overlap internal primary coverage changed.",
    )
    left = pd.DataFrame(
        {
            "session": internal_sessions,
            "close_primary": pd.to_numeric(internal_rows["close"], errors="coerce"),
        }
    )
    right = external_rows.rename(columns={"close": "close_external"})
    joined = left.merge(right, on="session", validate="one_to_one").sort_values(
        "session", kind="stable"
    )
    expected_overlap = _xnys_session_strings(
        _date(spec.get("overlap_start")), _date(spec.get("overlap_end"))
    )
    _require(
        tuple(joined["session"].astype(str)) == expected_overlap,
        "Pinned external overlap sessions changed.",
    )
    ratio = joined["close_primary"] / joined["close_external"]
    scale = float(ratio.median())
    normalized_error = (ratio / scale - 1.0).abs()
    correlation = float(
        joined["close_primary"].pct_change().corr(
            joined["close_external"].pct_change()
        )
    )
    p99 = float(normalized_error.quantile(0.99))
    tail = tuple(
        value
        for value in internal_sessions
        if value > _date(spec.get("overlap_end"))
    )
    _require(
        math.isfinite(scale)
        and scale > 0
        and math.isfinite(correlation)
        and correlation >= float(spec["minimum_return_correlation"])
        and math.isfinite(p99)
        and p99 <= float(spec["maximum_p99_scaled_close_error"])
        and len(tail)
        == _integer(
            spec.get("uncrosschecked_tail_sessions"),
            "uncrosschecked_tail_sessions",
        ),
        "Pinned external overlap metrics failed.",
    )
    return {
        "overlap_session_count": len(joined),
        "internal_history_session_count": len(internal_rows),
        "internal_history_start": internal_sessions.iloc[0],
        "internal_history_end": internal_sessions.iloc[-1],
        "external_overlap_start": joined["session"].iloc[0],
        "external_overlap_end": joined["session"].iloc[-1],
        "uncrosschecked_tail_sessions": len(tail),
        "uncrosschecked_tail_start": tail[0],
        "uncrosschecked_tail_end": tail[-1],
        "median_primary_to_external_close_scale": scale,
        "return_correlation": correlation,
        "p99_scaled_close_error": p99,
    }


def _terminal_window_detail(
    sessions: pd.DatetimeIndex,
    count: int,
) -> tuple[bool, dict[str, Any]]:
    if sessions.empty:
        return False, {
            "terminal_session": "",
            "expected_sessions": count,
            "present_sessions": 0,
            "missing": [],
        }
    terminal = sessions.max()
    calendar = xcals.get_calendar("XNYS")
    available = calendar.sessions_in_range(
        terminal - pd.Timedelta(days=count * 3), terminal
    )
    expected = tuple(
        pd.Timestamp(value).tz_localize(None).normalize() for value in available[-count:]
    )
    present = set(sessions)
    missing = [value.date().isoformat() for value in expected if value not in present]
    detail = {
        "terminal_session": terminal.date().isoformat(),
        "expected_sessions": count,
        "present_sessions": count - len(missing),
        "missing": missing,
    }
    return len(expected) == count and not missing, detail


def _terminal_event_date_matches(
    active_to: str,
    terminal_session: str,
    effective_date: str,
    *,
    identity_date_basis: str = "",
    derived_identity_active_to: str = "",
    terminal_calendar_complete: bool = False,
) -> bool:
    matched, expected_basis, expected_derived_active_to = (
        _terminal_event_date_binding(
            active_to,
            terminal_session,
            effective_date,
            terminal_calendar_complete=terminal_calendar_complete,
        )
    )
    return bool(
        matched
        and _text(identity_date_basis) in {"", expected_basis}
        and _date(derived_identity_active_to) == expected_derived_active_to
    )


def _terminal_event_date_binding(
    active_to: str,
    terminal_session: str,
    effective_date: str,
    *,
    terminal_calendar_complete: bool,
) -> tuple[bool, str, str]:
    """Bind one official event to the last complete local trading calendar.

    ``symbol_history.effective_to`` is an identity boundary, not necessarily a
    traded session.  It may therefore be a weekend immediately before the new
    identity starts, or the official completion session on which the retired
    identity no longer traded.  The accepted relations below are deliberately
    narrow: there may be no unaccounted XNYS session between the last stored
    price and the official event.  Open-ended identities additionally require
    the complete terminal-calendar proof before a boundary can be derived.
    """

    stored_active_to = _date(active_to)
    terminal = _date(terminal_session)
    effective = _date(effective_date)
    if not terminal or not effective:
        return False, "", ""

    terminal_day = pd.Timestamp(terminal)
    effective_day = pd.Timestamp(effective)
    calendar = xcals.get_calendar("XNYS")

    def sessions_after(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
        if end <= start:
            return []
        sessions = calendar.sessions_in_range(start + pd.Timedelta(days=1), end)
        return [pd.Timestamp(value).tz_localize(None).normalize() for value in sessions]

    if stored_active_to:
        basis = "stored_identity_active_to"
        boundary_day = pd.Timestamp(stored_active_to)
        if terminal_day > boundary_day or effective_day < terminal_day:
            return False, "", ""
        if terminal_day == boundary_day:
            between = sessions_after(boundary_day, effective_day)
            matched = effective_day == boundary_day or between == [effective_day]
            return (matched, basis, "") if matched else (False, "", "")

        # No traded identity session may be skipped.  The sole permitted XNYS
        # session between the last price and the stored boundary is the event
        # session itself, when the security ceased trading before the open.
        to_boundary = sessions_after(terminal_day, boundary_day)
        if effective_day == boundary_day and to_boundary == [boundary_day]:
            return True, basis, ""
        if to_boundary:
            return False, "", ""
        after_boundary = sessions_after(boundary_day, effective_day)
        matched = effective_day == boundary_day or after_boundary == [effective_day]
        return (matched, basis, "") if matched else (False, "", "")

    if not terminal_calendar_complete or effective_day < terminal_day:
        return False, "", ""
    basis = "derived_local_terminal_session"
    if effective_day == terminal_day:
        return True, basis, terminal
    after_terminal = sessions_after(terminal_day, effective_day)
    if after_terminal == [effective_day]:
        return True, basis, terminal
    # An exact official legal completion can occur over the weekend before the
    # next exchange session.  It is accepted only inside that sessionless gap.
    next_session = pd.Timestamp(calendar.next_session(terminal_day)).tz_localize(
        None
    ).normalize()
    if terminal_day < effective_day < next_session and not after_terminal:
        return True, basis, terminal
    return False, "", ""


_REVIEWED_NO_DATA_UNSUPPORTED_PATH_FIELDS = frozenset(
    {
        "target_id",
        "basis",
        "security_id",
        "provider_symbol",
        "identity_active_from",
        "identity_active_to",
        "last_price_date",
        "candidate_id",
        "event_id",
        "action_type",
        "event_effective_date",
        "official_source_url",
        "official_evidence_sha256",
        "source_sha256",
        "cache_wrapper_sha256",
        "internal_price_source_hash",
        "expected_internal_price_rows",
        "valuation_mark_close",
        "cash_available_session",
        "price_history_supported",
        "generic_date_tolerance",
        "index_scope",
        "limitation",
    }
)


def _canonical_reviewed_no_data_unsupported_path(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one exact official valuation-mark/no-price-path exception."""

    _require(
        isinstance(value, Mapping)
        and set(value) == set(_REVIEWED_NO_DATA_UNSUPPORTED_PATH_FIELDS),
        "Reviewed no-data unsupported-path fields are not exact.",
    )
    output: dict[str, Any] = {
        "target_id": _text(value.get("target_id")).lower(),
        "basis": _text(value.get("basis")),
        "security_id": _text(value.get("security_id")),
        "provider_symbol": _text(value.get("provider_symbol")).upper(),
        "identity_active_from": _date(value.get("identity_active_from")),
        "identity_active_to": _date(value.get("identity_active_to")),
        "last_price_date": _date(value.get("last_price_date")),
        "candidate_id": _text(value.get("candidate_id")).lower(),
        "event_id": _text(value.get("event_id")).lower(),
        "action_type": _text(value.get("action_type")).lower(),
        "event_effective_date": _date(value.get("event_effective_date")),
        "official_source_url": _text(value.get("official_source_url")),
        "official_evidence_sha256": _text(
            value.get("official_evidence_sha256")
        ).lower(),
        "source_sha256": _text(value.get("source_sha256")).lower(),
        "cache_wrapper_sha256": _text(
            value.get("cache_wrapper_sha256")
        ).lower(),
        "internal_price_source_hash": _text(
            value.get("internal_price_source_hash")
        ).lower(),
        "expected_internal_price_rows": _integer(
            value.get("expected_internal_price_rows"),
            "expected_internal_price_rows",
        ),
        "valuation_mark_close": _exact_number_text(
            value.get("valuation_mark_close"), "valuation_mark_close"
        ),
        "cash_available_session": _date(value.get("cash_available_session")),
        "price_history_supported": value.get("price_history_supported"),
        "generic_date_tolerance": value.get("generic_date_tolerance"),
        "index_scope": _text(value.get("index_scope")),
        "limitation": _text(value.get("limitation")),
    }
    hash_fields = (
        "target_id",
        "candidate_id",
        "event_id",
        "official_evidence_sha256",
        "source_sha256",
        "cache_wrapper_sha256",
        "internal_price_source_hash",
    )
    _require(
        all(
            len(str(output[field])) == 64
            and all(character in "0123456789abcdef" for character in str(output[field]))
            for field in hash_fields
        ),
        "Reviewed no-data unsupported-path entry has an invalid SHA-256 binding.",
    )
    _require(
        output["basis"] == REVIEWED_NO_DATA_UNSUPPORTED_PATH_BASIS
        and output["provider_symbol"] == "BMYRT"
        and output["action_type"] == "delisting"
        and output["event_effective_date"] == output["last_price_date"]
        and output["identity_active_from"] == output["last_price_date"]
        and output["identity_active_to"] > output["last_price_date"]
        and output["candidate_id"]
        == lifecycle_candidate_id(output["security_id"], output["last_price_date"])
        and output["event_id"]
        == canonical_lifecycle_event_id(
            output["security_id"],
            output["action_type"],
            output["event_effective_date"],
        )
        and output["expected_internal_price_rows"] == 1
        and output["valuation_mark_close"] == "2.3"
        and output["cash_available_session"] == "2019-11-22"
        and output["price_history_supported"] is False
        and output["generic_date_tolerance"] is False
        and output["index_scope"] == "non_index_child"
        and bool(output["official_source_url"])
        and bool(output["limitation"]),
        "Reviewed no-data unsupported-path entry changed its narrow scope.",
    )
    return output


def reviewed_no_data_unsupported_paths(
    prices_policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = prices_policy.get("reviewed_no_data_unsupported_paths")
    _require(
        isinstance(raw, list),
        "Policy reviewed_no_data_unsupported_paths must be a list.",
    )
    output: dict[str, dict[str, Any]] = {}
    for value in raw:
        normalized = _canonical_reviewed_no_data_unsupported_path(value)
        target_id = str(normalized["target_id"])
        _require(
            target_id not in output,
            "Duplicate reviewed no-data unsupported-path target: " + target_id,
        )
        output[target_id] = normalized
    return output


def reviewed_no_data_unsupported_path_sha256(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256(_canonical_reviewed_no_data_unsupported_path(value))


def reviewed_no_data_unsupported_path_inventory_sha256(
    prices_policy: Mapping[str, Any],
) -> str:
    registry = reviewed_no_data_unsupported_paths(prices_policy)
    return canonical_json_sha256([registry[key] for key in sorted(registry)])


def permanent_exception_no_data_binding(
    target: Mapping[str, Any],
    terminal_session: str,
    permanent_exception_checks: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Bind no-data only to one already-passed permanent exception candidate."""

    target_id = _text(target.get("target_id")).lower()
    security_id = _text(target.get("security_id"))
    symbol = _text(target.get("symbol") or target.get("provider_symbol")).upper()
    active_to = _date(
        target.get("active_to") or target.get("identity_active_to")
    )
    terminal = _date(terminal_session)
    terminal_event_id = _text(target.get("terminal_event_id"))
    successor_security_id = _text(target.get("successor_security_id"))
    if (
        not target_id
        or target_id
        not in TRUSTED_REVIEWED_PERMANENT_EXCEPTION_NO_DATA_TARGET_IDS
        or not security_id
        or not symbol
        or not terminal
        or terminal_event_id
        or successor_security_id
    ):
        return None
    matches = [
        item
        for item in permanent_exception_checks
        if isinstance(item, Mapping)
        and item.get("status") == "passed"
        and _text(item.get("validation_kind")) == PERMANENT_EXCEPTION_VALIDATION
        and _text(item.get("security_id")) == security_id
        and _text(item.get("symbol")).upper() == symbol
        and _date(item.get("last_price_date")) == terminal
        and _text(item.get("exception_code")) in PERMANENT_EXCEPTION_CODES
        and item.get("identity_date_bound") is True
        and item.get("registry_binding_passed") is True
        and item.get("reviewer_pin_passed") is True
        and item.get("official_original") is True
        and item.get("exact_archive_pair") is True
        and item.get("archive_payload_verified") is True
        and len(_text(item.get("candidate_id"))) == 64
        and len(_text(item.get("evidence_sha256"))) == 64
    ]
    _require(
        len(matches) <= 1,
        "Permanent no-data price exception has ambiguous candidate bindings: "
        + target_id,
    )
    if not matches:
        return None
    match = matches[0]
    # A stored terminal boundary may include non-trading legal days, but it may
    # not precede the reviewed last-price candidate.
    if active_to and active_to < terminal:
        return None
    return {
        "code": PERMANENT_EXCEPTION_NO_DATA_CODE,
        "validation_basis": REVIEWED_PERMANENT_EXCEPTION_NO_DATA_BASIS,
        "candidate_id": _text(match.get("candidate_id")),
        "evidence_id": _text(match.get("evidence_id")),
        "exception_code": _text(match.get("exception_code")),
        "exception_reason": _text(match.get("exception_reason")),
        "official_evidence_sha256": _text(match.get("evidence_sha256")).lower(),
        "official_source_url": _text(match.get("source_url")),
        "last_price_date": terminal,
        "identity_date_match": True,
        "identity_date_basis": "permanent_exception_candidate_last_price",
        "price_history_supported": False,
        "generic_date_tolerance": False,
    }


def unsupported_path_no_data_binding(
    target: Mapping[str, Any],
    terminal_session: str,
    event: Mapping[str, Any],
    internal_prices: pd.DataFrame,
    prices_policy: Mapping[str, Any],
    *,
    source_sha256: str,
    cache_wrapper_sha256: str,
) -> dict[str, Any] | None:
    """Bind the sole reviewed official valuation mark to exact no-data bytes."""

    target_id = _text(target.get("target_id")).lower()
    spec = reviewed_no_data_unsupported_paths(prices_policy).get(target_id)
    if spec is None:
        return None
    projection = {
        "security_id": _text(target.get("security_id")),
        "provider_symbol": _text(
            target.get("provider_symbol") or target.get("symbol")
        ).upper(),
        "identity_active_from": _date(
            target.get("active_from") or target.get("identity_active_from")
        ),
        "identity_active_to": _date(
            target.get("active_to") or target.get("identity_active_to")
        ),
        "last_price_date": _date(terminal_session),
        "event_id": _text(event.get("event_id")).lower(),
        "action_type": _text(event.get("action_type")).lower(),
        "event_effective_date": _date(event.get("effective_date")),
        "candidate_id": _text(event.get("candidate_id")).lower(),
        "official_evidence_sha256": _text(event.get("evidence_sha256")).lower(),
        "source_sha256": _text(source_sha256).lower(),
        "cache_wrapper_sha256": _text(cache_wrapper_sha256).lower(),
    }
    expected = {
        key: spec[key]
        for key in (
            "security_id",
            "provider_symbol",
            "identity_active_from",
            "identity_active_to",
            "last_price_date",
            "event_id",
            "action_type",
            "event_effective_date",
            "candidate_id",
            "official_evidence_sha256",
            "source_sha256",
            "cache_wrapper_sha256",
        )
    }
    if projection != expected or event.get("status") != "passed":
        return None
    rows = internal_prices.copy()
    sessions = pd.to_datetime(rows.get("session"), errors="coerce")
    rows = rows.loc[sessions.dt.date.astype(str).eq(spec["last_price_date"])]
    if (
        len(internal_prices) != spec["expected_internal_price_rows"]
        or len(rows) != 1
        or _text(rows.iloc[0].get("source_hash")).lower()
        != spec["internal_price_source_hash"]
        or any(
            _exact_number_text(rows.iloc[0].get(field), field)
            != spec["valuation_mark_close"]
            for field in ("open", "high", "low", "close")
        )
        or _exact_number_text(rows.iloc[0].get("volume"), "volume") != "0"
        or _text(rows.iloc[0].get("currency")).upper() != "USD"
    ):
        return None
    return {
        "code": UNSUPPORTED_PATH_NO_DATA_CODE,
        "validation_basis": REVIEWED_NO_DATA_UNSUPPORTED_PATH_BASIS,
        "reviewed_spec_sha256": reviewed_no_data_unsupported_path_sha256(spec),
        "reviewed_registry_sha256": (
            TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_SHA256
        ),
        "candidate_id": spec["candidate_id"],
        "official_event_id": spec["event_id"],
        "official_action_type": spec["action_type"],
        "official_evidence_sha256": spec["official_evidence_sha256"],
        "official_source_url": spec["official_source_url"],
        "last_price_date": spec["last_price_date"],
        "identity_date_match": True,
        "identity_date_basis": "reviewed_official_exit_mark_not_price_path",
        "price_history_supported": False,
        "generic_date_tolerance": False,
        "expected_internal_price_rows": spec["expected_internal_price_rows"],
        "valuation_mark_close": spec["valuation_mark_close"],
        "cash_available_session": spec["cash_available_session"],
        "index_scope": spec["index_scope"],
        "limitation": spec["limitation"],
    }


def _canonical_reviewed_no_data_successor_chain(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one finite, report-independent no-data successor review."""

    _require(
        isinstance(value, Mapping),
        "Reviewed no-data successor-chain entry must be an object.",
    )
    raw_nodes = value.get("nodes")
    final = value.get("final")
    _require(
        isinstance(raw_nodes, list) and 2 <= len(raw_nodes) <= 4,
        "Reviewed no-data successor chain must contain two to four nodes.",
    )
    _require(
        isinstance(final, Mapping),
        "Reviewed no-data successor chain lacks a final price target.",
    )

    exact_request_fields = {
        "source_url",
        "request_period1",
        "request_period2",
        "http_status",
        "no_data_evidence_kind",
    }
    nodes: list[dict[str, Any]] = []
    for raw in raw_nodes:
        _require(
            isinstance(raw, Mapping),
            "Reviewed no-data successor-chain node must be an object.",
        )
        present_request_fields = exact_request_fields & set(raw)
        _require(
            not present_request_fields
            or present_request_fields == exact_request_fields,
            "Reviewed no-data successor-chain request binding must be all-or-none.",
        )
        node: dict[str, Any] = {
            "target_id": _text(raw.get("target_id")).lower(),
            "security_id": _text(raw.get("security_id")),
            "provider_symbol": _text(raw.get("provider_symbol")).upper(),
            "event_id": _text(raw.get("event_id")).lower(),
            "official_evidence_sha256": _text(
                raw.get("official_evidence_sha256")
            ).lower(),
            "successor_security_id": _text(raw.get("successor_security_id")),
            "successor_symbol": _text(raw.get("successor_symbol")).upper(),
            "source_sha256": _text(raw.get("source_sha256")).lower(),
            "cache_wrapper_sha256": _text(
                raw.get("cache_wrapper_sha256")
            ).lower(),
        }
        if present_request_fields:
            node.update(
                {
                    "source_url": _text(raw.get("source_url")),
                    "request_period1": _integer(
                        raw.get("request_period1"), "request_period1"
                    ),
                    "request_period2": _integer(
                        raw.get("request_period2"), "request_period2"
                    ),
                    "http_status": _integer(raw.get("http_status"), "http_status"),
                    "no_data_evidence_kind": _text(
                        raw.get("no_data_evidence_kind")
                    ),
                }
            )
            request = _bounded_yahoo_source_request(node["source_url"])
            _require(
                request
                == (
                    node["provider_symbol"],
                    node["request_period1"],
                    node["request_period2"],
                )
                and node["http_status"] in {200, 400, 404, 410}
                and node["no_data_evidence_kind"]
                in {
                    "chart_not_found",
                    "http_200_empty_equity_chart",
                    "http_200_empty_retired_yhd_placeholder",
                    "http_400_bounded_history_not_found",
                },
                "Reviewed no-data successor-chain request binding is invalid.",
            )
        nodes.append(node)

    output = {
        "root_target_id": _text(value.get("root_target_id")).lower(),
        "basis": _text(value.get("basis")),
        "raw_price_only": value.get("raw_price_only"),
        "predecessor_backcast": value.get("predecessor_backcast"),
        "nodes": nodes,
        "final": {
            "target_id": _text(final.get("target_id")).lower(),
            "security_id": _text(final.get("security_id")),
            "provider_symbol": _text(final.get("provider_symbol")).upper(),
            "status": _text(final.get("status")).lower(),
            "source_sha256": _text(final.get("source_sha256")).lower(),
            "cache_wrapper_sha256": _text(
                final.get("cache_wrapper_sha256")
            ).lower(),
            "reviewed_price_evidence_sha256": _text(
                final.get("reviewed_price_evidence_sha256")
            ).lower(),
        },
    }

    hashes = [
        node[field]
        for node in nodes
        for field in (
            "target_id",
            "event_id",
            "official_evidence_sha256",
            "source_sha256",
            "cache_wrapper_sha256",
        )
    ] + [
        output["final"]["target_id"],
        output["final"]["source_sha256"],
        output["final"]["cache_wrapper_sha256"],
    ]
    reviewed_hash = output["final"]["reviewed_price_evidence_sha256"]
    if reviewed_hash:
        hashes.append(reviewed_hash)
    _require(
        all(
            len(value_hash) == 64
            and all(character in "0123456789abcdef" for character in value_hash)
            for value_hash in hashes
        ),
        "Reviewed no-data successor chain contains an invalid SHA-256 binding.",
    )
    _require(
        output["basis"] == REVIEWED_NO_DATA_SUCCESSOR_CHAIN_BASIS
        and output["raw_price_only"] is True
        and output["predecessor_backcast"] is False
        and output["final"]["status"] == "passed",
        "Reviewed no-data successor chain changed its narrow validation scope.",
    )
    _require(
        output["root_target_id"] == nodes[0]["target_id"],
        "Reviewed no-data successor-chain root does not match its first node.",
    )
    target_ids = [node["target_id"] for node in nodes] + [
        output["final"]["target_id"]
    ]
    _require(
        len(target_ids) == len(set(target_ids)),
        "Reviewed no-data successor chain is cyclic or repeats a target.",
    )
    _require(
        len({node["event_id"] for node in nodes}) == len(nodes),
        "Reviewed no-data successor chain repeats an official event.",
    )
    for index, node in enumerate(nodes):
        next_target = (
            nodes[index + 1] if index + 1 < len(nodes) else output["final"]
        )
        _require(
            bool(node["security_id"])
            and bool(node["provider_symbol"])
            and bool(node["successor_security_id"])
            and bool(node["successor_symbol"])
            and node["successor_security_id"] == next_target["security_id"]
            and node["successor_symbol"] == next_target["provider_symbol"],
            "Reviewed no-data successor chain has a broken identity link.",
        )
    return output


def reviewed_no_data_successor_chains(
    prices_policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = prices_policy.get("reviewed_no_data_successor_chains")
    _require(
        isinstance(raw, list),
        "Policy reviewed_no_data_successor_chains must be a list.",
    )
    output: dict[str, dict[str, Any]] = {}
    for value in raw:
        normalized = _canonical_reviewed_no_data_successor_chain(value)
        root_target_id = normalized["root_target_id"]
        _require(
            root_target_id not in output,
            "Duplicate reviewed no-data successor-chain root: " + root_target_id,
        )
        output[root_target_id] = normalized
    return output


def reviewed_no_data_successor_chain_sha256(
    value: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(_canonical_reviewed_no_data_successor_chain(value))


def reviewed_no_data_successor_chain_inventory_sha256(
    prices_policy: Mapping[str, Any],
) -> str:
    registry = reviewed_no_data_successor_chains(prices_policy)
    return canonical_json_sha256([registry[key] for key in sorted(registry)])


def _reviewed_no_data_successor_chain_binding(
    price_checks: list[Mapping[str, Any]],
    event_checks: list[Mapping[str, Any]],
    *,
    source_target_id: str,
    expected_successor_security_id: str,
    chain_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Recompute one exact finite chain without trusting exception statuses."""

    spec_value = chain_specs.get(_text(source_target_id).lower())
    if spec_value is None:
        return None
    try:
        spec = _canonical_reviewed_no_data_successor_chain(spec_value)
    except RuntimeError:
        return None

    price_by_target: dict[str, Mapping[str, Any]] = {}
    for item in price_checks:
        target_id = _text(item.get("target_id")).lower()
        if not target_id or target_id in price_by_target:
            return None
        price_by_target[target_id] = item
    event_by_id: dict[str, Mapping[str, Any]] = {}
    for item in event_checks:
        event_id = _text(item.get("event_id")).lower()
        if not event_id or event_id in event_by_id:
            return None
        event_by_id[event_id] = item

    nodes = spec["nodes"]
    final_spec = spec["final"]
    if nodes[0]["successor_security_id"] != _text(
        expected_successor_security_id
    ):
        return None

    for index, node in enumerate(nodes):
        item = price_by_target.get(node["target_id"])
        event = event_by_id.get(node["event_id"])
        if item is None or event is None:
            return None
        exception = item.get("exception")
        if not isinstance(exception, Mapping):
            return None
        local_requirements = (
            "official_event_verified",
            "identity_event_match",
            "identity_date_match",
            "terminal_calendar_complete",
            "response_identity_match",
            "no_data_evidence_validated",
        )
        if not all(exception.get(key) is True for key in local_requirements):
            return None
        try:
            item_symbol = normalize_yahoo_symbol(
                _text(item.get("provider_symbol") or item.get("symbol"))
            )
            event_successor_symbol = normalize_yahoo_symbol(
                _text(event.get("new_symbol"))
            )
            item_period1 = (
                _integer(item.get("request_period1"), "request_period1")
                if "source_url" in node
                else 0
            )
            item_period2 = (
                _integer(item.get("request_period2"), "request_period2")
                if "source_url" in node
                else 0
            )
            item_http_status = (
                _integer(item.get("http_status"), "http_status")
                if "source_url" in node
                else 0
            )
        except (RuntimeError, ValueError):
            return None
        exact_request = (
            "source_url" not in node
            or (
                _text(item.get("source_url")) == node["source_url"]
                and _text(item.get("expected_source_url")) == node["source_url"]
                and item_period1 == node["request_period1"]
                and item_period2 == node["request_period2"]
                and item_http_status == node["http_status"]
                and _text(item.get("no_data_evidence_kind"))
                == node["no_data_evidence_kind"]
            )
        )
        exact_item = (
            _text(item.get("target_id")).lower() == node["target_id"]
            and _text(item.get("security_id")) == node["security_id"]
            and item_symbol == node["provider_symbol"]
            and item.get("provider_support") == "no_data"
            and _text(item.get("source_sha256")).lower() == node["source_sha256"]
            and _text(item.get("cache_wrapper_sha256")).lower()
            == node["cache_wrapper_sha256"]
            and _text(item.get("terminal_event_id")).lower() == node["event_id"]
            and _text(exception.get("official_event_id")).lower()
            == node["event_id"]
            and _text(exception.get("official_evidence_sha256")).lower()
            == node["official_evidence_sha256"]
            and _text(exception.get("successor_security_id"))
            == node["successor_security_id"]
            and exact_request
        )
        exact_event = (
            event.get("status") == "passed"
            and _text(event.get("event_id")).lower() == node["event_id"]
            and _text(event.get("security_id")) == node["security_id"]
            and _text(event.get("action_type")).lower()
            in YAHOO_NO_DATA_TERMINAL_ACTION_TYPES
            and _text(event.get("evidence_sha256")).lower()
            == node["official_evidence_sha256"]
            and _text(event.get("new_security_id"))
            == node["successor_security_id"]
            and event_successor_symbol == node["successor_symbol"]
        )
        if not exact_item or not exact_event:
            return None

        next_spec = nodes[index + 1] if index + 1 < len(nodes) else final_spec
        effective = _date(event.get("effective_date"))
        candidates: list[Mapping[str, Any]] = []
        for candidate in price_checks:
            if _text(candidate.get("target_id")).lower() == node["target_id"]:
                continue
            if _text(candidate.get("security_id")) != node["successor_security_id"]:
                continue
            try:
                candidate_symbol = normalize_yahoo_symbol(
                    _text(candidate.get("provider_symbol") or candidate.get("symbol"))
                )
            except ValueError:
                continue
            active_from = _date(candidate.get("identity_active_from"))
            active_to = _date(candidate.get("identity_active_to"))
            if (
                candidate_symbol == node["successor_symbol"]
                and effective
                and (not active_from or active_from <= effective)
                and (not active_to or effective <= active_to)
            ):
                candidates.append(candidate)
        if (
            len(candidates) != 1
            or _text(candidates[0].get("target_id")).lower()
            != next_spec["target_id"]
        ):
            return None

    final_item = price_by_target.get(final_spec["target_id"])
    if final_item is None:
        return None
    final_reviewed_hash = _text(
        final_item.get("reviewed_price_evidence_sha256")
    ).lower()
    expected_reviewed_hash = final_spec["reviewed_price_evidence_sha256"]
    if (
        final_item.get("status") != "passed"
        or final_item.get("provider_support") == "no_data"
        or _text(final_item.get("security_id")) != final_spec["security_id"]
        or _text(final_item.get("provider_symbol") or final_item.get("symbol")).upper()
        != final_spec["provider_symbol"]
        or _text(final_item.get("source_sha256")).lower()
        != final_spec["source_sha256"]
        or _text(final_item.get("cache_wrapper_sha256")).lower()
        != final_spec["cache_wrapper_sha256"]
        or final_reviewed_hash != expected_reviewed_hash
        or bool(final_item.get("reviewed_price_evidence_applied"))
        != bool(expected_reviewed_hash)
    ):
        return None

    return {
        "required": True,
        "passed": True,
        "target_id": nodes[1]["target_id"],
        "provider_symbol": nodes[0]["successor_symbol"],
        "status": "reviewed_finite_successor_chain",
        "reason": "",
        "candidate_count": 1,
        "validation_basis": REVIEWED_NO_DATA_SUCCESSOR_CHAIN_BASIS,
        "chain_sha256": reviewed_no_data_successor_chain_sha256(spec),
        "chain_root_target_id": spec["root_target_id"],
        "chain_target_ids": [node["target_id"] for node in nodes]
        + [final_spec["target_id"]],
        "chain_event_ids": [node["event_id"] for node in nodes],
        "final_target_id": final_spec["target_id"],
        "final_provider_symbol": final_spec["provider_symbol"],
        "final_status": "passed",
        "raw_price_only": True,
        "predecessor_backcast": False,
    }


def successor_price_check_binding(
    price_checks: Iterable[Mapping[str, Any]],
    event: Mapping[str, Any],
    *,
    source_target_id: str,
    expected_successor_security_id: str,
    reviewed_successor_chains: Mapping[str, Mapping[str, Any]] | None = None,
    event_checks: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Select the one independently priced successor identity for an event.

    Requiring every historical symbol segment of the successor to pass creates
    false cascades (and makes same-security ticker changes impossible).  This
    instead binds the event's exact successor symbol/security/date to one
    identity interval.  The chosen interval must itself be a normal ``passed``
    Yahoo/EODHD comparison.  A no-data interval can satisfy the requirement
    only through the separate code-pinned finite-chain registry, whose last
    target must be a normal or reviewed-exact ``passed`` price comparison.
    """

    successor_id = _text(expected_successor_security_id)
    if not successor_id:
        return {
            "required": False,
            "passed": True,
            "target_id": "",
            "provider_symbol": "",
            "status": "not_required",
            "reason": "",
            "candidate_count": 0,
        }
    event_successor_id = _text(event.get("new_security_id"))
    event_symbol = _text(event.get("new_symbol")).upper()
    effective = _date(event.get("effective_date"))
    try:
        provider_symbol = normalize_yahoo_symbol(event_symbol)
    except ValueError:
        provider_symbol = ""
    if (
        event_successor_id != successor_id
        or not provider_symbol
        or not effective
    ):
        return {
            "required": True,
            "passed": False,
            "target_id": "",
            "provider_symbol": provider_symbol,
            "status": "event_successor_mismatch",
            "reason": "official event successor identity is incomplete or inconsistent",
            "candidate_count": 0,
        }

    price_check_list = list(price_checks)
    candidates: list[Mapping[str, Any]] = []
    for item in price_check_list:
        if (
            _text(item.get("target_id")) == _text(source_target_id)
            or _text(item.get("security_id")) != successor_id
        ):
            continue
        try:
            item_symbol = normalize_yahoo_symbol(
                _text(item.get("provider_symbol") or item.get("symbol"))
            )
        except ValueError:
            continue
        active_from = _date(item.get("identity_active_from"))
        active_to = _date(item.get("identity_active_to"))
        if (
            item_symbol == provider_symbol
            and (not active_from or active_from <= effective)
            and (not active_to or effective <= active_to)
        ):
            candidates.append(item)
    if len(candidates) != 1:
        return {
            "required": True,
            "passed": False,
            "target_id": "",
            "provider_symbol": provider_symbol,
            "status": "successor_target_not_unique",
            "reason": (
                "no exact successor identity interval"
                if not candidates
                else "multiple successor identity intervals"
            ),
            "candidate_count": len(candidates),
        }
    candidate = candidates[0]
    status = _text(candidate.get("status"))
    reason = _text(candidate.get("reason"))
    if status != "passed" and reviewed_successor_chains:
        reviewed_binding = _reviewed_no_data_successor_chain_binding(
            price_check_list,
            list(event_checks),
            source_target_id=source_target_id,
            expected_successor_security_id=expected_successor_security_id,
            chain_specs=reviewed_successor_chains,
        )
        if reviewed_binding is not None:
            return reviewed_binding
    return {
        "required": True,
        "passed": status == "passed",
        "target_id": _text(candidate.get("target_id")),
        "provider_symbol": provider_symbol,
        "status": status,
        "reason": reason,
        "candidate_count": 1,
    }


def _bounded_yahoo_source_request(value: Any) -> tuple[str, int, int] | None:
    parsed = urlparse(str(value).strip())
    query = parse_qs(parsed.query, keep_blank_values=True)
    safe = (
        parsed.scheme == "https"
        and (parsed.hostname or "").lower() == INDEPENDENT_PRICE_HOST
        and parsed.username is None
        and parsed.password is None
        and parsed.fragment == ""
        and parsed.path.startswith("/v8/finance/chart/")
        and len(parsed.path.removeprefix("/v8/finance/chart/")) > 0
        and set(query)
        == {"period1", "period2", "events", "includeAdjustedClose", "interval"}
        and query.get("events") == ["history"]
        and query.get("includeAdjustedClose") == ["true"]
        and query.get("interval") == ["1d"]
        and not ({"crumb", "token", "apikey", "api_key"} & set(query))
    )
    if not safe:
        return None
    try:
        period1 = int(query["period1"][0])
        period2 = int(query["period2"][0])
        symbol = normalize_yahoo_symbol(
            unquote(parsed.path.removeprefix("/v8/finance/chart/"))
        )
    except (TypeError, ValueError):
        return None
    if (
        str(period1) != query["period1"][0]
        or str(period2) != query["period2"][0]
        or period1 <= 0
        or period2 <= period1
        or period2 >= 4_102_444_800
    ):
        return None
    return symbol, period1, period2


def _safe_yahoo_source_url(value: Any) -> bool:
    return _bounded_yahoo_source_request(value) is not None


def _canonical_bounded_yahoo_url(symbol: str, period1: int, period2: int) -> str:
    normalized = normalize_yahoo_symbol(symbol)
    return (
        f"{INDEPENDENT_PRICE_ENDPOINT.format(symbol=normalized)}"
        f"?period1={period1}&period2={period2}"
        "&events=history&includeAdjustedClose=true&interval=1d"
    )


def _expected_bounded_yahoo_request(
    target: Mapping[str, Any],
    prices: pd.DataFrame,
) -> tuple[str, str, int, int]:
    start = _date(target.get("active_from"))
    end = _date(target.get("active_to"))
    _require(bool(start), "Yahoo target active_from is required for bounded requests.")
    if not end:
        sessions = pd.to_datetime(prices["session"], errors="coerce")
        _require(
            not bool(sessions.isna().any()) and not sessions.empty,
            "daily_price_raw cannot determine the bounded Yahoo as-of date.",
        )
        end = sessions.max().date().isoformat()
    start_day = pd.Timestamp(start, tz="UTC")
    end_day = pd.Timestamp(end, tz="UTC")
    _require(end_day >= start_day, "Yahoo bounded request dates are reversed.")
    return (
        start,
        end,
        int(start_day.timestamp()),
        int((end_day + pd.Timedelta(days=1)).timestamp()),
    )


def _source_url_symbol(value: Any) -> str:
    parsed = urlparse(str(value).strip())
    try:
        return normalize_yahoo_symbol(unquote(parsed.path.rsplit("/", 1)[-1]))
    except ValueError as exc:
        raise RuntimeError("Yahoo source URL symbol is invalid.") from exc


def _normalized_report_symbol(value: Any) -> str:
    try:
        return normalize_yahoo_symbol(str(value))
    except ValueError as exc:
        raise RuntimeError("Yahoo report symbol is invalid.") from exc


def _validate_policy_contract(policy: Mapping[str, Any]) -> None:
    _require(
        policy.get("schema_version") == 6,
        "Cross-validation policy schema is not terminal-date-correction v6.",
    )
    provider = policy.get("provider")
    events = policy.get("events")
    prices = policy.get("prices")
    _require(isinstance(provider, dict), "Cross-validation provider policy is missing.")
    _require(isinstance(events, dict), "Cross-validation event policy is missing.")
    _require(isinstance(prices, dict), "Cross-validation price policy is missing.")
    _require(
        provider.get("name") == INDEPENDENT_PRICE_PROVIDER
        and provider.get("endpoint_template") == INDEPENDENT_PRICE_ENDPOINT,
        "Cross-validation policy does not pin the Yahoo chart endpoint.",
    )
    _require(
        _integer(provider.get("max_http_attempts"), "max_http_attempts") == 400
        and _integer(
            provider.get("max_attempts_per_target"), "max_attempts_per_target"
        )
        == 1
        and _integer(provider.get("retry_count"), "retry_count") == 0,
        "Cross-validation policy must enforce 400 total attempts, one per target, and no retries.",
    )
    _require(
        provider.get("repository_visibility") == "private"
        and provider.get("r2_visibility") == "private"
        and provider.get("redistribution_allowed") is False
        and isinstance(provider.get("use_restriction"), str)
        and "personal" in provider["use_restriction"].lower()
        and "private" in provider["use_restriction"].lower(),
        "Yahoo chart policy must remain personal-use only in private storage.",
    )
    _require(
        set(events.get("action_types") or ())
        == {"cash_merger", "stock_merger", "spinoff", "ticker_change", "delisting"},
        "Cross-validation policy must cover all five lifecycle action types.",
    )
    terminal_hosts = {
        _text(item).lower() for item in events.get("official_hosts") or ()
    }
    provenance_hosts = {
        _text(item).lower()
        for item in events.get("official_provenance_hosts") or ()
    }
    _require(
        terminal_hosts == {"sec.gov"}
        and provenance_hosts
        == {
            "sec.gov",
            "spglobal.com",
            "investor.ovintiv.com",
            "investors.qvcgrp.com",
        }
        and {
            _text(item)
            for item in events.get("official_provenance_source_kinds") or ()
        }
        == {
            "official_crosscheck",
            "official_filing",
            "official_filing_exit_mark",
            "official_issuer_plus_sec_crosscheck",
            "official_issuer_market_transition",
        },
        "Cross-validation policy does not pin terminal/nonterminal official provenance.",
    )
    _require(
        {
            _text(item)
            for item in events.get("terminal_official_source_kinds") or ()
        }
        == {"official_crosscheck", "official_filing"},
        "Cross-validation policy does not pin terminal official source kinds.",
    )
    reviewed = reviewed_nonterminal_extractions(events)
    _require(
        bool(reviewed),
        "Cross-validation policy has no reviewed nonterminal extraction inventory.",
    )
    _require(
        reviewed_nonterminal_inventory_sha256(events)
        == TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
        "Reviewed nonterminal extraction inventory is not the code-pinned manifest.",
    )
    allowed_kinds = {
        _text(item)
        for item in events.get("official_provenance_source_kinds") or ()
    }
    for event_id, extraction in reviewed.items():
        _require(
            _official_url(extraction["source_url"], provenance_hosts)
            and extraction["source_kind"] in allowed_kinds,
            "Reviewed nonterminal extraction has unapproved official provenance: "
            + event_id,
        )
    event_gates = reviewed_terminal_event_gates(events)
    _require(
        set(event_gates) == set(TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_EVENT_IDS)
        and reviewed_terminal_event_gate_inventory_sha256(events)
        == TRUSTED_REVIEWED_TERMINAL_EVENT_GATES_SHA256,
        "Reviewed terminal event-gate inventory is not the code-pinned manifest.",
    )
    terminal_overrides = reviewed_terminal_overrides(events)
    _require(
        set(terminal_overrides)
        == set(TRUSTED_REVIEWED_TERMINAL_OVERRIDE_EVENT_IDS)
        and reviewed_terminal_override_inventory_sha256(events)
        == TRUSTED_REVIEWED_TERMINAL_OVERRIDES_SHA256,
        "Reviewed terminal override inventory is not the code-pinned manifest.",
    )
    for event_id, override in terminal_overrides.items():
        _require(
            _official_url(override["source_url"], terminal_hosts)
            and override["source_kind"] == "official_crosscheck",
            "Reviewed terminal override has unapproved official provenance: "
            + event_id,
        )
    market_date_corrections = reviewed_terminal_market_date_corrections(events)
    _require(
        set(market_date_corrections)
        == set(TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTION_EVENT_IDS)
        and reviewed_terminal_market_date_correction_inventory_sha256(events)
        == TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTIONS_SHA256,
        "Reviewed terminal market-date correction inventory is not the "
        "code-pinned manifest.",
    )
    _require(
        not (set(market_date_corrections) & set(terminal_overrides))
        and not (set(market_date_corrections) & set(reviewed)),
        "Terminal market-date corrections must remain separate from other "
        "reviewed exception inventories.",
    )
    for event_id, correction in market_date_corrections.items():
        _require(
            _official_url(correction["source_url"], terminal_hosts)
            and _official_url(correction["report_source_url"], terminal_hosts)
            and correction["source_kind"] == "official_crosscheck",
            "Reviewed terminal market-date correction has unapproved official "
            "provenance: "
            + event_id,
        )
    policy_exceptions = reviewed_terminal_policy_exceptions(events)
    _require(
        set(policy_exceptions)
        == set(TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTION_EVENT_IDS)
        and reviewed_terminal_policy_exception_inventory_sha256(events)
        == TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTIONS_SHA256,
        "Reviewed terminal policy exception inventory is not the code-pinned "
        "manifest.",
    )
    _require(
        not (set(policy_exceptions) & set(market_date_corrections))
        and not (set(policy_exceptions) & set(terminal_overrides))
        and not (set(policy_exceptions) & set(reviewed)),
        "Terminal policy exceptions must remain separate from every other "
        "reviewed exception inventory.",
    )
    for event_id, exception in policy_exceptions.items():
        _require(
            _official_url(exception["source_url"], terminal_hosts)
            and _official_url(exception["report_source_url"], terminal_hosts),
            "Reviewed terminal policy exception has unapproved SEC provenance: "
            + event_id,
        )
    tail_corrections = reviewed_terminal_price_tail_corrections(events)
    _require(
        set(tail_corrections)
        == set(TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS)
        and reviewed_terminal_price_tail_correction_inventory_sha256(events)
        == TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256,
        "Reviewed terminal price-tail correction inventory is not the "
        "code-pinned repair registry.",
    )
    _require(
        not (set(tail_corrections) & set(policy_exceptions))
        and not (set(tail_corrections) & set(market_date_corrections))
        and not (set(tail_corrections) & set(terminal_overrides))
        and not (set(tail_corrections) & set(reviewed)),
        "Terminal price-tail corrections must remain separate from every other "
        "reviewed exception inventory.",
    )
    for event_id, correction in tail_corrections.items():
        _require(
            _official_url(correction["official_source_url"], terminal_hosts),
            "Reviewed terminal price-tail correction has unapproved SEC "
            "provenance: " + event_id,
        )
    sivb_bindings = trusted_sivb_evidence_bindings()
    _require(
        set(sivb_bindings) == set(TRUSTED_SIVB_EVIDENCE_BINDING_EVENT_IDS)
        and trusted_sivb_evidence_binding_inventory_sha256()
        == TRUSTED_SIVB_EVIDENCE_BINDINGS_SHA256
        and not (set(sivb_bindings) & set(reviewed))
        and not (set(sivb_bindings) & set(terminal_overrides))
        and not (set(sivb_bindings) & set(policy_exceptions))
        and not (set(sivb_bindings) & set(tail_corrections))
        and not (set(sivb_bindings) & set(market_date_corrections)),
        "Trusted SIVB evidence bindings overlap an unreviewed policy path.",
    )
    frc_bindings = trusted_frc_evidence_bindings()
    _require(
        set(frc_bindings) == set(TRUSTED_FRC_EVIDENCE_BINDING_EVENT_IDS)
        and trusted_frc_evidence_binding_inventory_sha256()
        == TRUSTED_FRC_EVIDENCE_BINDINGS_SHA256
        and not (set(frc_bindings) & set(reviewed))
        and not (set(frc_bindings) & set(terminal_overrides))
        and not (set(frc_bindings) & set(policy_exceptions))
        and not (set(frc_bindings) & set(tail_corrections))
        and not (set(frc_bindings) & set(market_date_corrections)),
        "Trusted FRC evidence bindings overlap another reviewed policy path.",
    )
    ntco_bindings = trusted_ntco_evidence_bindings()
    configured_ntco_ids = {
        _text(value)
        for value in events.get("reviewed_ntco_transition_event_ids") or ()
    }
    _require(
        configured_ntco_ids == set(TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS)
        and set(ntco_bindings) == set(TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS)
        and trusted_ntco_evidence_binding_inventory_sha256()
        == TRUSTED_NTCO_EVIDENCE_BINDINGS_SHA256
        and not (set(ntco_bindings) & set(reviewed))
        and not (set(ntco_bindings) & set(terminal_overrides))
        and not (set(ntco_bindings) & set(policy_exceptions))
        and not (set(ntco_bindings) & set(tail_corrections))
        and not (set(ntco_bindings) & set(market_date_corrections))
        and not (set(ntco_bindings) & set(sivb_bindings))
        and not (set(ntco_bindings) & set(frc_bindings)),
        "Trusted NTCO evidence bindings are not the exact isolated policy set.",
    )
    reviewed_prices = reviewed_price_evidence_registry(prices)
    _require(
        set(reviewed_prices) == set(TRUSTED_REVIEWED_PRICE_EVIDENCE_TARGET_IDS)
        and reviewed_price_evidence_inventory_sha256(prices)
        == TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256,
        "Reviewed price-evidence inventory is not the code-pinned manifest.",
    )
    source_archive_price_only = source_archive_price_only_registry(prices)
    _require(
        set(source_archive_price_only)
        == set(TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS)
        and source_archive_price_only_inventory_sha256(prices)
        == TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256
        and not (set(source_archive_price_only) & set(reviewed_prices)),
        "Frozen WIKI price-only evidence inventory is not the exact isolated "
        "code-pinned pair.",
    )
    wiki14_price_only = wiki14_price_only_registry(prices)
    _require(
        set(wiki14_price_only)
        == set(TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS)
        and wiki14_price_only_inventory_sha256(prices)
        == TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256
        and not (set(wiki14_price_only) & set(reviewed_prices))
        and not (set(wiki14_price_only) & set(source_archive_price_only)),
        "Frozen WIKI14 price-only evidence inventory is not the exact "
        "isolated code-pinned set.",
    )
    unsupported_paths = reviewed_no_data_unsupported_paths(prices)
    _require(
        set(unsupported_paths)
        == set(TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_TARGET_IDS)
        and reviewed_no_data_unsupported_path_inventory_sha256(prices)
        == TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_SHA256
        and not (set(unsupported_paths) & set(reviewed_prices))
        and not (set(unsupported_paths) & set(source_archive_price_only))
        and not (set(unsupported_paths) & set(wiki14_price_only)),
        "Reviewed no-data unsupported-path inventory is not the exact isolated "
        "code-pinned set.",
    )
    for target_id, spec in unsupported_paths.items():
        _require(
            _official_url(spec["official_source_url"], terminal_hosts),
            "Reviewed no-data unsupported-path evidence has unapproved "
            "provenance: "
            + target_id,
        )
    reviewed_successor_chains = reviewed_no_data_successor_chains(prices)
    _require(
        set(reviewed_successor_chains)
        == set(TRUSTED_REVIEWED_NO_DATA_SUCCESSOR_CHAIN_ROOT_TARGET_IDS)
        and reviewed_no_data_successor_chain_inventory_sha256(prices)
        == TRUSTED_REVIEWED_NO_DATA_SUCCESSOR_CHAINS_SHA256,
        "Reviewed no-data successor-chain inventory is not code-pinned.",
    )
    _require(
        not (set(reviewed_successor_chains) & set(unsupported_paths)),
        "No-data finite chains and unsupported-path exceptions must be disjoint.",
    )
    reviewed_chain_target_ids = {
        target_id
        for chain in reviewed_successor_chains.values()
        for target_id in (
            [node["target_id"] for node in chain["nodes"]]
            + [chain["final"]["target_id"]]
        )
    }
    _require(
        not (
            reviewed_chain_target_ids
            & set(BLOCKED_NO_DATA_SUCCESSOR_CHAIN_TARGET_IDS)
        ),
        "Reviewed no-data successor chains include a blocked cycle or ticker reuse.",
    )
    _require(
        prices.get("currency") == "USD"
        and prices.get("instrument_type") == "EQUITY"
        and set(prices.get("allowed_exchange_names") or ())
        == set(ALLOWED_US_EXCHANGE_NAMES)
        and prices.get("exchange_timezone") == US_EXCHANGE_TIMEZONE
        and "indicators.quote" in str(prices.get("adjustment_basis", ""))
        and "adjclose is never" in str(prices.get("adjustment_basis", "")),
        "Cross-validation price policy must pin USD US-equity raw Yahoo quote OHLCV.",
    )
    no_data_action_types = prices.get("no_data_terminal_action_types")
    _require(
        isinstance(no_data_action_types, list)
        and len(no_data_action_types) == len(YAHOO_NO_DATA_TERMINAL_ACTION_TYPES)
        and {_text(value).lower() for value in no_data_action_types}
        == set(YAHOO_NO_DATA_TERMINAL_ACTION_TYPES),
        "Yahoo no-data terminal action allowlist must be exact.",
    )
    no_data_date_relations = prices.get("no_data_terminal_date_relations")
    _require(
        isinstance(no_data_date_relations, list)
        and len(no_data_date_relations)
        == len(YAHOO_NO_DATA_TERMINAL_DATE_RELATIONS)
        and {_text(value) for value in no_data_date_relations}
        == set(YAHOO_NO_DATA_TERMINAL_DATE_RELATIONS)
        and prices.get("no_data_successor_validation_basis")
        == YAHOO_NO_DATA_SUCCESSOR_VALIDATION_BASIS,
        "Yahoo no-data date/successor policy must be exact.",
    )
    overlap_specs = prices.get("pinned_external_overlaps")
    _require(
        isinstance(overlap_specs, list)
        and {
            _text(item.get("symbol")).upper()
            for item in overlap_specs
            if isinstance(item, dict)
        }
        == {"LILA", "LILAK"},
        "Cross-validation policy must pin OLD LILA/LILAK external overlaps.",
    )
    for spec in overlap_specs:
        _require(isinstance(spec, dict), "Pinned external overlap entry is invalid.")
        external_url = urlparse(_text(spec.get("external_source_url")))
        primary_url = urlparse(_text(spec.get("primary_source_url")))
        _require(
            pinned_external_overlap_spec_is_trusted(spec)
            and spec.get("primary_source") == "yahoo_chart_adjusted_basis_primary"
            and spec.get("external_source") == "boris_kaggle_cc0_v3"
            and primary_url.scheme == "https"
            and (primary_url.hostname or "").lower() == INDEPENDENT_PRICE_HOST
            and external_url.scheme == "https"
            and (external_url.hostname or "").lower() == "www.kaggle.com"
            and len(_text(spec.get("external_source_sha256"))) == 64
            and _integer(spec.get("overlap_sessions"), "overlap_sessions") == 597
            and _integer(spec.get("primary_sessions"), "primary_sessions") == 630
            and _integer(
                spec.get("uncrosschecked_tail_sessions"),
                "uncrosschecked_tail_sessions",
            )
            == 33
            and spec.get("upstream_provider_disclosed") is False
            and spec.get("independent_provider_claimed") is False,
            "Pinned external overlap controls changed.",
        )


def _validate_report_provider(provider: Any) -> None:
    _require(isinstance(provider, dict), "Archived report provider is invalid.")
    _require(
        provider.get("name") == INDEPENDENT_PRICE_PROVIDER,
        "Archived report does not use Yahoo chart.",
    )
    attempts = _integer(provider.get("http_attempts_this_run"), "http_attempts_this_run")
    _require(
        attempts <= 400
        and _integer(provider.get("request_cap"), "request_cap") == 400
        and _integer(
            provider.get("attempts_per_target_cap"), "attempts_per_target_cap"
        )
        == 1
        and _integer(provider.get("retry_count"), "retry_count") == 0
        and provider.get("raw_response_cache_required") is True
        and provider.get("exact_response_bytes_archived") is True
        and provider.get("adjustment_basis") == "raw_quote_ohlcv"
        and provider.get("personal_use_only") is True
        and provider.get("private_repository_required") is True
        and provider.get("private_r2_required") is True
        and provider.get("redistribution_allowed") is False
        and isinstance(provider.get("use_restriction"), str)
        and bool(provider["use_restriction"].strip()),
        "Archived Yahoo provider controls are incomplete.",
    )


def _safe_archive_path(root: Path, object_path: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / object_path).resolve()
    _require(
        resolved != resolved_root and resolved_root in resolved.parents,
        f"Cross-validation archive path escapes repository root: {object_path}",
    )
    return resolved


def _official_permanent_exception_url(value: Any) -> bool:
    parsed = urlparse(_text(value))
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        return False
    if host == "www.sec.gov":
        return parsed.path.startswith("/Archives/edgar/data/")
    if host == "www.fdic.gov":
        return parsed.path.startswith("/resources/resolutions/bank-failures/")
    return False


def _archive_payload(
    repository,
    archive: pd.DataFrame,
    archive_id: str,
    *,
    source_url: str = "",
) -> bytes:
    # Most archive rows are content addressed, so ``archive_id`` equals the
    # uncompressed payload hash.  A small number of provenance-preserving rows
    # predate that convention and retain a distinct record ID while storing the
    # exact payload digest in ``source_hash``.  Resolve either exact binding,
    # but still fail closed on absent or duplicated evidence and always verify
    # the uncompressed bytes below.
    if source_url:
        matches = archive.loc[
            archive["source_hash"].astype(str).eq(archive_id)
            & archive["source_url"].astype(str).eq(source_url)
        ]
    else:
        matches = archive.loc[archive["archive_id"].astype(str).eq(archive_id)]
        if matches.empty:
            matches = archive.loc[
                archive["source_hash"].astype(str).eq(archive_id)
            ]
    _require(
        len(matches) == 1,
        "Cross-validation evidence is absent or duplicated in source_archive: "
        f"{archive_id}",
    )
    row = matches.iloc[0]
    _require(
        str(row.get("source_hash", "")) == archive_id,
        f"Cross-validation source_archive source_hash mismatch: {archive_id}",
    )
    path = _safe_archive_path(repository.root, str(row["object_path"]))
    _require(path.is_file(), f"Missing cross-validation archive payload: {path}")
    encoded = path.read_bytes()
    try:
        payload = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    except Exception as exc:
        raise RuntimeError(
            f"Unreadable cross-validation archive payload: {path}"
        ) from exc
    _require(
        sha256_bytes(payload) == archive_id,
        f"Cross-validation archive payload hash mismatch: {archive_id}",
    )
    return payload


def _yahoo_archive_payload(
    repository,
    archive: pd.DataFrame,
    price_item: Mapping[str, Any],
) -> bytes:
    """Verify exact Yahoo response bytes and their target-specific request."""

    source_hash = _text(price_item.get("source_sha256")).lower()
    wrapper_hash = _text(price_item.get("cache_wrapper_sha256")).lower()
    source_url = _text(price_item.get("source_url"))
    raw_payload = _archive_payload(repository, archive, source_hash)
    wrapper_payload = _archive_payload(
        repository,
        archive,
        wrapper_hash,
        source_url=source_url,
    )
    try:
        envelope = json.loads(wrapper_payload)
        embedded = base64.b64decode(envelope["content_base64"], validate=True)
    except Exception as exc:
        raise RuntimeError("Archived Yahoo request envelope is invalid.") from exc
    _require(
        isinstance(envelope, dict)
        and wrapper_payload == canonical_json_bytes(envelope)
        and sha256_bytes(wrapper_payload) == wrapper_hash
        and _text(envelope.get("schema")) == "yahoo_chart_raw_response/v2"
        and _text(envelope.get("source_url")) == source_url
        and _text(envelope.get("symbol"))
        == _normalized_report_symbol(price_item.get("provider_symbol"))
        and _integer(envelope.get("request_period1"), "request_period1")
        == _integer(price_item.get("request_period1"), "request_period1")
        and _integer(envelope.get("request_period2"), "request_period2")
        == _integer(price_item.get("request_period2"), "request_period2")
        and _integer(envelope.get("http_status"), "http_status")
        == _integer(price_item.get("http_status"), "http_status")
        and _text(envelope.get("content_sha256")).lower() == source_hash
        and sha256_bytes(embedded) == source_hash
        and embedded == raw_payload
        and _text(envelope.get("content_type"))
        .lower()
        .split(";", 1)[0]
        .strip()
        == "application/json",
        "Archived Yahoo request/response provenance is not reproducible.",
    )
    return raw_payload


def _trusted_evidence_payload(
    repository,
    archive: pd.DataFrame,
    evidence: Mapping[str, Any],
) -> bytes:
    """Load one code-pinned archive row whose ID may differ from payload hash."""

    archive_id = _text(evidence.get("archive_id")) or _text(
        evidence.get("source_hash")
    ).lower()
    source_hash = _text(evidence.get("source_hash")).lower()
    matches = archive.loc[archive["archive_id"].map(_text).eq(archive_id)]
    _require(
        len(matches) == 1,
        "Trusted evidence is absent or duplicated in source_archive: " + archive_id,
    )
    row = matches.iloc[0]
    _require(
        _text(row.get("source_hash")).lower() == source_hash,
        "Trusted evidence source_hash changed: " + archive_id,
    )
    path = _safe_archive_path(repository.root, _text(row.get("object_path")))
    _require(path.is_file(), f"Missing trusted evidence payload: {path}")
    encoded = path.read_bytes()
    try:
        payload = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    except Exception as exc:
        raise RuntimeError(f"Unreadable trusted evidence payload: {path}") from exc
    _require(
        sha256_bytes(payload) == source_hash,
        "Trusted evidence payload hash mismatch: " + archive_id,
    )
    return payload


def _archive_payload_by_provenance(
    repository,
    archive: pd.DataFrame,
    *,
    source: str,
    source_url: str,
    source_hash: str,
) -> bytes:
    """Load one exact URL/hash/source row and attest its deterministic ID."""

    digest = _text(source_hash).lower()
    matches = archive.loc[
        archive["source_hash"].map(_text).str.lower().eq(digest)
        & archive["source_url"].map(_text).eq(_text(source_url))
        & archive["source"].map(_text).eq(_text(source))
    ]
    _require(
        len(matches) == 1
        and source_archive_binding_matches(
            matches.iloc[0].to_dict(),
            source=source,
            source_url=source_url,
            source_hash=digest,
        ),
        "Cross-validation exact provenance archive binding changed: " + digest,
    )
    row = matches.iloc[0]
    path = _safe_archive_path(repository.root, _text(row.get("object_path")))
    _require(path.is_file(), f"Missing cross-validation archive payload: {path}")
    encoded = path.read_bytes()
    try:
        payload = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    except Exception as exc:
        raise RuntimeError(
            f"Unreadable cross-validation archive payload: {path}"
        ) from exc
    _require(
        sha256_bytes(payload) == digest,
        "Cross-validation exact provenance payload hash mismatch: " + digest,
    )
    return payload


def _validate_terminal_price_tail_release_bindings(
    repository,
    release: DataRelease,
    corrections: Mapping[str, Mapping[str, Any]],
    used_event_ids: set[str],
    *,
    archive: pd.DataFrame,
    prices: pd.DataFrame,
    master: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[dict[str, str], ...]:
    """Re-attest exact repaired rows, evidence bytes, lineage and replay gaps."""

    if not used_event_ids:
        return ()
    _require(
        used_event_ids == (set(corrections) & used_event_ids),
        "Terminal price-tail release binding received an unknown event.",
    )
    versions = release.dataset_versions
    write_datasets = (
        "corporate_actions",
        "daily_price_raw",
        "adjustment_factors",
        "lifecycle_resolutions",
        "security_master",
        "symbol_history",
    )
    missing_versions = [dataset for dataset in write_datasets if not versions.get(dataset)]
    _require(
        not missing_versions,
        "Terminal price-tail release lacks rewritten datasets: "
        + ", ".join(missing_versions),
    )
    normalized_inventory = [
        _canonical_reviewed_terminal_price_tail_correction(value)
        for value in corrections.values()
    ]
    _require(
        canonical_json_sha256(normalized_inventory)
        == TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256,
        "Terminal price-tail policy registry is no longer code-pinned.",
    )
    used_legacy_event_ids = (
        used_event_ids & _TRUSTED_LEGACY_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS
    )
    used_short_event_ids = (
        used_event_ids & _TRUSTED_SHORT_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS
    )
    _require(
        used_event_ids == used_legacy_event_ids | used_short_event_ids,
        "Terminal price-tail release binding received an unclassified event.",
    )
    # The original three-row repair predates ``cash_amount`` and is preserved
    # byte-for-byte in these identity/price manifests.  The later short-tail
    # planner embeds its own complete six-row registry in every rewritten
    # dataset.  A future planner may instead embed the combined five-row
    # policy registry; all three paths remain exact code pins.
    legacy_manifest_datasets = {
        "daily_price_raw",
        "security_master",
        "symbol_history",
    }
    for dataset in write_datasets:
        manifest = repository.manifest_for_version(dataset, versions[dataset])
        metadata = manifest.metadata
        combined_inventory = metadata.get("terminal_tail_registry_draft")
        combined_match = bool(
            metadata.get("terminal_tail_registry_inventory_sha256")
            == TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
            and isinstance(combined_inventory, list)
            and canonical_json_sha256(combined_inventory)
            == TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
        )
        if combined_match:
            continue
        if used_legacy_event_ids and dataset in legacy_manifest_datasets:
            legacy_inventory = metadata.get("terminal_tail_registry_draft")
            _require(
                metadata.get("terminal_tail_registry_inventory_sha256")
                == _TRUSTED_LEGACY_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
                and isinstance(legacy_inventory, list)
                and canonical_json_sha256(legacy_inventory)
                == _TRUSTED_LEGACY_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
                and {
                    _text(item.get("event_id")).lower()
                    for item in legacy_inventory
                    if isinstance(item, Mapping)
                }
                == set(_TRUSTED_LEGACY_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS),
                "Legacy terminal price-tail manifest registry is missing or not "
                "code-pinned: "
                + dataset,
            )
        if used_short_event_ids:
            short_inventory = metadata.get("short_terminal_tail_registry")
            _require(
                metadata.get("short_terminal_tail_registry_sha256")
                == _TRUSTED_SHORT_TERMINAL_PRICE_TAIL_REGISTRY_SHA256
                and isinstance(short_inventory, list)
                and canonical_json_sha256(short_inventory)
                == _TRUSTED_SHORT_TERMINAL_PRICE_TAIL_REGISTRY_SHA256,
                "Short terminal price-tail manifest registry is missing or not "
                "code-pinned: "
                + dataset,
            )
            short_by_event = {
                _text(item.get("event_id")).lower(): item
                for item in short_inventory
                if isinstance(item, Mapping)
            }
            _require(
                len(short_by_event) == len(short_inventory)
                and set(_TRUSTED_SHORT_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS)
                <= set(short_by_event),
                "Short terminal price-tail manifest event inventory changed: "
                + dataset,
            )
            for event_id in sorted(used_short_event_ids):
                expected = corrections[event_id]
                actual = short_by_event[event_id]
                removed_sessions = actual.get("removed_sessions")
                exact_pairs = {
                    "symbol": (
                        _text(actual.get("symbol")).upper(),
                        expected["symbol"],
                    ),
                    "security_id": (
                        _text(actual.get("security_id")),
                        expected["security_id"],
                    ),
                    "old_candidate_id": (
                        _text(actual.get("old_candidate_id")).lower(),
                        expected["old_candidate_id"],
                    ),
                    "candidate_id": (
                        _text(actual.get("candidate_id")).lower(),
                        expected["candidate_id"],
                    ),
                    "old_event_id": (
                        _text(actual.get("old_event_id")).lower(),
                        expected["old_event_id"],
                    ),
                    "event_id": (
                        _text(actual.get("event_id")).lower(),
                        expected["event_id"],
                    ),
                    "action_type": (
                        _text(actual.get("action_type")).lower(),
                        expected["action_type"],
                    ),
                    "last_real_session": (
                        _date(actual.get("last_real_session")),
                        expected["last_real_session"],
                    ),
                    "legal_completion_date": (
                        _date(actual.get("legal_completion_date")),
                        expected["official_completion_date"],
                    ),
                    "market_transition_session": (
                        _date(actual.get("market_transition_session")),
                        expected["market_transition_session"],
                    ),
                    "raw_source_hash": (
                        _text(actual.get("raw_source_hash")).lower(),
                        expected["raw_source_hash"],
                    ),
                    "official_source_hash": (
                        _text(actual.get("official_source_hash")).lower(),
                        expected["official_source_hash"],
                    ),
                    "successor_source_hash": (
                        _text(actual.get("successor_raw_source_hash")).lower(),
                        expected["successor_source_hash"],
                    ),
                    "removed_tail_count": (
                        len(removed_sessions)
                        if isinstance(removed_sessions, list)
                        else -1,
                        expected["removed_tail_count"],
                    ),
                    "removed_tail_start": (
                        _date(removed_sessions[0])
                        if isinstance(removed_sessions, list)
                        and removed_sessions
                        else "",
                        expected["removed_tail_start"],
                    ),
                    "removed_tail_end": (
                        _date(removed_sessions[-1])
                        if isinstance(removed_sessions, list)
                        and removed_sessions
                        else "",
                        expected["removed_tail_end"],
                    ),
                }
                _require(
                    all(actual_value == expected_value for actual_value, expected_value in exact_pairs.values()),
                    "Short terminal price-tail manifest projection changed: "
                    + dataset
                    + ":"
                    + event_id,
                )

    price_sessions = pd.to_datetime(prices["session"], errors="coerce").dt.date.astype(str)
    for event_id in sorted(used_event_ids):
        correction = corrections[event_id]
        official_payload = _archive_payload(
            repository, archive, correction["official_source_hash"]
        )
        _require(
            len(official_payload) == correction["official_source_bytes"]
            and correction["filing_acceptance_datetime"].encode() in official_payload,
            "Terminal price-tail SEC payload does not prove the exact filing: "
            + event_id,
        )
        raw_payload = _archive_payload(
            repository, archive, correction["raw_source_hash"]
        )
        _require(
            len(raw_payload) == correction["raw_source_bytes"],
            "Terminal price-tail raw EOD payload size changed: " + event_id,
        )
        try:
            raw_records = json.loads(raw_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Terminal price-tail raw EOD payload is invalid JSON: " + event_id
            ) from exc
        _require(
            isinstance(raw_records, list)
            and all(isinstance(item, Mapping) for item in raw_records),
            "Terminal price-tail raw EOD payload is not an object list: " + event_id,
        )
        tail = [
            item
            for item in raw_records
            if correction["removed_tail_start"]
            <= _date(item.get("date"))
            <= correction["removed_tail_end"]
        ]
        _require(
            len(tail) == correction["removed_tail_count"]
            and bool(tail)
            and _date(tail[0].get("date")) == correction["removed_tail_start"]
            and _date(tail[-1].get("date")) == correction["removed_tail_end"]
            and canonical_json_sha256(tail) == correction["removed_tail_sha256"],
            "Terminal price-tail extraction no longer equals the reviewed rows: "
            + event_id,
        )
        if correction["successor_source_hash"]:
            _archive_payload(
                repository, archive, correction["successor_source_hash"]
            )

        source_rows = prices.loc[
            prices["security_id"].map(_text).eq(correction["security_id"])
        ]
        source_dates = price_sessions.loc[source_rows.index]
        identity_source = (
            "official_short_terminal_boundary_repair"
            if event_id
            in _TRUSTED_SHORT_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS
            else "official_terminal_boundary_repair"
        )
        _require(
            not source_rows.empty
            and source_dates.max() == correction["last_real_session"]
            and not source_dates.between(
                correction["removed_tail_start"], correction["removed_tail_end"]
            ).any(),
            "Terminal price-tail repaired price boundary is not exact: " + event_id,
        )
        master_rows = master.loc[
            master["security_id"].map(_text).eq(correction["security_id"])
        ]
        history_rows = history.loc[
            history["security_id"].map(_text).eq(correction["security_id"])
            & history["symbol"].map(_text).str.upper().eq(correction["symbol"])
        ]
        _require(
            len(master_rows) == 1
            and len(history_rows) == 1
            and _date(master_rows.iloc[0].get("active_to"))
            == correction["last_real_session"]
            and _date(history_rows.iloc[0].get("effective_to"))
            == correction["last_real_session"]
            and _text(master_rows.iloc[0].get("source")) == identity_source
            and _text(history_rows.iloc[0].get("source")) == identity_source
            and _text(master_rows.iloc[0].get("source_hash")).lower()
            == correction["official_source_hash"]
            and _text(history_rows.iloc[0].get("source_hash")).lower()
            == correction["official_source_hash"],
            "Terminal price-tail repaired identity boundary is not exact: " + event_id,
        )

    factors = repository.read_frame(
        "adjustment_factors", versions["adjustment_factors"]
    )
    lineage = (
        versions["daily_price_raw"] + "+" + versions["corporate_actions"]
    )
    factor_manifest = repository.manifest_for_version(
        "adjustment_factors", versions["adjustment_factors"]
    )
    _require(
        _text(factor_manifest.metadata.get("source_version")) == lineage
        and _text(factor_manifest.metadata.get("source_daily_price_version"))
        == versions["daily_price_raw"]
        and _text(factor_manifest.metadata.get("source_corporate_actions_version"))
        == versions["corporate_actions"]
        and set(factors["source_version"].map(_text)) == {lineage}
        and set(factors["source_hash"].map(_text)) == {lineage}
        and set(factors["source"].map(_text)) == {"derived"},
        "Terminal price-tail adjustment-factor lineage is not release-exact.",
    )

    gap_rows: list[dict[str, str]] = []
    membership_version = versions.get("index_membership_events", "")
    _require(
        bool(membership_version),
        "Terminal price-tail snapshot exception requires index membership events.",
    )
    membership = repository.read_frame(
        "index_membership_events", membership_version
    )
    for event_id, expected in TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS.items():
        if event_id not in used_event_ids:
            continue
        rows = membership.loc[
            membership["event_id"].map(_text).eq(expected["next_remove_event_id"])
        ]
        _require(
            len(rows) == 1
            and _text(rows.iloc[0].get("index_id")) == expected["index_id"]
            and _text(rows.iloc[0].get("security_id")) == expected["security_id"]
            and _text(rows.iloc[0].get("operation")).upper() == "REMOVE"
            and _date(rows.iloc[0].get("effective_date"))
            == expected["next_remove_effective_date"]
            and _text(rows.iloc[0].get("source")) == expected["next_remove_source"]
            and _text(rows.iloc[0].get("source_hash")).lower()
            == expected["next_remove_source_hash"],
            "Terminal price-tail snapshot exception removal lineage changed: "
            + event_id,
        )
        observed_fingerprint = index_member_identity_gap_fingerprint(
            index_id=expected["index_id"],
            replay_date=expected["replay_date"],
            security_id=expected["security_id"],
            next_remove_event_id=expected["next_remove_event_id"],
            next_remove_effective_date=expected["next_remove_effective_date"],
            next_remove_source=expected["next_remove_source"],
            next_remove_source_hash=expected["next_remove_source_hash"],
        )
        _require(
            observed_fingerprint == expected["fingerprint"],
            "Terminal price-tail snapshot exception fingerprint changed: " + event_id,
        )
        gap_rows.append(
            {
                "event_id": event_id,
                "index_id": expected["index_id"],
                "replay_date": expected["replay_date"],
                "security_id": expected["security_id"],
                "fingerprint": expected["fingerprint"],
            }
        )
    return tuple(gap_rows)


def _validated_versions(release: DataRelease) -> dict[str, str]:
    missing = [name for name in VALIDATED_DATASETS if not release.dataset_versions.get(name)]
    _require(
        not missing,
        "Release is missing cross-validation input datasets: " + ", ".join(missing),
    )
    return {name: release.dataset_versions[name] for name in VALIDATED_DATASETS}


def _load_versions_json(value: Any) -> dict[str, str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cross-validation validated_versions_json is invalid.") from exc
    _require(isinstance(parsed, dict), "Cross-validation versions must be an object.")
    return {str(key): str(item) for key, item in parsed.items()}


def _check_report_rows(report: Mapping[str, Any]) -> dict[str, int]:
    events = report.get("events")
    permanent_exceptions = report.get("permanent_exceptions")
    prices = report.get("prices")
    _require(isinstance(events, list), "Cross-validation events must be a list.")
    _require(
        isinstance(permanent_exceptions, list),
        "Cross-validation permanent exceptions must be a list.",
    )
    _require(isinstance(prices, list), "Cross-validation prices must be a list.")

    event_ids: set[str] = set()
    event_mismatches = 0
    nonterminal_events = 0
    reviewed_nonterminal_events = 0
    for item in events:
        _require(isinstance(item, dict), "Cross-validation event entry is invalid.")
        event_id = str(item.get("event_id", "")).strip()
        _require(event_id and event_id not in event_ids, "Duplicate cross-validation event_id.")
        event_ids.add(event_id)
        validation_kind = _text(item.get("validation_kind"))
        common_passed = (
            item.get("status") == "passed"
            and item.get("terms_match") is True
            and item.get("date_match") is True
            and item.get("official_original") is True
            and item.get("official_provenance_passed") is True
            and bool(_text(item.get("source_url")))
            and len(str(item.get("evidence_sha256", ""))) == 64
        )
        if validation_kind == TERMINAL_EVENT_VALIDATION:
            dedicated_ntco_required = (
                event_id in TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS
            )
            passed = (
                common_passed
                and item.get("lifecycle_report_extraction_approved") is True
                and bool(_text(item.get("candidate_id")))
                and (
                    not dedicated_ntco_required
                    or trusted_ntco_report_diagnostic_passed(item)
                )
            )
        elif validation_kind == NONTERMINAL_EVENT_VALIDATION:
            nonterminal_events += 1
            dedicated_sivb_passed = trusted_sivb_report_diagnostic_passed(item)
            dedicated_frc_passed = trusted_frc_report_diagnostic_passed(item)
            dedicated_ntco_passed = trusted_ntco_report_diagnostic_passed(item)
            passed = (
                common_passed
                and item.get("lifecycle_report_extraction_approved") is False
                and not _text(item.get("candidate_id"))
                and (
                    (
                        item.get("reviewed_extraction_match") is True
                        and len(_text(item.get("reviewed_extraction_sha256"))) == 64
                    )
                    or dedicated_sivb_passed
                    or dedicated_frc_passed
                    or dedicated_ntco_passed
                )
            )
            reviewed_nonterminal_events += int(passed)
        else:
            passed = False
        event_mismatches += int(not passed)

    permanent_candidate_ids: set[str] = set()
    permanent_exception_mismatches = 0
    for item in permanent_exceptions:
        _require(
            isinstance(item, dict),
            "Cross-validation permanent exception entry is invalid.",
        )
        candidate_id = _text(item.get("candidate_id"))
        _require(
            candidate_id and candidate_id not in permanent_candidate_ids,
            "Duplicate permanent lifecycle exception candidate_id.",
        )
        permanent_candidate_ids.add(candidate_id)
        passed = (
            _text(item.get("validation_kind"))
            == PERMANENT_EXCEPTION_VALIDATION
            and bool(_text(item.get("evidence_id")))
            and item.get("status") == "passed"
            and _text(item.get("exception_code")) in PERMANENT_EXCEPTION_CODES
            and bool(_text(item.get("exception_reason")))
            and item.get("identity_date_bound") is True
            and item.get("registry_binding_passed") is True
            and item.get("reviewer_pin_passed") is True
            and item.get("official_original") is True
            and item.get("exact_archive_pair") is True
            and item.get("archive_payload_verified") is True
            and _official_permanent_exception_url(item.get("source_url"))
            and len(_text(item.get("evidence_sha256"))) == 64
        )
        permanent_exception_mismatches += int(not passed)
    permanent_by_candidate_id = {
        _text(item.get("candidate_id")): item for item in permanent_exceptions
    }

    price_ids: set[str] = set()
    price_pass = 0
    price_exceptions = 0
    price_unresolved = 0
    price_mismatches = 0
    overlap_sessions = 0
    for item in prices:
        _require(isinstance(item, dict), "Cross-validation price entry is invalid.")
        target_id = str(item.get("target_id", "")).strip()
        _require(target_id and target_id not in price_ids, "Duplicate price target_id.")
        price_ids.add(target_id)
        status = str(item.get("status", ""))
        if status == "passed":
            price_pass += 1
            overlap_count = _integer(
                item.get("overlap_session_count"), "overlap_session_count"
            )
            if _text(item.get("validation_basis")) == REVIEWED_PRICE_EVIDENCE_BASIS:
                independent_rows = _integer(
                    item.get("independent_internal_price_rows"),
                    "independent_internal_price_rows",
                )
                _integer(
                    item.get("self_source_rows_excluded"),
                    "self_source_rows_excluded",
                )
                _require(
                    target_id in TRUSTED_REVIEWED_PRICE_EVIDENCE_TARGET_IDS
                    and item.get("reviewed_price_evidence_applied") is True
                    and _text(
                        item.get("reviewed_price_evidence_registry_sha256")
                    )
                    == TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256
                    and len(
                        _text(item.get("reviewed_price_evidence_sha256"))
                    )
                    == 64
                    and len(
                        _text(item.get("reviewed_price_projection_sha256"))
                    )
                    == 64
                    and all(
                        len(_text(item.get(key))) == 64
                        for key in (
                            "reviewed_internal_ohlcv_sha256",
                            "reviewed_provider_ohlcv_sha256",
                            "reviewed_overlap_ohlcv_sha256",
                            "reviewed_all_null_sessions_sha256",
                        )
                    )
                    and isinstance(item.get("reviewed_price_mismatch_rows"), list)
                    and isinstance(
                        item.get("reviewed_triple_supertrend_signal"), dict
                    )
                    and isinstance(item.get("reviewed_provider_metadata"), dict)
                    and bool(_text(item.get("reviewed_price_limitation")))
                    and item.get("reviewed_official_event_binding_passed") is True
                    and independent_rows >= overlap_count > 0
                    and item.get("all_overlap_sessions_compared") is True
                    and item.get("scale_stability_passed") is True
                    and item.get("price_tolerance_passed") is True
                    and item.get("session_coverage_passed") is True
                    and item.get("currency_passed") is True
                    and item.get("identity_boundary_passed") is True
                    and item.get("provider_adjustment_basis")
                    == "reviewed_exact_raw_quote_ohlcv"
                    and item.get("adjusted_close_used") is False,
                    f"Reviewed exact price report is incomplete: {target_id}",
                )
                overlap_sessions += overlap_count
                continue
            if _text(item.get("validation_basis")) == PINNED_EXTERNAL_OVERLAP_VALIDATION:
                _require(
                    overlap_count == 597
                    and _integer(
                        item.get("internal_history_session_count"),
                        "internal_history_session_count",
                    )
                    == 630
                    and _integer(
                        item.get("uncrosschecked_tail_sessions"),
                        "uncrosschecked_tail_sessions",
                    )
                    == 33
                    and item.get("all_overlap_sessions_compared") is True
                    and item.get("scale_stability_passed") is True
                    and item.get("price_tolerance_passed") is True
                    and item.get("session_coverage_passed") is True
                    and item.get("currency_passed") is True
                    and item.get("identity_boundary_passed") is True
                    and item.get("provider_currency") == "USD"
                    and item.get("provider_adjustment_basis")
                    == "scale_normalized_close_overlap"
                    and item.get("adjusted_close_used") is False
                    and item.get("upstream_provider_disclosed") is False
                    and item.get("independent_provider_claimed") is False
                    and len(_text(item.get("primary_source_sha256"))) == 64
                    and len(_text(item.get("source_sha256"))) == 64,
                    f"Pinned external overlap report is incomplete: {target_id}",
                )
                overlap_sessions += overlap_count
                continue
            independent_rows = _integer(
                item.get("independent_internal_price_rows"),
                "independent_internal_price_rows",
            )
            _integer(item.get("self_source_rows_excluded"), "self_source_rows_excluded")
            _require(
                independent_rows >= overlap_count > 0,
                f"Price target lacks independent internal overlap: {target_id}",
            )
            _require(
                item.get("all_overlap_sessions_compared") is True,
                f"Price target did not compare every overlap session: {target_id}",
            )
            _require(
                item.get("scale_stability_passed") is True,
                f"Price target has unstable adjustment scale: {target_id}",
            )
            _require(
                item.get("price_tolerance_passed") is True
                and item.get("session_coverage_passed") is True
                and item.get("currency_passed") is True
                and item.get("identity_boundary_passed") is True,
                f"Price target failed price/session policy: {target_id}",
            )
            _require(
                item.get("provider_currency") == "USD"
                and item.get("provider_adjustment_basis") == "raw_quote_ohlcv"
                and item.get("adjusted_close_used") is False,
                f"Price target did not use USD raw Yahoo quote OHLCV: {target_id}",
            )
            boundary_evidence = item.get("identity_boundary_evidence") or []
            _require(
                isinstance(boundary_evidence, list),
                f"Price target identity boundary evidence is invalid: {target_id}",
            )
            evidenced = {
                str(entry.get("boundary", ""))
                for entry in boundary_evidence
                if isinstance(entry, dict)
                and entry.get("official_original") is True
                and len(str(entry.get("evidence_sha256", ""))) == 64
            }
            provider_sessions_before = _integer(
                item.get("provider_sessions_before_identity"),
                "provider_sessions_before_identity",
            )
            provider_sessions_after = _integer(
                item.get("provider_sessions_after_identity"),
                "provider_sessions_after_identity",
            )
            _require(
                provider_sessions_before == 0
                or "active_from" in evidenced,
                f"Price target lacks official active_from evidence: {target_id}",
            )
            _require(
                provider_sessions_after == 0
                or "active_to" in evidenced,
                f"Price target lacks official active_to evidence: {target_id}",
            )
            overlap_sessions += overlap_count
        elif status == "explicit_exception":
            price_exceptions += 1
            if (
                _text(item.get("validation_basis"))
                == REVIEWED_REMAINING_PRICE_EXCEPTION_BASIS
            ):
                validate_reviewed_remaining_price_exception(item)
                overlap_sessions += _integer(
                    item.get("overlap_session_count", 0),
                    "overlap_session_count",
                )
                continue
            if (
                _text(item.get("validation_basis"))
                == REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS
            ):
                evidence = item.get(
                    "reviewed_source_archive_price_only_evidence"
                )
                overlap_count = _integer(
                    item.get("overlap_session_count"), "overlap_session_count"
                )
                _require(
                    target_id
                    in TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS
                    and item.get(
                        "reviewed_source_archive_price_only_evidence_applied"
                    )
                    is True
                    and _text(
                        item.get(
                            "reviewed_source_archive_price_only_registry_sha256"
                        )
                    )
                    == TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256
                    and len(
                        _text(
                            item.get(
                                "reviewed_source_archive_price_only_policy_sha256"
                            )
                        )
                    )
                    == 64
                    and len(
                        _text(
                            item.get(
                                "reviewed_source_archive_price_only_projection_sha256"
                            )
                        )
                    )
                    == 64
                    and isinstance(evidence, dict)
                    and evidence.get("validation_basis")
                    == REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS
                    and evidence.get("extract_sha256") == WIKI_EXTRACT_SHA256
                    and evidence.get("provenance_sha256")
                    == WIKI_PROVENANCE_SHA256
                    and evidence.get("action_factor_status")
                    == "incomplete_not_rewritten"
                    and evidence.get(
                        "price_only_pass_must_not_imply_action_factor_pass"
                    )
                    is True
                    and evidence.get("generic_ticker_reuse_allowed") is False
                    and evidence.get(
                        "yahoo_symbol_only_identity_reuse_allowed"
                    )
                    is False
                    and item.get("price_only_arbitration_passed") is True
                    and item.get("price_tolerance_passed") is False
                    and item.get("session_coverage_passed") is False
                    and item.get("corporate_actions_validated") is False
                    and item.get("adjustment_factors_validated") is False
                    and item.get("generic_ticker_reuse_allowed") is False
                    and item.get("provider_support")
                    == "reviewed_frozen_archive_price_only"
                    and item.get("provider_adjustment_basis")
                    == "frozen_wiki_raw_unadjusted_ohlcv_price_only"
                    and item.get("adjusted_close_used") is False
                    and overlap_count > 0,
                    f"Frozen WIKI price-only exception is incomplete: {target_id}",
                )
                if _text(item.get("symbol")).upper() == "BBT":
                    _require(
                        evidence.get("wiki_dividends_missing_from_current")
                        == [
                            {"date": "2015-05-13", "amount": 0.27},
                            {"date": "2015-08-12", "amount": 0.27},
                            {"date": "2015-11-10", "amount": 0.27},
                            {"date": "2016-02-10", "amount": 0.27},
                        ],
                        "Frozen WIKI BBT four-dividend gap was not preserved.",
                    )
                overlap_sessions += overlap_count
                continue
            if _text(item.get("validation_basis")) == REVIEWED_WIKI14_PRICE_ONLY_BASIS:
                evidence = item.get("reviewed_wiki14_price_only_evidence")
                overlap_count = _integer(
                    item.get("overlap_session_count"), "overlap_session_count"
                )
                _require(
                    target_id in TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS
                    and item.get("reviewed_wiki14_price_only_evidence_applied")
                    is True
                    and _text(
                        item.get("reviewed_wiki14_price_only_registry_sha256")
                    )
                    == TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256
                    and len(
                        _text(item.get("reviewed_wiki14_price_only_policy_sha256"))
                    )
                    == 64
                    and len(
                        _text(
                            item.get(
                                "reviewed_wiki14_price_only_projection_sha256"
                            )
                        )
                    )
                    == 64
                    and isinstance(evidence, dict)
                    and evidence.get("validation_basis")
                    == REVIEWED_WIKI14_PRICE_ONLY_BASIS
                    and evidence.get("provenance_sha256")
                    == WIKI14_PROVENANCE_SHA256
                    and evidence.get("action_factor_status")
                    == "incomplete_not_rewritten"
                    and evidence.get(
                        "price_only_pass_must_not_imply_action_factor_pass"
                    )
                    is True
                    and evidence.get("generic_ticker_reuse_allowed") is False
                    and evidence.get("yahoo_symbol_only_identity_reuse_allowed")
                    is False
                    and evidence.get("private_internal_only") is True
                    and evidence.get("redistribution_allowed") is False
                    and evidence.get("public_publication_allowed") is False
                    and item.get("price_only_arbitration_passed") is True
                    and item.get("price_tolerance_passed") is False
                    and item.get("session_coverage_passed") is False
                    and item.get("corporate_actions_validated") is False
                    and item.get("adjustment_factors_validated") is False
                    and item.get("generic_ticker_reuse_allowed") is False
                    and item.get("private_internal_only") is True
                    and item.get("redistribution_allowed") is False
                    and item.get("public_publication_allowed") is False
                    and item.get("provider_support")
                    == "reviewed_frozen_wiki14_archive_price_only"
                    and item.get("provider_adjustment_basis")
                    == "frozen_wiki_raw_unadjusted_ohlcv_price_only"
                    and item.get("adjusted_close_used") is False
                    and overlap_count > 0,
                    f"Frozen WIKI14 price-only exception is incomplete: {target_id}",
                )
                overlap_sessions += overlap_count
                continue
            exception = item.get("exception")
            validation_basis = _text(item.get("validation_basis"))
            if validation_basis == REVIEWED_PERMANENT_EXCEPTION_NO_DATA_BASIS:
                candidate_id = _text(
                    exception.get("candidate_id")
                    if isinstance(exception, Mapping)
                    else ""
                )
                permanent = permanent_by_candidate_id.get(candidate_id)
                _require(
                    isinstance(exception, dict)
                    and exception.get("code")
                    == PERMANENT_EXCEPTION_NO_DATA_CODE
                    and item.get(
                        "reviewed_permanent_exception_no_data_applied"
                    )
                    is True
                    and item.get("reviewed_unsupported_path_no_data_applied")
                    is False
                    and permanent is not None
                    and permanent.get("status") == "passed"
                    and _text(exception.get("evidence_id"))
                    == _text(permanent.get("evidence_id"))
                    and _text(exception.get("exception_code"))
                    == _text(permanent.get("exception_code"))
                    and _text(exception.get("exception_reason"))
                    == _text(permanent.get("exception_reason"))
                    and _text(exception.get("official_evidence_sha256")).lower()
                    == _text(permanent.get("evidence_sha256")).lower()
                    and _text(exception.get("official_source_url"))
                    == _text(permanent.get("source_url"))
                    and _text(item.get("security_id"))
                    == _text(permanent.get("security_id"))
                    and _text(item.get("symbol")).upper()
                    == _text(permanent.get("symbol")).upper()
                    and _date(exception.get("last_price_date"))
                    == _date(permanent.get("last_price_date"))
                    and exception.get("identity_date_match") is True
                    and exception.get("identity_date_basis")
                    == "permanent_exception_candidate_last_price"
                    and exception.get("price_history_supported") is False
                    and exception.get("generic_date_tolerance") is False
                    and not _text(item.get("terminal_event_id"))
                    and not _text(item.get("successor_security_id"))
                    and exception.get("successor_requirement_passed") is True,
                    "Permanent lifecycle no-data price exception is incomplete: "
                    + target_id,
                )
                _require(
                    item.get("provider_support") == "no_data"
                    and item.get("provider_currency")
                    == "unavailable_no_price_payload"
                    and item.get("provider_adjustment_basis")
                    == "no_price_payload"
                    and item.get("adjusted_close_used") is False
                    and item.get("response_identity_match") is True
                    and exception.get("response_identity_match") is True
                    and exception.get("no_data_evidence_validated") is True,
                    "Permanent lifecycle no-data price exception lacks exact "
                    "Yahoo evidence: "
                    + target_id,
                )
                continue
            if validation_basis == REVIEWED_NO_DATA_UNSUPPORTED_PATH_BASIS:
                _require(
                    isinstance(exception, dict)
                    and exception.get("code") == UNSUPPORTED_PATH_NO_DATA_CODE
                    and target_id
                    in TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_TARGET_IDS
                    and item.get("reviewed_unsupported_path_no_data_applied")
                    is True
                    and item.get(
                        "reviewed_permanent_exception_no_data_applied"
                    )
                    is False
                    and _text(exception.get("reviewed_registry_sha256"))
                    == TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_SHA256
                    and len(_text(exception.get("reviewed_spec_sha256"))) == 64
                    and exception.get("identity_date_match") is True
                    and exception.get("identity_date_basis")
                    == "reviewed_official_exit_mark_not_price_path"
                    and exception.get("price_history_supported") is False
                    and exception.get("generic_date_tolerance") is False
                    and exception.get("expected_internal_price_rows") == 1
                    and _text(exception.get("valuation_mark_close")) == "2.3"
                    and _date(exception.get("cash_available_session"))
                    == "2019-11-22"
                    and exception.get("index_scope") == "non_index_child"
                    and bool(_text(exception.get("limitation")))
                    and exception.get("official_event_verified") is True
                    and exception.get("identity_event_match") is True
                    and exception.get("successor_requirement_passed") is True
                    and not _text(item.get("successor_security_id"))
                    and item.get("provider_support") == "no_data"
                    and item.get("provider_currency")
                    == "unavailable_no_price_payload"
                    and item.get("provider_adjustment_basis")
                    == "no_price_payload"
                    and item.get("adjusted_close_used") is False
                    and item.get("response_identity_match") is True
                    and exception.get("response_identity_match") is True
                    and exception.get("no_data_evidence_validated") is True,
                    "Reviewed unsupported-path no-data exception is incomplete: "
                    + target_id,
                )
                continue
            if validation_basis == REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_BASIS:
                binding = (
                    exception.get("reviewed_nonterminal_same_sid_binding")
                    if isinstance(exception, Mapping)
                    else None
                )
                event_id = _text(
                    exception.get("official_event_id")
                    if isinstance(exception, Mapping)
                    else ""
                ).lower()
                spec = (
                    TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SPECS.get(
                        event_id
                    )
                )
                _require(
                    isinstance(binding, Mapping)
                    and spec is not None
                    and target_id == spec["source_target_id"]
                    and item.get(
                        "reviewed_nonterminal_same_sid_no_data_applied"
                    )
                    is True
                    and binding.get("code")
                    == "reviewed_nonterminal_same_sid_ticker_no_data"
                    and binding.get("validation_basis")
                    == REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_BASIS
                    and _text(binding.get("registry_sha256"))
                    == TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SHA256
                    and _text(binding.get("source_target_id"))
                    == spec["source_target_id"]
                    and _text(binding.get("successor_target_id"))
                    == spec["successor_target_id"]
                    and _text(binding.get("security_id"))
                    == spec["security_id"]
                    and binding.get("same_security_id_continuation") is True
                    and binding.get("terminal_resolution_required") is False
                    and binding.get("terminal_resolution_forbidden") is True,
                    "Reviewed nonterminal same-SID no-data exception is "
                    "incomplete: "
                    + target_id,
                )
            _require(
                isinstance(exception, dict)
                and exception.get("code") == "delisted_provider_unsupported"
                and _text(exception.get("official_action_type")).lower()
                in YAHOO_NO_DATA_TERMINAL_ACTION_TYPES
                and exception.get("official_event_verified") is True
                and exception.get("identity_event_match") is True
                and exception.get("identity_date_match") is True
                and exception.get("terminal_calendar_complete") is True
                and exception.get("successor_requirement_passed") is True,
                f"Invalid terminal-provider exception: {target_id}",
            )
            _require(
                item.get("provider_support") == "no_data"
                and item.get("provider_currency") == "unavailable_no_price_payload"
                and item.get("provider_adjustment_basis") == "no_price_payload"
                and item.get("adjusted_close_used") is False,
                f"Terminal-provider exception lacks valid Yahoo no-data evidence: {target_id}",
            )
            _require(
                item.get("response_identity_match") is True
                and exception.get("response_identity_match") is True
                and exception.get("no_data_evidence_validated") is True
                and item.get("no_data_evidence_kind")
                in {
                    "chart_not_found",
                    "http_200_empty_equity_chart",
                    "http_200_empty_retired_yhd_placeholder",
                    "http_400_bounded_history_not_found",
                },
                f"Terminal-provider exception lacks validated Yahoo no-data bytes: {target_id}",
            )
        elif status == "unresolved":
            price_unresolved += 1
        else:
            price_mismatches += 1

    return {
        "event_count": len(events),
        "event_mismatch_count": event_mismatches,
        "nonterminal_event_count": nonterminal_events,
        "reviewed_nonterminal_event_count": reviewed_nonterminal_events,
        "permanent_exception_count": len(permanent_exceptions),
        "permanent_exception_mismatch_count": permanent_exception_mismatches,
        "price_target_count": len(prices),
        "price_pass_count": price_pass,
        "price_exception_count": price_exceptions,
        "price_unresolved_count": price_unresolved,
        "price_mismatch_count": price_mismatches,
        "overlap_session_count": overlap_sessions,
    }


def validate_cross_validation_gate(
    repository,
    release: DataRelease,
) -> dict[str, Any]:
    """Prove independent event/price checks before any remote R2 access."""

    version = release.dataset_versions.get(CROSS_VALIDATION_DATASET)
    _require(
        bool(version),
        "Release must include cross_validation_reports before R2 access.",
    )
    archive_version = release.dataset_versions.get("source_archive")
    _require(bool(archive_version), "Release must include source_archive.")

    frame = repository.read_frame(CROSS_VALIDATION_DATASET, version)
    _require(len(frame) == 1, "Cross-validation release must contain exactly one report row.")
    row = frame.iloc[0]
    report_id = str(row["report_id"])
    _require(
        len(report_id) == 64
        and report_id == str(row["report_archive_id"])
        and report_id == str(row["source_hash"]),
        "Cross-validation report hashes are inconsistent.",
    )
    _require(str(row["status"]) == "passed", "Cross-validation status is not passed.")
    _require(
        str(row["provider"]) == INDEPENDENT_PRICE_PROVIDER,
        "Independent price provider must be Yahoo chart.",
    )

    expected_versions = _validated_versions(release)
    row_versions = _load_versions_json(row["validated_versions_json"])
    _require(
        row_versions == expected_versions,
        "Cross-validation input dataset versions do not match the current release.",
    )

    archive = repository.read_frame("source_archive", archive_version)
    report_payload = _archive_payload(repository, archive, report_id)
    try:
        report = json.loads(report_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Archived cross-validation report is not valid JSON.") from exc
    _require(
        isinstance(report, dict) and report.get("schema") == CROSS_VALIDATION_SCHEMA,
        "Cross-validation report schema is invalid.",
    )
    _require(
        report_payload == canonical_json_bytes(report),
        "Cross-validation report bytes are not canonical/reproducible.",
    )
    _require(report.get("status") == "passed", "Archived cross-validation report failed.")
    _validate_report_provider(report.get("provider"))
    _require(
        report.get("validated_versions") == expected_versions,
        "Archived report versions do not match the current release.",
    )

    resolutions = repository.read_frame(
        "lifecycle_resolutions", expected_versions["lifecycle_resolutions"]
    )
    lifecycle_manifest = repository.manifest_for_version(
        "lifecycle_resolutions", expected_versions["lifecycle_resolutions"]
    )
    lifecycle_evidence_report_sha256 = _text(
        lifecycle_manifest.metadata.get("evidence_report_sha256")
    ).lower()
    _require(
        len(lifecycle_evidence_report_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in lifecycle_evidence_report_sha256
        ),
        "Lifecycle evidence_report_sha256 is missing or invalid.",
    )
    try:
        lifecycle_evidence_report = json.loads(
            _archive_payload(
                repository, archive, lifecycle_evidence_report_sha256
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Lifecycle evidence report is invalid JSON.") from exc
    _require(
        isinstance(lifecycle_evidence_report, dict)
        and isinstance(lifecycle_evidence_report.get("records"), dict),
        "Lifecycle evidence report has no exact records object.",
    )
    expected_input_hashes = {
        "candidate_set_sha256": str(
            lifecycle_manifest.metadata.get("candidate_set_sha256", "")
        ),
        "lifecycle_resolutions_sha256": dataframe_sha256(
            resolutions, ("candidate_id",)
        ),
        "lifecycle_evidence_report_sha256": lifecycle_evidence_report_sha256,
    }
    _require(
        len(expected_input_hashes["candidate_set_sha256"]) == 64,
        "Lifecycle candidate_set_sha256 is missing.",
    )
    _require(
        report.get("input_hashes") == expected_input_hashes,
        "Cross-validation candidate/resolution hashes do not match the release.",
    )
    _require(
        _text(report.get("lifecycle_evidence_report_sha256")).lower()
        == lifecycle_evidence_report_sha256
        and _text(row.get("lifecycle_evidence_report_sha256")).lower()
        == lifecycle_evidence_report_sha256,
        "Cross-validation lifecycle evidence report fingerprint is inconsistent.",
    )
    policy = report.get("policy")
    _require(isinstance(policy, dict), "Cross-validation policy is missing.")
    _validate_policy_contract(policy)
    reviewed_extractions = reviewed_nonterminal_extractions(policy["events"])
    event_gates = reviewed_terminal_event_gates(policy["events"])
    terminal_overrides = reviewed_terminal_overrides(policy["events"])
    market_date_corrections = reviewed_terminal_market_date_corrections(
        policy["events"]
    )
    policy_exceptions = reviewed_terminal_policy_exceptions(policy["events"])
    tail_corrections = reviewed_terminal_price_tail_corrections(policy["events"])
    reviewed_price_registry = reviewed_price_evidence_registry(policy["prices"])
    reviewed_remaining_price_exceptions = (
        reviewed_remaining_price_exception_inventory()
    )
    source_archive_price_only_specs = source_archive_price_only_registry(
        policy["prices"]
    )
    wiki14_price_only_specs = wiki14_price_only_registry(policy["prices"])
    permanent_rows = resolutions.loc[
        resolutions["resolution"].astype(str).eq("exception")
        & resolutions["exception_code"].astype(str).isin(PERMANENT_EXCEPTION_CODES)
    ].copy()
    _require(
        permanent_rows["recheck_after"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .all(),
        "Permanent lifecycle exception cannot carry a temporary recheck date.",
    )
    report_permanent = report.get("permanent_exceptions")
    _require(
        isinstance(report_permanent, list),
        "Cross-validation permanent exception report is missing.",
    )
    report_permanent_by_id = {
        _text(item.get("candidate_id")): item
        for item in report_permanent
        if isinstance(item, dict) and _text(item.get("candidate_id"))
    }
    expected_permanent_ids = set(permanent_rows["candidate_id"].map(_text))
    _require(
        len(report_permanent_by_id) == len(report_permanent)
        and set(report_permanent_by_id) == expected_permanent_ids,
        "Cross-validation must cover every permanent lifecycle exception exactly once.",
    )
    permanent_exception_evidence_ids: set[str] = set()
    permanent_specs = (
        trusted_permanent_exception_specs() if len(permanent_rows) else {}
    )
    for resolution in permanent_rows.to_dict(orient="records"):
        candidate_id = _text(resolution.get("candidate_id"))
        security_id = _text(resolution.get("security_id"))
        last_price_date = _date(resolution.get("last_price_date"))
        item = report_permanent_by_id[candidate_id]
        official_spec = permanent_exception_spec_for_resolution(
            resolution, permanent_specs
        )
        source_url = _text(resolution.get("source_url"))
        source_hash = _text(resolution.get("source_hash")).lower()
        _require(
            official_spec is not None
            and official_spec.pinned
            and candidate_id == lifecycle_candidate_id(security_id, last_price_date)
            and _text(item.get("evidence_id")) == official_spec.evidence_id
            and _text(item.get("security_id")) == security_id
            and _text(item.get("symbol")).upper()
            == _text(resolution.get("symbol")).upper()
            and _date(item.get("last_price_date")) == last_price_date
            and _text(item.get("exception_code"))
            == _text(resolution.get("exception_code"))
            and _text(item.get("exception_reason"))
            == _text(resolution.get("exception_reason"))
            and _text(resolution.get("exception_code"))
            == official_spec.exception_code
            and _text(resolution.get("exception_reason")) == official_spec.claim
            and _text(item.get("source_url")) == source_url
            and _text(item.get("evidence_sha256")).lower() == source_hash
            and source_url == official_spec.source_url
            and source_hash == official_spec.source_sha256
            and _official_permanent_exception_url(source_url)
            and len(source_hash) == 64
            and item.get("status") == "passed"
            and item.get("identity_date_bound") is True
            and item.get("registry_binding_passed") is True
            and item.get("reviewer_pin_passed") is True
            and item.get("official_original") is True
            and item.get("exact_archive_pair") is True
            and item.get("archive_payload_verified") is True,
            "Permanent lifecycle exception identity/date/provenance is invalid: "
            + candidate_id,
        )
        archive_match = archive.loc[
            archive["archive_id"].map(_text).eq(source_hash)
        ]
        _require(
            len(archive_match) == 1
            and _text(archive_match.iloc[0].get("source_hash")).lower()
            == source_hash
            and _text(archive_match.iloc[0].get("source_url")) == source_url,
            "Permanent lifecycle exception URL/hash archive mismatch: "
            + candidate_id,
        )
        _archive_payload(repository, archive, source_hash)
        permanent_exception_evidence_ids.add(source_hash)
    applied = resolutions.loc[
        resolutions["resolution"].astype(str).eq("applied")
    ]
    actions = repository.read_frame(
        "corporate_actions", expected_versions["corporate_actions"]
    )
    master = repository.read_frame(
        "security_master", expected_versions["security_master"]
    )
    prices = repository.read_frame(
        "daily_price_raw", expected_versions["daily_price_raw"]
    )
    history = repository.read_frame(
        "symbol_history", expected_versions["symbol_history"]
    )
    required_action_columns = {
        "event_id",
        "security_id",
        "action_type",
        "effective_date",
        "new_security_id",
        "new_symbol",
        "cash_amount",
        "ratio",
        "currency",
        "source_url",
        "source_hash",
        "source_kind",
        "official",
    }
    _require(
        required_action_columns.issubset(actions.columns),
        "corporate_actions lacks lifecycle event evidence fields.",
    )
    lifecycle_actions = actions.loc[
        actions["action_type"].astype(str).str.lower().isin(
            {"cash_merger", "stock_merger", "spinoff", "ticker_change", "delisting"}
        )
    ].copy()
    lifecycle_actions["_event_id"] = lifecycle_actions["event_id"].map(_text)
    lifecycle_action_ids = lifecycle_actions["_event_id"].tolist()
    _require(
        all(lifecycle_action_ids)
        and len(lifecycle_action_ids) == len(set(lifecycle_action_ids)),
        "Every in-scope lifecycle corporate action requires one unique event_id.",
    )
    applied_groups = {
        event_id: group
        for event_id, group in applied.assign(
            _event_id=applied["event_id"].map(_text)
        ).groupby("_event_id", sort=False)
        if event_id
    }
    _require(
        set(applied_groups).issubset(set(lifecycle_action_ids))
        and all(len(group) == 1 for group in applied_groups.values())
        and len(applied_groups) == len(applied),
        "Cross-validation must bind every applied terminal resolution to exactly "
        "one in-scope lifecycle corporate action.",
    )
    report_events = report.get("events")
    _require(isinstance(report_events, list), "Cross-validation events must be a list.")
    report_event_by_id: dict[str, Mapping[str, Any]] = {}
    for item in report_events:
        _require(isinstance(item, dict), "Cross-validation event entry is invalid.")
        event_id = _text(item.get("event_id"))
        _require(
            event_id and event_id not in report_event_by_id,
            "Cross-validation event report is not one-to-one.",
        )
        report_event_by_id[event_id] = item
    _require(
        set(report_event_by_id) == set(lifecycle_action_ids),
        "Cross-validation does not cover every in-scope lifecycle corporate action exactly once.",
    )
    allowed_terminal_hosts = {
        _text(value).lower() for value in policy["events"]["official_hosts"]
    }
    allowed_provenance_hosts = {
        _text(value).lower()
        for value in policy["events"]["official_provenance_hosts"]
    }
    allowed_provenance_kinds = {
        _text(value)
        for value in policy["events"]["official_provenance_source_kinds"]
    }
    used_market_date_corrections: set[str] = set()
    used_policy_exceptions: set[str] = set()
    used_tail_corrections: set[str] = set()
    used_event_gates: set[str] = set()
    for action in lifecycle_actions.to_dict(orient="records"):
        event_id = _text(action.get("event_id"))
        event = report_event_by_id[event_id]
        event_gate = event_gates.get(event_id)
        security_id = _text(action.get("security_id"))
        source_url = _text(action.get("source_url"))
        source_hash = _text(action.get("source_hash")).lower()
        archive_match = archive.loc[
            archive["source_hash"].map(_text).str.lower().eq(source_hash)
            & archive["source_url"].map(_text).eq(source_url)
        ]
        if event_gate is None:
            raw_archive_source = (
                _text(archive_match.iloc[0].get("source"))
                if len(archive_match) == 1
                else ""
            )
            _require(
                len(archive_match) == 1
                and raw_archive_source
                and _text(archive_match.iloc[0].get("dataset"))
                == raw_archive_source
                and source_archive_binding_matches(
                    archive_match.iloc[0].to_dict(),
                    source=raw_archive_source,
                    source_url=source_url,
                    source_hash=source_hash,
                ),
                f"Lifecycle event URL/hash provenance mismatch: {event_id}",
            )
            _archive_payload_by_provenance(
                repository,
                archive,
                source=raw_archive_source,
                source_url=source_url,
                source_hash=source_hash,
            )
        else:
            _require(
                not archive_match.empty,
                f"Reviewed lifecycle event URL/hash evidence is missing: {event_id}",
            )
        common_action_report_binding = (
            _text(event.get("security_id")) == security_id
            and _text(event.get("action_type")).lower()
            == _text(action.get("action_type")).lower()
            and _date(event.get("effective_date"))
            == _date(action.get("effective_date"))
            and _text(event.get("new_security_id"))
            == _text(action.get("new_security_id"))
            and _text(event.get("new_symbol")).upper()
            == _text(action.get("new_symbol")).upper()
            and _exact_number_text(event.get("ratio"), "ratio")
            == _exact_number_text(action.get("ratio"), "ratio")
            and _exact_number_text(event.get("cash_amount"), "cash_amount")
            == _exact_number_text(action.get("cash_amount"), "cash_amount")
            and _text(event.get("currency")).upper()
            == _text(action.get("currency")).upper()
            and _text(event.get("source_kind"))
            == _text(action.get("source_kind"))
            and _text(event.get("source_url")) == source_url
            and _text(event.get("evidence_sha256")).lower() == source_hash
        )
        common_report_match = (
            common_action_report_binding
            and event.get("official_original") is True
            and event.get("official_provenance_passed") is True
        )
        resolution_group = applied_groups.get(event_id)
        sivb_evidence_binding = (
            trusted_sivb_evidence_binding_diagnostic(action, archive)
        )
        _require(
            event.get("trusted_sivb_evidence_binding")
            == sivb_evidence_binding,
            "Trusted SIVB evidence diagnostic differs: "
            + event_id,
        )
        if sivb_evidence_binding is not None:
            spec = trusted_sivb_evidence_bindings()[event_id]
            _require(
                sivb_evidence_binding["action_binding_exact"] is True
                and sivb_evidence_binding["status"] == "trusted"
                and sivb_evidence_binding["sec_raw_archived"] is True
                and sivb_evidence_binding["eodhd_raw_archived"] is True
                and sivb_evidence_binding["occ_raw_pdf_archived"] is True
                and sivb_evidence_binding["legacy_extraction_archived"] is True
                and sivb_evidence_binding["legacy_extraction_authoritative"] is False,
                "Trusted SIVB evidence binding is not exact: "
                + event_id,
            )
            for evidence in spec["evidence"]:
                payload = _archive_payload(
                    repository, archive, str(evidence["source_hash"])
                )
                _require(
                    len(payload) == int(evidence["content_bytes"]),
                    "Trusted SIVB evidence byte count changed: "
                    + str(evidence["role"]),
                )
        frc_evidence_binding = trusted_frc_evidence_binding_diagnostic(
            action, archive
        )
        _require(
            event.get("trusted_frc_evidence_binding") == frc_evidence_binding,
            "Trusted FRC evidence diagnostic differs: " + event_id,
        )
        if frc_evidence_binding is not None:
            spec = trusted_frc_evidence_bindings()[event_id]
            _require(
                frc_evidence_binding["action_binding_exact"] is True
                and frc_evidence_binding["status"] == "trusted"
                and frc_evidence_binding["occ_raw_pdf_archived"] is True
                and frc_evidence_binding["legacy_extraction_archived"] is True
                and frc_evidence_binding["legacy_extraction_authoritative"] is False,
                "Trusted FRC evidence binding is not exact: " + event_id,
            )
            for evidence in spec["evidence"]:
                payload = _trusted_evidence_payload(repository, archive, evidence)
                _require(
                    len(payload) == int(evidence["content_bytes"]),
                    "Trusted FRC evidence byte count changed: "
                    + str(evidence["role"]),
                )
                if evidence["role"] == "occ_memo_52352_raw_pdf":
                    _require(
                        payload.startswith(b"%PDF-")
                        and b"%%EOF" in payload[-4096:],
                        "Trusted FRC OCC 52352 payload is not a complete PDF.",
                    )
        ntco_evidence_binding = trusted_ntco_evidence_binding_diagnostic(
            action, archive
        )
        _require(
            event.get("trusted_ntco_evidence_binding")
            == ntco_evidence_binding,
            "Trusted NTCO evidence diagnostic differs: " + event_id,
        )
        if ntco_evidence_binding is not None:
            spec = trusted_ntco_evidence_bindings()[event_id]
            _require(
                ntco_evidence_binding["action_binding_exact"] is True
                and ntco_evidence_binding["status"] == "trusted"
                and ntco_evidence_binding["expected_terminal"]
                is (resolution_group is not None)
                and ntco_evidence_binding[
                    "derived_action_evidence_archived"
                ]
                is True
                and ntco_evidence_binding["raw_official_evidence_archived"]
                is True,
                "Trusted NTCO evidence binding is not exact: " + event_id,
            )
            for evidence in spec["evidence"]:
                payload = _trusted_evidence_payload(
                    repository, archive, evidence
                )
                _require(
                    len(payload) == int(evidence["content_bytes"]),
                    "Trusted NTCO evidence byte count changed: "
                    + str(evidence["role"]),
                )
                if evidence.get("raw_payload") is True:
                    _require(
                        payload.startswith(b"%PDF-")
                        and b"%%EOF" in payload[-4096:],
                        "Trusted NTCO raw official payload is not a complete PDF: "
                        + str(evidence["role"]),
                    )
        if resolution_group is not None:
            resolution = resolution_group.iloc[0]
            if event_gate is None:
                event_gate_attested = (
                    event.get("reviewed_terminal_event_gate_applied") is not True
                    and event.get("reviewed_terminal_event_gate_match") is not True
                    and not _text(
                        event.get("reviewed_terminal_event_gate_sha256")
                    )
                )
            else:
                record = lifecycle_evidence_report["records"].get(security_id)
                event_gate_attested = (
                    not reviewed_terminal_event_gate_mismatches(
                        action,
                        resolution,
                        record,
                        archive,
                        event_gate,
                        lifecycle_evidence_report_sha256,
                    )
                    and (
                        _text(event_gate.get("policy_code"))
                        != "sivbq_verified_legal_cancellation/v1"
                        or sivb_evidence_binding is not None
                    )
                    and event.get("reviewed_terminal_event_gate_applied") is True
                    and event.get("reviewed_terminal_event_gate_match") is True
                    and _text(event.get("reviewed_terminal_event_gate_sha256"))
                    == reviewed_terminal_event_gate_sha256(event_gate)
                    and event.get("date_match") is True
                    and event.get("terms_match") is True
                )
                if event_gate_attested:
                    used_event_gates.add(event_id)
                    for archive_id in event_gate["archive_ids"]:
                        gate_rows = archive.loc[
                            archive["archive_id"]
                            .map(_text)
                            .str.lower()
                            .eq(_text(archive_id).lower())
                        ]
                        gate_row = gate_rows.iloc[0]
                        _trusted_evidence_payload(
                            repository,
                            archive,
                            {
                                "archive_id": _text(archive_id).lower(),
                                "source_hash": _text(
                                    gate_row.get("source_hash")
                                ).lower(),
                            },
                        )
            terminal_override = terminal_overrides.get(event_id)
            if terminal_override is None:
                terminal_override_attested = (
                    event.get("reviewed_terminal_override_applied") is not True
                    and event.get("reviewed_terminal_override_match") is not True
                    and not _text(
                        event.get("reviewed_terminal_override_sha256")
                    )
                )
            else:
                record = lifecycle_evidence_report["records"].get(security_id)
                terminal_override_attested = (
                    not reviewed_terminal_override_mismatches(
                        action, terminal_override
                    )
                    and not reviewed_terminal_report_mismatches(
                        action, resolution, record
                    )
                    and event.get("reviewed_terminal_override_applied") is True
                    and event.get("reviewed_terminal_override_match") is True
                    and _text(event.get("reviewed_terminal_override_sha256"))
                    == reviewed_terminal_override_sha256(terminal_override)
                )
            policy_exception = policy_exceptions.get(event_id)
            if policy_exception is None:
                policy_exception_attested = (
                    event.get("reviewed_terminal_policy_exception_applied")
                    is not True
                    and event.get("reviewed_terminal_policy_exception_match")
                    is not True
                    and not _text(
                        event.get("reviewed_terminal_policy_exception_sha256")
                    )
                    and not _text(
                        event.get("reviewed_terminal_policy_exception_code")
                    )
                )
            else:
                record = lifecycle_evidence_report["records"].get(security_id)
                policy_exception_attested = (
                    not reviewed_terminal_policy_action_mismatches(
                        action, policy_exception
                    )
                    and not reviewed_terminal_policy_report_mismatches(
                        action,
                        resolution,
                        record,
                        policy_exception,
                        lifecycle_evidence_report_sha256,
                    )
                    and not reviewed_terminal_policy_release_warning_mismatches(
                        release.warnings, policy_exception
                    )
                    and event.get("reviewed_terminal_policy_exception_applied")
                    is True
                    and event.get("reviewed_terminal_policy_exception_match")
                    is True
                    and _text(
                        event.get("reviewed_terminal_policy_exception_sha256")
                    )
                    == reviewed_terminal_policy_exception_sha256(
                        policy_exception
                    )
                    and _text(
                        event.get("reviewed_terminal_policy_exception_code")
                    )
                    == policy_exception["policy_code"]
                    and event.get("date_match") is True
                    and event.get("terms_match") is True
                )
                if policy_exception_attested:
                    used_policy_exceptions.add(event_id)
            market_date_correction = market_date_corrections.get(event_id)
            tail_correction = tail_corrections.get(event_id)
            if market_date_correction is None:
                market_date_correction_attested = (
                    event.get(
                        "reviewed_terminal_market_date_correction_applied"
                    )
                    is not True
                    and event.get(
                        "reviewed_terminal_market_date_correction_match"
                    )
                    is not True
                    and not _text(
                        event.get(
                            "reviewed_terminal_market_date_correction_sha256"
                        )
                    )
                    and (
                        tail_correction is not None
                        or (
                            not _text(event.get("official_completion_date"))
                            and not _text(event.get("terminal_market_date_relation"))
                        )
                    )
                )
            else:
                record = lifecycle_evidence_report["records"].get(security_id)
                market_date_correction_attested = (
                    not reviewed_terminal_market_date_action_mismatches(
                        action, market_date_correction
                    )
                    and not reviewed_terminal_market_date_report_mismatches(
                        action,
                        resolution,
                        record,
                        market_date_correction,
                        lifecycle_evidence_report_sha256,
                    )
                    and event.get(
                        "reviewed_terminal_market_date_correction_applied"
                    )
                    is True
                    and event.get(
                        "reviewed_terminal_market_date_correction_match"
                    )
                    is True
                    and _text(
                        event.get(
                            "reviewed_terminal_market_date_correction_sha256"
                        )
                    )
                    == reviewed_terminal_market_date_correction_sha256(
                        market_date_correction
                    )
                    and _date(event.get("lifecycle_report_effective_date"))
                    == market_date_correction["report_effective_date"]
                    and _date(event.get("official_completion_date"))
                    == market_date_correction["official_completion_date"]
                    and _text(event.get("terminal_market_date_relation"))
                    == market_date_correction["date_relation"]
                    and event.get("date_match") is True
                )
                if market_date_correction_attested:
                    used_market_date_corrections.add(event_id)
            if tail_correction is None:
                tail_correction_attested = (
                    event.get(
                        "reviewed_terminal_price_tail_correction_applied"
                    )
                    is not True
                    and event.get(
                        "reviewed_terminal_price_tail_correction_match"
                    )
                    is not True
                    and not _text(
                        event.get(
                            "reviewed_terminal_price_tail_correction_sha256"
                        )
                    )
                )
            else:
                record = lifecycle_evidence_report["records"].get(security_id)
                tail_report_mismatches = (
                    reviewed_terminal_price_tail_report_mismatches(
                        action,
                        resolution,
                        record,
                        archive,
                        tail_correction,
                        lifecycle_evidence_report_sha256,
                    )
                )
                if event_gate is not None and event_gate_attested:
                    superseded_report_projection = {
                        "lifecycle_evidence_report_sha256",
                        "candidate_last_price_date",
                        "candidate_active_to",
                        "old_candidate_id",
                        "report_crosscheck_old_price_session",
                    }
                    tail_report_mismatches = tuple(
                        field
                        for field in tail_report_mismatches
                        if field not in superseded_report_projection
                    )
                tail_correction_attested = (
                    not reviewed_terminal_price_tail_action_mismatches(
                        action, tail_correction
                    )
                    and not tail_report_mismatches
                    and event.get(
                        "reviewed_terminal_price_tail_correction_applied"
                    )
                    is True
                    and event.get(
                        "reviewed_terminal_price_tail_correction_match"
                    )
                    is True
                    and _text(
                        event.get(
                            "reviewed_terminal_price_tail_correction_sha256"
                        )
                    )
                    == reviewed_terminal_price_tail_correction_sha256(
                        tail_correction
                    )
                    and _date(event.get("lifecycle_report_effective_date"))
                    == tail_correction["report_effective_date"]
                    and _date(event.get("official_completion_date"))
                    == tail_correction["official_completion_date"]
                    and _text(event.get("terminal_market_date_relation"))
                    == tail_correction["date_relation"]
                    and event.get("date_match") is True
                )
                if tail_correction_attested:
                    used_tail_corrections.add(event_id)
            _require(
                common_report_match
                and event_gate_attested
                and terminal_override_attested
                and policy_exception_attested
                and market_date_correction_attested
                and tail_correction_attested
                and _text(event.get("validation_kind"))
                == TERMINAL_EVENT_VALIDATION
                and _text(resolution.get("security_id")) == security_id
                and _text(event.get("candidate_id"))
                == _text(resolution.get("candidate_id"))
                and event.get("lifecycle_report_extraction_approved") is True
                and _text(action.get("official")).lower() == "true"
                and (
                    _text(action.get("source_kind"))
                    in {
                        _text(item)
                        for item in policy["events"].get(
                            "terminal_official_source_kinds", ()
                        )
                    }
                    or (
                        event_gate is not None
                        and event_gate_attested
                    )
                    or (
                        policy_exception is not None
                        and policy_exception_attested
                    )
                    or ntco_evidence_binding is not None
                )
                and (
                    _official_url(source_url, allowed_terminal_hosts)
                    or ntco_evidence_binding is not None
                )
                and len(source_hash) == 64,
                "Terminal lifecycle event identity/report/provenance is invalid: "
                + event_id,
            )
        else:
            if (
                sivb_evidence_binding is not None
                or frc_evidence_binding is not None
                or ntco_evidence_binding is not None
            ):
                _require(
                    common_report_match
                    and _text(event.get("validation_kind"))
                    == NONTERMINAL_EVENT_VALIDATION
                    and not _text(event.get("candidate_id"))
                    and event.get("lifecycle_report_extraction_approved") is False
                    and event.get("date_match") is True
                    and event.get("terms_match") is True
                    and event.get("reviewed_extraction_match") is False
                    and not _text(event.get("reviewed_extraction_sha256"))
                    and _text(action.get("official")).lower() == "true"
                    and _nonterminal_terms_complete(action),
                    "Dedicated exact ticker-action provenance is invalid: "
                    + event_id,
                )
                continue
            reviewed = reviewed_extractions.get(event_id)
            _require(
                reviewed is not None,
                "Nonterminal lifecycle event has no reviewed exact extraction: "
                + event_id,
            )
            reviewed_mismatches = reviewed_nonterminal_extraction_mismatches(
                action, reviewed
            )
            reviewed_hash = reviewed_nonterminal_extraction_sha256(reviewed)
            _require(
                common_report_match
                and _text(event.get("validation_kind"))
                == NONTERMINAL_EVENT_VALIDATION
                and not _text(event.get("candidate_id"))
                and event.get("lifecycle_report_extraction_approved") is False
                and event.get("date_match") is True
                and event.get("terms_match") is True
                and _text(action.get("official")).lower() == "true"
                and _text(action.get("source_kind")) in allowed_provenance_kinds
                and _official_url(source_url, allowed_provenance_hosts)
                and _nonterminal_terms_complete(action)
                and len(source_hash) == 64
                and not reviewed_mismatches
                and event.get("reviewed_extraction_match") is True
                and _text(event.get("reviewed_extraction_sha256"))
                == reviewed_hash,
                "Nonterminal lifecycle event official provenance is invalid: "
                + event_id,
            )

    _require(
        used_event_gates == (set(event_gates) & set(lifecycle_action_ids)),
        "Cross-validation must attest every reviewed terminal event gate "
        "present in the release exactly once.",
    )
    _require(
        used_market_date_corrections
        == (set(market_date_corrections) & set(lifecycle_action_ids)),
        "Cross-validation must attest every reviewed terminal market-date "
        "correction present in the release exactly once.",
    )
    _require(
        used_policy_exceptions
        == (set(policy_exceptions) & set(lifecycle_action_ids)),
        "Cross-validation must attest every reviewed terminal policy exception "
        "present in the release exactly once.",
    )
    _require(
        used_tail_corrections
        == (set(tail_corrections) & set(lifecycle_action_ids)),
        "Cross-validation must attest every reviewed terminal price-tail "
        "correction present in the release exactly once.",
    )
    reviewed_snapshot_identity_gaps = _validate_terminal_price_tail_release_bindings(
        repository,
        release,
        tail_corrections,
        used_tail_corrections,
        archive=archive,
        prices=prices,
        master=master,
        history=history,
    )

    expected_price_security_ids = {
        str(value).strip()
        for value in lifecycle_actions["security_id"]
        if pd.notna(value) and str(value).strip()
    }
    expected_price_security_ids.update(
        str(value).strip()
        for value in lifecycle_actions["new_security_id"]
        if pd.notna(value) and str(value).strip()
    )
    expected_price_security_ids.update(
        str(value).strip()
        for value in applied["security_id"]
        if pd.notna(value) and str(value).strip()
    )
    expected_price_security_ids.update(
        str(value).strip()
        for value in applied["successor_security_id"]
        if pd.notna(value) and str(value).strip()
    )
    expected_price_security_ids.update(provider_affected_identity_ids(master, prices))
    report_price_security_ids = {
        str(item.get("security_id", "")) for item in report.get("prices", ())
    }
    _require(
        report_price_security_ids == expected_price_security_ids,
        "Cross-validation price targets do not cover every lifecycle identity "
        "and every independent-provider-affected identity.",
    )
    expected_price_targets = _expected_price_targets(
        master, history, expected_price_security_ids
    )
    report_price_targets = {
        _text(item.get("target_id")): item for item in report.get("prices", ())
    }
    _require(
        len(report_price_targets) == len(report.get("prices", ()))
        and set(report_price_targets) == set(expected_price_targets),
        "Cross-validation price targets must cover every symbol_history interval "
        "for every lifecycle identity and every independent-provider-affected identity.",
    )
    for target_id, expected_target in expected_price_targets.items():
        item = report_price_targets[target_id]
        _require(
            _text(item.get("security_id")) == expected_target["security_id"]
            and _text(item.get("symbol")).upper() == expected_target["symbol"]
            and _normalized_report_symbol(item.get("provider_symbol"))
            == expected_target["provider_symbol"]
            and _date(item.get("identity_active_from"))
            == expected_target["active_from"]
            and _date(item.get("identity_active_to")) == expected_target["active_to"],
            f"Cross-validation canonical symbol interval does not match: {target_id}",
        )
    independent_price_ids = {
        str(value).strip()
        for value in prices.loc[~independent_provider_source_mask(prices), "security_id"]
        if pd.notna(value) and str(value).strip()
    }
    for item in report.get("prices", ()):
        if (
            item.get("status") == "passed"
            and _text(item.get("validation_basis"))
            != PINNED_EXTERNAL_OVERLAP_VALIDATION
        ):
            _require(
                str(item.get("security_id", "")).strip() in independent_price_ids,
                "Passed price target has no provider-independent internal source: "
                + str(item.get("security_id", "")),
            )
    for key in ("access_class", "stability_note", "use_restriction"):
        _require(
            report["provider"].get(key) == policy["provider"].get(key),
            f"Archived Yahoo provider report does not preserve policy field {key}.",
        )
    _require(
        report["provider"].get("request_mode")
        == "bounded_period1_period2_daily"
        and report["provider"].get("range_max_allowed") is False
        and report["provider"].get("period2_semantics")
        == "exclusive_next_utc_midnight"
        and report["provider"].get("data_granularity_required") == "1d"
        and report["provider"].get("xnys_inventory_recomputed") is True,
        "Archived Yahoo provider report lacks the bounded daily request contract.",
    )
    policy_sha256 = canonical_json_sha256(policy)
    _require(
        policy_sha256 == str(row["policy_sha256"]),
        "Cross-validation policy hash mismatch.",
    )

    computed = _check_report_rows(report)
    pinned_overlap_count = sum(
        _text(item.get("validation_basis")) == PINNED_EXTERNAL_OVERLAP_VALIDATION
        for item in report.get("prices", ())
    )
    reviewed_price_count = sum(
        _text(item.get("validation_basis")) == REVIEWED_PRICE_EVIDENCE_BASIS
        and item.get("status") == "passed"
        for item in report.get("prices", ())
    )
    expected_reviewed_price_ids = set(reviewed_price_registry) & set(
        expected_price_targets
    )
    source_archive_price_only_count = sum(
        _text(item.get("validation_basis"))
        == REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS
        and item.get("status") == "explicit_exception"
        for item in report.get("prices", ())
    )
    expected_source_archive_price_only_ids = set(
        source_archive_price_only_specs
    ) & set(expected_price_targets)
    wiki14_price_only_count = sum(
        _text(item.get("validation_basis")) == REVIEWED_WIKI14_PRICE_ONLY_BASIS
        and item.get("status") == "explicit_exception"
        for item in report.get("prices", ())
    )
    expected_wiki14_price_only_ids = set(wiki14_price_only_specs) & set(
        expected_price_targets
    )
    reviewed_remaining_price_exception_count = sum(
        _text(item.get("validation_basis"))
        == REVIEWED_REMAINING_PRICE_EXCEPTION_BASIS
        and item.get("status") == "explicit_exception"
        for item in report.get("prices", ())
    )
    expected_reviewed_remaining_price_exception_ids = set(
        reviewed_remaining_price_exceptions
    ) & set(expected_price_targets)
    reviewed_remaining_price_exception_feature_claimed = (
        "reviewed_remaining_price_exception_targets" in report["provider"]
        or "reviewed_remaining_price_exception_inventory_sha256"
        in report["provider"]
        or reviewed_remaining_price_exception_count > 0
    )
    _require(
        _integer(
            report["provider"].get("pinned_external_overlap_targets", 0),
            "pinned_external_overlap_targets",
        )
        == pinned_overlap_count,
        "Archived provider pinned-overlap target count is inconsistent.",
    )
    _require(
        _integer(
            report["provider"].get("reviewed_exact_price_evidence_targets", 0),
            "reviewed_exact_price_evidence_targets",
        )
        == reviewed_price_count
        == len(expected_reviewed_price_ids)
        and (
            not expected_reviewed_price_ids
            or _text(
                report["provider"].get(
                    "reviewed_exact_price_evidence_registry_sha256"
                )
            )
            == TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256
        ),
        "Archived provider reviewed-price target inventory is inconsistent.",
    )
    _require(
        _integer(
            report["provider"].get(
                "reviewed_source_archive_price_only_targets", 0
            ),
            "reviewed_source_archive_price_only_targets",
        )
        == source_archive_price_only_count
        == len(expected_source_archive_price_only_ids)
        and (
            not expected_source_archive_price_only_ids
            or _text(
                report["provider"].get(
                    "reviewed_source_archive_price_only_registry_sha256"
                )
            )
            == TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256
        ),
        "Archived provider frozen WIKI price-only inventory is inconsistent.",
    )
    _require(
        _integer(
            report["provider"].get("reviewed_wiki14_price_only_targets", 0),
            "reviewed_wiki14_price_only_targets",
        )
        == wiki14_price_only_count
        == len(expected_wiki14_price_only_ids)
        and (
            not expected_wiki14_price_only_ids
            or _text(
                report["provider"].get(
                    "reviewed_wiki14_price_only_registry_sha256"
                )
            )
            == TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256
        ),
        "Archived provider frozen WIKI14 price-only inventory is inconsistent.",
    )
    if reviewed_remaining_price_exception_feature_claimed:
        _require(
            _integer(
                report["provider"].get(
                    "reviewed_remaining_price_exception_targets", 0
                ),
                "reviewed_remaining_price_exception_targets",
            )
            == reviewed_remaining_price_exception_count
            == len(expected_reviewed_remaining_price_exception_ids)
            and expected_reviewed_remaining_price_exception_ids
            == set(reviewed_remaining_price_exceptions)
            and _text(
                report["provider"].get(
                    "reviewed_remaining_price_exception_inventory_sha256"
                )
            )
            == TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_INVENTORY_SHA256,
            "Archived provider reviewed remaining-price exception inventory is "
            "inconsistent.",
        )
    summary = report.get("summary")
    _require(isinstance(summary, dict), "Cross-validation summary is missing.")
    for key, expected in computed.items():
        _require(
            _integer(summary.get(key), key) == expected
            and _integer(row[key], key) == expected,
            f"Cross-validation count mismatch for {key}.",
        )
    _require(
        computed["event_mismatch_count"] == 0
        and computed["reviewed_nonterminal_event_count"]
        == computed["nonterminal_event_count"]
        and computed["permanent_exception_mismatch_count"] == 0
        and computed["price_unresolved_count"] == 0
        and computed["price_mismatch_count"] == 0,
        "Cross-validation contains unresolved or mismatched checks.",
    )

    evidence_ids = {
        report_id,
        lifecycle_evidence_report_sha256,
        *permanent_exception_evidence_ids,
    }
    evidence_ids.update(str(item["evidence_sha256"]) for item in report["events"])
    for item in report["prices"]:
        target_id = _text(item.get("target_id"))
        target = expected_price_targets[target_id]
        source_hash = str(item.get("source_sha256", ""))
        if (
            _text(item.get("validation_basis"))
            == REVIEWED_REMAINING_PRICE_EXCEPTION_BASIS
        ):
            validate_reviewed_remaining_price_exception(item)
            _require(
                len(source_hash) == 64
                and len(_text(item.get("cache_wrapper_sha256"))) == 64
                and _safe_yahoo_source_url(item.get("source_url")),
                "Reviewed remaining-price exception lacks exact Yahoo evidence: "
                + target_id,
            )
            evidence_ids.update(
                (source_hash, _text(item.get("cache_wrapper_sha256")).lower())
            )
            for boundary in item.get("identity_boundary_evidence") or ():
                boundary_hash = _text(boundary.get("evidence_sha256")).lower()
                _require(
                    len(boundary_hash) == 64,
                    "Reviewed remaining-price identity boundary evidence is invalid.",
                )
                evidence_ids.add(boundary_hash)
            continue
        if (
            _text(item.get("validation_basis"))
            == REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS
        ):
            spec = source_archive_price_only_specs.get(target_id)
            diagnostic = item.get(
                "reviewed_source_archive_price_only_evidence"
            )
            exception = item.get("exception")
            _require(
                spec is not None
                and item.get("status") == "explicit_exception"
                and item.get(
                    "reviewed_source_archive_price_only_evidence_applied"
                )
                is True
                and source_hash == WIKI_EXTRACT_SHA256
                and _text(item.get("provenance_sha256"))
                == WIKI_PROVENANCE_SHA256
                and _text(item.get("source_url")) == WIKI_DOWNLOAD_URL
                and _text(item.get("expected_source_url")) == WIKI_DOWNLOAD_URL
                and not _text(item.get("cache_wrapper_sha256"))
                and item.get("http_status") is None
                and item.get("response_identity_match") is False
                and _text(
                    item.get(
                        "reviewed_source_archive_price_only_policy_sha256"
                    )
                )
                == source_archive_price_only_spec_sha256(spec)
                and _text(
                    item.get(
                        "reviewed_source_archive_price_only_registry_sha256"
                    )
                )
                == TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256
                and isinstance(diagnostic, dict)
                and diagnostic.get("target_id") == target_id
                and diagnostic.get("security_id") == target["security_id"]
                and diagnostic.get("symbol") == target["symbol"]
                and diagnostic.get("target_provider_symbol")
                == target["provider_symbol"]
                and diagnostic.get("generic_ticker_reuse_allowed") is False
                and diagnostic.get(
                    "yahoo_symbol_only_identity_reuse_allowed"
                )
                is False
                and diagnostic.get("action_factor_status")
                == "incomplete_not_rewritten"
                and isinstance(exception, dict)
                and exception.get("code")
                == "reviewed_frozen_wiki_price_only"
                and exception.get("price_only_arbitration_passed") is True
                and exception.get("action_factor_status")
                == "incomplete_not_rewritten"
                and exception.get(
                    "price_only_pass_must_not_imply_action_factor_pass"
                )
                is True
                and exception.get("generic_ticker_reuse_allowed") is False,
                "Frozen WIKI price-only report/spec binding changed: "
                + target_id,
            )
            evidence_ids.update((WIKI_EXTRACT_SHA256, WIKI_PROVENANCE_SHA256))
            continue
        if _text(item.get("validation_basis")) == REVIEWED_WIKI14_PRICE_ONLY_BASIS:
            spec = wiki14_price_only_specs.get(target_id)
            diagnostic = item.get("reviewed_wiki14_price_only_evidence")
            exception = item.get("exception")
            _require(
                spec is not None
                and item.get("status") == "explicit_exception"
                and item.get("reviewed_wiki14_price_only_evidence_applied") is True
                and source_hash == spec["extract_sha256"]
                and _text(item.get("provenance_sha256"))
                == WIKI14_PROVENANCE_SHA256
                and _text(item.get("source_url")) == WIKI14_DOWNLOAD_URL
                and _text(item.get("expected_source_url")) == WIKI14_DOWNLOAD_URL
                and not _text(item.get("cache_wrapper_sha256"))
                and item.get("http_status") is None
                and item.get("response_identity_match") is False
                and _text(item.get("reviewed_wiki14_price_only_policy_sha256"))
                == wiki14_price_only_spec_sha256(spec)
                and _text(item.get("reviewed_wiki14_price_only_registry_sha256"))
                == TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256
                and isinstance(diagnostic, dict)
                and diagnostic.get("target_id") == target_id
                and diagnostic.get("security_id") == target["security_id"]
                and diagnostic.get("symbol") == target["symbol"]
                and diagnostic.get("target_provider_symbol")
                == target["provider_symbol"]
                and diagnostic.get("extract_sha256") == spec["extract_sha256"]
                and diagnostic.get("provenance_sha256")
                == WIKI14_PROVENANCE_SHA256
                and diagnostic.get("generic_ticker_reuse_allowed") is False
                and diagnostic.get("yahoo_symbol_only_identity_reuse_allowed")
                is False
                and diagnostic.get("action_factor_status")
                == "incomplete_not_rewritten"
                and diagnostic.get("private_internal_only") is True
                and diagnostic.get("redistribution_allowed") is False
                and diagnostic.get("public_publication_allowed") is False
                and isinstance(exception, dict)
                and exception.get("code") == "reviewed_frozen_wiki14_price_only"
                and exception.get("price_only_arbitration_passed") is True
                and exception.get("action_factor_status")
                == "incomplete_not_rewritten"
                and exception.get(
                    "price_only_pass_must_not_imply_action_factor_pass"
                )
                is True
                and exception.get("generic_ticker_reuse_allowed") is False
                and exception.get("private_internal_only") is True
                and exception.get("redistribution_allowed") is False
                and exception.get("public_publication_allowed") is False,
                "Frozen WIKI14 price-only report/spec binding changed: "
                + target_id,
            )
            evidence_ids.update((spec["extract_sha256"], WIKI14_PROVENANCE_SHA256))
            continue
        if _text(item.get("validation_basis")) == PINNED_EXTERNAL_OVERLAP_VALIDATION:
            spec = _pinned_overlap_spec(policy, target)
            _require(spec is not None, "Pinned overlap target is not in policy.")
            primary_hash = _text(item.get("primary_source_sha256")).lower()
            _require(
                source_hash == _text(spec.get("external_source_sha256")).lower()
                and _text(item.get("source_url"))
                == _text(spec.get("external_source_url"))
                and _text(item.get("external_source"))
                == _text(spec.get("external_source"))
                and _text(item.get("primary_source"))
                == _text(spec.get("primary_source"))
                and _text(item.get("primary_source_url"))
                == _text(spec.get("primary_source_url"))
                and len(primary_hash) == 64
                and item.get("upstream_provider_disclosed")
                is spec.get("upstream_provider_disclosed")
                and item.get("independent_provider_claimed")
                is spec.get("independent_provider_claimed")
                and _text(item.get("license")) == _text(spec.get("license"))
                and _text(item.get("license_url"))
                == _text(spec.get("license_url")),
                "Pinned overlap report does not match its exact policy entry.",
            )
            evidence_ids.update((source_hash, primary_hash))
            continue
        _require(
            len(source_hash) == 64,
            "Price check is missing the exact Yahoo response hash.",
        )
        _require(
            len(str(item.get("cache_wrapper_sha256", ""))) == 64,
            "Price check is missing the Yahoo cache wrapper hash.",
        )
        _require(
            _safe_yahoo_source_url(item.get("source_url")),
            "Price check contains an unsafe or unpinned Yahoo source URL.",
        )
        request = _bounded_yahoo_source_request(item.get("source_url"))
        _require(request is not None, "Yahoo bounded source request is invalid.")
        request_start, request_end, period1, period2 = (
            _expected_bounded_yahoo_request(target, prices)
        )
        _require(
            request == (target["provider_symbol"], period1, period2)
            and _text(item.get("source_url"))
            == _canonical_bounded_yahoo_url(
                target["provider_symbol"], period1, period2
            )
            and _date(item.get("request_start_date")) == request_start
            and _date(item.get("request_end_date")) == request_end
            and _integer(item.get("request_period1"), "request_period1") == period1
            and _integer(item.get("request_period2"), "request_period2") == period2
            and item.get("request_period2_is_exclusive") is True
            and _text(item.get("expected_source_url"))
            == _text(item.get("source_url")),
            "Yahoo source URL/report bounds do not match the exact identity interval.",
        )
        provider_symbol = _normalized_report_symbol(item.get("provider_symbol"))
        _require(
            provider_symbol == _normalized_report_symbol(item.get("symbol"))
            and provider_symbol == _source_url_symbol(item.get("source_url")),
            "Yahoo report, identity, and source URL symbols do not match.",
        )
        http_status = _integer(item.get("http_status"), "http_status")
        if item.get("status") == "explicit_exception":
            _require(
                http_status in {200, 400, 404, 410},
                "Yahoo no-data evidence has an unsupported HTTP status.",
            )
        else:
            _require(
                http_status == 200,
                "Yahoo price evidence was not an HTTP 200 response.",
            )
        reviewed_spec = reviewed_price_registry.get(target_id)
        if reviewed_spec is None:
            _require(
                _text(item.get("validation_basis"))
                != REVIEWED_PRICE_EVIDENCE_BASIS
                and item.get("reviewed_price_evidence_applied") is not True,
                "Unregistered target claims reviewed exact price evidence: "
                + target_id,
            )
        else:
            official_event_id = reviewed_spec["official_event_id"]
            official_event = report_event_by_id.get(official_event_id)
            official_binding = not official_event_id or (
                official_event is not None
                and official_event.get("status") == "passed"
                and _text(official_event.get("security_id"))
                == reviewed_spec["security_id"]
                and _date(official_event.get("effective_date"))
                == reviewed_spec["official_effective_date"]
                and _text(official_event.get("evidence_sha256")).lower()
                == reviewed_spec["official_evidence_sha256"]
            )
            _require(
                item.get("status") == "passed"
                and _text(item.get("validation_basis"))
                == REVIEWED_PRICE_EVIDENCE_BASIS
                and item.get("reviewed_price_evidence_applied") is True
                and source_hash == reviewed_spec["source_sha256"]
                and _text(item.get("cache_wrapper_sha256")).lower()
                == reviewed_spec["cache_wrapper_sha256"]
                and _text(item.get("reviewed_price_evidence_sha256"))
                == reviewed_price_evidence_sha256(reviewed_spec)
                and _text(
                    item.get("reviewed_price_evidence_registry_sha256")
                )
                == TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256
                and _text(item.get("reviewed_price_projection_sha256"))
                == reviewed_spec["expected_projection_sha256"]
                and _text(item.get("reviewed_price_evidence_case_code"))
                == reviewed_spec["case_code"]
                and _text(item.get("reviewed_price_limitation"))
                == reviewed_spec["limitation"]
                and item.get("reviewed_official_event_binding_passed") is True
                and _text(item.get("reviewed_official_event_id"))
                == official_event_id
                and _text(
                    item.get("reviewed_official_evidence_sha256")
                ).lower()
                == reviewed_spec["official_evidence_sha256"]
                and _date(item.get("reviewed_official_effective_date"))
                == reviewed_spec["official_effective_date"]
                and official_binding,
                "Reviewed exact price report/spec/event binding changed: "
                + target_id,
            )
            if reviewed_spec["official_evidence_sha256"]:
                evidence_ids.add(reviewed_spec["official_evidence_sha256"])
        evidence_ids.update(
            (source_hash, _text(item.get("cache_wrapper_sha256")).lower())
        )
        for boundary in item.get("identity_boundary_evidence") or ():
            boundary_hash = str(boundary.get("evidence_sha256", ""))
            _require(
                len(boundary_hash) == 64,
                "Identity boundary evidence lacks an exact source hash.",
            )
            evidence_ids.add(boundary_hash)
        if item.get("status") == "explicit_exception":
            official_hash = str(item["exception"].get("official_evidence_sha256", ""))
            _require(len(official_hash) == 64, "Price exception lacks official evidence hash.")
            evidence_ids.add(official_hash)
    archive_ids = set(archive["archive_id"].astype(str)) | set(
        archive["source_hash"].astype(str)
    )
    missing_evidence = sorted(evidence_ids - archive_ids)
    _require(
        not missing_evidence,
        "Cross-validation evidence is missing from source_archive: "
        + ", ".join(missing_evidence),
    )
    for evidence_id in sorted(evidence_ids):
        _archive_payload(repository, archive, evidence_id)

    reviewed_successor_chains = reviewed_no_data_successor_chains(policy["prices"])
    recomputed_source_archive_price_only: dict[str, dict[str, Any]] = {}
    factors: pd.DataFrame | None = None
    if expected_source_archive_price_only_ids:
        _require(
            expected_source_archive_price_only_ids
            == set(TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS),
            "Frozen WIKI price-only gate requires the complete BBBY/BBT pair.",
        )
        factors = repository.read_frame(
            "adjustment_factors", expected_versions["adjustment_factors"]
        )
        source_archive_targets = {
            target_id: {
                "target_id": target_id,
                **expected_price_targets[target_id],
                "terminal_event_id": _text(
                    report_price_targets[target_id].get("terminal_event_id")
                ),
            }
            for target_id in sorted(expected_source_archive_price_only_ids)
        }
        recomputed_source_archive_price_only = (
            verify_source_archive_price_only_evidence(
                repository,
                archive,
                prices=prices,
                factors=factors,
                master=master,
                history=history,
                actions=actions,
                targets=source_archive_targets,
                prices_policy=policy["prices"],
                release_warnings=release.warnings,
            )
        )
    recomputed_wiki14_price_only: dict[str, dict[str, Any]] = {}
    if expected_wiki14_price_only_ids:
        _require(
            expected_wiki14_price_only_ids
            == set(TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS),
            "Frozen WIKI14 price-only gate requires the complete 14-target set.",
        )
        if factors is None:
            factors = repository.read_frame(
                "adjustment_factors", expected_versions["adjustment_factors"]
            )
        wiki14_targets = {
            target_id: {
                "target_id": target_id,
                **expected_price_targets[target_id],
                "terminal_event_id": _text(
                    report_price_targets[target_id].get("terminal_event_id")
                ),
            }
            for target_id in sorted(expected_wiki14_price_only_ids)
        }
        recomputed_wiki14_price_only = verify_wiki14_price_only_evidence(
            repository,
            archive,
            prices=prices,
            factors=factors,
            master=master,
            history=history,
            actions=actions,
            targets=wiki14_targets,
            prices_policy=policy["prices"],
            release_warnings=release.warnings,
        )
    for price_item in report["prices"]:
        source_hash = str(price_item["source_sha256"])
        if (
            _text(price_item.get("validation_basis"))
            == REVIEWED_REMAINING_PRICE_EXCEPTION_BASIS
        ):
            validate_reviewed_remaining_price_exception(price_item)
            _yahoo_archive_payload(repository, archive, price_item)
            continue
        if (
            _text(price_item.get("validation_basis"))
            == REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS
        ):
            target_id = _text(price_item.get("target_id"))
            diagnostic = recomputed_source_archive_price_only.get(target_id)
            _require(
                diagnostic is not None
                and price_item.get(
                    "reviewed_source_archive_price_only_evidence"
                )
                == diagnostic
                and _text(
                    price_item.get(
                        "reviewed_source_archive_price_only_projection_sha256"
                    )
                )
                == diagnostic["projection_sha256"]
                and _integer(
                    price_item.get("overlap_session_count"),
                    "overlap_session_count",
                )
                == diagnostic["overlap_session_count"]
                and _text(price_item.get("action_factor_status"))
                == "incomplete_not_rewritten"
                and price_item.get("corporate_actions_validated") is False
                and price_item.get("adjustment_factors_validated") is False
                and price_item.get("generic_ticker_reuse_allowed") is False,
                "Frozen WIKI price-only projection is not reproducible: "
                + target_id,
            )
            continue
        if _text(price_item.get("validation_basis")) == REVIEWED_WIKI14_PRICE_ONLY_BASIS:
            target_id = _text(price_item.get("target_id"))
            diagnostic = recomputed_wiki14_price_only.get(target_id)
            _require(
                diagnostic is not None
                and price_item.get("reviewed_wiki14_price_only_evidence")
                == diagnostic
                and _text(
                    price_item.get(
                        "reviewed_wiki14_price_only_projection_sha256"
                    )
                )
                == diagnostic["projection_sha256"]
                and _integer(
                    price_item.get("overlap_session_count"),
                    "overlap_session_count",
                )
                == diagnostic["overlap_session_count"]
                and _text(price_item.get("action_factor_status"))
                == "incomplete_not_rewritten"
                and price_item.get("corporate_actions_validated") is False
                and price_item.get("adjustment_factors_validated") is False
                and price_item.get("generic_ticker_reuse_allowed") is False
                and price_item.get("private_internal_only") is True
                and price_item.get("redistribution_allowed") is False
                and price_item.get("public_publication_allowed") is False,
                "Frozen WIKI14 price-only projection is not reproducible: "
                + target_id,
            )
            continue
        response_url = str(price_item.get("source_url", "")).strip()
        if (
            _text(price_item.get("validation_basis"))
            == PINNED_EXTERNAL_OVERLAP_VALIDATION
        ):
            response_payload = _archive_payload(repository, archive, source_hash)
        else:
            response_payload = _yahoo_archive_payload(
                repository,
                archive,
                price_item,
            )
        if (
            price_item.get("status") == "passed"
            and _text(price_item.get("validation_basis"))
            == PINNED_EXTERNAL_OVERLAP_VALIDATION
        ):
            target = expected_price_targets[str(price_item["target_id"])]
            spec = _pinned_overlap_spec(policy, target)
            _require(spec is not None, "Pinned overlap target is not in policy.")
            internal_rows = _all_internal_target_rows(prices, target)
            primary_hashes = {
                _text(value).lower()
                for value in internal_rows["source_hash"]
                if _text(value)
            }
            primary_hash = _text(price_item.get("primary_source_sha256")).lower()
            _require(
                set(internal_rows["source"].astype(str))
                == {_text(spec.get("primary_source"))}
                and set(internal_rows["source_url"].astype(str))
                == {_text(spec.get("primary_source_url"))}
                and primary_hashes == {primary_hash},
                "Pinned overlap internal Yahoo primary provenance changed.",
            )
            primary_payload = _archive_payload(repository, archive, primary_hash)
            try:
                parsed_primary = parse_yahoo_chart_json(
                    primary_payload, _text(price_item.get("provider_symbol"))
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Archived pinned Yahoo primary failed strict validation: "
                    + str(exc)
                ) from exc
            primary_bars = parsed_primary.bars.copy()
            primary_bars["_session"] = pd.to_datetime(
                primary_bars["session"], errors="coerce"
            ).dt.normalize()
            primary_bars = primary_bars.loc[
                primary_bars["_session"].ge(pd.Timestamp(target["active_from"]))
                & primary_bars["_session"].le(pd.Timestamp(target["active_to"]))
            ].sort_values("_session", kind="stable")
            _require(
                parsed_primary.currency == "USD"
                and tuple(primary_bars["_session"])
                == tuple(internal_rows["_session"]),
                "Archived Yahoo primary sessions/currency differ from stored rows.",
            )
            for column in ("open", "high", "low", "close", "volume"):
                stored = pd.to_numeric(
                    internal_rows[column], errors="coerce"
                ).reset_index(drop=True)
                archived = pd.to_numeric(
                    primary_bars[column], errors="coerce"
                ).reset_index(drop=True)
                _require(
                    not bool(stored.isna().any())
                    and stored.equals(archived),
                    f"Stored pinned Yahoo primary {column} differs from archived bytes.",
                )
            external_rows = _parse_pinned_external_payload(response_payload, spec)
            metrics = _recompute_pinned_overlap(internal_rows, external_rows, spec)
            integer_fields = (
                "overlap_session_count",
                "internal_history_session_count",
                "uncrosschecked_tail_sessions",
            )
            text_fields = (
                "internal_history_start",
                "internal_history_end",
                "external_overlap_start",
                "external_overlap_end",
                "uncrosschecked_tail_start",
                "uncrosschecked_tail_end",
            )
            _require(
                all(
                    _integer(price_item.get(key), key) == int(metrics[key])
                    for key in integer_fields
                )
                and all(_text(price_item.get(key)) == str(metrics[key]) for key in text_fields)
                and math.isclose(
                    float(price_item.get("median_primary_to_external_close_scale")),
                    float(metrics["median_primary_to_external_close_scale"]),
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(price_item.get("return_correlation")),
                    float(metrics["return_correlation"]),
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(price_item.get("p99_scaled_close_error")),
                    float(metrics["p99_scaled_close_error"]),
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(price_item.get("external_overlap_ratio")),
                    len(external_rows) / len(internal_rows),
                    abs_tol=1e-12,
                ),
                "Pinned external overlap report is not reproducible.",
            )
            primary_archive = archive.loc[
                archive["archive_id"].map(_text).eq(primary_hash)
            ]
            external_archive = archive.loc[
                archive["archive_id"].map(_text).eq(source_hash)
            ]
            _require(
                len(primary_archive) == 1
                and _text(primary_archive.iloc[0].get("source_hash")).lower()
                == primary_hash
                and _text(primary_archive.iloc[0].get("source_url"))
                == _text(spec.get("primary_source_url"))
                and len(external_archive) == 1
                and _text(external_archive.iloc[0].get("source_hash")).lower()
                == source_hash
                and _text(external_archive.iloc[0].get("source_url"))
                == _text(spec.get("external_source_url")),
                "Pinned overlap URL/hash archive provenance mismatch.",
            )
            continue
        reviewed_spec = reviewed_price_registry.get(
            _text(price_item.get("target_id"))
        )
        if reviewed_spec is not None:
            _require(
                price_item.get("status") == "passed"
                and _text(price_item.get("validation_basis"))
                == REVIEWED_PRICE_EVIDENCE_BASIS,
                "Code-pinned reviewed price target did not use its exact path.",
            )
            target = expected_price_targets[_text(price_item.get("target_id"))]
            independent_prices = prices.loc[
                ~independent_provider_source_mask(prices)
            ].copy()
            internal_rows = _all_internal_target_rows(
                independent_prices, target
            ).drop(columns=["_session"])
            split_dates = [
                _date(value)
                for value in actions.loc[
                    actions["security_id"].map(_text).eq(target["security_id"])
                    & actions["action_type"]
                    .map(_text)
                    .str.lower()
                    .isin({"split", "capital_reduction", "stock_dividend"}),
                    "effective_date",
                ]
                if _date(value)
            ]
            provider_rows, projection = build_reviewed_price_projection(
                content=response_payload,
                spec=reviewed_spec,
                target={
                    "target_id": _text(price_item.get("target_id")),
                    "security_id": target["security_id"],
                    "symbol": target["symbol"],
                    "active_from": target["active_from"],
                    "active_to": target["active_to"],
                },
                internal_prices=internal_rows,
                split_dates=split_dates,
                policy_prices=policy["prices"],
            )
            projection_sha256 = verify_reviewed_price_projection(
                projection, reviewed_spec
            )
            provider_start = (
                provider_rows["session"].min().date().isoformat()
                if not provider_rows.empty
                else ""
            )
            provider_end = (
                provider_rows["session"].max().date().isoformat()
                if not provider_rows.empty
                else ""
            )
            _require(
                _text(price_item.get("reviewed_price_projection_sha256"))
                == projection_sha256
                and price_item.get("reviewed_price_mismatch_rows")
                == projection["mismatch_rows"]
                and price_item.get("reviewed_triple_supertrend_signal")
                == projection["signal"]
                and price_item.get("reviewed_provider_metadata")
                == projection["metadata"]
                and _text(
                    price_item.get("reviewed_internal_ohlcv_sha256")
                )
                == projection["internal_ohlcv_sha256"]
                and _text(
                    price_item.get("reviewed_provider_ohlcv_sha256")
                )
                == projection["provider_ohlcv_sha256"]
                and _text(
                    price_item.get("reviewed_overlap_ohlcv_sha256")
                )
                == projection["overlap_ohlcv_sha256"]
                and _integer(
                    price_item.get("reviewed_all_null_row_count"),
                    "reviewed_all_null_row_count",
                )
                == projection["all_null_row_count"]
                and _text(
                    price_item.get("reviewed_all_null_sessions_sha256")
                )
                == projection["all_null_sessions_sha256"]
                and _integer(
                    price_item.get("overlap_session_count"),
                    "overlap_session_count",
                )
                == projection["overlap_row_count"]
                and _integer(
                    price_item.get("provider_history_session_count"),
                    "provider_history_session_count",
                )
                == len(provider_rows)
                and _date(price_item.get("provider_history_start"))
                == provider_start
                and _date(price_item.get("provider_history_end")) == provider_end
                and math.isclose(
                    float(price_item.get("session_coverage_ratio")),
                    float(projection["coverage_ratio"]),
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(
                        price_item.get(
                            "provider_internal_session_coverage_ratio"
                        )
                    ),
                    float(projection["coverage_ratio"]),
                    abs_tol=1e-12,
                ),
                "Reviewed exact Yahoo price projection is not reproducible: "
                + _text(price_item.get("target_id")),
            )
            continue
        if price_item.get("status") == "passed":
            try:
                parsed_price = parse_yahoo_chart_json(
                    response_payload, str(price_item["provider_symbol"])
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Archived Yahoo response failed strict validation: " + str(exc)
                ) from exc
            _require(
                parsed_price.currency == "USD"
                and parsed_price.adjustment_basis == "raw_quote_ohlcv",
                "Archived Yahoo response is not USD raw quote OHLCV.",
            )
            _require(
                not parsed_price.bars.empty,
                "Passed Yahoo price check has no raw quote bars.",
            )
            provider_count = _integer(
                price_item.get("provider_history_session_count"),
                "provider_history_session_count",
            )
            _require(
                provider_count == len(parsed_price.bars)
                and str(price_item.get("provider_history_start", ""))
                == parsed_price.bars["session"].min().date().isoformat()
                and str(price_item.get("provider_history_end", ""))
                == parsed_price.bars["session"].max().date().isoformat(),
                "Yahoo report history bounds do not match the archived response.",
            )
            target = expected_price_targets[str(price_item["target_id"])]
            request_start, request_end, period1, period2 = (
                _expected_bounded_yahoo_request(target, prices)
            )
            expected_xnys = pd.DatetimeIndex(
                xcals.get_calendar("XNYS").sessions_in_range(
                    request_start, request_end
                )
            ).tz_localize(None).normalize()
            provider_index = pd.DatetimeIndex(
                pd.to_datetime(parsed_price.bars["session"], errors="coerce")
            ).normalize()
            expected_xnys_set = set(expected_xnys)
            provider_set = set(provider_index)
            unexpected_provider = provider_set - expected_xnys_set
            missing_provider = sorted(expected_xnys_set - provider_set)
            provider_request_coverage = (
                len(provider_set & expected_xnys_set) / len(expected_xnys_set)
                if expected_xnys_set
                else 0.0
            )
            minimum_coverage = float(
                policy["prices"].get("minimum_session_coverage_ratio")
            )
            minimum_request_sessions = min(
                _integer(
                    policy["prices"].get("minimum_overlap_sessions"),
                    "minimum_overlap_sessions",
                ),
                len(expected_xnys_set),
            )
            _require(
                not provider_index.has_duplicates
                and not unexpected_provider
                and len(provider_set & expected_xnys_set)
                >= minimum_request_sessions
                and provider_request_coverage >= minimum_coverage
                and _date(price_item.get("request_start_date")) == request_start
                and _date(price_item.get("request_end_date")) == request_end
                and _integer(price_item.get("request_period1"), "request_period1")
                == period1
                and _integer(price_item.get("request_period2"), "request_period2")
                == period2
                and price_item.get("request_period2_is_exclusive") is True
                and _integer(
                    price_item.get("request_xnys_session_count"),
                    "request_xnys_session_count",
                )
                == len(expected_xnys_set)
                and _integer(
                    price_item.get("provider_xnys_session_count"),
                    "provider_xnys_session_count",
                )
                == len(provider_set & expected_xnys_set)
                and _integer(
                    price_item.get("provider_unexpected_session_count"),
                    "provider_unexpected_session_count",
                )
                == 0
                and list(price_item.get("provider_unexpected_sessions") or []) == []
                and _integer(
                    price_item.get("provider_missing_xnys_session_count"),
                    "provider_missing_xnys_session_count",
                )
                == len(missing_provider)
                and list(price_item.get("provider_missing_xnys_sessions") or [])
                == [value.date().isoformat() for value in missing_provider]
                and _integer(
                    price_item.get("provider_outside_request_session_count"),
                    "provider_outside_request_session_count",
                )
                == 0
                and list(price_item.get("provider_outside_request_sessions") or [])
                == []
                and math.isclose(
                    float(
                        price_item.get("provider_request_xnys_coverage_ratio")
                    ),
                    provider_request_coverage,
                    abs_tol=1e-12,
                )
                and price_item.get("provider_request_inventory_passed") is True,
                "Yahoo report failed bounded exact XNYS inventory validation.",
            )
            internal_sessions = _internal_target_sessions(prices, target)
            provider_sessions = _provider_target_sessions(parsed_price.bars, target)
            _require(
                not internal_sessions.empty,
                "Passed price target has no provider-independent internal history: "
                + str(price_item["target_id"]),
            )
            overlap = internal_sessions.intersection(provider_sessions)
            overlap_count = len(overlap)
            internal_count = len(internal_sessions)
            coverage_ratio = overlap_count / internal_count
            _require(
                _integer(
                    price_item.get("eodhd_history_session_count"),
                    "eodhd_history_session_count",
                )
                == internal_count
                and _date(price_item.get("eodhd_history_start"))
                == internal_sessions.min().date().isoformat()
                and _date(price_item.get("eodhd_history_end"))
                == internal_sessions.max().date().isoformat()
                and _integer(
                    price_item.get("overlap_session_count"),
                    "overlap_session_count",
                )
                == overlap_count,
                "Price report does not match full EODHD identity-interval history: "
                + str(price_item["target_id"]),
            )
            try:
                reported_full_ratio = float(
                    price_item.get("eodhd_full_history_overlap_ratio")
                )
                reported_coverage_ratio = float(
                    price_item.get("session_coverage_ratio")
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Price report full-history coverage ratios are invalid."
                ) from exc
            minimum_overlap = min(
                _integer(
                    policy["prices"].get("minimum_overlap_sessions"),
                    "minimum_overlap_sessions",
                ),
                internal_count,
            )
            _require(
                math.isclose(reported_full_ratio, coverage_ratio, abs_tol=1e-12)
                and math.isclose(reported_coverage_ratio, coverage_ratio, abs_tol=1e-12)
                and math.isclose(
                    float(
                        price_item.get("provider_internal_session_coverage_ratio")
                    ),
                    coverage_ratio,
                    abs_tol=1e-12,
                )
                and overlap_count >= minimum_overlap
                and coverage_ratio >= minimum_coverage,
                "Price target lacks full EODHD history coverage: "
                + str(price_item["target_id"]),
            )
        elif price_item.get("status") == "explicit_exception":
            try:
                no_data = parse_yahoo_chart_no_data_evidence(
                    response_payload,
                    str(price_item["provider_symbol"]),
                    http_status=_integer(price_item.get("http_status"), "http_status"),
                    request_period1=_integer(
                        price_item.get("request_period1"), "request_period1"
                    ),
                    request_period2=_integer(
                        price_item.get("request_period2"), "request_period2"
                    ),
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Archived Yahoo no-data response failed strict validation: "
                    + str(exc)
                ) from exc
            _require(
                no_data.kind == _text(price_item.get("no_data_evidence_kind")),
                "Archived Yahoo no-data kind does not match the report.",
            )
            target = expected_price_targets[str(price_item["target_id"])]
            internal_sessions = _internal_target_sessions(prices, target)
            terminal_passed, terminal_detail = _terminal_window_detail(
                internal_sessions,
                _integer(
                    policy["prices"].get("terminal_calendar_window_sessions"),
                    "terminal_calendar_window_sessions",
                ),
            )
            exception = price_item["exception"]
            validation_basis = _text(price_item.get("validation_basis"))
            _require(
                exception.get("terminal_calendar") == terminal_detail
                and exception.get("terminal_calendar_complete")
                is terminal_passed,
                "Terminal-provider exception calendar is not reproducible.",
            )
            if validation_basis == REVIEWED_PERMANENT_EXCEPTION_NO_DATA_BASIS:
                expected_binding = permanent_exception_no_data_binding(
                    {
                        "target_id": _text(price_item.get("target_id")),
                        **target,
                        "symbol": _text(price_item.get("symbol")),
                        "terminal_event_id": _text(
                            price_item.get("terminal_event_id")
                        ),
                        "successor_security_id": _text(
                            price_item.get("successor_security_id")
                        ),
                    },
                    _date(terminal_detail.get("terminal_session")),
                    report_permanent,
                )
                _require(
                    expected_binding is not None
                    and all(
                        exception.get(key) == value
                        for key, value in expected_binding.items()
                    )
                    and exception.get("official_event_verified") is False
                    and exception.get("identity_event_match") is False
                    and exception.get("successor_security_id") == ""
                    and exception.get("successor_requirement_passed") is True
                    and exception.get("response_identity_match") is True
                    and exception.get("no_data_evidence_validated") is True,
                    "Permanent lifecycle no-data exception is not exactly bound: "
                    + _text(price_item.get("target_id")),
                )
                successor_binding = successor_price_check_binding(
                    report["prices"],
                    {},
                    source_target_id=_text(price_item.get("target_id")),
                    expected_successor_security_id="",
                    reviewed_successor_chains=reviewed_successor_chains,
                    event_checks=report["events"],
                )
                _require(
                    exception.get("successor_validation") == successor_binding
                    and successor_binding["passed"] is True,
                    "Permanent lifecycle no-data successor state changed.",
                )
                continue
            if validation_basis == REVIEWED_NO_DATA_UNSUPPORTED_PATH_BASIS:
                official_event_id = _text(exception.get("official_event_id"))
                event = report_event_by_id.get(official_event_id)
                _require(
                    event is not None,
                    "Reviewed unsupported-path no-data event is missing.",
                )
                security_rows = prices.loc[
                    prices["security_id"].map(_text).eq(target["security_id"])
                ].copy()
                security_rows = security_rows.loc[
                    ~independent_provider_source_mask(security_rows)
                ].copy()
                parsed_sessions = pd.to_datetime(
                    security_rows["session"], errors="coerce"
                ).dt.normalize()
                if target.get("active_from"):
                    security_rows = security_rows.loc[
                        parsed_sessions.ge(pd.Timestamp(target["active_from"]))
                    ].copy()
                    parsed_sessions = parsed_sessions.loc[security_rows.index]
                if target.get("active_to"):
                    security_rows = security_rows.loc[
                        parsed_sessions.le(pd.Timestamp(target["active_to"]))
                    ].copy()
                expected_binding = unsupported_path_no_data_binding(
                    {
                        "target_id": _text(price_item.get("target_id")),
                        **target,
                        "symbol": _text(price_item.get("symbol")),
                        "terminal_event_id": _text(
                            price_item.get("terminal_event_id")
                        ),
                        "successor_security_id": _text(
                            price_item.get("successor_security_id")
                        ),
                    },
                    _date(terminal_detail.get("terminal_session")),
                    event,
                    security_rows,
                    policy["prices"],
                    source_sha256=_text(price_item.get("source_sha256")),
                    cache_wrapper_sha256=_text(
                        price_item.get("cache_wrapper_sha256")
                    ),
                )
                _require(
                    expected_binding is not None
                    and all(
                        exception.get(key) == value
                        for key, value in expected_binding.items()
                    )
                    and exception.get("successor_security_id") == ""
                    and exception.get("successor_requirement_passed") is True
                    and exception.get("response_identity_match") is True
                    and exception.get("no_data_evidence_validated") is True,
                    "Reviewed unsupported-path no-data exception is not exactly "
                    "bound.",
                )
                resolution = applied_groups.get(official_event_id)
                _require(
                    resolution is not None
                    and len(resolution) == 1
                    and _text(resolution.iloc[0].get("candidate_id"))
                    == _text(exception.get("candidate_id"))
                    and not _text(
                        resolution.iloc[0].get("successor_security_id")
                    ),
                    "Reviewed unsupported-path no-data resolution changed.",
                )
                successor_binding = successor_price_check_binding(
                    report["prices"],
                    event,
                    source_target_id=_text(price_item.get("target_id")),
                    expected_successor_security_id="",
                    reviewed_successor_chains=reviewed_successor_chains,
                    event_checks=report["events"],
                )
                _require(
                    exception.get("successor_validation") == successor_binding
                    and successor_binding["passed"] is True,
                    "Reviewed unsupported-path no-data successor state changed.",
                )
                continue
            official_event_id = _text(exception.get("official_event_id"))
            event = report_event_by_id.get(official_event_id)
            official_action_type = _text(
                exception.get("official_action_type")
            ).lower()
            _require(
                terminal_passed
                and event is not None
                and _text(price_item.get("terminal_event_id")) == official_event_id
                and _text(event.get("security_id")) == target["security_id"]
                and official_action_type in YAHOO_NO_DATA_TERMINAL_ACTION_TYPES
                and _text(event.get("action_type")).lower()
                == official_action_type
                and _text(exception.get("official_evidence_sha256")).lower()
                == _text(event.get("evidence_sha256")).lower()
                and _terminal_event_date_matches(
                    target["active_to"],
                    terminal_detail["terminal_session"],
                    _date(event.get("effective_date")),
                    identity_date_basis=_text(
                        exception.get("identity_date_basis")
                    ),
                    derived_identity_active_to=_date(
                        exception.get("derived_identity_active_to")
                    ),
                    terminal_calendar_complete=terminal_passed,
                ),
                "Terminal-provider exception is not bound to the exact official "
                "terminal identity/date.",
            )
            reviewed_nonterminal_binding = (
                reviewed_nonterminal_same_sid_no_data_binding(
                    {
                        "target_id": _text(price_item.get("target_id")),
                        **target,
                        "symbol": _text(price_item.get("symbol")),
                        "provider_symbol": _text(
                            price_item.get("provider_symbol")
                        ),
                        "terminal_event_id": _text(
                            price_item.get("terminal_event_id")
                        ),
                        "successor_security_id": _text(
                            price_item.get("successor_security_id")
                        ),
                    },
                    event,
                    reviewed_extractions,
                )
            )
            validation_basis = _text(price_item.get("validation_basis"))
            if validation_basis == REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_BASIS:
                _require(
                    reviewed_nonterminal_binding is not None
                    and price_item.get(
                        "reviewed_nonterminal_same_sid_no_data_applied"
                    )
                    is True
                    and exception.get(
                        "reviewed_nonterminal_same_sid_binding"
                    )
                    == reviewed_nonterminal_binding
                    and applied_groups.get(official_event_id) is None,
                    "Reviewed nonterminal same-SID no-data transition is not "
                    "exactly bound or gained a terminal resolution.",
                )
                expected_successor = _text(
                    reviewed_nonterminal_binding["security_id"]
                )
            else:
                _require(
                    reviewed_nonterminal_binding is None,
                    "Reviewed nonterminal same-SID no-data transition lost its "
                    "dedicated validation basis.",
                )
                resolution = applied_groups.get(official_event_id)
                _require(
                    resolution is not None and len(resolution) == 1,
                    "Terminal-provider exception lacks one applied resolution.",
                )
                expected_successor = _text(
                    resolution.iloc[0].get("successor_security_id")
                )
            reported_successor = _text(exception.get("successor_security_id"))
            successor_binding = successor_price_check_binding(
                report["prices"],
                event,
                source_target_id=_text(price_item.get("target_id")),
                expected_successor_security_id=expected_successor,
                reviewed_successor_chains=reviewed_successor_chains,
                event_checks=report["events"],
            )
            _require(
                reported_successor == expected_successor
                and exception.get("successor_validation") == successor_binding
                and successor_binding["passed"] is True,
                "Terminal-provider exception successor coverage is invalid.",
            )
            if reviewed_nonterminal_binding is not None:
                _require(
                    _text(successor_binding.get("target_id"))
                    == reviewed_nonterminal_binding["successor_target_id"],
                    "Reviewed nonterminal same-SID successor target changed.",
                )
    boundary_specs = {
        (
            str(item.get("symbol", "")).strip().upper(),
            str(item.get("boundary", "")).strip(),
            str(item.get("date", "")).strip(),
            str(item.get("source_url", "")).strip(),
            str(item.get("source_kind", "")).strip(),
        )
        for item in (policy.get("identity_boundaries") or ())
        if isinstance(item, dict)
    }
    for price_item in report["prices"]:
        for boundary in price_item.get("identity_boundary_evidence") or ():
            boundary_name = str(boundary.get("boundary", "")).strip()
            expected_date = (
                str(price_item.get("identity_active_from", "")).strip()
                if boundary_name == "active_from"
                else str(price_item.get("identity_active_to", "")).strip()
                if boundary_name == "active_to"
                else ""
            )
            _require(
                bool(expected_date)
                and str(boundary.get("date", "")).strip() == expected_date,
                "Identity boundary evidence date does not match the target identity.",
            )
            parsed_url = urlparse(str(boundary.get("source_url", "")).strip())
            boundary_host = (parsed_url.hostname or "").lower()
            _require(
                parsed_url.scheme == "https"
                and any(
                    boundary_host == host or boundary_host.endswith("." + host)
                    for host in ("sec.gov", "nasdaqtrader.com", "nyse.com")
                ),
                "Identity boundary evidence is not an approved official URL.",
            )
            boundary_key = (
                str(price_item.get("symbol", "")).strip().upper(),
                boundary_name,
                str(boundary.get("date", "")).strip(),
                str(boundary.get("source_url", "")).strip(),
                str(boundary.get("source_kind", "")).strip(),
            )
            _require(
                boundary_key in boundary_specs,
                "Identity boundary evidence is not an exact policy entry.",
            )
            boundary_hash = str(boundary.get("evidence_sha256", ""))
            archive_row = archive.loc[
                archive["archive_id"].astype(str).eq(boundary_hash)
            ].iloc[0]
            _require(
                str(archive_row.get("source_url", "")).strip()
                == str(boundary.get("source_url", "")).strip(),
                "Identity boundary URL/hash provenance mismatch.",
            )

    manifest = repository.manifest_for_version(CROSS_VALIDATION_DATASET, version)
    expected_metadata = {
        "report_id": report_id,
        "status": "passed",
        "provider": INDEPENDENT_PRICE_PROVIDER,
        "policy_sha256": policy_sha256,
        "lifecycle_evidence_report_sha256": lifecycle_evidence_report_sha256,
        "validated_versions": expected_versions,
        "input_hashes": expected_input_hashes,
        **computed,
    }
    for key, expected in expected_metadata.items():
        _require(
            manifest.metadata.get(key) == expected,
            f"Cross-validation manifest metadata mismatch for {key}.",
        )
    return {
        **computed,
        "report_id": report_id,
        "policy_sha256": policy_sha256,
        "provider": INDEPENDENT_PRICE_PROVIDER,
        "lifecycle_evidence_report_sha256": lifecycle_evidence_report_sha256,
        "validated_versions": expected_versions,
        "evidence_artifact_count": len(evidence_ids),
        "reviewed_index_identity_gap_exceptions": [
            dict(item) for item in reviewed_snapshot_identity_gaps
        ],
    }
