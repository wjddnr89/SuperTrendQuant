#!/usr/bin/env python3
"""Plan six exact US terminal-price tail repairs without external access.

The reviewed cohort is deliberately code-pinned rather than policy-driven:

* FLIR, QEP, NLSN, and HBI have provider rows after an official terminal
  trading boundary;
* KORS and NLOK have old-symbol rows after the official new-symbol market
  date, with the same sessions present in the archived successor response.

The default command is a read-only plan.  ``--apply`` exists for a later,
explicitly authorised local commit and uses a writer lock, compare-and-swap
pointers, a transaction journal, and rollback.  There is no network, EODHD,
Yahoo, R2, or other remote code path in this module.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import html
import json
import math
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.lifecycle_coverage import (
    lifecycle_candidate_id,
    lifecycle_candidate_set_sha256,
    lifecycle_resolution_set_sha256,
)
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.terminal_readiness import (
    audit_terminal_transitions,
)
from supertrend_quant.market_store.validation import (
    index_member_identity_gap_fingerprint,
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
WRITE_DATASETS = (
    "corporate_actions",
    "daily_price_raw",
    "adjustment_factors",
    "lifecycle_resolutions",
    "security_master",
    "symbol_history",
)
REQUIRED_DATASETS = (
    *WRITE_DATASETS,
    "source_archive",
    "index_membership_events",
    "index_constituent_anchors",
)
OPERATION = "repair_us_short_terminal_price_tails"
TRANSACTION_DIR = "transactions/us-short-terminal-price-tails"
RECOVERY_DIR = "recovery/us-short-terminal-price-tails"
REPAIR_REVIEWED_AT = "2026-07-19T00:00:00Z"
REPAIRED_IDENTITY_SOURCE = "official_short_terminal_boundary_repair"
REPAIRED_RESOLUTION_SOURCE = "short_terminal_boundary_repair"
REPAIRED_REVIEWER = "short_terminal_boundary_repair_v1"
OLD_IDENTITY_SOURCE = "eodhd_exchange_symbols"
OLD_IDENTITY_URL = "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
OLD_IDENTITY_HASH = (
    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
)
OLD_IDENTITY_RETRIEVED_AT = "2026-07-16T15:56:01.033938Z"
OLD_RESOLUTION_SOURCE = "lifecycle_finalizer"
OLD_RESOLUTION_REVIEWER = "us_lifecycle_finalizer_v1"
OLD_RESOLUTION_REVIEWED_AT = "2026-07-18T00:00:00Z"
ACTION_SOURCE = "sec_edgar+stored_price_crosscheck"
ACTION_SOURCE_KIND = "official_crosscheck"
ARCHIVE_SESSION = "2026-07-15"
EXPECTED_SNAPSHOT_IDENTITY_GAP = {
    "code": "index_member_missing_active_symbol",
    "row_count": 1,
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
}


@dataclass(frozen=True)
class ExactTerminalCase:
    symbol: str
    security_id: str
    exchange: str
    action_type: str
    old_active_to: str
    last_real_session: str
    legal_completion_date: str
    market_transition_session: str
    removed_sessions: tuple[str, ...]
    old_event_id: str
    new_event_id: str
    old_candidate_id: str
    new_candidate_id: str
    announcement_date: str
    cash_amount: float | None
    ratio: float | None
    successor_symbol: str
    successor_security_id: str
    official_source_url: str
    official_source_hash: str
    official_source_bytes: int
    official_retrieved_at: str
    filing_acceptance_datetime: str
    official_patterns: tuple[str, ...]
    raw_source_url: str
    raw_source_hash: str
    raw_source_bytes: int
    raw_source_rows: int
    raw_retrieved_at: str
    raw_selected_sessions: tuple[str, ...]
    raw_selected_sha256: str
    successor_raw_source_url: str = ""
    successor_raw_source_hash: str = ""
    successor_raw_source_bytes: int = 0
    successor_raw_source_rows: int = 0
    successor_raw_retrieved_at: str = ""
    successor_selected_sessions: tuple[str, ...] = ()
    successor_selected_sha256: str = ""


CASES: tuple[ExactTerminalCase, ...] = (
    ExactTerminalCase(
        symbol="FLIR",
        security_id="US:EODHD:0c47238f-bf19-5faa-a3ae-25a34ef3d3f5",
        exchange="NASDAQ",
        action_type="stock_merger",
        old_active_to="2021-05-17",
        last_real_session="2021-05-13",
        legal_completion_date="2021-05-14",
        market_transition_session="2021-05-14",
        removed_sessions=("2021-05-14", "2021-05-17"),
        old_event_id="cff77a9d1a8fbd905c0254118710c572c56a14da2086b77d8ba3900a9ac627f6",
        new_event_id="cff77a9d1a8fbd905c0254118710c572c56a14da2086b77d8ba3900a9ac627f6",
        old_candidate_id="4596c4615b5b92431ec8d740dca52c7a2ba7b8434034fddf114b232cb028a966",
        new_candidate_id="f8f7358ae36981dcc5f346f7aaa4c88bf6dbfc3796f685e2bea9648d0442fe3d",
        announcement_date="2021-05-14",
        cash_amount=28.0,
        ratio=0.0718,
        successor_symbol="TDY",
        successor_security_id="US:EODHD:02738ac8-c50a-5089-bf68-f174ac71704b",
        official_source_url="https://www.sec.gov/Archives/edgar/data/1094285/000119312521161542/0001193125-21-161542.txt",
        official_source_hash="354312dc20154537f038d2bde1390789b770e8e5c1b62bd20c34449ecadac101",
        official_source_bytes=345_708,
        official_retrieved_at="2026-07-18T20:58:44.041522Z",
        filing_acceptance_datetime="20210514105133",
        official_patterns=(
            r"On May 14, 2021, Teledyne completed its acquisition of FLIR Systems",
            r"Merger Consideration.{0,300}?\$28\.00 per share in cash.{0,150}?0\.0718 of a share",
        ),
        raw_source_url="https://eodhd.com/api/eod/FLIR.US?from=2015-01-01&to=2026-07-15",
        raw_source_hash="6131bb18500249e42b462230fa1adbc43159b9b853e737970a043a9724e9cdb8",
        raw_source_bytes=184_471,
        raw_source_rows=1_604,
        raw_retrieved_at="2026-07-16T15:56:18.544807Z",
        raw_selected_sessions=("2021-05-13", "2021-05-14", "2021-05-17"),
        raw_selected_sha256="9624fcd637c2e1f49651324425021be3e9bf016bd6cd4bb783d498408a317436",
    ),
    ExactTerminalCase(
        symbol="QEP",
        security_id="US:EODHD:716dea51-f3a0-5381-9696-d097c877695f",
        exchange="NYSE",
        action_type="stock_merger",
        old_active_to="2021-03-18",
        last_real_session="2021-03-16",
        legal_completion_date="2021-03-17",
        market_transition_session="2021-03-17",
        removed_sessions=("2021-03-17", "2021-03-18"),
        old_event_id="2e8f5c5e5a3a887eb38b579ef45c47ab458770cd37002bcf5634cbfad0ae16da",
        new_event_id="2e8f5c5e5a3a887eb38b579ef45c47ab458770cd37002bcf5634cbfad0ae16da",
        old_candidate_id="548707cb96365a8ac74c2eadee81e552dcf04477b4fb55ec62ac33172f81658b",
        new_candidate_id="0134aa07f07f6f8a1ae42fdcf66dc87d2d42752242f8f3508a9cde35759efadb",
        announcement_date="2021-03-17",
        cash_amount=None,
        ratio=0.05,
        successor_symbol="FANG",
        successor_security_id="US:EODHD:c1f9ab83-05e8-57fc-8c07-f775826662c6",
        official_source_url="https://www.sec.gov/Archives/edgar/data/1539838/000119312521084144/0001193125-21-084144.txt",
        official_source_hash="c38c9f61ea9ddcdaef61f393012e620c21dea7c3062634c7aa8dbb53bec8af2a",
        official_source_bytes=533_895,
        official_retrieved_at="2026-07-18T20:58:41.780146Z",
        filing_acceptance_datetime="20210317164058",
        official_patterns=(
            r"On March 17, 2021, Diamondback Energy.{0,120}?completed its previously announced merger.{0,120}?QEP Resources",
            r"QEP common stock will no longer be listed for trading on NYSE",
            r"receive 0\.050.{0,80}?shares of common stock",
        ),
        raw_source_url="https://eodhd.com/api/eod/QEP.US?from=2015-01-01&to=2026-07-15",
        raw_source_hash="3ffd22cc3f12ada8b10cc47ba48a6ac21065d25bfef82877240a7461cb540287",
        raw_source_bytes=176_917,
        raw_source_rows=1_563,
        raw_retrieved_at="2026-07-16T15:57:16.749646Z",
        raw_selected_sessions=("2021-03-16", "2021-03-17", "2021-03-18"),
        raw_selected_sha256="7826aa9e40c0a86b40ee838adacde3bd352c1c132a9b7ae991b1e2e794da618a",
    ),
    ExactTerminalCase(
        symbol="KORS",
        security_id="US:EODHD:7623d5d2-1c3d-595f-8e96-408208fc7d37",
        exchange="NYSE MKT",
        action_type="ticker_change",
        old_active_to="2019-01-04",
        last_real_session="2018-12-31",
        legal_completion_date="2019-01-02",
        market_transition_session="2019-01-02",
        removed_sessions=("2019-01-02", "2019-01-03", "2019-01-04"),
        old_event_id="951960822646a8f972002508d162bd37a3ab3aaec42349fbff37caa1919b7b51",
        new_event_id="951960822646a8f972002508d162bd37a3ab3aaec42349fbff37caa1919b7b51",
        old_candidate_id="6513ade63d6a0c1e6a22f656593873f833dc1ab89a377d9a9b13ff6f3a9c436c",
        new_candidate_id="2a6ac318cc5c6e6c42b4788a75c400278ac09cc59abab518aff6ca44bb2b8512",
        announcement_date="2018-12-31",
        cash_amount=None,
        ratio=None,
        successor_symbol="CPRI",
        successor_security_id="US:EODHD:27f39ea8-f202-53a2-83bf-41a211b5f3d9",
        official_source_url="https://www.sec.gov/Archives/edgar/data/1530721/000119312518362322/0001193125-18-362322.txt",
        official_source_hash="ff4732e714524028c56a66c96e6ac8c50a401a36a4a46037cb80b01bb8454d25",
        official_source_bytes=483_284,
        official_retrieved_at="2026-07-18T20:58:24.767711Z",
        filing_acceptance_datetime="20181231163337",
        official_patterns=(
            r"January 2, 2019, the Company will trade on the New York Stock Exchange under the ticker .CPRI",
        ),
        raw_source_url="https://eodhd.com/api/eod/KORS.US?from=2015-01-01&to=2026-07-15",
        raw_source_hash="312c31c9cf1704c0638985f60c61cea3f4bbe088ecbdb080b9d9d7be728c657b",
        raw_source_bytes=114_444,
        raw_source_rows=1_009,
        raw_retrieved_at="2026-07-16T15:57:19.138389Z",
        raw_selected_sessions=("2018-12-31", "2019-01-02", "2019-01-03", "2019-01-04"),
        raw_selected_sha256="b7129e2427ff34d6dd6c9df350d60089b95302d81f8107d1cf94fce41e75e77e",
        successor_raw_source_url="https://eodhd.com/api/eod/CPRI.US?from=2015-01-01&to=2026-07-15",
        successor_raw_source_hash="878f32150378c2d4e2e2a4d922fdfe2c3cf2fd10d85d1385cb65498542398d5f",
        successor_raw_source_bytes=328_115,
        successor_raw_source_rows=2_899,
        successor_raw_retrieved_at="2026-07-16T15:56:32.278258Z",
        successor_selected_sessions=("2019-01-02", "2019-01-03", "2019-01-04"),
        successor_selected_sha256="abf26111357cecafaf7fe5a13b3d02ea94b8a15478f99c7a9cb6d8a54cc8864c",
    ),
    ExactTerminalCase(
        symbol="NLSN",
        security_id="US:EODHD:99eac7c1-6892-5b3a-bf4b-bc8143e3bfe2",
        exchange="NYSE",
        action_type="cash_merger",
        old_active_to="2022-10-12",
        last_real_session="2022-10-11",
        legal_completion_date="2022-10-11",
        market_transition_session="2022-10-12",
        removed_sessions=("2022-10-12",),
        old_event_id="0079876b484ced964d0395f5427978616ac571d778a8eff54a42c994cc459177",
        new_event_id="2aa7c18ca6ac8f0e4680a7e5456a04ba2f401fabfb2cc7dc0a5326e298f71176",
        old_candidate_id="fecbb5633630dd39cda2b8bf9d5a3d5e9b4b1c3d7962538da588ab1fc9adf211",
        new_candidate_id="4a4091e9f1adb59e12e4c1418588ea83c38b7110d94ba0295f3e0a3ddb2542b4",
        announcement_date="2022-10-11",
        cash_amount=28.0,
        ratio=None,
        successor_symbol="",
        successor_security_id="",
        official_source_url="https://www.sec.gov/Archives/edgar/data/1492633/000119312522260583/0001193125-22-260583.txt",
        official_source_hash="893b9f658c505f40fa304c7b18c89fd7ade6d29a6dcd166590181fae4d8e11fd",
        official_source_bytes=985_757,
        official_retrieved_at="2026-07-18T20:59:19.348906Z",
        filing_acceptance_datetime="20221011160209",
        official_patterns=(
            r"On October 11, 2022, Nielsen Holdings.{0,120}?completed the transactions",
            r"suspend trading of Company Ordinary Shares.{0,100}?prior to the opening of trading on October 12, 2022",
            r"payment of \$28\.00 in cash for each Company Ordinary Share",
        ),
        raw_source_url="https://eodhd.com/api/eod/NLSN.US?from=2015-01-01&to=2026-07-15",
        raw_source_hash="ff96ad118e42dc2c9dbebfe9d113a2424cb49ca21ff65a08cfd861e52f45c343",
        raw_source_bytes=226_517,
        raw_source_rows=1_959,
        raw_retrieved_at="2026-07-16T15:57:39.159390Z",
        raw_selected_sessions=("2022-10-11", "2022-10-12"),
        raw_selected_sha256="30dd45d1848577e9c76fe135b0a33c9de969fceb9ab48467335654e48eb9fa74",
    ),
    ExactTerminalCase(
        symbol="NLOK",
        security_id="US:EODHD:e9eea478-61d8-5762-9f5b-fbdfd69a02a3",
        exchange="NASDAQ",
        action_type="ticker_change",
        old_active_to="2022-11-10",
        last_real_session="2022-11-07",
        legal_completion_date="2022-11-07",
        market_transition_session="2022-11-08",
        removed_sessions=("2022-11-08", "2022-11-09", "2022-11-10"),
        old_event_id="002f1c86383d22157a2bfc5602decaf3880b85300a2c410ba1c84608eb43b967",
        new_event_id="d82975bc819ca47d10c7b2e2ca963422629980682933a4ee13b355fe564e6344",
        old_candidate_id="ded9b9f5f78cf788136a21b8b5899614d546208e4c7cf33eba64e66a4fd654a2",
        new_candidate_id="777f53ec86359bd2c6a62c385d4c1aac6bd432cf9e9c9633b4362ed49a3f54c2",
        announcement_date="2022-11-07",
        cash_amount=None,
        ratio=None,
        successor_symbol="GEN",
        successor_security_id="US:EODHD:cb0b8e57-3e09-542c-adf8-fe2c98d97b55",
        official_source_url="https://www.sec.gov/Archives/edgar/data/849399/000110465922115277/0001104659-22-115277.txt",
        official_source_hash="a4732aaa030033aebda1d508bed1742e237694dc97fdb1a71f9af02f20d95d83",
        official_source_bytes=467_634,
        official_retrieved_at="2026-07-18T20:59:19.501210Z",
        filing_acceptance_datetime="20221107123441",
        official_patterns=(
            r"Amendment is effective as of November 7, 2022",
            r"cease trading under the ticker symbol .NLOK.{0,120}?begin trading under its new ticker symbol, .GEN.{0,160}?effective on November 8, 2022",
        ),
        raw_source_url="https://eodhd.com/api/eod/NLOK.US?from=2015-01-01&to=2026-07-15",
        raw_source_hash="4935d389ce3dd31477c0906d12a9b8eda3e67332ec14bc715d57c6ce7a313d4a",
        raw_source_bytes=229_114,
        raw_source_rows=1_980,
        raw_retrieved_at="2026-07-16T15:58:25.371610Z",
        raw_selected_sessions=("2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10"),
        raw_selected_sha256="fc69183320ccbd12de4acad626c288d7455d393eed7435e4a26f7f3f8c61cd3c",
        successor_raw_source_url="https://eodhd.com/api/eod/GEN.US?from=2015-01-01&to=2026-07-15",
        successor_raw_source_hash="94bc6e419cd9d30a5f494641d83a136d68f31ccb337daf6632189ef5c153eef5",
        successor_raw_source_bytes=333_992,
        successor_raw_source_rows=2_899,
        successor_raw_retrieved_at="2026-07-16T15:58:09.417592Z",
        successor_selected_sessions=("2022-11-08", "2022-11-09", "2022-11-10"),
        successor_selected_sha256="d6f094306caa8f75b2619cc4d96e9888c701d317d6ff8ea4db651ef29bf74298",
    ),
    ExactTerminalCase(
        symbol="HBI",
        security_id="US:EODHD:865c1483-a99b-5066-b55a-649e24804d68",
        exchange="NYSE",
        action_type="stock_merger",
        old_active_to="2025-12-02",
        last_real_session="2025-11-28",
        legal_completion_date="2025-12-01",
        market_transition_session="2025-12-01",
        removed_sessions=("2025-12-01", "2025-12-02"),
        old_event_id="11d34abfe232c1b262916a2b845cc5987819f7a73b4a76f9cdb95dd1137ae85f",
        new_event_id="11d34abfe232c1b262916a2b845cc5987819f7a73b4a76f9cdb95dd1137ae85f",
        old_candidate_id="c42a1f3354d431227c2f3ec6e826c5c6cd19991c91e5b4f3b6c84bf1f8f69bc1",
        new_candidate_id="34a78e61e5ac60c2a56899790d79053e94af92f73ad968069624f22b4f6563d2",
        announcement_date="2025-12-01",
        cash_amount=0.8,
        ratio=0.102,
        successor_symbol="GIL",
        successor_security_id="US:EODHD:c18d86ee-dd25-509d-a4de-8a552fd6c69d",
        official_source_url="https://www.sec.gov/Archives/edgar/data/1359841/000119312525303276/0001193125-25-303276.txt",
        official_source_hash="85b0662c74efc9afdbd9e40babd6bb690c65287fd0ea4189cad7981b0148bde8",
        official_source_bytes=207_011,
        official_retrieved_at="2026-07-18T20:59:33.208240Z",
        filing_acceptance_datetime="20251201093656",
        official_patterns=(
            r"On December 1, 2025.{0,100}?Gildan Activewear.{0,100}?acquired Hanesbrands",
            r"Trading of Hanesbrands Common Stock on the NYSE was suspended prior to the opening of trading on the Closing Date",
            r"0\.102.{0,100}?common shares of Gildan.{0,100}?\$0\.80 in cash",
        ),
        raw_source_url="https://eodhd.com/api/eod/HBI.US?from=2015-01-01&to=2026-07-15",
        raw_source_hash="d451b649d888b78de99cfe53a9a666ddecefce370fa5a624a4cae8eb8865194d",
        raw_source_bytes=312_383,
        raw_source_rows=2_746,
        raw_retrieved_at="2026-07-16T15:57:29.612499Z",
        raw_selected_sessions=("2025-11-28", "2025-12-01", "2025-12-02"),
        raw_selected_sha256="e14af707eb731e5ab01b40ea6d259f3827cdc11a6034053b9c7d9e506a5533d9",
    ),
)

EXPECTED_REMOVED_ROWS = sum(len(case.removed_sessions) for case in CASES)
EARLY_HISTORY_OPERATION = "repair_us_early_terminal_history_supplements"
EARLY_HISTORY_LINEAGE_PREFIX = "early-terminal-history:"
EARLY_HISTORY_REQUEST_INVENTORY_SHA256 = (
    "8f8c46ef969a06085270efc50c3e50dae0d3aebed3b1edb07929fff718762ac7"
)
EARLY_HISTORY_INSERTED_ROWS = 118


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    planned_versions: dict[str, str]
    frames: dict[str, pd.DataFrame]
    summary: dict[str, Any]


@dataclass(frozen=True)
class FactorManifestBinding:
    """Current-release versions bound to the validated factor manifest."""

    manifest: DatasetManifest
    factor_version: str
    daily_price_version: str
    corporate_actions_version: str


FailureInjector = Callable[[str], None]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    if not _text(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).date().isoformat()


def _number(value: Any) -> float | None:
    if not _text(value):
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _same_number(value: Any, expected: float | None) -> bool:
    observed = _number(value)
    if expected is None:
        return observed is None
    return observed is not None and math.isclose(
        observed, expected, rel_tol=0, abs_tol=1e-12
    )


def _canonical_json_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalized_official_text(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    without_tags = re.sub(r"<[^>]+>", " ", decoded)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _safe_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    target = (base / object_path).resolve()
    if target == base or base not in target.parents:
        raise ValueError(f"Evidence path escapes repository: {object_path}.")
    return target


def _static_contract() -> None:
    if len(CASES) != 6 or {case.symbol for case in CASES} != {
        "FLIR",
        "QEP",
        "KORS",
        "NLSN",
        "NLOK",
        "HBI",
    }:
        raise RuntimeError("Short terminal-tail inventory changed.")
    if EXPECTED_REMOVED_ROWS != 13:
        raise RuntimeError("Short terminal-tail row inventory must equal 13.")
    for case in CASES:
        if canonical_lifecycle_event_id(
            case.security_id, case.action_type, case.legal_completion_date
        ) != case.old_event_id:
            raise RuntimeError(f"{case.symbol} old event ID changed.")
        if canonical_lifecycle_event_id(
            case.security_id, case.action_type, case.market_transition_session
        ) != case.new_event_id:
            raise RuntimeError(f"{case.symbol} repaired event ID changed.")
        if lifecycle_candidate_id(
            case.security_id, case.old_active_to
        ) != case.old_candidate_id:
            raise RuntimeError(f"{case.symbol} old candidate ID changed.")
        if lifecycle_candidate_id(
            case.security_id, case.last_real_session
        ) != case.new_candidate_id:
            raise RuntimeError(f"{case.symbol} repaired candidate ID changed.")
        if tuple(sorted(case.removed_sessions)) != case.removed_sessions:
            raise RuntimeError(f"{case.symbol} removed sessions are not ordered.")
        if min(case.removed_sessions) <= case.last_real_session:
            raise RuntimeError(f"{case.symbol} tail overlaps its final real session.")


def registry_draft() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in CASES:
        row = {
            "symbol": case.symbol,
            "security_id": case.security_id,
            "action_type": case.action_type,
            "old_event_id": case.old_event_id,
            "event_id": case.new_event_id,
            "old_candidate_id": case.old_candidate_id,
            "candidate_id": case.new_candidate_id,
            "legal_completion_date": case.legal_completion_date,
            "last_real_session": case.last_real_session,
            "market_transition_session": case.market_transition_session,
            "removed_sessions": list(case.removed_sessions),
            "raw_source_hash": case.raw_source_hash,
            "raw_selected_sha256": case.raw_selected_sha256,
            "official_source_hash": case.official_source_hash,
            "successor_raw_source_hash": case.successor_raw_source_hash,
            "successor_selected_sha256": case.successor_selected_sha256,
        }
        row["registry_item_sha256"] = _canonical_json_sha256(row)
        output.append(row)
    return output


def registry_inventory_sha256() -> str:
    return _canonical_json_sha256(registry_draft())


def _archive_row(
    archive: pd.DataFrame,
    *,
    digest: str,
    suffix: str,
    dataset: str,
    content_type: str,
    source_url: str,
    retrieved_at: str,
) -> Mapping[str, Any]:
    object_path = f"archives/{ARCHIVE_SESSION}/{digest}.{suffix}.gz"
    related = (
        archive["archive_id"].astype(str).eq(digest)
        | archive["source_hash"].astype(str).eq(digest)
        | archive["object_path"].astype(str).eq(object_path)
        | archive["source_url"].astype(str).eq(source_url)
    )
    rows = archive.loc[related]
    if len(rows) != 1:
        raise ValueError(
            f"Exact archive inventory for {digest} is not one row; found {len(rows)}."
        )
    row = rows.iloc[0]
    expected = {
        "archive_id": digest,
        "dataset": dataset,
        "object_path": object_path,
        "content_type": content_type,
        "effective_date": ARCHIVE_SESSION,
        "source": dataset,
        "retrieved_at": retrieved_at,
        "source_hash": digest,
        "source_url": source_url,
    }
    changed = [
        field
        for field, value in expected.items()
        if (
            _date(row.get(field)) != value
            if field == "effective_date"
            else _text(row.get(field)) != value
        )
    ]
    if changed:
        raise ValueError(
            f"Exact archive row {digest} changed: {', '.join(changed)}."
        )
    return row


def _archive_payload(
    repository: LocalDatasetRepository,
    row: Mapping[str, Any],
    *,
    digest: str,
    expected_bytes: int,
) -> bytes:
    path = _safe_path(repository.root, _text(row.get("object_path")))
    if not path.is_file():
        raise FileNotFoundError(f"Exact archive payload is missing: {path}.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError(f"Exact archive payload is invalid gzip: {path}.") from exc
    observed = hashlib.sha256(payload).hexdigest()
    if observed != digest or len(payload) != expected_bytes:
        raise ValueError(
            "Exact archive payload hash/size changed: "
            f"expected={digest}/{expected_bytes}; observed={observed}/{len(payload)}."
        )
    return payload


def _raw_records(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    source_url: str,
    source_hash: str,
    source_bytes: int,
    source_rows: int,
    retrieved_at: str,
    selected_sessions: tuple[str, ...],
    selected_sha256: str,
) -> list[Mapping[str, Any]]:
    row = _archive_row(
        archive,
        digest=source_hash,
        suffix="json",
        dataset="eodhd_eod",
        content_type="application/json",
        source_url=source_url,
        retrieved_at=retrieved_at,
    )
    payload = _archive_payload(
        repository, row, digest=source_hash, expected_bytes=source_bytes
    )
    try:
        records = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Archived EODHD response is invalid JSON.") from exc
    if not isinstance(records, list) or len(records) != source_rows or not all(
        isinstance(value, Mapping) for value in records
    ):
        raise ValueError("Archived EODHD response inventory changed.")
    selected = [
        value for value in records if _date(value.get("date")) in selected_sessions
    ]
    if tuple(_date(value.get("date")) for value in selected) != selected_sessions:
        raise ValueError("Archived EODHD selected-session inventory changed.")
    if _canonical_json_sha256(selected) != selected_sha256:
        raise ValueError("Archived EODHD selected-session hash changed.")
    return list(records)


def _verify_case_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    case: ExactTerminalCase,
) -> dict[str, list[Mapping[str, Any]]]:
    official_row = _archive_row(
        archive,
        digest=case.official_source_hash,
        suffix="txt",
        dataset="sec_edgar_filing",
        content_type="text/plain",
        source_url=case.official_source_url,
        retrieved_at=case.official_retrieved_at,
    )
    official = _archive_payload(
        repository,
        official_row,
        digest=case.official_source_hash,
        expected_bytes=case.official_source_bytes,
    )
    decoded = official.decode("utf-8", errors="replace")
    if not re.search(
        rf"ACCEPTANCE-DATETIME:?[ >]*{case.filing_acceptance_datetime}",
        decoded,
        re.I,
    ):
        raise ValueError(f"{case.symbol} SEC acceptance timestamp changed.")
    normalized = _normalized_official_text(official)
    missing = [
        pattern
        for pattern in case.official_patterns
        if not re.search(pattern, normalized, re.I)
    ]
    if missing:
        raise ValueError(
            f"{case.symbol} SEC payload no longer proves the exact boundary: "
            + ", ".join(missing)
            + "."
        )
    source = _raw_records(
        repository,
        archive,
        source_url=case.raw_source_url,
        source_hash=case.raw_source_hash,
        source_bytes=case.raw_source_bytes,
        source_rows=case.raw_source_rows,
        retrieved_at=case.raw_retrieved_at,
        selected_sessions=case.raw_selected_sessions,
        selected_sha256=case.raw_selected_sha256,
    )
    successor: list[Mapping[str, Any]] = []
    if case.successor_raw_source_hash:
        successor = _raw_records(
            repository,
            archive,
            source_url=case.successor_raw_source_url,
            source_hash=case.successor_raw_source_hash,
            source_bytes=case.successor_raw_source_bytes,
            source_rows=case.successor_raw_source_rows,
            retrieved_at=case.successor_raw_retrieved_at,
            selected_sessions=case.successor_selected_sessions,
            selected_sha256=case.successor_selected_sha256,
        )
    return {"source": source, "successor": successor}


def _raw_by_date(
    records: list[Mapping[str, Any]], sessions: tuple[str, ...]
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for session in sessions:
        rows = [value for value in records if _date(value.get("date")) == session]
        if len(rows) != 1:
            raise ValueError(f"Raw EOD inventory for {session} is not exact.")
        output[session] = rows[0]
    return output


def _parquet_row_matches_raw(row: Mapping[str, Any], raw: Mapping[str, Any]) -> bool:
    return bool(
        all(
            _same_number(row.get(field), _number(raw.get(field)))
            for field in ("open", "high", "low", "close", "volume")
        )
        and _text(row.get("source")) == "eodhd_eod"
        and _text(row.get("source_hash"))
        and _text(row.get("retrieved_at"))
    )


def _price_state(
    prices: pd.DataFrame,
    case: ExactTerminalCase,
    source_records: list[Mapping[str, Any]],
) -> str:
    target = prices.loc[prices["security_id"].astype(str).eq(case.security_id)].copy()
    sessions = pd.to_datetime(target["session"], errors="raise").dt.date.astype(str)
    if len(target) == 0:
        raise ValueError(f"{case.symbol} price history is missing.")
    after = tuple(sorted(sessions.loc[sessions.gt(case.last_real_session)]))
    old = after == case.removed_sessions and sessions.max() == case.old_active_to
    repaired = not after and sessions.max() == case.last_real_session
    if old == repaired:
        raise ValueError(
            f"{case.symbol} prices are neither exact raw nor exact repaired state."
        )
    expected_sessions = (
        case.raw_selected_sessions if old else (case.last_real_session,)
    )
    raw = _raw_by_date(source_records, expected_sessions)
    for session in expected_sessions:
        rows = target.loc[sessions.eq(session)]
        if len(rows) != 1 or not _parquet_row_matches_raw(rows.iloc[0], raw[session]):
            raise ValueError(f"{case.symbol} Parquet OHLCV differs on {session}.")
        if (
            _text(rows.iloc[0].get("source_hash")) != case.raw_source_hash
            or _text(rows.iloc[0].get("retrieved_at")) != case.raw_retrieved_at
        ):
            raise ValueError(f"{case.symbol} Parquet price provenance changed.")
    return "old" if old else "repaired"


def _verify_successor_duplicate_evidence(
    prices: pd.DataFrame,
    case: ExactTerminalCase,
    successor_records: list[Mapping[str, Any]],
) -> None:
    if not case.successor_raw_source_hash:
        return
    raw = _raw_by_date(successor_records, case.successor_selected_sessions)
    target = prices.loc[
        prices["security_id"].astype(str).eq(case.successor_security_id)
    ].copy()
    sessions = pd.to_datetime(target["session"], errors="raise").dt.date.astype(str)
    for session in case.successor_selected_sessions:
        rows = target.loc[sessions.eq(session)]
        if len(rows) != 1 or not _parquet_row_matches_raw(rows.iloc[0], raw[session]):
            raise ValueError(
                f"{case.symbol}->{case.successor_symbol} successor evidence changed on {session}."
            )
        if (
            _text(rows.iloc[0].get("source_hash"))
            != case.successor_raw_source_hash
            or _text(rows.iloc[0].get("retrieved_at"))
            != case.successor_raw_retrieved_at
        ):
            raise ValueError(
                f"{case.symbol}->{case.successor_symbol} successor provenance changed."
            )


def _identity_state(
    frame: pd.DataFrame, case: ExactTerminalCase, *, history: bool
) -> str:
    symbol_field = "symbol" if history else "primary_symbol"
    end_field = "effective_to" if history else "active_to"
    rows = frame.loc[
        frame["security_id"].astype(str).eq(case.security_id)
        & frame[symbol_field].astype(str).str.upper().eq(case.symbol)
    ]
    if len(rows) != 1:
        raise ValueError(f"{case.symbol} identity inventory is not one exact row.")
    row = rows.iloc[0]
    old_end = {"", case.old_active_to} if history else {case.old_active_to}
    old = bool(
        _date(row.get(end_field)) in old_end
        and _text(row.get("exchange")) == case.exchange
        and _text(row.get("source")) == OLD_IDENTITY_SOURCE
        and _text(row.get("source_url")) == OLD_IDENTITY_URL
        and _text(row.get("source_hash")) == OLD_IDENTITY_HASH
        and _text(row.get("retrieved_at")) == OLD_IDENTITY_RETRIEVED_AT
    )
    repaired = bool(
        _date(row.get(end_field)) == case.last_real_session
        and _text(row.get("exchange")) == case.exchange
        and _text(row.get("source")) == REPAIRED_IDENTITY_SOURCE
        and _text(row.get("source_url")) == case.official_source_url
        and _text(row.get("source_hash")) == case.official_source_hash
        and _text(row.get("retrieved_at")) == REPAIR_REVIEWED_AT
    )
    if old == repaired:
        raise ValueError(
            f"{case.symbol} identity is neither exact old nor repaired state."
        )
    return "old" if old else "repaired"


def _action_base(row: Mapping[str, Any], case: ExactTerminalCase) -> bool:
    return bool(
        _text(row.get("security_id")) == case.security_id
        and _text(row.get("action_type")) == case.action_type
        and _date(row.get("announcement_date")) == case.announcement_date
        and _same_number(row.get("cash_amount"), case.cash_amount)
        and _same_number(row.get("ratio"), case.ratio)
        and _text(row.get("currency")) == "USD"
        and _text(row.get("new_security_id")) == case.successor_security_id
        and _text(row.get("new_symbol")).upper() == case.successor_symbol
        and bool(row.get("official"))
        and _text(row.get("source_url")) == case.official_source_url
        and _text(row.get("source_kind")) == ACTION_SOURCE_KIND
        and _text(row.get("source")) == ACTION_SOURCE
        and _text(row.get("retrieved_at")) == case.official_retrieved_at
        and _text(row.get("source_hash")) == case.official_source_hash
    )


def _action_state(actions: pd.DataFrame, case: ExactTerminalCase) -> str:
    rows = actions.loc[
        actions["security_id"].astype(str).eq(case.security_id)
        & actions["action_type"].astype(str).eq(case.action_type)
    ]
    if len(rows) != 1 or not _action_base(rows.iloc[0], case):
        raise ValueError(f"{case.symbol} exact lifecycle action changed.")
    row = rows.iloc[0]
    old = bool(
        _text(row.get("event_id")) == case.old_event_id
        and _date(row.get("effective_date")) == case.legal_completion_date
        and _date(row.get("ex_date")) == case.legal_completion_date
    )
    repaired = bool(
        _text(row.get("event_id")) == case.new_event_id
        and _date(row.get("effective_date")) == case.market_transition_session
        and _date(row.get("ex_date")) == case.market_transition_session
    )
    if case.old_event_id == case.new_event_id:
        if not old:
            raise ValueError(f"{case.symbol} unchanged action state drifted.")
        return "neutral"
    if old == repaired:
        raise ValueError(
            f"{case.symbol} action is neither exact legal-date nor market-date state."
        )
    return "old" if old else "repaired"


def _resolution_state(
    resolutions: pd.DataFrame, case: ExactTerminalCase
) -> str:
    rows = resolutions.loc[
        resolutions["security_id"].astype(str).eq(case.security_id)
    ]
    if len(rows) != 1:
        raise ValueError(f"{case.symbol} lifecycle resolution inventory changed.")
    row = rows.iloc[0]
    common = bool(
        _text(row.get("symbol")).upper() == case.symbol
        and _text(row.get("resolution")) == "applied"
        and not _text(row.get("exception_code"))
        and not _text(row.get("exception_reason"))
        and _text(row.get("successor_security_id"))
        == case.successor_security_id
        and _text(row.get("successor_symbol")).upper() == case.successor_symbol
        and _text(row.get("source_url")) == case.official_source_url
        and _text(row.get("source_hash")) == case.official_source_hash
    )
    if not common:
        raise ValueError(f"{case.symbol} lifecycle resolution terms changed.")
    old = bool(
        _text(row.get("candidate_id")) == case.old_candidate_id
        and _date(row.get("last_price_date")) == case.old_active_to
        and _text(row.get("event_id")) == case.old_event_id
        and _text(row.get("reviewed_by")) == OLD_RESOLUTION_REVIEWER
        and _text(row.get("reviewed_at")) == OLD_RESOLUTION_REVIEWED_AT
        and _text(row.get("source")) == OLD_RESOLUTION_SOURCE
        and _text(row.get("retrieved_at")) == case.official_retrieved_at
    )
    repaired = bool(
        _text(row.get("candidate_id")) == case.new_candidate_id
        and _date(row.get("last_price_date")) == case.last_real_session
        and _text(row.get("event_id")) == case.new_event_id
        and _text(row.get("reviewed_by")) == REPAIRED_REVIEWER
        and _text(row.get("reviewed_at")) == REPAIR_REVIEWED_AT
        and _text(row.get("source")) == REPAIRED_RESOLUTION_SOURCE
        and _text(row.get("retrieved_at")) == REPAIR_REVIEWED_AT
    )
    if old == repaired:
        raise ValueError(
            f"{case.symbol} resolution is neither exact old nor repaired state."
        )
    return "old" if old else "repaired"


def _case_state(
    *,
    symbol: str,
    price_state: str,
    master_state: str,
    history_state: str,
    action_state: str,
    resolution_state: str,
) -> str:
    states = {price_state, master_state, history_state, resolution_state}
    if action_state != "neutral":
        states.add(action_state)
    if len(states) != 1:
        raise RuntimeError(
            f"{symbol} short terminal-tail repair is partially applied: {sorted(states)}."
        )
    return next(iter(states))


def _rewrite_prices(prices: pd.DataFrame) -> pd.DataFrame:
    output = prices.copy(deep=True)
    sessions = pd.to_datetime(output["session"], errors="raise").dt.date.astype(str)
    remove = pd.Series(False, index=output.index)
    for case in CASES:
        remove |= output["security_id"].astype(str).eq(case.security_id) & sessions.isin(
            case.removed_sessions
        )
    if int(remove.sum()) != EXPECTED_REMOVED_ROWS:
        raise ValueError("Exact short terminal-tail price inventory changed.")
    return output.loc[~remove].reset_index(drop=True)


def _rewrite_identity(frame: pd.DataFrame, *, history: bool) -> pd.DataFrame:
    output = frame.copy(deep=True)
    end_field = "effective_to" if history else "active_to"
    symbol_field = "symbol" if history else "primary_symbol"
    for case in CASES:
        rows = (
            output["security_id"].astype(str).eq(case.security_id)
            & output[symbol_field].astype(str).str.upper().eq(case.symbol)
        )
        updates = {
            end_field: case.last_real_session,
            "source": REPAIRED_IDENTITY_SOURCE,
            "source_url": case.official_source_url,
            "retrieved_at": REPAIR_REVIEWED_AT,
            "source_hash": case.official_source_hash,
        }
        for field, value in updates.items():
            output.loc[rows, field] = value
    return output.reset_index(drop=True)


def _rewrite_actions(actions: pd.DataFrame) -> pd.DataFrame:
    output = actions.copy(deep=True)
    for case in CASES:
        if case.old_event_id == case.new_event_id:
            continue
        rows = (
            output["security_id"].astype(str).eq(case.security_id)
            & output["action_type"].astype(str).eq(case.action_type)
        )
        output.loc[rows, ["event_id", "effective_date", "ex_date"]] = [
            case.new_event_id,
            case.market_transition_session,
            case.market_transition_session,
        ]
    return output.reset_index(drop=True)


def _rewrite_resolutions(resolutions: pd.DataFrame) -> pd.DataFrame:
    output = resolutions.copy(deep=True)
    for case in CASES:
        rows = output["security_id"].astype(str).eq(case.security_id)
        updates = {
            "candidate_id": case.new_candidate_id,
            "last_price_date": case.last_real_session,
            "event_id": case.new_event_id,
            "reviewed_by": REPAIRED_REVIEWER,
            "reviewed_at": REPAIR_REVIEWED_AT,
            "source": REPAIRED_RESOLUTION_SOURCE,
            "retrieved_at": REPAIR_REVIEWED_AT,
        }
        for field, value in updates.items():
            output.loc[rows, field] = value
    return output.reset_index(drop=True)


def _adjustment_source_version(price_version: str, action_version: str) -> str:
    if not price_version or not action_version:
        raise RuntimeError("Exact factor lineage requires price and action versions.")
    return f"{price_version}+{action_version}"


def _early_history_source_version(
    daily_price_version: str, corporate_actions_version: str
) -> str:
    if not daily_price_version or not corporate_actions_version:
        raise RuntimeError("Exact early-history lineage requires both input versions.")
    digest = hashlib.sha256(
        f"{daily_price_version}|{corporate_actions_version}".encode()
    ).hexdigest()
    return f"{EARLY_HISTORY_LINEAGE_PREFIX}{digest}"


def _is_exact_early_history_factor_rebuild(
    current: pd.DataFrame,
    binding: FactorManifestBinding,
) -> bool:
    """Recognise only the deterministic downstream full rebuild of this repair.

    The early-terminal-history operation legitimately rebuilt every factor after
    the short-tail repair.  Its timestamps and lineage therefore differ from the
    original short-tail write.  Idempotency may accept that later provenance only
    when the *current* release versions, factor manifest, preserved short-tail
    registry, and every factor row form one exact content-bound batch.
    """

    manifest = binding.manifest
    metadata = manifest.metadata
    factor_suffix = "-adjustment_factors"
    if not (
        manifest.dataset == "adjustment_factors"
        and manifest.version == binding.factor_version
        and manifest.quality == "valid"
        and manifest.conflict_count == 0
        and manifest.unresolved_action_count == 0
        and not manifest.warnings
        and binding.factor_version.endswith(factor_suffix)
    ):
        return False

    stem = binding.factor_version[: -len(factor_suffix)]
    prefix = f"early-terminal-history-{manifest.completed_session}-"
    token = stem[len(prefix) :] if stem.startswith(prefix) else ""
    if not (
        len(token) == 32
        and all(character in "0123456789abcdef" for character in token)
        and binding.daily_price_version == f"{stem}-daily_price_raw"
        and binding.corporate_actions_version == f"{stem}-corporate_actions"
    ):
        return False

    expected_lineage = _early_history_source_version(
        binding.daily_price_version, binding.corporate_actions_version
    )
    exact_metadata = bool(
        metadata.get("operation") == EARLY_HISTORY_OPERATION
        and metadata.get("source_version") == expected_lineage
        and metadata.get("source_daily_price_version")
        == binding.daily_price_version
        and metadata.get("source_corporate_actions_version")
        == binding.corporate_actions_version
        and metadata.get("request_inventory_sha256")
        == EARLY_HISTORY_REQUEST_INVENTORY_SHA256
        and type(metadata.get("inserted_price_rows")) is int
        and metadata.get("inserted_price_rows") == EARLY_HISTORY_INSERTED_ROWS
        and type(metadata.get("inserted_rows")) is int
        and metadata.get("inserted_rows") == EARLY_HISTORY_INSERTED_ROWS
        and type(metadata.get("existing_economic_rows_changed")) is int
        and metadata.get("existing_economic_rows_changed") == 0
        and metadata.get("inherits_parent") is False
        and metadata.get("apply_network_accessed") is False
        and metadata.get("r2_accessed") is False
        and metadata.get("short_terminal_tail_registry_sha256")
        == registry_inventory_sha256()
        and metadata.get("short_terminal_tail_registry") == registry_draft()
    )
    if not exact_metadata:
        return False

    # ``inherits_parent=False`` plus an exact manifest row count proves this is
    # a full factor inventory, not a delta that happens to carry the same label.
    if not manifest.files or sum(item.row_count for item in manifest.files) != len(
        current
    ):
        return False
    if not (
        set(current["source_version"].astype(str)) == {expected_lineage}
        and set(current["source_hash"].astype(str)) == {expected_lineage}
        and set(current["source"].astype(str)) == {"derived"}
    ):
        return False

    calculated = set(current["calculated_at"].astype(str))
    retrieved = set(current["retrieved_at"].astype(str))
    if len(calculated) != 1 or calculated != retrieved:
        return False
    try:
        factor_timestamp = pd.Timestamp(next(iter(calculated)))
        manifest_timestamp = pd.Timestamp(manifest.created_at)
    except (TypeError, ValueError):
        return False
    return bool(
        not pd.isna(factor_timestamp)
        and not pd.isna(manifest_timestamp)
        and factor_timestamp.tzinfo is not None
        and manifest_timestamp.tzinfo is not None
        and factor_timestamp <= manifest_timestamp
    )


def _new_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"short-terminal-tails-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


def _expected_factors(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
    columns: list[str],
) -> pd.DataFrame:
    output = build_adjustment_factors(
        prices, actions, source_version=source_version
    ).reindex(columns=columns)
    output["source_version"] = source_version
    output["calculated_at"] = REPAIR_REVIEWED_AT
    output["source"] = "derived"
    output["retrieved_at"] = REPAIR_REVIEWED_AT
    output["source_hash"] = source_version
    return output.reset_index(drop=True)


def _normalized_factor_values(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        ["security_id", "session", "split_factor", "total_return_factor"]
    ].copy()
    output["security_id"] = output["security_id"].astype(str)
    output["session"] = pd.to_datetime(output["session"], errors="raise").dt.normalize()
    if output.duplicated(["security_id", "session"]).any():
        raise ValueError("Adjustment-factor keys are duplicated.")
    return output.sort_values(["security_id", "session"], ignore_index=True)


def _price_keys(prices: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    values = pd.DataFrame(
        {
            "security_id": prices["security_id"].astype(str),
            "session": pd.to_datetime(
                prices["session"], errors="raise"
            ).dt.normalize(),
        }
    )
    if values.duplicated(["security_id", "session"]).any():
        raise ValueError("Daily-price keys are duplicated.")
    return set(values.itertuples(index=False, name=None))


def _factor_economics_equal(current: pd.DataFrame, expected: pd.DataFrame) -> int:
    left = _normalized_factor_values(current)
    right = _normalized_factor_values(expected)
    if len(left) != len(right) or not left[["security_id", "session"]].equals(
        right[["security_id", "session"]]
    ):
        raise ValueError("Retained adjustment-factor inventory changed.")
    changed = np.zeros(len(left), dtype=bool)
    for column in ("split_factor", "total_return_factor"):
        old = pd.to_numeric(left[column], errors="raise").to_numpy(dtype=float)
        new = pd.to_numeric(right[column], errors="raise").to_numpy(dtype=float)
        changed |= ~((old == new) | (np.isnan(old) & np.isnan(new)))
    if changed.any():
        sample = left.loc[changed, ["security_id", "session"]].head(10)
        raise ValueError(
            "Short terminal-tail repair changed retained factor economics: "
            + json.dumps(sample.to_dict("records"), default=str, sort_keys=True)
        )
    return 0


def _prepare_factors(
    current: pd.DataFrame,
    old_prices: pd.DataFrame,
    new_prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
    repaired_state: bool,
    factor_manifest_binding: FactorManifestBinding | None = None,
) -> tuple[pd.DataFrame, int]:
    current_keys = set(
        _normalized_factor_values(current)[["security_id", "session"]].itertuples(
            index=False, name=None
        )
    )
    old_keys = _price_keys(old_prices)
    if current_keys != old_keys:
        raise ValueError("Current factor keys do not exactly match current prices.")
    expected = _expected_factors(
        new_prices,
        actions,
        source_version=source_version,
        columns=list(current.columns),
    )
    expected_keys = set(
        _normalized_factor_values(expected)[["security_id", "session"]].itertuples(
            index=False, name=None
        )
    )
    if expected_keys != _price_keys(new_prices):
        raise ValueError("Rebuilt factors do not exactly cover repaired prices.")
    removed_keys = current_keys - expected_keys
    if repaired_state:
        if removed_keys:
            raise ValueError("Already-repaired factors still have extra keys.")
        retained = current
    else:
        wanted = {
            (case.security_id, pd.Timestamp(session).normalize())
            for case in CASES
            for session in case.removed_sessions
        }
        if removed_keys != wanted or len(removed_keys) != EXPECTED_REMOVED_ROWS:
            raise ValueError("Adjustment-factor tail inventory changed.")
        keep = [
            (str(row.security_id), pd.Timestamp(row.session).normalize())
            not in removed_keys
            for row in current.itertuples(index=False)
        ]
        retained = current.loc[keep].reset_index(drop=True)
    _factor_economics_equal(retained, expected)
    if repaired_state:
        exact_lineage = bool(
            set(current["source_version"].astype(str)) == {source_version}
            and set(current["source_hash"].astype(str)) == {source_version}
            and set(current["source"].astype(str)) == {"derived"}
            and set(current["calculated_at"].astype(str)) == {REPAIR_REVIEWED_AT}
            and set(current["retrieved_at"].astype(str)) == {REPAIR_REVIEWED_AT}
        )
        exact_downstream_rebuild = bool(
            factor_manifest_binding is not None
            and _is_exact_early_history_factor_rebuild(
                current, factor_manifest_binding
            )
        )
        if not (exact_lineage or exact_downstream_rebuild):
            raise RuntimeError(
                "Repaired factor provenance is stale; neither the original "
                "short-tail lineage nor the exact current early-history full "
                "rebuild is present."
            )
        if exact_downstream_rebuild:
            return current.copy(deep=True).reset_index(drop=True), len(removed_keys)
    return expected, len(removed_keys)


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        overrides: Mapping[str, pd.DataFrame],
    ):
        self.base = base
        self.versions = dict(versions)
        self.overrides = dict(overrides)

    def current_manifest(self, dataset: str):
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        version = self.versions.get(dataset)
        return self.base.read_frame(dataset, version) if version else pd.DataFrame()


def _pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}.")
        output[dataset] = etag
    return output


def _terminal_report(frames: Mapping[str, pd.DataFrame], release_version: str):
    return audit_terminal_transitions(
        corporate_actions=frames["corporate_actions"],
        lifecycle_resolutions=frames["lifecycle_resolutions"],
        daily_price_raw=frames["daily_price_raw"],
        index_constituent_anchors=frames["index_constituent_anchors"],
        index_membership_events=frames["index_membership_events"],
        symbol_history=frames["symbol_history"],
        security_master=frames["security_master"],
        release_version=release_version,
    )


def _issue_fingerprint(report, *, exclude_targets: bool) -> tuple[str, ...]:
    target_ids = {case.security_id for case in CASES}
    return tuple(
        sorted(
            json.dumps(issue.to_dict(), sort_keys=True)
            for issue in report.issues
            if (issue.security_id not in target_ids) == exclude_targets
        )
    )


def _validate_candidate_snapshot(repository: Any) -> bool:
    """Allow only the independently reviewed, pre-existing NBL replay gap."""

    report = validate_repository_snapshot(repository)
    errors = tuple(issue for issue in report.issues if issue.severity == "error")
    if not errors:
        return False
    expected = EXPECTED_SNAPSHOT_IDENTITY_GAP
    fingerprint = index_member_identity_gap_fingerprint(
        index_id=expected["index_id"],
        replay_date=expected["replay_date"],
        security_id=expected["security_id"],
        next_remove_event_id=expected["next_remove_event_id"],
        next_remove_effective_date=expected["next_remove_effective_date"],
        next_remove_source=expected["next_remove_source"],
        next_remove_source_hash=expected["next_remove_source_hash"],
    )
    if fingerprint != expected["fingerprint"]:
        raise AssertionError("Code-pinned NBL identity-gap fingerprint changed.")
    if not (
        len(errors) == 1
        and errors[0].code == expected["code"]
        and errors[0].row_count == expected["row_count"]
        and tuple(errors[0].fingerprints) == (fingerprint,)
    ):
        report.raise_for_errors()
    return True


def prepare_repair(repository: LocalDatasetRepository) -> PreparedRepair:
    _static_contract()
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError(
            "Current release lacks short-tail datasets: " + ", ".join(missing)
        )
    pointer_etags = _pointer_etags(repository, release)
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    evidence: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    states: dict[str, str] = {}
    for case in CASES:
        evidence[case.symbol] = _verify_case_evidence(
            repository, frames["source_archive"], case
        )
        _verify_successor_duplicate_evidence(
            frames["daily_price_raw"], case, evidence[case.symbol]["successor"]
        )
        states[case.symbol] = _case_state(
            symbol=case.symbol,
            price_state=_price_state(
                frames["daily_price_raw"], case, evidence[case.symbol]["source"]
            ),
            master_state=_identity_state(
                frames["security_master"], case, history=False
            ),
            history_state=_identity_state(
                frames["symbol_history"], case, history=True
            ),
            action_state=_action_state(frames["corporate_actions"], case),
            resolution_state=_resolution_state(
                frames["lifecycle_resolutions"], case
            ),
        )
    if len(set(states.values())) != 1:
        raise RuntimeError(f"Short terminal-tail batch is partially applied: {states}.")
    state = next(iter(states.values()))
    before_report = _terminal_report(frames, release.version)

    if state == "old":
        planned_versions = _new_versions(release)
        overrides: dict[str, pd.DataFrame] = {
            "corporate_actions": _rewrite_actions(frames["corporate_actions"]),
            "daily_price_raw": _rewrite_prices(frames["daily_price_raw"]),
            "lifecycle_resolutions": _rewrite_resolutions(
                frames["lifecycle_resolutions"]
            ),
            "security_master": _rewrite_identity(
                frames["security_master"], history=False
            ),
            "symbol_history": _rewrite_identity(
                frames["symbol_history"], history=True
            ),
        }
        lineage = _adjustment_source_version(
            planned_versions["daily_price_raw"],
            planned_versions["corporate_actions"],
        )
        repaired_state = False
        factor_manifest_binding = None
    elif state == "repaired":
        planned_versions = {}
        overrides = {
            dataset: frames[dataset]
            for dataset in WRITE_DATASETS
            if dataset != "adjustment_factors"
        }
        lineage = _adjustment_source_version(
            release.dataset_versions["daily_price_raw"],
            release.dataset_versions["corporate_actions"],
        )
        repaired_state = True
        factor_manifest = repository.current_manifest("adjustment_factors")
        if (
            factor_manifest is None
            or factor_manifest.version
            != release.dataset_versions["adjustment_factors"]
        ):
            raise RuntimeError(
                "Current adjustment-factor manifest is not release-bound."
            )
        factor_manifest_binding = FactorManifestBinding(
            manifest=factor_manifest,
            factor_version=release.dataset_versions["adjustment_factors"],
            daily_price_version=release.dataset_versions["daily_price_raw"],
            corporate_actions_version=release.dataset_versions[
                "corporate_actions"
            ],
        )
    else:  # pragma: no cover
        raise AssertionError(state)

    factors, factor_rows_removed = _prepare_factors(
        frames["adjustment_factors"],
        frames["daily_price_raw"],
        overrides["daily_price_raw"],
        overrides["corporate_actions"],
        source_version=lineage,
        repaired_state=repaired_state,
        factor_manifest_binding=factor_manifest_binding,
    )
    overrides["adjustment_factors"] = factors
    for dataset in WRITE_DATASETS:
        validate_dataset(
            dataset,
            overrides[dataset],
            completed_session=release.completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    candidate = _CandidateRepository(
        repository, release.dataset_versions, overrides
    )
    snapshot_identity_gap_recorded = _validate_candidate_snapshot(candidate)
    after_frames = dict(frames)
    after_frames.update(overrides)
    after_report = _terminal_report(after_frames, f"{release.version}:short-tail-plan")
    if _issue_fingerprint(
        before_report, exclude_targets=True
    ) != _issue_fingerprint(after_report, exclude_targets=True):
        raise RuntimeError("Short terminal-tail repair changed unrelated readiness issues.")
    target_issues_after = _issue_fingerprint(after_report, exclude_targets=False)
    if target_issues_after:
        raise RuntimeError(
            "Short terminal-tail targets still fail readiness: "
            + json.dumps(target_issues_after)
        )
    candidate_hash = lifecycle_candidate_set_sha256(
        overrides["lifecycle_resolutions"][["security_id", "last_price_date"]]
    )
    resolution_hash = lifecycle_resolution_set_sha256(
        overrides["lifecycle_resolutions"]
    )
    changed = state == "old"
    factor_source_version = lineage
    if repaired_state:
        observed_lineages = set(
            frames["adjustment_factors"]["source_version"].astype(str)
        )
        if len(observed_lineages) != 1:
            raise RuntimeError("Repaired factor lineage is not uniform.")
        factor_source_version = next(iter(observed_lineages))
    targets = [
        {
            "symbol": case.symbol,
            "security_id": case.security_id,
            "legal_completion_date": case.legal_completion_date,
            "last_real_session": case.last_real_session,
            "market_transition_session": case.market_transition_session,
            "removed_sessions": list(case.removed_sessions),
            "old_event_id": case.old_event_id,
            "new_event_id": case.new_event_id,
            "old_candidate_id": case.old_candidate_id,
            "new_candidate_id": case.new_candidate_id,
            "official_source_hash": case.official_source_hash,
            "raw_source_hash": case.raw_source_hash,
            "raw_selected_sha256": case.raw_selected_sha256,
            "successor_raw_source_hash": case.successor_raw_source_hash,
        }
        for case in CASES
    ]
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frames=overrides if changed else {},
        summary={
            "status": "validated_offline_plan" if changed else "already_repaired",
            "base_release_version": release.version,
            "target_count": len(CASES),
            "targets": targets,
            "removed_daily_price_rows": (
                len(frames["daily_price_raw"])
                - len(overrides["daily_price_raw"])
                if changed
                else 0
            ),
            "removed_adjustment_factor_rows": factor_rows_removed,
            "factor_source_version": factor_source_version,
            "adjustment_factor_economic_rows_changed": 0,
            "candidate_set_sha256": candidate_hash,
            "resolution_set_sha256": resolution_hash,
            "registry_draft": registry_draft(),
            "registry_inventory_sha256": registry_inventory_sha256(),
            "planned_versions": dict(planned_versions),
            "write_datasets": list(WRITE_DATASETS),
            "source_archive_immutable": True,
            "index_membership_events_unchanged": True,
            "index_constituent_anchors_unchanged": True,
            "snapshot_identity_gap_recorded": snapshot_identity_gap_recorded,
            "snapshot_identity_gap": (
                dict(EXPECTED_SNAPSHOT_IDENTITY_GAP)
                if snapshot_identity_gap_recorded
                else None
            ),
            "non_target_terminal_issues_unchanged": True,
            "target_terminal_issues_after": 0,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


@contextmanager
def _exclusive_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved short-tail recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted short-tail transaction blocks writes: {journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_inputs_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, etag = repository.current_release()
    if (
        release is None
        or release.to_bytes() != prepared.release.to_bytes()
        or etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after short-tail planning.")
    for dataset in REQUIRED_DATASETS:
        pointer, pointer_etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or pointer.version != prepared.release.dataset_versions[dataset]
            or pointer_etag != prepared.pointer_etags[dataset]
        ):
            raise RuntimeError(f"{dataset} pointer changed after short-tail planning.")


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
) -> dict[str, Any]:
    manifest = repository.manifest_for_version(
        dataset, prepared.release.dataset_versions[dataset]
    )
    metadata = dict(manifest.metadata)
    output_versions = dict(prepared.release.dataset_versions)
    output_versions.update(prepared.planned_versions)
    metadata.update(
        {
            "operation": OPERATION,
            "input_release_version": prepared.release.version,
            "output_versions": output_versions,
            "short_terminal_tail_registry": registry_draft(),
            "short_terminal_tail_registry_sha256": registry_inventory_sha256(),
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    lineage = _adjustment_source_version(
        prepared.planned_versions["daily_price_raw"],
        prepared.planned_versions["corporate_actions"],
    )
    if dataset == "daily_price_raw":
        metadata["removed_rows"] = EXPECTED_REMOVED_ROWS
    elif dataset == "adjustment_factors":
        metadata.update(
            {
                "source_version": lineage,
                "source_daily_price_version": prepared.planned_versions[
                    "daily_price_raw"
                ],
                "source_corporate_actions_version": prepared.planned_versions[
                    "corporate_actions"
                ],
                "removed_rows": EXPECTED_REMOVED_ROWS,
                "expected_economic_rows_changed": 0,
            }
        )
    elif dataset == "lifecycle_resolutions":
        metadata.update(
            {
                "candidate_set_sha256": prepared.summary["candidate_set_sha256"],
                "resolution_set_sha256": prepared.summary["resolution_set_sha256"],
                "adjustment_source_version": lineage,
            }
        )
    return metadata


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
    old_versions: Mapping[str, str],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            expected_versions = {**dict(old_versions), **dict(planned_versions)}
            belongs = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or observed.dataset_versions == expected_versions
            if not belongs:
                raise RuntimeError(
                    f"unexpected release during short-tail rollback: {observed.version}"
                )
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            old = old_pointer_bytes[dataset]
            if current.data != old:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected {dataset} pointer during short-tail rollback: {pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository,
    committed: DataRelease,
    *,
    expected_out_of_scope_pointer_etags: Mapping[str, str | None],
) -> None:
    current, _ = repository.current_release()
    if current is None or current.to_bytes() != committed.to_bytes():
        raise RuntimeError("Committed short-tail release is not current.")
    for dataset, version in committed.dataset_versions.items():
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"Applied short-tail pointer mismatch: {dataset}.")
        if (
            dataset not in WRITE_DATASETS
            and etag != expected_out_of_scope_pointer_etags.get(dataset)
        ):
            raise RuntimeError(f"Out-of-scope pointer changed: {dataset}.")
    replay = prepare_repair(repository)
    if replay.summary["status"] != "already_repaired":
        raise RuntimeError("Applied short-tail repair is not idempotent.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        current_plan = prepare_repair(repository)
        if current_plan.summary["status"] == "already_repaired":
            return {
                **current_plan.summary,
                "mode": "apply",
                "writes_performed": False,
            }
        planned = dict(current_plan.planned_versions)
        if set(planned) != set(WRITE_DATASETS) or len(set(planned.values())) != len(
            WRITE_DATASETS
        ):
            raise RuntimeError("Prepared short-tail versions are invalid.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version
                != current_plan.release.dataset_versions[dataset]
                or value.etag != current_plan.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} changed before short-tail apply.")
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_short_terminal_price_tails_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": current_plan.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": planned,
            "registry_inventory_sha256": registry_inventory_sha256(),
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            versions = dict(current_plan.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    current_plan.frames[dataset],
                    completed_session=current_plan.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(
                        repository, current_plan, dataset
                    ),
                    expected_pointer_etag=current_plan.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict or result.manifest.version != planned[dataset]:
                    raise RuntimeError(f"Unexpected {dataset} write result.")
                versions[dataset] = result.manifest.version
                inject(f"after_write:{dataset}")
            for dataset in REQUIRED_DATASETS:
                if dataset in WRITE_DATASETS:
                    continue
                pointer, etag = repository.current_pointer(dataset)
                if (
                    pointer is None
                    or pointer.version
                    != current_plan.release.dataset_versions[dataset]
                    or etag != current_plan.pointer_etags[dataset]
                ):
                    raise RuntimeError(
                        f"Out-of-scope pointer changed during apply: {dataset}."
                    )
            committed = repository.commit_release(
                current_plan.release.completed_session,
                versions,
                quality=current_plan.release.quality,
                warnings=current_plan.release.warnings,
                expected_etag=current_plan.release_etag,
            )
            inject("after_release_commit")
            _validate_candidate_snapshot(repository)
            _assert_applied_release(
                repository,
                committed,
                expected_out_of_scope_pointer_etags=current_plan.pointer_etags,
            )
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": committed.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            return {
                **current_plan.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "transaction_id": transaction_id,
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=committed.version if committed else "",
                old_versions=current_plan.release.dataset_versions,
            )
            journal.update(
                {
                    "status": "rollback_failed" if rollback_errors else "rolled_back",
                    "original_error": f"{type(original).__name__}: {original}",
                    "rollback_errors": list(rollback_errors),
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            if rollback_errors:
                recovery = repository.root / RECOVERY_DIR / f"{transaction_id}.json"
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "Short-tail rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}."
                ) from original
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan six exact US short terminal-price tail repairs offline."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository)
    result = (
        apply_repair(repository, prepared)
        if args.apply
        else {**prepared.summary, "mode": "plan", "writes_performed": False}
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
