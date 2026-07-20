#!/usr/bin/env python3
"""Repair NBL, XLNX, and CXO terminal boundaries without network access.

The immutable EODHD responses contain provider tail rows after the source
securities stopped trading.  NBL and XLNX already have the correct merger
engine date; their post-merger rows must be removed.  CXO traded through the
2021-01-15 close, so its one zero-volume 2021-01-19 row must be removed and
the stock-merger engine date must move from the legal completion date to the
first following XNYS session.

Plan mode is the default.  The implementation has no network, EODHD, or R2
code path.  Apply is explicit and uses the same local CAS/rollback discipline
as the other exact lifecycle repairs.
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
from dataclasses import asdict, dataclass
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
TRANSACTION_DIR = "transactions/us-terminal-price-tails"
RECOVERY_DIR = "recovery/us-terminal-price-tails"
OPERATION = "repair_us_terminal_price_tails"
REPAIR_REVIEWED_AT = "2026-07-18T14:00:00Z"
REPAIRED_IDENTITY_SOURCE = "official_terminal_boundary_repair"
REPAIRED_RESOLUTION_SOURCE = "terminal_boundary_repair"
REPAIRED_REVIEWER = "terminal_boundary_repair_v1"

OLD_IDENTITY_SOURCE = "eodhd_exchange_symbols"
OLD_IDENTITY_URL = "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
OLD_IDENTITY_RETRIEVED_AT = "2026-07-16T15:56:01.033938Z"
OLD_IDENTITY_HASH = (
    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
)
ACTION_SOURCE = "sec_edgar+stored_price_crosscheck"
ACTION_SOURCE_KIND = "official_crosscheck"
RESOLUTION_OLD_SOURCE = "lifecycle_finalizer"
RESOLUTION_OLD_REVIEWER = "us_lifecycle_finalizer_v1"
RESOLUTION_OLD_REVIEWED_AT = "2026-07-18T00:00:00Z"

EVIDENCE_REPORT_HASH = (
    "109813ac49e2be1b05cce11b9f042ae0cf5f23d6c317006ed1a168007768b4f9"
)
EVIDENCE_REPORT_BYTES = 714_105
EVIDENCE_REPORT_URL = (
    "file://results/data_quality/us_lifecycle/"
    "sec_collection_final_pre_finalize.json"
)
EVIDENCE_REPORT_RETRIEVED_AT = "2026-07-18T10:26:52.350026Z"
EXPECTED_CURRENT_CANDIDATE_SET_SHA256 = (
    "399555a7bc7a1a2cf5a922c3d94c19eb7b37dd40b654e8c935fe5b9b4ce5d098"
)
EXPECTED_SNAPSHOT_IDENTITY_GAP = {
    "code": "index_member_missing_active_symbol",
    "row_count": 1,
    "index_id": "sp500",
    "replay_date": "2020-10-07",
    "security_id": "US:EODHD:3dd6d6ce-e7a1-5078-b258-df5b18404c9d",
    "symbol": "NBL",
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
    "reason": "community_removal_lags_official_terminal_identity",
}


@dataclass(frozen=True)
class TerminalTailSpec:
    symbol: str
    security_id: str
    old_exchange: str
    repaired_exchange: str
    old_active_to: str
    last_real_session: str
    legal_completion_date: str
    market_transition_session: str
    old_event_id: str
    new_event_id: str
    old_candidate_id: str
    new_candidate_id: str
    successor_symbol: str
    successor_security_id: str
    ratio: float
    official_source_url: str
    official_source_hash: str
    official_source_bytes: int
    official_retrieved_at: str
    filing_accession_number: str
    filing_acceptance_datetime: str
    official_patterns: tuple[str, ...]
    raw_source_url: str
    raw_source_hash: str
    raw_source_bytes: int
    raw_source_rows: int
    raw_retrieved_at: str
    removed_tail_start: str
    removed_tail_end: str
    removed_tail_count: int
    removed_tail_sha256: str
    terminal_ohlcv: tuple[float, float, float, float, float]
    successor_source_url: str
    successor_source_hash: str
    successor_source_bytes: int
    successor_source_rows: int
    successor_retrieved_at: str
    successor_ohlcv: tuple[float, float, float, float, float]
    report_candidate_active_to: str
    report_candidate_last_price_date: str
    report_crosscheck_old_price_session: str
    report_effective_date: str
    index_removals: tuple[tuple[str, str], ...]


CASES: tuple[TerminalTailSpec, ...] = (
    TerminalTailSpec(
        symbol="NBL",
        security_id="US:EODHD:3dd6d6ce-e7a1-5078-b258-df5b18404c9d",
        old_exchange="NYSE",
        repaired_exchange="NASDAQ",
        old_active_to="2020-10-16",
        last_real_session="2020-10-02",
        legal_completion_date="2020-10-05",
        market_transition_session="2020-10-05",
        old_event_id="dd616ed974225aa84c3b537b9d21ce2a27e13a1f5af8c7e2488b765a007cac31",
        new_event_id="dd616ed974225aa84c3b537b9d21ce2a27e13a1f5af8c7e2488b765a007cac31",
        old_candidate_id="c2f14e962e176ec48993b81250c074de0a22f5ba0b38a666a494e8654a907c32",
        new_candidate_id="ac20b7b59563fe0b7f96a244092df17a3c5de3188d75144ff1c46cfa4ccb7955",
        successor_symbol="CVX",
        successor_security_id="US:EODHD:fe5efe55-952b-5929-9eb0-f61c469314e8",
        ratio=0.1191,
        official_source_url="https://www.sec.gov/Archives/edgar/data/72207/000119312520263378/0001193125-20-263378.txt",
        official_source_hash="fe5554317c372cb8fe924762d304049d0605b2f335ac4a3641bf2c22945ddffc",
        official_source_bytes=277_498,
        official_retrieved_at="2026-07-18T10:30:44.927075Z",
        filing_accession_number="0001193125-20-263378",
        filing_acceptance_datetime="20201005084839",
        official_patterns=(
            r"completion, on October 5, 2020",
            r"suspended from trading.{0,100}?prior to the open of trading on October 5, 2020",
            r"receive 0\.1191 of a share of common stock of Chevron",
        ),
        raw_source_url="https://eodhd.com/api/eod/NBL.US?from=2015-01-01&to=2026-07-15",
        raw_source_hash="f468258431303cd0278595d4457bac8af8e741cb2737cea142f28f1e25d5c5da",
        raw_source_bytes=168_139,
        raw_source_rows=1_459,
        raw_retrieved_at="2026-07-16T15:56:47.970768Z",
        removed_tail_start="2020-10-05",
        removed_tail_end="2020-10-16",
        removed_tail_count=10,
        removed_tail_sha256="43174e67dc9c4faedd3eb34b645f5f394c8dc83f264dfc0697525fc1661dc5ff",
        terminal_ohlcv=(8.14, 8.51, 8.12, 8.46, 13_126_428.0),
        successor_source_url="https://eodhd.com/api/eod/CVX.US?from=2015-01-01&to=2026-07-15",
        successor_source_hash="60bf38eef9fd9c22d5d7fcde27c65a64794a426af67cce7e35c35304485cd4a9",
        successor_source_bytes=345_341,
        successor_source_rows=2_899,
        successor_retrieved_at="2026-07-16T15:58:37.616426Z",
        successor_ohlcv=(71.52, 72.73, 70.71, 72.70, 12_049_800.0),
        report_candidate_active_to="2020-10-16",
        report_candidate_last_price_date="2020-10-16",
        report_crosscheck_old_price_session="2020-10-05",
        report_effective_date="2020-10-05",
        index_removals=(("sp500", "2020-10-12"),),
    ),
    TerminalTailSpec(
        symbol="XLNX",
        security_id="US:EODHD:597fd8aa-aa0e-5109-b7f4-8f9781e5e9a9",
        old_exchange="NASDAQ",
        repaired_exchange="NASDAQ",
        old_active_to="2022-02-16",
        last_real_session="2022-02-11",
        legal_completion_date="2022-02-14",
        market_transition_session="2022-02-14",
        old_event_id="5607a776f99741c085a54e45eddc90282d7d7fe5fe86a3b8cab2350bb7188ca7",
        new_event_id="5607a776f99741c085a54e45eddc90282d7d7fe5fe86a3b8cab2350bb7188ca7",
        old_candidate_id="f73ff16a040088b4d9c5fc9df9c220bc19eb0a458d41f6b9fbfad0c1df00ef9a",
        new_candidate_id="93d51e30eff9d6edf166c175d59a20421be00dc3790335712bd1e3ef758a08db",
        successor_symbol="AMD",
        successor_security_id="US:EODHD:18d4628a-052b-5aba-a4e7-a620d1bd07c7",
        ratio=1.7234,
        official_source_url="https://www.sec.gov/Archives/edgar/data/2488/000000248822000031/0000002488-22-000031.txt",
        official_source_hash="f3fccce84370f2606849a031af574d221dcaed59f283533bff74ae83084bddc1",
        official_source_bytes=431_982,
        official_retrieved_at="2026-07-18T10:30:50.080306Z",
        filing_accession_number="0000002488-22-000031",
        filing_acceptance_datetime="20220214084109",
        official_patterns=(
            r"On February 14, 2022.{0,100}?completed the previously announced acquisition of Xilinx",
            r"Merger became effective on February 14, 2022",
            r"received 1\.7234 shares of AMD common stock",
            r"Xilinx common stock will no longer be listed for trading",
        ),
        raw_source_url="https://eodhd.com/api/eod/XLNX.US?from=2015-01-01&to=2026-07-15",
        raw_source_hash="5e23869bd317191de08bc0fb9c021b9cac03d1a1c0bc2d9e0d93f090888b1fb3",
        raw_source_bytes=210_148,
        raw_source_rows=1_795,
        raw_retrieved_at="2026-07-16T15:57:03.560554Z",
        removed_tail_start="2022-02-14",
        removed_tail_end="2022-02-16",
        removed_tail_count=3,
        removed_tail_sha256="a77056ceded0f8213430f8943774378ec591c47f7efd5236b410acdade52b0ac",
        terminal_ohlcv=(217.27, 218.95, 192.49, 194.92, 35_522_431.0),
        successor_source_url="https://eodhd.com/api/eod/AMD.US?from=2015-01-01&to=2026-07-15",
        successor_source_hash="54782a1e3ea8e551f0774a2b3034d1e5d360324874da50d41e45218a2ba2e66b",
        successor_source_bytes=333_325,
        successor_source_rows=2_899,
        successor_retrieved_at="2026-07-16T15:56:23.117541Z",
        successor_ohlcv=(115.51, 118.37, 113.46, 114.27, 135_146_400.0),
        report_candidate_active_to="2022-02-16",
        report_candidate_last_price_date="2022-02-16",
        report_crosscheck_old_price_session="2022-02-14",
        report_effective_date="2022-02-14",
        index_removals=(("sp500", "2022-02-15"), ("nasdaq100", "2022-02-22")),
    ),
    TerminalTailSpec(
        symbol="CXO",
        security_id="US:EODHD:b32ce08e-4158-5aea-b7ba-6175e716fa41",
        old_exchange="NYSE",
        repaired_exchange="NYSE",
        old_active_to="2021-01-19",
        last_real_session="2021-01-15",
        legal_completion_date="2021-01-15",
        market_transition_session="2021-01-19",
        old_event_id="db752821ea192e7c3ea7ebc90f02f8474b017b053d52334cb0cca7e6a803396b",
        new_event_id="162dee2832998354021e79efcafa8a915d3c2f0744b939c5f570c93deb2f1a55",
        old_candidate_id="3346f9b460655bf4f5dd8405f0b24d23e905456274d5f7f0f24b9723705d2b88",
        new_candidate_id="f326b63d0229f68816a663f468e1723b8925db58b8f21607cb3b16e72cfb531c",
        successor_symbol="COP",
        successor_security_id="US:EODHD:cfbc7973-3e6d-5334-a29f-7d0d83693ae0",
        ratio=1.46,
        official_source_url="https://www.sec.gov/Archives/edgar/data/1163165/000110465921004775/0001104659-21-004775.txt",
        official_source_hash="65047e754c1838fd1c5cee20d03d8941ee6990540b5b2307beb2e9f8839bcc9e",
        official_source_bytes=316_992,
        official_retrieved_at="2026-07-18T10:30:46.496018Z",
        filing_accession_number="0001104659-21-004775",
        filing_acceptance_datetime="20210115165006",
        official_patterns=(
            r"On January 15, 2021.{0,100}?completed its previously announced acquisition of Concho",
            r"right to receive 1\.46.{0,100}?shares of common stock",
        ),
        raw_source_url="https://eodhd.com/api/eod/CXO.US?from=2015-01-01&to=2026-07-15",
        raw_source_hash="59e7e2843948065b50ced2daf2012cab37061643a03e0929426b7507997a64bb",
        raw_source_bytes=180_668,
        raw_source_rows=1_522,
        raw_retrieved_at="2026-07-16T15:57:56.073822Z",
        removed_tail_start="2021-01-19",
        removed_tail_end="2021-01-19",
        removed_tail_count=1,
        removed_tail_sha256="d50b081e7df7a511ac7a3ec685191b6e3c9af09927502da9a045cbfd999bfe17",
        terminal_ohlcv=(69.12, 69.12, 64.60, 65.60, 28_565_330.0),
        successor_source_url="https://eodhd.com/api/eod/COP.US?from=2015-01-01&to=2026-07-15",
        successor_source_hash="4d0dd5898022ca5c32fb548c86be65c6d5ff115e5a7e4532a7ecedafc224841a",
        successor_source_bytes=338_051,
        successor_source_rows=2_899,
        successor_retrieved_at="2026-07-16T15:58:12.348065Z",
        successor_ohlcv=(45.15, 46.15, 44.86, 46.00, 14_498_800.0),
        report_candidate_active_to="2021-01-19",
        report_candidate_last_price_date="2021-01-19",
        report_crosscheck_old_price_session="2021-01-15",
        report_effective_date="2021-01-15",
        index_removals=(("sp500", "2021-01-21"),),
    ),
)


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    planned_versions: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


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
    value = _text(value)
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _number(value: Any) -> float | None:
    if not _text(value):
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _archive_path(completed_session: str, digest: str, suffix: str) -> str:
    return f"archives/{completed_session}/{digest}.{suffix}.gz"


def _safe_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    target = (base / object_path).resolve()
    if target == base or base not in target.parents:
        raise ValueError(f"Terminal-tail evidence path escapes repository: {object_path}.")
    return target


def _normalized_official_text(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    without_tags = re.sub(r"<[^>]+>", " ", decoded)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _static_contract() -> None:
    if len(CASES) != 3 or {case.symbol for case in CASES} != {"NBL", "XLNX", "CXO"}:
        raise RuntimeError("Terminal-tail inventory must contain exactly NBL/XLNX/CXO.")
    for case in CASES:
        expected_old = canonical_lifecycle_event_id(
            case.security_id, "stock_merger", case.legal_completion_date
        )
        expected_new = canonical_lifecycle_event_id(
            case.security_id, "stock_merger", case.market_transition_session
        )
        if expected_old != case.old_event_id or expected_new != case.new_event_id:
            raise RuntimeError(f"{case.symbol} canonical lifecycle event IDs changed.")
        if lifecycle_candidate_id(case.security_id, case.old_active_to) != case.old_candidate_id:
            raise RuntimeError(f"{case.symbol} old candidate ID changed.")
        if lifecycle_candidate_id(case.security_id, case.last_real_session) != case.new_candidate_id:
            raise RuntimeError(f"{case.symbol} repaired candidate ID changed.")
        if pd.Timestamp(case.market_transition_session) <= pd.Timestamp(case.last_real_session):
            raise RuntimeError(f"{case.symbol} transition is not after its final close.")


def registry_draft() -> list[dict[str, Any]]:
    """Return the code-pinned draft for a future publication-gate registry."""

    values: list[dict[str, Any]] = []
    for case in CASES:
        item = {
            "symbol": case.symbol,
            "security_id": case.security_id,
            "old_candidate_id": case.old_candidate_id,
            "candidate_id": case.new_candidate_id,
            "old_event_id": case.old_event_id,
            "event_id": case.new_event_id,
            "action_type": "stock_merger",
            "report_candidate_active_to": case.report_candidate_active_to,
            "report_candidate_last_price_date": case.report_candidate_last_price_date,
            "report_crosscheck_old_price_session": case.report_crosscheck_old_price_session,
            "report_effective_date": case.report_effective_date,
            "official_completion_date": case.legal_completion_date,
            "last_real_session": case.last_real_session,
            "market_transition_session": case.market_transition_session,
            "date_relation": "next_xnys_session_after_terminal_close",
            "new_security_id": case.successor_security_id,
            "new_symbol": case.successor_symbol,
            "ratio": case.ratio,
            "raw_source_url": case.raw_source_url,
            "raw_source_hash": case.raw_source_hash,
            "raw_source_bytes": case.raw_source_bytes,
            "removed_tail_start": case.removed_tail_start,
            "removed_tail_end": case.removed_tail_end,
            "removed_tail_count": case.removed_tail_count,
            "removed_tail_sha256": case.removed_tail_sha256,
            "official_source_url": case.official_source_url,
            "official_source_hash": case.official_source_hash,
            "official_source_bytes": case.official_source_bytes,
            "filing_accession_number": case.filing_accession_number,
            "filing_acceptance_datetime": case.filing_acceptance_datetime,
            "successor_source_hash": case.successor_source_hash,
            "index_removals_observed": [
                {"index_id": index_id, "effective_date": effective_date}
                for index_id, effective_date in case.index_removals
            ],
            "lifecycle_evidence_report_sha256": EVIDENCE_REPORT_HASH,
        }
        item["registry_item_sha256"] = _canonical_json_sha256(item)
        values.append(item)
    return values


def registry_inventory_sha256() -> str:
    return _canonical_json_sha256(registry_draft())


def _archive_row(
    archive: pd.DataFrame,
    *,
    completed_session: str,
    digest: str,
    suffix: str,
    dataset: str,
    content_type: str,
    source_url: str,
    retrieved_at: str,
) -> Mapping[str, Any]:
    expected_path = _archive_path(completed_session, digest, suffix)
    related = (
        archive["archive_id"].astype(str).eq(digest)
        | archive["source_hash"].astype(str).eq(digest)
        | archive["object_path"].astype(str).eq(expected_path)
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
        "object_path": expected_path,
        "content_type": content_type,
        "effective_date": completed_session,
        "source": dataset,
        "retrieved_at": retrieved_at,
        "source_hash": digest,
        "source_url": source_url,
    }
    mismatches = [
        field
        for field, value in expected.items()
        if (
            _date(row.get(field)) != value
            if field == "effective_date"
            else _text(row.get(field)) != value
        )
    ]
    if mismatches:
        raise ValueError(
            f"Exact archive row {digest} changed: {', '.join(mismatches)}."
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


def _verify_report(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
) -> Mapping[str, Any]:
    row = _archive_row(
        archive,
        completed_session=completed_session,
        digest=EVIDENCE_REPORT_HASH,
        suffix="json",
        dataset="lifecycle_evidence_report",
        content_type="application/json",
        source_url=EVIDENCE_REPORT_URL,
        retrieved_at=EVIDENCE_REPORT_RETRIEVED_AT,
    )
    payload = _archive_payload(
        repository,
        row,
        digest=EVIDENCE_REPORT_HASH,
        expected_bytes=EVIDENCE_REPORT_BYTES,
    )
    try:
        report = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Lifecycle evidence report is invalid JSON.") from exc
    records = report.get("records") if isinstance(report, Mapping) else None
    if not isinstance(records, Mapping):
        raise ValueError("Lifecycle evidence report has no exact records inventory.")
    for case in CASES:
        record = records.get(case.security_id)
        if not isinstance(record, Mapping):
            raise ValueError(f"{case.symbol} lifecycle report record is missing.")
        candidate = record.get("candidate")
        parsed = record.get("parsed")
        filing = record.get("filing")
        crosscheck = record.get("crosscheck")
        if not all(
            isinstance(value, Mapping)
            for value in (candidate, parsed, filing, crosscheck)
        ):
            raise ValueError(f"{case.symbol} lifecycle report structure changed.")
        expected_pairs = {
            "candidate_security_id": (
                _text(candidate.get("security_id")),
                case.security_id,
            ),
            "candidate_symbol": (
                _text(candidate.get("symbol")).upper(),
                case.symbol,
            ),
            "candidate_active_to": (
                _date(candidate.get("active_to")),
                case.report_candidate_active_to,
            ),
            "candidate_last_price_date": (
                _date(candidate.get("last_price_date")),
                case.report_candidate_last_price_date,
            ),
            "parsed_action_type": (
                _text(parsed.get("action_type")),
                "stock_merger",
            ),
            "parsed_effective_date": (
                _date(parsed.get("effective_date")),
                case.report_effective_date,
            ),
            "parsed_new_symbol": (
                _text(parsed.get("new_symbol")).upper(),
                case.successor_symbol,
            ),
            "filing_accession": (
                _text(filing.get("accession_number")),
                case.filing_accession_number,
            ),
            "crosscheck_old_price_session": (
                _date(crosscheck.get("old_price_session")),
                case.report_crosscheck_old_price_session,
            ),
            "successor_security_id": (
                _text(record.get("successor_security_id")),
                case.successor_security_id,
            ),
            "source_url": (
                _text(record.get("source_url")),
                case.official_source_url,
            ),
            "source_hash": (
                _text(record.get("source_hash")),
                case.official_source_hash,
            ),
        }
        mismatches = [
            field for field, (observed, expected) in expected_pairs.items()
            if observed != expected
        ]
        if not math.isclose(
            float(_number(parsed.get("ratio")) or 0.0),
            case.ratio,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            mismatches.append("parsed_ratio")
        if mismatches:
            raise ValueError(
                f"{case.symbol} lifecycle report exact fields changed: "
                + ", ".join(mismatches)
                + "."
            )
    return report


def _raw_record(records: list[Mapping[str, Any]], session: str) -> Mapping[str, Any]:
    matches = [record for record in records if _date(record.get("date")) == session]
    if len(matches) != 1:
        raise ValueError(f"Raw EOD inventory does not contain one {session} record.")
    return matches[0]


def _ohlcv_matches(
    row: Mapping[str, Any],
    expected: tuple[float, float, float, float, float],
) -> bool:
    observed = tuple(_number(row.get(field)) for field in ("open", "high", "low", "close", "volume"))
    return all(
        value is not None
        and math.isclose(float(value), float(wanted), rel_tol=0, abs_tol=1e-8)
        for value, wanted in zip(observed, expected, strict=True)
    )


def _verify_case_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    case: TerminalTailSpec,
    *,
    completed_session: str,
) -> dict[str, list[Mapping[str, Any]]]:
    official_row = _archive_row(
        archive,
        completed_session=completed_session,
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
        rf"ACCEPTANCE-DATETIME>?{re.escape(case.filing_acceptance_datetime)}",
        decoded,
        re.I,
    ):
        raise ValueError(f"{case.symbol} SEC acceptance timestamp changed.")
    normalized = _normalized_official_text(official)
    missing = [
        pattern for pattern in case.official_patterns
        if not re.search(pattern, normalized, re.I)
    ]
    if missing:
        raise ValueError(
            f"{case.symbol} SEC payload no longer proves exact completion terms: "
            + ", ".join(missing)
            + "."
        )

    def raw_payload(
        *,
        digest: str,
        source_url: str,
        retrieved_at: str,
        expected_bytes: int,
        expected_rows: int,
    ) -> list[Mapping[str, Any]]:
        row = _archive_row(
            archive,
            completed_session=completed_session,
            digest=digest,
            suffix="json",
            dataset="eodhd_eod",
            content_type="application/json",
            source_url=source_url,
            retrieved_at=retrieved_at,
        )
        payload = _archive_payload(
            repository, row, digest=digest, expected_bytes=expected_bytes
        )
        try:
            values = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{case.symbol} raw EOD payload is invalid JSON.") from exc
        if not isinstance(values, list) or len(values) != expected_rows:
            raise ValueError(
                f"{case.symbol} raw EOD record count changed: {len(values) if isinstance(values, list) else 'invalid'}."
            )
        if not all(isinstance(value, Mapping) for value in values):
            raise ValueError(f"{case.symbol} raw EOD records are not objects.")
        return list(values)

    source_records = raw_payload(
        digest=case.raw_source_hash,
        source_url=case.raw_source_url,
        retrieved_at=case.raw_retrieved_at,
        expected_bytes=case.raw_source_bytes,
        expected_rows=case.raw_source_rows,
    )
    successor_records = raw_payload(
        digest=case.successor_source_hash,
        source_url=case.successor_source_url,
        retrieved_at=case.successor_retrieved_at,
        expected_bytes=case.successor_source_bytes,
        expected_rows=case.successor_source_rows,
    )
    tail = [
        record for record in source_records
        if case.removed_tail_start <= _date(record.get("date")) <= case.removed_tail_end
    ]
    if (
        len(tail) != case.removed_tail_count
        or _date(tail[0].get("date")) != case.removed_tail_start
        or _date(tail[-1].get("date")) != case.removed_tail_end
        or _canonical_json_sha256(tail) != case.removed_tail_sha256
    ):
        raise ValueError(f"{case.symbol} exact removed-tail extraction changed.")
    terminal = _raw_record(source_records, case.last_real_session)
    successor = _raw_record(successor_records, case.market_transition_session)
    if not _ohlcv_matches(terminal, case.terminal_ohlcv):
        raise ValueError(f"{case.symbol} terminal OHLCV changed in raw EOD evidence.")
    if not _ohlcv_matches(successor, case.successor_ohlcv):
        raise ValueError(f"{case.symbol} successor transition OHLCV changed.")
    if any(
        _date(record.get("date")) <= case.last_real_session
        for record in tail
    ):
        raise ValueError(f"{case.symbol} reviewed tail overlaps valid trading history.")
    return {"source": source_records, "successor": successor_records}


def _verify_index_removals(
    membership: pd.DataFrame,
    case: TerminalTailSpec,
) -> None:
    rows = membership.loc[
        membership["security_id"].astype(str).eq(case.security_id)
        & membership["operation"].astype(str).str.upper().eq("REMOVE")
    ]
    observed = {
        (_text(row.index_id).lower(), _date(row.effective_date))
        for row in rows.itertuples(index=False)
    }
    if observed != set(case.index_removals):
        raise ValueError(
            f"{case.symbol} exact community index-removal inventory changed: {sorted(observed)}."
        )


def _price_state(
    prices: pd.DataFrame,
    case: TerminalTailSpec,
    records: list[Mapping[str, Any]],
) -> str:
    rows = prices.loc[prices["security_id"].astype(str).eq(case.security_id)].copy()
    if rows.empty:
        raise ValueError(f"{case.symbol} price inventory is missing.")
    rows["_session"] = pd.to_datetime(rows["session"], errors="coerce").dt.date.astype(str)
    if rows["_session"].eq("NaT").any() or rows["_session"].duplicated().any():
        raise ValueError(f"{case.symbol} price sessions are invalid or duplicated.")
    by_session = {str(row["_session"]): row for row in rows.to_dict("records")}
    raw_by_session = {_date(record.get("date")): record for record in records}
    old_dates = set(raw_by_session)
    removed_dates = {
        date
        for date in old_dates
        if case.removed_tail_start <= date <= case.removed_tail_end
    }
    repaired_dates = old_dates - removed_dates
    actual_dates = set(by_session)
    if actual_dates == old_dates:
        state = "old"
    elif actual_dates == repaired_dates:
        state = "repaired"
    else:
        missing = sorted(old_dates - actual_dates)[:10]
        extra = sorted(actual_dates - old_dates)[:10]
        raise ValueError(
            f"{case.symbol} price inventory is neither exact raw nor exact repaired; "
            f"missing={missing}; extra={extra}."
        )
    for session, row in by_session.items():
        if not _ohlcv_matches(row, tuple(
            float(_number(raw_by_session[session].get(field)) or 0.0)
            for field in ("open", "high", "low", "close", "volume")
        )):
            raise ValueError(f"{case.symbol}/{session} Parquet OHLCV differs from raw archive.")
        expected_pairs = {
            "source": "eodhd_eod",
            "retrieved_at": case.raw_retrieved_at,
            "source_hash": case.raw_source_hash,
            "currency": "USD",
        }
        if any(_text(row.get(field)) != value for field, value in expected_pairs.items()):
            raise ValueError(f"{case.symbol}/{session} Parquet source lineage changed.")
    terminal = by_session.get(case.last_real_session)
    if terminal is None or not _ohlcv_matches(terminal, case.terminal_ohlcv):
        raise ValueError(f"{case.symbol} final real Parquet OHLCV changed.")
    return state


def _verify_successor_price(prices: pd.DataFrame, case: TerminalTailSpec) -> None:
    rows = prices.loc[
        prices["security_id"].astype(str).eq(case.successor_security_id)
        & pd.to_datetime(prices["session"], errors="coerce")
        .dt.date.astype(str)
        .eq(case.market_transition_session)
    ]
    if len(rows) != 1 or not _ohlcv_matches(rows.iloc[0], case.successor_ohlcv):
        raise ValueError(f"{case.symbol} successor is not priceable on transition.")
    row = rows.iloc[0]
    if (
        _text(row.get("source")) != "eodhd_eod"
        or _text(row.get("retrieved_at")) != case.successor_retrieved_at
        or _text(row.get("source_hash")) != case.successor_source_hash
        or _text(row.get("currency")) != "USD"
    ):
        raise ValueError(f"{case.symbol} successor transition lineage changed.")


def _identity_state(
    frame: pd.DataFrame,
    case: TerminalTailSpec,
    *,
    history: bool,
) -> str:
    rows = frame.loc[frame["security_id"].astype(str).eq(case.security_id)]
    if len(rows) != 1:
        raise ValueError(f"{case.symbol} identity row is missing or duplicated.")
    row = rows.iloc[0]
    symbol_field = "symbol" if history else "primary_symbol"
    end_field = "effective_to" if history else "active_to"
    common = _text(row.get(symbol_field)).upper() == case.symbol
    old = bool(
        common
        and _text(row.get("exchange")).upper() == case.old_exchange
        and (
            _date(row.get(end_field)) == ""
            if history
            else _date(row.get(end_field)) == case.old_active_to
        )
        and _text(row.get("source")) == OLD_IDENTITY_SOURCE
        and _text(row.get("source_url")) == OLD_IDENTITY_URL
        and _text(row.get("retrieved_at")) == OLD_IDENTITY_RETRIEVED_AT
        and _text(row.get("source_hash")) == OLD_IDENTITY_HASH
    )
    repaired = bool(
        common
        and _text(row.get("exchange")).upper() == case.repaired_exchange
        and _date(row.get(end_field)) == case.last_real_session
        and _text(row.get("source")) == REPAIRED_IDENTITY_SOURCE
        and _text(row.get("source_url")) == case.official_source_url
        and _text(row.get("retrieved_at")) == REPAIR_REVIEWED_AT
        and _text(row.get("source_hash")) == case.official_source_hash
    )
    if old == repaired:
        raise ValueError(
            f"{case.symbol} {'symbol_history' if history else 'security_master'} "
            "is neither exact old nor exact repaired state."
        )
    return "old" if old else "repaired"


def _action_base(row: Mapping[str, Any], case: TerminalTailSpec) -> bool:
    return bool(
        _text(row.get("security_id")) == case.security_id
        and _text(row.get("action_type")) == "stock_merger"
        and _date(row.get("announcement_date")) == case.legal_completion_date
        and _date(row.get("record_date")) == ""
        and _date(row.get("payment_date")) == ""
        and _number(row.get("cash_amount")) is None
        and _number(row.get("ratio")) is not None
        and math.isclose(float(_number(row.get("ratio"))), case.ratio, rel_tol=0, abs_tol=1e-12)
        and _text(row.get("currency")) == "USD"
        and _text(row.get("new_security_id")) == case.successor_security_id
        and _text(row.get("new_symbol")).upper() == case.successor_symbol
        and _text(row.get("official")).lower() == "true"
        and _text(row.get("source_url")) == case.official_source_url
        and _text(row.get("source_kind")) == ACTION_SOURCE_KIND
        and _text(row.get("source")) == ACTION_SOURCE
        and _text(row.get("retrieved_at")) == case.official_retrieved_at
        and _text(row.get("source_hash")) == case.official_source_hash
        and _text(row.get("metadata")) == ""
    )


def _action_state(actions: pd.DataFrame, case: TerminalTailSpec) -> str:
    rows = actions.loc[
        actions["security_id"].astype(str).eq(case.security_id)
        & actions["action_type"].astype(str).eq("stock_merger")
    ]
    if len(rows) != 1 or not _action_base(rows.iloc[0], case):
        raise ValueError(f"{case.symbol} stock-merger action terms changed.")
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
    if old and repaired:
        return "neutral"
    if old == repaired:
        raise ValueError(f"{case.symbol} action is neither exact old nor repaired state.")
    return "old" if old else "repaired"


def _resolution_state(resolutions: pd.DataFrame, case: TerminalTailSpec) -> str:
    rows = resolutions.loc[resolutions["security_id"].astype(str).eq(case.security_id)]
    if len(rows) != 1:
        raise ValueError(f"{case.symbol} lifecycle resolution is missing or duplicated.")
    row = rows.iloc[0]
    common = bool(
        _text(row.get("symbol")).upper() == case.symbol
        and _text(row.get("resolution")) == "applied"
        and _text(row.get("exception_code")) == ""
        and _text(row.get("exception_reason")) == ""
        and _text(row.get("recheck_after")) == ""
        and _text(row.get("successor_security_id")) == case.successor_security_id
        and _text(row.get("successor_symbol")).upper() == case.successor_symbol
        and _text(row.get("source_url")) == case.official_source_url
        and _text(row.get("source_hash")) == case.official_source_hash
    )
    old = bool(
        common
        and _text(row.get("candidate_id")) == case.old_candidate_id
        and _date(row.get("last_price_date")) == case.old_active_to
        and _text(row.get("event_id")) == case.old_event_id
        and _text(row.get("reviewed_by")) == RESOLUTION_OLD_REVIEWER
        and _text(row.get("reviewed_at")) == RESOLUTION_OLD_REVIEWED_AT
        and _text(row.get("source")) == RESOLUTION_OLD_SOURCE
        and _text(row.get("retrieved_at")) == case.official_retrieved_at
    )
    repaired = bool(
        common
        and _text(row.get("candidate_id")) == case.new_candidate_id
        and _date(row.get("last_price_date")) == case.last_real_session
        and _text(row.get("event_id")) == case.new_event_id
        and _text(row.get("reviewed_by")) == REPAIRED_REVIEWER
        and _text(row.get("reviewed_at")) == REPAIR_REVIEWED_AT
        and _text(row.get("source")) == REPAIRED_RESOLUTION_SOURCE
        and _text(row.get("retrieved_at")) == REPAIR_REVIEWED_AT
    )
    if old == repaired:
        raise ValueError(f"{case.symbol} resolution is neither exact old nor repaired state.")
    return "old" if old else "repaired"


def _case_state(
    *,
    price_state: str,
    master_state: str,
    history_state: str,
    action_state: str,
    resolution_state: str,
    symbol: str,
) -> str:
    states = {price_state, master_state, history_state, resolution_state}
    if action_state != "neutral":
        states.add(action_state)
    if len(states) != 1:
        raise RuntimeError(
            f"{symbol} terminal-tail repair is partially applied: {sorted(states)}."
        )
    return next(iter(states))


def _rewrite_prices(prices: pd.DataFrame) -> pd.DataFrame:
    output = prices.copy(deep=True)
    remove = pd.Series(False, index=output.index)
    sessions = pd.to_datetime(output["session"], errors="coerce").dt.date.astype(str)
    for case in CASES:
        remove |= (
            output["security_id"].astype(str).eq(case.security_id)
            & sessions.ge(case.removed_tail_start)
            & sessions.le(case.removed_tail_end)
        )
    return output.loc[~remove].reset_index(drop=True)


def _rewrite_identity(
    frame: pd.DataFrame,
    *,
    history: bool,
) -> pd.DataFrame:
    output = frame.copy(deep=True)
    end_field = "effective_to" if history else "active_to"
    for case in CASES:
        rows = output["security_id"].astype(str).eq(case.security_id)
        updates = {
            end_field: case.last_real_session,
            "exchange": case.repaired_exchange,
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
            & output["action_type"].astype(str).eq("stock_merger")
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
        raise RuntimeError("Terminal-tail factor lineage requires exact input versions.")
    return f"{price_version}+{action_version}"


def _new_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"terminal-price-tails-{session}-{token}-{dataset}"
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
        prices,
        actions,
        source_version=source_version,
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
            "session": pd.to_datetime(prices["session"], errors="raise").dt.normalize(),
        }
    )
    if values.duplicated(["security_id", "session"]).any():
        raise ValueError("Daily-price keys are duplicated.")
    return set(values.itertuples(index=False, name=None))


def _expected_removed_factor_keys() -> set[tuple[str, pd.Timestamp]]:
    output: set[tuple[str, pd.Timestamp]] = set()
    for case in CASES:
        for session in pd.date_range(case.removed_tail_start, case.removed_tail_end):
            output.add((case.security_id, session.normalize()))
    return output


def _factor_economics_equal(
    current: pd.DataFrame,
    expected: pd.DataFrame,
) -> int:
    left = _normalized_factor_values(current)
    right = _normalized_factor_values(expected)
    if len(left) != len(right) or not left[["security_id", "session"]].equals(
        right[["security_id", "session"]]
    ):
        raise ValueError("Retained adjustment-factor key inventory changed.")
    changed = np.zeros(len(left), dtype=bool)
    for column in ("split_factor", "total_return_factor"):
        old = pd.to_numeric(left[column], errors="raise").to_numpy(dtype=float)
        new = pd.to_numeric(right[column], errors="raise").to_numpy(dtype=float)
        changed |= ~((old == new) | (np.isnan(old) & np.isnan(new)))
    count = int(changed.sum())
    if count:
        sample = left.loc[changed, ["security_id", "session"]].head(10)
        raise ValueError(
            "Terminal-tail removal unexpectedly changes retained factor economics: "
            + json.dumps(sample.to_dict("records"), default=str, sort_keys=True)
        )
    return count


def _prepare_factors(
    current: pd.DataFrame,
    old_prices: pd.DataFrame,
    new_prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
    repaired_state: bool,
) -> tuple[pd.DataFrame, int, int]:
    old_price_keys = _price_keys(old_prices)
    current_values = _normalized_factor_values(current)
    current_keys = set(
        current_values[["security_id", "session"]].itertuples(index=False, name=None)
    )
    if repaired_state:
        if current_keys != old_price_keys:
            raise ValueError("Repaired adjustment factors do not match repaired prices.")
    elif current_keys != old_price_keys:
        raise ValueError("Original adjustment factors do not match original prices.")

    expected = _expected_factors(
        new_prices,
        actions,
        source_version=source_version,
        columns=list(current.columns),
    )
    expected_keys = set(
        _normalized_factor_values(expected)[["security_id", "session"]]
        .itertuples(index=False, name=None)
    )
    new_price_keys = _price_keys(new_prices)
    if expected_keys != new_price_keys:
        raise ValueError("Rebuilt adjustment factors do not exactly cover repaired prices.")

    if repaired_state:
        removed_keys: set[tuple[str, pd.Timestamp]] = set()
        retained = current
    else:
        removed_keys = current_keys - expected_keys
        wanted_removed = _expected_removed_factor_keys()
        # The reviewed ranges contain only exchange sessions in the archived
        # source payloads, so the range-derived key set is exact here.
        wanted_removed &= current_keys
        if removed_keys != wanted_removed or len(removed_keys) != sum(
            case.removed_tail_count for case in CASES
        ):
            raise ValueError(
                "Adjustment-factor removals are not the exact reviewed 14-row tail."
            )
        mask = [
            (str(row.security_id), pd.Timestamp(row.session).normalize())
            not in removed_keys
            for row in current.itertuples(index=False)
        ]
        retained = current.loc[mask].reset_index(drop=True)
    economic_changed = _factor_economics_equal(retained, expected)

    exact_lineage = bool(
        set(current["source_version"].astype(str)) == {source_version}
        and set(current["source_hash"].astype(str)) == {source_version}
        and set(current["source"].astype(str)) == {"derived"}
        and set(current["calculated_at"].astype(str)) == {REPAIR_REVIEWED_AT}
        and set(current["retrieved_at"].astype(str)) == {REPAIR_REVIEWED_AT}
    )
    if repaired_state and not exact_lineage:
        raise RuntimeError("Repaired factors have stale or non-exact provenance.")
    return expected, len(removed_keys), economic_changed


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
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}.")
        output[dataset] = etag
    return output


def _terminal_report(
    frames: Mapping[str, pd.DataFrame],
    *,
    release_version: str,
):
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


def _target_issues(report) -> tuple[Any, ...]:
    target_ids = {case.security_id for case in CASES}
    return tuple(issue for issue in report.issues if issue.security_id in target_ids)


def _non_target_issue_fingerprint(report) -> tuple[str, ...]:
    target_ids = {case.security_id for case in CASES}
    return tuple(
        sorted(
            json.dumps(issue.to_dict(), sort_keys=True)
            for issue in report.issues
            if issue.security_id not in target_ids
        )
    )


def _verify_lifecycle_manifest(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> None:
    manifest = repository.manifest_for_version(
        "lifecycle_resolutions",
        release.dataset_versions["lifecycle_resolutions"],
    )
    if _text(manifest.metadata.get("evidence_report_sha256")) != EVIDENCE_REPORT_HASH:
        raise ValueError("Lifecycle manifest is not pinned to the reviewed evidence report.")


def _validate_candidate_snapshot(repository: _CandidateRepository) -> bool:
    """Validate the snapshot, pinning the one known stale-community replay gap.

    NBL's official terminal identity ends on 2020-10-02, while the community
    S&P 500 REMOVE is dated 2020-10-12.  An unrelated S&P event on 2020-10-07
    makes the generic replay validator ask for an active NBL symbol on that
    intermediate date.  Preserving the official identity boundary is the
    reviewed outcome, so this exact one-row validator finding is recorded and
    permitted; any additional or different finding remains blocking.
    """

    report = validate_repository_snapshot(repository)
    errors = tuple(issue for issue in report.issues if issue.severity == "error")
    if not errors:
        return False
    expected_fingerprint = index_member_identity_gap_fingerprint(
        index_id=EXPECTED_SNAPSHOT_IDENTITY_GAP["index_id"],
        replay_date=EXPECTED_SNAPSHOT_IDENTITY_GAP["replay_date"],
        security_id=EXPECTED_SNAPSHOT_IDENTITY_GAP["security_id"],
        next_remove_event_id=EXPECTED_SNAPSHOT_IDENTITY_GAP[
            "next_remove_event_id"
        ],
        next_remove_effective_date=EXPECTED_SNAPSHOT_IDENTITY_GAP[
            "next_remove_effective_date"
        ],
        next_remove_source=EXPECTED_SNAPSHOT_IDENTITY_GAP[
            "next_remove_source"
        ],
        next_remove_source_hash=EXPECTED_SNAPSHOT_IDENTITY_GAP[
            "next_remove_source_hash"
        ],
    )
    if expected_fingerprint != EXPECTED_SNAPSHOT_IDENTITY_GAP["fingerprint"]:
        raise AssertionError("Code-pinned NBL identity-gap fingerprint changed.")
    if not (
        len(errors) == 1
        and errors[0].code == EXPECTED_SNAPSHOT_IDENTITY_GAP["code"]
        and errors[0].row_count == EXPECTED_SNAPSHOT_IDENTITY_GAP["row_count"]
        and tuple(errors[0].fingerprints) == (expected_fingerprint,)
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
            "Current release lacks terminal-tail repair datasets: " + ", ".join(missing)
        )
    pointer_etags = _pointer_etags(repository, release)
    _verify_lifecycle_manifest(repository, release)
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    _verify_report(
        repository,
        frames["source_archive"],
        completed_session=release.completed_session,
    )

    evidence: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    states: dict[str, str] = {}
    for case in CASES:
        evidence[case.symbol] = _verify_case_evidence(
            repository,
            frames["source_archive"],
            case,
            completed_session=release.completed_session,
        )
        _verify_index_removals(frames["index_membership_events"], case)
        _verify_successor_price(frames["daily_price_raw"], case)
        states[case.symbol] = _case_state(
            price_state=_price_state(
                frames["daily_price_raw"],
                case,
                evidence[case.symbol]["source"],
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
            symbol=case.symbol,
        )
    if len(set(states.values())) != 1:
        raise RuntimeError(f"Terminal-tail repair is partially applied: {states}.")
    state = next(iter(states.values()))

    before_report = _terminal_report(frames, release_version=release.version)
    before_target = _target_issues(before_report)
    if state == "old":
        if len(before_target) != 3 or {
            issue.code for issue in before_target
        } != {"source_reentry_after_terminal_action"}:
            raise RuntimeError(
                "Original target terminal-readiness inventory is not exact: "
                + json.dumps([issue.to_dict() for issue in before_target], sort_keys=True)
            )
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
        factor_source_version = _adjustment_source_version(
            planned_versions["daily_price_raw"],
            planned_versions["corporate_actions"],
        )
        repaired_state = False
    elif state == "repaired":
        if before_target:
            raise RuntimeError("Repaired target rows still fail terminal readiness.")
        planned_versions = {}
        overrides = {
            dataset: frames[dataset]
            for dataset in WRITE_DATASETS
            if dataset != "adjustment_factors"
        }
        factor_source_version = _adjustment_source_version(
            release.dataset_versions["daily_price_raw"],
            release.dataset_versions["corporate_actions"],
        )
        repaired_state = True
    else:  # pragma: no cover - guarded by _case_state
        raise AssertionError(f"Unknown terminal-tail repair state: {state}.")

    factors, removed_factor_rows, economic_changed = _prepare_factors(
        frames["adjustment_factors"],
        frames["daily_price_raw"],
        overrides["daily_price_raw"],
        overrides["corporate_actions"],
        source_version=factor_source_version,
        repaired_state=repaired_state,
    )
    overrides["adjustment_factors"] = factors
    if economic_changed:
        raise AssertionError("Terminal-tail factor economics changed.")
    if state == "old" and removed_factor_rows != 14:
        raise AssertionError("Terminal-tail factor removal count is not 14.")

    for dataset in WRITE_DATASETS:
        validate_dataset(
            dataset,
            overrides[dataset],
            completed_session=release.completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    candidate_repository = _CandidateRepository(
        repository,
        release.dataset_versions,
        overrides,
    )
    snapshot_identity_gap_recorded = _validate_candidate_snapshot(
        candidate_repository
    )
    after_frames = dict(frames)
    after_frames.update(overrides)
    after_report = _terminal_report(
        after_frames,
        release_version=f"{release.version}:terminal-tail-plan",
    )
    after_target = _target_issues(after_report)
    if after_target:
        raise RuntimeError(
            "Repaired target rows still fail terminal readiness: "
            + json.dumps([issue.to_dict() for issue in after_target], sort_keys=True)
        )
    if state == "old":
        if len(before_report.issues) - len(after_report.issues) != 3:
            raise RuntimeError("Terminal-tail repair did not remove exactly three issues.")
        if _non_target_issue_fingerprint(before_report) != _non_target_issue_fingerprint(
            after_report
        ):
            raise RuntimeError("Terminal-tail repair changed unrelated readiness issues.")

    candidate_set_sha256 = lifecycle_candidate_set_sha256(
        overrides["lifecycle_resolutions"][["security_id", "last_price_date"]]
    )
    resolution_set_sha256 = lifecycle_resolution_set_sha256(
        overrides["lifecycle_resolutions"]
    )
    if (
        release.version == "20260715-20260718T122234681534Z"
        and candidate_set_sha256 != EXPECTED_CURRENT_CANDIDATE_SET_SHA256
    ):
        raise RuntimeError("Exact current-release candidate-set hash changed.")

    target_summary = []
    for case in CASES:
        target_summary.append(
            {
                "symbol": case.symbol,
                "security_id": case.security_id,
                "old_event_id": case.old_event_id,
                "new_event_id": case.new_event_id,
                "old_candidate_id": case.old_candidate_id,
                "new_candidate_id": case.new_candidate_id,
                "last_real_session": case.last_real_session,
                "legal_completion_date": case.legal_completion_date,
                "market_transition_session": case.market_transition_session,
                "removed_tail_start": case.removed_tail_start,
                "removed_tail_end": case.removed_tail_end,
                "removed_tail_count": case.removed_tail_count,
                "removed_tail_sha256": case.removed_tail_sha256,
                "official_source_hash": case.official_source_hash,
                "raw_source_hash": case.raw_source_hash,
                "index_removals": [
                    {"index_id": index_id, "effective_date": effective_date}
                    for index_id, effective_date in case.index_removals
                ],
            }
        )
    changed = state == "old"
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frames=overrides if changed else {},
        summary={
            "status": "validated_offline_plan" if changed else "already_repaired",
            "base_release_version": release.version,
            "targets": target_summary,
            "target_count": len(CASES),
            "removed_daily_price_rows": (
                len(frames["daily_price_raw"])
                - len(overrides["daily_price_raw"])
                if changed
                else 0
            ),
            "daily_price_rows_before": len(frames["daily_price_raw"]),
            "daily_price_rows_after": len(overrides["daily_price_raw"]),
            "removed_adjustment_factor_rows": removed_factor_rows,
            "adjustment_factor_rows_before": len(frames["adjustment_factors"]),
            "adjustment_factor_rows_after": len(factors),
            "adjustment_factor_economic_rows_changed": 0,
            "adjustment_factor_provenance_rows_rebound": len(factors) if changed else 0,
            "factor_source_version": factor_source_version,
            "terminal_issues_before_total": len(before_report.issues),
            "terminal_issues_after_total": len(after_report.issues),
            "target_terminal_issues_before": len(before_target),
            "target_terminal_issues_after": len(after_target),
            "target_terminal_issue_codes_before": sorted(
                issue.code for issue in before_target
            ),
            "target_terminal_issue_codes_after": sorted(
                issue.code for issue in after_target
            ),
            "non_target_terminal_issues_unchanged": (
                _non_target_issue_fingerprint(before_report)
                == _non_target_issue_fingerprint(after_report)
            ),
            "candidate_set_sha256": candidate_set_sha256,
            "resolution_set_sha256": resolution_set_sha256,
            "lifecycle_evidence_report_sha256": EVIDENCE_REPORT_HASH,
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
            "other_dataset_versions_unchanged": True,
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
            raise RuntimeError("Unresolved terminal-tail recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted terminal-tail transaction blocks writes: {journal}."
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
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    release, release_etag = repository.current_release()
    if (
        release is None
        or release.to_bytes() != prepared.release.to_bytes()
        or release_etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after terminal-tail planning.")
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or pointer.version != prepared.release.dataset_versions[dataset]
            or etag != prepared.pointer_etags[dataset]
        ):
            raise RuntimeError(
                f"{dataset} pointer changed after terminal-tail planning."
            )


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
) -> dict[str, Any]:
    current = repository.manifest_for_version(
        dataset,
        prepared.release.dataset_versions[dataset],
    )
    metadata = dict(current.metadata)
    input_versions = dict(prepared.release.dataset_versions)
    output_versions = dict(input_versions)
    output_versions.update(prepared.planned_versions)
    metadata.update(
        {
            "operation": OPERATION,
            "input_release_version": prepared.release.version,
            "input_versions": input_versions,
            "output_versions": output_versions,
            "terminal_tail_symbols": [case.symbol for case in CASES],
            "terminal_tail_removed_rows": {
                case.symbol: case.removed_tail_count for case in CASES
            },
            "terminal_tail_sha256": {
                case.symbol: case.removed_tail_sha256 for case in CASES
            },
            "terminal_tail_registry_draft": registry_draft(),
            "terminal_tail_registry_inventory_sha256": registry_inventory_sha256(),
            "lifecycle_evidence_report_sha256": EVIDENCE_REPORT_HASH,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    factor_lineage = _adjustment_source_version(
        prepared.planned_versions["daily_price_raw"],
        prepared.planned_versions["corporate_actions"],
    )
    if dataset == "daily_price_raw":
        metadata.update(
            {
                "removed_rows": 14,
                "removed_source_archive_hashes": {
                    case.symbol: case.raw_source_hash for case in CASES
                },
                "removed_tail_hashes": {
                    case.symbol: case.removed_tail_sha256 for case in CASES
                },
            }
        )
    elif dataset == "adjustment_factors":
        factors = prepared.frames["adjustment_factors"]
        if (
            set(factors["source_version"].astype(str)) != {factor_lineage}
            or set(factors["source_hash"].astype(str)) != {factor_lineage}
            or set(factors["source"].astype(str)) != {"derived"}
            or set(factors["calculated_at"].astype(str)) != {REPAIR_REVIEWED_AT}
            or set(factors["retrieved_at"].astype(str)) != {REPAIR_REVIEWED_AT}
        ):
            raise RuntimeError("Prepared terminal-tail factors have stale lineage.")
        metadata.update(
            {
                "source_version": factor_lineage,
                "source_daily_price_version": prepared.planned_versions[
                    "daily_price_raw"
                ],
                "source_corporate_actions_version": prepared.planned_versions[
                    "corporate_actions"
                ],
                "removed_rows": 14,
                "expected_economic_rows_changed": 0,
            }
        )
    elif dataset == "lifecycle_resolutions":
        metadata.update(
            {
                "candidate_set_sha256": prepared.summary["candidate_set_sha256"],
                "resolution_set_sha256": prepared.summary["resolution_set_sha256"],
                "adjustment_source_version": factor_lineage,
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
                    "unexpected release during terminal-tail rollback: "
                    f"{observed.version}"
                )
            repository.objects.put(
                "releases/current.json",
                old_release_bytes,
                if_match=current.etag,
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
                        f"unexpected {dataset} pointer during terminal-tail rollback: "
                        f"{pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
    *,
    expected_candidate_set_sha256: str,
    expected_out_of_scope_pointer_etags: Mapping[str, str | None],
) -> None:
    current, _ = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes():
        raise RuntimeError("Committed terminal-tail release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(
                f"Applied terminal-tail release pointer mismatch: {dataset}."
            )
        if (
            dataset not in WRITE_DATASETS
            and etag != expected_out_of_scope_pointer_etags.get(dataset)
        ):
            raise RuntimeError(
                f"Applied terminal-tail out-of-scope pointer changed: {dataset}."
            )
    price_version = release.dataset_versions["daily_price_raw"]
    action_version = release.dataset_versions["corporate_actions"]
    factor_version = release.dataset_versions["adjustment_factors"]
    lineage = _adjustment_source_version(price_version, action_version)
    factor_manifest = repository.manifest_for_version(
        "adjustment_factors", factor_version
    )
    if any(
        _text(factor_manifest.metadata.get(key)) != value
        for key, value in {
            "source_version": lineage,
            "source_daily_price_version": price_version,
            "source_corporate_actions_version": action_version,
        }.items()
    ):
        raise RuntimeError("Terminal-tail factor manifest lineage is not release-exact.")
    lifecycle_manifest = repository.manifest_for_version(
        "lifecycle_resolutions",
        release.dataset_versions["lifecycle_resolutions"],
    )
    if (
        _text(lifecycle_manifest.metadata.get("candidate_set_sha256"))
        != expected_candidate_set_sha256
        or _text(lifecycle_manifest.metadata.get("evidence_report_sha256"))
        != EVIDENCE_REPORT_HASH
    ):
        raise RuntimeError("Terminal-tail lifecycle manifest hashes are not exact.")
    replay = prepare_repair(repository)
    if replay.summary["status"] != "already_repaired":
        raise RuntimeError("Terminal-tail repair is not idempotent.")
    if replay.summary["target_terminal_issues_after"] != 0:
        raise RuntimeError("Applied terminal-tail targets are not ready.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_lock(repository):
        # The caller-owned plan contains mutable DataFrames and mappings even
        # though PreparedRepair itself is frozen.  Use it only to prove that
        # the repository is still on the exact base release/pointers observed
        # during planning, then rebuild and exclusively use a fresh plan while
        # holding the shared market-store writer lock.  Planned versions are
        # intentionally regenerated because they contain a per-attempt UUID.
        _assert_inputs_unchanged(repository, prepared)
        current_plan = prepare_repair(repository)
        if current_plan.summary["status"] == "already_repaired":
            return {
                **current_plan.summary,
                "mode": "apply",
                "writes_performed": False,
            }
        archive = repository.read_frame(
            "source_archive",
            current_plan.release.dataset_versions["source_archive"],
        )
        _verify_report(
            repository,
            archive,
            completed_session=current_plan.release.completed_session,
        )
        for case in CASES:
            _verify_case_evidence(
                repository,
                archive,
                case,
                completed_session=current_plan.release.completed_session,
            )
        planned = dict(current_plan.planned_versions)
        if set(planned) != set(WRITE_DATASETS) or len(set(planned.values())) != len(
            WRITE_DATASETS
        ):
            raise RuntimeError("Prepared terminal-tail repair has invalid versions.")
        old_release = repository.objects.get("releases/current.json")
        if old_release.etag != current_plan.release_etag:
            raise RuntimeError("Release CAS changed during locked terminal-tail planning.")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != current_plan.release.dataset_versions[dataset]
                or value.etag != current_plan.pointer_etags[dataset]
            ):
                raise RuntimeError(
                    f"{dataset} pointer changed before terminal-tail apply."
                )
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_terminal_price_tails_transaction/v1",
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
                    metadata=_metadata_for_write(repository, current_plan, dataset),
                    expected_pointer_etag=current_plan.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                if result.manifest.version != planned[dataset]:
                    raise RuntimeError(f"Unexpected {dataset} version was written.")
                versions[dataset] = result.manifest.version
                inject(f"after_write:{dataset}")
            if any(
                versions.get(dataset) != version
                for dataset, version in current_plan.release.dataset_versions.items()
                if dataset not in WRITE_DATASETS
            ):
                raise RuntimeError(
                    "Terminal-tail repair changed an out-of-scope dataset version."
                )
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
                expected_candidate_set_sha256=str(
                    current_plan.summary["candidate_set_sha256"]
                ),
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
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
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
                    "Terminal-tail rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair NBL/XLNX/CXO terminal provider tails offline."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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
