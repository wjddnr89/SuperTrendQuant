#!/usr/bin/env python3
"""Plan or atomically canonicalize the 2019 SYMC -> NLOK identity.

The current release contains two EODHD security IDs for one issuer.  NLOK's
ID already owns the complete 2015--2022 provider price/action history, while
the SYMC ID duplicates 1,455 price/factor rows and causes ticker look-ahead in
the 2015 Nasdaq-100 anchor.  This offline repair makes NLOK's ID canonical,
installs exact SYMC/NLOK symbol intervals, removes the redundant old-ID data,
rebinds the S&P anchor, and collapses the redundant 2019-11-05 index swap.

Plan is the default.  There is no network, EODHD or R2 code path.  Apply is
protected by one writer lock, release and dataset-pointer CAS, a durable
rollback journal, exact repaired-state replay, and code-pinned source/data
projections.  It never creates a terminal resolution for this same-security
ticker continuation.
"""

from __future__ import annotations

import argparse
import base64
import gc
import gzip
import fcntl
import json
import os
import shutil
import sys
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_us_identity_tail_repairs as audit  # noqa: E402
import finalize_us_lifecycle_coverage as lifecycle_finalizer  # noqa: E402
import repair_us_identity_price_tails as identity_tails  # noqa: E402
from supertrend_quant.market_store.lifecycle import (  # noqa: E402
    build_lifecycle_candidates,
    canonical_lifecycle_event_id,
)
from supertrend_quant.market_store.lifecycle_coverage import (  # noqa: E402
    LifecycleCoverageReport,
    validate_lifecycle_coverage,
)
from supertrend_quant.market_store.lifecycle_report_provenance import (  # noqa: E402
    REPORT_BINDING_FIELDS,
    build_lifecycle_report_binding,
    validate_lifecycle_report_binding,
)
from supertrend_quant.market_store.manifest import (  # noqa: E402
    CurrentPointer,
    DataRelease,
    DatasetManifest,
    ManifestFile,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.models import DataQuality  # noqa: E402
from supertrend_quant.market_store.official_lifecycle_evidence import (  # noqa: E402
    include_bound_official_applied_event_candidates,
    load_official_lifecycle_exception_evidence,
)
from supertrend_quant.market_store.repository import (  # noqa: E402
    DatasetWriteResult,
    LocalDatasetRepository,
)
from supertrend_quant.market_store.schemas import dataset_spec  # noqa: E402
from supertrend_quant.market_store.storage import (  # noqa: E402
    ConditionalWriteFailed,
    ObjectNotFound,
)
from supertrend_quant.market_store.symc_nlok_identity import (  # noqa: E402
    CANONICAL_EVENT_ID,
    CANONICAL_SECURITY_ID,
    CANONICAL_SYMBOL,
    CANONICAL_SYMBOL_FROM,
    CANONICAL_SYMBOL_TO,
    GEN_SECURITY_ID,
    NLOK_TO_GEN_EVENT_ID,
    OLD_CANDIDATE_ID,
    OLD_EVENT_ID,
    OLD_SECURITY_ID,
    OLD_SYMBOL,
    OLD_SYMBOL_FROM,
    OLD_SYMBOL_TO,
    OFFICIAL_SOURCE_HASH,
    OFFICIAL_SOURCE_URL,
    REVIEWED_NONTERMINAL_EXTRACTION_SHA256,
    SP500_SWAP_ADD_EVENT_ID,
    SP500_SWAP_REMOVE_EVENT_ID,
    TRANSITION_DATE,
    reviewed_nonterminal_extraction,
    reviewed_same_sid_no_data_spec,
)
from supertrend_quant.market_store.validation import (  # noqa: E402
    ValidationReport,
    validate_dataset,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
OPERATION = "repair_us_symc_nlok_identity"
REPAIR_SCHEMA = "us_symc_nlok_identity_repair/v1"
TRANSACTION_DIR = "transactions/us-symc-nlok-identity"
RECOVERY_DIR = "recovery/us-symc-nlok-identity"
TRANSACTION_SCHEMA = "us_symc_nlok_identity_transaction/v1"
IDENTITY_WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "lifecycle_resolutions",
    "index_constituent_anchors",
    "index_membership_events",
)
WRITE_DATASETS = (*IDENTITY_WRITE_DATASETS, "source_archive")
REQUIRED_DATASETS = WRITE_DATASETS
HEAVY_WRITE_DATASETS = ("daily_price_raw", "adjustment_factors")
LIFECYCLE_CANDIDATE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "lifecycle_resolutions",
    "index_constituent_anchors",
    "index_membership_events",
    "source_archive",
)

PINNED_RELEASE_VERSION = audit.PINNED_RELEASE_VERSION
PINNED_DATASET_VERSIONS = dict(audit.PINNED_DATASET_VERSIONS)
PINNED_DATASET_VERSIONS["lifecycle_resolutions"] = (
    "short-terminal-tails-20260715-fa0f2cfd8b2248c1b8315cdbe0167610-"
    "lifecycle_resolutions"
)

OLD_PRICE_ROWS = 1_455
CANONICAL_PRICE_ROWS = 1_977
OLD_PRICE_SHA256 = "bc3878bbd989b7c2f2d3307a66e93f221e050ac1d14bb507b0354fcdd5160d7d"
CANONICAL_PRICE_SHA256 = (
    "77f154621cf2f40ca3a8d7a8ace36c06c27fd522845a2ff9058f6212bd184d44"
)
BASELINE_IDENTITY_GAP_FINGERPRINT = (
    "989c5d44ef1b8cf8a682d807b63a62ebe3c3f38eb6f57e6314b3fe381d5c7d04"
)
BASE_LIFECYCLE_EVIDENCE_REPORT_SHA256 = (
    "f5bc93126819503178cfb26a420173a06b63807bbb4ed9158f46cf6668c99390"
)
BASE_LIFECYCLE_EVIDENCE_REPORT_BYTES = 742_455
CURRENT_LIFECYCLE_REPORT_PATH = Path(
    "results/data_quality/us_lifecycle/sec_collection_20260719_current_release.json"
)
CURRENT_LIFECYCLE_REPORT_SHA256 = (
    "14845368ca771dbea5bc82e8fed4077152fd2c92b424059961e02239188c45ae"
)
CURRENT_LIFECYCLE_REPORT_BYTES = 739_698
LEGACY_LIFECYCLE_HINTS_SHA256 = (
    "2dc55661d72c9f993cd233339ad14858ff44b86d8ceca3d3bb33059fc30369e6"
)
CURRENT_LIFECYCLE_HINTS_SHA256 = (
    "3fd60e760466ffba8b00c6ea191944dc4c00ceb21199f0bb45e96f0d5d969866"
)
FRESH_REPORT_SOURCE_URL = (
    "generated://repair_us_symc_nlok_identity/lifecycle-evidence-report"
)
DEFAULT_HINTS_PATH = SCRIPT_DIR.parent / "configs/us_lifecycle_hints.yaml"

# Filled from exact current and candidate projections.  These pins cover data
# not already exposed by the eight-case audit and make tampering fail before a
# write can be planned.
BASE_PROJECTION_SHA256: Mapping[str, str] = {
    "security_master": "3456d3060d4c5284d0893f3c7368ec74b466b1947bdf9d01c793954ca11d359f",
    "symbol_history": "8b291c7ffb21a5298a548992616eb96f88a8d350e1be543d9d28a38b085ab7a5",
    "daily_price_raw": "5378c65ef6b01e2be72c283ceb1dcc389a23d8bb1f9326f282bacc322df8a292",
    "adjustment_factors": "c02e6ab946a6ffdac6d7d24b796a3beaf11f19ed414288c61ed9c256bdc503f9",
    "corporate_actions": "a1da678dc7c7f0c99592d3b106f6e07f8e5f1af9c6240a5ec6826ba1759830f3",
    "lifecycle_resolutions": "0839689ffe7388430aed8c137ed184b28406f4aec498012c183613cead7d6ceb",
    "index_constituent_anchors": "8671e73b044acd38d435b8743f2003ec5728e2ca03621179d02fe50c2034edcb",
    "index_membership_events": "9b21e75d125dc6de7eabb30f1245566611f220ea99d3dc2030c5ba449cab11ce",
}
CANDIDATE_PROJECTION_SHA256: Mapping[str, str] = {
    "security_master": "bf4231f19769da8594905615d7eccd762b929b36a9704de4bde7b2ad04f613a0",
    "symbol_history": "3427a14d606c41c9667f204379f15bf2147e737e12b19a6fa1b7863d6a280d91",
    "daily_price_raw": "77f154621cf2f40ca3a8d7a8ace36c06c27fd522845a2ff9058f6212bd184d44",
    "corporate_actions": "d8ee52c42046bcfa9e278f2563687497c39220c01cae210a48cb09103b55007c",
    "adjustment_factors": "2336dd0f139c601dda9ce6c940a61d313a59c7c528de42215cde419de88513c4",
    "lifecycle_resolutions": "2615a166fb985ea4d072972c1dadfb81d03b3e72d7abf368cd8007d93cb12c4d",
    "index_constituent_anchors": "4efe101a9632c0701b86f26c54515211e6375abb24ea8ee06ec19b270ce5b1f3",
    "index_membership_events": "fd6c88cc8d84f5256f922156b5b90e2e1fcda58376aadb07202582d6f5f43ae7",
}
BASE_FACTOR_ECONOMICS_SHA256 = (
    "a16532679111a10a66668afd12ab0bab1585d85403c0ea112606dcb017d3b01b"
)
CANDIDATE_FACTOR_ECONOMICS_SHA256 = (
    "19f419ad615eb12f6a1a6ed51f1f454dd6ec50752f57ba7609c6e2850016c8fd"
)
FRESH_LIFECYCLE_CANDIDATE_SET_SHA256 = (
    "a4f2097dc6c6be40cd4d1aa465e3221d6082d240db070800038e4a31b70e6de8"
)
FRESH_LIFECYCLE_RESOLUTION_SET_SHA256 = (
    "7ff3df6c09324d5baa7793a4a76c71379ae78bb05c13e27ef4b9504623ba5afd"
)
DESCENDANT_LIFECYCLE_CANDIDATE_SET_SHA256 = (
    "32cf8a701a37041584b4a8117064c858d122d3fa50b6f76f19f3e05bd4060c64"
)
DESCENDANT_LIFECYCLE_RESOLUTION_SET_SHA256 = (
    "150ea3f58b1ccc638955b8411a4b0fd7f0c7efec68876f6991cfbcb2264253c5"
)
DESCENDANT_AUDIT_SHA256 = (
    "175210bdf903990ebf8f11ac40ff2797d3d35bee0e4fab7ea2b3546c33cc7791"
)


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    version_state: Mapping[str, Mapping[str, Any]]
    frames: Mapping[str, pd.DataFrame]
    planned_versions: Mapping[str, str]
    planned_release: DataRelease | None
    lifecycle_report_content: bytes
    lifecycle_report_object_path: str
    lifecycle_metadata: Mapping[str, Any]
    warnings: tuple[str, ...]
    summary: Mapping[str, Any]
    candidate_frames: Mapping[str, pd.DataFrame] = field(default_factory=dict)


@dataclass(frozen=True)
class FreshLifecycleEvidence:
    content: bytes
    sha256: str
    object_path: str
    archive_row: Mapping[str, Any]
    coverage: LifecycleCoverageReport
    metadata: Mapping[str, Any]
    finalizer_compatibility: Mapping[str, Any]


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
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _projected_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(float(value), ".17g")
    return str(value)


def _frame_sha256(frame: pd.DataFrame, *, sort_by: tuple[str, ...]) -> str:
    ordered = frame.sort_values(list(sort_by), kind="stable")
    records = [
        {column: _projected_value(row[column]) for column in frame.columns}
        for _, row in ordered.iterrows()
    ]
    return sha256_bytes(_canonical_json_bytes(records))


def _scope(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    ids = {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
    if dataset in {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "lifecycle_resolutions",
    }:
        return frame.loc[frame["security_id"].map(_text).isin(ids)].copy()
    if dataset in {"index_constituent_anchors", "index_membership_events"}:
        return frame.loc[frame["security_id"].map(_text).isin(ids)].copy()
    raise KeyError(dataset)


PROJECTION_SORT = {
    "security_master": ("security_id",),
    "symbol_history": ("security_id", "effective_from", "symbol"),
    "daily_price_raw": ("security_id", "session"),
    "corporate_actions": ("event_id",),
    "adjustment_factors": ("security_id", "session"),
    "lifecycle_resolutions": ("candidate_id",),
    "index_constituent_anchors": ("index_id", "anchor_date", "security_id"),
    "index_membership_events": ("event_id",),
}


def _projection_sha256(frames: Mapping[str, pd.DataFrame], dataset: str) -> str:
    return _frame_sha256(
        _scope(frames[dataset], dataset), sort_by=PROJECTION_SORT[dataset]
    )


def _factor_economics_sha256(frame: pd.DataFrame) -> str:
    scoped = _scope(frame, "adjustment_factors").loc[
        :, ["security_id", "session", "split_factor", "total_return_factor"]
    ]
    return _frame_sha256(scoped, sort_by=("security_id", "session"))


def _version_state(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    etags: dict[str, str | None] = {}
    state: dict[str, dict[str, Any]] = {}
    for dataset in REQUIRED_DATASETS:
        version = release.dataset_versions.get(dataset, "")
        if not version:
            raise RuntimeError(f"Current release lacks {dataset}.")
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}.")
        manifest = repository.manifest_for_version(dataset, version)
        manifest_path = (
            repository.root
            / repository.version_prefix(dataset, version)
            / "manifest.json"
        )
        etags[dataset] = etag
        state[dataset] = {
            "version": version,
            "pointer_etag": etag,
            "pointer_sha256": sha256_bytes(
                repository.objects.get(repository.current_key(dataset)).data
            ),
            "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
            "files": [
                {
                    "path": item.path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "row_count": item.row_count,
                }
                for item in manifest.files
            ],
        }
    return etags, state


def _load_plan_frames(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, pd.DataFrame]:
    """Load full small tables and only the two reviewed IDs from heavy tables."""

    security_ids = {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
    frames: dict[str, pd.DataFrame] = {}
    for dataset in REQUIRED_DATASETS:
        version = release.dataset_versions[dataset]
        if dataset in HEAVY_WRITE_DATASETS:
            frames[dataset] = identity_tails._read_security_subset(
                repository,
                dataset,
                version,
                security_ids,
            )
        else:
            frames[dataset] = repository.read_frame(dataset, version)
    return frames


def _official_action_row(actions: pd.DataFrame) -> pd.Series:
    rows = actions.loc[actions["event_id"].map(_text).eq(OLD_EVENT_ID)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one original SYMC transition; found {len(rows)}.")
    row = rows.iloc[0]
    exact = (
        _text(row.get("security_id")) == OLD_SECURITY_ID
        and _text(row.get("action_type")).lower() == "ticker_change"
        and _date(row.get("effective_date")) == TRANSITION_DATE
        and _date(row.get("ex_date")) == TRANSITION_DATE
        and _text(row.get("new_security_id")) == CANONICAL_SECURITY_ID
        and _text(row.get("new_symbol")).upper() == CANONICAL_SYMBOL
        and bool(row.get("official"))
        and _text(row.get("source_url")) == OFFICIAL_SOURCE_URL
        and _text(row.get("source_hash")).lower() == OFFICIAL_SOURCE_HASH
        and pd.isna(row.get("ratio"))
        and pd.isna(row.get("cash_amount"))
    )
    if not exact:
        raise RuntimeError("Original SYMC transition economics/provenance changed.")
    return row


def _canonical_action(row: pd.Series, columns: pd.Index) -> pd.DataFrame:
    value = row.to_dict()
    value.update(
        {
            "event_id": CANONICAL_EVENT_ID,
            "security_id": CANONICAL_SECURITY_ID,
            "new_security_id": CANONICAL_SECURITY_ID,
        }
    )
    if canonical_lifecycle_event_id(
        CANONICAL_SECURITY_ID, "ticker_change", TRANSITION_DATE
    ) != CANONICAL_EVENT_ID:
        raise RuntimeError("Canonical SYMC/NLOK event ID pin changed.")
    return pd.DataFrame([value]).reindex(columns=columns)


def _base_lifecycle_report_document(
    repository: LocalDatasetRepository,
    release: DataRelease,
    source_archive: pd.DataFrame,
) -> tuple[dict[str, Any], bytes, Mapping[str, Any]]:
    manifest = repository.manifest_for_version(
        "lifecycle_resolutions", release.dataset_versions["lifecycle_resolutions"]
    )
    if manifest.metadata.get("evidence_report_sha256") != (
        BASE_LIFECYCLE_EVIDENCE_REPORT_SHA256
    ):
        raise RuntimeError("Pinned lifecycle evidence-report manifest binding changed.")
    rows = source_archive.loc[
        source_archive["archive_id"]
        .map(_text)
        .str.lower()
        .eq(BASE_LIFECYCLE_EVIDENCE_REPORT_SHA256)
    ]
    if len(rows) != 1:
        raise RuntimeError("Pinned lifecycle evidence report is absent or duplicated.")
    row = rows.iloc[0].to_dict()
    if not (
        _text(row.get("source_hash")).lower()
        == BASE_LIFECYCLE_EVIDENCE_REPORT_SHA256
        and _text(row.get("source")) == "lifecycle_evidence_report"
        and _text(row.get("content_type")) == "application/json"
    ):
        raise RuntimeError("Pinned lifecycle evidence-report archive row changed.")
    root = repository.root.resolve()
    path = (repository.root / _text(row.get("object_path"))).resolve()
    if root != path and root not in path.parents:
        raise RuntimeError("Pinned lifecycle report path escapes the repository.")
    if not path.is_file():
        raise RuntimeError("Pinned lifecycle evidence-report payload is missing.")
    encoded = path.read_bytes()
    try:
        payload = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    except Exception as exc:
        raise RuntimeError("Pinned lifecycle evidence-report payload is invalid.") from exc
    if not (
        len(payload) == BASE_LIFECYCLE_EVIDENCE_REPORT_BYTES
        and sha256_bytes(payload) == BASE_LIFECYCLE_EVIDENCE_REPORT_SHA256
    ):
        raise RuntimeError("Pinned lifecycle evidence-report content changed.")
    report = json.loads(payload)
    return report, payload, row


def _current_lifecycle_report_document(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, Any], bytes]:
    _base_lifecycle_report_document(
        repository, release, frames["source_archive"]
    )
    path = repository.root.parent.parent / CURRENT_LIFECYCLE_REPORT_PATH
    if not path.is_file():
        path = CURRENT_LIFECYCLE_REPORT_PATH
    if not path.is_file():
        raise RuntimeError("Pinned current-release lifecycle report is missing.")
    payload = path.read_bytes()
    if not (
        len(payload) == CURRENT_LIFECYCLE_REPORT_BYTES
        and sha256_bytes(payload) == CURRENT_LIFECYCLE_REPORT_SHA256
    ):
        raise RuntimeError("Pinned current-release lifecycle report changed.")
    report = json.loads(payload)
    if not (
        report.get("release_version") == release.version
        and report.get("input_dataset_versions") == release.dataset_versions
        and report.get("candidate_count") == 182
        and isinstance(report.get("records"), dict)
        and len(report["records"]) == 182
        and report.get("hints_sha256") == LEGACY_LIFECYCLE_HINTS_SHA256
    ):
        raise RuntimeError("Pinned current-release lifecycle report binding changed.")
    return report, payload


def _assert_current_lifecycle_hints() -> None:
    if not DEFAULT_HINTS_PATH.is_file():
        raise RuntimeError("Current lifecycle hints are missing.")
    digest = sha256_bytes(DEFAULT_HINTS_PATH.read_bytes())
    if digest != CURRENT_LIFECYCLE_HINTS_SHA256:
        raise RuntimeError(
            "Current lifecycle hints changed; exact simple7 -> SYMC handoff is "
            "fail-closed."
        )


def _validate_descendant_lifecycle_report_payload(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    *,
    report: Mapping[str, Any],
    payload: bytes,
    digest: str,
    parent_candidates: tuple[Any, ...] | None = None,
) -> tuple[dict[str, Any], bytes, str]:
    _assert_current_lifecycle_hints()
    if sha256_bytes(payload) != digest:
        raise RuntimeError("Seven-identity lifecycle report payload hash changed.")
    report = dict(report)
    if not (
        report.get("release_version") == release.version
        and report.get("input_dataset_versions") == release.dataset_versions
        and report.get("candidate_count") == 182
        and isinstance(report.get("records"), dict)
        and len(report["records"]) == 182
        and report.get("hints_sha256") == CURRENT_LIFECYCLE_HINTS_SHA256
    ):
        raise RuntimeError("Seven-identity lifecycle report binding changed.")
    candidates = parent_candidates
    if candidates is None:
        parent = _CandidateRepository(repository, release.dataset_versions, frames)
        candidates = _candidate_values(parent, release)
    expected_binding = build_lifecycle_report_binding(
        release_version=release.version,
        completed_session=release.completed_session,
        dataset_versions=release.dataset_versions,
        candidates=candidates,
        hints_path=DEFAULT_HINTS_PATH,
        sec_fetch_policy=report.get("sec_fetch_policy"),
        sec_max_http_attempts=report.get("sec_max_http_attempts"),
        sec_max_http_attempts_per_candidate=report.get(
            "sec_max_http_attempts_per_candidate"
        ),
        sec_max_http_attempts_per_request=report.get(
            "sec_max_http_attempts_per_request"
        ),
        sec_http_attempts=report.get("sec_http_attempts"),
        sec_http_attempts_by_candidate=report.get(
            "sec_http_attempts_by_candidate"
        ),
    )
    validate_lifecycle_report_binding(
        report,
        expected_binding,
        purpose="SYMC/NLOK exact simple7 parent",
    )
    return report, payload, digest


def _descendant_lifecycle_report_document(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    *,
    parent_candidates: tuple[Any, ...] | None = None,
) -> tuple[dict[str, Any], bytes, str]:
    manifest = repository.manifest_for_version(
        "lifecycle_resolutions", release.dataset_versions["lifecycle_resolutions"]
    )
    primary = _text(manifest.metadata.get("lifecycle_evidence_report_sha256"))
    publication = _text(manifest.metadata.get("evidence_report_sha256"))
    if primary and publication and primary != publication:
        raise RuntimeError("Seven-identity lifecycle report metadata conflicts.")
    digest = (primary or publication).lower()
    if len(digest) != 64:
        raise RuntimeError("Seven-identity lifecycle report hash is missing.")
    payload = _archived_report_payload(
        repository, frames["source_archive"], digest
    )
    report = json.loads(payload)
    return _validate_descendant_lifecycle_report_payload(
        repository,
        release,
        frames,
        report=report,
        payload=payload,
        digest=digest,
        parent_candidates=parent_candidates,
    )


def _parent_lifecycle_report_document(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    *,
    parent_release_kind: str,
) -> tuple[dict[str, Any], bytes, str]:
    if parent_release_kind == "pinned_base":
        raise RuntimeError(
            "SYMC/NLOK publication requires the exact reviewed seven-identity "
            "price-tail descendant; the pinned base is intentionally fail-closed."
        )
    if parent_release_kind == "identity_price_tails_descendant":
        return _descendant_lifecycle_report_document(
            repository, release, frames
        )
    raise RuntimeError("Unsupported SYMC/NLOK parent release kind.")


def _verify_lifecycle_report_supersession(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    *,
    parent_release_kind: str,
    parent_report_payload: tuple[dict[str, Any], bytes, str] | None = None,
) -> dict[str, Any]:
    # The manifest-bound report is retained and verified as the immutable
    # provenance base.  Its six pre-short-tail candidate identities are stale,
    # so the exact current-release report is separately hash-pinned below.
    if parent_release_kind == "pinned_base":
        _base_lifecycle_report_document(
            repository, release, frames["source_archive"]
        )
    if parent_report_payload is None:
        report, _payload, source_digest = _parent_lifecycle_report_document(
            repository,
            release,
            frames,
            parent_release_kind=parent_release_kind,
        )
    else:
        report, _payload, source_digest = parent_report_payload
    records = report.get("records")
    if not isinstance(records, dict):
        raise RuntimeError("Pinned lifecycle report records are invalid.")
    record = records.get(OLD_SECURITY_ID)
    if not isinstance(record, dict):
        raise RuntimeError("Pinned lifecycle report lacks the old SYMC candidate.")
    candidate = record.get("candidate") or {}
    parsed = record.get("parsed") or {}
    exact = (
        candidate.get("security_id") == OLD_SECURITY_ID
        and candidate.get("symbol") == OLD_SYMBOL
        and candidate.get("last_price_date") == "2020-11-05"
        and parsed.get("action_type") == "ticker_change"
        and parsed.get("effective_date") == TRANSITION_DATE
        and parsed.get("new_symbol") == CANONICAL_SYMBOL
        and record.get("eligible_for_apply") is True
        and (record.get("crosscheck") or {}).get("passed") is True
        and record.get("source_hash") == OFFICIAL_SOURCE_HASH
    )
    if not exact:
        raise RuntimeError("Old SYMC lifecycle report projection changed.")
    return {
        "report_sha256": source_digest,
        "manifest_bound_predecessor_report_sha256": (
            BASE_LIFECYCLE_EVIDENCE_REPORT_SHA256
        ),
        "old_candidate_id": OLD_CANDIDATE_ID,
        "old_event_id": OLD_EVENT_ID,
        "superseded_by_event_id": CANONICAL_EVENT_ID,
        "classification": "same_security_nonterminal_ticker_continuation",
        "old_terminal_resolution_must_not_be_reapplied": True,
        "finalizer_requires_exact_supersession_guard": True,
    }


def _parent_release_kind(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> str:
    if release.version == PINNED_RELEASE_VERSION:
        for dataset, version in PINNED_DATASET_VERSIONS.items():
            if release.dataset_versions.get(dataset) != version:
                raise RuntimeError(f"Pinned {dataset} version changed.")
        _assert_current_lifecycle_hints()
        raise RuntimeError(
            "SYMC/NLOK repair order violation: apply the exact reviewed "
            "seven-identity price-tail repair before planning SYMC/NLOK."
        )

    _assert_current_lifecycle_hints()
    if not identity_tails._exact_repair_manifests(repository, release):
        raise RuntimeError(
            "Current release is neither the pinned base nor the exact reviewed "
            "seven-identity price-tail descendant."
        )
    for dataset, version in PINNED_DATASET_VERSIONS.items():
        if dataset in identity_tails.WRITE_DATASETS:
            continue
        if release.dataset_versions.get(dataset) != version:
            raise RuntimeError(
                f"Seven-identity descendant changed unrelated {dataset}."
            )
    if identity_tails.registry_inventory_sha256() != (
        identity_tails.TRUSTED_REGISTRY_INVENTORY_SHA256
    ):
        raise RuntimeError("Seven-identity descendant registry pin changed.")
    verified = identity_tails.prepare_repair(repository)
    if not (
        verified.release.to_bytes() == release.to_bytes()
        and verified.summary.get("status") == "already_repaired"
        and verified.summary.get("registry_inventory_sha256")
        == identity_tails.TRUSTED_REGISTRY_INVENTORY_SHA256
        and verified.summary.get("candidate_content_sha256")
        == identity_tails.EXPECTED_CANDIDATE_CONTENT_SHA256
        and verified.summary.get("lifecycle_candidate_set_sha256")
        == identity_tails.EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
        and verified.summary.get("lifecycle_resolution_set_sha256")
        == identity_tails.EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256
    ):
        raise RuntimeError("Seven-identity descendant exact replay failed.")
    return "identity_price_tails_descendant"


def _parent_release_manifest_kind(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> str:
    if release.version == PINNED_RELEASE_VERSION:
        if all(
            release.dataset_versions.get(dataset) == version
            for dataset, version in PINNED_DATASET_VERSIONS.items()
        ):
            raise RuntimeError(
                "Applied SYMC/NLOK release has an invalid pinned-base parent; "
                "the required order is simple7 -> SYMC/NLOK."
            )
        raise RuntimeError("Pinned SYMC/NLOK parent versions changed.")
    _assert_current_lifecycle_hints()
    if not identity_tails._exact_repair_manifests(repository, release):
        raise RuntimeError("SYMC/NLOK parent repair manifests are not exact.")
    for dataset, version in PINNED_DATASET_VERSIONS.items():
        if (
            dataset not in identity_tails.WRITE_DATASETS
            and release.dataset_versions.get(dataset) != version
        ):
            raise RuntimeError(
                f"Seven-identity parent changed unrelated {dataset}."
            )
    if getattr(identity_tails, "_verify_repaired_lifecycle_state", None) is None:
        raise RuntimeError("SYMC/NLOK parent lacks the simple7 lifecycle verifier.")
    return "identity_price_tails_descendant"


def _verify_base(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    *,
    parent_kind: str | None = None,
    parent_candidates: tuple[Any, ...] | None = None,
    parent_report_payload: tuple[dict[str, Any], bytes, str] | None = None,
) -> dict[str, Any]:
    parent_kind = parent_kind or _parent_release_kind(repository, release, frames)

    # Re-run the completed finite audit.  It verifies the official archive,
    # action, price, identity and membership hashes and the signal signature.
    if parent_kind == "pinned_base":
        audited = audit.build_audit(repository)
        diagnostic = audited["symc_full_identity_diagnostic"]
        audit_sha256 = sha256_bytes(audit._canonical_json_bytes(audited))
    else:
        diagnostic = audit._symc_full_identity_diagnostic(
            frames["daily_price_raw"], frames["adjustment_factors"]
        )
        audit_sha256 = sha256_bytes(audit._canonical_json_bytes(diagnostic))
        if audit_sha256 != DESCENDANT_AUDIT_SHA256:
            raise RuntimeError("Reviewed simple7 descendant SYMC audit changed.")
    if not (
        diagnostic["old_sid_rows"] == OLD_PRICE_ROWS
        and diagnostic["successor_rows"] == CANONICAL_PRICE_ROWS
        and diagnostic["old_rows_covered_by_successor"] == OLD_PRICE_ROWS
        and diagnostic["old_price_inventory_sha256"] == OLD_PRICE_SHA256
        and diagnostic["successor_price_inventory_sha256"]
        == CANONICAL_PRICE_SHA256
    ):
        raise RuntimeError("SYMC/NLOK price coverage pins changed.")
    for mode in ("raw", "total_return_adjusted"):
        diff = diagnostic["pre_transition_triple_supertrend_diff"][mode]
        if diff["TripleBuySignal"]["count"] or diff["TripleSellSignal"]["count"]:
            raise RuntimeError("SYMC/NLOK pre-transition trade signals changed.")
    adjusted = diagnostic["pre_transition_triple_supertrend_diff"][
        "total_return_adjusted"
    ]
    if adjusted["TripleST1_Trend"]["sessions"] != ["2015-08-26"]:
        raise RuntimeError("SYMC/NLOK reviewed adjusted ST1 difference changed.")

    _official_action_row(frames["corporate_actions"])
    old_resolution = frames["lifecycle_resolutions"].loc[
        frames["lifecycle_resolutions"]["security_id"].map(_text).eq(OLD_SECURITY_ID)
    ]
    if not (
        len(old_resolution) == 1
        and _text(old_resolution.iloc[0].get("candidate_id")) == OLD_CANDIDATE_ID
        and _text(old_resolution.iloc[0].get("event_id")) == OLD_EVENT_ID
        and _text(old_resolution.iloc[0].get("resolution")) == "applied"
        and _text(old_resolution.iloc[0].get("successor_security_id"))
        == CANONICAL_SECURITY_ID
        and _text(old_resolution.iloc[0].get("source_hash")).lower()
        == OFFICIAL_SOURCE_HASH
    ):
        raise RuntimeError("Old SYMC terminal resolution projection changed.")
    for dataset, expected in BASE_PROJECTION_SHA256.items():
        if dataset == "adjustment_factors" and parent_kind != "pinned_base":
            continue
        if expected and _projection_sha256(frames, dataset) != expected:
            raise RuntimeError(f"Pinned base {dataset} projection changed.")
    if _factor_economics_sha256(frames["adjustment_factors"]) != (
        BASE_FACTOR_ECONOMICS_SHA256
    ):
        raise RuntimeError("Pinned base adjustment-factor economics changed.")
    if parent_kind != "identity_price_tails_descendant":
        raise RuntimeError("SYMC/NLOK base verification requires exact simple7 ancestry.")
    if parent_candidates is None:
        parent_candidate_repository = _CandidateRepository(
            repository, release.dataset_versions, frames
        )
        parent_candidates = _candidate_values(parent_candidate_repository, release)
    if parent_report_payload is None:
        parent_report_payload = _descendant_lifecycle_report_document(
            repository,
            release,
            frames,
            parent_candidates=parent_candidates,
        )
    return {
        "audit_sha256": audit_sha256,
        "parent_release_kind": parent_kind,
        "signal_diagnostic": diagnostic,
        "finalizer_supersession": _verify_lifecycle_report_supersession(
            repository,
            release,
            frames,
            parent_release_kind=parent_kind,
            parent_report_payload=parent_report_payload,
        ),
        "_parent_lifecycle_report_payload": parent_report_payload,
        "_parent_lifecycle_candidates": parent_candidates,
    }


def _repaired_state(
    frames: Mapping[str, pd.DataFrame],
    *,
    require_pins: bool = True,
    allow_parent_factor_lineage: bool = False,
) -> bool:
    for dataset in WRITE_DATASETS:
        if dataset not in frames:
            return False
    old_presence = {
        dataset: int(
            frames[dataset]["security_id"].map(_text).eq(OLD_SECURITY_ID).sum()
        )
        for dataset in IDENTITY_WRITE_DATASETS
    }
    if any(old_presence.values()):
        return False
    history = frames["symbol_history"].loc[
        frames["symbol_history"]["security_id"].map(_text).eq(CANONICAL_SECURITY_ID)
    ]
    intervals = {
        (_text(row.get("symbol")).upper(), _date(row.get("effective_from")), _date(row.get("effective_to")))
        for row in history.to_dict(orient="records")
    }
    if intervals != {
        (OLD_SYMBOL, OLD_SYMBOL_FROM, OLD_SYMBOL_TO),
        (CANONICAL_SYMBOL, CANONICAL_SYMBOL_FROM, CANONICAL_SYMBOL_TO),
    }:
        return False
    actions = frames["corporate_actions"]
    installed = actions.loc[actions["event_id"].map(_text).eq(CANONICAL_EVENT_ID)]
    if len(installed) != 1:
        return False
    row = installed.iloc[0]
    if not (
        _text(row.get("security_id")) == CANONICAL_SECURITY_ID
        and _text(row.get("new_security_id")) == CANONICAL_SECURITY_ID
        and _date(row.get("effective_date")) == TRANSITION_DATE
        and _text(row.get("source_hash")).lower() == OFFICIAL_SOURCE_HASH
        and pd.isna(row.get("ratio"))
        and pd.isna(row.get("cash_amount"))
    ):
        return False
    if frames["lifecycle_resolutions"]["event_id"].map(_text).isin(
        {OLD_EVENT_ID, CANONICAL_EVENT_ID}
    ).any():
        return False
    if frames["index_membership_events"]["event_id"].map(_text).isin(
        {SP500_SWAP_REMOVE_EVENT_ID, SP500_SWAP_ADD_EVENT_ID}
    ).any():
        return False
    canonical_prices = _scope(frames["daily_price_raw"], "daily_price_raw")
    if not (
        len(canonical_prices) == CANONICAL_PRICE_ROWS
        and _frame_sha256(canonical_prices, sort_by=("security_id", "session"))
        == CANONICAL_PRICE_SHA256
    ):
        return False
    if require_pins:
        for dataset, expected in CANDIDATE_PROJECTION_SHA256.items():
            if dataset == "adjustment_factors" and allow_parent_factor_lineage:
                continue
            if not expected or _projection_sha256(frames, dataset) != expected:
                return False
        if _factor_economics_sha256(frames["adjustment_factors"]) != (
            CANDIDATE_FACTOR_ECONOMICS_SHA256
        ):
            return False
    return True


def prepare_candidate_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    completed_session: str,
    require_candidate_pins: bool = True,
    parent_release_kind: str = "pinned_base",
    consume_inputs: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if consume_inputs:
        if not isinstance(frames, dict):
            raise TypeError("Consumed SYMC/NLOK inputs must be a mutable dict.")
        output = frames
    else:
        output = {dataset: frame.copy() for dataset, frame in frames.items()}
    missing = sorted(set(REQUIRED_DATASETS) - set(output))
    if missing:
        raise RuntimeError("SYMC/NLOK transform inputs are missing: " + ", ".join(missing))
    for dataset, expected in BASE_PROJECTION_SHA256.items():
        if (
            dataset == "adjustment_factors"
            and parent_release_kind == "identity_price_tails_descendant"
        ):
            continue
        if _projection_sha256(output, dataset) != expected:
            raise RuntimeError(f"Pinned base {dataset} projection changed.")
    if _factor_economics_sha256(output["adjustment_factors"]) != (
        BASE_FACTOR_ECONOMICS_SHA256
    ):
        raise RuntimeError("Pinned base adjustment-factor economics changed.")
    scoped_factors_before = _scope(
        output["adjustment_factors"], "adjustment_factors"
    )
    old_factor_rows_retired = int(
        scoped_factors_before["security_id"].map(_text).eq(OLD_SECURITY_ID).sum()
    )
    if old_factor_rows_retired != OLD_PRICE_ROWS:
        raise RuntimeError("Old SYMC adjustment-factor row count changed.")
    del scoped_factors_before
    action_rows = output["corporate_actions"]
    canonical_event_collisions = int(
        action_rows["event_id"].map(_text).eq(CANONICAL_EVENT_ID).sum()
    )
    if canonical_event_collisions:
        raise RuntimeError(
            "Canonical SYMC/NLOK ticker event ID already exists globally."
        )
    action = _official_action_row(action_rows)

    before_canonical = {
        dataset: _projection_sha256(output, dataset)
        for dataset in IDENTITY_WRITE_DATASETS
    }
    canonical_economic_before = {
        "prices": _frame_sha256(
            output["daily_price_raw"].loc[
                output["daily_price_raw"]["security_id"].map(_text).eq(
                    CANONICAL_SECURITY_ID
                )
            ],
            sort_by=("security_id", "session"),
        ),
        "factors": _frame_sha256(
            output["adjustment_factors"].loc[
                output["adjustment_factors"]["security_id"].map(_text).eq(
                    CANONICAL_SECURITY_ID
                )
            ],
            sort_by=("security_id", "session"),
        ),
        "actions": _frame_sha256(
            output["corporate_actions"].loc[
                output["corporate_actions"]["security_id"].map(_text).eq(
                    CANONICAL_SECURITY_ID
                )
            ],
            sort_by=("event_id",),
        ),
        "resolutions": _frame_sha256(
            output["lifecycle_resolutions"].loc[
                output["lifecycle_resolutions"]["security_id"].map(_text).eq(
                    CANONICAL_SECURITY_ID
                )
            ],
            sort_by=("candidate_id",),
        ),
    }

    output["security_master"] = output["security_master"].loc[
        ~output["security_master"]["security_id"].map(_text).eq(OLD_SECURITY_ID)
    ].copy()

    history = output["symbol_history"]
    old_history = history.loc[
        history["security_id"].map(_text).eq(OLD_SECURITY_ID)
        & history["symbol"].map(_text).str.upper().eq(OLD_SYMBOL)
    ]
    nlok_history = history.loc[
        history["security_id"].map(_text).eq(CANONICAL_SECURITY_ID)
        & history["symbol"].map(_text).str.upper().eq(CANONICAL_SYMBOL)
    ]
    if len(old_history) != 1 or len(nlok_history) != 1:
        raise RuntimeError("SYMC/NLOK symbol-history source rows changed.")
    history = history.loc[
        ~history["security_id"].map(_text).eq(OLD_SECURITY_ID)
    ].copy()
    nlok_mask = (
        history["security_id"].map(_text).eq(CANONICAL_SECURITY_ID)
        & history["symbol"].map(_text).str.upper().eq(CANONICAL_SYMBOL)
    )
    history.loc[nlok_mask, "effective_from"] = CANONICAL_SYMBOL_FROM
    symc = old_history.iloc[0].to_dict()
    symc.update(
        {
            "security_id": CANONICAL_SECURITY_ID,
            "effective_from": OLD_SYMBOL_FROM,
            "effective_to": OLD_SYMBOL_TO,
            "source": "official_canonical_identity_repair",
            "source_url": OFFICIAL_SOURCE_URL,
            "retrieved_at": _text(action.get("retrieved_at")),
            "source_hash": OFFICIAL_SOURCE_HASH,
        }
    )
    output["symbol_history"] = pd.concat(
        [history, pd.DataFrame([symc]).reindex(columns=history.columns)],
        ignore_index=True,
    )

    for dataset in ("daily_price_raw", "adjustment_factors"):
        output[dataset] = output[dataset].loc[
            ~output[dataset]["security_id"].map(_text).eq(OLD_SECURITY_ID)
        ].copy()

    actions = output["corporate_actions"]
    old_action_count = int(actions["security_id"].map(_text).eq(OLD_SECURITY_ID).sum())
    actions = actions.loc[
        ~actions["security_id"].map(_text).eq(OLD_SECURITY_ID)
    ].copy()
    output["corporate_actions"] = pd.concat(
        [actions, _canonical_action(action, actions.columns)], ignore_index=True
    )

    resolutions = output["lifecycle_resolutions"]
    removed_resolution = resolutions.loc[
        resolutions["security_id"].map(_text).eq(OLD_SECURITY_ID)
        | resolutions["event_id"].map(_text).eq(OLD_EVENT_ID)
        | resolutions["candidate_id"].map(_text).eq(OLD_CANDIDATE_ID)
    ]
    if len(removed_resolution) != 1:
        raise RuntimeError("SYMC terminal-resolution removal is not exact.")
    output["lifecycle_resolutions"] = resolutions.drop(
        index=removed_resolution.index
    ).copy()

    anchors = output["index_constituent_anchors"]
    rebound_anchors = int(anchors["security_id"].map(_text).eq(OLD_SECURITY_ID).sum())
    if rebound_anchors != 1:
        raise RuntimeError("Expected exactly one S&P anchor to rebind.")
    anchors.loc[
        anchors["security_id"].map(_text).eq(OLD_SECURITY_ID), "security_id"
    ] = CANONICAL_SECURITY_ID
    output["index_constituent_anchors"] = anchors

    events = output["index_membership_events"]
    swap = events.loc[
        events["event_id"].map(_text).isin(
            {SP500_SWAP_REMOVE_EVENT_ID, SP500_SWAP_ADD_EVENT_ID}
        )
    ]
    if not (
        len(swap) == 2
        and set(swap["operation"].map(_text).str.upper()) == {"ADD", "REMOVE"}
        and set(swap["effective_date"].map(_date)) == {"2019-11-05"}
        and set(swap["security_id"].map(_text))
        == {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
    ):
        raise RuntimeError("2019-11-05 SYMC/NLOK S&P swap changed.")
    events = events.drop(index=swap.index).copy()
    if events["security_id"].map(_text).eq(OLD_SECURITY_ID).any():
        raise RuntimeError("Unexpected old SYMC membership events remain.")
    output["index_membership_events"] = events

    for dataset in IDENTITY_WRITE_DATASETS:
        validate_dataset(
            dataset,
            output[dataset],
            completed_session=(
                CANONICAL_SYMBOL_TO
                if dataset in HEAVY_WRITE_DATASETS
                else completed_session
            ),
            incomplete_action_policy="warn",
        ).raise_for_errors()

    if not _repaired_state(
        output,
        require_pins=require_candidate_pins,
        allow_parent_factor_lineage=(
            parent_release_kind == "identity_price_tails_descendant"
        ),
    ):
        raise RuntimeError("Prepared SYMC/NLOK snapshot failed exact repaired-state pins.")

    canonical_economic_after = {
        "prices": _frame_sha256(
            output["daily_price_raw"].loc[
                output["daily_price_raw"]["security_id"].map(_text).eq(
                    CANONICAL_SECURITY_ID
                )
            ],
            sort_by=("security_id", "session"),
        ),
        "factors": _frame_sha256(
            output["adjustment_factors"].loc[
                output["adjustment_factors"]["security_id"].map(_text).eq(
                    CANONICAL_SECURITY_ID
                )
            ],
            sort_by=("security_id", "session"),
        ),
        "actions": _frame_sha256(
            output["corporate_actions"].loc[
                output["corporate_actions"]["security_id"].map(_text).eq(
                    CANONICAL_SECURITY_ID
                )
                & ~output["corporate_actions"]["event_id"].map(_text).eq(
                    CANONICAL_EVENT_ID
                )
            ],
            sort_by=("event_id",),
        ),
        "resolutions": _frame_sha256(
            output["lifecycle_resolutions"].loc[
                output["lifecycle_resolutions"]["security_id"].map(_text).eq(
                    CANONICAL_SECURITY_ID
                )
            ],
            sort_by=("candidate_id",),
        ),
    }
    if canonical_economic_after != canonical_economic_before:
        raise RuntimeError("Canonical NLOK economic datasets changed during repair.")

    # Ticker change carries no ratio/cash economics.  Canonical NLOK prices,
    # actions, factors and the later NLOK -> GEN terminal resolution are exact.
    after_canonical = {
        dataset: _projection_sha256(output, dataset)
        for dataset in IDENTITY_WRITE_DATASETS
    }
    summary = {
        "status": "validated_offline_plan",
        "old_security_id": OLD_SECURITY_ID,
        "canonical_security_id": CANONICAL_SECURITY_ID,
        "old_price_rows_retired": OLD_PRICE_ROWS,
        "canonical_price_rows_preserved": CANONICAL_PRICE_ROWS,
        "old_factor_rows_retired": old_factor_rows_retired,
        "old_action_rows_retired": old_action_count,
        "canonical_ticker_action_added": 1,
        "canonical_event_id_preexisting_rows": canonical_event_collisions,
        "old_terminal_resolutions_removed": 1,
        "new_terminal_resolutions_added": 0,
        "sp500_anchors_rebound": rebound_anchors,
        "redundant_sp500_swap_rows_removed": 2,
        "canonical_symbol_intervals": [
            {"symbol": OLD_SYMBOL, "from": OLD_SYMBOL_FROM, "to": OLD_SYMBOL_TO},
            {
                "symbol": CANONICAL_SYMBOL,
                "from": CANONICAL_SYMBOL_FROM,
                "to": CANONICAL_SYMBOL_TO,
            },
        ],
        "canonical_event_id": CANONICAL_EVENT_ID,
        "reviewed_nonterminal_extraction": reviewed_nonterminal_extraction(),
        "reviewed_nonterminal_extraction_sha256": (
            REVIEWED_NONTERMINAL_EXTRACTION_SHA256
        ),
        "cross_validation_same_sid_no_data_spec": reviewed_same_sid_no_data_spec(),
        "projection_sha256_before": before_canonical,
        "projection_sha256_after": after_canonical,
        "canonical_economic_inventory_sha256_before": canonical_economic_before,
        "canonical_economic_inventory_sha256_after": canonical_economic_after,
        "canonical_price_action_factor_resolution_economics_preserved": True,
        "network_accessed": False,
        "http_attempts": 0,
        "eodhd_calls": 0,
        "r2_accessed": False,
        "writes_performed": False,
    }
    return output, summary


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
        if dataset in self.overrides:
            return object()
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        current_version = self.versions.get(dataset)
        if dataset in self.overrides and (
            _version is None or _version == current_version
        ):
            return self.overrides[dataset]
        version = _version or current_version
        if not version:
            return pd.DataFrame(columns=dataset_spec(dataset).required_columns)
        return self.base.read_frame(dataset, version)


def _project_lifecycle_candidate_context(
    parent_context: Mapping[str, pd.DataFrame],
    transformed: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build the post-repair candidate context without full daily-price rows."""

    output = {
        dataset: transformed[dataset]
        for dataset in (
            "security_master",
            "symbol_history",
            "corporate_actions",
            "index_constituent_anchors",
            "index_membership_events",
        )
    }
    terminals = parent_context["daily_price_raw"]
    affected = {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
    terminals = terminals.loc[
        ~terminals["security_id"].map(_text).isin(affected)
    ].copy()
    canonical = transformed["daily_price_raw"].loc[
        transformed["daily_price_raw"]["security_id"]
        .map(_text)
        .eq(CANONICAL_SECURITY_ID)
    ]
    if canonical.empty:
        raise RuntimeError("Canonical NLOK prices are missing from the sparse plan.")
    terminal = pd.DataFrame(
        [
            {
                "security_id": CANONICAL_SECURITY_ID,
                "session": pd.to_datetime(
                    canonical["session"], errors="raise"
                ).max(),
            }
        ]
    ).reindex(columns=terminals.columns)
    output["daily_price_raw"] = pd.concat(
        [terminals, terminal], ignore_index=True
    )
    output["lifecycle_resolutions"] = transformed["lifecycle_resolutions"]
    output["source_archive"] = transformed["source_archive"]
    return output


def _new_planned_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"symc-nlok-identity-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


def _candidate_values(
    candidate: _CandidateRepository,
    release: DataRelease,
) -> tuple[Any, ...]:
    raw = build_lifecycle_candidates(candidate, release=release)
    specs = load_official_lifecycle_exception_evidence(DEFAULT_HINTS_PATH)
    return include_bound_official_applied_event_candidates(
        raw, candidate, release, specs
    )


def _candidate_frame(values: tuple[Any, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": item.security_id,
                "symbol": item.symbol,
                "name": item.name,
                "exchange": item.exchange,
                "last_price_date": item.last_price_date,
                "active_to": item.active_to,
                "index_remove_dates": list(item.index_remove_dates),
            }
            for item in values
        ]
    )


def _refresh_report_summary(report: dict[str, Any]) -> None:
    records = list(report["records"].values())
    counts = Counter(
        str((record.get("parsed") or {}).get("action_type") or "unresolved")
        for record in records
    )
    eligible = sum(bool(record.get("eligible_for_apply")) for record in records)
    summary: dict[str, Any] = {
        "candidate_count": len(records),
        "collected_count": len(records),
        "eligible_count": eligible,
        "unresolved_count": len(records) - eligible,
        "action_type_counts": dict(sorted(counts.items())),
        "sec_fetch_policy": report["sec_fetch_policy"],
        "sec_max_http_attempts": int(report["sec_max_http_attempts"]),
        "sec_max_http_attempts_per_candidate": int(
            report["sec_max_http_attempts_per_candidate"]
        ),
        "sec_max_http_attempts_per_request": int(
            report["sec_max_http_attempts_per_request"]
        ),
        "sec_http_attempts": int(report["sec_http_attempts"]),
        "sec_http_attempts_remaining": int(report["sec_max_http_attempts"])
        - int(report["sec_http_attempts"]),
        "sec_http_attempts_by_candidate": dict(
            report["sec_http_attempts_by_candidate"]
        ),
    }
    official = report.get("official_exception_evidence") or {}
    if official:
        statuses = Counter(
            str((value or {}).get("status") or "invalid")
            for value in official.values()
        )
        summary["official_exception_evidence"] = {
            "evidence_count": len(official),
            "status_counts": dict(sorted(statuses.items())),
            "http_attempts": int(
                report.get("official_exception_evidence_http_attempts", 0)
            ),
        }
    report["summary"] = summary


def _verify_fresh_report(
    report: Mapping[str, Any],
    *,
    planned_release: DataRelease,
    candidates: tuple[Any, ...],
) -> None:
    expected_binding = build_lifecycle_report_binding(
        release_version=planned_release.version,
        completed_session=planned_release.completed_session,
        dataset_versions=planned_release.dataset_versions,
        candidates=candidates,
        hints_path=DEFAULT_HINTS_PATH,
        sec_fetch_policy=report.get("sec_fetch_policy"),
        sec_max_http_attempts=report.get("sec_max_http_attempts"),
        sec_max_http_attempts_per_candidate=report.get(
            "sec_max_http_attempts_per_candidate"
        ),
        sec_max_http_attempts_per_request=report.get(
            "sec_max_http_attempts_per_request"
        ),
        sec_http_attempts=report.get("sec_http_attempts"),
        sec_http_attempts_by_candidate=report.get(
            "sec_http_attempts_by_candidate"
        ),
    )
    validate_lifecycle_report_binding(
        report, expected_binding, purpose="SYMC/NLOK transactional repair"
    )
    records = report.get("records")
    if not isinstance(records, dict):
        raise RuntimeError("Fresh lifecycle report lacks its full records mapping.")
    expected = {item.security_id: item for item in candidates}
    if set(records) != set(expected):
        raise RuntimeError("Fresh lifecycle report candidate inventory is not exact.")
    for security_id, candidate in expected.items():
        record = records[security_id]
        identity = record.get("candidate") if isinstance(record, dict) else None
        if not isinstance(identity, dict):
            raise RuntimeError(f"Fresh lifecycle record is malformed: {security_id}")
        actual = (
            _text(identity.get("security_id")),
            _text(identity.get("symbol")).upper(),
            _date(identity.get("last_price_date")),
        )
        wanted = (
            candidate.security_id,
            candidate.symbol.upper(),
            _date(candidate.last_price_date),
        )
        if actual != wanted:
            raise RuntimeError(
                f"Fresh lifecycle candidate identity changed: {security_id}."
            )
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("Fresh lifecycle report summary is missing.")
    eligible = sum(bool(item.get("eligible_for_apply")) for item in records.values())
    if not (
        summary.get("candidate_count") == len(candidates)
        and summary.get("collected_count") == len(candidates)
        and summary.get("eligible_count") == eligible
        and summary.get("unresolved_count") == len(candidates) - eligible
    ):
        raise RuntimeError("Fresh lifecycle report summary is stale.")


def _fresh_lifecycle_evidence(
    repository: LocalDatasetRepository,
    parent_release: DataRelease,
    planned_release: DataRelease,
    candidate: _CandidateRepository,
    frames: Mapping[str, pd.DataFrame],
    *,
    parent_release_kind: str,
    parent_report_payload: tuple[Mapping[str, Any], bytes, str] | None = None,
    parent_report_frames: Mapping[str, pd.DataFrame] | None = None,
    parent_report_candidates: tuple[Any, ...] | None = None,
) -> FreshLifecycleEvidence:
    if parent_release_kind != "identity_price_tails_descendant":
        raise RuntimeError(
            "Fresh SYMC/NLOK lifecycle evidence may only descend from the exact "
            "reviewed seven-identity release."
        )
    if parent_report_payload is None:
        report, _old_content, source_report_sha256 = (
            _parent_lifecycle_report_document(
                repository,
                parent_release,
                frames,
                parent_release_kind=parent_release_kind,
            )
        )
    else:
        if parent_report_frames is None and parent_report_candidates is None:
            raise RuntimeError(
                "A supplied simple7 lifecycle report requires its exact parent "
                "candidates or candidate frames."
            )
        supplied_report, supplied_payload, supplied_digest = parent_report_payload
        report, _old_content, source_report_sha256 = (
            _validate_descendant_lifecycle_report_payload(
                repository,
                parent_release,
                parent_report_frames or {},
                report=supplied_report,
                payload=supplied_payload,
                digest=supplied_digest,
                parent_candidates=parent_report_candidates,
            )
        )
    report = json.loads(json.dumps(report))
    records = report.get("records")
    if not isinstance(records, dict) or set(records).issuperset({OLD_SECURITY_ID}) is False:
        raise RuntimeError("Old lifecycle report lacks the exact SYMC record.")
    removed = records.pop(OLD_SECURITY_ID)
    if not (
        bool(removed.get("eligible_for_apply"))
        and _text((removed.get("parsed") or {}).get("action_type"))
        == "ticker_change"
    ):
        raise RuntimeError("Removed lifecycle report record is not the reviewed SYMC row.")

    raw_candidates = build_lifecycle_candidates(
        candidate, release=planned_release
    )
    specs = load_official_lifecycle_exception_evidence(DEFAULT_HINTS_PATH)
    candidates = include_bound_official_applied_event_candidates(
        raw_candidates, candidate, planned_release, specs
    )
    binding = build_lifecycle_report_binding(
        release_version=planned_release.version,
        completed_session=planned_release.completed_session,
        dataset_versions=planned_release.dataset_versions,
        candidates=candidates,
        hints_path=DEFAULT_HINTS_PATH,
        sec_fetch_policy=report.get("sec_fetch_policy"),
        sec_max_http_attempts=report.get("sec_max_http_attempts"),
        sec_max_http_attempts_per_candidate=report.get(
            "sec_max_http_attempts_per_candidate"
        ),
        sec_max_http_attempts_per_request=report.get(
            "sec_max_http_attempts_per_request"
        ),
        sec_http_attempts=report.get("sec_http_attempts"),
        sec_http_attempts_by_candidate=report.get(
            "sec_http_attempts_by_candidate"
        ),
    )
    for field in REPORT_BINDING_FIELDS:
        report[field] = binding[field]
    _refresh_report_summary(report)
    _verify_fresh_report(
        report, planned_release=planned_release, candidates=candidates
    )
    if report["summary"].get("action_type_counts") != {
        "cash_merger": 52,
        "delisting": 10,
        "stock_merger": 87,
        "ticker_change": 29,
        "unresolved": 3,
    } or not (
        report["summary"].get("eligible_count") == 160
        and report["summary"].get("unresolved_count") == 21
    ):
        raise RuntimeError(
            "Descendant lifecycle report classification inventory changed."
        )

    candidate_frame = _candidate_frame(candidates)
    coverage = validate_lifecycle_coverage(
        candidate_frame,
        frames["lifecycle_resolutions"],
        frames["corporate_actions"],
        completed_session=planned_release.completed_session,
    )
    expected_coverage = {
        "coverage_gate_version": 1,
        "selection_rule": "us_terminal_v1",
        "candidate_set_sha256": DESCENDANT_LIFECYCLE_CANDIDATE_SET_SHA256,
        "resolution_set_sha256": DESCENDANT_LIFECYCLE_RESOLUTION_SET_SHA256,
        "candidate_count": 181,
        "resolution_count": 181,
        "applied_count": 169,
        "exception_count": 12,
        "open_count": 0,
    }
    if not coverage.valid or coverage.manifest_metadata() != expected_coverage:
        raise RuntimeError("Fresh SYMC/NLOK lifecycle coverage projection changed.")
    temporary = frames["lifecycle_resolutions"].loc[
        frames["lifecycle_resolutions"]["resolution"].map(_text).eq("exception")
        & frames["lifecycle_resolutions"]["recheck_after"]
        .fillna("")
        .map(_text)
        .ne("")
    ]
    if not temporary.empty:
        raise RuntimeError("Fresh lifecycle state contains temporary exceptions.")

    content = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    digest = sha256_bytes(content)
    object_path = (
        f"archives/{planned_release.completed_session}/{digest}.json.gz"
    )
    finalizer_records = lifecycle_finalizer._validate_full_report(
        lifecycle_finalizer.ReportDocument(
            path=Path(object_path),
            content=content,
            value=report,
        ),
        planned_release,
        candidates,
        hints_path=DEFAULT_HINTS_PATH,
    )
    if len(finalizer_records) != 181:
        raise RuntimeError("Full lifecycle finalizer report gate changed.")
    archive_row = {
        "archive_id": digest,
        "dataset": "lifecycle_evidence_report",
        "object_path": object_path,
        "content_type": "application/json",
        "effective_date": planned_release.completed_session,
        "source": "lifecycle_evidence_report",
        "source_url": FRESH_REPORT_SOURCE_URL,
        "retrieved_at": planned_release.created_at,
        "source_hash": digest,
    }
    prior_metadata = dict(
        repository.manifest_for_version(
            "lifecycle_resolutions",
            parent_release.dataset_versions["lifecycle_resolutions"],
        ).metadata
    )
    metadata = {
        **prior_metadata,
        "schema": REPAIR_SCHEMA,
        "operation": OPERATION,
        "input_release_version": parent_release.version,
        "input_versions": dict(parent_release.dataset_versions),
        "output_versions": dict(planned_release.dataset_versions),
        "canonical_security_id": CANONICAL_SECURITY_ID,
        "retired_security_id": OLD_SECURITY_ID,
        "canonical_event_id": CANONICAL_EVENT_ID,
        "evidence_report_sha256": digest,
        "evidence_report_object_path": object_path,
        "lifecycle_evidence_report_sha256": digest,
        "lifecycle_evidence_report_object_path": object_path,
        "lifecycle_report_release_version": planned_release.version,
        "lifecycle_report_input_versions": dict(
            planned_release.dataset_versions
        ),
        "superseded_evidence_report_sha256": (
            source_report_sha256
        ),
        "pinned_base_ancestor_report_sha256": (
            BASE_LIFECYCLE_EVIDENCE_REPORT_SHA256
        ),
        "current_source_report_sha256": source_report_sha256,
        "parent_release_kind": parent_release_kind,
        "lifecycle_hints_sha256": CURRENT_LIFECYCLE_HINTS_SHA256,
        "full_lifecycle_finalizer_gate_passed": True,
        "lifecycle_coverage": coverage.manifest_metadata(),
        "lifecycle_candidate_set_sha256": coverage.candidate_set_sha256,
        "lifecycle_resolution_set_sha256": coverage.resolution_set_sha256,
        "network_accessed": False,
        "eodhd_calls": 0,
        "r2_accessed": False,
        "inherits_parent": False,
        **coverage.manifest_metadata(),
    }
    security_ids = {value.security_id for value in candidates}
    finalizer_compatibility = {
        "frozen_report_candidate_count": 182,
        "candidate_raw_count_after_repair": len(raw_candidates),
        "candidate_expanded_count_after_repair": len(candidates),
        "old_symc_candidate_present_after_repair": OLD_SECURITY_ID in security_ids,
        "canonical_nlok_candidate_present_after_repair": (
            CANONICAL_SECURITY_ID in security_ids
        ),
        "frozen_report_is_stale_after_repair": True,
        "existing_finalizer_fails_before_write": True,
        "generic_candidate_tolerance_added": False,
    }
    if finalizer_compatibility != {
        "frozen_report_candidate_count": 182,
        "candidate_raw_count_after_repair": 180,
        "candidate_expanded_count_after_repair": 181,
        "old_symc_candidate_present_after_repair": False,
        "canonical_nlok_candidate_present_after_repair": True,
        "frozen_report_is_stale_after_repair": True,
        "existing_finalizer_fails_before_write": True,
        "generic_candidate_tolerance_added": False,
    }:
        raise RuntimeError("SYMC/NLOK finalizer candidate-set delta changed.")
    return FreshLifecycleEvidence(
        content=content,
        sha256=digest,
        object_path=object_path,
        archive_row=archive_row,
        coverage=coverage,
        metadata=metadata,
        finalizer_compatibility=finalizer_compatibility,
    )


def _archived_report_payload(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    digest: str,
) -> bytes:
    rows = source_archive.loc[
        source_archive["archive_id"].map(_text).str.lower().eq(digest)
        & source_archive["source_hash"].map(_text).str.lower().eq(digest)
        & source_archive["source"].map(_text).eq("lifecycle_evidence_report")
    ]
    if len(rows) != 1:
        raise RuntimeError("Lifecycle evidence report archive binding is not unique.")
    path = (repository.root / _text(rows.iloc[0].get("object_path"))).resolve()
    root = repository.root.resolve()
    if root != path and root not in path.parents:
        raise RuntimeError("Lifecycle evidence report path escapes the repository.")
    if not path.is_file():
        raise RuntimeError("Lifecycle evidence report archive payload is missing.")
    encoded = path.read_bytes()
    try:
        payload = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    except Exception as exc:
        raise RuntimeError("Lifecycle evidence report archive payload is invalid.") from exc
    if sha256_bytes(payload) != digest:
        raise RuntimeError("Lifecycle evidence report archive payload hash changed.")
    return payload


def _exact_applied_repair(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    *,
    candidate_frames: Mapping[str, pd.DataFrame] | None = None,
) -> bool:
    lifecycle_manifest = repository.manifest_for_version(
        "lifecycle_resolutions",
        release.dataset_versions["lifecycle_resolutions"],
    )
    if lifecycle_manifest.metadata.get("operation") != OPERATION:
        return False
    metadata = lifecycle_manifest.metadata
    input_release_version = _text(metadata.get("input_release_version"))
    input_versions = metadata.get("input_versions")
    evidence_sha256 = _text(metadata.get("evidence_report_sha256")).lower()
    evidence_object_path = _text(metadata.get("evidence_report_object_path"))
    source_report_sha256 = _text(
        metadata.get("current_source_report_sha256")
    ).lower()
    if not (
        metadata.get("schema") == REPAIR_SCHEMA
        and input_release_version
        and isinstance(input_versions, dict)
        and metadata.get("output_versions") == release.dataset_versions
        and metadata.get("canonical_security_id") == CANONICAL_SECURITY_ID
        and metadata.get("retired_security_id") == OLD_SECURITY_ID
        and metadata.get("canonical_event_id") == CANONICAL_EVENT_ID
        and metadata.get("parent_release_kind")
        == "identity_price_tails_descendant"
        and metadata.get("lifecycle_hints_sha256")
        == CURRENT_LIFECYCLE_HINTS_SHA256
        and metadata.get("full_lifecycle_finalizer_gate_passed") is True
        and len(evidence_sha256) == 64
        and evidence_sha256 in Path(evidence_object_path).name
        and metadata.get("lifecycle_evidence_report_sha256")
        == evidence_sha256
        and metadata.get("lifecycle_evidence_report_object_path")
        == evidence_object_path
        and metadata.get("lifecycle_report_release_version") == release.version
        and metadata.get("lifecycle_report_input_versions")
        == release.dataset_versions
        and len(source_report_sha256) == 64
        and metadata.get("superseded_evidence_report_sha256")
        == source_report_sha256
        and metadata.get("pinned_base_ancestor_report_sha256")
        == BASE_LIFECYCLE_EVIDENCE_REPORT_SHA256
        and metadata.get("lifecycle_coverage")
        == {
            key: metadata.get(key)
            for key in (
                "coverage_gate_version",
                "selection_rule",
                "candidate_set_sha256",
                "resolution_set_sha256",
                "candidate_count",
                "resolution_count",
                "applied_count",
                "exception_count",
                "open_count",
            )
        }
        and metadata.get("lifecycle_candidate_set_sha256")
        == metadata.get("candidate_set_sha256")
        and metadata.get("lifecycle_resolution_set_sha256")
        == metadata.get("resolution_set_sha256")
        and metadata.get("network_accessed") is False
        and metadata.get("eodhd_calls") == 0
        and metadata.get("r2_accessed") is False
    ):
        raise RuntimeError("Applied SYMC/NLOK lifecycle manifest contract changed.")
    for dataset in WRITE_DATASETS:
        manifest = repository.manifest_for_version(
            dataset, release.dataset_versions[dataset]
        )
        item = manifest.metadata
        if not (
            item.get("schema") == REPAIR_SCHEMA
            and item.get("operation") == OPERATION
            and item.get("input_release_version") == input_release_version
            and item.get("input_versions") == input_versions
            and item.get("output_versions") == release.dataset_versions
            and item.get("canonical_security_id") == CANONICAL_SECURITY_ID
            and item.get("retired_security_id") == OLD_SECURITY_ID
            and item.get("canonical_event_id") == CANONICAL_EVENT_ID
            and item.get("parent_release_kind")
            == "identity_price_tails_descendant"
            and item.get("lifecycle_hints_sha256")
            == CURRENT_LIFECYCLE_HINTS_SHA256
            and item.get("full_lifecycle_finalizer_gate_passed") is True
            and item.get("evidence_report_sha256") == evidence_sha256
            and item.get("evidence_report_object_path") == evidence_object_path
            and item.get("network_accessed") is False
            and item.get("eodhd_calls") == 0
            and item.get("r2_accessed") is False
        ):
            raise RuntimeError(f"Applied SYMC/NLOK {dataset} manifest changed.")
    if not _repaired_state(
        frames, require_pins=True, allow_parent_factor_lineage=True
    ):
        raise RuntimeError("Applied SYMC/NLOK data projection changed.")

    try:
        parent_value = repository.objects.get(
            f"releases/{input_release_version}.json"
        )
    except ObjectNotFound as exc:
        raise RuntimeError("Applied SYMC/NLOK parent release is missing.") from exc
    parent = DataRelease.from_bytes(parent_value.data)
    if parent.version != input_release_version or parent.dataset_versions != input_versions:
        raise RuntimeError("Applied SYMC/NLOK parent release binding changed.")
    parent_kind = _parent_release_manifest_kind(repository, parent)
    if parent_kind != "identity_price_tails_descendant":
        raise RuntimeError("Applied SYMC/NLOK parent kind is invalid.")

    lifecycle_frames = candidate_frames or frames
    candidate = _CandidateRepository(
        repository, release.dataset_versions, lifecycle_frames
    )
    candidates = _candidate_values(candidate, release)
    coverage = validate_lifecycle_coverage(
        _candidate_frame(candidates),
        lifecycle_frames["lifecycle_resolutions"],
        lifecycle_frames["corporate_actions"],
        completed_session=release.completed_session,
    )
    for key, value in coverage.manifest_metadata().items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Applied SYMC/NLOK lifecycle metadata changed: {key}."
            )
    if not coverage.valid or coverage.open_count != 0:
        raise RuntimeError("Applied SYMC/NLOK lifecycle coverage is not closed.")
    payload = _archived_report_payload(
        repository, lifecycle_frames["source_archive"], evidence_sha256
    )
    report = json.loads(payload)
    _verify_fresh_report(report, planned_release=release, candidates=candidates)
    finalizer_records = lifecycle_finalizer._validate_full_report(
        lifecycle_finalizer.ReportDocument(
            path=Path(_text(metadata.get("evidence_report_object_path"))),
            content=payload,
            value=report,
        ),
        release,
        candidates,
        hints_path=DEFAULT_HINTS_PATH,
    )
    if len(finalizer_records) != 181:
        raise RuntimeError("Applied full lifecycle finalizer report gate changed.")
    if OLD_SECURITY_ID in report["records"]:
        raise RuntimeError("Applied lifecycle report resurrected the old SYMC record.")
    return True


def _finalizer_candidate_set_compatibility(
    repository: LocalDatasetRepository,
    parent_release: DataRelease,
    candidate_release: DataRelease,
    candidate: _CandidateRepository,
) -> dict[str, Any]:
    """Prove that the frozen 182-record report becomes stale, fail-closed.

    This is a plan-time compatibility audit, not a broad finalizer bypass.  The
    exact official-applied expansion is recomputed from the reviewed hints.
    """

    raw = build_lifecycle_candidates(candidate, release=candidate_release)
    specs = load_official_lifecycle_exception_evidence(DEFAULT_HINTS_PATH)
    expanded = include_bound_official_applied_event_candidates(
        raw, candidate, candidate_release, specs
    )
    security_ids = {value.security_id for value in expanded}
    report, _payload, _row = _base_lifecycle_report_document(
        repository,
        parent_release,
        candidate.read_frame("source_archive"),
    )
    report_count = int(report.get("candidate_count", -1))
    if not (
        len(raw) == 180
        and len(expanded) == 181
        and report_count == 182
        and OLD_SECURITY_ID not in security_ids
        and CANONICAL_SECURITY_ID in security_ids
    ):
        raise RuntimeError("SYMC/NLOK finalizer candidate-set delta changed.")
    return {
        "frozen_report_candidate_count": report_count,
        "candidate_raw_count_after_repair": len(raw),
        "candidate_expanded_count_after_repair": len(expanded),
        "old_symc_candidate_present_after_repair": False,
        "canonical_nlok_candidate_present_after_repair": True,
        "frozen_report_is_stale_after_repair": True,
        "existing_finalizer_fails_before_write": True,
        "generic_candidate_tolerance_added": False,
    }


def prepare_repair(repository: LocalDatasetRepository) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    pointer_etags, version_state = _version_state(repository, release)
    lifecycle_manifest = repository.manifest_for_version(
        "lifecycle_resolutions",
        release.dataset_versions["lifecycle_resolutions"],
    )
    is_applied_release = lifecycle_manifest.metadata.get("operation") == OPERATION
    parent_kind: str | None = None
    if is_applied_release:
        # Validate immutable files and the exact repaired identity from bounded
        # Parquet scopes.  Lifecycle reconstruction needs only terminal price
        # summaries, so an idempotency check never reloads both 2M-row tables.
        for dataset in REQUIRED_DATASETS:
            manifest = repository.current_manifest(dataset)
            if (
                manifest is None
                or manifest.version != release.dataset_versions[dataset]
            ):
                raise RuntimeError(
                    f"Applied SYMC/NLOK {dataset} files changed."
                )
        repaired_scopes = _written_identity_scopes(
            repository, release.dataset_versions
        )
        lifecycle_frames = identity_tails._read_candidate_context(
            repository, release
        )
        lifecycle_frames["lifecycle_resolutions"] = repository.read_frame(
            "lifecycle_resolutions",
            release.dataset_versions["lifecycle_resolutions"],
        )
        lifecycle_frames["source_archive"] = repaired_scopes[
            "source_archive"
        ]
        if not _exact_applied_repair(
            repository,
            release,
            repaired_scopes,
            candidate_frames=lifecycle_frames,
        ):
            raise RuntimeError("Applied SYMC/NLOK release is not exact.")
        del repaired_scopes, lifecycle_frames
        gc.collect()
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            version_state=version_state,
            frames={},
            planned_versions={},
            planned_release=None,
            lifecycle_report_content=b"",
            lifecycle_report_object_path="",
            lifecycle_metadata={},
            warnings=release.warnings,
            summary={
                "status": "already_repaired",
                "base_release_version": release.version,
                "canonical_event_id": CANONICAL_EVENT_ID,
                "idempotency_validation_mode": (
                    "manifest_hashes_plus_affected_scopes_and_terminal_summaries"
                ),
                "full_market_price_frames_materialized": False,
                "network_accessed": False,
                "http_attempts": 0,
                "eodhd_calls": 0,
                "r2_accessed": False,
                "writes_performed": False,
            },
        )
    if not is_applied_release:
        # Enforce exact simple7 ancestry before reading any repair inputs.  This
        # also makes the pinned base and legacy-hints states fail closed.
        parent_kind = _parent_release_kind(repository, release, {})

    # Candidate reconstruction needs one terminal row per security, not every
    # daily bar.  Keep that bounded context separate from the exact two-SID
    # repair scopes used by the identity transform.
    parent_context = identity_tails._read_candidate_context(repository, release)
    parent_candidate_repository = _CandidateRepository(
        repository,
        release.dataset_versions,
        parent_context,
    )
    parent_report_candidates = _candidate_values(
        parent_candidate_repository,
        release,
    )
    frames = _load_plan_frames(repository, release)
    parent_report_payload = _descendant_lifecycle_report_document(
        repository,
        release,
        frames,
        parent_candidates=parent_report_candidates,
    )
    proof = _verify_base(
        repository,
        release,
        frames,
        parent_kind=parent_kind,
        parent_candidates=parent_report_candidates,
        parent_report_payload=parent_report_payload,
    )
    parent_report_payload = proof.pop("_parent_lifecycle_report_payload")
    parent_report_candidates = proof.pop("_parent_lifecycle_candidates")
    candidate_frames, summary = prepare_candidate_frames(
        frames,
        completed_session=release.completed_session,
        require_candidate_pins=True,
        parent_release_kind=proof["parent_release_kind"],
        consume_inputs=True,
    )
    del frames
    gc.collect()
    planned_versions = _new_planned_versions(release)
    output_versions = dict(release.dataset_versions)
    output_versions.update(planned_versions)
    planned_release = DataRelease.create(
        release.completed_session,
        output_versions,
        quality=DataQuality.DEGRADED if release.warnings else DataQuality.VALID,
        warnings=release.warnings,
    )
    lifecycle_candidate_frames = _project_lifecycle_candidate_context(
        parent_context,
        candidate_frames,
    )
    candidate = _CandidateRepository(
        repository,
        planned_release.dataset_versions,
        lifecycle_candidate_frames,
    )
    fresh = _fresh_lifecycle_evidence(
        repository,
        release,
        planned_release,
        candidate,
        candidate_frames,
        parent_release_kind=proof["parent_release_kind"],
        parent_report_payload=parent_report_payload,
        parent_report_candidates=parent_report_candidates,
    )
    archive = candidate_frames["source_archive"]
    if archive["archive_id"].map(_text).eq(fresh.sha256).any():
        raise RuntimeError("Fresh lifecycle report archive ID already exists.")
    archive_row = pd.DataFrame([fresh.archive_row]).reindex(columns=archive.columns)
    candidate_frames["source_archive"] = pd.concat(
        [archive, archive_row], ignore_index=True
    )
    validate_dataset(
        "source_archive",
        candidate_frames["source_archive"],
        completed_session=release.completed_session,
        incomplete_action_policy="warn",
    ).raise_for_errors()
    lifecycle_candidate_frames["source_archive"] = candidate_frames[
        "source_archive"
    ]
    lifecycle_candidate_frames["lifecycle_resolutions"] = candidate_frames[
        "lifecycle_resolutions"
    ]

    expected_output_row_counts: dict[str, int] = {}
    for dataset in WRITE_DATASETS:
        files = version_state[dataset].get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(
                f"Prepared SYMC/NLOK {dataset} input row inventory is missing."
            )
        input_count = sum(int(item["row_count"]) for item in files)
        expected = (
            input_count - OLD_PRICE_ROWS
            if dataset in HEAVY_WRITE_DATASETS
            else len(candidate_frames[dataset])
        )
        if expected < 0:
            raise RuntimeError(
                f"Prepared SYMC/NLOK {dataset} output row count is invalid."
            )
        expected_output_row_counts[dataset] = expected

    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        version_state=version_state,
        frames={dataset: candidate_frames[dataset] for dataset in WRITE_DATASETS},
        planned_versions=planned_versions,
        planned_release=planned_release,
        lifecycle_report_content=fresh.content,
        lifecycle_report_object_path=fresh.object_path,
        lifecycle_metadata=fresh.metadata,
        warnings=release.warnings,
        candidate_frames=lifecycle_candidate_frames,
        summary={
            **summary,
            "base_release_version": release.version,
            "parent_release_kind": proof["parent_release_kind"],
            "planned_release_version": planned_release.version,
            "planned_versions": planned_versions,
            "audit_sha256": proof["audit_sha256"],
            "finalizer_supersession": proof["finalizer_supersession"],
            "finalizer_candidate_set_compatibility": (
                fresh.finalizer_compatibility
            ),
            "lifecycle_evidence_report": {
                "superseded_sha256": fresh.metadata[
                    "current_source_report_sha256"
                ],
                "pinned_base_ancestor_sha256": (
                    BASE_LIFECYCLE_EVIDENCE_REPORT_SHA256
                ),
                "fresh_sha256": fresh.sha256,
                "object_path": fresh.object_path,
                "candidate_count": fresh.coverage.candidate_count,
                "resolution_count": fresh.coverage.resolution_count,
                "applied_count": fresh.coverage.applied_count,
                "exception_count": fresh.coverage.exception_count,
                "open_count": fresh.coverage.open_count,
                "publication_gate_passed": True,
            },
            "pre_transition_triple_supertrend_diff": proof["signal_diagnostic"][
                "pre_transition_triple_supertrend_diff"
            ],
            "snapshot_validation_passed": True,
            "snapshot_validation_mode": (
                "exact_pinned_parent_transform_plus_dataset_and_lifecycle_gates"
            ),
            "unchanged_baseline_identity_gap_fingerprint": (
                BASELINE_IDENTITY_GAP_FINGERPRINT
            ),
            "expected_output_row_counts": expected_output_row_counts,
            "plan_materialization": (
                "affected_heavy_tables_plus_full_small_tables"
            ),
            "full_market_price_frames_materialized": False,
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
        pending = tuple(recovery.rglob("*.json")) if recovery.exists() else ()
        if pending:
            raise RuntimeError(
                "A SYMC/NLOK recovery marker blocks writes: "
                + ", ".join(str(item) for item in pending)
            )
        transactions = repository.root / TRANSACTION_DIR
        interrupted: list[Path] = []
        if transactions.exists():
            for item in transactions.rglob("*.json"):
                try:
                    status = _text(json.loads(item.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    interrupted.append(item)
        if interrupted:
            raise RuntimeError(
                "An interrupted SYMC/NLOK transaction blocks writes: "
                + ", ".join(str(item) for item in interrupted)
            )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    write_atomic(path, _canonical_json_bytes(dict(value)))


def _assert_inputs_unchanged(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    """Revalidate the exact release, pointers, manifests and immutable files.

    Planning records more than pointer etags.  Consume that complete state here
    before the transaction journal or archive payload is created so an in-place
    manifest/file mutation cannot be published under a still-matching pointer.
    The file validation streams hashes and does not materialize Parquet rows.
    """

    current, release_etag = repository.current_release()
    if not (
        current is not None
        and current.to_bytes() == prepared.release.to_bytes()
        and release_etag == prepared.release_etag
    ):
        raise RuntimeError("Current release changed after SYMC/NLOK planning.")
    if not (
        set(prepared.pointer_etags) == set(REQUIRED_DATASETS)
        and set(prepared.version_state) == set(REQUIRED_DATASETS)
    ):
        raise RuntimeError("Prepared SYMC/NLOK input inventory is incomplete.")
    pointer_etags, version_state = _version_state(repository, current)
    if pointer_etags != dict(prepared.pointer_etags):
        raise RuntimeError("SYMC/NLOK pointer inventory changed after planning.")
    if version_state != dict(prepared.version_state):
        raise RuntimeError("SYMC/NLOK manifest inventory changed after planning.")
    for dataset in REQUIRED_DATASETS:
        manifest = repository.current_manifest(dataset)
        if (
            manifest is None
            or manifest.version != current.dataset_versions[dataset]
        ):
            raise RuntimeError(f"Current {dataset} files changed after planning.")


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_release_bytes: bytes,
    owned_pointer_bytes: Mapping[str, bytes],
) -> tuple[str, ...]:
    """Restore only an exactly owned publication, with an all-or-none preflight."""

    # Inspect every mutable value before restoring any of them.  If another
    # publisher changed even one byte, do not partially roll it back.
    try:
        current_release = repository.objects.get("releases/current.json")
        if current_release.data not in {old_release_bytes, planned_release_bytes}:
            observed = DataRelease.from_bytes(current_release.data)
            raise RuntimeError(
                "Unexpected release during SYMC/NLOK rollback: "
                + observed.version
            )
        current_pointers: dict[str, Any] = {}
        for dataset in reversed(WRITE_DATASETS):
            key = repository.current_key(dataset)
            current = repository.objects.get(key)
            old = old_pointer_bytes[dataset]
            owned = owned_pointer_bytes.get(dataset)
            if current.data != old and (owned is None or current.data != owned):
                pointer = CurrentPointer.from_bytes(current.data)
                raise RuntimeError(
                    f"Unexpected SYMC/NLOK pointer during rollback: "
                    f"{dataset}/{pointer.version}"
                )
            current_pointers[dataset] = current
    except Exception as exc:
        return (f"rollback preflight: {type(exc).__name__}: {exc}",)

    errors: list[str] = []
    try:
        if current_release.data != old_release_bytes:
            repository.objects.put(
                "releases/current.json",
                old_release_bytes,
                if_match=current_release.etag,
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = current_pointers[dataset]
            if current.data != old_pointer_bytes[dataset]:
                repository.objects.put(
                    key, old_pointer_bytes[dataset], if_match=current.etag
                )
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _lifecycle_report_destination(
    repository: LocalDatasetRepository,
    *,
    object_path: str,
    content: bytes,
) -> Path:
    if not object_path or sha256_bytes(content) not in Path(object_path).name:
        raise RuntimeError("Prepared lifecycle report object path is not content-addressed.")
    root = repository.root.resolve()
    destination = (repository.root / object_path).resolve()
    if root != destination and root not in destination.parents:
        raise RuntimeError("Prepared lifecycle report object path escapes repository.")
    return destination


def _persist_lifecycle_report(
    repository: LocalDatasetRepository,
    *,
    object_path: str,
    content: bytes,
) -> bool:
    destination = _lifecycle_report_destination(
        repository,
        object_path=object_path,
        content=content,
    )
    if destination.is_file():
        try:
            observed = (
                gzip.decompress(destination.read_bytes())
                if destination.suffix == ".gz"
                else destination.read_bytes()
            )
        except Exception as exc:
            raise RuntimeError("Existing lifecycle report payload is invalid.") from exc
        if observed != content:
            raise RuntimeError("Existing lifecycle report payload conflicts.")
        return False
    write_atomic(destination, gzip.compress(content, mtime=0))
    if gzip.decompress(destination.read_bytes()) != content:
        raise RuntimeError("Lifecycle report payload verification failed.")
    return True


def _remove_created_lifecycle_report(
    repository: LocalDatasetRepository,
    *,
    object_path: str,
    content: bytes,
    created: bool,
) -> tuple[str, ...]:
    """Remove only the exact archive payload created by this transaction."""

    if not created:
        return ()
    try:
        destination = _lifecycle_report_destination(
            repository,
            object_path=object_path,
            content=content,
        )
        if not destination.is_file():
            raise RuntimeError("Created lifecycle report payload disappeared.")
        observed = (
            gzip.decompress(destination.read_bytes())
            if destination.suffix == ".gz"
            else destination.read_bytes()
        )
        if observed != content:
            raise RuntimeError("Created lifecycle report payload changed.")
        destination.unlink()
        return ()
    except Exception as exc:
        return (f"archive_payload: {type(exc).__name__}: {exc}",)


def _capture_owned_pointer(
    repository: LocalDatasetRepository,
    *,
    dataset: str,
    version: str,
    manifest_bytes: bytes,
) -> bytes:
    """Capture the exact pointer bytes created by one successful dataset write."""

    value = repository.objects.get(repository.current_key(dataset))
    pointer = CurrentPointer.from_bytes(value.data)
    expected_path = f"{repository.version_prefix(dataset, version)}/manifest.json"
    if not (
        pointer.dataset == dataset
        and pointer.version == version
        and pointer.manifest_path == expected_path
        and pointer.manifest_sha256 == sha256_bytes(manifest_bytes)
    ):
        raise RuntimeError(f"Written SYMC/NLOK pointer is not exact: {dataset}.")
    return value.data


def _written_identity_scopes(
    repository: LocalDatasetRepository,
    versions: Mapping[str, str],
) -> dict[str, pd.DataFrame]:
    """Read only the two affected identities from written Parquet datasets."""

    security_ids = {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
    frames = {
        dataset: identity_tails._read_security_subset(
            repository,
            dataset,
            versions[dataset],
            security_ids,
        )
        for dataset in IDENTITY_WRITE_DATASETS
    }
    # source_archive is small and its full row inventory binds the report row.
    frames["source_archive"] = repository.read_frame(
        "source_archive", versions["source_archive"]
    )
    return frames


def _expected_output_row_counts(prepared: PreparedRepair) -> dict[str, int]:
    value = prepared.summary.get("expected_output_row_counts")
    if not isinstance(value, Mapping) or set(value) != set(WRITE_DATASETS):
        raise RuntimeError("Prepared SYMC/NLOK output row inventory is incomplete.")
    output: dict[str, int] = {}
    for dataset in WRITE_DATASETS:
        count = value.get(dataset)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise RuntimeError(
                f"Prepared SYMC/NLOK {dataset} output row count is invalid."
            )
        output[dataset] = count
    return output


@dataclass(frozen=True)
class _HeavyParquetInspection:
    path: Path
    relative_path: str
    schema: Any = field(repr=False, compare=False)
    row_count: int
    retired_rows: int
    min_session: str
    max_session: str
    size_bytes: int
    sha256: str


def _arrow_modules():
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pyarrow is required for the bounded SYMC/NLOK Parquet rewrite."
        ) from exc
    return pc, pq


def _session_date(value: Any, *, dataset: str, path: str) -> date:
    if value is None:
        raise RuntimeError(f"SYMC/NLOK {dataset} has a null session: {path}.")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if len(text) < 10 or text[4:5] != "-" or text[7:8] != "-":
            raise RuntimeError(
                f"SYMC/NLOK {dataset} has a non-ISO session: {path}."
            )
        try:
            return date.fromisoformat(text[:10])
        except ValueError as exc:
            raise RuntimeError(
                f"SYMC/NLOK {dataset} has an invalid session: {path}."
            ) from exc
    raise RuntimeError(
        f"SYMC/NLOK {dataset} has an unsupported session scalar: {path}."
    )


def _partition_key(
    dataset: str,
    relative_path: str,
) -> tuple[int, ...]:
    spec = dataset_spec(dataset)
    parts = Path(relative_path).parts
    if not (
        spec.partition_columns
        and len(parts) == len(spec.partition_columns) + 1
        and parts[-1].endswith(".parquet")
    ):
        raise RuntimeError(
            f"SYMC/NLOK {dataset} partition path changed: {relative_path}."
        )
    values: list[int] = []
    for name, part in zip(spec.partition_columns, parts[:-1]):
        prefix = name + "="
        if not part.startswith(prefix):
            raise RuntimeError(
                f"SYMC/NLOK {dataset} partition path changed: {relative_path}."
            )
        try:
            value = int(part[len(prefix) :])
        except ValueError as exc:
            raise RuntimeError(
                f"SYMC/NLOK {dataset} partition value changed: {relative_path}."
            ) from exc
        if name == "month" and not 1 <= value <= 12:
            raise RuntimeError(
                f"SYMC/NLOK {dataset} partition month changed: {relative_path}."
            )
        values.append(value)
    return tuple(values)


def _inspect_heavy_parquet_file(
    dataset: str,
    path: Path,
    relative_path: str,
    *,
    retired_security_id: str,
    batch_size: int = 16_384,
) -> _HeavyParquetInspection:
    """Stream one Parquet file and prove its physical dataset contract."""

    if dataset not in HEAVY_WRITE_DATASETS:
        raise RuntimeError(f"SYMC/NLOK streaming inspection forbids {dataset}.")
    if batch_size != 16_384:
        raise RuntimeError("SYMC/NLOK heavy rewrite batch size must remain 16384.")
    if not path.is_file():
        raise RuntimeError(f"SYMC/NLOK {dataset} source file is missing: {relative_path}.")
    spec = dataset_spec(dataset)
    if not (
        spec.primary_key == ("security_id", "session")
        and spec.date_columns
        and spec.date_columns[0] == "session"
    ):
        raise RuntimeError(f"SYMC/NLOK {dataset} physical-key contract changed.")
    partition = _partition_key(dataset, relative_path)
    pc, pq = _arrow_modules()
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if not set(spec.required_columns).issubset(schema.names):
        raise RuntimeError(
            f"SYMC/NLOK {dataset} Arrow schema lacks required columns: "
            f"{relative_path}."
        )
    if "security_id" not in schema.names or "session" not in schema.names:
        raise RuntimeError(
            f"SYMC/NLOK {dataset} physical key is missing: {relative_path}."
        )

    row_count = 0
    retired_rows = 0
    minimum: date | None = None
    maximum: date | None = None
    previous_key: tuple[str, str] | None = None
    security_index = schema.get_field_index("security_id")
    session_index = schema.get_field_index("session")
    for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
        if not batch.schema.equals(schema, check_metadata=True):
            raise RuntimeError(
                f"SYMC/NLOK {dataset} batch schema changed: {relative_path}."
            )
        security_values = batch.column(security_index).to_pylist()
        session_values = batch.column(session_index).to_pylist()
        if len(security_values) != len(session_values):  # pragma: no cover
            raise RuntimeError(f"SYMC/NLOK {dataset} batch length changed.")
        for security_id, raw_session in zip(security_values, session_values):
            if security_id is None:
                raise RuntimeError(
                    f"SYMC/NLOK {dataset} has a null security_id: {relative_path}."
                )
            if not isinstance(security_id, str) or not security_id:
                raise RuntimeError(
                    f"SYMC/NLOK {dataset} has an invalid security_id: "
                    f"{relative_path}."
                )
            session = _session_date(
                raw_session,
                dataset=dataset,
                path=relative_path,
            )
            current_key = (security_id, session.isoformat())
            if previous_key is not None and current_key <= previous_key:
                raise RuntimeError(
                    f"SYMC/NLOK {dataset} physical PK order changed: "
                    f"{relative_path}."
                )
            previous_key = current_key
            expected_partition = tuple(
                session.year if name == "year" else session.month
                for name in spec.partition_columns
            )
            if expected_partition != partition:
                raise RuntimeError(
                    f"SYMC/NLOK {dataset} row escaped its partition: "
                    f"{relative_path}."
                )
            minimum = session if minimum is None or session < minimum else minimum
            maximum = session if maximum is None or session > maximum else maximum
            retired_rows += int(security_id == retired_security_id)
            row_count += 1

    if parquet.metadata.num_rows != row_count:
        raise RuntimeError(
            f"SYMC/NLOK {dataset} Parquet row metadata changed: {relative_path}."
        )
    return _HeavyParquetInspection(
        path=path,
        relative_path=relative_path,
        schema=schema,
        row_count=row_count,
        retired_rows=retired_rows,
        min_session=minimum.isoformat() if minimum is not None else "",
        max_session=maximum.isoformat() if maximum is not None else "",
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def _assert_manifest_file_matches_inspection(
    dataset: str,
    item: ManifestFile,
    inspected: _HeavyParquetInspection,
) -> None:
    actual = (
        inspected.relative_path,
        inspected.sha256,
        inspected.size_bytes,
        inspected.row_count,
        inspected.min_session,
        inspected.max_session,
    )
    expected = (
        item.path,
        item.sha256,
        item.size_bytes,
        item.row_count,
        item.min_session,
        item.max_session,
    )
    if actual != expected:
        raise RuntimeError(
            f"SYMC/NLOK {dataset} manifest/Parquet inventory changed: {item.path}."
        )


def _expected_parent_file_inventory(
    expected_files: Any,
) -> tuple[tuple[str, str, int, int], ...]:
    if not isinstance(expected_files, list) or not expected_files:
        raise RuntimeError("Prepared SYMC/NLOK heavy parent inventory is missing.")
    output: list[tuple[str, str, int, int]] = []
    for item in expected_files:
        if not isinstance(item, Mapping):
            raise RuntimeError("Prepared SYMC/NLOK heavy parent inventory is invalid.")
        try:
            output.append(
                (
                    str(item["path"]),
                    str(item["sha256"]),
                    int(item["size_bytes"]),
                    int(item["row_count"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Prepared SYMC/NLOK heavy parent inventory is invalid."
            ) from exc
    return tuple(output)


def _scan_heavy_parent_files(
    repository: LocalDatasetRepository,
    *,
    dataset: str,
    parent_version: str,
    expected_parent_files: Any,
    retired_security_id: str,
    expected_removed_rows: int,
    expected_output_rows: int,
) -> tuple[DatasetManifest, tuple[_HeavyParquetInspection, ...]]:
    """Complete the all-file read-only pass before staging is created."""

    manifest = repository.manifest_for_version(dataset, parent_version)
    if not (
        manifest.dataset == dataset
        and manifest.version == parent_version
        and not bool(manifest.metadata.get("inherits_parent"))
        and manifest.files
    ):
        raise RuntimeError(f"SYMC/NLOK {dataset} parent manifest is not standalone.")
    expected_inventory = _expected_parent_file_inventory(expected_parent_files)
    observed_inventory = tuple(
        (item.path, item.sha256, item.size_bytes, item.row_count)
        for item in manifest.files
    )
    if observed_inventory != expected_inventory:
        raise RuntimeError(f"SYMC/NLOK {dataset} parent manifest inventory changed.")
    paths = tuple(item.path for item in manifest.files)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise RuntimeError(f"SYMC/NLOK {dataset} manifest file order changed.")
    partitions = tuple(_partition_key(dataset, item.path) for item in manifest.files)
    if len(partitions) != len(set(partitions)):
        raise RuntimeError(f"SYMC/NLOK {dataset} partition inventory changed.")

    version_root = repository.root / repository.version_prefix(
        dataset, parent_version
    )
    inspections: list[_HeavyParquetInspection] = []
    expected_schema: Any | None = None
    total_rows = 0
    removed_rows = 0
    for item in manifest.files:
        inspected = _inspect_heavy_parquet_file(
            dataset,
            version_root / item.path,
            item.path,
            retired_security_id=retired_security_id,
        )
        _assert_manifest_file_matches_inspection(dataset, item, inspected)
        if expected_schema is None:
            expected_schema = inspected.schema
        elif not inspected.schema.equals(expected_schema, check_metadata=True):
            raise RuntimeError(
                f"SYMC/NLOK {dataset} parent Arrow schemas are inconsistent."
            )
        inspections.append(inspected)
        total_rows += inspected.row_count
        removed_rows += inspected.retired_rows
    if removed_rows != expected_removed_rows:
        raise RuntimeError(
            f"SYMC/NLOK {dataset} retired row count changed: "
            f"{removed_rows} != {expected_removed_rows}."
        )
    if total_rows - removed_rows != expected_output_rows:
        raise RuntimeError(
            f"SYMC/NLOK {dataset} output row count changed: "
            f"{total_rows - removed_rows} != {expected_output_rows}."
        )
    return manifest, tuple(inspections)


def _rewrite_affected_heavy_file(
    dataset: str,
    inspected: _HeavyParquetInspection,
    destination: Path,
    *,
    retired_security_id: str,
    source_path: Path,
) -> None:
    pc, pq = _arrow_modules()
    source = pq.ParquetFile(source_path)
    if not source.schema_arrow.equals(inspected.schema, check_metadata=True):
        raise RuntimeError(
            f"SYMC/NLOK {dataset} source schema changed after pre-scan: "
            f"{inspected.relative_path}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen_rows = 0
    removed_rows = 0
    security_index = inspected.schema.get_field_index("security_id")
    with pq.ParquetWriter(
        destination,
        inspected.schema,
        compression="zstd",
    ) as writer:
        for batch in source.iter_batches(batch_size=16_384, use_threads=False):
            if not batch.schema.equals(inspected.schema, check_metadata=True):
                raise RuntimeError(
                    f"SYMC/NLOK {dataset} source batch changed after pre-scan: "
                    f"{inspected.relative_path}."
                )
            equal = pc.equal(batch.column(security_index), retired_security_id)
            if equal.null_count:
                raise RuntimeError(
                    f"SYMC/NLOK {dataset} null security_id appeared during rewrite."
                )
            removed_rows += int(pc.sum(equal).as_py() or 0)
            seen_rows += batch.num_rows
            retained = batch.filter(pc.invert(equal))
            if retained.num_rows:
                writer.write_batch(retained, row_group_size=16_384)
    if seen_rows != inspected.row_count or removed_rows != inspected.retired_rows:
        raise RuntimeError(
            f"SYMC/NLOK {dataset} source changed during rewrite: "
            f"{inspected.relative_path}."
        )


def _write_delete_only_heavy_dataset(
    repository: LocalDatasetRepository,
    *,
    dataset: str,
    parent_version: str,
    version: str,
    completed_session: str,
    metadata: Mapping[str, Any],
    expected_pointer_etag: str | None,
    expected_parent_files: Any,
    retired_security_id: str,
    expected_removed_rows: int,
    expected_output_rows: int,
) -> DatasetWriteResult:
    """Publish one standalone heavy version without constructing a pandas frame."""

    if dataset not in HEAVY_WRITE_DATASETS:
        raise RuntimeError(f"SYMC/NLOK heavy writer forbids {dataset}.")
    current, actual_etag = repository.current_pointer(dataset)
    if current is None or current.version != parent_version:
        raise RuntimeError(f"SYMC/NLOK {dataset} parent pointer changed.")
    if expected_pointer_etag is None:
        expected_pointer_etag = actual_etag
    final_prefix = repository.version_prefix(dataset, version)
    final_root = repository.root / final_prefix
    if final_root.exists():
        raise FileExistsError(f"Dataset version already exists: {dataset}/{version}")

    # This is deliberately the only read phase before staging.  It scans every
    # source row/file, validates all manifest and physical invariants, and
    # proves the exact delete cardinality before any filesystem mutation.
    parent_manifest, inspected_files = _scan_heavy_parent_files(
        repository,
        dataset=dataset,
        parent_version=parent_version,
        expected_parent_files=expected_parent_files,
        retired_security_id=retired_security_id,
        expected_removed_rows=expected_removed_rows,
        expected_output_rows=expected_output_rows,
    )

    staging = (
        repository.root
        / ".staging"
        / dataset
        / f"{version}-{uuid.uuid4().hex}"
    )
    snapshot_root = staging / ".source-snapshots"
    output_items: list[ManifestFile] = []
    try:
        for parent_item, inspected in zip(
            parent_manifest.files, inspected_files
        ):
            destination = staging / parent_item.path
            if inspected.retired_rows == 0:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(inspected.path, destination)
                if not (
                    destination.stat().st_size == parent_item.size_bytes
                    and sha256_file(destination) == parent_item.sha256
                ):
                    raise RuntimeError(
                        f"SYMC/NLOK {dataset} unaffected copy changed: "
                        f"{parent_item.path}."
                    )
                output_items.append(parent_item)
                continue

            # Bind the rewrite to an immutable, hash-verified copy of the
            # source file.  Re-reading the live parent after pre-scan would
            # otherwise leave a mutate/rewrite/restore TOCTOU window.
            snapshot = snapshot_root / parent_item.path
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inspected.path, snapshot)
            if not (
                snapshot.stat().st_size == parent_item.size_bytes
                and sha256_file(snapshot) == parent_item.sha256
            ):
                raise RuntimeError(
                    f"SYMC/NLOK {dataset} affected snapshot changed: "
                    f"{parent_item.path}."
                )
            _rewrite_affected_heavy_file(
                dataset,
                inspected,
                destination,
                retired_security_id=retired_security_id,
                source_path=snapshot,
            )
            output = _inspect_heavy_parquet_file(
                dataset,
                destination,
                parent_item.path,
                retired_security_id=retired_security_id,
            )
            if not (
                output.schema.equals(inspected.schema, check_metadata=True)
                and output.row_count
                == inspected.row_count - inspected.retired_rows
                and output.retired_rows == 0
            ):
                raise RuntimeError(
                    f"SYMC/NLOK {dataset} affected output changed: "
                    f"{parent_item.path}."
                )
            output_items.append(
                ManifestFile(
                    path=parent_item.path,
                    sha256=output.sha256,
                    size_bytes=output.size_bytes,
                    row_count=output.row_count,
                    min_session=output.min_session,
                    max_session=output.max_session,
                )
            )

        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
        if sum(item.row_count for item in output_items) != expected_output_rows:
            raise RuntimeError(f"SYMC/NLOK {dataset} staged row count changed.")
        # Close the scan/rewrite race window before the immutable version is
        # installed.  This is streaming hashing only; no tabular frame exists.
        for parent_item, inspected in zip(
            parent_manifest.files, inspected_files
        ):
            if not (
                inspected.path.stat().st_size == parent_item.size_bytes
                and sha256_file(inspected.path) == parent_item.sha256
            ):
                raise RuntimeError(
                    f"SYMC/NLOK {dataset} source changed after pre-scan: "
                    f"{parent_item.path}."
                )
        if repository.manifest_for_version(
            dataset, parent_version
        ).to_bytes() != parent_manifest.to_bytes():
            raise RuntimeError(
                f"SYMC/NLOK {dataset} parent manifest changed after pre-scan."
            )

        manifest_metadata = {
            **dict(parent_manifest.metadata),
            **dict(metadata),
            "inherits_parent": False,
        }
        for forbidden in (
            "_logical_quality",
            "_logical_warnings",
            "_unresolved_action_count",
        ):
            if forbidden in manifest_metadata:
                raise RuntimeError(
                    f"SYMC/NLOK heavy metadata override is forbidden: {forbidden}."
                )
        manifest = DatasetManifest.create(
            dataset=dataset,
            version=version,
            completed_session=completed_session,
            files=tuple(output_items),
            quality=parent_manifest.quality,
            parent_version=parent_manifest.version,
            source_mode=parent_manifest.source_mode,
            official_coverage_start=parent_manifest.official_coverage_start,
            official_coverage_end=parent_manifest.official_coverage_end,
            unresolved_action_count=parent_manifest.unresolved_action_count,
            conflict_count=parent_manifest.conflict_count,
            warnings=parent_manifest.warnings,
            metadata=manifest_metadata,
        )
        write_atomic(staging / "manifest.json", manifest.to_bytes())
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    manifest_path = f"{final_prefix}/manifest.json"
    pointer = CurrentPointer.create(manifest, manifest_path)
    report = ValidationReport(dataset)
    try:
        repository.objects.put(
            repository.current_key(dataset),
            pointer.to_bytes(),
            if_match=expected_pointer_etag,
            if_none_match=expected_pointer_etag is None,
        )
    except ConditionalWriteFailed:
        conflict_path = f"conflicts/{dataset}/{version}/manifest.json"
        repository.objects.put(
            conflict_path,
            manifest.to_bytes(),
            if_none_match=True,
        )
        return DatasetWriteResult(
            manifest,
            report,
            conflict=True,
            conflict_path=conflict_path,
        )
    return DatasetWriteResult(manifest, report)


def _write_prepared_heavy_dataset(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
    *,
    metadata: Mapping[str, Any],
) -> DatasetWriteResult:
    if dataset not in HEAVY_WRITE_DATASETS:
        raise RuntimeError(f"SYMC/NLOK prepared heavy writer forbids {dataset}.")
    expected_counts = _expected_output_row_counts(prepared)
    if expected_counts[dataset] != 2_095_793:
        raise RuntimeError(f"Prepared SYMC/NLOK {dataset} output total changed.")
    planned_scope = prepared.frames.get(dataset)
    if planned_scope is None or not (
        len(planned_scope) == CANONICAL_PRICE_ROWS
        and planned_scope["security_id"]
        .map(_text)
        .eq(CANONICAL_SECURITY_ID)
        .all()
    ):
        raise RuntimeError(f"Prepared sparse SYMC/NLOK {dataset} scope changed.")
    return _write_delete_only_heavy_dataset(
        repository,
        dataset=dataset,
        parent_version=prepared.release.dataset_versions[dataset],
        version=prepared.planned_versions[dataset],
        completed_session=prepared.release.completed_session,
        metadata=metadata,
        expected_pointer_etag=prepared.pointer_etags[dataset],
        expected_parent_files=prepared.version_state[dataset].get("files"),
        retired_security_id=OLD_SECURITY_ID,
        expected_removed_rows=OLD_PRICE_ROWS,
        expected_output_rows=expected_counts[dataset],
    )


def _prepared_small_write_frame(
    prepared: PreparedRepair,
    dataset: str,
) -> pd.DataFrame:
    if dataset in HEAVY_WRITE_DATASETS:
        raise RuntimeError(f"SYMC/NLOK heavy pandas materialization is forbidden: {dataset}.")
    expected_counts = _expected_output_row_counts(prepared)
    frame = prepared.frames.get(dataset)
    if frame is None or len(frame) != expected_counts[dataset]:
        raise RuntimeError(f"Prepared SYMC/NLOK {dataset} frame row count changed.")
    return frame


def _verify_written_publication(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    planned_release: DataRelease,
    manifest_bytes: Mapping[str, bytes],
) -> None:
    """Verify written files without reloading both multi-million-row tables."""

    if set(manifest_bytes) != set(WRITE_DATASETS):
        raise RuntimeError("Written SYMC/NLOK manifest inventory is incomplete.")
    expected_counts = _expected_output_row_counts(prepared)
    for dataset in WRITE_DATASETS:
        pointer, _etag = repository.current_pointer(dataset)
        manifest = repository.current_manifest(dataset)
        if not (
            pointer is not None
            and pointer.version == planned_release.dataset_versions[dataset]
            and manifest is not None
            and manifest.version == planned_release.dataset_versions[dataset]
            and manifest.to_bytes() == manifest_bytes[dataset]
            and sum(item.row_count for item in manifest.files)
            == expected_counts[dataset]
        ):
            raise RuntimeError(f"Written SYMC/NLOK manifest changed: {dataset}.")

    written = _written_identity_scopes(
        repository, planned_release.dataset_versions
    )
    try:
        if not _repaired_state(
            written,
            require_pins=True,
            allow_parent_factor_lineage=True,
        ):
            raise RuntimeError("Written SYMC/NLOK snapshot failed exact replay.")
        expected_archive = _frame_sha256(
            prepared.frames["source_archive"], sort_by=("archive_id",)
        )
        observed_archive = _frame_sha256(
            written["source_archive"], sort_by=("archive_id",)
        )
        if observed_archive != expected_archive:
            raise RuntimeError("Written lifecycle archive inventory changed.")
    finally:
        del written
        gc.collect()

    # The full lifecycle candidate/report gates reuse the already validated
    # planned frames.  Actual Parquet integrity and the affected identity rows
    # were independently checked above, so no second full price/factor reload
    # is needed here.
    if not _exact_applied_repair(
        repository,
        planned_release,
        prepared.frames,
        candidate_frames=prepared.candidate_frames,
    ):
        raise RuntimeError("Written lifecycle publication projection failed.")


def _assert_committed_publication(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    committed: DataRelease,
    owned_pointer_bytes: Mapping[str, bytes],
) -> None:
    """Check commit visibility without re-running the memory-heavy plan."""

    if set(owned_pointer_bytes) != set(WRITE_DATASETS):
        raise RuntimeError("Committed SYMC/NLOK pointer inventory is incomplete.")
    current, _etag = repository.current_release()
    if current is None or current.to_bytes() != committed.to_bytes():
        raise RuntimeError("Committed SYMC/NLOK release is not current.")
    for dataset, version in committed.dataset_versions.items():
        pointer, pointer_etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"Committed SYMC/NLOK pointer mismatch: {dataset}.")
        if dataset in owned_pointer_bytes:
            observed = repository.objects.get(repository.current_key(dataset))
            if observed.data != owned_pointer_bytes[dataset]:
                raise RuntimeError(
                    f"Committed SYMC/NLOK pointer bytes changed: {dataset}."
                )
        elif pointer_etag != prepared.pointer_etags.get(dataset):
            raise RuntimeError(
                f"Out-of-scope pointer changed during SYMC/NLOK apply: {dataset}."
            )


def _commit_exact_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
    *,
    expected_etag: str | None,
) -> DataRelease:
    payload = release.to_bytes()
    immutable_key = f"releases/{release.version}.json"
    try:
        repository.objects.put(immutable_key, payload, if_none_match=True)
    except ConditionalWriteFailed:
        try:
            existing = repository.objects.get(immutable_key)
        except ObjectNotFound as exc:  # pragma: no cover - race guard
            raise RuntimeError("Prepared immutable release conflicted.") from exc
        if existing.data != payload:
            raise RuntimeError("Prepared immutable release version conflicts.")
    repository.objects.put(
        "releases/current.json",
        payload,
        if_match=expected_etag,
        if_none_match=expected_etag is None,
    )
    return release


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    failure_injector: FailureInjector | None = None,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_repaired":
        return dict(prepared.summary)
    inject = failure_injector or (lambda _stage: None)
    with _exclusive_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"SYMC/NLOK pointer changed before apply: {dataset}")
            old_pointers[dataset] = value.data

        planned = dict(prepared.planned_versions)
        planned_release = prepared.planned_release
        if (
            planned_release is None
            or set(planned) != set(WRITE_DATASETS)
            or set(prepared.frames) != set(WRITE_DATASETS)
            or set(prepared.candidate_frames)
            != set(LIFECYCLE_CANDIDATE_DATASETS)
            or planned_release.dataset_versions
            != {**dict(prepared.release.dataset_versions), **planned}
            or not prepared.lifecycle_report_content
            or not prepared.lifecycle_report_object_path
            or prepared.lifecycle_metadata.get("parent_release_kind")
            != "identity_price_tails_descendant"
            or prepared.lifecycle_metadata.get("lifecycle_hints_sha256")
            != CURRENT_LIFECYCLE_HINTS_SHA256
            or prepared.lifecycle_metadata.get(
                "full_lifecycle_finalizer_gate_passed"
            )
            is not True
            or prepared.summary.get("plan_materialization")
            != "affected_heavy_tables_plus_full_small_tables"
            or prepared.summary.get("full_market_price_frames_materialized")
            is not False
        ):
            raise RuntimeError("Prepared SYMC/NLOK transaction contract is incomplete.")
        _expected_output_row_counts(prepared)
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": TRANSACTION_SCHEMA,
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value).decode("ascii")
                for dataset, value in old_pointers.items()
            },
            "planned_versions": planned,
            "planned_release_version": planned_release.version,
            "planned_release_sha256": sha256_bytes(planned_release.to_bytes()),
            "lifecycle_report_object_path": prepared.lifecycle_report_object_path,
            "lifecycle_report_sha256": sha256_bytes(
                prepared.lifecycle_report_content
            ),
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        archive_created = False
        owned_pointer_bytes: dict[str, bytes] = {}
        written_manifest_bytes: dict[str, bytes] = {}
        try:
            archive_destination = _lifecycle_report_destination(
                repository,
                object_path=prepared.lifecycle_report_object_path,
                content=prepared.lifecycle_report_content,
            )
            archive_preexisting = archive_destination.exists()
            try:
                archive_created = _persist_lifecycle_report(
                    repository,
                    object_path=prepared.lifecycle_report_object_path,
                    content=prepared.lifecycle_report_content,
                )
            except BaseException:
                # `_persist_lifecycle_report` may fail during its post-write
                # readback.  Preserve creation ownership so rollback either
                # removes the exact bytes or emits a recovery marker.
                archive_created = (
                    not archive_preexisting and archive_destination.exists()
                )
                raise
            inject("after_archive_payload")
            versions = dict(prepared.release.dataset_versions)
            common_metadata = {
                "schema": REPAIR_SCHEMA,
                "operation": OPERATION,
                "input_release_version": prepared.release.version,
                "input_versions": dict(prepared.release.dataset_versions),
                "output_versions": dict(planned_release.dataset_versions),
                "canonical_security_id": CANONICAL_SECURITY_ID,
                "retired_security_id": OLD_SECURITY_ID,
                "canonical_event_id": CANONICAL_EVENT_ID,
                "evidence_report_sha256": sha256_bytes(
                    prepared.lifecycle_report_content
                ),
                "evidence_report_object_path": (
                    prepared.lifecycle_report_object_path
                ),
                "parent_release_kind": prepared.lifecycle_metadata.get(
                    "parent_release_kind"
                ),
                "lifecycle_hints_sha256": prepared.lifecycle_metadata.get(
                    "lifecycle_hints_sha256"
                ),
                "full_lifecycle_finalizer_gate_passed": (
                    prepared.lifecycle_metadata.get(
                        "full_lifecycle_finalizer_gate_passed"
                    )
                ),
                "strict_symc_nlok_gate": "passed",
                "network_accessed": False,
                "eodhd_calls": 0,
                "r2_accessed": False,
                "inherits_parent": False,
            }
            for dataset in WRITE_DATASETS:
                parent_metadata = dict(
                    repository.manifest_for_version(
                        dataset,
                        prepared.release.dataset_versions[dataset],
                    ).metadata
                )
                metadata = {**parent_metadata, **common_metadata}
                if dataset == "lifecycle_resolutions":
                    metadata.update(prepared.lifecycle_metadata)
                if dataset in HEAVY_WRITE_DATASETS:
                    result = _write_prepared_heavy_dataset(
                        repository,
                        prepared,
                        dataset,
                        metadata=metadata,
                    )
                else:
                    write_frame = _prepared_small_write_frame(
                        prepared,
                        dataset,
                    )
                    result = repository.write_frame(
                        dataset,
                        write_frame,
                        completed_session=prepared.release.completed_session,
                        incomplete_action_policy="warn",
                        metadata=metadata,
                        expected_pointer_etag=prepared.pointer_etags[dataset],
                        version=planned[dataset],
                    )
                if result.conflict:
                    raise RuntimeError(
                        f"SYMC/NLOK write conflicted: {dataset}/{result.conflict_path}"
                    )
                manifest_bytes = result.manifest.to_bytes()
                if result.manifest.version != planned[dataset]:
                    raise RuntimeError(
                        f"Written SYMC/NLOK version changed: {dataset}."
                    )
                written_manifest_bytes[dataset] = manifest_bytes
                owned_pointer_bytes[dataset] = _capture_owned_pointer(
                    repository,
                    dataset=dataset,
                    version=planned[dataset],
                    manifest_bytes=manifest_bytes,
                )
                versions[dataset] = result.manifest.version
                inject("after_write:" + dataset)
            _verify_written_publication(
                repository,
                prepared,
                planned_release,
                written_manifest_bytes,
            )
            inject("before_commit")
            if versions != planned_release.dataset_versions:
                raise RuntimeError("Written SYMC/NLOK versions changed before commit.")
            committed = _commit_exact_release(
                repository,
                planned_release,
                expected_etag=prepared.release_etag,
            )
            inject("after_commit")
            _assert_committed_publication(
                repository,
                prepared,
                committed,
                owned_pointer_bytes,
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
                **prepared.summary,
                "status": "applied",
                "transaction_id": transaction_id,
                "new_release_version": committed.version,
                "quality": committed.quality,
                "warnings": list(committed.warnings),
                "writes_performed": True,
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_release_bytes=planned_release.to_bytes(),
                owned_pointer_bytes=owned_pointer_bytes,
            )
            # Preserve the payload whenever rollback ownership or any pointer
            # restoration is uncertain.  The recovery marker then retains the
            # exact evidence needed for manual reconciliation.
            if not rollback_errors:
                rollback_errors = (
                    *rollback_errors,
                    *_remove_created_lifecycle_report(
                        repository,
                        object_path=prepared.lifecycle_report_object_path,
                        content=prepared.lifecycle_report_content,
                        created=archive_created,
                    ),
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
                    "SYMC/NLOK rollback failed; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}"
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline canonical SYMC -> NLOK identity repair."
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[[str | Path], LocalDatasetRepository] = (
        LocalDatasetRepository
    ),
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    prepared = prepare_repair(repository)
    if not bool(getattr(args, "apply", False)):
        return dict(prepared.summary)
    return apply_repair(repository, prepared)


def main() -> int:
    args = _parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
