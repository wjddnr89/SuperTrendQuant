#!/usr/bin/env python3
"""Stage FISV's post-transition SEC confirmation into the local archive.

This offline repair does not alter the FI -> FISV corporate action.  The
hash-pinned 2025 Form 8-K remains the primary source for the exact 2025-11-11
transition date; the later 2026 Form 10-Q is archived only as confirmation that
the resulting security is registered on Nasdaq under FISV.

The default command is a read-only plan.  A later explicitly approved
``--apply`` writes only a new immutable ``source_archive`` version and release;
it has no network, EODHD, or R2 path.  The plan emits the exact existing
``reviewed_nonterminal_extractions`` row that may be activated only after the
confirmation payload has been reviewed and archived and the inventory hash is
updated in code.
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
from urllib.parse import urlparse

import pandas as pd

from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_EVIDENCE_DIR = (
    DEFAULT_CACHE_ROOT / "state/issuer_lifecycle/fisv_confirmation"
)
EVIDENCE_REPORT = "fisv_confirmation_evidence.json"
EVIDENCE_SCHEMA = "us_fisv_confirmation_evidence/v1"
DATASET = "source_archive"
OPERATION = "repair_us_fisv_confirmation_evidence"
TRANSACTION_DIR = "transactions/us-fisv-confirmation"
RECOVERY_DIR = "recovery/us-fisv-confirmation"
MAX_RESPONSE_BYTES = 8_000_000

EVENT_ID = "2df5c4c0298e5ff531aaa785146a20cba98d22080c970eabbd841b802ec60e7e"
SECURITY_ID = "US:EODHD:30662d16-c6e4-5187-9721-2b23ac10e4d0"
ACTION_TYPE = "ticker_change"
EFFECTIVE_DATE = "2025-11-11"
NEW_SYMBOL = "FISV"

PRIMARY_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/798354/"
    "000119312525254670/0001193125-25-254670.txt"
)
PRIMARY_SOURCE_HASH = (
    "d4cd0c2f981bfd0be14d2ebccfc8e852a94177e5fba86abe2c027c5510fc07d3"
)
PRIMARY_EXACT_BYTES = 311_689
PRIMARY_REQUIRED_TEXT_GROUPS = (
    (
        "trading will begin on nasdaq at market open on or about "
        "november 11, 2025",
    ),
    ('trade under the symbols, "fisv"', 'trade under the symbols, “fisv”'),
)

CONFIRMATION_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/798354/"
    "000079835426000018/fisv-20260331.htm"
)
CONFIRMATION_REQUIRED_TEXT_GROUPS = (
    ("Fiserv, Inc.",),
    ("March 31, 2026",),
    ("Trading Symbol", "Trading Symbol(s)"),
    ("FISV",),
    ("The Nasdaq Stock Market LLC", "Nasdaq Stock Market LLC"),
)

REVIEWED_EXTRACTION: dict[str, Any] = {
    "event_id": EVENT_ID,
    "security_id": SECURITY_ID,
    "action_type": ACTION_TYPE,
    "effective_date": EFFECTIVE_DATE,
    "new_security_id": SECURITY_ID,
    "new_symbol": NEW_SYMBOL,
    "ratio": None,
    "cash_amount": None,
    "currency": "USD",
    "source_kind": "official_crosscheck",
    "source_url": PRIMARY_SOURCE_URL,
    "source_hash": PRIMARY_SOURCE_HASH,
}


@dataclass(frozen=True)
class ConfirmationEvidence:
    source_url: str
    source_hash: str
    size: int
    filename: str
    retrieved_at: str
    content: bytes

    def object_path(self, completed_session: str) -> str:
        return f"archives/{completed_session}/{self.source_hash}.html.gz"


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etag: str | None
    frame: pd.DataFrame
    confirmation: ConfirmationEvidence
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    raw = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(raw) else raw.date().isoformat()


def _number_is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalized_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    decoded = html.unescape(re.sub(r"<[^>]+>", " ", decoded))
    return re.sub(r"\s+", " ", decoded).strip().casefold()


def _verify_terms(
    content: bytes,
    groups: tuple[tuple[str, ...], ...],
    *,
    label: str,
) -> None:
    if not content or len(content) > MAX_RESPONSE_BYTES:
        raise ValueError(f"{label} size is outside the reviewed envelope.")
    text = _normalized_text(content)
    for alternatives in groups:
        if not any(value.casefold() in text for value in alternatives):
            raise ValueError(
                f"{label} lacks reviewed official term: "
                + " | ".join(alternatives)
            )


def _safe_path(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if path == base or base not in path.parents:
        raise ValueError(f"FISV evidence path escapes its root: {relative}.")
    return path


def _require_exact_confirmation_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        url != CONFIRMATION_SOURCE_URL
        or parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "www.sec.gov"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("FISV confirmation URL is not the reviewed one-URL target.")


def _load_confirmation(evidence_dir: Path) -> ConfirmationEvidence:
    report_path = evidence_dir / EVIDENCE_REPORT
    if not report_path.is_file():
        raise FileNotFoundError(
            "FISV confirmation cache is missing; run the one-URL collector only "
            f"after network fetch is authorized: {report_path}."
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("FISV confirmation report is unreadable.") from exc
    if set(report) != {
        "schema",
        "status",
        "evidence",
        "http_attempts_total",
        "eodhd_calls",
        "r2_accessed",
    }:
        raise ValueError("FISV confirmation report fields are not exact.")
    if (
        report.get("schema") != EVIDENCE_SCHEMA
        or report.get("status") != "collected"
        or report.get("http_attempts_total") != 1
        or report.get("eodhd_calls") != 0
        or report.get("r2_accessed") is not False
    ):
        raise ValueError("FISV confirmation report contract changed.")
    row = report.get("evidence")
    if not isinstance(row, Mapping) or set(row) != {
        "label",
        "source_url",
        "source_hash",
        "size",
        "filename",
        "retrieved_at",
        "form",
        "period_end",
    }:
        raise ValueError("FISV confirmation evidence fields are not exact.")
    if (
        row.get("label") != "fisv_nasdaq_post_transition_confirmation"
        or row.get("form") != "10-Q"
        or row.get("period_end") != "2026-03-31"
    ):
        raise ValueError("FISV confirmation filing identity changed.")
    source_url = _text(row.get("source_url"))
    _require_exact_confirmation_url(source_url)
    digest = _text(row.get("source_hash")).lower()
    filename = _text(row.get("filename"))
    size = row.get("size")
    retrieved_at = _text(row.get("retrieved_at"))
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or filename != f"{digest}.html"
        or not isinstance(size, int)
        or size <= 0
        or size > MAX_RESPONSE_BYTES
        or not retrieved_at.endswith("Z")
    ):
        raise ValueError("FISV confirmation hash/size metadata is invalid.")
    payload_path = _safe_path(evidence_dir, filename)
    if not payload_path.is_file():
        raise FileNotFoundError(f"FISV confirmation payload is missing: {payload_path}.")
    content = payload_path.read_bytes()
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("FISV confirmation payload hash/size changed.")
    _verify_terms(
        content,
        CONFIRMATION_REQUIRED_TEXT_GROUPS,
        label="FISV 2026 Form 10-Q confirmation",
    )
    return ConfirmationEvidence(
        source_url=source_url,
        source_hash=digest,
        size=size,
        filename=filename,
        retrieved_at=retrieved_at,
        content=content,
    )


def _exact_action(actions: pd.DataFrame) -> Mapping[str, Any]:
    rows = actions.loc[actions["event_id"].astype(str).eq(EVENT_ID)]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one FI -> FISV event; observed {len(rows)}."
        )
    row = rows.iloc[0]
    expected = {
        "event_id": EVENT_ID,
        "security_id": SECURITY_ID,
        "action_type": ACTION_TYPE,
        "effective_date": EFFECTIVE_DATE,
        "ex_date": EFFECTIVE_DATE,
        "announcement_date": "2025-10-29",
        "record_date": "",
        "payment_date": "",
        "currency": "USD",
        "new_security_id": SECURITY_ID,
        "new_symbol": NEW_SYMBOL,
        "official": "True",
        "source_url": PRIMARY_SOURCE_URL,
        "source_kind": "official_crosscheck",
        "source_hash": PRIMARY_SOURCE_HASH,
    }
    mismatches = []
    for field, value in expected.items():
        observed = _date(row.get(field)) if field.endswith("_date") else _text(row.get(field))
        if observed != value:
            mismatches.append(field)
    if not _number_is_null(row.get("ratio")):
        mismatches.append("ratio")
    if not _number_is_null(row.get("cash_amount")):
        mismatches.append("cash_amount")
    if mismatches:
        raise ValueError(
            "FI -> FISV corporate action differs from the exact reviewed row: "
            + ", ".join(sorted(set(mismatches)))
            + "."
        )
    return row.to_dict()


def _archive_payload(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    source_hash: str,
    source_url: str,
) -> bytes:
    rows = archive.loc[
        archive["archive_id"].astype(str).eq(source_hash)
        | archive["source_hash"].astype(str).eq(source_hash)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one primary FISV archive row; observed {len(rows)}."
        )
    row = rows.iloc[0]
    if (
        _text(row.get("archive_id")) != source_hash
        or _text(row.get("source_hash")) != source_hash
        or _text(row.get("source_url")) != source_url
    ):
        raise ValueError("Primary FISV archive URL/hash binding changed.")
    path = _safe_path(repository.root, _text(row.get("object_path")))
    if not path.is_file():
        raise FileNotFoundError(f"Primary FISV archive payload is missing: {path}.")
    try:
        content = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError("Primary FISV archive payload is not valid gzip.") from exc
    if len(content) != PRIMARY_EXACT_BYTES or hashlib.sha256(content).hexdigest() != source_hash:
        raise ValueError("Primary FISV archive payload hash/size changed.")
    _verify_terms(
        content,
        PRIMARY_REQUIRED_TEXT_GROUPS,
        label="FISV 2025 Form 8-K scheduling evidence",
    )
    return content


def _confirmation_archive_row(
    archive: pd.DataFrame,
    confirmation: ConfirmationEvidence,
    completed_session: str,
) -> dict[str, Any]:
    row = {column: None for column in archive.columns}
    values = {
        "archive_id": confirmation.source_hash,
        "dataset": "sec_edgar_filing",
        "object_path": confirmation.object_path(completed_session),
        "content_type": "text/html",
        "effective_date": completed_session,
        "source": "sec_edgar_filing",
        "retrieved_at": confirmation.retrieved_at,
        "source_hash": confirmation.source_hash,
        "source_url": confirmation.source_url,
    }
    for field, value in values.items():
        if field not in row:
            raise ValueError(f"source_archive lacks required field: {field}.")
        row[field] = value
    return row


def _rewrite_archive(
    archive: pd.DataFrame,
    confirmation: ConfirmationEvidence,
    *,
    completed_session: str,
) -> tuple[pd.DataFrame, bool]:
    expected = _confirmation_archive_row(archive, confirmation, completed_session)
    related = (
        archive["archive_id"].astype(str).eq(confirmation.source_hash)
        | archive["source_hash"].astype(str).eq(confirmation.source_hash)
        | archive["object_path"].astype(str).eq(
            confirmation.object_path(completed_session)
        )
        | archive["source_url"].fillna("").astype(str).eq(confirmation.source_url)
    )
    rows = archive.loc[related]
    if not rows.empty:
        if len(rows) != 1 or any(
            _text(rows.iloc[0].get(field)) != _text(value)
            for field, value in expected.items()
        ):
            raise ValueError("Conflicting FISV confirmation source_archive row exists.")
        return archive.copy(), False
    addition = pd.DataFrame([expected]).loc[:, archive.columns]
    output = pd.concat([archive, addition], ignore_index=True, sort=False)
    primary_key = list(dataset_spec(DATASET).primary_key)
    if output.duplicated(primary_key, keep=False).any():
        raise ValueError("FISV confirmation would duplicate a source_archive key.")
    return output.reset_index(drop=True), True


def _verify_persisted_confirmation(
    repository: LocalDatasetRepository,
    confirmation: ConfirmationEvidence,
    *,
    completed_session: str,
) -> None:
    path = _safe_path(repository.root, confirmation.object_path(completed_session))
    if not path.is_file():
        raise FileNotFoundError(
            f"Archived FISV confirmation payload is missing: {path}."
        )
    try:
        content = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError("Archived FISV confirmation payload is not valid gzip.") from exc
    if content != confirmation.content:
        raise ValueError("Archived FISV confirmation payload conflicts with the cache.")
    if hashlib.sha256(content).hexdigest() != confirmation.source_hash:
        raise ValueError("Archived FISV confirmation payload hash changed.")


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        archive: pd.DataFrame,
    ):
        self.base = base
        self.versions = dict(versions)
        self.archive = archive.copy()

    def current_manifest(self, dataset: str):
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset == DATASET:
            return self.archive.copy()
        version = self.versions.get(dataset)
        return self.base.read_frame(dataset, version) if version else pd.DataFrame()


def _evidence_roles(confirmation: ConfirmationEvidence) -> dict[str, Any]:
    return {
        "transition_schedule": {
            "form": "8-K",
            "role": "exact_transition_date_and_market_open_schedule",
            "effective_date": EFFECTIVE_DATE,
            "source_url": PRIMARY_SOURCE_URL,
            "source_hash": PRIMARY_SOURCE_HASH,
        },
        "post_transition_confirmation": {
            "form": "10-Q",
            "role": "completed_state_exchange_and_ticker_confirmation_only",
            "period_end": "2026-03-31",
            "source_url": confirmation.source_url,
            "source_hash": confirmation.source_hash,
        },
        "primary_action_source_unchanged": True,
    }


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted({DATASET, "corporate_actions"} - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    pointer, pointer_etag = repository.current_pointer(DATASET)
    if pointer is None or pointer.version != release.dataset_versions[DATASET]:
        raise RuntimeError("source_archive release/current pointer mismatch.")
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    archive = repository.read_frame(DATASET, release.dataset_versions[DATASET])
    _exact_action(actions)
    _archive_payload(
        repository,
        archive,
        source_hash=PRIMARY_SOURCE_HASH,
        source_url=PRIMARY_SOURCE_URL,
    )
    confirmation = _load_confirmation(evidence_dir)
    repaired, changed = _rewrite_archive(
        archive,
        confirmation,
        completed_session=release.completed_session,
    )
    if not changed:
        _verify_persisted_confirmation(
            repository,
            confirmation,
            completed_session=release.completed_session,
        )
    validate_dataset(
        DATASET, repaired, completed_session=release.completed_session
    ).raise_for_errors()
    validate_repository_snapshot(
        _CandidateRepository(repository, release.dataset_versions, repaired)
    ).raise_for_errors()
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etag=pointer_etag,
        frame=repaired,
        confirmation=confirmation,
        summary={
            "status": "validated_offline_plan" if changed else "already_archived",
            "base_release_version": release.version,
            "event_id": EVENT_ID,
            "security_id": SECURITY_ID,
            "source_archive_rows_added": int(changed),
            "source_archive_only": True,
            "corporate_action_changed": False,
            "evidence_roles": _evidence_roles(confirmation),
            "reviewed_nonterminal_extraction": dict(REVIEWED_EXTRACTION),
            "activation_requirements": [
                "apply this source_archive-only repair after explicit approval",
                "review and pin the emitted confirmation SHA-256",
                "append the exact reviewed extraction to us_cross_validation.yaml",
                "update TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256",
                "rerun offline cross-validation before publication",
            ],
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


def readiness_plan(
    repository: LocalDatasetRepository,
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> dict[str, Any]:
    report_path = evidence_dir / EVIDENCE_REPORT
    if not report_path.is_file():
        return {
            "status": "blocked_pending_authorized_one_url_fetch",
            "event_id": EVENT_ID,
            "source_url": CONFIRMATION_SOURCE_URL,
            "reason": (
                "FISV confirmation cache is missing; run the one-URL collector "
                "only after network fetch is authorized: "
                f"{report_path}."
            ),
            "reviewed_nonterminal_extraction": dict(REVIEWED_EXTRACTION),
            "network_accessed": False,
            "writes_performed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    prepared = prepare_repair(repository, evidence_dir=evidence_dir)
    return {**prepared.summary, "mode": "plan", "writes_performed": False}


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    lock_path = repository.root / ".locks/market-store-write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved FISV confirmation recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        "Interrupted FISV confirmation transaction blocks writes: "
                        f"{journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_base_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, release_etag = repository.current_release()
    if (
        release is None
        or release.version != prepared.release.version
        or release_etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after FISV confirmation planning.")
    pointer, pointer_etag = repository.current_pointer(DATASET)
    if (
        pointer is None
        or pointer.version != prepared.release.dataset_versions[DATASET]
        or pointer_etag != prepared.pointer_etag
    ):
        raise RuntimeError("source_archive pointer changed after FISV planning.")


def _persist_confirmation(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    evidence = prepared.confirmation
    if hashlib.sha256(evidence.content).hexdigest() != evidence.source_hash:
        raise ValueError("Prepared FISV confirmation bytes changed before apply.")
    path = _safe_path(
        repository.root,
        evidence.object_path(prepared.release.completed_session),
    )
    if path.is_file():
        try:
            existing = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise ValueError("Persisted FISV confirmation is not valid gzip.") from exc
        if existing != evidence.content:
            raise ValueError("Persisted FISV confirmation bytes conflict.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, gzip.compress(evidence.content, mtime=0))
    if gzip.decompress(path.read_bytes()) != evidence.content:
        raise RuntimeError("FISV confirmation post-write verification failed.")


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: bytes,
    planned_version: str,
    committed_release_version: str,
    old_versions: Mapping[str, str],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            expected_versions = {**dict(old_versions), DATASET: planned_version}
            belongs = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or observed.dataset_versions == expected_versions
            if not belongs:
                raise RuntimeError(f"unexpected release during rollback: {observed.version}")
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    try:
        key = repository.current_key(DATASET)
        current = repository.objects.get(key)
        if current.data != old_pointer_bytes:
            observed = CurrentPointer.from_bytes(current.data)
            if observed.version != planned_version:
                raise RuntimeError(
                    f"unexpected source_archive pointer during rollback: {observed.version}"
                )
            repository.objects.put(key, old_pointer_bytes, if_match=current.etag)
    except Exception as exc:
        errors.append(f"{repository.current_key(DATASET)}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_archived":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    inject_failure = inject_failure or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _assert_base_unchanged(repository, prepared)
        old_release = repository.objects.get("releases/current.json")
        old_pointer = repository.objects.get(repository.current_key(DATASET))
        transaction_id = uuid.uuid4().hex
        planned_version = (
            "fisv-confirmation-"
            f"{prepared.release.completed_session.replace('-', '')}-"
            f"{transaction_id}-{DATASET}"
        )
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_fisv_confirmation_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": base64.b64encode(old_pointer.data).decode("ascii"),
            "planned_source_archive_version": planned_version,
            "confirmation_source_hash": prepared.confirmation.source_hash,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            inject_failure("after_journal")
            _persist_confirmation(repository, prepared)
            inject_failure("after_evidence_write")
            current_manifest = repository.manifest_for_version(
                DATASET, prepared.release.dataset_versions[DATASET]
            )
            metadata = dict(current_manifest.metadata)
            metadata.update(
                {
                    "operation": OPERATION,
                    "fisv_event_id": EVENT_ID,
                    "primary_schedule_source_hash": PRIMARY_SOURCE_HASH,
                    "confirmation_source_hash": prepared.confirmation.source_hash,
                    "source_archive_rows_added": 1,
                    "network_accessed": False,
                    "eodhd_calls": 0,
                    "r2_accessed": False,
                }
            )
            result = repository.write_frame(
                DATASET,
                prepared.frame,
                completed_session=prepared.release.completed_session,
                metadata=metadata,
                expected_pointer_etag=prepared.pointer_etag,
                version=planned_version,
            )
            if result.conflict:
                raise RuntimeError(
                    f"source_archive write conflicted: {result.conflict_path}."
                )
            inject_failure("after_source_archive_write")
            versions = dict(prepared.release.dataset_versions)
            versions[DATASET] = result.manifest.version
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=prepared.release.quality,
                warnings=prepared.release.warnings,
                expected_etag=prepared.release_etag,
            )
            inject_failure("after_release_commit")
            validate_repository_snapshot(repository).raise_for_errors()
            replay = prepare_repair(repository, evidence_dir=evidence_dir)
            if replay.summary["status"] != "already_archived":
                raise RuntimeError("FISV confirmation repair is not idempotent.")
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
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "new_source_archive_version": result.manifest.version,
                "transaction_id": transaction_id,
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointer.data,
                planned_version=planned_version,
                committed_release_version=committed.version if committed else "",
                old_versions=prepared.release.dataset_versions,
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
                    "FISV confirmation rollback was incomplete; recovery marker "
                    f"blocks writes: {recovery}; errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or apply the offline FISV confirmation archive repair."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence_dir = args.evidence_dir or (
        args.cache_root / "state/issuer_lifecycle/fisv_confirmation"
    )
    repository = LocalDatasetRepository(args.cache_root)
    if args.apply:
        prepared = prepare_repair(repository, evidence_dir=evidence_dir)
        result = apply_repair(
            repository, prepared, evidence_dir=evidence_dir
        )
    else:
        result = readiness_plan(repository, evidence_dir=evidence_dir)
        result["mode"] = "plan"
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
