#!/usr/bin/env python3
"""Bind FRC -> FRCB to the exact OCC 52352 PDF, offline.

The current lifecycle row is economically complete, but its provenance points
to a deterministic reviewed JSON extraction.  This repair accepts a locally
downloaded copy of OCC Information Memo 52352, verifies the PDF and every
reviewed term, preserves the legacy extraction for audit history, appends the
raw PDF to ``source_archive``, and rewrites only the ticker-change provenance.

There is deliberately no networking, EODHD, SEC, or R2 code in this module.
The default mode is a read-only plan.  ``--apply`` is an explicit offline,
CAS-guarded transaction with a repository writer lock, journal, rollback, and
an idempotency replay.  A successful visual review and confirmation that the
file came from the exact official URL are mandatory inputs.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import io
import json
import math
import re
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
ACTION_DATASET = "corporate_actions"
ARCHIVE_DATASET = "source_archive"
WRITE_DATASETS = (ARCHIVE_DATASET, ACTION_DATASET)
OPERATION = "repair_us_frc_occ_pdf_binding"
TRANSACTION_DIR = "transactions/us-frc-occ-52352"
RECOVERY_DIR = "recovery/us-frc-occ-52352"

OFFICIAL_OCC_URL = "https://infomemo.theocc.com/infomemos?number=52352"
OFFICIAL_OCC_HOST = "infomemo.theocc.com"
MEMO_NUMBER = "52352"
EVENT_ID = "e351f774b133eae45d49e0fbe60215e5bbceec540c3386076f4c3f2b6c57d9ea"
SECURITY_ID = "US:EODHD:3453b450-03d1-52ee-9100-cf80856b06ef"
EFFECTIVE_DATE = "2023-05-03"
ANNOUNCEMENT_DATE = "2023-05-02"
OLD_SYMBOL = "FRC"
NEW_SYMBOL = "FRCB"
CUSIP = "33616C100"
LEGACY_REVIEWED_EXTRACTION_SHA256 = (
    "377bcc0663eb9666b9f639edaf541b0ec729fd4b84ac876e345baaa9bf413668"
)
LEGACY_REVIEWED_EXTRACTION_ARCHIVE_ID = (
    "c568a6ac21ddc05d3c5821c228b94b7bd7e52a602a96b1cfb2f5f08ee24af658"
)
# Exact raw bytes independently reviewed after the one-shot official download.
# Semantic PDF checks are necessary but never sufficient to authorize a
# replacement: only these exact bytes may bind the corporate-action source.
REVIEWED_OCC_PDF_SHA256 = (
    "0ab8f0c49076f97049dcde80078affda258108ce7c904241bf1eba46b44aec66"
)
EXPECTED_PAGE_COUNT = 2
MAX_PDF_BYTES = 16 * 1024 * 1024

LEGACY_REVIEWED_EXTRACTION: dict[str, Any] = {
    "contract_multiplier": 1,
    "cusip": CUSIP,
    "deliverable_per_contract": "100 First Republic Bank (FRCB) Common Shares",
    "effective_date": EFFECTIVE_DATE,
    "market": "OTC",
    "memo_number": MEMO_NUMBER,
    "new_symbol": NEW_SYMBOL,
    "old_symbol": OLD_SYMBOL,
    "reviewed_claim": (
        "FRC and FRCB are the same First Republic common-share identity; "
        "only the market and ticker changed on 2023-05-03."
    ),
    "schema": "occ_reviewed_memo_extraction/v1",
    "source_url": OFFICIAL_OCC_URL,
    "subject": "First Republic Bank - Symbol Change",
}


class EvidenceError(ValueError):
    """Raised when an evidence or repository prerequisite is not exact."""


@dataclass(frozen=True)
class OccPdfEvidence:
    input_path: Path
    resolved_path: Path
    content: bytes
    source_hash: str
    exact_bytes: int
    page_count: int
    page_character_counts: tuple[int, ...]
    extracted_text_sha256: str
    claims: Mapping[str, bool]
    reviewed_by: str
    reviewed_at: str
    official_origin_confirmed: bool

    def object_path(self, completed_session: str) -> str:
        return f"archives/{completed_session}/{self.source_hash}.pdf.gz"


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    frames: Mapping[str, pd.DataFrame]
    evidence: OccPdfEvidence
    imported_at: str
    baseline_repository_error_signatures: tuple[str, ...]
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
    return str(value).strip()


def _date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _integer(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_json_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_utc_timestamp(value: str, *, label: str) -> str:
    text = _text(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError(f"{label} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise EvidenceError(f"{label} must be an explicit UTC timestamp.")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize_pdf_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\x00", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _claim_checks(text: str) -> dict[str, bool]:
    lowered = _normalize_pdf_text(text).casefold()
    return {
        "memo_number_52352": bool(re.search(r"(?:^|\s)#\s*52352(?:\s|$)", lowered)),
        "subject_first_republic_symbol_change": bool(
            re.search(
                r"first republic bank\s*[-\u2013\u2014]\s*symbol change",
                lowered,
            )
        ),
        "old_and_new_symbol_fields": bool(
            re.search(r"option symbol\s*:\s*frc\b", lowered)
            and re.search(r"new symbol\s*:\s*frcb\b", lowered)
        ),
        "effective_date_2023_05_03": bool(
            re.search(
                r"(?:\b05/03/2023\b|\bmay\s+0?3,?\s+2023\b|"
                r"\b2023-05-03\b)",
                lowered,
            )
        ),
        "frc_to_frcb_transition": bool(
            re.search(
                r"\bfrc\b.{0,100}?change(?:s| its trading symbol)?\s+to\s+\bfrcb\b",
                lowered,
            )
        ),
        "otc_market_reason": bool(re.search(r"listing of the company on an otc market", lowered)),
        "opening_of_business": bool(
            re.search(r"opening of business on may\s+0?3,?\s+2023", lowered)
        ),
        "option_terms_unchanged": bool(
            re.search(r"strike prices and all other option terms will not change", lowered)
        ),
        "clearing_member_new_symbol": bool(
            re.search(r"clearing member input to occ must use the new option symbol frcb", lowered)
        ),
        "underlying_security_change": bool(
            re.search(r"underlying\s+security\s*:\s*frc changes to frcb", lowered)
        ),
        "contract_multiplier_one": bool(
            re.search(r"contract\s+multiplier\s*:\s*1\b", lowered)
        ),
        "strike_divisor_one": bool(re.search(r"strike divisor\s*:\s*1\b", lowered)),
        "new_multiplier_100": bool(re.search(r"new multiplier\s*:\s*100\b", lowered)),
        "deliverable_100_common_shares": bool(
            re.search(
                r"100\s+first republic bank\s*\(\s*frcb\s*\)\s*common shares",
                lowered,
            )
        ),
        "cusip_33616c100": bool(re.search(r"cusip\s*:\s*33616c100\b", lowered)),
        "unofficial_summary_disclaimer": bool(
            re.search(r"provides an unofficial summary", lowered)
        ),
        "clearing_member_footer": bool(
            re.search(
                r"all clearing members are requested to immediately advise "
                r"all branch offices",
                lowered,
            )
        ),
    }


def _extract_pdf_pages(content: bytes) -> tuple[str, ...]:
    """Extract every PDF page with the available reviewed PDF runtime."""

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        PdfReader = None
    if PdfReader is not None:
        reader = PdfReader(io.BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise EvidenceError("OCC PDF must not be encrypted.")
        return tuple((page.extract_text() or "").strip() for page in reader.pages)

    try:
        import pymupdf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pypdf or pymupdf is required for OCC PDF validation."
        ) from exc
    with pymupdf.open(stream=content, filetype="pdf") as document:
        if document.needs_pass or document.is_encrypted:
            raise EvidenceError("OCC PDF must not be encrypted.")
        return tuple(page.get_text("text").strip() for page in document)


def load_occ_pdf(
    path: Path,
    *,
    reviewed_by: str,
    reviewed_at: str,
    official_origin_confirmed: bool,
) -> OccPdfEvidence:
    """Load one reviewed local PDF; never access the network or write files."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"OCC PDF does not exist: {input_path}")
    if input_path.is_symlink() or not input_path.is_file():
        raise EvidenceError("OCC PDF must be a regular, non-symlink file.")
    if not official_origin_confirmed:
        raise EvidenceError(
            "The reviewer must confirm download from the exact official OCC URL."
        )
    reviewer = _text(reviewed_by)
    if not reviewer:
        raise EvidenceError("A non-empty PDF reviewer identity is required.")
    reviewed_at = _validate_utc_timestamp(reviewed_at, label="reviewed_at")
    resolved = input_path.resolve(strict=True)
    content = resolved.read_bytes()
    if len(content) < 100 or len(content) > MAX_PDF_BYTES:
        raise EvidenceError(f"OCC PDF byte size is outside the envelope: {len(content)}.")
    if not content.startswith(b"%PDF-") or b"%%EOF" not in content[-4096:]:
        raise EvidenceError("OCC input is not a complete PDF byte stream.")
    observed_hash = _sha256(content)
    if observed_hash != REVIEWED_OCC_PDF_SHA256:
        raise EvidenceError(
            "OCC PDF SHA-256 does not match the independently reviewed raw pin."
        )

    try:
        page_texts = _extract_pdf_pages(content)
        if len(page_texts) != EXPECTED_PAGE_COUNT:
            raise EvidenceError(
                f"OCC memo 52352 must have {EXPECTED_PAGE_COUNT} pages; "
                f"observed {len(page_texts)}."
            )
    except EvidenceError:
        raise
    except Exception as exc:
        raise EvidenceError("OCC PDF cannot be parsed safely by pypdf.") from exc
    if any(not text for text in page_texts):
        raise EvidenceError("Every OCC PDF page must contain extractable text.")
    joined = "\n\f\n".join(page_texts)
    claims = _claim_checks(joined)
    missing = [name for name, passed in claims.items() if not passed]
    if missing:
        raise EvidenceError("OCC PDF lacks reviewed claims: " + ", ".join(missing))
    return OccPdfEvidence(
        input_path=input_path,
        resolved_path=resolved,
        content=content,
        source_hash=observed_hash,
        exact_bytes=len(content),
        page_count=len(page_texts),
        page_character_counts=tuple(len(value) for value in page_texts),
        extracted_text_sha256=_sha256(joined.encode("utf-8")),
        claims=claims,
        reviewed_by=reviewer,
        reviewed_at=reviewed_at,
        official_origin_confirmed=True,
    )


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = _text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvidenceError("FRC ticker-change metadata is invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise EvidenceError("FRC ticker-change metadata must be an object.")
    return parsed


def _exact_action(
    actions: pd.DataFrame,
    evidence: OccPdfEvidence,
    *,
    object_path: str = "",
) -> tuple[int, str, dict[str, Any]]:
    rows = actions.loc[actions["event_id"].astype(str).eq(EVENT_ID)]
    if len(rows) != 1:
        raise EvidenceError(f"Current release must contain exactly one FRC event {EVENT_ID}.")
    index = int(rows.index[0])
    row = rows.iloc[0]
    expected = {
        "security_id": SECURITY_ID,
        "action_type": "ticker_change",
        "effective_date": EFFECTIVE_DATE,
        "ex_date": EFFECTIVE_DATE,
        "announcement_date": ANNOUNCEMENT_DATE,
        "currency": "USD",
        "new_security_id": SECURITY_ID,
        "new_symbol": NEW_SYMBOL,
        "source_url": OFFICIAL_OCC_URL,
    }
    mismatches: list[str] = []
    for field, wanted in expected.items():
        actual = _date(row.get(field)) if "date" in field else _text(row.get(field))
        if actual != wanted:
            mismatches.append(field)
    if not _is_null(row.get("ratio")):
        mismatches.append("ratio")
    if not _is_null(row.get("cash_amount")):
        mismatches.append("cash_amount")
    if _text(row.get("official")).casefold() not in {"true", "1"}:
        mismatches.append("official")
    metadata = _metadata(row.get("metadata"))
    metadata_expected = {
        "memo_number": MEMO_NUMBER,
        "cusip": CUSIP,
    }
    for field, wanted in metadata_expected.items():
        if _text(metadata.get(field)) != wanted:
            mismatches.append(f"metadata.{field}")
    if mismatches:
        raise EvidenceError(
            "FRC ticker-change economics or identity fields changed: "
            + ", ".join(sorted(set(mismatches)))
            + "."
        )

    legacy = (
        _text(row.get("source_kind")) == "clearing_notice_reviewed_extraction"
        and _text(row.get("source")) == "occ_reviewed_memo_extraction"
        and _text(row.get("source_hash")) == LEGACY_REVIEWED_EXTRACTION_SHA256
    )
    raw = (
        _text(row.get("source_kind")) == "official_crosscheck"
        and _text(row.get("source")) == "occ_information_memo"
        and _text(row.get("source_hash")) == evidence.source_hash
        and _text(metadata.get("evidence_binding_schema"))
        == "occ_information_memo_binding/v1"
        and _text(metadata.get("occ_raw_pdf_sha256")) == evidence.source_hash
        and _integer(metadata.get("occ_raw_pdf_bytes")) == evidence.exact_bytes
        and _integer(metadata.get("occ_raw_pdf_page_count")) == evidence.page_count
        and _text(metadata.get("occ_raw_pdf_extracted_text_sha256"))
        == evidence.extracted_text_sha256
        and (
            not object_path
            or _text(metadata.get("occ_raw_pdf_object_path")) == object_path
        )
        and _text(metadata.get("occ_raw_pdf_reviewed_by")) == evidence.reviewed_by
        and _text(metadata.get("occ_raw_pdf_reviewed_at")) == evidence.reviewed_at
        and metadata.get("occ_official_origin_confirmed") is True
        and _text(metadata.get("occ_legacy_reviewed_extraction_sha256"))
        == LEGACY_REVIEWED_EXTRACTION_SHA256
    )
    if not (legacy or raw):
        raise EvidenceError(
            "FRC provenance is neither the exact legacy extraction nor this raw PDF."
        )
    return index, "legacy_extraction" if legacy else "raw_pdf_bound", metadata


def _safe_path(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if path == base or base not in path.parents:
        raise EvidenceError(f"Archive path escapes repository root: {relative}.")
    return path


def _legacy_payload() -> bytes:
    payload = _canonical_json_bytes(LEGACY_REVIEWED_EXTRACTION)
    if _sha256(payload) != LEGACY_REVIEWED_EXTRACTION_SHA256:
        raise RuntimeError("Legacy FRC reviewed-extraction code pin changed.")
    return payload


def _verify_legacy_archive(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
) -> None:
    rows = archive.loc[
        archive["archive_id"].astype(str).eq(LEGACY_REVIEWED_EXTRACTION_SHA256)
        | archive["source_hash"].astype(str).eq(LEGACY_REVIEWED_EXTRACTION_SHA256)
    ]
    if len(rows) != 1:
        raise EvidenceError("Legacy FRC reviewed extraction must have one archive row.")
    row = rows.iloc[0]
    object_path = (
        f"archives/{completed_session}/{LEGACY_REVIEWED_EXTRACTION_SHA256}.json.gz"
    )
    expected = {
        "archive_id": LEGACY_REVIEWED_EXTRACTION_ARCHIVE_ID,
        "dataset": "occ_reviewed_memo_extraction",
        "object_path": object_path,
        "content_type": "application/json",
        "effective_date": completed_session,
        "source": "occ_reviewed_memo_extraction",
        "source_hash": LEGACY_REVIEWED_EXTRACTION_SHA256,
        "source_url": OFFICIAL_OCC_URL,
    }
    changed = [
        field
        for field, wanted in expected.items()
        if (_date(row.get(field)) if field == "effective_date" else _text(row.get(field)))
        != wanted
    ]
    if changed:
        raise EvidenceError("Legacy FRC archive binding changed: " + ", ".join(changed))
    path = _safe_path(repository.root, object_path)
    if not path.is_file():
        raise EvidenceError("Legacy FRC reviewed-extraction payload is missing.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise EvidenceError("Legacy FRC archive payload is not valid gzip.") from exc
    if payload != _legacy_payload():
        raise EvidenceError("Legacy FRC reviewed-extraction payload changed.")


def _raw_archive_row(
    archive: pd.DataFrame,
    evidence: OccPdfEvidence,
    *,
    completed_session: str,
    imported_at: str,
) -> dict[str, Any]:
    row = {column: None for column in archive.columns}
    values = {
        "archive_id": evidence.source_hash,
        "dataset": "occ_information_memo",
        "object_path": evidence.object_path(completed_session),
        "content_type": "application/pdf",
        "effective_date": completed_session,
        "source": "occ_information_memo",
        "retrieved_at": imported_at,
        "source_hash": evidence.source_hash,
        "source_url": OFFICIAL_OCC_URL,
    }
    missing = sorted(set(values) - set(row))
    if missing:
        raise EvidenceError("source_archive lacks fields: " + ", ".join(missing))
    row.update(values)
    return row


def _rewrite_archive(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    evidence: OccPdfEvidence,
    *,
    completed_session: str,
    imported_at: str,
) -> tuple[pd.DataFrame, str]:
    expected = _raw_archive_row(
        archive,
        evidence,
        completed_session=completed_session,
        imported_at=imported_at,
    )
    related = archive.loc[
        archive["archive_id"].astype(str).eq(evidence.source_hash)
        | archive["source_hash"].astype(str).eq(evidence.source_hash)
        | archive["object_path"].astype(str).eq(expected["object_path"])
    ]
    if related.empty:
        output = pd.concat(
            [archive.copy(deep=True), pd.DataFrame([expected], columns=archive.columns)],
            ignore_index=True,
        )
        state = "missing"
    elif len(related) == 1:
        existing = related.iloc[0]
        stable_fields = (
            "archive_id",
            "dataset",
            "object_path",
            "content_type",
            "effective_date",
            "source",
            "source_hash",
            "source_url",
        )
        changed = [
            field
            for field in stable_fields
            if (
                _date(existing.get(field))
                if field == "effective_date"
                else _text(existing.get(field))
            )
            != _text(expected[field])
        ]
        if changed:
            raise EvidenceError("Raw OCC archive row conflicts: " + ", ".join(changed))
        path = _safe_path(repository.root, _text(existing.get("object_path")))
        if not path.is_file():
            raise EvidenceError("Raw OCC archive row exists but payload is missing.")
        try:
            raw = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise EvidenceError("Raw OCC archive object is not valid gzip.") from exc
        if raw != evidence.content:
            raise EvidenceError("Raw OCC archive bytes conflict with the reviewed PDF.")
        output = archive.copy(deep=True)
        state = "present"
    else:
        raise EvidenceError("Raw OCC PDF has duplicate archive bindings.")
    primary_key = list(dataset_spec(ARCHIVE_DATASET).primary_key)
    if output.duplicated(primary_key, keep=False).any():
        raise EvidenceError("Raw OCC repair duplicates source_archive primary keys.")
    return output.reset_index(drop=True), state


def _rewrite_action(
    actions: pd.DataFrame,
    evidence: OccPdfEvidence,
    *,
    imported_at: str,
    object_path: str,
) -> tuple[pd.DataFrame, str]:
    output = actions.copy(deep=True)
    index, state, metadata = _exact_action(
        output, evidence, object_path=object_path
    )
    if state == "raw_pdf_bound":
        return output, state
    binding = dict(metadata)
    binding.update(
        {
            "evidence_binding_schema": "occ_information_memo_binding/v1",
            "occ_raw_pdf_sha256": evidence.source_hash,
            "occ_raw_pdf_bytes": evidence.exact_bytes,
            "occ_raw_pdf_page_count": evidence.page_count,
            "occ_raw_pdf_extracted_text_sha256": evidence.extracted_text_sha256,
            "occ_raw_pdf_object_path": object_path,
            "occ_raw_pdf_reviewed_by": evidence.reviewed_by,
            "occ_raw_pdf_reviewed_at": evidence.reviewed_at,
            "occ_official_origin_confirmed": True,
            "occ_legacy_reviewed_extraction_sha256": (
                LEGACY_REVIEWED_EXTRACTION_SHA256
            ),
            "occ_disclaimer_role": "unofficial_corporate_event_summary",
        }
    )
    updates = {
        "source_url": OFFICIAL_OCC_URL,
        "source_kind": "official_crosscheck",
        "source": "occ_information_memo",
        "retrieved_at": imported_at,
        "source_hash": evidence.source_hash,
        "metadata": _canonical_json(binding),
    }
    for field, value in updates.items():
        output.at[index, field] = value
    return output, state


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        frames: Mapping[str, pd.DataFrame],
    ):
        self.base = base
        self.versions = dict(versions)
        self.frames = {name: value.copy() for name, value in frames.items()}

    def current_manifest(self, dataset: str):
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.frames:
            return self.frames[dataset].copy()
        version = self.versions.get(dataset)
        return self.base.read_frame(dataset, version) if version else pd.DataFrame()


def _repository_error_signatures(report: Any) -> tuple[str, ...]:
    signatures = []
    for issue in report.issues:
        if issue.severity != "error":
            continue
        signatures.append(
            _canonical_json(
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity,
                    "row_count": int(issue.row_count),
                    "fingerprints": list(issue.fingerprints),
                }
            )
        )
    return tuple(sorted(signatures))


def _require_no_new_repository_errors(
    baseline_signatures: Sequence[str], candidate_report: Any
) -> tuple[str, ...]:
    candidate_signatures = _repository_error_signatures(candidate_report)
    baseline_counts = {
        value: tuple(baseline_signatures).count(value)
        for value in set(baseline_signatures)
    }
    candidate_counts = {
        value: candidate_signatures.count(value) for value in set(candidate_signatures)
    }
    new_or_worse = sorted(
        value
        for value, count in candidate_counts.items()
        if count > baseline_counts.get(value, 0)
    )
    if new_or_worse:
        raise EvidenceError(
            "FRC OCC candidate introduces repository validation errors: "
            + "; ".join(new_or_worse)
        )
    return candidate_signatures


def prepare_repair(
    repository: LocalDatasetRepository,
    evidence: OccPdfEvidence,
    *,
    imported_at: str | None = None,
) -> PreparedRepair:
    """Prepare and validate a pure in-memory two-dataset candidate."""

    imported_at = _validate_utc_timestamp(
        imported_at or utc_now_iso(), label="imported_at"
    )
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(WRITE_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    pointer_etags: dict[str, str | None] = {}
    current: dict[str, pd.DataFrame] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions[dataset]:
            raise RuntimeError(f"{dataset} release/current pointer mismatch.")
        pointer_etags[dataset] = etag
        current[dataset] = repository.read_frame(dataset, pointer.version)

    _verify_legacy_archive(
        repository,
        current[ARCHIVE_DATASET],
        completed_session=release.completed_session,
    )
    archive, archive_state = _rewrite_archive(
        repository,
        current[ARCHIVE_DATASET],
        evidence,
        completed_session=release.completed_session,
        imported_at=imported_at,
    )
    action, action_state = _rewrite_action(
        current[ACTION_DATASET],
        evidence,
        imported_at=imported_at,
        object_path=evidence.object_path(release.completed_session),
    )
    if (archive_state, action_state) not in {
        ("missing", "legacy_extraction"),
        ("present", "raw_pdf_bound"),
    }:
        raise RuntimeError(
            "FRC OCC PDF binding is partially applied: "
            f"archive={archive_state}, action={action_state}."
        )
    frames = {ARCHIVE_DATASET: archive, ACTION_DATASET: action}
    validate_dataset(
        ARCHIVE_DATASET,
        archive,
        completed_session=release.completed_session,
    ).raise_for_errors()
    validate_dataset(
        ACTION_DATASET,
        action,
        completed_session=release.completed_session,
        incomplete_action_policy="block",
    ).raise_for_errors()
    baseline_repository_errors = _repository_error_signatures(
        validate_repository_snapshot(repository)
    )
    candidate_repository_errors = _require_no_new_repository_errors(
        baseline_repository_errors,
        validate_repository_snapshot(
            _CandidateRepository(repository, release.dataset_versions, frames)
        ),
    )

    changed = archive_state == "missing"
    summary: dict[str, Any] = {
        "schema": "us_frc_occ_pdf_binding_plan/v1",
        "status": "validated_offline_plan" if changed else "already_bound",
        "base_release_version": release.version,
        "completed_session": release.completed_session,
        "event_id": EVENT_ID,
        "security_id": SECURITY_ID,
        "event": {
            "action_type": "ticker_change",
            "effective_date": EFFECTIVE_DATE,
            "old_symbol": OLD_SYMBOL,
            "new_symbol": NEW_SYMBOL,
            "same_security_identity": True,
            "economic_fields_changed": False,
        },
        "official_source": {
            "url": OFFICIAL_OCC_URL,
            "host": OFFICIAL_OCC_HOST,
            "memo_number": MEMO_NUMBER,
            "document_owner": "The Options Clearing Corporation",
            "document_self_description": "unofficial corporate-event summary",
        },
        "raw_pdf": {
            "sha256": evidence.source_hash,
            "bytes": evidence.exact_bytes,
            "page_count": evidence.page_count,
            "page_character_counts": list(evidence.page_character_counts),
            "extracted_text_sha256": evidence.extracted_text_sha256,
            "claims": dict(evidence.claims),
            "object_path": evidence.object_path(release.completed_session),
            "gzip_sha256_mtime_0": _sha256(gzip.compress(evidence.content, mtime=0)),
            "reviewed_by": evidence.reviewed_by,
            "reviewed_at": evidence.reviewed_at,
            "official_origin_confirmed": evidence.official_origin_confirmed,
            "independent_reviewer_pin_matched": (
                evidence.source_hash == REVIEWED_OCC_PDF_SHA256
            ),
        },
        "legacy_extraction": {
            "preserved": True,
            "sha256": LEGACY_REVIEWED_EXTRACTION_SHA256,
        },
        "source_archive_rows_added": int(changed),
        "corporate_action_rows_changed": int(changed),
        "write_datasets": list(WRITE_DATASETS),
        "repository_validation": {
            "preexisting_error_count": len(baseline_repository_errors),
            "candidate_error_count": len(candidate_repository_errors),
            "new_or_worsened_error_count": 0,
        },
        "follow_up_not_applied": {
            "files": [
                "unified_quant/src/supertrend_quant/market_store/cross_validation.py",
                "unified_quant/scripts/validate_us_lifecycle_cross_sources.py",
                "unified_quant/scripts/finalize_us_lifecycle_coverage.py",
                "unified_quant/configs/us_cross_validation.yaml",
            ],
            "required_change": (
                "replace the FRC raw-primary-missing blocker and reviewed-"
                "extraction binding with this raw PDF SHA-256 after apply"
            ),
        },
        "network_accessed": False,
        "eodhd_calls": 0,
        "sec_calls": 0,
        "r2_accessed": False,
    }
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        frames=frames,
        evidence=evidence,
        imported_at=imported_at,
        baseline_repository_error_signatures=baseline_repository_errors,
        summary=summary,
    )


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
            raise RuntimeError("Unresolved FRC OCC recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        "Interrupted FRC OCC transaction blocks writes: " + str(journal)
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
        raise RuntimeError("Current release changed after FRC OCC planning.")
    for dataset in WRITE_DATASETS:
        pointer, pointer_etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or pointer.version != prepared.release.dataset_versions[dataset]
            or pointer_etag != prepared.pointer_etags[dataset]
        ):
            raise RuntimeError(f"{dataset} pointer changed after FRC OCC planning.")


def _persist_pdf(repository: LocalDatasetRepository, prepared: PreparedRepair) -> None:
    evidence = prepared.evidence
    if _sha256(evidence.content) != evidence.source_hash:
        raise EvidenceError("Prepared OCC PDF bytes changed before apply.")
    path = _safe_path(
        repository.root,
        evidence.object_path(prepared.release.completed_session),
    )
    if path.is_file():
        try:
            raw = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise EvidenceError("Persisted OCC PDF is not valid gzip.") from exc
        if raw != evidence.content:
            raise EvidenceError("Persisted OCC PDF conflicts with reviewed bytes.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, gzip.compress(evidence.content, mtime=0))
    if gzip.decompress(path.read_bytes()) != evidence.content:
        raise RuntimeError("OCC PDF post-write verification failed.")


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
                raise RuntimeError(f"unexpected release during rollback: {observed.version}")
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        try:
            key = repository.current_key(dataset)
            current = repository.objects.get(key)
            if current.data == old_pointer_bytes[dataset]:
                continue
            observed = CurrentPointer.from_bytes(current.data)
            if observed.version != planned_versions[dataset]:
                raise RuntimeError(
                    f"unexpected {dataset} pointer during rollback: {observed.version}"
                )
            repository.objects.put(key, old_pointer_bytes[dataset], if_match=current.etag)
        except Exception as exc:
            errors.append(f"{repository.current_key(dataset)}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    """Apply the prepared repair transaction; callers must opt in explicitly."""

    if prepared.summary["status"] == "already_bound":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    inject_failure = inject_failure or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _assert_base_unchanged(repository, prepared)
        old_release = repository.objects.get("releases/current.json")
        old_pointers = {
            dataset: repository.objects.get(repository.current_key(dataset))
            for dataset in WRITE_DATASETS
        }
        transaction_id = uuid.uuid4().hex
        prefix = (
            f"frc-occ-52352-{prepared.release.completed_session.replace('-', '')}-"
            f"{transaction_id}"
        )
        planned_versions = {
            dataset: f"{prefix}-{dataset}" for dataset in WRITE_DATASETS
        }
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_frc_occ_pdf_binding_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value.data).decode("ascii")
                for dataset, value in old_pointers.items()
            },
            "planned_versions": planned_versions,
            "raw_pdf_sha256": prepared.evidence.source_hash,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        written_versions: dict[str, str] = {}
        try:
            inject_failure("after_journal")
            _persist_pdf(repository, prepared)
            inject_failure("after_pdf_write")
            for dataset in WRITE_DATASETS:
                current_manifest = repository.manifest_for_version(
                    dataset, prepared.release.dataset_versions[dataset]
                )
                metadata = dict(current_manifest.metadata)
                metadata.update(
                    {
                        "operation": OPERATION,
                        "frc_event_id": EVENT_ID,
                        "occ_memo_number": MEMO_NUMBER,
                        "occ_raw_pdf_sha256": prepared.evidence.source_hash,
                        "network_accessed": False,
                        "eodhd_calls": 0,
                        "sec_calls": 0,
                        "r2_accessed": False,
                    }
                )
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy=(
                        "block" if dataset == ACTION_DATASET else "warn"
                    ),
                    metadata=metadata,
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned_versions[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                if result.manifest.version != planned_versions[dataset]:
                    raise RuntimeError(f"Unexpected {dataset} version was written.")
                written_versions[dataset] = result.manifest.version
                inject_failure(f"after_{dataset}_write")
            versions = dict(prepared.release.dataset_versions)
            versions.update(written_versions)
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=prepared.release.quality,
                warnings=prepared.release.warnings,
                expected_etag=prepared.release_etag,
            )
            inject_failure("after_release_commit")
            _require_no_new_repository_errors(
                prepared.baseline_repository_error_signatures,
                validate_repository_snapshot(repository),
            )
            replay = prepare_repair(
                repository,
                prepared.evidence,
                imported_at=prepared.imported_at,
            )
            if replay.summary["status"] != "already_bound":
                raise RuntimeError("FRC OCC repair is not idempotent.")
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
                "new_dataset_versions": written_versions,
                "transaction_id": transaction_id,
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes={
                    dataset: value.data for dataset, value in old_pointers.items()
                },
                planned_versions=planned_versions,
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
                    "FRC OCC rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or apply the offline FRC OCC 52352 raw-PDF binding."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--occ-pdf", type=Path, required=True)
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument(
        "--confirm-official-origin",
        action="store_true",
        help="Confirm the file was downloaded from the exact official OCC URL.",
    )
    parser.add_argument("--imported-at", default=None)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence = load_occ_pdf(
        args.occ_pdf,
        reviewed_by=args.reviewed_by,
        reviewed_at=args.reviewed_at,
        official_origin_confirmed=args.confirm_official_origin,
    )
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(
        repository,
        evidence,
        imported_at=args.imported_at,
    )
    if args.apply:
        result = apply_repair(repository, prepared)
    else:
        result = {**prepared.summary, "mode": "plan", "writes_performed": False}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
