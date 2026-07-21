#!/usr/bin/env python3
"""Plan an exact-byte binding for OCC memo 52352 without changing the store.

The existing FRC -> FRCB lifecycle action was built from a manually reviewed
extraction because the OCC endpoint is protected by a browser challenge.  This
planner accepts a PDF that the owner downloaded from that exact OCC URL and
prepares the narrowly scoped changes needed to replace the extraction with the
real PDF hash.

This file deliberately has no apply mode and no networking code.  It only:

* reads the path explicitly supplied with ``--occ-pdf``;
* extracts text with :mod:`pypdf` and verifies every reviewed memo term;
* computes the exact PDF and deterministic-gzip hashes and byte counts;
* checks the current release, action, archive index, and review registry;
* emits a JSON dry-run plan for a later reviewed implementation.

The official URL and the local acquisition path are reported in separate
objects.  Passing the automated checks does not prove that a local file was
downloaded from OCC, so the plan always blocks apply/publication pending a
rendered visual review and confirmation of the download provenance.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd
import yaml

from supertrend_quant.market_store.cross_validation import (
    TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
    reviewed_nonterminal_extraction_mismatches,
    reviewed_nonterminal_extraction_sha256,
    reviewed_nonterminal_extractions,
    reviewed_nonterminal_inventory_sha256,
)
from supertrend_quant.market_store.manifest import utc_now_iso
from supertrend_quant.market_store.repository import LocalDatasetRepository


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_POLICY_PATH = Path("unified_quant/configs/us_cross_validation.yaml")

OFFICIAL_OCC_URL = "https://infomemo.theocc.com/infomemos?number=52352"
OFFICIAL_OCC_HOST = "infomemo.theocc.com"
MEMO_NUMBER = "52352"
FRC_CUSIP = "33616C100"
EVENT_ID = "e351f774b133eae45d49e0fbe60215e5bbceec540c3386076f4c3f2b6c57d9ea"
SECURITY_ID = "US:EODHD:3453b450-03d1-52ee-9100-cf80856b06ef"
EFFECTIVE_DATE = "2023-05-03"
OLD_SYMBOL = "FRC"
NEW_SYMBOL = "FRCB"

# This is the hash of the existing self-authored reviewed JSON extraction.  It
# is intentionally *not* an expected OCC PDF hash.  The PDF hash is learned
# only from the bytes supplied by the user and is emitted for explicit review.
LEGACY_REVIEWED_EXTRACTION_SHA256 = (
    "377bcc0663eb9666b9f639edaf541b0ec729fd4b84ac876e345baaa9bf413668"
)

MAX_PDF_BYTES = 16 * 1024 * 1024
EXPECTED_PAGE_COUNT = 2
SOURCE_ARCHIVE_DATASET = "source_archive"
ACTION_DATASET = "corporate_actions"


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


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
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
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _normalize_pdf_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\x00", " ")
    return re.sub(r"\s+", " ", value).strip()


def _claim_checks(text: str) -> dict[str, bool]:
    normalized = _normalize_pdf_text(text)
    lowered = normalized.casefold()
    flags = {
        "memo_number_52352": bool(
            re.search(r"(?:^|\s)#\s*52352(?:\s|$)", lowered)
        ),
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
        "frc_to_frcb_transition": bool(
            re.search(
                r"\bfrc\b\s+(?:changes\s+to|will\s+change(?:\s+its)?"
                r"\s+trading\s+symbol\s+to)\s+\bfrcb\b",
                lowered,
            )
        ),
        "effective_date_2023_05_03": bool(
            re.search(
                r"(?:\b05/03/2023\b|\bmay\s+0?3,?\s+2023\b|"
                r"\b2023-05-03\b)",
                lowered,
            )
        ),
        "cusip_33616c100": bool(
            re.search(r"cusip\s*:\s*33616c100\b", lowered)
        ),
        "contract_multiplier_one": bool(
            re.search(r"contract multiplier\s*:\s*1\b", lowered)
        ),
        "new_multiplier_100": bool(
            re.search(r"new multiplier\s*:\s*100\b", lowered)
        ),
        "deliverable_100_common_shares": bool(
            re.search(
                r"100\s+first republic bank\s*\(\s*frcb\s*\)\s*"
                r"common shares\b",
                lowered,
            )
        ),
        "all_option_terms_unchanged": bool(
            re.search(
                r"strike prices and all other option terms will not change\b",
                lowered,
            )
        ),
        "unofficial_summary_disclaimer": bool(
            re.search(r"provides an unofficial summary\b", lowered)
        ),
    }
    return flags


def _extract_pdf_pages(content: bytes) -> tuple[str, ...]:
    """Extract every page with either supported, local-only PDF runtime."""

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        PdfReader = None
    if PdfReader is not None:
        reader = PdfReader(io.BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise ValueError("OCC PDF must not be encrypted.")
        return tuple((page.extract_text() or "").strip() for page in reader.pages)

    try:
        import pymupdf
    except ModuleNotFoundError:
        try:
            import fitz as pymupdf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pypdf or pymupdf is required for OCC PDF validation; "
                "install project dependencies."
            ) from exc
    with pymupdf.open(stream=content, filetype="pdf") as document:
        if document.needs_pass or document.is_encrypted:
            raise ValueError("OCC PDF must not be encrypted.")
        return tuple(page.get_text("text").strip() for page in document)


def load_occ_pdf(path: Path) -> OccPdfEvidence:
    """Read and validate one user-supplied OCC PDF without writing anything."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"OCC PDF does not exist: {input_path}")
    if input_path.is_symlink():
        raise ValueError("OCC PDF must be a regular file, not a symbolic link.")
    if not input_path.is_file():
        raise ValueError(f"OCC PDF is not a regular file: {input_path}")
    resolved = input_path.resolve(strict=True)
    content = resolved.read_bytes()
    if len(content) < 100 or len(content) > MAX_PDF_BYTES:
        raise ValueError(
            f"OCC PDF byte size is outside the reviewed range: {len(content)}."
        )
    if not content.startswith(b"%PDF-") or b"%%EOF" not in content[-2048:]:
        raise ValueError("OCC input is not a complete PDF byte stream.")

    try:
        page_texts = _extract_pdf_pages(content)
        if len(page_texts) != EXPECTED_PAGE_COUNT:
            raise ValueError(
                "OCC memo 52352 must contain exactly "
                f"{EXPECTED_PAGE_COUNT} pages; observed {len(page_texts)}."
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("OCC PDF cannot be parsed safely by the local PDF runtime.") from exc
    if any(not text for text in page_texts):
        raise ValueError("Every OCC PDF page must contain extractable text.")

    joined = "\n\f\n".join(page_texts)
    claims = _claim_checks(joined)
    missing = tuple(name for name, passed in claims.items() if not passed)
    if missing:
        raise ValueError(
            "OCC PDF is missing or changed reviewed claims: " + ", ".join(missing)
        )
    return OccPdfEvidence(
        input_path=input_path,
        resolved_path=resolved,
        content=content,
        source_hash=hashlib.sha256(content).hexdigest(),
        exact_bytes=len(content),
        page_count=len(page_texts),
        page_character_counts=tuple(len(text) for text in page_texts),
        extracted_text_sha256=hashlib.sha256(joined.encode("utf-8")).hexdigest(),
        claims=claims,
    )


def _validate_imported_at(value: str) -> str:
    text = _text(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("imported_at must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("imported_at must be an explicit UTC timestamp.")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = _text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("FRC corporate-action metadata is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("FRC corporate-action metadata must be an object.")
    return parsed


def _exact_action(actions: pd.DataFrame, evidence: OccPdfEvidence) -> dict[str, Any]:
    rows = actions.loc[actions["event_id"].astype(str).eq(EVENT_ID)]
    if len(rows) != 1:
        raise ValueError(
            f"Current release must contain exactly one FRC event {EVENT_ID}."
        )
    row = rows.iloc[0].to_dict()
    expected = {
        "security_id": SECURITY_ID,
        "action_type": "ticker_change",
        "effective_date": EFFECTIVE_DATE,
        "ex_date": EFFECTIVE_DATE,
        "announcement_date": "2023-05-02",
        "currency": "USD",
        "new_security_id": SECURITY_ID,
        "new_symbol": NEW_SYMBOL,
        "source_url": OFFICIAL_OCC_URL,
    }
    mismatches: list[str] = []
    for field, expected_value in expected.items():
        actual = _date(row.get(field)) if "date" in field else _text(row.get(field))
        if actual != expected_value:
            mismatches.append(field)
    if row.get("ratio") is not None and not pd.isna(row.get("ratio")):
        mismatches.append("ratio")
    if row.get("cash_amount") is not None and not pd.isna(row.get("cash_amount")):
        mismatches.append("cash_amount")
    if _text(row.get("official")).lower() not in {"true", "1"}:
        mismatches.append("official")
    metadata = _metadata(row.get("metadata"))
    if _text(metadata.get("memo_number")) != MEMO_NUMBER:
        mismatches.append("metadata.memo_number")
    if _text(metadata.get("cusip")).upper() != FRC_CUSIP:
        mismatches.append("metadata.cusip")
    if mismatches:
        raise ValueError(
            "Current FRC corporate action changed outside the PDF binding: "
            + ", ".join(sorted(set(mismatches)))
            + "."
        )

    legacy = (
        _text(row.get("source_kind")) == "clearing_notice_reviewed_extraction"
        and _text(row.get("source")) == "occ_reviewed_memo_extraction"
        and _text(row.get("source_hash")) == LEGACY_REVIEWED_EXTRACTION_SHA256
    )
    already_bound = (
        _text(row.get("source_kind")) == "official_crosscheck"
        and _text(row.get("source")) == "occ_information_memo"
        and _text(row.get("source_hash")) == evidence.source_hash
    )
    if not (legacy or already_bound):
        raise ValueError(
            "Current FRC corporate-action provenance is neither the reviewed "
            "legacy extraction nor this exact PDF binding."
        )
    row["_binding_state"] = "legacy_extraction" if legacy else "already_bound"
    return row


def _reviewed_extraction(evidence: OccPdfEvidence) -> dict[str, Any]:
    return {
        "event_id": EVENT_ID,
        "security_id": SECURITY_ID,
        "action_type": "ticker_change",
        "effective_date": EFFECTIVE_DATE,
        "new_security_id": SECURITY_ID,
        "new_symbol": NEW_SYMBOL,
        "ratio": None,
        "cash_amount": None,
        "currency": "USD",
        "source_kind": "official_crosscheck",
        "source_url": OFFICIAL_OCC_URL,
        "source_hash": evidence.source_hash,
    }


def _policy_plan(
    policy: Mapping[str, Any], evidence: OccPdfEvidence
) -> dict[str, Any]:
    events = policy.get("events")
    if not isinstance(events, Mapping):
        raise ValueError("Cross-validation policy has no events object.")
    current_inventory = reviewed_nonterminal_extractions(events)
    current_hash = reviewed_nonterminal_inventory_sha256(events)
    if current_hash != TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256:
        raise ValueError(
            "Current reviewed-nonterminal registry does not match its code pin."
        )
    proposed = _reviewed_extraction(evidence)
    existing = current_inventory.get(EVENT_ID)
    if existing is not None:
        mismatches = reviewed_nonterminal_extraction_mismatches(proposed, existing)
        if mismatches:
            raise ValueError(
                "Existing FRC reviewed extraction conflicts with this PDF: "
                + ", ".join(mismatches)
                + "."
            )

    proposed_events = copy.deepcopy(dict(events))
    values = list(proposed_events.get("reviewed_nonterminal_extractions") or ())
    if existing is None:
        values.append(proposed)
    proposed_events["reviewed_nonterminal_extractions"] = values
    proposed_hash = reviewed_nonterminal_inventory_sha256(proposed_events)

    current_hosts = tuple(
        _text(value).lower()
        for value in events.get("official_provenance_hosts") or ()
    )
    proposed_hosts = tuple(sorted(set(current_hosts) | {OFFICIAL_OCC_HOST}))
    allowed_kinds = tuple(
        _text(value)
        for value in events.get("official_provenance_source_kinds") or ()
    )
    if "official_crosscheck" not in allowed_kinds:
        raise ValueError(
            "Cross-validation policy no longer permits official_crosscheck."
        )
    return {
        "reviewed_nonterminal_rows_added": int(existing is None),
        "reviewed_nonterminal_extraction": proposed,
        "reviewed_nonterminal_extraction_sha256": (
            reviewed_nonterminal_extraction_sha256(proposed)
        ),
        "current_inventory_sha256": current_hash,
        "proposed_inventory_sha256": proposed_hash,
        "trusted_code_pin_current": (
            TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256
        ),
        "trusted_code_pin_proposed": proposed_hash,
        "official_provenance_hosts_current": list(current_hosts),
        "official_provenance_hosts_proposed": list(proposed_hosts),
        "official_provenance_host_contract_change_required": (
            OFFICIAL_OCC_HOST not in current_hosts
        ),
        "official_provenance_source_kinds_unchanged": list(allowed_kinds),
    }


def _archive_plan(
    archive: pd.DataFrame,
    evidence: OccPdfEvidence,
    *,
    completed_session: str,
    imported_at: str,
) -> dict[str, Any]:
    object_path = (
        f"archives/{completed_session}/{evidence.source_hash}.pdf.gz"
    )
    row = {
        "archive_id": evidence.source_hash,
        "dataset": "occ_information_memo",
        "object_path": object_path,
        "content_type": "application/pdf",
        "effective_date": completed_session,
        "source": "occ_information_memo",
        "source_url": OFFICIAL_OCC_URL,
        "retrieved_at": imported_at,
        "source_hash": evidence.source_hash,
    }
    if archive.empty:
        related = archive
    else:
        related = archive.loc[
            archive["archive_id"].astype(str).eq(evidence.source_hash)
            | archive["source_hash"].astype(str).eq(evidence.source_hash)
        ]
    if len(related) > 1:
        raise ValueError("OCC PDF hash has duplicate source_archive rows.")
    if len(related) == 1:
        existing = related.iloc[0].to_dict()
        stable_fields = (
            "archive_id",
            "dataset",
            "content_type",
            "source",
            "source_url",
            "source_hash",
        )
        mismatches = [
            field
            for field in stable_fields
            if _text(existing.get(field)) != _text(row[field])
        ]
        if mismatches:
            raise ValueError(
                "Existing OCC source_archive row conflicts with the PDF plan: "
                + ", ".join(mismatches)
                + "."
            )
        row = existing
        rows_added = 0
    else:
        rows_added = 1
    compressed = gzip.compress(evidence.content, mtime=0)
    return {
        "rows_added": rows_added,
        "candidate_row": row,
        "candidate_row_sha256": _canonical_json_sha256(row),
        "payload_write_required": bool(rows_added),
        "payload_object_path": _text(row.get("object_path")) or object_path,
        "payload_raw_sha256": evidence.source_hash,
        "payload_raw_bytes": evidence.exact_bytes,
        "payload_gzip_sha256_mtime_0": hashlib.sha256(compressed).hexdigest(),
        "payload_gzip_bytes_mtime_0": len(compressed),
    }


def _action_plan(
    current: Mapping[str, Any], evidence: OccPdfEvidence, *, imported_at: str
) -> dict[str, Any]:
    binding_metadata = {
        "schema": "occ_information_memo_binding/v1",
        "memo_number": MEMO_NUMBER,
        "cusip": FRC_CUSIP,
        "official_source_url": OFFICIAL_OCC_URL,
        "raw_pdf_sha256": evidence.source_hash,
        "raw_pdf_bytes": evidence.exact_bytes,
        "extracted_text_sha256": evidence.extracted_text_sha256,
        "acquisition_method": "user_supplied_local_file",
        "official_download_timestamp": None,
        "local_imported_at": imported_at,
        "origin_authentication_status": "pending_manual_confirmation",
        "visual_review_status": "pending",
        "evidence_role": "official_occ_clearing_notice_crosscheck",
        "occ_disclaimer_role": "unofficial_corporate_event_summary",
    }
    changes = {
        "source_url": OFFICIAL_OCC_URL,
        "source_kind": "official_crosscheck",
        "source": "occ_information_memo",
        "retrieved_at": imported_at,
        "source_hash": evidence.source_hash,
        "metadata": json.dumps(
            binding_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    }
    changed = any(_text(current.get(field)) != _text(value) for field, value in changes.items())
    return {
        "rows_changed": int(changed),
        "event_id": EVENT_ID,
        "current_binding_state": current["_binding_state"],
        "proposed_provenance_fields": changes,
        "economic_fields_changed": False,
    }


def build_plan(
    *,
    evidence: OccPdfEvidence,
    actions: pd.DataFrame,
    archive: pd.DataFrame,
    policy: Mapping[str, Any],
    base_release_version: str,
    completed_session: str,
    dataset_versions: Mapping[str, str],
    imported_at: str,
) -> dict[str, Any]:
    """Build a pure in-memory plan from a frozen repository snapshot."""

    imported_at = _validate_imported_at(imported_at)
    current_action = _exact_action(actions, evidence)
    archive_plan = _archive_plan(
        archive,
        evidence,
        completed_session=completed_session,
        imported_at=imported_at,
    )
    policy_plan = _policy_plan(policy, evidence)
    action_plan = _action_plan(current_action, evidence, imported_at=imported_at)
    official = urlparse(OFFICIAL_OCC_URL)
    plan: dict[str, Any] = {
        "schema": "us_frc_occ_pdf_binding_plan/v1",
        "status": "validated_offline_plan",
        "mode": "plan",
        "publication_ready": False,
        "base": {
            "release_version": base_release_version,
            "completed_session": completed_session,
            "dataset_versions": {
                ACTION_DATASET: _text(dataset_versions.get(ACTION_DATASET)),
                SOURCE_ARCHIVE_DATASET: _text(
                    dataset_versions.get(SOURCE_ARCHIVE_DATASET)
                ),
            },
        },
        "event": {
            "event_id": EVENT_ID,
            "security_id": SECURITY_ID,
            "action_type": "ticker_change",
            "effective_date": EFFECTIVE_DATE,
            "old_symbol": OLD_SYMBOL,
            "new_symbol": NEW_SYMBOL,
            "cusip": FRC_CUSIP,
        },
        "evidence": {
            "official_source_metadata": {
                "url": OFFICIAL_OCC_URL,
                "scheme": official.scheme,
                "host": official.hostname,
                "memo_number": MEMO_NUMBER,
                "document_owner": "The Options Clearing Corporation",
                "evidence_role": "official_occ_clearing_notice_crosscheck",
                "document_self_description": "unofficial corporate-event summary",
            },
            "local_file_provenance": {
                "input_path": str(evidence.input_path),
                "resolved_path": str(evidence.resolved_path),
                "acquisition_method": "user_supplied_local_file",
                "imported_at": imported_at,
                "official_download_timestamp": None,
                "origin_authentication_status": "pending_manual_confirmation",
                "network_retrieval_performed_by_planner": False,
            },
            "exact_pdf": {
                "sha256": evidence.source_hash,
                "bytes": evidence.exact_bytes,
                "page_count": evidence.page_count,
                "page_character_counts": list(evidence.page_character_counts),
                "extracted_text_sha256": evidence.extracted_text_sha256,
                "required_claims": dict(evidence.claims),
                "expected_sha256_predeclared": False,
            },
        },
        "source_archive_plan": archive_plan,
        "corporate_action_plan": action_plan,
        "reviewed_nonterminal_plan": policy_plan,
        "visual_review": {
            "status": "required_before_apply",
            "automated_text_extraction_is_not_layout_or_origin_proof": True,
            "expected_page_count": EXPECTED_PAGE_COUNT,
            "render_command_argv": [
                "pdftoppm",
                "-png",
                "<OCC_PDF>",
                "tmp/pdfs/frc-occ-52352",
            ],
            "checklist": [
                "confirm both rendered pages are legible and complete",
                "confirm memo number, symbols, date, CUSIP, and deliverable visually",
                "confirm the file was downloaded from the exact official OCC URL",
                "record reviewer and review timestamp before any apply implementation",
            ],
        },
        "activation_requirements": [
            "complete rendered two-page visual review and download-origin confirmation",
            "persist the exact PDF bytes under the planned content-addressed object path",
            "rewrite only the FRC corporate-action provenance fields shown in this plan",
            "append the exact reviewed nonterminal extraction shown in this plan",
            "add infomemo.theocc.com to the exact official-provenance host contract",
            "update TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256 to the proposed hash",
            "run offline cross-validation and publication tests before release commit",
        ],
        "safety": {
            "writes_performed": False,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
            "apply_mode_available": False,
            "official_pdf_hash_invented": False,
        },
    }
    plan["plan_sha256"] = _canonical_json_sha256(plan)
    return plan


def build_repository_plan(
    repository: LocalDatasetRepository,
    *,
    occ_pdf: Path,
    policy_path: Path = DEFAULT_POLICY_PATH,
    imported_at: str | None = None,
) -> dict[str, Any]:
    """Read one frozen release and produce a no-write binding plan."""

    evidence = load_occ_pdf(occ_pdf)
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    required_versions = {
        dataset: release.dataset_versions.get(dataset, "")
        for dataset in (ACTION_DATASET, SOURCE_ARCHIVE_DATASET)
    }
    if not all(required_versions.values()):
        raise RuntimeError(
            "Current release must contain corporate_actions and source_archive."
        )
    for dataset, version in required_versions.items():
        pointer, _pointer_etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"{dataset} release/current pointer mismatch.")
    try:
        policy = yaml.safe_load(Path(policy_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cross-validation policy is unreadable: {policy_path}") from exc
    if not isinstance(policy, dict):
        raise RuntimeError("Cross-validation policy root must be an object.")
    plan = build_plan(
        evidence=evidence,
        actions=repository.read_frame(ACTION_DATASET, required_versions[ACTION_DATASET]),
        archive=repository.read_frame(
            SOURCE_ARCHIVE_DATASET, required_versions[SOURCE_ARCHIVE_DATASET]
        ),
        policy=policy,
        base_release_version=release.version,
        completed_session=release.completed_session,
        dataset_versions=required_versions,
        imported_at=imported_at or utc_now_iso(),
    )
    observed_release, observed_etag = repository.current_release()
    if (
        observed_release is None
        or observed_release.version != release.version
        or observed_etag != release_etag
    ):
        raise RuntimeError("Current release changed while the OCC plan was built.")
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a user-downloaded OCC memo 52352 PDF and emit a dry-run "
            "FRC/FRCB provenance-binding plan."
        )
    )
    parser.add_argument(
        "--occ-pdf",
        type=Path,
        required=True,
        help="Local PDF downloaded from the exact OCC memo 52352 URL.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Read-only local market-store root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        plan = build_repository_plan(
            LocalDatasetRepository(args.cache_root),
            occ_pdf=args.occ_pdf,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
