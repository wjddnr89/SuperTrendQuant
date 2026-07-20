#!/usr/bin/env python3
"""Superseded NTCO NYSE-only repair retained as a non-runnable audit draft.

Do not run this repair.  Later official Cboe/OCC/BNY evidence establishes an
NTCO -> NTCOY same-ADS OTC continuation and a subsequent mandatory cash
exchange.  Deleting the post-2024-02-09 rows or closing an
``unsupported_consideration`` exception would therefore be unsafe.  Every
acquisition and mutation entry point in this module is fail-closed; use
``plan_us_ntco_ntcoy_transition.py`` for the replacement evidence plan.

Natura &Co's NYSE ADSs last traded on 2024-02-09.  The stored EODHD payload
contains 43 later rows through 2024-04-12 plus two later provider dividends,
but the official filing says that no replacement quotation medium was
arranged.  The ADS program continued until 2024-08-07 and an ADS represented
two underlying B3 shares, so the terminal economics are not a zero-cash
delisting.  They are closed as an exact ``unsupported_consideration``
exception.

The command is deliberately two-stage and fail-closed:

* ``--fetch-official`` performs at most one HTTP attempt to one code-pinned
  SEC URL.  It writes only a staging cache and observed SHA-256.
* the default plan and ``--apply`` never use the network.  They require the
  staged bytes to match a reviewer-pinned SHA-256 in ``us_lifecycle_hints``.

Plan is the default.  Apply uses the shared market-store writer lock, exact
pointer/release CAS checks, a durable journal, and rollback of every current
pointer and the release pointer.  The immutable original EODHD EOD/dividend
envelopes remain in ``source_archive``; release metadata binds the quarantined
row inventories to those archive objects and exact record hashes.
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
import os
import re
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.lifecycle import build_lifecycle_candidates
from supertrend_quant.market_store.lifecycle_coverage import (
    lifecycle_candidate_id,
    validate_lifecycle_coverage,
)
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.official_lifecycle_evidence import (
    OfficialLifecycleExceptionEvidenceSpec,
    include_bound_official_applied_event_candidates,
    load_official_lifecycle_exception_evidence,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_HINTS = Path("unified_quant/configs/us_lifecycle_hints.yaml")
DEFAULT_EVIDENCE_DIR = (
    DEFAULT_CACHE_ROOT / "state/issuer_lifecycle/ntco_nyse_boundary"
)
EVIDENCE_REPORT = "ntco_official_boundary_evidence.json"
EVIDENCE_SCHEMA = "us_ntco_official_boundary_evidence/v1"
MAX_HTTP_ATTEMPTS = 1
MAX_RESPONSE_BYTES = 12_000_000
TIMEOUT_SECONDS = 30

OPERATION = "repair_us_ntco_nyse_boundary"
TRANSACTION_DIR = "transactions/us-ntco-nyse-boundary"
RECOVERY_DIR = "recovery/us-ntco-nyse-boundary"
REVIEWED_BY = "us_lifecycle_finalizer_v1"
REVIEWED_AT = "2026-07-18T00:00:00Z"
REPAIRED_IDENTITY_SOURCE = "official_ntco_nyse_boundary"
SUPERSEDED_REASON = (
    "Superseded by official Cboe/OCC/BNY evidence: NTCO changed to NTCOY on "
    "Other-OTC on 2024-02-12 and later received a mandatory cash exchange. "
    "The NYSE-only trim/unsupported-exception repair is permanently disabled."
)


def _raise_superseded() -> None:
    raise RuntimeError(SUPERSEDED_REASON)
RESOLUTION_SOURCE = "sec_edgar_filing"

SECURITY_ID = "US:EODHD:7e3cea59-409f-5cf0-8429-9d4245013622"
SYMBOL = "NTCO"
NAME_TOKEN = "Natura"
LAST_NYSE_SESSION = "2024-02-09"
FIRST_QUARANTINED_SESSION = "2024-02-12"
LAST_QUARANTINED_SESSION = "2024-04-12"
BOUNDARY_EFFECTIVE_DATE = "2024-02-12"
ADS_PROGRAM_TERMINATION_DATE = "2024-08-07"
CANDIDATE_ID = (
    "3da15f8f6e43b34b9c3a55d1d1e82c6f0a3c2c44d52f8f078abe03c7f8b8dd3c"
)
EVIDENCE_ID = "ntco_2024_nyse_ads_delisting"
EXCEPTION_CODE = "unsupported_consideration"
EXCEPTION_CLAIM = (
    "NTCO's NYSE ADS quotation ended before the depositary program; each ADS "
    "continued to represent two B3 shares that holders could surrender for, so "
    "neither a zero-cash delisting nor one fixed terminal consideration row is "
    "economically complete."
)
OFFICIAL_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1776967/"
    "000155485524000366/ntco-20231231.htm"
)

# These groups express the four independent facts required by the repair.  The
# alternatives are intentionally narrow, but allow harmless HTML/typography
# differences after whitespace and entity normalization.
OFFICIAL_REQUIRED_TEXT_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        r"last day of trading.{0,80}?february 9, 2024",
        r"february 9, 2024.{0,80}?last day of trading",
    ),
    (
        r"(?:has|had|have|was) not arranged.{0,180}?quotation medium",
        r"no.{0,120}?quotation medium.{0,120}?arranged",
    ),
    (
        r"(?:termination|terminate|terminated).{0,180}?august 7, 2024",
        r"august 7, 2024.{0,180}?(?:termination|terminate|terminated)",
    ),
    (
        r"(?:one|1|each).{0,30}?ads.{0,80}?(?:two|2).{0,40}?(?:common )?shares",
        r"(?:two|2).{0,40}?(?:common )?shares.{0,80}?(?:one|1).{0,30}?ads",
    ),
    (
        r"surrender.{0,120}?ads.{0,180}?(?:underlying|common) shares",
        r"holders.{0,120}?ads.{0,180}?(?:withdraw|receive).{0,100}?(?:underlying|common) shares",
    ),
)

# The shared official-evidence registry uses literal substring matching, not
# regular expressions.  Keep its reviewed phrases separate from the stricter
# local regex contract above.
REGISTRY_REQUIRED_TEXT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("last day of trading was February 9, 2024",),
    (
        "has not arranged for the listing or quotation",
        "has not arranged for quotation",
        "will not arrange for the listing or quotation",
    ),
    ("quotation medium",),
    ("August 7, 2024",),
    (
        "Each ADS represents two common shares",
        "each ADS representing two common shares",
        "one ADS represents two common shares",
    ),
    ("surrender their ADSs", "surrender of their ADSs"),
)

EOD_ARCHIVE_ID = (
    "e88684de37208bd947df3140593aff81082126aefbc353d545f3ef0ae9fd8883"
)
EOD_SOURCE_URL = (
    "https://eodhd.com/api/eod/NTCO.US?from=2015-01-01&to=2026-07-15"
)
EOD_RAW_SHA256 = (
    "91cb9baec50c86d49447d78f2882256a991884e46fda1a6019f5df792cb02dde"
)
EOD_RAW_BYTES = 120_644
EOD_RAW_ROWS = 1_075
EOD_RETRIEVED_AT = "2026-07-17T20:37:19.646249Z"
QUARANTINED_PRICE_ROWS = 43
QUARANTINED_PRICE_RECORDS_SHA256 = (
    "c1d5c74407f010ee56d829b565900752858e034c505b166a7218cbec3d4d8677"
)
LAST_NYSE_OHLCV = (6.5, 6.6075, 6.465, 6.57, 1_561_496.0)

DIVIDEND_ARCHIVE_ID = (
    "50a475c8a45f25d19d831ce7eaaf1f3fbad758600eec7dec45ea5c63d4a171a8"
)
DIVIDEND_SOURCE_URL = (
    "https://eodhd.com/api/div/NTCO.US?from=2015-01-01&to=2026-07-15"
)
DIVIDEND_RAW_SHA256 = (
    "b2a5b7c6a26165cf4f92618e4a76c06b0cd7de55673fd5cc7162073374469fa0"
)
DIVIDEND_RAW_BYTES = 649
DIVIDEND_RAW_ROWS = 4
QUARANTINED_DIVIDEND_ROWS = 2
QUARANTINED_DIVIDEND_RECORDS_SHA256 = (
    "018d1a12ac421f62ef2a052a4858ae24b21ff29ba8522d07febd0bfa20c916e5"
)
QUARANTINED_DIVIDEND_EVENT_IDS = frozenset(
    {
        "658cb5351b78504a2c20ca3ae75d4d5a2660ea884fc1e2650b1c9a0370551cc0",
        "ebbf2e8b20dfeb94521486d8ed81342ae1fb631c01796857e53795fcafbd163c",
    }
)

OLD_MASTER_SOURCE = "eodhd_exchange_symbols"
OLD_MASTER_URL = "https://eodhd.com/api/exchange-symbol-list/US?delisted=1"
OLD_MASTER_HASH = (
    "8a64e65e316b71e5d165265db2796b68a31f821812f74b63367435b8fcb2ed13"
)
OLD_MASTER_ACTIVE_TO = "2024-04-12"
OLD_HISTORY_SOURCE = "official_market_transition_repair"
OLD_HISTORY_URL = (
    "https://www.sec.gov/Archives/edgar/data/8868/"
    "000095015720000022/0000950157-20-000022.txt"
)
OLD_HISTORY_HASH = (
    "12ca5855e19d9c0c0542f393964ef1e9ee0b1f831c26296f389f143d4bad42a4"
)
OLD_HISTORY_RETRIEVED_AT = "2026-07-18T15:30:00Z"

WRITE_DATASETS = (
    "corporate_actions",
    "daily_price_raw",
    "adjustment_factors",
    "source_archive",
    "lifecycle_resolutions",
    "security_master",
    "symbol_history",
)
REQUIRED_DATASETS = (
    *WRITE_DATASETS,
    "index_membership_events",
    "index_constituent_anchors",
)


@dataclass(frozen=True)
class StagedEvidence:
    source_url: str
    source_sha256: str
    content_bytes: int
    filename: str
    retrieved_at: str
    content: bytes

    def archive_path(self, completed_session: str) -> str:
        return f"archives/{completed_session}/{self.source_sha256}.html.gz"


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    planned_versions: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    evidence: StagedEvidence
    summary: Mapping[str, Any]


Fetcher = Callable[[str, str], bytes]
FailureInjector = Callable[[str], None]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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
    try:
        return None if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if path == base or base not in path.parents:
        raise ValueError(f"NTCO evidence path escapes its root: {relative}.")
    return path


def _normalized_official_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    decoded = html.unescape(re.sub(r"<[^>]+>", " ", decoded))
    return re.sub(r"\s+", " ", decoded).strip().casefold()


def _verify_official_terms(content: bytes) -> None:
    if not content or len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("NTCO SEC response size is outside the reviewed envelope.")
    text = _normalized_official_text(content)
    for alternatives in OFFICIAL_REQUIRED_TEXT_GROUPS:
        if not any(re.search(pattern, text, flags=re.I) for pattern in alternatives):
            raise ValueError(
                "NTCO SEC filing lacks reviewed official term group: "
                + " | ".join(alternatives)
            )


def _require_exact_official_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        url != OFFICIAL_SOURCE_URL
        or parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "www.sec.gov"
        or parsed.query
        or parsed.fragment
        or parsed.path
        != "/Archives/edgar/data/1776967/000155485524000366/ntco-20231231.htm"
    ):
        raise ValueError("NTCO official fetch target is not the exact code-pinned URL.")


def _validate_user_agent(value: str) -> str:
    output = value.strip()
    if not output or "@" not in output:
        raise RuntimeError(
            "SEC_USER_AGENT with a contact email is required for --fetch-official."
        )
    return output


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject 3xx responses before urllib can issue a follow-up request."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, msg, headers
        raise RuntimeError(
            "NTCO official filing returned a redirect; automatic follow-up "
            f"requests are disabled (HTTP {code}, location={newurl})."
        )


def _fetch_once(url: str, user_agent: str) -> bytes:
    _require_exact_official_url(url)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _validate_user_agent(user_agent),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Encoding": "identity",
        },
    )
    opener = urllib.request.build_opener(_NoRedirectHandler())
    with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
        status = int(getattr(response, "status", response.getcode()))
        final_url = str(response.geturl())
        if status != 200:
            raise RuntimeError(f"NTCO official filing returned HTTP {status}.")
        if final_url != url:
            raise RuntimeError(
                "NTCO official filing redirected outside the exact URL: " + final_url
            )
        content = response.read(MAX_RESPONSE_BYTES + 1)
    _verify_official_terms(content)
    return content


def _report_path(evidence_dir: Path) -> Path:
    return evidence_dir / EVIDENCE_REPORT


def _payload_path(evidence_dir: Path, filename: str) -> Path:
    return _safe_path(evidence_dir, filename)


def verify_staged_evidence(evidence_dir: Path) -> StagedEvidence | None:
    report_path = _report_path(evidence_dir)
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("NTCO staged evidence report is unreadable.") from exc
    if set(report) != {
        "schema",
        "status",
        "evidence",
        "http_attempts_total",
        "eodhd_calls",
        "r2_accessed",
    }:
        raise ValueError("NTCO staged evidence report fields are not exact.")
    if (
        report.get("schema") != EVIDENCE_SCHEMA
        or report.get("status") != "collected_pending_reviewer_pin"
        or report.get("http_attempts_total") != MAX_HTTP_ATTEMPTS
        or report.get("eodhd_calls") != 0
        or report.get("r2_accessed") is not False
    ):
        raise ValueError("NTCO staged evidence report contract changed.")
    row = report.get("evidence")
    if not isinstance(row, Mapping) or set(row) != {
        "evidence_id",
        "source_url",
        "source_sha256",
        "content_bytes",
        "filename",
        "retrieved_at",
        "form",
        "period_end",
    }:
        raise ValueError("NTCO staged evidence fields are not exact.")
    if (
        row.get("evidence_id") != EVIDENCE_ID
        or row.get("source_url") != OFFICIAL_SOURCE_URL
        or row.get("form") != "20-F"
        or row.get("period_end") != "2023-12-31"
    ):
        raise ValueError("NTCO staged filing identity changed.")
    _require_exact_official_url(str(row["source_url"]))
    digest = _text(row.get("source_sha256")).lower()
    filename = _text(row.get("filename"))
    size = row.get("content_bytes")
    retrieved_at = _text(row.get("retrieved_at"))
    if (
        len(digest) != 64
        or any(value not in "0123456789abcdef" for value in digest)
        or filename != f"{digest}.html"
        or not isinstance(size, int)
        or size <= 0
        or size > MAX_RESPONSE_BYTES
        or not retrieved_at.endswith("Z")
    ):
        raise ValueError("NTCO staged evidence hash/size metadata is invalid.")
    path = _payload_path(evidence_dir, filename)
    if not path.is_file():
        raise FileNotFoundError(f"NTCO staged evidence payload is missing: {path}.")
    content = path.read_bytes()
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("NTCO staged evidence hash/size verification failed.")
    _verify_official_terms(content)
    return StagedEvidence(
        source_url=OFFICIAL_SOURCE_URL,
        source_sha256=digest,
        content_bytes=size,
        filename=filename,
        retrieved_at=retrieved_at,
        content=content,
    )


def fetch_official(
    evidence_dir: Path,
    *,
    user_agent: str,
    fetcher: Fetcher = _fetch_once,
) -> dict[str, Any]:
    _raise_superseded()
    cached = verify_staged_evidence(evidence_dir)
    if cached is not None:
        return {
            "schema": EVIDENCE_SCHEMA,
            "status": "cache_verified_pending_reviewer_pin",
            "mode": "fetch_official",
            "source_sha256": cached.source_sha256,
            "payload_path": str(_payload_path(evidence_dir, cached.filename)),
            "http_attempts_this_run": 0,
            "network_accessed": False,
            "writes_performed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    url = OFFICIAL_SOURCE_URL
    _require_exact_official_url(url)
    content = fetcher(url, _validate_user_agent(user_agent))
    _verify_official_terms(content)
    digest = hashlib.sha256(content).hexdigest()
    filename = f"{digest}.html"
    path = _payload_path(evidence_dir, filename)
    report_path = _report_path(evidence_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() != content:
        raise RuntimeError(f"Immutable NTCO staged-evidence collision at {path}.")
    if not path.is_file():
        write_atomic(path, content)
    report = {
        "schema": EVIDENCE_SCHEMA,
        "status": "collected_pending_reviewer_pin",
        "evidence": {
            "evidence_id": EVIDENCE_ID,
            "source_url": url,
            "source_sha256": digest,
            "content_bytes": len(content),
            "filename": filename,
            "retrieved_at": _now(),
            "form": "20-F",
            "period_end": "2023-12-31",
        },
        "http_attempts_total": MAX_HTTP_ATTEMPTS,
        "eodhd_calls": 0,
        "r2_accessed": False,
    }
    write_atomic(
        report_path,
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )
    return {
        **report,
        "mode": "fetch_official",
        "payload_path": str(path),
        "http_attempts_this_run": MAX_HTTP_ATTEMPTS,
        "network_accessed": True,
        "writes_performed": True,
    }


def _load_pinned_spec(
    hints_path: Path,
    evidence: StagedEvidence,
) -> tuple[OfficialLifecycleExceptionEvidenceSpec, Mapping[str, OfficialLifecycleExceptionEvidenceSpec]]:
    specs = load_official_lifecycle_exception_evidence(hints_path)
    spec = specs.get(EVIDENCE_ID)
    if spec is None:
        raise RuntimeError(
            f"Reviewer registry entry is missing: {EVIDENCE_ID}.  Stage one never "
            "self-approves its observed hash."
        )
    exact = bool(
        spec.candidate_symbols == (SYMBOL,)
        and spec.candidate_security_ids == (SECURITY_ID,)
        and spec.candidate_last_price_dates == (LAST_NYSE_SESSION,)
        and spec.binding_status == "bound"
        and spec.effective_date == BOUNDARY_EFFECTIVE_DATE
        and spec.resolution_kind == "exception"
        and spec.exception_code == EXCEPTION_CODE
        and not spec.action_type
        and spec.cash_amount is None
        and spec.claim == EXCEPTION_CLAIM
        and spec.source_url == OFFICIAL_SOURCE_URL
        and spec.required_text_groups == REGISTRY_REQUIRED_TEXT_GROUPS
    )
    if not exact:
        raise RuntimeError("NTCO reviewer registry identity/claim/term contract changed.")
    if not spec.source_sha256:
        raise RuntimeError(
            "NTCO official evidence is staged but not reviewer-pinned.  Review the "
            f"payload, then copy exactly {evidence.source_sha256} into "
            f"official_exception_evidence.{EVIDENCE_ID}.source_sha256."
        )
    if spec.source_sha256 != evidence.source_sha256:
        raise RuntimeError(
            "NTCO reviewer-pinned SHA-256 does not match the staged payload: "
            f"pinned={spec.source_sha256}, observed={evidence.source_sha256}."
        )
    _verify_official_terms(evidence.content)
    return spec, specs


def reviewer_registry_draft(evidence: StagedEvidence | None) -> dict[str, Any]:
    return {
        "evidence_id": EVIDENCE_ID,
        "candidate_symbols": [SYMBOL],
        "candidate_name_contains": [NAME_TOKEN],
        "candidate_security_ids": [SECURITY_ID],
        "candidate_last_price_dates": [LAST_NYSE_SESSION],
        "binding_status": "bound",
        "effective_date": BOUNDARY_EFFECTIVE_DATE,
        "resolution_kind": "exception",
        "exception_code": EXCEPTION_CODE,
        "claim": EXCEPTION_CLAIM,
        "source_url": OFFICIAL_SOURCE_URL,
        "source_sha256": evidence.source_sha256 if evidence else "",
        "required_text_groups": [
            list(group) for group in REGISTRY_REQUIRED_TEXT_GROUPS
        ],
    }


def _archive_envelope_records(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    archive_id: str,
    dataset: str,
    source_url: str,
    raw_sha256: str,
    raw_bytes: int,
    raw_rows: int,
) -> list[Mapping[str, Any]]:
    rows = archive.loc[archive["archive_id"].astype(str).eq(archive_id)]
    if len(rows) != 1:
        raise ValueError(f"Expected one archived NTCO {dataset} envelope; got {len(rows)}.")
    row = rows.iloc[0]
    if any(
        _text(row.get(field)) != value
        for field, value in {
            "archive_id": archive_id,
            "source_hash": archive_id,
            "dataset": dataset,
            "source": dataset,
            "source_url": source_url,
            "content_type": "application/vnd.supertrendquant.source-envelope+json",
        }.items()
    ):
        raise ValueError(f"Archived NTCO {dataset} envelope binding changed.")
    path = _safe_path(repository.root, _text(row.get("object_path")))
    if not path.is_file():
        raise FileNotFoundError(f"Archived NTCO {dataset} envelope is missing: {path}.")
    try:
        packed = path.read_bytes()
        envelope_bytes = gzip.decompress(packed)
        envelope = json.loads(envelope_bytes)
        raw = base64.b64decode(envelope["content_base64"], validate=True)
    except (OSError, EOFError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Archived NTCO {dataset} envelope is invalid.") from exc
    if hashlib.sha256(envelope_bytes).hexdigest() != archive_id:
        raise ValueError(f"Archived NTCO {dataset} envelope SHA-256 changed.")
    expected_envelope = {
        "content_sha256": raw_sha256,
        "content_type": "application/json",
        "source": dataset,
        "source_url": source_url,
    }
    if any(_text(envelope.get(field)) != value for field, value in expected_envelope.items()):
        raise ValueError(f"Archived NTCO {dataset} raw-envelope metadata changed.")
    if len(raw) != raw_bytes or hashlib.sha256(raw).hexdigest() != raw_sha256:
        raise ValueError(f"Archived NTCO {dataset} raw payload hash/size changed.")
    try:
        records = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Archived NTCO {dataset} raw payload is invalid JSON.") from exc
    if (
        not isinstance(records, list)
        or len(records) != raw_rows
        or not all(isinstance(item, Mapping) for item in records)
    ):
        raise ValueError(f"Archived NTCO {dataset} raw inventory changed.")
    return list(records)


def _verify_eodhd_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    prices = _archive_envelope_records(
        repository,
        archive,
        archive_id=EOD_ARCHIVE_ID,
        dataset="eodhd_eod",
        source_url=EOD_SOURCE_URL,
        raw_sha256=EOD_RAW_SHA256,
        raw_bytes=EOD_RAW_BYTES,
        raw_rows=EOD_RAW_ROWS,
    )
    dividends = _archive_envelope_records(
        repository,
        archive,
        archive_id=DIVIDEND_ARCHIVE_ID,
        dataset="eodhd_div",
        source_url=DIVIDEND_SOURCE_URL,
        raw_sha256=DIVIDEND_RAW_SHA256,
        raw_bytes=DIVIDEND_RAW_BYTES,
        raw_rows=DIVIDEND_RAW_ROWS,
    )
    tail = [item for item in prices if _date(item.get("date")) > LAST_NYSE_SESSION]
    if (
        len(tail) != QUARANTINED_PRICE_ROWS
        or _date(tail[0].get("date")) != FIRST_QUARANTINED_SESSION
        or _date(tail[-1].get("date")) != LAST_QUARANTINED_SESSION
        or _canonical_json_sha256(tail) != QUARANTINED_PRICE_RECORDS_SHA256
    ):
        raise ValueError("NTCO exact post-NYSE EODHD tail inventory changed.")
    dividend_tail = [
        item for item in dividends if _date(item.get("date")) > LAST_NYSE_SESSION
    ]
    if (
        len(dividend_tail) != QUARANTINED_DIVIDEND_ROWS
        or _canonical_json_sha256(dividend_tail)
        != QUARANTINED_DIVIDEND_RECORDS_SHA256
    ):
        raise ValueError("NTCO exact post-NYSE dividend inventory changed.")
    terminal = [item for item in prices if _date(item.get("date")) == LAST_NYSE_SESSION]
    if len(terminal) != 1 or not _ohlcv_matches(terminal[0], LAST_NYSE_OHLCV):
        raise ValueError("NTCO official terminal-session raw OHLCV changed.")
    return prices, dividends


def _ohlcv_matches(
    row: Mapping[str, Any],
    expected: tuple[float, float, float, float, float],
) -> bool:
    values = tuple(_number(row.get(field)) for field in ("open", "high", "low", "close", "volume"))
    return all(
        value is not None
        and math.isclose(float(value), float(wanted), rel_tol=0, abs_tol=1e-8)
        for value, wanted in zip(values, expected, strict=True)
    )


def _price_state(
    frame: pd.DataFrame,
    raw_records: list[Mapping[str, Any]],
) -> str:
    rows = frame.loc[frame["security_id"].astype(str).eq(SECURITY_ID)].copy()
    if rows.empty:
        raise ValueError("NTCO price inventory is missing.")
    rows["_session"] = pd.to_datetime(rows["session"], errors="coerce").dt.date.astype(str)
    if rows["_session"].eq("NaT").any() or rows["_session"].duplicated().any():
        raise ValueError("NTCO price sessions are invalid or duplicated.")
    raw = {_date(item.get("date")): item for item in raw_records}
    old_dates = set(raw)
    repaired_dates = {value for value in old_dates if value <= LAST_NYSE_SESSION}
    actual_dates = set(rows["_session"])
    if actual_dates == old_dates:
        state = "old"
    elif actual_dates == repaired_dates:
        state = "repaired"
    else:
        raise ValueError(
            "NTCO price inventory is neither exact raw nor exact repaired: "
            f"missing={sorted(old_dates - actual_dates)[:10]}, "
            f"extra={sorted(actual_dates - old_dates)[:10]}."
        )
    for row in rows.to_dict("records"):
        session = str(row["_session"])
        if not _ohlcv_matches(
            row,
            tuple(
                float(_number(raw[session].get(field)) or 0.0)
                for field in ("open", "high", "low", "close", "volume")
            ),
        ):
            raise ValueError(f"NTCO/{session} Parquet OHLCV differs from raw archive.")
        if any(
            _text(row.get(field)) != value
            for field, value in {
                "currency": "USD",
                "source": "eodhd_eod",
                "retrieved_at": EOD_RETRIEVED_AT,
                "source_hash": EOD_RAW_SHA256,
            }.items()
        ) or _text(row.get("source_url")) not in {"", EOD_SOURCE_URL}:
            raise ValueError(f"NTCO/{session} price lineage changed.")
    terminal = rows.loc[rows["_session"].eq(LAST_NYSE_SESSION)]
    if len(terminal) != 1 or not _ohlcv_matches(terminal.iloc[0], LAST_NYSE_OHLCV):
        raise ValueError("NTCO final NYSE Parquet OHLCV changed.")
    return state


def _action_state(
    actions: pd.DataFrame,
    raw_dividends: list[Mapping[str, Any]],
) -> str:
    rows = actions.loc[actions["security_id"].astype(str).eq(SECURITY_ID)].copy()
    if not rows["action_type"].astype(str).eq("cash_dividend").all():
        raise ValueError("NTCO has an unexpected non-dividend corporate action.")
    raw_by_date = {_date(item.get("date")): item for item in raw_dividends}
    old_dates = set(raw_by_date)
    repaired_dates = {value for value in old_dates if value <= LAST_NYSE_SESSION}
    actual_dates = {_date(value) for value in rows["effective_date"]}
    if actual_dates == old_dates:
        state = "old"
    elif actual_dates == repaired_dates:
        state = "repaired"
    else:
        raise ValueError("NTCO provider-dividend inventory is neither old nor repaired.")
    for row in rows.to_dict("records"):
        session = _date(row.get("effective_date"))
        raw = raw_by_date.get(session)
        expected_amount = _number((raw or {}).get("value"))
        if (
            raw is None
            or _date(row.get("ex_date")) != session
            or _number(row.get("cash_amount")) != expected_amount
            or _text(row.get("currency")) != "USD"
            or _text(row.get("source")) != "eodhd_div"
            or _text(row.get("source_url")) != DIVIDEND_SOURCE_URL
            or _text(row.get("source_hash")) != DIVIDEND_RAW_SHA256
            or _text(row.get("retrieved_at")) != EOD_RETRIEVED_AT
        ):
            raise ValueError(f"NTCO/{session} provider dividend differs from raw archive.")
    old_tail_ids = set(
        rows.loc[
            pd.to_datetime(rows["effective_date"]).dt.date.astype(str).gt(
                LAST_NYSE_SESSION
            ),
            "event_id",
        ].astype(str)
    )
    if state == "old" and old_tail_ids != QUARANTINED_DIVIDEND_EVENT_IDS:
        raise ValueError("NTCO post-NYSE dividend event IDs changed.")
    return state


def _identity_state(frame: pd.DataFrame, *, history: bool, source_hash: str) -> str:
    rows = frame.loc[frame["security_id"].astype(str).eq(SECURITY_ID)]
    if len(rows) != 1:
        raise ValueError("NTCO identity row is missing or duplicated.")
    row = rows.iloc[0]
    symbol_field = "symbol" if history else "primary_symbol"
    end_field = "effective_to" if history else "active_to"
    if _text(row.get(symbol_field)).upper() != SYMBOL or _text(row.get("exchange")).upper() != "NYSE":
        raise ValueError("NTCO identity symbol/exchange changed.")
    if history:
        old = bool(
            _date(row.get(end_field)) == OLD_MASTER_ACTIVE_TO
            and _text(row.get("source")) == OLD_HISTORY_SOURCE
            and _text(row.get("source_url")) == OLD_HISTORY_URL
            and _text(row.get("source_hash")) == OLD_HISTORY_HASH
            and _text(row.get("retrieved_at")) == OLD_HISTORY_RETRIEVED_AT
        )
    else:
        old = bool(
            _date(row.get(end_field)) == OLD_MASTER_ACTIVE_TO
            and _text(row.get("source")) == OLD_MASTER_SOURCE
            and _text(row.get("source_url")) == OLD_MASTER_URL
            and _text(row.get("source_hash")) == OLD_MASTER_HASH
        )
    repaired = bool(
        _date(row.get(end_field)) == LAST_NYSE_SESSION
        and _text(row.get("source")) == REPAIRED_IDENTITY_SOURCE
        and _text(row.get("source_url")) == OFFICIAL_SOURCE_URL
        and _text(row.get("source_hash")) == source_hash
        and _text(row.get("retrieved_at")) == REVIEWED_AT
    )
    if old == repaired:
        raise ValueError("NTCO identity is neither exact old nor exact repaired state.")
    return "old" if old else "repaired"


def _resolution_state(
    resolutions: pd.DataFrame,
    evidence: StagedEvidence,
) -> str:
    rows = resolutions.loc[resolutions["security_id"].astype(str).eq(SECURITY_ID)]
    if rows.empty:
        return "old"
    if len(rows) != 1:
        raise ValueError("NTCO lifecycle resolution is duplicated.")
    row = rows.iloc[0]
    expected = _resolution_row(resolutions, evidence)
    if any(_text(row.get(field)) != _text(value) for field, value in expected.items()):
        raise ValueError("NTCO lifecycle resolution differs from the exact reviewed row.")
    return "repaired"


def _resolution_row(
    resolutions: pd.DataFrame,
    evidence: StagedEvidence,
) -> dict[str, Any]:
    row = {column: None for column in resolutions.columns}
    values = {
        "candidate_id": CANDIDATE_ID,
        "security_id": SECURITY_ID,
        "symbol": SYMBOL,
        "last_price_date": LAST_NYSE_SESSION,
        "resolution": "exception",
        "event_id": "",
        "exception_code": EXCEPTION_CODE,
        "exception_reason": EXCEPTION_CLAIM,
        "reviewed_by": REVIEWED_BY,
        "reviewed_at": REVIEWED_AT,
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        "source_url": OFFICIAL_SOURCE_URL,
        "source": RESOLUTION_SOURCE,
        "retrieved_at": evidence.retrieved_at,
        "source_hash": evidence.source_sha256,
    }
    for field, value in values.items():
        if field not in row:
            raise ValueError(f"lifecycle_resolutions lacks required field: {field}.")
        row[field] = value
    return row


def _official_archive_row(
    archive: pd.DataFrame,
    evidence: StagedEvidence,
    completed_session: str,
) -> dict[str, Any]:
    row = {column: None for column in archive.columns}
    values = {
        "archive_id": evidence.source_sha256,
        "dataset": RESOLUTION_SOURCE,
        "object_path": evidence.archive_path(completed_session),
        "content_type": "text/html",
        "effective_date": completed_session,
        "source": RESOLUTION_SOURCE,
        "retrieved_at": evidence.retrieved_at,
        "source_hash": evidence.source_sha256,
        "source_url": OFFICIAL_SOURCE_URL,
    }
    for field, value in values.items():
        if field not in row:
            raise ValueError(f"source_archive lacks required field: {field}.")
        row[field] = value
    return row


def _archive_state(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    evidence: StagedEvidence,
    *,
    completed_session: str,
) -> str:
    expected = _official_archive_row(archive, evidence, completed_session)
    related = (
        archive["archive_id"].astype(str).eq(evidence.source_sha256)
        | archive["source_hash"].astype(str).eq(evidence.source_sha256)
        | archive["source_url"].fillna("").astype(str).eq(OFFICIAL_SOURCE_URL)
    )
    rows = archive.loc[related]
    if rows.empty:
        return "old"
    if len(rows) != 1 or any(
        _text(rows.iloc[0].get(field)) != _text(value)
        for field, value in expected.items()
    ):
        raise ValueError("Conflicting NTCO official source_archive row exists.")
    path = _safe_path(repository.root, evidence.archive_path(completed_session))
    if not path.is_file():
        raise FileNotFoundError(f"Archived NTCO official payload is missing: {path}.")
    try:
        content = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError("Archived NTCO official payload is invalid gzip.") from exc
    if content != evidence.content:
        raise ValueError("Archived NTCO official payload differs from staged evidence.")
    return "repaired"


def _combined_state(states: Mapping[str, str]) -> str:
    unique = set(states.values())
    if len(unique) != 1:
        raise RuntimeError(f"NTCO boundary repair is partially applied: {dict(states)}.")
    return next(iter(unique))


def _rewrite_prices(prices: pd.DataFrame) -> pd.DataFrame:
    sessions = pd.to_datetime(prices["session"], errors="coerce")
    remove = prices["security_id"].astype(str).eq(SECURITY_ID) & sessions.dt.date.astype(str).gt(
        LAST_NYSE_SESSION
    )
    if int(remove.sum()) != QUARANTINED_PRICE_ROWS:
        raise ValueError("NTCO price rewrite is not the exact 43-row quarantine.")
    return prices.loc[~remove].reset_index(drop=True)


def _rewrite_actions(actions: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(actions["effective_date"], errors="coerce")
    remove = (
        actions["security_id"].astype(str).eq(SECURITY_ID)
        & actions["action_type"].astype(str).eq("cash_dividend")
        & dates.dt.date.astype(str).gt(LAST_NYSE_SESSION)
    )
    if int(remove.sum()) != QUARANTINED_DIVIDEND_ROWS or set(
        actions.loc[remove, "event_id"].astype(str)
    ) != QUARANTINED_DIVIDEND_EVENT_IDS:
        raise ValueError("NTCO action rewrite is not the exact two-row quarantine.")
    return actions.loc[~remove].reset_index(drop=True)


def _rewrite_identity(
    frame: pd.DataFrame,
    evidence: StagedEvidence,
    *,
    history: bool,
) -> pd.DataFrame:
    output = frame.copy(deep=True)
    rows = output["security_id"].astype(str).eq(SECURITY_ID)
    if int(rows.sum()) != 1:
        raise ValueError("NTCO identity rewrite requires exactly one row.")
    end_field = "effective_to" if history else "active_to"
    for field, value in {
        end_field: LAST_NYSE_SESSION,
        "source": REPAIRED_IDENTITY_SOURCE,
        "source_url": OFFICIAL_SOURCE_URL,
        "retrieved_at": REVIEWED_AT,
        "source_hash": evidence.source_sha256,
    }.items():
        output.loc[rows, field] = value
    return output.reset_index(drop=True)


def _rewrite_archive(
    archive: pd.DataFrame,
    evidence: StagedEvidence,
    *,
    completed_session: str,
) -> pd.DataFrame:
    row = _official_archive_row(archive, evidence, completed_session)
    output = pd.concat(
        [archive, pd.DataFrame([row]).loc[:, archive.columns]],
        ignore_index=True,
        sort=False,
    )
    keys = list(dataset_spec("source_archive").primary_key)
    if output.duplicated(keys, keep=False).any():
        raise ValueError("NTCO official archive row duplicates an immutable key.")
    return output.reset_index(drop=True)


def _rewrite_resolutions(
    resolutions: pd.DataFrame,
    evidence: StagedEvidence,
) -> pd.DataFrame:
    row = _resolution_row(resolutions, evidence)
    output = pd.concat(
        [resolutions, pd.DataFrame([row]).loc[:, resolutions.columns]],
        ignore_index=True,
        sort=False,
    )
    if output.duplicated(["candidate_id"], keep=False).any():
        raise ValueError("NTCO lifecycle resolution duplicates a candidate_id.")
    return output.reset_index(drop=True)


def _adjustment_source_version(price_version: str, action_version: str) -> str:
    if not price_version or not action_version:
        raise ValueError("NTCO factor lineage requires exact price/action versions.")
    return f"{price_version}+{action_version}"


def _normalized_factor_values(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        ["security_id", "session", "split_factor", "total_return_factor"]
    ].copy()
    output["security_id"] = output["security_id"].astype(str)
    output["session"] = pd.to_datetime(output["session"], errors="raise").dt.normalize()
    if output.duplicated(["security_id", "session"]).any():
        raise ValueError("Adjustment-factor keys are duplicated.")
    return output.sort_values(["security_id", "session"], ignore_index=True)


def _factor_change_counts(
    current: pd.DataFrame,
    expected: pd.DataFrame,
) -> tuple[int, int, int]:
    left = _normalized_factor_values(current)
    right = _normalized_factor_values(expected)
    merged = left.merge(
        right,
        on=["security_id", "session"],
        suffixes=("_old", "_new"),
        how="inner",
        validate="one_to_one",
    )
    split_changed = ~np.isclose(
        pd.to_numeric(merged["split_factor_old"]).to_numpy(float),
        pd.to_numeric(merged["split_factor_new"]).to_numpy(float),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    total_changed = ~np.isclose(
        pd.to_numeric(merged["total_return_factor_old"]).to_numpy(float),
        pd.to_numeric(merged["total_return_factor_new"]).to_numpy(float),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    changed = split_changed | total_changed
    non_target = changed & ~merged["security_id"].eq(SECURITY_ID).to_numpy()
    return int(split_changed.sum()), int(total_changed.sum()), int(non_target.sum())


def _rebuild_factors(
    current: pd.DataFrame,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    expected = build_adjustment_factors(
        prices,
        actions,
        source_version=source_version,
    ).reindex(columns=current.columns)
    expected["source_version"] = source_version
    expected["calculated_at"] = REVIEWED_AT
    expected["source"] = "derived"
    expected["retrieved_at"] = REVIEWED_AT
    expected["source_hash"] = source_version
    expected = expected.reset_index(drop=True)
    current_keys = set(
        zip(
            current["security_id"].astype(str),
            pd.to_datetime(current["session"], errors="raise").dt.normalize(),
        )
    )
    expected_keys = set(
        zip(
            expected["security_id"].astype(str),
            pd.to_datetime(expected["session"], errors="raise").dt.normalize(),
        )
    )
    removed_keys = current_keys - expected_keys
    added_keys = expected_keys - current_keys
    if added_keys or len(removed_keys) != QUARANTINED_PRICE_ROWS or any(
        security_id != SECURITY_ID
        or not (
            FIRST_QUARANTINED_SESSION
            <= session.date().isoformat()
            <= LAST_QUARANTINED_SESSION
        )
        for security_id, session in removed_keys
    ):
        raise ValueError(
            "NTCO factor-key delta is not the exact 43-row quarantined tail."
        )
    split_changed, total_changed, non_target_changed = _factor_change_counts(
        current, expected
    )
    if split_changed != 0 or non_target_changed != 0:
        raise ValueError(
            "NTCO repair unexpectedly changes split factors or non-target economics: "
            f"split={split_changed}, non_target={non_target_changed}."
        )
    return expected, {
        "removed_rows": int(len(current) - len(expected)),
        "retained_total_return_rows_changed": total_changed,
        "retained_split_rows_changed": split_changed,
        "non_target_economic_rows_changed": non_target_changed,
    }


def _new_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"ntco-nyse-boundary-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        frames: Mapping[str, pd.DataFrame],
    ):
        self.base = base
        self.versions = dict(versions)
        self.frames = {key: value.copy(deep=True) for key, value in frames.items()}

    def current_release(self):
        return None, None

    def current_manifest(self, dataset: str):
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.frames:
            return self.frames[dataset].copy(deep=True)
        version = self.versions.get(dataset)
        return self.base.read_frame(dataset, version) if version else pd.DataFrame()


def _candidate_release(
    release: DataRelease,
    versions: Mapping[str, str],
) -> DataRelease:
    return DataRelease(
        version=release.version,
        created_at=release.created_at,
        completed_session=release.completed_session,
        dataset_versions=dict(versions),
        quality=release.quality,
        warnings=release.warnings,
    )


def _validate_candidate_snapshot(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    specs: Mapping[str, OfficialLifecycleExceptionEvidenceSpec],
) -> dict[str, Any]:
    base_report = validate_repository_snapshot(repository)
    unexpected_base_errors = [
        issue
        for issue in base_report.issues
        if issue.severity == "error"
        and issue.code != "index_member_missing_active_symbol"
    ]
    if unexpected_base_errors:
        raise ValueError(
            "Base release has unrelated repository errors before NTCO repair: "
            + "; ".join(issue.message for issue in unexpected_base_errors)
        )
    allowed_identity_gaps = tuple(
        fingerprint
        for issue in base_report.issues
        if issue.code == "index_member_missing_active_symbol"
        for fingerprint in issue.fingerprints
    )
    versions = dict(release.dataset_versions)
    candidate_repository = _CandidateRepository(repository, versions, frames)
    candidate_release = _candidate_release(release, versions)
    candidates = include_bound_official_applied_event_candidates(
        build_lifecycle_candidates(
            candidate_repository,
            release=candidate_release,
        ),
        candidate_repository,
        candidate_release,
        specs,
    )
    candidate_frame = pd.DataFrame([asdict(item) for item in candidates])
    report = validate_lifecycle_coverage(
        candidate_frame,
        frames["lifecycle_resolutions"],
        frames["corporate_actions"],
        completed_session=release.completed_session,
    )
    report.raise_for_errors()
    if not report.valid or report.open_count:
        raise ValueError("NTCO repair does not close the expanded lifecycle set.")
    matches = [item for item in candidates if item.security_id == SECURITY_ID]
    if (
        len(matches) != 1
        or matches[0].last_price_date != LAST_NYSE_SESSION
        or lifecycle_candidate_id(SECURITY_ID, LAST_NYSE_SESSION) != CANDIDATE_ID
    ):
        raise ValueError("Expanded lifecycle candidate does not bind exact NTCO/date.")
    validate_repository_snapshot(
        candidate_repository,
        allowed_index_identity_gap_fingerprints=allowed_identity_gaps,
    ).raise_for_errors()
    return report.manifest_metadata()


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    hints_path: Path = DEFAULT_HINTS,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> PreparedRepair:
    _raise_superseded()
    evidence = verify_staged_evidence(evidence_dir)
    if evidence is None:
        raise FileNotFoundError(
            "NTCO official evidence cache is missing; --apply is cache-only.  "
            "Run --fetch-official only after SEC network access is authorized."
        )
    _spec, specs = _load_pinned_spec(hints_path, evidence)
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    pointer_etags: dict[str, str | None] = {}
    current: dict[str, pd.DataFrame] = {}
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions[dataset]:
            raise RuntimeError(f"{dataset} release/current pointer mismatch.")
        pointer_etags[dataset] = etag
        current[dataset] = repository.read_frame(
            dataset, release.dataset_versions[dataset]
        )

    raw_prices, raw_dividends = _verify_eodhd_evidence(
        repository, current["source_archive"]
    )
    states = {
        "prices": _price_state(current["daily_price_raw"], raw_prices),
        "actions": _action_state(current["corporate_actions"], raw_dividends),
        "master": _identity_state(
            current["security_master"],
            history=False,
            source_hash=evidence.source_sha256,
        ),
        "history": _identity_state(
            current["symbol_history"],
            history=True,
            source_hash=evidence.source_sha256,
        ),
        "resolution": _resolution_state(
            current["lifecycle_resolutions"], evidence
        ),
        "archive": _archive_state(
            repository,
            current["source_archive"],
            evidence,
            completed_session=release.completed_session,
        ),
    }
    state = _combined_state(states)
    if state == "repaired":
        factor_keys = set(
            zip(
                current["adjustment_factors"]["security_id"].astype(str),
                pd.to_datetime(current["adjustment_factors"]["session"]).dt.normalize(),
            )
        )
        price_keys = set(
            zip(
                current["daily_price_raw"]["security_id"].astype(str),
                pd.to_datetime(current["daily_price_raw"]["session"]).dt.normalize(),
            )
        )
        if factor_keys != price_keys:
            raise ValueError("Repaired NTCO factor keys do not match repaired prices.")
        coverage = _validate_candidate_snapshot(repository, release, current, specs)
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            planned_versions={},
            frames={},
            evidence=evidence,
            summary={
                "status": "already_repaired",
                "base_release_version": release.version,
                **coverage,
                "network_accessed": False,
                "eodhd_calls": 0,
                "r2_accessed": False,
            },
        )

    planned_versions = _new_versions(release)
    frames = {key: value.copy(deep=True) for key, value in current.items()}
    frames["daily_price_raw"] = _rewrite_prices(current["daily_price_raw"])
    frames["corporate_actions"] = _rewrite_actions(current["corporate_actions"])
    frames["security_master"] = _rewrite_identity(
        current["security_master"], evidence, history=False
    )
    frames["symbol_history"] = _rewrite_identity(
        current["symbol_history"], evidence, history=True
    )
    frames["source_archive"] = _rewrite_archive(
        current["source_archive"],
        evidence,
        completed_session=release.completed_session,
    )
    frames["lifecycle_resolutions"] = _rewrite_resolutions(
        current["lifecycle_resolutions"], evidence
    )
    factor_lineage = _adjustment_source_version(
        planned_versions["daily_price_raw"],
        planned_versions["corporate_actions"],
    )
    frames["adjustment_factors"], factor_changes = _rebuild_factors(
        current["adjustment_factors"],
        frames["daily_price_raw"],
        frames["corporate_actions"],
        source_version=factor_lineage,
    )
    if factor_changes != {
        "removed_rows": QUARANTINED_PRICE_ROWS,
        "retained_total_return_rows_changed": EOD_RAW_ROWS - QUARANTINED_PRICE_ROWS,
        "retained_split_rows_changed": 0,
        "non_target_economic_rows_changed": 0,
    }:
        raise ValueError(f"NTCO expected factor delta changed: {factor_changes}.")
    for dataset in WRITE_DATASETS:
        validate_dataset(
            dataset,
            frames[dataset],
            incomplete_action_policy="block",
            completed_session=release.completed_session,
        ).raise_for_errors()
    coverage = _validate_candidate_snapshot(repository, release, frames, specs)
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frames={dataset: frames[dataset] for dataset in WRITE_DATASETS},
        evidence=evidence,
        summary={
            "status": "validated_offline_plan",
            "base_release_version": release.version,
            "security_id": SECURITY_ID,
            "symbol": SYMBOL,
            "last_nyse_session": LAST_NYSE_SESSION,
            "boundary_effective_date": BOUNDARY_EFFECTIVE_DATE,
            "ads_program_termination_date": ADS_PROGRAM_TERMINATION_DATE,
            "quarantined_price_rows": QUARANTINED_PRICE_ROWS,
            "quarantined_price_records_sha256": QUARANTINED_PRICE_RECORDS_SHA256,
            "quarantined_dividend_rows": QUARANTINED_DIVIDEND_ROWS,
            "quarantined_dividend_records_sha256": QUARANTINED_DIVIDEND_RECORDS_SHA256,
            "raw_eod_archive_id": EOD_ARCHIVE_ID,
            "raw_dividend_archive_id": DIVIDEND_ARCHIVE_ID,
            "official_source_sha256": evidence.source_sha256,
            "source_archive_rows_added": 1,
            "lifecycle_resolution_rows_added": 1,
            "factor_changes": factor_changes,
            "planned_versions": dict(planned_versions),
            **coverage,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


def readiness_plan(
    repository: LocalDatasetRepository,
    *,
    hints_path: Path = DEFAULT_HINTS,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> dict[str, Any]:
    del repository, hints_path, evidence_dir
    return {
        "status": "superseded_do_not_apply",
        "mode": "plan",
        "reason": SUPERSEDED_REASON,
        "replacement_script": "plan_us_ntco_ntcoy_transition.py",
        "network_accessed": False,
        "writes_performed": False,
        "eodhd_calls": 0,
        "r2_accessed": False,
    }

    evidence = verify_staged_evidence(evidence_dir)
    if evidence is None:
        _require_exact_official_url(OFFICIAL_SOURCE_URL)
        return {
            "status": "ready_for_authorized_one_url_fetch",
            "mode": "plan",
            "source_url": OFFICIAL_SOURCE_URL,
            "max_http_attempts": MAX_HTTP_ATTEMPTS,
            "http_attempts_this_run": 0,
            "reviewer_registry_draft": reviewer_registry_draft(None),
            "network_accessed": False,
            "writes_performed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    try:
        _load_pinned_spec(hints_path, evidence)
    except RuntimeError as exc:
        if "not reviewer-pinned" not in str(exc) and "registry entry is missing" not in str(exc):
            raise
        return {
            "status": "blocked_pending_reviewer_pin",
            "mode": "plan",
            "reason": str(exc),
            "observed_source_sha256": evidence.source_sha256,
            "reviewer_registry_draft": reviewer_registry_draft(evidence),
            "http_attempts_this_run": 0,
            "network_accessed": False,
            "writes_performed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    prepared = prepare_repair(
        repository,
        hints_path=hints_path,
        evidence_dir=evidence_dir,
    )
    return {**prepared.summary, "mode": "plan", "writes_performed": False}


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
            raise RuntimeError("Unresolved NTCO recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted NTCO boundary transaction blocks writes: {journal}."
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
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    release, release_etag = repository.current_release()
    if (
        release is None
        or release.to_bytes() != prepared.release.to_bytes()
        or release_etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after NTCO planning.")
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or pointer.version != prepared.release.dataset_versions[dataset]
            or etag != prepared.pointer_etags[dataset]
        ):
            raise RuntimeError(f"{dataset} pointer changed after NTCO planning.")


def _persist_official(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    evidence = prepared.evidence
    if hashlib.sha256(evidence.content).hexdigest() != evidence.source_sha256:
        raise ValueError("Prepared NTCO official bytes changed before apply.")
    path = _safe_path(
        repository.root,
        evidence.archive_path(prepared.release.completed_session),
    )
    if path.is_file():
        try:
            existing = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise ValueError("Persisted NTCO official payload is invalid gzip.") from exc
        if existing != evidence.content:
            raise ValueError("Persisted NTCO official payload conflicts.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, gzip.compress(evidence.content, mtime=0))
    if gzip.decompress(path.read_bytes()) != evidence.content:
        raise RuntimeError("NTCO official payload post-write verification failed.")


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
) -> dict[str, Any]:
    manifest = repository.manifest_for_version(
        dataset, prepared.release.dataset_versions[dataset]
    )
    metadata = dict(manifest.metadata)
    metadata.update(
        {
            "operation": OPERATION,
            "input_release_version": prepared.release.version,
            "ntco_security_id": SECURITY_ID,
            "ntco_last_nyse_session": LAST_NYSE_SESSION,
            "ntco_boundary_effective_date": BOUNDARY_EFFECTIVE_DATE,
            "ntco_ads_program_termination_date": ADS_PROGRAM_TERMINATION_DATE,
            "ntco_exception_code": EXCEPTION_CODE,
            "ntco_exception_claim": EXCEPTION_CLAIM,
            "ntco_official_source_sha256": prepared.evidence.source_sha256,
            "ntco_quarantined_price_rows": QUARANTINED_PRICE_ROWS,
            "ntco_quarantined_price_records_sha256": QUARANTINED_PRICE_RECORDS_SHA256,
            "ntco_quarantined_dividend_rows": QUARANTINED_DIVIDEND_ROWS,
            "ntco_quarantined_dividend_records_sha256": QUARANTINED_DIVIDEND_RECORDS_SHA256,
            "ntco_raw_eod_archive_id": EOD_ARCHIVE_ID,
            "ntco_raw_dividend_archive_id": DIVIDEND_ARCHIVE_ID,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    if dataset == "adjustment_factors":
        lineage = _adjustment_source_version(
            prepared.planned_versions["daily_price_raw"],
            prepared.planned_versions["corporate_actions"],
        )
        metadata.update(
            {
                "source_version": lineage,
                "source_daily_price_version": prepared.planned_versions[
                    "daily_price_raw"
                ],
                "source_corporate_actions_version": prepared.planned_versions[
                    "corporate_actions"
                ],
                **dict(prepared.summary["factor_changes"]),
            }
        )
    if dataset == "lifecycle_resolutions":
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
        ):
            metadata[key] = prepared.summary[key]
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
                    f"unexpected release during NTCO rollback: {observed.version}"
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
                observed = CurrentPointer.from_bytes(current.data)
                if observed.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected {dataset} pointer during NTCO rollback: "
                        f"{observed.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
    *,
    hints_path: Path,
    evidence_dir: Path,
) -> None:
    current, _ = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes():
        raise RuntimeError("Committed NTCO release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, _etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"Committed NTCO pointer mismatch: {dataset}.")
    replay = prepare_repair(
        repository,
        hints_path=hints_path,
        evidence_dir=evidence_dir,
    )
    if replay.summary["status"] != "already_repaired":
        raise RuntimeError("NTCO boundary repair is not idempotent.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    hints_path: Path = DEFAULT_HINTS,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    _raise_superseded()
    if prepared.summary["status"] == "already_repaired":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_lock(repository):
        _assert_base_unchanged(repository, prepared)
        # Replan while holding the global writer lock.  The caller's mutable
        # DataFrames and UUID versions are never trusted for writes.
        current_plan = prepare_repair(
            repository,
            hints_path=hints_path,
            evidence_dir=evidence_dir,
        )
        if current_plan.summary["status"] == "already_repaired":
            return {
                **current_plan.summary,
                "mode": "apply",
                "writes_performed": False,
            }
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != current_plan.release.dataset_versions[dataset]
                or value.etag != current_plan.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before NTCO apply.")
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_ntco_nyse_boundary_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": current_plan.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": dict(current_plan.planned_versions),
            "official_source_sha256": current_plan.evidence.source_sha256,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            inject("after_journal")
            _persist_official(repository, current_plan)
            inject("after_official_write")
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
                    version=current_plan.planned_versions[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                versions[dataset] = result.manifest.version
                inject(f"after_write:{dataset}")
            for dataset in REQUIRED_DATASETS:
                if dataset in WRITE_DATASETS:
                    continue
                pointer, etag = repository.current_pointer(dataset)
                if (
                    pointer is None
                    or pointer.version != current_plan.release.dataset_versions[dataset]
                    or etag != current_plan.pointer_etags[dataset]
                ):
                    raise RuntimeError(
                        f"Out-of-scope pointer changed during NTCO apply: {dataset}."
                    )
            committed = repository.commit_release(
                current_plan.release.completed_session,
                versions,
                quality=current_plan.release.quality,
                warnings=current_plan.release.warnings,
                expected_etag=current_plan.release_etag,
            )
            inject("after_release_commit")
            _assert_applied_release(
                repository,
                committed,
                hints_path=hints_path,
                evidence_dir=evidence_dir,
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
                "quality": committed.quality,
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=current_plan.planned_versions,
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
                    "NTCO rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage, plan, or transactionally apply the exact NTCO NYSE boundary repair."
        )
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--hints", type=Path, default=DEFAULT_HINTS)
    parser.add_argument("--evidence-dir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch-official", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence_dir = args.evidence_dir or (
        args.cache_root / "state/issuer_lifecycle/ntco_nyse_boundary"
    )
    if args.fetch_official:
        result = fetch_official(
            evidence_dir,
            user_agent=os.getenv("SEC_USER_AGENT", ""),
        )
    else:
        repository = LocalDatasetRepository(args.cache_root)
        if args.apply:
            prepared = prepare_repair(
                repository,
                hints_path=args.hints,
                evidence_dir=evidence_dir,
            )
            result = apply_repair(
                repository,
                prepared,
                hints_path=args.hints,
                evidence_dir=evidence_dir,
            )
        else:
            result = readiness_plan(
                repository,
                hints_path=args.hints,
                evidence_dir=evidence_dir,
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
