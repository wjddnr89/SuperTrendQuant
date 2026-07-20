#!/usr/bin/env python3
"""Fail-closed evidence plan for the NTCO -> NTCOY ADS lifecycle.

The former NYSE-only repair is unsafe: official Cboe and OCC notices identify
an Other-OTC continuation under NTCOY on 2024-02-12 with the same CUSIP and
100-ADS deliverable, and BNY Mellon later identifies a mandatory cash exchange
of USD 5.043659 per ADS effective 2024-09-04.  The intended model is therefore
one same-security ticker change followed by one cash-settled delisting.

This module stages one exact official source per invocation and acquires one
bounded EODHD eod/div/splits bundle with no redirects or retries. Separately
pinned BNY termination notices bind the 2024-08-07 tradability boundary without
changing the original six-raw acquisition signature. Fetch, reviewer promotion,
and transactional apply are deliberately separate modes. Its default mode is
read-only and prints the exact three-call provider plan plus the overlap gates
that must pass before a transactional repair is written.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import json
import os
import re
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import exchange_calendars as xcals
import pandas as pd
import yaml

from supertrend_quant.env import load_env
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import (
    EodhdCallBudget,
    EodhdClient,
    SourceArtifact,
)
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.lifecycle import build_lifecycle_candidates
from supertrend_quant.market_store.lifecycle_coverage import (
    lifecycle_candidate_id,
    validate_lifecycle_coverage,
)
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
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
    DEFAULT_CACHE_ROOT / "state/issuer_lifecycle/ntco_ntcoy_transition/official"
)
DEFAULT_SUPPLEMENTAL_EVIDENCE_DIR = Path("tmp/pdfs/ntco_bny_ad1140774")
DEFAULT_SUPPLEMENTAL_BOOKS_CLOSED_EVIDENCE_DIR = Path(
    "tmp/pdfs/ntco_bny_books_closed"
)
DEFAULT_PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/us_ntco_ntcoy_transition_pins.yaml"
)

SECURITY_ID = "US:EODHD:7e3cea59-409f-5cf0-8429-9d4245013622"
OLD_SYMBOL = "NTCO"
NEW_SYMBOL = "NTCOY"
PROVIDER_SYMBOL = "NTCOY.US"
CUSIP = "63884N108"
ADS_DELIVERABLE = "100 American Depositary Shares"
TICKER_CHANGE_DATE = "2024-02-12"
OFFICIAL_DESTINATION_MARKET = "Other-OTC"
CANONICAL_EXCHANGE = "OTC"
TERMINAL_EFFECTIVE_DATE = "2024-09-04"
TERMINAL_CASH_AMOUNT = Decimal("5.043659")
TERMINAL_CURRENCY = "USD"
PROVIDER_END = "2024-09-03"
MIN_PROVIDER_PRICE_ROWS = 100
MIN_PROVIDER_TERMINAL_SESSION = "2024-08-26"
MAX_PROVIDER_CALENDAR_GAP_DAYS = 10

TICKER_CHANGE_EVENT_ID = canonical_lifecycle_event_id(
    SECURITY_ID, "ticker_change", TICKER_CHANGE_DATE
)
TERMINAL_EVENT_ID = canonical_lifecycle_event_id(
    SECURITY_ID, "delisting", TERMINAL_EFFECTIVE_DATE
)

OFFICIAL_URLS: Mapping[str, str] = {
    "cboe": (
        "https://cdn.cboe.com/resources/product_restriction/2024/"
        "Cboe-Options-Exchanges-Restrictions-on-Transactions-in-Options-on-"
        "Natura-Co-Holding-S-A.pdf"
    ),
    "occ": "https://infomemo.theocc.com/infomemos?number=54105",
    "bny": (
        "https://www.adrbny.com/content/dam/adr/documents/"
        "corporate-actions-dr/files/ad1145447.pdf"
    ),
}
SUPPLEMENTAL_OFFICIAL_URLS: Mapping[str, str] = {
    "bny_termination": (
        "https://www.adrbny.com/content/dam/adr/documents/"
        "corporate-actions-dr/files/ad1140774.pdf"
    ),
    "bny_books_closed": (
        "https://www.adrbny.com/content/dam/adr/documents/"
        "books-closed/files/bc1141635.pdf"
    ),
}
ALL_OFFICIAL_URLS: Mapping[str, str] = {
    **OFFICIAL_URLS,
    **SUPPLEMENTAL_OFFICIAL_URLS,
}
OFFICIAL_FILENAMES = {
    "cboe": "cboe-c2024020910.pdf",
    "occ": "occ-54105.bin",
    "bny": "bny-ad1145447.pdf",
    "bny_termination": "ad1140774.pdf",
    "bny_books_closed": "bc1141635.pdf",
}
OFFICIAL_REVIEWED_CLAIMS: Mapping[str, Mapping[str, str]] = {
    "cboe": {
        "notice_id": "C2024020910",
        "old_symbol": OLD_SYMBOL,
        "new_symbol": NEW_SYMBOL,
        "transition_date": TICKER_CHANGE_DATE,
        "destination_market": OFFICIAL_DESTINATION_MARKET,
    },
    "occ": {
        "memo_number": "54105",
        "old_symbol": OLD_SYMBOL,
        "new_symbol": NEW_SYMBOL,
        "cusip": CUSIP,
        "deliverable": ADS_DELIVERABLE,
    },
    "bny": {
        "action": "mandatory cash exchange",
        "ads_symbol": NEW_SYMBOL,
        "ads_to_underlying_ratio": "1:2",
        "effective_date": TERMINAL_EFFECTIVE_DATE,
        "gross_cash_usd_per_ads": str(TERMINAL_CASH_AMOUNT),
        "fee_usd_per_ads": "0",
        "net_cash_usd_per_ads": str(TERMINAL_CASH_AMOUNT),
    },
}
SUPPLEMENTAL_OFFICIAL_REVIEWED_CLAIMS: Mapping[str, Mapping[str, str]] = {
    "bny_termination": {
        "notice_id": "ad1140774",
        "action": "ADR facility termination",
        "ads_symbol": NEW_SYMBOL,
        "termination_date": "2024-08-07",
        "termination_time": "5:00 PM ET",
    },
    "bny_books_closed": {
        "notice_id": "bc1141635",
        "ads_symbol": NEW_SYMBOL,
        "cusip": CUSIP,
        "exchange": CANONICAL_EXCHANGE,
        "ads_to_underlying_ratio": "1:2",
        "issuance_close_date": "2024-08-08",
        "cancellation_close_date": "2024-08-13",
        "close_reason": "Termination of DR Facility",
    },
}
OFFICIAL_EVIDENCE_SCHEMA = "us_ntco_ntcoy_official_evidence/v1"
MAX_OFFICIAL_HTTP_ATTEMPTS_PER_RUN = 1
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
TIMEOUT_SECONDS = 90

EODHD_ENDPOINTS = ("eod", "div", "splits")
MAX_EODHD_HTTP_ATTEMPTS = 3
EODHD_REQUEST_PARAMS: Mapping[str, str] = {
    "from": TICKER_CHANGE_DATE,
    "to": PROVIDER_END,
}
EODHD_REQUEST_URLS: Mapping[str, str] = {
    endpoint: (
        f"https://eodhd.com/api/{endpoint}/{PROVIDER_SYMBOL}"
        f"?from={TICKER_CHANGE_DATE}&to={PROVIDER_END}"
    )
    for endpoint in EODHD_ENDPOINTS
}

STATE_DIR = "state/issuer_lifecycle/ntco_ntcoy_transition"
QUARANTINE_DIR = f"{STATE_DIR}/quarantine"
REVIEWED_DIR = f"{STATE_DIR}/reviewed"
TRANSACTION_DIR = "transactions/us-ntco-ntcoy-transition"
RECOVERY_DIR = "recovery/us-ntco-ntcoy-transition"
REVIEWED_BY = "us_ntco_ntcoy_transition_repair_v1"
REVIEWED_AT = "2026-07-19T00:00:00Z"
PRICE_IDENTITY_TERMINAL_ONLY = "price_identity_terminal_only"
REJECTED_DIVIDEND_CONFLICT_POLICY = (
    "archive_exact_ntcoy_raw_reject_economics_preserve_ntco_actions"
)
MAX_DIVIDEND_SENSITIVITY_USD_PER_ADS = Decimal("0.01585")
PRICE_ONLY_RELEASE_WARNING = (
    "NTCO/NTCOY: exact NTCOY price/identity/terminal evidence applied; "
    "two conflicting NTCOY provider dividend amounts were rejected and the "
    "existing NTCO dividend economics were preserved (maximum absolute "
    "two-event sensitivity USD 0.01585 per ADS)."
)

WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "lifecycle_resolutions",
    "adjustment_factors",
    "source_archive",
)
OUT_OF_SCOPE_CAS_DATASETS = (
    "index_constituent_anchors",
    "index_membership_events",
)
REQUIRED_DATASETS = (*WRITE_DATASETS, *OUT_OF_SCOPE_CAS_DATASETS)

CURRENT_EOD_SOURCE_URL = (
    "https://eodhd.com/api/eod/NTCO.US?from=2015-01-01&to=2026-07-15"
)
CURRENT_EOD_RAW_BYTES = 120_644
CURRENT_EOD_RAW_ROWS = 1_075
CURRENT_DIVIDEND_SOURCE_URL = (
    "https://eodhd.com/api/div/NTCO.US?from=2015-01-01&to=2026-07-15"
)
CURRENT_DIVIDEND_RAW_BYTES = 649
CURRENT_DIVIDEND_RAW_ROWS = 4
CURRENT_DIVIDEND_RETRIEVED_AT = "2026-07-17T20:37:19.646249Z"
CURRENT_DIVIDEND_EVENT_IDS = frozenset(
    {
        "658cb5351b78504a2c20ca3ae75d4d5a2660ea884fc1e2650b1c9a0370551cc0",
        "ebbf2e8b20dfeb94521486d8ed81342ae1fb631c01796857e53795fcafbd163c",
    }
)
PRESERVED_DIVIDEND_ACTIONS: Mapping[str, Mapping[str, str]] = {
    "658cb5351b78504a2c20ca3ae75d4d5a2660ea884fc1e2650b1c9a0370551cc0": {
        "effective_date": "2024-03-21",
        "cash_amount": "0.28427",
    },
    "ebbf2e8b20dfeb94521486d8ed81342ae1fb631c01796857e53795fcafbd163c": {
        "effective_date": "2024-04-09",
        "cash_amount": "0.01099",
    },
}

# Exact immutable audit bindings of the currently stored NTCO.US responses.
# These rows remain untouched until a separately acquired NTCOY bundle passes
# every overlap gate below.
CURRENT_EOD_ARCHIVE_ID = (
    "e88684de37208bd947df3140593aff81082126aefbc353d545f3ef0ae9fd8883"
)
CURRENT_EOD_RAW_SHA256 = (
    "91cb9baec50c86d49447d78f2882256a991884e46fda1a6019f5df792cb02dde"
)
CURRENT_OVERLAP_PRICE_ROWS = 43
CURRENT_OVERLAP_PRICE_SHA256 = (
    "c1d5c74407f010ee56d829b565900752858e034c505b166a7218cbec3d4d8677"
)
CURRENT_OVERLAP_FIRST_SESSION = "2024-02-12"
CURRENT_OVERLAP_LAST_SESSION = "2024-04-12"
CURRENT_IDENTITY_ACTIVE_FROM = "2020-01-06"
CURRENT_DIVIDEND_ARCHIVE_ID = (
    "50a475c8a45f25d19d831ce7eaaf1f3fbad758600eec7dec45ea5c63d4a171a8"
)
CURRENT_DIVIDEND_RAW_SHA256 = (
    "b2a5b7c6a26165cf4f92618e4a76c06b0cd7de55673fd5cc7162073374469fa0"
)
CURRENT_OVERLAP_DIVIDEND_ROWS = 2
CURRENT_OVERLAP_DIVIDEND_SHA256 = (
    "018d1a12ac421f62ef2a052a4858ae24b21ff29ba8522d07febd0bfa20c916e5"
)

# Exact observed-but-unreviewed NTCOY response profile.  These bindings do not
# approve promotion: they make the real alias divergence a deterministic
# decision input while the reviewer pin fields remain blank.  Any new provider
# response or any changed inventory falls back to the generic fail-closed path.
OBSERVED_UNREVIEWED_QUARANTINE_ID = (
    "7374ab1e5fca9813a25549ea4bd0e6dcbdc3da8465fb9240bb1c9f749ef24f93"
)
OBSERVED_PROVIDER_RAW_SHA256: Mapping[str, str] = {
    "eod": "3ef3a1f03ec97252ac4db079298cdb90ddc32bdeb41fd64a71aaf6d667153e54",
    "div": "6adc67e2b64dd8dcf0acfc0a3bf20bb0d275844f2305b66c1ff4d2a3789d8175",
    "splits": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
}
SUPPLEMENTAL_OFFICIAL_RAW_SHA256: Mapping[str, str] = {
    "bny_termination": (
        "abc23f86378f82f5efb475fc23e0a82e398db4201dc7e52fd4064d088eb47b83"
    ),
    "bny_books_closed": (
        "b5d449fcad19cd38fb461c8f9bae4c1856c0fb7ccd4903e062f87840315ba675"
    ),
}
OBSERVED_PROVIDER_PRICE_ROWS = 123
OBSERVED_PROVIDER_LAST_TRADE_SESSION = "2024-08-07"
MAX_REVIEWED_HIGH_LOW_PRECISION_DELTA = Decimal("0.0005")
OBSERVED_HIGH_LOW_DIFF_INVENTORY = (
    ("2024-02-27", "high", "6.8055", "6.806", "0.0005"),
    ("2024-02-29", "low", "6.4875", "6.488", "0.0005"),
    ("2024-03-19", "high", "7.4189", "7.419", "0.0001"),
    ("2024-04-09", "low", "7.0048", "7.005", "0.0002"),
)
OBSERVED_HIGH_LOW_DIFF_SHA256 = (
    "f990a7be5e12a8547ae2c2361d30c6c816a20d217815aae14e5331e730b747db"
)
OBSERVED_VOLUME_DIFF_INVENTORY = (
    ("2024-02-12", 296642, 296600, -42),
    ("2024-02-13", 2092007, 2117000, 24993),
    ("2024-02-14", 745382, 745400, 18),
    ("2024-02-15", 196222, 196200, -22),
    ("2024-02-16", 54232, 54200, -32),
    ("2024-02-20", 177406, 177400, -6),
    ("2024-02-21", 88720, 88700, -20),
    ("2024-02-22", 322702, 322700, -2),
    ("2024-02-23", 7830, 7800, -30),
    ("2024-02-26", 62943, 62900, -43),
    ("2024-02-27", 142836, 142800, -36),
    ("2024-02-28", 104786, 104800, 14),
    ("2024-02-29", 176812, 176800, -12),
    ("2024-03-01", 94623, 94600, -23),
    ("2024-03-04", 111022, 111000, -22),
    ("2024-03-05", 146391, 146400, 9),
    ("2024-03-06", 154758, 154800, 42),
    ("2024-03-07", 247259, 247900, 641),
    ("2024-03-08", 68103, 68100, -3),
    ("2024-03-11", 20081, 20100, 19),
    ("2024-03-12", 335220, 331100, -4120),
    ("2024-03-13", 39293, 39300, 7),
    ("2024-03-14", 536958, 537000, 42),
    ("2024-03-15", 94157, 94200, 43),
    ("2024-03-18", 97728, 97700, -28),
    ("2024-03-19", 70808, 70800, -8),
    ("2024-03-20", 152248, 152200, -48),
    ("2024-03-21", 55953, 56000, 47),
    ("2024-03-22", 47429, 47400, -29),
    ("2024-03-25", 5026884, 10026900, 5000016),
    ("2024-03-26", 1097026, 1097000, -26),
    ("2024-03-27", 146171, 146200, 29),
    ("2024-03-28", 193921, 193900, -21),
    ("2024-04-01", 30575, 30600, 25),
    ("2024-04-02", 34312, 34300, -12),
    ("2024-04-03", 46402, 46400, -2),
    ("2024-04-04", 279024, 279000, -24),
    ("2024-04-05", 69475, 69500, 25),
    ("2024-04-08", 387002, 387000, -2),
    ("2024-04-09", 85204, 85200, -4),
    ("2024-04-10", 39996, 42900, 2904),
    ("2024-04-11", 76620, 76600, -20),
    ("2024-04-12", 400597, 400800, 203),
)
OBSERVED_VOLUME_DIFF_SHA256 = (
    "ba4f72fd20f2f11b0970c675565244f98e8f0117e6803feeb39a79a6662bf0dc"
)
OBSERVED_DIVIDEND_DIFF_INVENTORY = (
    ("2024-03-21", "0.28427", "0.27036", "-0.01391"),
    ("2024-04-09", "0.01099", "0.01293", "0.00194"),
)
OBSERVED_DIVIDEND_DIFF_SHA256 = (
    "3e4481e59437b4b4257cb8e0ec563f0091884ef6f190c83da2e364e357c23a89"
)

Fetcher = Callable[[str, str], bytes]


@dataclass(frozen=True)
class StagedOfficialEvidence:
    source_key: str
    source_url: str
    source_sha256: str
    content_bytes: int
    filename: str
    retrieved_at: str
    content: bytes


@dataclass(frozen=True)
class RawQuarantine:
    quarantine_id: str
    path: Path
    artifacts: tuple[SourceArtifact, ...]
    budget_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ReviewedBundle:
    artifacts: tuple[SourceArtifact, ...]
    supplemental_artifacts: tuple[SourceArtifact, ...]
    prices: pd.DataFrame
    provider_actions: pd.DataFrame
    provider_last_session: str
    budget_receipt: Mapping[str, Any]
    base_release_version: str
    overlap_report: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedTransition:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    planned_versions: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    archive_artifacts: tuple[SourceArtifact, ...]
    summary: Mapping[str, Any]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject 3xx before urllib can make a hidden follow-up request."""

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
            "Official evidence returned a redirect; automatic follow-up "
            f"requests are disabled (HTTP {code}, location={newurl})."
        )


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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


def _decimal(value: Any) -> Decimal | None:
    text = _text(value)
    if not text:
        return None
    try:
        output = Decimal(text)
    except InvalidOperation:
        return None
    return output if output.is_finite() else None


def _safe_path(root: Path, name: str) -> Path:
    base = root.resolve()
    path = (base / name).resolve()
    if path.parent != base:
        raise ValueError("Evidence filename escapes the exact staging directory.")
    return path


def _source_url(source_key: str) -> str:
    if source_key not in ALL_OFFICIAL_URLS:
        raise ValueError(f"Unknown NTCO official source key: {source_key!r}.")
    url = ALL_OFFICIAL_URLS[source_key]
    if not url.startswith("https://") or "#" in url:
        raise RuntimeError("Code-pinned official evidence URL is invalid.")
    return url


def _validate_user_agent(value: str) -> str:
    output = value.strip()
    if not output or "@" not in output:
        raise RuntimeError(
            "SEC_USER_AGENT with a contact email is required for an official fetch."
        )
    return output


def _validate_media(source_key: str, content: bytes) -> None:
    if not content or len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("Official evidence response size is invalid.")
    if source_key in {
        "cboe",
        "bny",
        "bny_termination",
        "bny_books_closed",
    } and not content.startswith(b"%PDF-"):
        raise ValueError(f"{source_key} official evidence is not a PDF payload.")
    lowered = content[:4096].lower()
    if b"access denied" in lowered or b"request rate threshold" in lowered:
        raise ValueError("Official evidence response is an access-denied page.")


def _fetch_once(url: str, user_agent: str) -> bytes:
    if url not in ALL_OFFICIAL_URLS.values():
        raise ValueError("Official fetch target is not one of the exact code pins.")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _validate_user_agent(user_agent),
            "Accept": "application/pdf,text/html,application/xhtml+xml",
            "Accept-Encoding": "identity",
        },
    )
    opener = urllib.request.build_opener(_NoRedirectHandler())
    with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
        status = int(getattr(response, "status", response.getcode()))
        final_url = str(response.geturl())
        if status != 200:
            raise RuntimeError(f"Official evidence returned HTTP {status}.")
        if final_url != url:
            raise RuntimeError("Official evidence final URL differs from the exact pin.")
        content = response.read(MAX_RESPONSE_BYTES + 1)
    return content


def _report_path(evidence_dir: Path, source_key: str) -> Path:
    return _safe_path(evidence_dir, f"{source_key}.json")


def verify_staged_official(
    evidence_dir: Path, source_key: str
) -> StagedOfficialEvidence | None:
    report_path = _report_path(evidence_dir, source_key)
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid staged {source_key} evidence report.") from exc
    expected = {
        "schema": OFFICIAL_EVIDENCE_SCHEMA,
        "source_key": source_key,
        "source_url": _source_url(source_key),
        "filename": OFFICIAL_FILENAMES[source_key],
    }
    if any(_text(report.get(field)) != value for field, value in expected.items()):
        raise ValueError(f"Staged {source_key} evidence identity changed.")
    digest = _text(report.get("source_sha256")).lower()
    retrieved_at = _text(report.get("retrieved_at"))
    try:
        size = int(report.get("content_bytes"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Staged {source_key} evidence size is invalid.") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or not retrieved_at or size <= 0:
        raise ValueError(f"Staged {source_key} evidence metadata is incomplete.")
    payload = _safe_path(evidence_dir, OFFICIAL_FILENAMES[source_key])
    if not payload.is_file():
        raise FileNotFoundError(f"Staged {source_key} evidence payload is missing.")
    content = payload.read_bytes()
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"Staged {source_key} evidence hash/size changed.")
    _validate_media(source_key, content)
    return StagedOfficialEvidence(
        source_key=source_key,
        source_url=expected["source_url"],
        source_sha256=digest,
        content_bytes=size,
        filename=expected["filename"],
        retrieved_at=retrieved_at,
        content=content,
    )


@contextmanager
def _exclusive_file_lock(path: Path, *, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"{label} lock is already held.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fetch_official_unlocked(
    evidence_dir: Path,
    source_key: str,
    *,
    user_agent: str,
    fetcher: Fetcher = _fetch_once,
) -> dict[str, Any]:
    cached = verify_staged_official(evidence_dir, source_key)
    if cached is not None:
        return {
            "status": "cache_verified_pending_reviewer_pin",
            "source_key": source_key,
            "source_url": cached.source_url,
            "source_sha256": cached.source_sha256,
            "http_attempts_this_run": 0,
            "network_accessed": False,
            "writes_performed": False,
        }
    user_agent = _validate_user_agent(user_agent)
    url = _source_url(source_key)
    content = fetcher(url, user_agent)
    _validate_media(source_key, content)
    digest = hashlib.sha256(content).hexdigest()
    retrieved_at = _now()
    filename = OFFICIAL_FILENAMES[source_key]
    report = {
        "schema": OFFICIAL_EVIDENCE_SCHEMA,
        "source_key": source_key,
        "source_url": url,
        "source_sha256": digest,
        "content_bytes": len(content),
        "filename": filename,
        "retrieved_at": retrieved_at,
        "review_status": "pending_manual_review_and_pin",
    }
    write_atomic(_safe_path(evidence_dir, filename), content)
    write_atomic(_report_path(evidence_dir, source_key), _canonical_json(report) + b"\n")
    return {
        "status": "collected_pending_reviewer_pin",
        "source_key": source_key,
        "source_url": url,
        "source_sha256": digest,
        "http_attempts_this_run": 1,
        "network_accessed": True,
        "writes_performed": True,
    }


def fetch_official(
    evidence_dir: Path,
    source_key: str,
    *,
    user_agent: str,
    fetcher: Fetcher = _fetch_once,
) -> dict[str, Any]:
    if not _report_path(evidence_dir, source_key).is_file():
        _validate_user_agent(user_agent)
    lock_path = evidence_dir / ".locks" / f"{source_key}.lock"
    with _exclusive_file_lock(lock_path, label=f"NTCO official {source_key}"):
        return _fetch_official_unlocked(
            evidence_dir,
            source_key,
            user_agent=user_agent,
            fetcher=fetcher,
        )


def _load_pin_document(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read exact NTCO/NTCOY pin file: {path}.") from exc
    if not isinstance(value, Mapping):
        raise ValueError("NTCO/NTCOY pin file must be a mapping.")
    if value.get("schema") != "us_ntco_ntcoy_transition_pins/v1":
        raise ValueError("NTCO/NTCOY pin schema changed.")
    return value


def validate_pin_contract(path: Path = DEFAULT_PINS) -> Mapping[str, Any]:
    document = _load_pin_document(path)
    sources = document.get("official_sources")
    supplemental_sources = document.get("supplemental_official_sources")
    provider = document.get("provider")
    policy = document.get("decision_policy")
    if not isinstance(sources, Mapping) or set(sources) != set(OFFICIAL_URLS):
        raise ValueError("Official source pin inventory must be exactly Cboe/OCC/BNY.")
    for key, url in OFFICIAL_URLS.items():
        row = sources[key]
        if not isinstance(row, Mapping) or _text(row.get("source_url")) != url:
            raise ValueError(f"Official source URL pin changed for {key}.")
        digest = _text(row.get("source_sha256")).lower()
        if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Official source SHA-256 pin is invalid for {key}.")
        claims = row.get("reviewed_claims")
        expected_claims = OFFICIAL_REVIEWED_CLAIMS[key]
        if (
            not isinstance(claims, Mapping)
            or set(claims) != set(expected_claims)
            or any(
                _text(claims.get(field)) != expected
                for field, expected in expected_claims.items()
            )
        ):
            raise ValueError(f"Official reviewed claim contract changed for {key}.")
    if (
        not isinstance(supplemental_sources, Mapping)
        or set(supplemental_sources) != set(SUPPLEMENTAL_OFFICIAL_URLS)
    ):
        raise ValueError(
            "Supplemental official source pin inventory must contain exactly the "
            "reviewed BNY termination notices."
        )
    for key, url in SUPPLEMENTAL_OFFICIAL_URLS.items():
        row = supplemental_sources[key]
        if not isinstance(row, Mapping) or _text(row.get("source_url")) != url:
            raise ValueError(f"Supplemental official source URL pin changed for {key}.")
        digest = _text(row.get("source_sha256")).lower()
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or digest != SUPPLEMENTAL_OFFICIAL_RAW_SHA256[key]
        ):
            raise ValueError(
                f"Supplemental official source SHA-256 pin is not exact for {key}."
            )
        claims = row.get("reviewed_claims")
        expected_claims = SUPPLEMENTAL_OFFICIAL_REVIEWED_CLAIMS[key]
        if (
            not isinstance(claims, Mapping)
            or set(claims) != set(expected_claims)
            or any(
                _text(claims.get(field)) != expected
                for field, expected in expected_claims.items()
            )
        ):
            raise ValueError(
                f"Supplemental official reviewed claim contract changed for {key}."
            )
    if not isinstance(provider, Mapping):
        raise ValueError("Provider pin contract is missing.")
    if (
        _text(provider.get("provider_symbol")) != PROVIDER_SYMBOL
        or _date(provider.get("from")) != TICKER_CHANGE_DATE
        or _date(provider.get("to")) != PROVIDER_END
        or int(provider.get("max_http_attempts", -1)) != MAX_EODHD_HTTP_ATTEMPTS
    ):
        raise ValueError("EODHD NTCOY request boundary or call cap changed.")
    requests = provider.get("requests")
    if not isinstance(requests, Mapping) or set(requests) != set(EODHD_ENDPOINTS):
        raise ValueError("EODHD NTCOY request inventory changed.")
    for endpoint in EODHD_ENDPOINTS:
        row = requests[endpoint]
        if not isinstance(row, Mapping) or _text(row.get("source_url")) != EODHD_REQUEST_URLS[endpoint]:
            raise ValueError(f"EODHD exact request URL changed for {endpoint}.")
        digest = _text(row.get("source_sha256")).lower()
        if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"EODHD raw SHA-256 pin is invalid for {endpoint}.")
    if not isinstance(policy, Mapping):
        raise ValueError("NTCO/NTCOY decision policy is missing.")
    exact_policy = {
        "security_id": SECURITY_ID,
        "ticker_change_effective_date": TICKER_CHANGE_DATE,
        "ticker_change_old_symbol": OLD_SYMBOL,
        "ticker_change_new_symbol": NEW_SYMBOL,
        "ticker_change_new_exchange": CANONICAL_EXCHANGE,
        "terminal_action_type": "delisting",
        "terminal_effective_date": TERMINAL_EFFECTIVE_DATE,
        "terminal_cash_amount_usd": str(TERMINAL_CASH_AMOUNT),
        "provider_price_policy": "approve_exact_observed_ntcoy_raw",
        "provider_splits_policy": "approve_exact_empty_ntcoy_raw",
        "provider_dividend_policy": REJECTED_DIVIDEND_CONFLICT_POLICY,
        "preserved_dividend_event_ids_sha256": _canonical_sha256(
            sorted(CURRENT_DIVIDEND_EVENT_IDS)
        ),
        "max_dividend_sensitivity_usd_per_ads": str(
            MAX_DIVIDEND_SENSITIVITY_USD_PER_ADS
        ),
        "index_scope_policy": "require_zero_ntco_anchor_and_membership_rows",
    }
    if any(_text(policy.get(field)) != value for field, value in exact_policy.items()):
        raise ValueError("NTCO/NTCOY exact lifecycle decision policy changed.")
    return document


def _safe_hash(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", _text(value).lower()))


def _official_artifacts(evidence_dir: Path) -> tuple[SourceArtifact, ...]:
    artifacts: list[SourceArtifact] = []
    for key in OFFICIAL_URLS:
        staged = verify_staged_official(evidence_dir, key)
        if staged is None:
            raise FileNotFoundError(f"Official {key} raw cache is missing.")
        artifacts.append(
            SourceArtifact(
                source=f"official_{key}",
                source_url=staged.source_url,
                retrieved_at=staged.retrieved_at,
                content=staged.content,
                content_type=(
                    "application/pdf" if key in {"cboe", "bny"} else "text/html"
                ),
            )
        )
    return tuple(artifacts)


def _supplemental_official_artifact(
    evidence_dir: Path = DEFAULT_SUPPLEMENTAL_EVIDENCE_DIR,
    *,
    source_key: str = "bny_termination",
) -> SourceArtifact:
    if source_key not in SUPPLEMENTAL_OFFICIAL_URLS:
        raise ValueError(f"Unknown supplemental official source: {source_key}.")
    staged = verify_staged_official(evidence_dir, source_key)
    if staged is None:
        raise FileNotFoundError(
            f"Supplemental BNY {source_key} raw cache is missing."
        )
    return SourceArtifact(
        source=f"official_{source_key}",
        source_url=staged.source_url,
        retrieved_at=staged.retrieved_at,
        content=staged.content,
        content_type="application/pdf",
    )


def _pin_map(document: Mapping[str, Any]) -> dict[str, str]:
    sources = document["official_sources"]
    requests = document["provider"]["requests"]
    return {
        **{
            OFFICIAL_URLS[key]: _text(sources[key].get("source_sha256")).lower()
            for key in OFFICIAL_URLS
        },
        **{
            EODHD_REQUEST_URLS[endpoint]: _text(
                requests[endpoint].get("source_sha256")
            ).lower()
            for endpoint in EODHD_ENDPOINTS
        },
    }


def _validate_artifact_pins(
    artifacts: Sequence[SourceArtifact],
    document: Mapping[str, Any],
    *,
    official_only: bool = False,
) -> dict[str, str]:
    pins = _pin_map(document)
    expected_urls = (
        tuple(OFFICIAL_URLS.values())
        if official_only
        else (*OFFICIAL_URLS.values(), *EODHD_REQUEST_URLS.values())
    )
    pending = [url for url in expected_urls if not _safe_hash(pins[url])]
    if pending:
        raise ValueError("Reviewer SHA-256 pins are pending for: " + ", ".join(pending))
    observed = {item.source_url: item.source_hash for item in artifacts}
    expected = {url: pins[url] for url in expected_urls}
    if observed != expected:
        raise ValueError("Reviewed raw artifact URL/SHA-256 pins do not match.")
    return expected


def _validate_supplemental_artifact_pin(
    artifact: SourceArtifact,
    document: Mapping[str, Any],
    *,
    source_key: str = "bny_termination",
) -> str:
    expected_url = SUPPLEMENTAL_OFFICIAL_URLS[source_key]
    expected_hash = _text(
        document["supplemental_official_sources"][source_key].get("source_sha256")
    ).lower()
    if artifact.source_url != expected_url or artifact.source_hash != expected_hash:
        raise ValueError(
            f"Supplemental BNY {source_key} raw URL/SHA-256 pin does not match."
        )
    return expected_hash


def _extract_document_text(artifact: SourceArtifact) -> str:
    if artifact.content.startswith(b"%PDF-"):
        try:
            import fitz

            document = fitz.open(stream=artifact.content, filetype="pdf")
            raw = " ".join(page.get_text("text") for page in document)
            document.close()
        except Exception as exc:
            raise ValueError(
                f"Cannot extract reviewed PDF text: {artifact.source_url}."
            ) from exc
    else:
        try:
            raw = artifact.content.decode("utf-8")
        except UnicodeDecodeError:
            raw = artifact.content.decode("latin-1")
        raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", raw).strip().lower()


def _require_claim_groups(
    text: str, groups: Sequence[Sequence[str]], *, label: str
) -> None:
    missing = [
        tuple(group)
        for group in groups
        if not any(term.lower() in text for term in group)
    ]
    if missing:
        raise ValueError(f"Reviewed {label} raw lost required claim groups: {missing}.")


def _validate_official_semantics(artifacts: Sequence[SourceArtifact]) -> dict[str, Any]:
    by_url = {item.source_url: item for item in artifacts}
    if set(by_url) != set(OFFICIAL_URLS.values()):
        raise ValueError("Official raw inventory must be exactly Cboe/OCC/BNY.")
    cboe = _extract_document_text(by_url[OFFICIAL_URLS["cboe"]])
    occ = _extract_document_text(by_url[OFFICIAL_URLS["occ"]])
    bny = _extract_document_text(by_url[OFFICIAL_URLS["bny"]])
    _require_claim_groups(
        cboe,
        (
            ("c2024020910", "c2024-020910"),
            ("natura",),
            ("ntco",),
            ("ntcoy",),
            ("other-otc", "other otc"),
            ("february 12, 2024", "02/12/2024", "2/12/2024"),
        ),
        label="Cboe transition",
    )
    _require_claim_groups(
        occ,
        (
            ("54105",),
            ("ntco",),
            ("ntcoy",),
            ("63884n108",),
            ("deliverable per contract: 100", "new multiplier: 100"),
            ("american depositary shares",),
        ),
        label="OCC identity",
    )
    _require_claim_groups(
        bny,
        (
            ("mandatory cash exchange", "mandatory exchange for cash"),
            ("ntcoy",),
            ("ratio (ads: underlying shares): 1:2", "ratio (ads:underlying shares): 1:2"),
            ("september 4, 2024", "09/04/2024", "9/4/2024"),
            ("5.043659",),
            ("0.000000", "fee $0", "fee: 0", "fee 0"),
        ),
        label="BNY cash termination",
    )
    return {
        "cboe": dict(OFFICIAL_REVIEWED_CLAIMS["cboe"]),
        "occ": dict(OFFICIAL_REVIEWED_CLAIMS["occ"]),
        "bny": dict(OFFICIAL_REVIEWED_CLAIMS["bny"]),
    }


def _validate_supplemental_official_semantics(
    artifact: SourceArtifact,
    *,
    source_key: str = "bny_termination",
) -> dict[str, Any]:
    expected_url = SUPPLEMENTAL_OFFICIAL_URLS[source_key]
    if artifact.source_url != expected_url:
        raise ValueError(f"Supplemental official raw must be BNY notice {source_key}.")
    text = _extract_document_text(artifact)
    if source_key == "bny_termination":
        _require_claim_groups(
            text,
            (
                ("termination notice",),
                ("natura & co holding s.a.", "natura & co holding sa"),
                ("one ads represents two common shares",),
                ("63884n108",),
                ("existing adr facility will be terminated",),
                ("5:00 pm (eastern time)", "5:00 pm eastern time"),
                ("august 7, 2024", "08/07/2024", "8/7/2024"),
                ("until at least august 12, 2024",),
                ("may attempt to sell the underlying shares",),
            ),
            label="BNY ADR-facility termination",
        )
        return {
            **dict(SUPPLEMENTAL_OFFICIAL_REVIEWED_CLAIMS[source_key]),
            "cusip": CUSIP,
            "ads_to_underlying_ratio": "1:2",
            "identity_binding": (
                "CUSIP 63884N108 linked to NTCOY by reviewed OCC memo 54105"
            ),
        }
    if source_key == "bny_books_closed":
        _require_claim_groups(
            text,
            (
                ("books closed / open announcement",),
                ("natura & co holding s.a.", "natura & co holding sa"),
                ("ntcoy",),
                ("63884n108",),
                ("otc",),
                ("1 : 2", "1:2"),
                ("aug 08, 2024", "august 8, 2024", "08/08/2024"),
                ("issuance",),
                ("aug 13, 2024", "august 13, 2024", "08/13/2024"),
                ("cancellation",),
                ("termination of dr facility",),
                ("indefinitely",),
            ),
            label="BNY books-closed termination corroboration",
        )
        return dict(SUPPLEMENTAL_OFFICIAL_REVIEWED_CLAIMS[source_key])
    raise ValueError(f"Unsupported supplemental official source: {source_key}.")


def _load_supplemental_official_artifacts(
    document: Mapping[str, Any],
    *,
    supplemental_evidence_dir: Path = DEFAULT_SUPPLEMENTAL_EVIDENCE_DIR,
    supplemental_books_closed_evidence_dir: Path = (
        DEFAULT_SUPPLEMENTAL_BOOKS_CLOSED_EVIDENCE_DIR
    ),
) -> tuple[SourceArtifact, ...]:
    directories = {
        "bny_termination": supplemental_evidence_dir,
        "bny_books_closed": supplemental_books_closed_evidence_dir,
    }
    output: list[SourceArtifact] = []
    for source_key in SUPPLEMENTAL_OFFICIAL_URLS:
        artifact = _supplemental_official_artifact(
            directories[source_key], source_key=source_key
        )
        _validate_supplemental_artifact_pin(
            artifact, document, source_key=source_key
        )
        _validate_supplemental_official_semantics(
            artifact, source_key=source_key
        )
        output.append(artifact)
    return tuple(output)


def _release_index_absence_audit(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    """Prove the target has no index anchor or membership rows in this release."""

    datasets: dict[str, Any] = {}
    total_target_rows = 0
    for dataset in OUT_OF_SCOPE_CAS_DATASETS:
        version = release.dataset_versions.get(dataset)
        if not version:
            raise ValueError(f"Release lacks index-scope dataset: {dataset}.")
        frame = repository.read_frame(dataset, version)
        if "security_id" not in frame.columns:
            raise ValueError(f"Index-scope dataset lacks security_id: {dataset}.")
        target = frame.loc[frame["security_id"].astype(str).eq(SECURITY_ID)]
        records = target.to_dict("records")
        total_target_rows += len(records)
        datasets[dataset] = {
            "version": version,
            "total_rows": len(frame),
            "target_security_rows": len(records),
            "target_rows_sha256": _canonical_sha256(records),
        }
    if total_target_rows:
        raise ValueError(
            "NTCO price-only policy requires zero index anchor/membership rows."
        )
    return {
        "schema": "us_ntco_ntcoy_index_scope_audit/v1",
        "release_version": release.version,
        "completed_session": release.completed_session,
        "security_id": SECURITY_ID,
        "datasets": datasets,
        "target_row_count": total_target_rows,
        "index_backtest_membership_impact": "none_in_release",
        "absence_proven": True,
    }


def _validate_release_index_absence_audit(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema") != "us_ntco_ntcoy_index_scope_audit/v1"
        or _text(value.get("security_id")) != SECURITY_ID
        or int(value.get("target_row_count", -1)) != 0
        or value.get("absence_proven") is not True
        or _text(value.get("index_backtest_membership_impact"))
        != "none_in_release"
    ):
        raise ValueError("NTCO release index-absence audit changed.")
    datasets = value.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(
        OUT_OF_SCOPE_CAS_DATASETS
    ):
        raise ValueError("NTCO release index-absence dataset inventory changed.")
    empty_sha = _canonical_sha256([])
    for dataset in OUT_OF_SCOPE_CAS_DATASETS:
        row = datasets[dataset]
        if (
            not isinstance(row, Mapping)
            or not _text(row.get("version"))
            or int(row.get("target_security_rows", -1)) != 0
            or _text(row.get("target_rows_sha256")) != empty_sha
        ):
            raise ValueError(f"NTCO index-absence proof changed for {dataset}.")


def _budget_used(budget: EodhdCallBudget) -> int:
    used = int(budget.seed_used)
    if not budget.state_path.is_file():
        return used
    try:
        value = json.loads(budget.state_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("EODHD call budget state is unreadable; fetch refused.") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("EODHD call budget state is invalid; fetch refused.")
    if _text(value.get("period")) == budget.period:
        try:
            persisted = int(value["used"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("EODHD call budget usage is invalid; fetch refused.") from exc
        if persisted < 0:
            raise RuntimeError("EODHD call budget usage is negative; fetch refused.")
        used = max(used, persisted)
    return used


def _assert_budget_capacity(budget: EodhdCallBudget, used: int) -> None:
    if used + MAX_EODHD_HTTP_ATTEMPTS > budget.ceiling:
        raise RuntimeError(
            "EODHD preflight refused a partial three-call NTCOY acquisition: "
            f"used={used}, required={MAX_EODHD_HTTP_ATTEMPTS}, "
            f"safety_ceiling={budget.ceiling}."
        )


class ExactNtcoyEodhdClient(EodhdClient):
    """Exactly eod/div/splits, one shared-ledger claim and one attempt each."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.attempted_endpoints: list[str] = []
        self.claim_positions: list[int] = []

    def get_raw_artifact(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any],
        retrieved_at: str,
    ) -> SourceArtifact:
        normalized = endpoint.strip("/")
        position = len(self.attempted_endpoints)
        if position >= MAX_EODHD_HTTP_ATTEMPTS:
            raise RuntimeError("NTCOY client refused a fourth EODHD request.")
        short = EODHD_ENDPOINTS[position]
        expected = f"{short}/{PROVIDER_SYMBOL}"
        if normalized != expected or dict(params) != dict(EODHD_REQUEST_PARAMS):
            raise RuntimeError("NTCOY client refused a non-reviewed provider request.")
        claim_position = int(self.budget.claim())
        self.claim_positions.append(claim_position)
        self.attempted_endpoints.append(short)
        response = self.session.get(
            f"{self.base_url}/{normalized}",
            params={**EODHD_REQUEST_PARAMS, "api_token": self.token, "fmt": "json"},
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
            timeout=120,
            allow_redirects=False,
        )
        status = int(getattr(response, "status_code", 0) or 0)
        if 300 <= status < 400:
            raise RuntimeError(f"NTCOY EODHD returned forbidden redirect HTTP {status}.")
        response.raise_for_status()
        return SourceArtifact(
            source=f"eodhd_{short}",
            source_url=EODHD_REQUEST_URLS[short],
            retrieved_at=retrieved_at,
            content=bytes(response.content),
            content_type=str(
                getattr(response, "headers", {}).get(
                    "Content-Type", "application/json"
                )
            ),
        )


def _budget_receipt(
    budget: EodhdCallBudget,
    *,
    before: int,
    after: int,
    claim_positions: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema": "eodhd_budget_receipt/v2",
        "period": budget.period,
        "used_before": int(before),
        "used_after": int(after),
        "delta": int(after) - int(before),
        "own_claim_count": len(claim_positions),
        "claim_positions": [int(item) for item in claim_positions],
        "daily_limit": int(budget.limit),
        "reserve": int(budget.reserve),
        "safety_ceiling": int(budget.ceiling),
    }


def _validate_budget_receipt(
    receipt: Mapping[str, Any], *, complete: bool
) -> dict[str, Any]:
    value = dict(receipt)
    required = {
        "schema",
        "period",
        "used_before",
        "used_after",
        "delta",
        "own_claim_count",
        "claim_positions",
        "daily_limit",
        "reserve",
        "safety_ceiling",
    }
    if set(value) != required or value.get("schema") != "eodhd_budget_receipt/v2":
        raise ValueError("NTCOY budget receipt schema changed.")
    before = int(value["used_before"])
    after = int(value["used_after"])
    positions = [int(item) for item in value["claim_positions"]]
    if (
        after - before != int(value["delta"])
        or int(value["own_claim_count"]) != len(positions)
        or positions != sorted(set(positions))
        or any(item <= before or item > after for item in positions)
        or len(positions) > int(value["delta"])
    ):
        raise ValueError("NTCOY budget receipt own-claim proof changed.")
    if complete and len(positions) != MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError("Complete NTCOY receipt must prove exactly three own calls.")
    if int(value["safety_ceiling"]) != int(value["daily_limit"]) - int(
        value["reserve"]
    ):
        raise ValueError("NTCOY budget receipt safety ceiling changed.")
    return value


def _stage_signature() -> dict[str, Any]:
    return {
        "schema": "us_ntco_ntcoy_stage_signature/v1",
        "official_urls": list(OFFICIAL_URLS.values()),
        "eodhd_request_order": [
            {
                "endpoint": endpoint,
                "url": EODHD_REQUEST_URLS[endpoint],
                "params": dict(EODHD_REQUEST_PARAMS),
            }
            for endpoint in EODHD_ENDPOINTS
        ],
        "eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "retry_count": 0,
        "redirects": False,
    }


def _artifact_rows(artifacts: Sequence[SourceArtifact]) -> list[dict[str, Any]]:
    return [
        {
            "source": item.source,
            "source_url": item.source_url,
            "retrieved_at": item.retrieved_at,
            "content_type": item.content_type,
            "content_sha256": item.source_hash,
            "content_base64": base64.b64encode(item.content).decode("ascii"),
        }
        for item in artifacts
    ]


def _write_quarantine(
    cache_root: Path,
    artifacts: Sequence[SourceArtifact],
    receipt: Mapping[str, Any],
    *,
    status: str,
    error: str = "",
) -> tuple[str, Path]:
    if status not in {"complete_unreviewed", "incomplete"}:
        raise ValueError("Unknown NTCOY quarantine status.")
    if status == "complete_unreviewed" and len(artifacts) != 6:
        raise ValueError("Complete NTCOY quarantine requires exactly six raws.")
    envelope = {
        "schema": "us_ntco_ntcoy_raw_quarantine/v1",
        "signature": _stage_signature(),
        "status": status,
        "error": _text(error),
        "budget_receipt": dict(receipt),
        "artifacts": _artifact_rows(artifacts),
    }
    content = _canonical_json(envelope)
    quarantine_id = sha256_bytes(content)
    path = cache_root / QUARANTINE_DIR / f"{quarantine_id}.json.gz"
    encoded = gzip.compress(content, mtime=0)
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Immutable NTCOY quarantine collision: {path}")
    else:
        write_atomic(path, encoded)
    return quarantine_id, path


def _quarantine_path(cache_root: Path, quarantine_id: str) -> Path:
    normalized = _text(quarantine_id).lower()
    if not _safe_hash(normalized):
        raise ValueError("NTCOY quarantine id must be an exact lowercase SHA-256.")
    return cache_root / QUARANTINE_DIR / f"{normalized}.json.gz"


def read_quarantine(cache_root: Path, quarantine_id: str) -> RawQuarantine:
    path = _quarantine_path(cache_root, quarantine_id)
    if not path.is_file():
        raise FileNotFoundError(f"NTCOY quarantine does not exist: {path}")
    try:
        content = gzip.decompress(path.read_bytes())
        envelope = json.loads(content)
    except Exception as exc:
        raise ValueError(f"NTCOY quarantine is unreadable: {path}") from exc
    if content != _canonical_json(envelope):
        raise ValueError("NTCOY quarantine is not canonical JSON.")
    if sha256_bytes(content) != _text(quarantine_id).lower():
        raise ValueError("NTCOY quarantine content-address hash changed.")
    if set(envelope) != {
        "schema",
        "signature",
        "status",
        "error",
        "budget_receipt",
        "artifacts",
    } or envelope.get("schema") != "us_ntco_ntcoy_raw_quarantine/v1":
        raise ValueError("NTCOY quarantine wrapper changed.")
    if envelope.get("signature") != _stage_signature():
        raise ValueError("NTCOY quarantine acquisition signature changed.")
    if envelope.get("status") != "complete_unreviewed":
        raise ValueError("Only a complete NTCOY quarantine can be promoted.")
    receipt = _validate_budget_receipt(envelope["budget_receipt"], complete=True)
    rows = envelope.get("artifacts")
    expected_urls = (*OFFICIAL_URLS.values(), *EODHD_REQUEST_URLS.values())
    if not isinstance(rows, list) or len(rows) != len(expected_urls):
        raise ValueError("NTCOY quarantine does not contain exactly six raws.")
    artifacts: list[SourceArtifact] = []
    required = {
        "source",
        "source_url",
        "retrieved_at",
        "content_type",
        "content_sha256",
        "content_base64",
    }
    for row, expected_url in zip(rows, expected_urls, strict=True):
        if set(row) != required or row.get("source_url") != expected_url:
            raise ValueError("NTCOY quarantine raw URL/order changed.")
        try:
            raw = base64.b64decode(row["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError("NTCOY quarantine raw body is not valid base64.") from exc
        if sha256_bytes(raw) != row.get("content_sha256"):
            raise ValueError("NTCOY quarantine raw body hash changed.")
        artifacts.append(
            SourceArtifact(
                source=_text(row["source"]),
                source_url=_text(row["source_url"]),
                retrieved_at=_text(row["retrieved_at"]),
                content=raw,
                content_type=_text(row["content_type"]),
            )
        )
    return RawQuarantine(
        quarantine_id=_text(quarantine_id).lower(),
        path=path,
        artifacts=tuple(artifacts),
        budget_receipt=receipt,
    )


def assess_quarantine_decision(
    cache_root: Path,
    quarantine_id: str,
    *,
    repository: LocalDatasetRepository | None = None,
    supplemental_evidence_dir: Path = DEFAULT_SUPPLEMENTAL_EVIDENCE_DIR,
    supplemental_books_closed_evidence_dir: Path = (
        DEFAULT_SUPPLEMENTAL_BOOKS_CLOSED_EVIDENCE_DIR
    ),
    pins_path: Path = DEFAULT_PINS,
) -> dict[str, Any]:
    """Audit a complete raw quarantine without pins, promotion, or writes."""

    quarantine = read_quarantine(cache_root, quarantine_id)
    if quarantine.quarantine_id != OBSERVED_UNREVIEWED_QUARANTINE_ID:
        raise ValueError("NTCOY decision audit only accepts the code-pinned quarantine.")
    _validate_official_semantics(quarantine.artifacts[:3])
    provider_rows: dict[str, list[Mapping[str, Any]]] = {}
    provider_hashes: dict[str, str] = {}
    for endpoint, artifact in zip(
        EODHD_ENDPOINTS,
        quarantine.artifacts[3:],
        strict=True,
    ):
        try:
            rows = json.loads(artifact.content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"NTCOY {endpoint} quarantine raw is invalid JSON.") from exc
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError(f"NTCOY {endpoint} quarantine raw must be a row list.")
        provider_rows[endpoint] = rows
        provider_hashes[endpoint] = artifact.source_hash
    if provider_hashes != dict(OBSERVED_PROVIDER_RAW_SHA256):
        raise ValueError("NTCOY observed provider raw hash inventory changed.")
    supplemental_sources_report: dict[str, Any] = {}
    official_last_trade_session = ""
    pin_document: Mapping[str, Any] | None = None
    try:
        pin_document = validate_pin_contract(pins_path)
    except Exception as exc:
        for source_key, source_url in SUPPLEMENTAL_OFFICIAL_URLS.items():
            supplemental_sources_report[source_key] = {
                "state": "missing_or_invalid",
                "source_url": source_url,
                "source_sha256": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        supplemental_dirs = {
            "bny_termination": supplemental_evidence_dir,
            "bny_books_closed": supplemental_books_closed_evidence_dir,
        }
        for source_key, source_url in SUPPLEMENTAL_OFFICIAL_URLS.items():
            try:
                artifact = _supplemental_official_artifact(
                    supplemental_dirs[source_key],
                    source_key=source_key,
                )
                supplemental_hash = _validate_supplemental_artifact_pin(
                    artifact,
                    pin_document,
                    source_key=source_key,
                )
                supplemental_claims = _validate_supplemental_official_semantics(
                    artifact,
                    source_key=source_key,
                )
            except Exception as exc:
                supplemental_sources_report[source_key] = {
                    "state": "missing_or_invalid",
                    "source_url": source_url,
                    "source_sha256": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                supplemental_sources_report[source_key] = {
                    "state": "pinned_semantics_validated",
                    "source_url": artifact.source_url,
                    "source_sha256": supplemental_hash,
                    "reviewed_claims": supplemental_claims,
                }
                if source_key == "bny_termination":
                    official_last_trade_session = _date(
                        supplemental_claims["termination_date"]
                    )
    primary_boundary_gate_closed = bool(
        official_last_trade_session == OBSERVED_PROVIDER_LAST_TRADE_SESSION
    )
    corroboration_valid = bool(
        supplemental_sources_report.get("bny_books_closed", {}).get("state")
        == "pinned_semantics_validated"
    )
    supplemental_report = {
        "primary_source_key": "bny_termination",
        "boundary_gate_closed": primary_boundary_gate_closed,
        "books_closed_corroboration_valid": corroboration_valid,
        "sources": supplemental_sources_report,
    }
    repository = repository or LocalDatasetRepository(cache_root)
    release, _ = repository.current_release()
    if release is None:
        raise RuntimeError("Current release is required for NTCOY decision audit.")
    release_scope_audit = _release_index_absence_audit(repository, release)
    current_prices, current_dividends = current_overlap_records(repository, release)
    report = assess_provider_overlap(
        current_prices=current_prices,
        current_dividends=current_dividends,
        ntcoy_prices=provider_rows["eod"],
        ntcoy_dividends=provider_rows["div"],
        ntcoy_splits=provider_rows["splits"],
        provider_raw_sha256=provider_hashes,
        official_last_trade_session=official_last_trade_session,
    )
    if not report.get("observed_actual_profile"):
        raise ValueError("NTCOY quarantine no longer matches the reviewed diff profile.")
    provider_pins_validated = bool(
        pin_document is not None
        and {
            endpoint: _text(
                pin_document["provider"]["requests"][endpoint]["source_sha256"]
            ).lower()
            for endpoint in EODHD_ENDPOINTS
        }
        == provider_hashes
    )
    promotion_eligible = bool(
        report["status"] == "ready_for_price_identity_terminal_review"
        and primary_boundary_gate_closed
        and corroboration_valid
        and provider_pins_validated
        and release_scope_audit["absence_proven"]
    )
    return {
        **report,
        "schema": "us_ntco_ntcoy_quarantine_decision/v1",
        "quarantine_id": quarantine.quarantine_id,
        "base_release_version": release.version,
        "provider_raw_sha256": provider_hashes,
        "supplemental_official_evidence": supplemental_report,
        "release_scope_audit": release_scope_audit,
        "raw_review_status": (
            "price_identity_terminal_only_ready_for_promotion"
            if promotion_eligible
            else "observed_exact_profile_not_yet_promotable"
        ),
        "budget_receipt": dict(quarantine.budget_receipt),
        "provider_pins_written": False,
        "provider_pins_validated": provider_pins_validated,
        "promotion_eligible": promotion_eligible,
        "promotion_performed": False,
        "network_accessed": False,
        "writes_performed": False,
        "audit_priorities": {
            "P0": [
                *(
                    []
                    if report["terminal_boundary"]["official_boundary_confirmed"]
                    else [
                        "confirm 2024-08-07 last tradable session or obtain the missing tail"
                    ]
                ),
                *(
                    []
                    if provider_pins_validated
                    else ["pin the exact observed EOD, dividend, and empty-splits raws"]
                ),
                "do not promote or apply before every remaining P0 gate closes",
            ],
            "P1": [
                "approve canonical NTCOY exact-hash OHLCV and exact empty splits only",
                "archive the exact NTCOY dividend raw as rejected conflict evidence",
                "preserve both existing NTCO dividend action rows and economic values",
                "retain both alias raws and the full high/low, volume, and dividend inventories",
                "record zero release index-membership rows and USD 0.01585/ADS maximum sensitivity",
            ],
        },
    }


def collect_eodhd_stage(
    cache_root: Path,
    *,
    pins_path: Path = DEFAULT_PINS,
    client_factory: Callable[..., ExactNtcoyEodhdClient] = ExactNtcoyEodhdClient,
    budget_factory: Callable[[], EodhdCallBudget] = EodhdCallBudget,
) -> dict[str, Any]:
    """Use pinned official caches, then spend at most three provider calls."""

    load_env()
    evidence_dir = cache_root / STATE_DIR / "official"
    official = _official_artifacts(evidence_dir)
    document = validate_pin_contract(pins_path)
    _validate_artifact_pins(official, document, official_only=True)
    _validate_official_semantics(official)
    lock_path = cache_root / STATE_DIR / ".locks/eodhd-stage.lock"
    with _exclusive_file_lock(lock_path, label="NTCOY EODHD stage"):
        # Capacity must be checked after acquiring the stage-wide process lock.
        # Otherwise two processes can both preflight the same old counter, then
        # the later process can spend one or two calls before the shared ledger
        # stops its third request.
        budget = budget_factory()
        before = _budget_used(budget)
        _assert_budget_capacity(budget, before)
        client = client_factory(budget=budget)
        artifacts = list(official)
        retrieved_at = utc_now_iso()
        try:
            for endpoint in EODHD_ENDPOINTS:
                artifact = client.get_raw_artifact(
                    f"{endpoint}/{PROVIDER_SYMBOL}",
                    params=EODHD_REQUEST_PARAMS,
                    retrieved_at=retrieved_at,
                )
                artifacts.append(artifact)
                rows = json.loads(artifact.content)
                if not isinstance(rows, list):
                    raise ValueError(f"NTCOY {endpoint} payload must be a JSON list.")
        except BaseException as exc:
            after = _budget_used(budget)
            receipt = _budget_receipt(
                budget,
                before=before,
                after=after,
                claim_positions=client.claim_positions,
            )
            _, path = _write_quarantine(
                cache_root,
                artifacts,
                receipt,
                status="incomplete",
                error=f"{type(exc).__name__}: {exc}",
            )
            if hasattr(exc, "add_note"):
                exc.add_note(f"Partial exact NTCOY raws were preserved at {path}.")
            raise
        after = _budget_used(budget)
        receipt = _budget_receipt(
            budget,
            before=before,
            after=after,
            claim_positions=client.claim_positions,
        )
        _validate_budget_receipt(receipt, complete=True)
        quarantine_id, path = _write_quarantine(
            cache_root,
            artifacts,
            receipt,
            status="complete_unreviewed",
        )
    return {
        "status": "eodhd_stage_fetched_needs_reviewer_pins",
        "network_accessed": True,
        "official_http_attempts_this_run": 0,
        "eodhd_http_attempts_this_run": len(client.attempted_endpoints),
        "retry_count": 0,
        "quarantine_id": quarantine_id,
        "quarantine_path": str(path),
        "budget_receipt": receipt,
        "artifact_sha256": {
            item.source_url: item.source_hash for item in artifacts
        },
    }


def transition_model() -> dict[str, Any]:
    return {
        "security_id": SECURITY_ID,
        "identity_policy": "same_security_id",
        "ticker_change": {
            "event_id": TICKER_CHANGE_EVENT_ID,
            "action_type": "ticker_change",
            "effective_date": TICKER_CHANGE_DATE,
            "old_symbol": OLD_SYMBOL,
            "new_symbol": NEW_SYMBOL,
            "new_security_id": SECURITY_ID,
            "new_exchange": CANONICAL_EXCHANGE,
            "official_destination_market": OFFICIAL_DESTINATION_MARKET,
            "cash_amount": None,
            "ratio": None,
            "official_source_keys": ["cboe", "occ"],
            "official_identity_terms": {
                "cusip": CUSIP,
                "deliverable": ADS_DELIVERABLE,
            },
        },
        "tradability_boundary": {
            "last_trade_session": OBSERVED_PROVIDER_LAST_TRADE_SESSION,
            "facility_termination_time": "5:00 PM ET",
            "next_state": "pending_cash_conversion",
            "forward_fill_allowed": False,
            "primary_official_source_key": "bny_termination",
            "corroborating_official_source_key": "bny_books_closed",
        },
        "terminal": {
            "event_id": TERMINAL_EVENT_ID,
            "action_type": "delisting",
            "effective_date": TERMINAL_EFFECTIVE_DATE,
            "cash_amount": str(TERMINAL_CASH_AMOUNT),
            "currency": TERMINAL_CURRENCY,
            "new_security_id": "",
            "new_symbol": "",
            "official_source_keys": ["bny"],
            "ads_to_underlying_ratio": "1:2",
            "fee_per_ads": "0",
        },
    }


def _price_map(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, tuple[Decimal, ...]], list[str]]:
    output: dict[str, tuple[Decimal, ...]] = {}
    problems: list[str] = []
    for row in rows:
        session = _date(row.get("date", row.get("session")))
        if not session or session in output:
            problems.append(f"invalid_or_duplicate_price_session:{session or '<blank>'}")
            continue
        values = tuple(
            _decimal(row.get(field)) for field in ("open", "high", "low", "close", "volume")
        )
        if any(value is None for value in values):
            problems.append(f"invalid_ohlcv:{session}")
            continue
        open_, high, low, close, volume = values
        assert all(value is not None for value in values)
        if (
            min(open_, high, low, close) <= 0
            or high < max(open_, low, close)
            or low > min(open_, high, close)
            or volume < 0
        ):
            problems.append(f"invalid_ohlcv_envelope:{session}")
            continue
        output[session] = values  # type: ignore[assignment]
    return output, problems


def _dividend_map(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Decimal], list[str]]:
    output: dict[str, Decimal] = {}
    problems: list[str] = []
    for row in rows:
        session = _date(row.get("date", row.get("ex_date")))
        value = _decimal(
            row.get("unadjustedValue", row.get("value", row.get("cash_amount")))
        )
        if not session or value is None or value < 0 or session in output:
            problems.append(f"invalid_or_duplicate_dividend:{session or '<blank>'}")
            continue
        output[session] = value
    return output, problems


def _decimal_string(value: Decimal) -> str:
    return format(value, "f")


def _price_field_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Decimal]]:
    output: dict[str, dict[str, Decimal]] = {}
    for row in rows:
        session = _date(row.get("date", row.get("session")))
        if not session or session in output:
            continue
        values = {
            field: _decimal(row.get(field))
            for field in ("open", "high", "low", "close", "volume")
        }
        values["adjusted_close"] = _decimal(
            row.get("adjusted_close", row.get("close"))
        )
        if any(value is None for value in values.values()):
            continue
        output[session] = {
            field: value for field, value in values.items() if value is not None
        }
    return output


def _high_low_diff_inventory(
    current: Mapping[str, Mapping[str, Decimal]],
    canonical: Mapping[str, Mapping[str, Decimal]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for session in sorted(set(current) & set(canonical)):
        for field in ("high", "low"):
            old = current[session][field]
            new = canonical[session][field]
            if old != new:
                output.append(
                    {
                        "session": session,
                        "field": field,
                        "current": _decimal_string(old),
                        "canonical": _decimal_string(new),
                        "delta": _decimal_string(new - old),
                    }
                )
    return output


def _volume_diff_inventory(
    current: Mapping[str, Mapping[str, Decimal]],
    canonical: Mapping[str, Mapping[str, Decimal]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for session in sorted(set(current) & set(canonical)):
        old = current[session]["volume"]
        new = canonical[session]["volume"]
        if old != new:
            if old != old.to_integral_value() or new != new.to_integral_value():
                return []
            old_int = int(old)
            new_int = int(new)
            output.append(
                {
                    "session": session,
                    "current": old_int,
                    "canonical": new_int,
                    "delta": new_int - old_int,
                }
            )
    return output


def _dividend_diff_inventory(
    current: Mapping[str, Decimal],
    canonical: Mapping[str, Decimal],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for session in sorted(set(current) & set(canonical)):
        old = current[session]
        new = canonical[session]
        if old != new:
            output.append(
                {
                    "session": session,
                    "current": _decimal_string(old),
                    "canonical": _decimal_string(new),
                    "delta": _decimal_string(new - old),
                }
            )
    return output


def _expected_high_low_diff_inventory() -> list[dict[str, str]]:
    return [
        {
            "session": session,
            "field": field,
            "current": current,
            "canonical": canonical,
            "delta": delta,
        }
        for session, field, current, canonical, delta in OBSERVED_HIGH_LOW_DIFF_INVENTORY
    ]


def _expected_volume_diff_inventory() -> list[dict[str, Any]]:
    return [
        {
            "session": session,
            "current": current,
            "canonical": canonical,
            "delta": delta,
        }
        for session, current, canonical, delta in OBSERVED_VOLUME_DIFF_INVENTORY
    ]


def _expected_dividend_diff_inventory() -> list[dict[str, str]]:
    return [
        {
            "session": session,
            "current": current,
            "canonical": canonical,
            "delta": delta,
        }
        for session, current, canonical, delta in OBSERVED_DIVIDEND_DIFF_INVENTORY
    ]


def assess_provider_overlap(
    *,
    current_prices: Sequence[Mapping[str, Any]],
    current_dividends: Sequence[Mapping[str, Any]],
    ntcoy_prices: Sequence[Mapping[str, Any]],
    ntcoy_dividends: Sequence[Mapping[str, Any]],
    ntcoy_splits: Sequence[Mapping[str, Any]],
    provider_raw_sha256: Mapping[str, str] | None = None,
    official_last_trade_session: str = "",
) -> dict[str, Any]:
    """Classify cached provider data without treating alias equality as truth.

    The official transition makes NTCOY the canonical provider symbol after
    2024-02-12, but it does not prove dividend economics or the final trading
    session.  The one observed divergent response is therefore code-bound as a
    replacement *candidate* and remains blocked pending independent evidence.
    A last-trade session is trusted only when the caller first validates the
    separately pinned BNY facility-termination notice.
    """

    blockers: list[str] = []
    if len(current_prices) != CURRENT_OVERLAP_PRICE_ROWS:
        blockers.append("current_ntco_price_row_count_changed")
    if _canonical_sha256(current_prices) != CURRENT_OVERLAP_PRICE_SHA256:
        blockers.append("current_ntco_price_archive_binding_changed")
    if len(current_dividends) != CURRENT_OVERLAP_DIVIDEND_ROWS:
        blockers.append("current_ntco_dividend_row_count_changed")
    if _canonical_sha256(current_dividends) != CURRENT_OVERLAP_DIVIDEND_SHA256:
        blockers.append("current_ntco_dividend_archive_binding_changed")

    current_price_map, problems = _price_map(current_prices)
    blockers.extend(f"current:{item}" for item in problems)
    current_sessions = sorted(current_price_map)
    if (
        not current_sessions
        or current_sessions[0] != CURRENT_OVERLAP_FIRST_SESSION
        or current_sessions[-1] != CURRENT_OVERLAP_LAST_SESSION
    ):
        blockers.append("current_ntco_price_boundary_changed")
    provider_price_map, problems = _price_map(ntcoy_prices)
    blockers.extend(f"ntcoy:{item}" for item in problems)
    current_fields = _price_field_map(current_prices)
    provider_fields = _price_field_map(ntcoy_prices)
    current_dividend_map, problems = _dividend_map(current_dividends)
    blockers.extend(f"current:{item}" for item in problems)
    provider_dividend_map, problems = _dividend_map(ntcoy_dividends)
    blockers.extend(f"ntcoy:{item}" for item in problems)

    provider_sessions = sorted(provider_price_map)
    if not provider_sessions:
        blockers.append("ntcoy_eod_is_empty")
    else:
        if provider_sessions[0] != TICKER_CHANGE_DATE:
            blockers.append("ntcoy_first_session_is_not_transition_date")
        if provider_sessions[-1] > PROVIDER_END:
            blockers.append("ntcoy_price_after_bounded_pretermination_window")
        if len(provider_sessions) < MIN_PROVIDER_PRICE_ROWS:
            blockers.append("ntcoy_price_tail_too_short")
        gaps = [
            (pd.Timestamp(right) - pd.Timestamp(left)).days
            for left, right in zip(provider_sessions, provider_sessions[1:])
        ]
        if gaps and max(gaps) > MAX_PROVIDER_CALENDAR_GAP_DAYS:
            blockers.append("ntcoy_price_tail_has_unreviewed_gap")
        valid_xnys_sessions = {
            value.date().isoformat()
            for value in xcals.get_calendar("XNYS").sessions_in_range(
                TICKER_CHANGE_DATE,
                PROVIDER_END,
            )
        }
        if any(session not in valid_xnys_sessions for session in provider_sessions):
            blockers.append("ntcoy_price_contains_non_xnys_session")
    if ntcoy_splits:
        blockers.append("ntcoy_split_requires_separate_official_review")
    if any(
        session < TICKER_CHANGE_DATE or session > PROVIDER_END
        for session in provider_dividend_map
    ):
        blockers.append("ntcoy_dividend_outside_bounded_request_window")

    compared_fields = ("open", "high", "low", "close", "adjusted_close", "volume")
    field_mismatch_sessions = {
        field: sorted(
            session
            for session in current_fields
            if session not in provider_fields
            or current_fields[session][field] != provider_fields[session][field]
        )
        for field in compared_fields
    }
    price_mismatches = sorted(
        {
            session
            for sessions in field_mismatch_sessions.values()
            for session in sessions
        }
    )
    dividend_mismatches = sorted(
        session
        for session, expected in current_dividend_map.items()
        if provider_dividend_map.get(session) != expected
    )
    high_low_inventory = _high_low_diff_inventory(current_fields, provider_fields)
    volume_inventory = _volume_diff_inventory(current_fields, provider_fields)
    dividend_inventory = _dividend_diff_inventory(
        current_dividend_map,
        provider_dividend_map,
    )
    expected_high_low = _expected_high_low_diff_inventory()
    expected_volume = _expected_volume_diff_inventory()
    expected_dividends = _expected_dividend_diff_inventory()
    if (
        _canonical_sha256(expected_high_low) != OBSERVED_HIGH_LOW_DIFF_SHA256
        or _canonical_sha256(expected_volume) != OBSERVED_VOLUME_DIFF_SHA256
        or _canonical_sha256(expected_dividends) != OBSERVED_DIVIDEND_DIFF_SHA256
    ):
        raise RuntimeError("Code-pinned NTCO alias-difference inventory is corrupt.")
    core_price_fields_exact = all(
        not field_mismatch_sessions[field]
        for field in ("open", "close", "adjusted_close")
    )
    precision_inventory_exact = high_low_inventory == expected_high_low and all(
        abs(Decimal(row["delta"])) <= MAX_REVIEWED_HIGH_LOW_PRECISION_DELTA
        for row in high_low_inventory
    )
    volume_inventory_exact = volume_inventory == expected_volume
    dividend_inventory_exact = dividend_inventory == expected_dividends
    overlap_session_inventory_exact = set(current_fields).issubset(provider_fields)
    raw_hashes = {
        endpoint: _text((provider_raw_sha256 or {}).get(endpoint)).lower()
        for endpoint in EODHD_ENDPOINTS
    }
    observed_raw_hash_profile = raw_hashes == dict(OBSERVED_PROVIDER_RAW_SHA256)
    observed_boundary_profile = bool(
        provider_sessions
        and len(provider_sessions) == OBSERVED_PROVIDER_PRICE_ROWS
        and provider_sessions[0] == TICKER_CHANGE_DATE
        and provider_sessions[-1] == OBSERVED_PROVIDER_LAST_TRADE_SESSION
    )
    observed_alias_diff_profile = bool(
        overlap_session_inventory_exact
        and core_price_fields_exact
        and precision_inventory_exact
        and volume_inventory_exact
        and dividend_inventory_exact
        and set(current_dividend_map) == set(provider_dividend_map)
    )
    observed_actual_profile = bool(
        observed_alias_diff_profile
        and observed_boundary_profile
        and observed_raw_hash_profile
        and not ntcoy_splits
    )
    official_boundary_confirmed = bool(
        observed_actual_profile
        and official_last_trade_session == OBSERVED_PROVIDER_LAST_TRADE_SESSION
        and provider_sessions[-1] == official_last_trade_session
    )
    if observed_actual_profile:
        # The two alias-dividend values are deliberately not selected.  Their
        # exact raw is retained as rejected conflict evidence while the two
        # already stored NTCO corporate-action rows remain the economic input.
        # This narrow policy is allowed only for the code-pinned raw profile.
        if not official_boundary_confirmed:
            blockers.append("independent_last_trade_boundary_evidence_required")
    else:
        if price_mismatches:
            blockers.append("ntco_ntcoy_ohlcv_overlap_mismatch")
        if dividend_mismatches:
            blockers.append("ntco_ntcoy_dividend_overlap_mismatch")
        if (
            provider_sessions
            and provider_sessions[-1] < MIN_PROVIDER_TERMINAL_SESSION
        ):
            blockers.append("ntcoy_terminal_price_too_early")
        if observed_alias_diff_profile and not observed_raw_hash_profile:
            blockers.append("observed_alias_profile_requires_exact_raw_hash_binding")
    if any(session >= TERMINAL_EFFECTIVE_DATE for session in provider_dividend_map):
        blockers.append("provider_dividend_on_or_after_official_cash_termination")

    unique_blockers = sorted(set(blockers))
    independent_evidence_blockers = {
        "independent_last_trade_boundary_evidence_required",
    }
    independent_only = bool(unique_blockers) and set(unique_blockers).issubset(
        independent_evidence_blockers
    )
    if not unique_blockers:
        status = (
            "ready_for_price_identity_terminal_review"
            if observed_actual_profile
            else "ready_for_transaction_design_review"
        )
    elif independent_only:
        status = "blocked_pending_independent_evidence"
    else:
        status = "blocked_provider_mismatch"
    unobserved_sessions: list[str] = []
    if provider_sessions and provider_sessions[-1] < PROVIDER_END:
        unobserved_sessions = [
            value.date().isoformat()
            for value in xcals.get_calendar("XNYS").sessions_in_range(
                provider_sessions[-1],
                PROVIDER_END,
            )
            if value.date().isoformat() > provider_sessions[-1]
        ]
    return {
        "status": status,
        "apply_allowed": False,
        "blockers": unique_blockers,
        "price_overlap_rows": len(current_price_map),
        "price_overlap_mismatch_sessions": sorted(price_mismatches),
        "field_overlap_exact_rows": {
            field: len(current_fields) - len(field_mismatch_sessions[field])
            for field in compared_fields
        },
        "field_overlap_mismatch_sessions": field_mismatch_sessions,
        "dividend_overlap_rows": len(current_dividend_map),
        "dividend_overlap_mismatch_sessions": sorted(dividend_mismatches),
        "ntcoy_price_rows": len(provider_price_map),
        "ntcoy_first_session": provider_sessions[0] if provider_sessions else "",
        "ntcoy_last_session": provider_sessions[-1] if provider_sessions else "",
        "ntcoy_new_price_rows_after_current_tail": sum(
            session > CURRENT_OVERLAP_LAST_SESSION for session in provider_price_map
        ),
        "observed_raw_hash_profile": observed_raw_hash_profile,
        "observed_alias_diff_profile": observed_alias_diff_profile,
        "observed_actual_profile": observed_actual_profile,
        "canonical_price_replacement_candidate": observed_actual_profile,
        "decision_mode": (
            PRICE_IDENTITY_TERMINAL_ONLY
            if observed_actual_profile
            else "full_exact_provider_bundle"
        ),
        "difference_inventory": {
            "high_low": high_low_inventory,
            "high_low_sha256": _canonical_sha256(high_low_inventory),
            "high_low_max_allowed_delta": str(
                MAX_REVIEWED_HIGH_LOW_PRECISION_DELTA
            ),
            "volume": volume_inventory,
            "volume_sha256": _canonical_sha256(volume_inventory),
            "dividends": dividend_inventory,
            "dividends_sha256": _canonical_sha256(dividend_inventory),
        },
        "dividend_ambiguity": {
            "classification": (
                "exact_provider_alias_conflict_rejected"
                if observed_actual_profile
                else "gross_net_or_fx_fee_unresolved"
            ),
            "automatic_selection_allowed": False,
            "provider_economics_accepted": False,
            "policy": (
                REJECTED_DIVIDEND_CONFLICT_POLICY
                if observed_actual_profile
                else "unreviewed"
            ),
            "preserved_event_ids": (
                sorted(CURRENT_DIVIDEND_EVENT_IDS)
                if observed_actual_profile
                else []
            ),
            "maximum_absolute_sensitivity_usd_per_ads": (
                str(MAX_DIVIDEND_SENSITIVITY_USD_PER_ADS)
                if observed_actual_profile
                else ""
            ),
            "records": [
                {
                    "session": row["session"],
                    "ntco_alias_amount": row["current"],
                    "ntcoy_canonical_amount": row["canonical"],
                    "accepted_amount": row["current"],
                    "accepted_source": "preserved_current_ntco_corporate_action",
                    "rejected_amount": row["canonical"],
                    "rejection_reason": "conflicts_with_preserved_ntco_economics",
                }
                for row in dividend_inventory
            ]
            if observed_actual_profile
            else [],
        },
        "terminal_boundary": {
            "last_observed_trade_session": (
                provider_sessions[-1] if provider_sessions else ""
            ),
            "provider_request_end": PROVIDER_END,
            "unobserved_xnys_sessions_before_request_end": unobserved_sessions,
            "official_facility_termination_session": (
                official_last_trade_session if official_boundary_confirmed else ""
            ),
            "official_boundary_confirmed": official_boundary_confirmed,
            "official_cash_conversion_effective_date": TERMINAL_EFFECTIVE_DATE,
            "modeling_rule": (
                "never forward-fill a tradable price; keep the position pending "
                "conversion after the independently confirmed last trade, then "
                "apply BNY cash on 2024-09-04"
            ),
        },
        "decision": {
            "canonical_prices": (
                "exact-hash replacement candidate because official sources bind "
                "NTCO->NTCOY as the same ADS; price-only promotion is allowed "
                "after the official last-trade boundary validates"
                if observed_actual_profile
                else "not eligible"
            ),
            "volume": (
                "preserve the exact 43-row alias difference inventory and use "
                "canonical NTCOY only if the replacement is later approved"
                if observed_actual_profile
                else "unreviewed mismatch"
            ),
            "dividends": (
                "reject both exact NTCOY alias amounts as transaction economics, "
                "archive their raw and diff inventory, and preserve the existing "
                "NTCO corporate-action rows and values"
                if observed_actual_profile
                else "unreviewed mismatch"
            ),
            "last_trade": (
                "confirmed as 2024-08-07 by exact-pinned BNY facility-termination "
                "notice ad1140774; do not forward-fill afterward"
                if official_boundary_confirmed
                else "blocked until an independent official source confirms "
                "2024-08-07 as the last tradable session or supplies the missing tail"
                if observed_actual_profile
                else "unreviewed boundary"
            ),
        },
        "accepted_overlap_policy": (
            "approve exact NTCOY prices and exact empty splits only; archive "
            "the NTCOY dividend raw as rejected conflict evidence and preserve "
            "the existing NTCO dividend actions"
            if observed_actual_profile and not unique_blockers
            else "replace the 43 post-NYSE rows and matching dividends with the "
            "exact pinned NTCOY raw bundle; preserve the original NTCO.US "
            "envelopes and both comparison hashes in source_archive"
            if not unique_blockers
            else (
                "none yet; retain the exact-hash NTCOY canonical replacement "
                "candidate in quarantine pending exact official last-trade evidence"
                if observed_actual_profile
                else "none; quarantine both raw bundles and perform no release write"
            )
        ),
        "transition_model": transition_model(),
    }


def _safe_repository_path(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if path == base or base not in path.parents:
        raise ValueError("Archived object path escapes the repository root.")
    return path


def _archived_raw_records(
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
        raise ValueError(f"Exact archived {dataset} envelope is missing or ambiguous.")
    row = rows.iloc[0]
    expected = {
        "dataset": dataset,
        "source_url": source_url,
    }
    # The immutable archive rows produced by the original ingestion stored the
    # content-addressed envelope id in ``source_hash``.  Newer rows store the
    # embedded raw-response hash there instead.  Both representations remain
    # cryptographically bound below: the row is selected by the exact pinned
    # archive id, the decompressed envelope must hash to that id, and its raw
    # payload must hash to ``raw_sha256``.
    legacy_or_current_source_hash = _text(row.get("source_hash")) in {
        archive_id,
        raw_sha256,
    }
    if (
        any(_text(row.get(field)) != value for field, value in expected.items())
        or not legacy_or_current_source_hash
    ):
        raise ValueError(f"Archived {dataset} envelope binding changed.")
    path = _safe_repository_path(repository.root, _text(row.get("object_path")))
    if not path.is_file():
        raise FileNotFoundError(f"Archived {dataset} envelope object is missing: {path}")
    try:
        envelope_bytes = gzip.decompress(path.read_bytes())
        envelope = json.loads(envelope_bytes)
        raw = base64.b64decode(envelope["content_base64"], validate=True)
    except Exception as exc:
        raise ValueError(f"Archived {dataset} envelope is invalid.") from exc
    if sha256_bytes(envelope_bytes) != archive_id:
        raise ValueError(f"Archived {dataset} envelope content-address changed.")
    if (
        _text(envelope.get("source")) != dataset
        or _text(envelope.get("source_url")) != source_url
        or _text(envelope.get("content_sha256")) != raw_sha256
        or len(raw) != raw_bytes
        or sha256_bytes(raw) != raw_sha256
    ):
        raise ValueError(f"Archived {dataset} raw binding changed.")
    try:
        records = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Archived {dataset} raw JSON is invalid.") from exc
    if not isinstance(records, list) or len(records) != raw_rows:
        raise ValueError(f"Archived {dataset} raw row inventory changed.")
    return records


def current_overlap_records(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    all_prices = _archived_raw_records(
        repository,
        archive,
        archive_id=CURRENT_EOD_ARCHIVE_ID,
        dataset="eodhd_eod",
        source_url=CURRENT_EOD_SOURCE_URL,
        raw_sha256=CURRENT_EOD_RAW_SHA256,
        raw_bytes=CURRENT_EOD_RAW_BYTES,
        raw_rows=CURRENT_EOD_RAW_ROWS,
    )
    all_dividends = _archived_raw_records(
        repository,
        archive,
        archive_id=CURRENT_DIVIDEND_ARCHIVE_ID,
        dataset="eodhd_div",
        source_url=CURRENT_DIVIDEND_SOURCE_URL,
        raw_sha256=CURRENT_DIVIDEND_RAW_SHA256,
        raw_bytes=CURRENT_DIVIDEND_RAW_BYTES,
        raw_rows=CURRENT_DIVIDEND_RAW_ROWS,
    )
    prices = [
        item
        for item in all_prices
        if _date(item.get("date")) >= TICKER_CHANGE_DATE
    ]
    dividends = [
        item
        for item in all_dividends
        if _date(item.get("date")) >= TICKER_CHANGE_DATE
    ]
    if (
        len(prices) != CURRENT_OVERLAP_PRICE_ROWS
        or _canonical_sha256(prices) != CURRENT_OVERLAP_PRICE_SHA256
        or len(dividends) != CURRENT_OVERLAP_DIVIDEND_ROWS
        or _canonical_sha256(dividends) != CURRENT_OVERLAP_DIVIDEND_SHA256
    ):
        raise ValueError("Current NTCO 43-price/2-dividend raw inventory changed.")
    return prices, dividends


def _provider_price_frame(
    rows: Sequence[Mapping[str, Any]], artifact: SourceArtifact
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        session = _date(row.get("date"))
        if not session or session < TICKER_CHANGE_DATE or session > PROVIDER_END:
            raise ValueError("NTCOY EOD row is outside the exact bounded request.")
        values = {
            field: _decimal(row.get(field))
            for field in ("open", "high", "low", "close", "volume")
        }
        if any(value is None for value in values.values()):
            raise ValueError(f"NTCOY EOD OHLCV is invalid on {session}.")
        open_ = values["open"]
        high = values["high"]
        low = values["low"]
        close = values["close"]
        volume = values["volume"]
        assert None not in (open_, high, low, close, volume)
        if (
            min(open_, high, low, close) <= 0
            or high < max(open_, low, close)
            or low > min(open_, high, close)
            or volume < 0
        ):
            raise ValueError(f"NTCOY EOD OHLCV envelope is invalid on {session}.")
        records.append(
            {
                "security_id": SECURITY_ID,
                "session": session,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
                "currency": "USD",
                "source": "eodhd_eod",
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty or frame["session"].duplicated().any():
        raise ValueError("NTCOY EOD sessions are empty or duplicated.")
    if frame["session"].min() != TICKER_CHANGE_DATE:
        raise ValueError("NTCOY first EOD session is not 2024-02-12.")
    if frame["session"].max() >= TERMINAL_EFFECTIVE_DATE:
        raise ValueError("NTCOY EOD contains a price on/after cash termination.")
    return frame


def _provider_event_id(action_type: str, effective_date: str) -> str:
    return hashlib.sha256(
        f"eodhd_{action_type}|{SECURITY_ID}|{effective_date}".encode("utf-8")
    ).hexdigest()


def _provider_action_frame(
    dividend_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    dividend_artifact: SourceArtifact,
) -> pd.DataFrame:
    if split_rows:
        raise ValueError("NTCOY split payload must be exactly empty.")
    records: list[dict[str, Any]] = []
    for row in dividend_rows:
        effective = _date(row.get("date"))
        amount = _decimal(
            row.get("unadjustedValue", row.get("value", row.get("cash_amount")))
        )
        if (
            not effective
            or effective < TICKER_CHANGE_DATE
            or effective > PROVIDER_END
            or amount is None
            or amount <= 0
        ):
            raise ValueError("NTCOY dividend row is outside range or invalid.")
        records.append(
            {
                "event_id": _provider_event_id("cash_dividend", effective),
                "security_id": SECURITY_ID,
                "action_type": "cash_dividend",
                "effective_date": effective,
                "ex_date": effective,
                "announcement_date": _date(row.get("declarationDate")),
                "record_date": _date(row.get("recordDate")),
                "payment_date": _date(row.get("paymentDate")),
                "cash_amount": float(amount),
                "ratio": None,
                "currency": _text(row.get("currency")) or "USD",
                "new_security_id": "",
                "new_symbol": "",
                "official": False,
                "source_url": dividend_artifact.source_url,
                "source_kind": "provider",
                "source": "eodhd_div",
                "retrieved_at": dividend_artifact.retrieved_at,
                "source_hash": dividend_artifact.source_hash,
                "metadata": None,
            }
        )
    columns = tuple(
        dict.fromkeys((*dataset_spec("corporate_actions").required_columns, "metadata"))
    )
    frame = pd.DataFrame(records, columns=columns)
    if not frame.empty and frame["event_id"].duplicated().any():
        raise ValueError("NTCOY provider dividends contain duplicate effective dates.")
    return frame


def bundle_from_artifacts(
    artifacts: Sequence[SourceArtifact],
    *,
    current_prices: Sequence[Mapping[str, Any]],
    current_dividends: Sequence[Mapping[str, Any]],
    budget_receipt: Mapping[str, Any],
    base_release_version: str,
    supplemental_artifacts: Sequence[SourceArtifact] = (),
    release_scope_audit: Mapping[str, Any] | None = None,
) -> ReviewedBundle:
    expected_urls = (*OFFICIAL_URLS.values(), *EODHD_REQUEST_URLS.values())
    if tuple(item.source_url for item in artifacts) != expected_urls:
        raise ValueError("Reviewed NTCOY bundle URL/order changed.")
    _validate_budget_receipt(budget_receipt, complete=True)
    _validate_official_semantics(artifacts[:3])
    provider_rows: dict[str, list[Mapping[str, Any]]] = {}
    for endpoint, artifact in zip(EODHD_ENDPOINTS, artifacts[3:], strict=True):
        try:
            value = json.loads(artifact.content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"NTCOY {endpoint} raw is invalid JSON.") from exc
        if not isinstance(value, list) or not all(
            isinstance(item, Mapping) for item in value
        ):
            raise ValueError(f"NTCOY {endpoint} raw must be a row list.")
        provider_rows[endpoint] = value
    supplemental = tuple(supplemental_artifacts)
    official_last_trade_session = ""
    if supplemental:
        if tuple(item.source_url for item in supplemental) != tuple(
            SUPPLEMENTAL_OFFICIAL_URLS.values()
        ):
            raise ValueError("Reviewed supplemental official raw URL/order changed.")
        for source_key, artifact in zip(
            SUPPLEMENTAL_OFFICIAL_URLS, supplemental, strict=True
        ):
            if artifact.source_hash != SUPPLEMENTAL_OFFICIAL_RAW_SHA256[source_key]:
                raise ValueError(
                    f"Reviewed supplemental official raw hash changed: {source_key}."
                )
            claims = _validate_supplemental_official_semantics(
                artifact, source_key=source_key
            )
            if source_key == "bny_termination":
                official_last_trade_session = _date(claims["termination_date"])
    overlap = assess_provider_overlap(
        current_prices=current_prices,
        current_dividends=current_dividends,
        ntcoy_prices=provider_rows["eod"],
        ntcoy_dividends=provider_rows["div"],
        ntcoy_splits=provider_rows["splits"],
        provider_raw_sha256={
            endpoint: artifact.source_hash
            for endpoint, artifact in zip(
                EODHD_ENDPOINTS,
                artifacts[3:],
                strict=True,
            )
        },
        official_last_trade_session=official_last_trade_session,
    )
    accepted_statuses = {
        "ready_for_transaction_design_review",
        "ready_for_price_identity_terminal_review",
    }
    if overlap["status"] not in accepted_statuses:
        raise ValueError(
            "NTCO/NTCOY exact overlap validation failed: "
            + ", ".join(overlap["blockers"])
        )
    price_only = overlap["decision_mode"] == PRICE_IDENTITY_TERMINAL_ONLY
    if price_only:
        if len(supplemental) != len(SUPPLEMENTAL_OFFICIAL_URLS):
            raise ValueError(
                "Price-only NTCOY approval requires both exact supplemental BNY raws."
            )
        if release_scope_audit is None:
            raise ValueError(
                "Price-only NTCOY approval requires release index-absence evidence."
            )
        _validate_release_index_absence_audit(release_scope_audit)
        overlap["release_scope_audit"] = dict(release_scope_audit)
    prices = _provider_price_frame(provider_rows["eod"], artifacts[3])
    actions = _provider_action_frame(
        [] if price_only else provider_rows["div"],
        provider_rows["splits"],
        artifacts[4],
    )
    return ReviewedBundle(
        artifacts=tuple(artifacts),
        supplemental_artifacts=supplemental,
        prices=prices,
        provider_actions=actions,
        provider_last_session=_text(prices["session"].max()),
        budget_receipt=dict(budget_receipt),
        base_release_version=base_release_version,
        overlap_report=dict(overlap),
    )


def _reviewed_cache_path(cache_root: Path) -> Path:
    digest = sha256_bytes(_canonical_json(_stage_signature()))
    return cache_root / REVIEWED_DIR / f"{digest}.json.gz"


def _write_reviewed_bundle(
    cache_root: Path,
    bundle: ReviewedBundle,
    pins: Mapping[str, str],
) -> Path:
    payload = {
        "signature": _stage_signature(),
        "base_release_version": bundle.base_release_version,
        "budget_receipt": dict(bundle.budget_receipt),
        "reviewed_pins": dict(pins),
        "reviewed_supplemental_pins": {
            item.source_url: item.source_hash
            for item in bundle.supplemental_artifacts
        },
        "overlap_report": dict(bundle.overlap_report),
        "artifacts": _artifact_rows(bundle.artifacts),
        "supplemental_artifacts": _artifact_rows(bundle.supplemental_artifacts),
    }
    envelope = {
        "schema": "us_ntco_ntcoy_reviewed_bundle/v1",
        "payload": payload,
        "payload_sha256": sha256_bytes(_canonical_json(payload)),
    }
    content = _canonical_json(envelope)
    encoded = gzip.compress(content, mtime=0)
    path = _reviewed_cache_path(cache_root)
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Immutable NTCOY reviewed bundle conflict: {path}")
    else:
        write_atomic(path, encoded)
    return path


def _read_reviewed_bundle(
    cache_root: Path,
    *,
    pins_path: Path,
    repository: LocalDatasetRepository,
) -> ReviewedBundle | None:
    path = _reviewed_cache_path(cache_root)
    if not path.is_file():
        return None
    try:
        content = gzip.decompress(path.read_bytes())
        envelope = json.loads(content)
    except Exception as exc:
        raise ValueError(f"NTCOY reviewed bundle is unreadable: {path}") from exc
    if content != _canonical_json(envelope):
        raise ValueError("NTCOY reviewed bundle is not canonical JSON.")
    if set(envelope) != {"schema", "payload", "payload_sha256"} or envelope.get(
        "schema"
    ) != "us_ntco_ntcoy_reviewed_bundle/v1":
        raise ValueError("NTCOY reviewed bundle wrapper changed.")
    payload = envelope["payload"]
    if envelope.get("payload_sha256") != sha256_bytes(_canonical_json(payload)):
        raise ValueError("NTCOY reviewed bundle payload hash changed.")
    if payload.get("signature") != _stage_signature():
        raise ValueError("NTCOY reviewed acquisition signature changed.")
    document = validate_pin_contract(pins_path)
    pins = _pin_map(document)
    if payload.get("reviewed_pins") != pins:
        raise ValueError("NTCOY reviewed bundle pins changed.")
    artifacts: list[SourceArtifact] = []
    for row in payload.get("artifacts", []):
        raw = base64.b64decode(row["content_base64"], validate=True)
        if sha256_bytes(raw) != row.get("content_sha256"):
            raise ValueError("NTCOY reviewed raw body hash changed.")
        artifacts.append(
            SourceArtifact(
                source=_text(row["source"]),
                source_url=_text(row["source_url"]),
                retrieved_at=_text(row["retrieved_at"]),
                content=raw,
                content_type=_text(row["content_type"]),
            )
        )
    _validate_artifact_pins(artifacts, document)
    supplemental_artifacts: list[SourceArtifact] = []
    for row in payload.get("supplemental_artifacts", []):
        raw = base64.b64decode(row["content_base64"], validate=True)
        if sha256_bytes(raw) != row.get("content_sha256"):
            raise ValueError("NTCOY supplemental reviewed raw body hash changed.")
        supplemental_artifacts.append(
            SourceArtifact(
                source=_text(row["source"]),
                source_url=_text(row["source_url"]),
                retrieved_at=_text(row["retrieved_at"]),
                content=raw,
                content_type=_text(row["content_type"]),
            )
        )
    expected_supplemental_pins = {
        SUPPLEMENTAL_OFFICIAL_URLS[key]: _text(
            document["supplemental_official_sources"][key]["source_sha256"]
        ).lower()
        for key in SUPPLEMENTAL_OFFICIAL_URLS
    }
    observed_supplemental_pins = {
        item.source_url: item.source_hash for item in supplemental_artifacts
    }
    if (
        payload.get("reviewed_supplemental_pins")
        != expected_supplemental_pins
        or observed_supplemental_pins != expected_supplemental_pins
    ):
        raise ValueError("NTCOY reviewed supplemental raw pins changed.")
    reviewed_overlap = payload.get("overlap_report")
    if not isinstance(reviewed_overlap, Mapping):
        raise ValueError("NTCOY reviewed overlap report changed.")
    reviewed_scope = reviewed_overlap.get("release_scope_audit")
    base_release_version = _text(payload["base_release_version"])
    if reviewed_scope is not None:
        if not isinstance(reviewed_scope, Mapping):
            raise ValueError("NTCOY reviewed release-scope audit changed.")
        _validate_release_index_absence_audit(reviewed_scope)
        if _text(reviewed_scope.get("release_version")) != base_release_version:
            raise ValueError(
                "NTCOY reviewed release-scope audit is not bound to its base release."
            )
    release, _ = repository.current_release()
    if release is None:
        raise RuntimeError("Current release is required to replay NTCOY bundle.")
    if reviewed_scope is not None:
        current_scope = _release_index_absence_audit(repository, release)
        # The reviewed decision artifact must remain bound to the immutable
        # base-release audit.  A successful apply creates a new release id but
        # deliberately retains the two out-of-scope index dataset versions.
        # Compare that lineage while excluding only the derived release id so
        # post-commit replay cannot change its own expected artifact hash.
        reviewed_lineage = {
            key: value
            for key, value in reviewed_scope.items()
            if key != "release_version"
        }
        current_lineage = {
            key: value
            for key, value in current_scope.items()
            if key != "release_version"
        }
        if _canonical_json(reviewed_lineage) != _canonical_json(current_lineage):
            raise ValueError(
                "NTCOY current index-scope lineage differs from the reviewed base."
            )
    current_prices, current_dividends = current_overlap_records(repository, release)
    bundle = bundle_from_artifacts(
        artifacts,
        current_prices=current_prices,
        current_dividends=current_dividends,
        budget_receipt=payload["budget_receipt"],
        base_release_version=base_release_version,
        supplemental_artifacts=supplemental_artifacts,
        release_scope_audit=reviewed_scope,
    )
    if _canonical_json(bundle.overlap_report) != _canonical_json(reviewed_overlap):
        raise ValueError("NTCOY reviewed overlap decision no longer replays exactly.")
    return bundle


def promote_quarantine(
    cache_root: Path,
    quarantine_id: str,
    *,
    pins_path: Path = DEFAULT_PINS,
    repository: LocalDatasetRepository | None = None,
    supplemental_evidence_dir: Path = DEFAULT_SUPPLEMENTAL_EVIDENCE_DIR,
    supplemental_books_closed_evidence_dir: Path = (
        DEFAULT_SUPPLEMENTAL_BOOKS_CLOSED_EVIDENCE_DIR
    ),
) -> dict[str, Any]:
    repository = repository or LocalDatasetRepository(cache_root)
    quarantine = read_quarantine(cache_root, quarantine_id)
    document = validate_pin_contract(pins_path)
    pins = _validate_artifact_pins(quarantine.artifacts, document)
    supplemental_artifacts = _load_supplemental_official_artifacts(
        document,
        supplemental_evidence_dir=supplemental_evidence_dir,
        supplemental_books_closed_evidence_dir=(
            supplemental_books_closed_evidence_dir
        ),
    )
    release, _ = repository.current_release()
    if release is None:
        raise RuntimeError("Current release is required for NTCOY promotion.")
    current_prices, current_dividends = current_overlap_records(repository, release)
    bundle = bundle_from_artifacts(
        quarantine.artifacts,
        current_prices=current_prices,
        current_dividends=current_dividends,
        budget_receipt=quarantine.budget_receipt,
        base_release_version=release.version,
        supplemental_artifacts=supplemental_artifacts,
        release_scope_audit=_release_index_absence_audit(repository, release),
    )
    path = _write_reviewed_bundle(cache_root, bundle, pins)
    return {
        "status": "reviewed_bundle_promoted",
        "network_accessed": False,
        "quarantine_id": quarantine.quarantine_id,
        "reviewed_bundle_path": str(path),
        "provider_last_session": bundle.provider_last_session,
        "price_rows": len(bundle.prices),
        "provider_action_rows": len(bundle.provider_actions),
        "decision_mode": bundle.overlap_report["decision_mode"],
        "rejected_provider_dividend_rows": (
            len(bundle.overlap_report["dividend_ambiguity"]["records"])
        ),
        "preserved_dividend_event_ids": sorted(CURRENT_DIVIDEND_EVENT_IDS),
        "maximum_dividend_sensitivity_usd_per_ads": str(
            MAX_DIVIDEND_SENSITIVITY_USD_PER_ADS
        ),
        "release_scope_audit": bundle.overlap_report.get("release_scope_audit"),
        "artifact_sha256": {
            item.source_url: item.source_hash
            for item in (*bundle.artifacts, *bundle.supplemental_artifacts)
        },
    }


def _reviewed_extraction_artifacts(bundle: ReviewedBundle) -> tuple[SourceArtifact, ...]:
    raw = {item.source_url: item for item in bundle.artifacts}
    all_raw = (*bundle.artifacts, *bundle.supplemental_artifacts)
    retrieved_at = max(item.retrieved_at for item in all_raw)
    price_only = (
        bundle.overlap_report.get("decision_mode")
        == PRICE_IDENTITY_TERMINAL_ONLY
    )
    identity = SourceArtifact(
        source="official_ntco_ntcoy_identity",
        source_url=OFFICIAL_URLS["occ"],
        retrieved_at=retrieved_at,
        content=_canonical_json(
            {
                "schema": "official_ntco_ntcoy_identity/v1",
                "security_id": SECURITY_ID,
                "effective_date": TICKER_CHANGE_DATE,
                "old_symbol": OLD_SYMBOL,
                "new_symbol": NEW_SYMBOL,
                "canonical_exchange": CANONICAL_EXCHANGE,
                "official_destination_market": OFFICIAL_DESTINATION_MARKET,
                "cusip": CUSIP,
                "deliverable": ADS_DELIVERABLE,
                "cboe_raw_sha256": raw[OFFICIAL_URLS["cboe"]].source_hash,
                "occ_raw_sha256": raw[OFFICIAL_URLS["occ"]].source_hash,
            }
        ),
        content_type="application/json",
    )
    terminal = SourceArtifact(
        source="official_ntcoy_cash_termination",
        source_url=OFFICIAL_URLS["bny"],
        retrieved_at=retrieved_at,
        content=_canonical_json(
            {
                "schema": "official_ntcoy_cash_termination/v1",
                "security_id": SECURITY_ID,
                "action_type": "delisting",
                "effective_date": TERMINAL_EFFECTIVE_DATE,
                "cash_amount": str(TERMINAL_CASH_AMOUNT),
                "currency": TERMINAL_CURRENCY,
                "ads_to_underlying_ratio": "1:2",
                "fee_per_ads": "0",
                "bny_raw_sha256": raw[OFFICIAL_URLS["bny"]].source_hash,
            }
        ),
        content_type="application/json",
    )
    decision_audit = SourceArtifact(
        source="reviewed_ntco_ntcoy_transition_decision",
        source_url=EODHD_REQUEST_URLS["div"],
        retrieved_at=retrieved_at,
        content=_canonical_json(
            {
                "schema": "reviewed_ntco_ntcoy_transition_decision/v1",
                "security_id": SECURITY_ID,
                "decision_mode": bundle.overlap_report.get("decision_mode"),
                "provider_price_raw_sha256": raw[
                    EODHD_REQUEST_URLS["eod"]
                ].source_hash,
                "provider_splits_raw_sha256": raw[
                    EODHD_REQUEST_URLS["splits"]
                ].source_hash,
                "provider_dividend_economics_accepted": not price_only,
                "provider_dividend_raw_decision": (
                    REJECTED_DIVIDEND_CONFLICT_POLICY
                    if price_only
                    else "accepted_exact_full_bundle"
                ),
                "provider_dividend_raw_sha256": raw[
                    EODHD_REQUEST_URLS["div"]
                ].source_hash,
                "preserved_dividend_actions": (
                    PRESERVED_DIVIDEND_ACTIONS if price_only else {}
                ),
                "maximum_absolute_sensitivity_usd_per_ads": str(
                    MAX_DIVIDEND_SENSITIVITY_USD_PER_ADS
                ),
                "difference_inventory": bundle.overlap_report.get(
                    "difference_inventory", {}
                ),
                "release_scope_audit": bundle.overlap_report.get(
                    "release_scope_audit", {}
                ),
                "supplemental_official_raw_sha256": {
                    item.source_url: item.source_hash
                    for item in bundle.supplemental_artifacts
                },
            }
        ),
        content_type="application/json",
    )
    return identity, terminal, decision_audit


def _official_actions(bundle: ReviewedBundle) -> pd.DataFrame:
    identity, terminal, _decision_audit = _reviewed_extraction_artifacts(bundle)
    rows = [
        {
            "event_id": TICKER_CHANGE_EVENT_ID,
            "security_id": SECURITY_ID,
            "action_type": "ticker_change",
            "effective_date": TICKER_CHANGE_DATE,
            "ex_date": TICKER_CHANGE_DATE,
            "announcement_date": "2024-02-09",
            "record_date": "",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "new_security_id": SECURITY_ID,
            "new_symbol": NEW_SYMBOL,
            "official": True,
            "source_url": identity.source_url,
            "source_kind": "clearing_and_exchange_notices",
            "source": identity.source,
            "retrieved_at": identity.retrieved_at,
            "source_hash": identity.source_hash,
            "metadata": json.dumps(
                {
                    "cboe_source_url": OFFICIAL_URLS["cboe"],
                    "occ_source_url": OFFICIAL_URLS["occ"],
                    "official_destination_market": OFFICIAL_DESTINATION_MARKET,
                    "canonical_exchange": CANONICAL_EXCHANGE,
                    "cusip": CUSIP,
                    "deliverable": ADS_DELIVERABLE,
                },
                sort_keys=True,
            ),
        },
        {
            "event_id": TERMINAL_EVENT_ID,
            "security_id": SECURITY_ID,
            "action_type": "delisting",
            "effective_date": TERMINAL_EFFECTIVE_DATE,
            "ex_date": TERMINAL_EFFECTIVE_DATE,
            "announcement_date": "2024-08-26",
            "record_date": "",
            "payment_date": TERMINAL_EFFECTIVE_DATE,
            "cash_amount": float(TERMINAL_CASH_AMOUNT),
            "ratio": None,
            "currency": TERMINAL_CURRENCY,
            "new_security_id": "",
            "new_symbol": "",
            "official": True,
            "source_url": terminal.source_url,
            "source_kind": "depositary_corporate_action_notice",
            "source": terminal.source,
            "retrieved_at": terminal.retrieved_at,
            "source_hash": terminal.source_hash,
            "metadata": json.dumps(
                {
                    "mandatory_exchange": True,
                    "gross_rate_per_ads": str(TERMINAL_CASH_AMOUNT),
                    "cancellation_fee_per_ads": "0",
                    "net_rate_per_ads": str(TERMINAL_CASH_AMOUNT),
                    "ads_to_underlying_ratio": "1:2",
                },
                sort_keys=True,
            ),
        },
    ]
    columns = tuple(
        dict.fromkeys((*dataset_spec("corporate_actions").required_columns, "metadata"))
    )
    return pd.DataFrame(rows, columns=columns)


def _archive_id(artifact: SourceArtifact) -> str:
    # Publication uses the payload digest as the canonical archive id.  The
    # exact NTCOY empty-splits response shares the universal ``[]`` digest with
    # unrelated endpoints, so that one provenance tuple uses the registry-
    # approved composite id instead of collapsing another source row.
    if (
        artifact.source == "eodhd_splits"
        and artifact.source_url == EODHD_REQUEST_URLS["splits"]
        and artifact.source_hash == OBSERVED_PROVIDER_RAW_SHA256["splits"]
    ):
        return "6a09ccaafcdf8ad57177fd1be2146ce912c84c4269cdc11ce736c7b4faad4461"
    return artifact.source_hash


def _archive_row(artifact: SourceArtifact, completed_session: str) -> dict[str, Any]:
    suffix = "json" if artifact.content_type == "application/json" else "bin"
    return {
        "archive_id": _archive_id(artifact),
        "dataset": artifact.source,
        "object_path": f"archives/{completed_session}/{artifact.source_hash}.{suffix}.gz",
        "content_type": artifact.content_type,
        "effective_date": completed_session,
        "source": artifact.source,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
        "source_url": artifact.source_url,
    }


def _identity_artifact(bundle: ReviewedBundle) -> SourceArtifact:
    return _reviewed_extraction_artifacts(bundle)[0]


def _rewrite_master(master: pd.DataFrame, bundle: ReviewedBundle) -> pd.DataFrame:
    output = master.copy()
    mask = output["security_id"].astype(str).eq(SECURITY_ID)
    if int(mask.sum()) != 1:
        raise ValueError("NTCO security_master identity is missing or ambiguous.")
    evidence = _identity_artifact(bundle)
    output.loc[mask, "primary_symbol"] = NEW_SYMBOL
    if "provider_symbol" in output:
        output.loc[mask, "provider_symbol"] = PROVIDER_SYMBOL
    if "action_provider_symbol" in output:
        output.loc[mask, "action_provider_symbol"] = PROVIDER_SYMBOL
    output.loc[mask, "exchange"] = CANONICAL_EXCHANGE
    output.loc[mask, "active_to"] = TERMINAL_EFFECTIVE_DATE
    output.loc[mask, "source"] = evidence.source
    output.loc[mask, "retrieved_at"] = evidence.retrieved_at
    output.loc[mask, "source_hash"] = evidence.source_hash
    if "source_url" in output:
        output.loc[mask, "source_url"] = evidence.source_url
    return output


def _rewrite_history(history: pd.DataFrame, bundle: ReviewedBundle) -> pd.DataFrame:
    own = history.loc[history["security_id"].astype(str).eq(SECURITY_ID)].copy()
    old = own.loc[own["symbol"].astype(str).str.upper().eq(OLD_SYMBOL)]
    if old.empty:
        raise ValueError("NTCO symbol_history lacks the predecessor row.")
    effective_from = min(_date(value) for value in old["effective_from"])
    if not effective_from:
        raise ValueError("NTCO predecessor symbol_history start is invalid.")
    evidence = _identity_artifact(bundle)
    rows = [
        {
            "security_id": SECURITY_ID,
            "symbol": OLD_SYMBOL,
            "exchange": "NYSE",
            "effective_from": effective_from,
            "effective_to": "2024-02-09",
            "source": evidence.source,
            "retrieved_at": evidence.retrieved_at,
            "source_hash": evidence.source_hash,
            "source_url": evidence.source_url,
        },
        {
            "security_id": SECURITY_ID,
            "symbol": NEW_SYMBOL,
            "exchange": CANONICAL_EXCHANGE,
            "effective_from": TICKER_CHANGE_DATE,
            # Symbol history is a tradable-alias interval. The ADS remains an
            # economic cash claim until 2024-09-04 (security_master.active_to),
            # but the exact BNY facility termination ends tradability 8/07.
            "effective_to": OBSERVED_PROVIDER_LAST_TRADE_SESSION,
            "source": evidence.source,
            "retrieved_at": evidence.retrieved_at,
            "source_hash": evidence.source_hash,
            "source_url": evidence.source_url,
        },
    ]
    output = history.loc[~history["security_id"].astype(str).eq(SECURITY_ID)].copy()
    replacement = pd.DataFrame(rows)
    for column in output.columns:
        if column not in replacement:
            replacement[column] = None
    return pd.concat(
        [output, replacement.loc[:, output.columns]], ignore_index=True, sort=False
    )


def _rewrite_prices(prices: pd.DataFrame, bundle: ReviewedBundle) -> pd.DataFrame:
    sessions = pd.to_datetime(prices["session"], errors="coerce").dt.date.astype(str)
    remove = prices["security_id"].astype(str).eq(SECURITY_ID) & sessions.ge(
        TICKER_CHANGE_DATE
    )
    output = prices.loc[~remove].copy()
    provider = bundle.prices.copy()
    for column in output.columns:
        if column not in provider:
            provider[column] = None
    output = pd.concat(
        [output, provider.loc[:, output.columns]], ignore_index=True, sort=False
    )
    if output.duplicated(["security_id", "session"]).any():
        raise ValueError("NTCOY price rewrite duplicates a security/session key.")
    return output


def _rewrite_actions(actions: pd.DataFrame, bundle: ReviewedBundle) -> pd.DataFrame:
    effective = pd.to_datetime(actions["effective_date"], errors="coerce").dt.date.astype(
        str
    )
    target_tail = actions["security_id"].astype(str).eq(SECURITY_ID) & effective.ge(
        TICKER_CHANGE_DATE
    )
    price_only = (
        bundle.overlap_report.get("decision_mode")
        == PRICE_IDENTITY_TERMINAL_ONLY
    )
    if price_only:
        target_ids = set(actions.loc[target_tail, "event_id"].astype(str))
        if target_ids != set(CURRENT_DIVIDEND_EVENT_IDS):
            raise ValueError(
                "Price-only NTCOY rewrite requires exactly the two preserved "
                "NTCO dividend actions."
            )
        _validate_preserved_dividend_actions(actions.loc[target_tail])
        output = actions.copy()
    else:
        output = actions.loc[~target_tail].copy()
    additions = pd.concat(
        [bundle.provider_actions, _official_actions(bundle)],
        ignore_index=True,
        sort=False,
    )
    for column in output.columns:
        if column not in additions:
            additions[column] = None
    output = pd.concat(
        [output, additions.loc[:, output.columns]], ignore_index=True, sort=False
    )
    if output["event_id"].astype(str).duplicated().any():
        raise ValueError("NTCOY action rewrite duplicates event_id.")
    return output


def _rewrite_resolutions(
    resolutions: pd.DataFrame, bundle: ReviewedBundle
) -> pd.DataFrame:
    terminal = _reviewed_extraction_artifacts(bundle)[1]
    output = resolutions.loc[
        ~resolutions["security_id"].astype(str).eq(SECURITY_ID)
    ].copy()
    row = {column: None for column in output.columns}
    values = {
        "candidate_id": lifecycle_candidate_id(
            SECURITY_ID, bundle.provider_last_session
        ),
        "security_id": SECURITY_ID,
        "symbol": NEW_SYMBOL,
        "last_price_date": bundle.provider_last_session,
        "resolution": "applied",
        "event_id": TERMINAL_EVENT_ID,
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": REVIEWED_BY,
        "reviewed_at": REVIEWED_AT,
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        "source_url": terminal.source_url,
        "source": terminal.source,
        "retrieved_at": terminal.retrieved_at,
        "source_hash": terminal.source_hash,
    }
    missing = set(values) - set(row)
    if missing:
        raise ValueError("lifecycle_resolutions schema lacks: " + ", ".join(sorted(missing)))
    row.update(values)
    output = pd.concat(
        [output, pd.DataFrame([row]).loc[:, output.columns]],
        ignore_index=True,
        sort=False,
    )
    if output["candidate_id"].astype(str).duplicated().any():
        raise ValueError("NTCOY lifecycle resolution duplicates candidate_id.")
    return output


def _rewrite_archive(
    archive: pd.DataFrame,
    artifacts: Sequence[SourceArtifact],
    completed_session: str,
) -> pd.DataFrame:
    ids = {_archive_id(item) for item in artifacts}
    output = archive.loc[~archive["archive_id"].astype(str).isin(ids)].copy()
    additions = pd.DataFrame(
        [_archive_row(item, completed_session) for item in artifacts]
    )
    for column in output.columns:
        if column not in additions:
            additions[column] = None
    output = pd.concat(
        [output, additions.loc[:, output.columns]], ignore_index=True, sort=False
    )
    if output["archive_id"].astype(str).duplicated().any():
        raise ValueError("NTCOY source_archive duplicates archive_id.")
    return output


def _new_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"ntco-ntcoy-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


def _target_prices_match(frame: pd.DataFrame, bundle: ReviewedBundle) -> bool:
    own = frame.loc[frame["security_id"].astype(str).eq(SECURITY_ID)].copy()
    own["session"] = pd.to_datetime(own["session"], errors="coerce").dt.date.astype(str)
    own = own.loc[own["session"].ge(TICKER_CHANGE_DATE)]
    observed, problems = _price_map(own.to_dict("records"))
    expected, expected_problems = _price_map(bundle.prices.to_dict("records"))
    if problems or expected_problems or observed != expected:
        return False

    def lineage_rows(value: pd.DataFrame) -> list[dict[str, str]]:
        return sorted(
            (
                {
                    "security_id": _text(row.get("security_id")),
                    "session": _date(row.get("session", row.get("date"))),
                    "currency": _text(row.get("currency")),
                    "source": _text(row.get("source")),
                    "source_url": _text(row.get("source_url")),
                    "retrieved_at": _text(row.get("retrieved_at")),
                    "source_hash": _text(row.get("source_hash")),
                }
                for row in value.to_dict("records")
            ),
            key=lambda row: row["session"],
        )

    return lineage_rows(own) == lineage_rows(bundle.prices)


def _normalized_action_inventory(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        cash = _decimal(row.get("cash_amount"))
        ratio = _decimal(row.get("ratio"))
        records.append(
            {
                "event_id": _text(row.get("event_id")),
                "security_id": _text(row.get("security_id")),
                "action_type": _text(row.get("action_type")),
                "effective_date": _date(row.get("effective_date")),
                "ex_date": _date(row.get("ex_date")),
                "announcement_date": _date(row.get("announcement_date")),
                "record_date": _date(row.get("record_date")),
                "payment_date": _date(row.get("payment_date")),
                "cash_amount": "" if cash is None else format(cash, "f"),
                "ratio": "" if ratio is None else format(ratio, "f"),
                "currency": _text(row.get("currency")),
                "new_security_id": _text(row.get("new_security_id")),
                "new_symbol": _text(row.get("new_symbol")),
                "official": _text(row.get("official")).lower() in {"1", "true"},
                "source_url": _text(row.get("source_url")),
                "source_kind": _text(row.get("source_kind")),
                "source": _text(row.get("source")),
                "retrieved_at": _text(row.get("retrieved_at")),
                "source_hash": _text(row.get("source_hash")),
                "metadata": _text(row.get("metadata")),
            }
        )
    return sorted(records, key=lambda row: row["event_id"])


def _validate_preserved_dividend_actions(frame: pd.DataFrame) -> None:
    expected: list[dict[str, Any]] = []
    for event_id, row in PRESERVED_DIVIDEND_ACTIONS.items():
        effective = row["effective_date"]
        expected.append(
            {
                "event_id": event_id,
                "security_id": SECURITY_ID,
                "action_type": "cash_dividend",
                "effective_date": effective,
                "ex_date": effective,
                "announcement_date": "",
                "record_date": "",
                "payment_date": "",
                "cash_amount": row["cash_amount"],
                "ratio": "",
                "currency": "USD",
                "new_security_id": "",
                "new_symbol": "",
                "official": False,
                "source_url": CURRENT_DIVIDEND_SOURCE_URL,
                "source_kind": "provider",
                "source": "eodhd_div",
                "retrieved_at": CURRENT_DIVIDEND_RETRIEVED_AT,
                "source_hash": CURRENT_DIVIDEND_RAW_SHA256,
                "metadata": "",
            }
        )
    expected.sort(key=lambda row: row["event_id"])
    if _normalized_action_inventory(frame) != expected:
        raise ValueError(
            "The two preserved NTCO dividend actions or their lineage changed."
        )


def _target_factors_match(frames: Mapping[str, pd.DataFrame]) -> bool:
    prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"].astype(str).eq(SECURITY_ID)
    ]
    actions = frames["corporate_actions"].loc[
        frames["corporate_actions"]["security_id"].astype(str).eq(SECURITY_ID)
    ]
    factors = frames["adjustment_factors"].loc[
        frames["adjustment_factors"]["security_id"].astype(str).eq(SECURITY_ID)
    ].copy()
    if prices.empty or len(factors) != len(prices):
        return False
    lineages = set(factors["source_version"].map(_text))
    if len(lineages) != 1:
        return False
    lineage = next(iter(lineages))
    match = re.fullmatch(
        r"ntco-ntcoy-(\d{8})-([0-9a-f]{32})-daily_price_raw\+"
        r"ntco-ntcoy-\1-\2-corporate_actions",
        lineage,
    )
    if match is None:
        return False
    if (
        set(factors["source_hash"].map(_text)) != {lineage}
        or set(factors["source"].map(_text)) != {"derived"}
        or set(factors["calculated_at"].map(_text)) != {REVIEWED_AT}
        or set(factors["retrieved_at"].map(_text)) != {REVIEWED_AT}
    ):
        return False
    expected = build_adjustment_factors(
        prices,
        actions,
        source_version=lineage,
    )

    def economics(value: pd.DataFrame) -> dict[str, tuple[Decimal | None, Decimal | None]]:
        output: dict[str, tuple[Decimal | None, Decimal | None]] = {}
        for row in value.to_dict("records"):
            session = _date(row.get("session"))
            if not session or session in output:
                return {}
            output[session] = (
                _decimal(row.get("split_factor")),
                _decimal(row.get("total_return_factor")),
            )
        return output

    return economics(factors) == economics(expected)


def _target_archives_match(
    archive: pd.DataFrame,
    artifacts: Sequence[SourceArtifact],
    repository: LocalDatasetRepository | None,
) -> bool:
    effective_dates: set[str] = set()
    for artifact in artifacts:
        rows = archive.loc[
            archive["archive_id"].astype(str).eq(_archive_id(artifact))
        ]
        if len(rows) != 1:
            return False
        row = rows.iloc[0]
        suffix = "json" if artifact.content_type == "application/json" else "bin"
        effective_date = _date(row.get("effective_date"))
        expected_path = (
            f"archives/{effective_date}/{artifact.source_hash}.{suffix}.gz"
        )
        if (
            not effective_date
            or _text(row.get("dataset")) != artifact.source
            or _text(row.get("source")) != artifact.source
            or _text(row.get("source_url")) != artifact.source_url
            or _text(row.get("source_hash")) != artifact.source_hash
            or _text(row.get("content_type")) != artifact.content_type
            or _text(row.get("retrieved_at")) != artifact.retrieved_at
            or _text(row.get("object_path")) != expected_path
        ):
            return False
        effective_dates.add(effective_date)
        if repository is not None:
            try:
                path = _safe_repository_path(
                    repository.root,
                    _text(row.get("object_path")),
                )
                if gzip.decompress(path.read_bytes()) != artifact.content:
                    return False
            except Exception:
                return False
    return len(effective_dates) == 1


def _is_repaired(
    frames: Mapping[str, pd.DataFrame],
    bundle: ReviewedBundle,
    repository: LocalDatasetRepository | None = None,
) -> bool:
    master = frames["security_master"].loc[
        frames["security_master"]["security_id"].astype(str).eq(SECURITY_ID)
    ]
    if len(master) != 1:
        return False
    row = master.iloc[0]
    identity = _identity_artifact(bundle)
    if not (
        _text(row.get("primary_symbol")) == NEW_SYMBOL
        and _text(row.get("provider_symbol")) == PROVIDER_SYMBOL
        and _text(row.get("action_provider_symbol")) == PROVIDER_SYMBOL
        and _text(row.get("exchange")) == CANONICAL_EXCHANGE
        and _date(row.get("active_from")) == CURRENT_IDENTITY_ACTIVE_FROM
        and _date(row.get("active_to")) == TERMINAL_EFFECTIVE_DATE
        and _text(row.get("source")) == identity.source
        and _text(row.get("source_url")) == identity.source_url
        and _text(row.get("source_hash")) == identity.source_hash
        and _text(row.get("retrieved_at")) == identity.retrieved_at
    ):
        return False
    history = frames["symbol_history"].loc[
        frames["symbol_history"]["security_id"].astype(str).eq(SECURITY_ID)
    ]
    history_keys = {
        (
            _text(item.get("symbol")),
            _text(item.get("exchange")),
            _date(item.get("effective_from")),
            _date(item.get("effective_to")),
        )
        for item in history.to_dict("records")
    }
    if history_keys != {
        (OLD_SYMBOL, "NYSE", CURRENT_IDENTITY_ACTIVE_FROM, "2024-02-09"),
        (
            NEW_SYMBOL,
            CANONICAL_EXCHANGE,
            TICKER_CHANGE_DATE,
            OBSERVED_PROVIDER_LAST_TRADE_SESSION,
        ),
    }:
        return False
    if any(
        _text(row.get("source")) != identity.source
        or _text(row.get("source_url")) != identity.source_url
        or _text(row.get("source_hash")) != identity.source_hash
        or _text(row.get("retrieved_at")) != identity.retrieved_at
        for row in history.to_dict("records")
    ):
        return False
    if not _target_prices_match(frames["daily_price_raw"], bundle):
        return False
    actions = frames["corporate_actions"]
    own_actions = actions.loc[actions["security_id"].astype(str).eq(SECURITY_ID)]
    effective = own_actions["effective_date"].map(_date)
    tail = own_actions.loc[effective.ge(TICKER_CHANGE_DATE)]
    if bundle.overlap_report.get("decision_mode") == PRICE_IDENTITY_TERMINAL_ONLY:
        preserved = tail.loc[
            tail["event_id"].astype(str).isin(CURRENT_DIVIDEND_EVENT_IDS)
        ]
        official = tail.loc[
            tail["event_id"].astype(str).isin(
                {TICKER_CHANGE_EVENT_ID, TERMINAL_EVENT_ID}
            )
        ]
        if len(tail) != 4:
            return False
        try:
            _validate_preserved_dividend_actions(preserved)
        except ValueError:
            return False
        if _normalized_action_inventory(official) != _normalized_action_inventory(
            _official_actions(bundle)
        ):
            return False
    else:
        expected_tail = pd.concat(
            [bundle.provider_actions, _official_actions(bundle)],
            ignore_index=True,
            sort=False,
        )
        if _normalized_action_inventory(tail) != _normalized_action_inventory(
            expected_tail
        ):
            return False
    if not _target_factors_match(frames):
        return False
    resolutions = frames["lifecycle_resolutions"]
    resolution = resolutions.loc[
        resolutions["candidate_id"].astype(str).eq(
            lifecycle_candidate_id(SECURITY_ID, bundle.provider_last_session)
        )
    ]
    terminal = _reviewed_extraction_artifacts(bundle)[1]
    if len(resolution) != 1:
        return False
    resolved = resolution.iloc[0]
    if not (
        _text(resolved.get("security_id")) == SECURITY_ID
        and _text(resolved.get("symbol")) == NEW_SYMBOL
        and _date(resolved.get("last_price_date")) == bundle.provider_last_session
        and _text(resolved.get("resolution")) == "applied"
        and _text(resolved.get("event_id")) == TERMINAL_EVENT_ID
        and not _text(resolved.get("exception_code"))
        and _text(resolved.get("reviewed_by")) == REVIEWED_BY
        and _text(resolved.get("reviewed_at")) == REVIEWED_AT
        and _text(resolved.get("source_url")) == terminal.source_url
        and _text(resolved.get("source")) == terminal.source
        and _text(resolved.get("source_hash")) == terminal.source_hash
    ):
        return False
    return _target_archives_match(
        frames["source_archive"],
        (
            *bundle.artifacts,
            *bundle.supplemental_artifacts,
            *_reviewed_extraction_artifacts(bundle),
        ),
        repository,
    )


def _has_partial_markers(frames: Mapping[str, pd.DataFrame]) -> bool:
    master = frames["security_master"]
    own = master.loc[master["security_id"].astype(str).eq(SECURITY_ID)]
    master_marker = not own.empty and _text(own.iloc[0].get("primary_symbol")) == NEW_SYMBOL
    history_marker = frames["symbol_history"]["symbol"].astype(str).str.upper().eq(
        NEW_SYMBOL
    ) & frames["symbol_history"]["security_id"].astype(str).eq(SECURITY_ID)
    action_marker = frames["corporate_actions"]["event_id"].astype(str).isin(
        {TICKER_CHANGE_EVENT_ID, TERMINAL_EVENT_ID}
    )
    resolution_marker = frames["lifecycle_resolutions"]["event_id"].astype(str).eq(
        TERMINAL_EVENT_ID
    )
    archive_marker = frames["source_archive"]["source_url"].fillna("").astype(str).isin(
        {
            *OFFICIAL_URLS.values(),
            *SUPPLEMENTAL_OFFICIAL_URLS.values(),
            *EODHD_REQUEST_URLS.values(),
        }
    )
    return bool(
        master_marker
        or history_marker.any()
        or action_marker.any()
        or resolution_marker.any()
        or archive_marker.any()
    )


def _validate_old_tables(
    frames: Mapping[str, pd.DataFrame],
    current_prices: Sequence[Mapping[str, Any]],
    current_dividends: Sequence[Mapping[str, Any]],
) -> None:
    prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"].astype(str).eq(SECURITY_ID)
    ].copy()
    prices["session"] = pd.to_datetime(prices["session"], errors="coerce").dt.date.astype(str)
    prices = prices.loc[prices["session"].ge(TICKER_CHANGE_DATE)]
    observed, problems = _price_map(prices.to_dict("records"))
    expected, expected_problems = _price_map(current_prices)
    if problems or expected_problems or observed != expected:
        raise ValueError("Current NTCO table does not match the exact 43 archived raws.")
    actions = frames["corporate_actions"]
    effective = pd.to_datetime(actions["effective_date"], errors="coerce").dt.date.astype(str)
    tail = actions.loc[
        actions["security_id"].astype(str).eq(SECURITY_ID)
        & effective.ge(TICKER_CHANGE_DATE)
    ]
    if (
        len(tail) != CURRENT_OVERLAP_DIVIDEND_ROWS
        or set(tail["event_id"].astype(str)) != CURRENT_DIVIDEND_EVENT_IDS
        or not tail["action_type"].astype(str).eq("cash_dividend").all()
    ):
        raise ValueError("Current NTCO table does not contain the exact two dividends.")
    expected_dividends, _ = _dividend_map(current_dividends)
    observed_dividends = {
        _date(row.get("effective_date")): _decimal(row.get("cash_amount"))
        for row in tail.to_dict("records")
    }
    if observed_dividends != expected_dividends:
        raise ValueError("Current NTCO dividend economics differ from raw archive.")


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        frames: Mapping[str, pd.DataFrame],
    ):
        self.base = base
        self.root = base.root
        self.versions = dict(versions)
        self.frames = {key: value.copy(deep=True) for key, value in frames.items()}

    def current_release(self):  # type: ignore[no-untyped-def]
        return None, None

    def current_manifest(self, dataset: str):  # type: ignore[no-untyped-def]
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.frames:
            return self.frames[dataset].copy(deep=True)
        version = self.versions.get(dataset)
        return self.base.read_frame(dataset, version) if version else pd.DataFrame()


def _validate_candidate_frames(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    bundle: ReviewedBundle,
) -> dict[str, Any]:
    versions = dict(release.dataset_versions)
    candidate_repository = _CandidateRepository(repository, versions, frames)
    candidate_release = DataRelease(
        version=release.version,
        created_at=release.created_at,
        completed_session=release.completed_session,
        dataset_versions=versions,
        quality=release.quality,
        warnings=release.warnings,
    )
    candidates = build_lifecycle_candidates(
        candidate_repository, release=candidate_release
    )
    def as_candidate_frame(values):  # type: ignore[no-untyped-def]
        return pd.DataFrame(
            [
                {
                    "candidate_id": lifecycle_candidate_id(
                        item.security_id,
                        item.last_price_date,
                    ),
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

    candidate_frame = as_candidate_frame(candidates)
    report = validate_lifecycle_coverage(
        candidate_frame,
        frames["lifecycle_resolutions"],
        frames["corporate_actions"],
        completed_session=release.completed_session,
    )
    matches = [item for item in candidates if item.security_id == SECURITY_ID]
    if len(matches) != 1 or matches[0].last_price_date != bundle.provider_last_session:
        raise ValueError("Expanded lifecycle does not bind exact NTCOY terminal price.")

    # This repository snapshot already carries a separately tracked lifecycle
    # backlog.  The narrow repair must close NTCO's old candidate and may not
    # hide, add, or otherwise alter any unrelated coverage issue.  Comparing
    # exact candidate-id sets keeps the gate fail-closed without expanding this
    # transaction into repairs for unrelated issuers.
    base_candidates = build_lifecycle_candidates(repository, release=release)
    base_matches = [item for item in base_candidates if item.security_id == SECURITY_ID]
    if len(base_matches) != 1:
        raise ValueError("Base release does not contain one exact NTCO candidate.")
    old_candidate_id = lifecycle_candidate_id(
        base_matches[0].security_id,
        base_matches[0].last_price_date,
    )
    new_candidate_id = lifecycle_candidate_id(
        matches[0].security_id,
        matches[0].last_price_date,
    )
    base_report = validate_lifecycle_coverage(
        as_candidate_frame(base_candidates),
        existing_resolutions := repository.read_frame(
            "lifecycle_resolutions",
            release.dataset_versions["lifecycle_resolutions"],
        ),
        repository.read_frame(
            "corporate_actions",
            release.dataset_versions["corporate_actions"],
        ),
        completed_session=release.completed_session,
    )

    def issue_inventory(value):  # type: ignore[no-untyped-def]
        return {
            (issue.code, tuple(sorted(issue.candidate_ids)))
            for issue in value.issues
            if issue.candidate_ids
        }

    expected_issues: set[tuple[str, tuple[str, ...]]] = set()
    for issue in base_report.issues:
        candidate_ids = tuple(
            sorted(value for value in issue.candidate_ids if value != old_candidate_id)
        )
        if candidate_ids:
            expected_issues.add((issue.code, candidate_ids))
    if issue_inventory(report) != expected_issues:
        raise ValueError("NTCO repair changes unrelated lifecycle coverage inventory.")
    if any(new_candidate_id in issue.candidate_ids for issue in report.issues):
        raise ValueError("NTCOY terminal lifecycle candidate remains unresolved.")
    target_resolution = frames["lifecycle_resolutions"].loc[
        frames["lifecycle_resolutions"]["candidate_id"].astype(str).eq(new_candidate_id)
    ]
    if (
        len(target_resolution) != 1
        or _text(target_resolution.iloc[0].get("resolution")) != "applied"
        or _text(target_resolution.iloc[0].get("event_id")) != TERMINAL_EVENT_ID
    ):
        raise ValueError("NTCOY terminal candidate resolution is not exact.")
    base = validate_repository_snapshot(repository)
    allowed = tuple(
        fingerprint
        for issue in base.issues
        if issue.code == "index_member_missing_active_symbol"
        for fingerprint in issue.fingerprints
    )
    validate_repository_snapshot(
        candidate_repository,
        allowed_index_identity_gap_fingerprints=allowed,
    ).raise_for_errors()
    metadata = report.manifest_metadata()
    metadata.update(
        {
            "inherited_issue_count": len(report.issues),
            "inherited_issue_candidate_count": sum(
                len(issue.candidate_ids) for issue in report.issues
            ),
            "closed_candidate_id": old_candidate_id,
            "replacement_candidate_id": new_candidate_id,
            "base_resolution_rows": len(existing_resolutions),
            "prewrite_allowed_index_identity_gap_fingerprints": list(allowed),
        }
    )
    return metadata


def prepare_frames(
    existing: Mapping[str, pd.DataFrame],
    bundle: ReviewedBundle,
    *,
    completed_session: str,
    planned_versions: Mapping[str, str],
) -> tuple[dict[str, pd.DataFrame], tuple[SourceArtifact, ...]]:
    master = _rewrite_master(existing["security_master"], bundle)
    history = _rewrite_history(existing["symbol_history"], bundle)
    prices = _rewrite_prices(existing["daily_price_raw"], bundle)
    actions = _rewrite_actions(existing["corporate_actions"], bundle)
    resolutions = _rewrite_resolutions(existing["lifecycle_resolutions"], bundle)
    factors = existing["adjustment_factors"].loc[
        ~existing["adjustment_factors"]["security_id"].astype(str).eq(SECURITY_ID)
    ].copy()
    rebuilt = build_adjustment_factors(
        prices.loc[prices["security_id"].astype(str).eq(SECURITY_ID)],
        actions.loc[actions["security_id"].astype(str).eq(SECURITY_ID)],
        source_version=(
            f"{planned_versions['daily_price_raw']}+"
            f"{planned_versions['corporate_actions']}"
        ),
    )
    # The generic builder timestamps at execution time.  This reviewed repair
    # is an immutable replay, so normalize the derived audit timestamps to the
    # reviewer pin rather than making dataset bytes depend on wall-clock time.
    if not rebuilt.empty:
        rebuilt["calculated_at"] = REVIEWED_AT
        rebuilt["retrieved_at"] = REVIEWED_AT
    factors = pd.concat([factors, rebuilt], ignore_index=True, sort=False)
    artifacts = (
        *bundle.artifacts,
        *bundle.supplemental_artifacts,
        *_reviewed_extraction_artifacts(bundle),
    )
    archive = _rewrite_archive(
        existing["source_archive"], artifacts, completed_session
    )
    frames = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "adjustment_factors": factors,
        "source_archive": archive,
    }
    for dataset, frame in frames.items():
        validate_dataset(
            dataset,
            frame,
            incomplete_action_policy="block",
            completed_session=completed_session,
        ).raise_for_errors()
    return frames, tuple(artifacts)


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    pins_path: Path = DEFAULT_PINS,
) -> PreparedTransition:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current release is required for NTCOY repair.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    bundle = _read_reviewed_bundle(
        repository.root,
        pins_path=pins_path,
        repository=repository,
    )
    if bundle is None:
        raise FileNotFoundError("Reviewed NTCOY bundle has not been promoted.")
    pointer_etags: dict[str, str | None] = {}
    existing: dict[str, pd.DataFrame] = {}
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions[dataset]:
            raise RuntimeError(f"NTCOY release/current pointer mismatch: {dataset}")
        pointer_etags[dataset] = etag
        existing[dataset] = repository.read_frame(
            dataset, release.dataset_versions[dataset]
        )
    if _is_repaired(existing, bundle, repository):
        return PreparedTransition(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            planned_versions={},
            frames=existing,
            archive_artifacts=(),
            summary={
                "status": "already_repaired",
                "network_accessed": False,
                "eodhd_http_attempts_this_run": 0,
                "r2_accessed": False,
            },
        )
    if _has_partial_markers(existing):
        raise RuntimeError("NTCO/NTCOY repair is partially applied; fail closed.")
    current_prices, current_dividends = current_overlap_records(repository, release)
    _validate_old_tables(existing, current_prices, current_dividends)
    if bundle.base_release_version != release.version:
        raise RuntimeError("Promoted NTCOY bundle base release changed before plan.")
    planned = _new_versions(release)
    frames, artifacts = prepare_frames(
        existing,
        bundle,
        completed_session=release.completed_session,
        planned_versions=planned,
    )
    if bundle.overlap_report.get("decision_mode") == PRICE_IDENTITY_TERMINAL_ONLY:
        before_dividends = existing["corporate_actions"].loc[
            existing["corporate_actions"]["event_id"]
            .astype(str)
            .isin(CURRENT_DIVIDEND_EVENT_IDS)
        ].sort_values("event_id", ignore_index=True)
        after_dividends = frames["corporate_actions"].loc[
            frames["corporate_actions"]["event_id"]
            .astype(str)
            .isin(CURRENT_DIVIDEND_EVENT_IDS)
        ].sort_values("event_id", ignore_index=True)
        if _normalized_action_inventory(
            before_dividends
        ) != _normalized_action_inventory(after_dividends):
            raise ValueError(
                "Price-only NTCOY plan changed a preserved dividend row."
            )
    coverage = _validate_candidate_frames(repository, release, frames, bundle)
    non_target_before = existing["adjustment_factors"].loc[
        ~existing["adjustment_factors"]["security_id"].astype(str).eq(SECURITY_ID)
    ].sort_values(["security_id", "session"], ignore_index=True)
    non_target_after = frames["adjustment_factors"].loc[
        ~frames["adjustment_factors"]["security_id"].astype(str).eq(SECURITY_ID)
    ].sort_values(["security_id", "session"], ignore_index=True)
    if not non_target_before.equals(non_target_after):
        raise ValueError("NTCOY factor rebuild changed non-target securities.")
    return PreparedTransition(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned,
        frames=frames,
        archive_artifacts=artifacts,
        summary={
            "status": "validated_offline_plan",
            "base_release_version": release.version,
            "security_id": SECURITY_ID,
            "provider_symbol": PROVIDER_SYMBOL,
            "provider_price_rows": len(bundle.prices),
            "provider_last_session": bundle.provider_last_session,
            "existing_overlap_price_rows": CURRENT_OVERLAP_PRICE_ROWS,
            "existing_overlap_dividend_rows": CURRENT_OVERLAP_DIVIDEND_ROWS,
            "decision_mode": bundle.overlap_report.get("decision_mode"),
            "provider_dividend_raw_decision": bundle.overlap_report[
                "dividend_ambiguity"
            ]["policy"],
            "preserved_dividend_event_ids": sorted(CURRENT_DIVIDEND_EVENT_IDS),
            "maximum_dividend_sensitivity_usd_per_ads": str(
                MAX_DIVIDEND_SENSITIVITY_USD_PER_ADS
            ),
            "release_scope_audit": bundle.overlap_report.get(
                "release_scope_audit"
            ),
            "ticker_change_event_id": TICKER_CHANGE_EVENT_ID,
            "terminal_event_id": TERMINAL_EVENT_ID,
            "terminal_cash_amount": str(TERMINAL_CASH_AMOUNT),
            "archive_rows_added": len(artifacts),
            "planned_versions": dict(planned),
            "network_accessed": False,
            "eodhd_http_attempts_this_run": 0,
            "r2_accessed": False,
            **coverage,
        },
    )


def _persist_artifacts(
    repository: LocalDatasetRepository,
    artifacts: Sequence[SourceArtifact],
    completed_session: str,
) -> None:
    for artifact in artifacts:
        suffix = "json" if artifact.content_type == "application/json" else "bin"
        path = (
            repository.root
            / f"archives/{completed_session}/{artifact.source_hash}.{suffix}.gz"
        )
        encoded = gzip.compress(artifact.content, mtime=0)
        if path.is_file():
            try:
                existing = gzip.decompress(path.read_bytes())
            except Exception as exc:
                raise ValueError(f"Immutable NTCOY archive is unreadable: {path}") from exc
            if existing != artifact.content:
                raise RuntimeError(f"Immutable NTCOY archive conflict: {path}")
        else:
            write_atomic(path, encoded)


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    with _exclusive_file_lock(path, label="Market-store writer"):
        recovery = repository.root / "recovery"
        if recovery.exists() and tuple(recovery.rglob("*.json")):
            raise RuntimeError("A recovery marker blocks NTCOY writes.")
        transaction_root = repository.root / TRANSACTION_DIR
        for journal_path in sorted(transaction_root.glob("*.json")):
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
                status = _text(journal.get("status")) if isinstance(journal, Mapping) else ""
            except Exception as exc:
                raise RuntimeError(
                    f"Unreadable NTCOY transaction journal blocks writes: {journal_path}"
                ) from exc
            if status not in {"committed", "rolled_back"}:
                raise RuntimeError(
                    "Unfinished NTCOY transaction journal blocks writes: "
                    f"{journal_path} (status={status or '<blank>'})."
                )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    write_atomic(path, _canonical_json(dict(value)) + b"\n")


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release: bytes,
    old_pointers: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release: bytes | None,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release:
            if committed_release is None or current.data != committed_release:
                raise RuntimeError(
                    "current release is not owned by this NTCOY transaction"
                )
            repository.objects.put(
                "releases/current.json", old_release, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"release: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            if current.data == old_pointers[dataset]:
                continue
            pointer = CurrentPointer.from_bytes(current.data)
            if pointer.version != planned_versions[dataset]:
                raise RuntimeError(f"unexpected pointer {pointer.version}")
            repository.objects.put(
                key, old_pointers[dataset], if_match=current.etag
            )
        except Exception as exc:
            errors.append(f"{dataset}: {type(exc).__name__}: {exc}")
    for dataset in OUT_OF_SCOPE_CAS_DATASETS:
        try:
            current = repository.objects.get(repository.current_key(dataset))
            if current.data != old_pointers[dataset]:
                raise RuntimeError("out-of-scope current pointer changed")
        except Exception as exc:
            errors.append(f"{dataset}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedTransition,
    *,
    pins_path: Path = DEFAULT_PINS,
    inject_failure: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if prepared.summary.get("network_accessed"):
        raise RuntimeError("Fetch/promote and apply must be separate invocations.")
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        if prepared.summary.get("status") == "already_repaired":
            verified = prepare_repair(repository, pins_path=pins_path)
            if verified.summary.get("status") != "already_repaired":
                raise RuntimeError(
                    "Caller-reported NTCOY repaired state failed locked replay."
                )
            return {
                **verified.summary,
                "mode": "apply",
                "writes_performed": False,
            }
        release, release_etag = repository.current_release()
        if (
            release is None
            or release.version != prepared.release.version
            or release_etag != prepared.release_etag
        ):
            raise RuntimeError("Current release changed after NTCOY preflight.")
        # Replan under the writer lock. Caller-owned frames and UUID versions
        # are never trusted for writes.
        current_plan = prepare_repair(repository, pins_path=pins_path)
        if current_plan.summary["status"] != "validated_offline_plan":
            raise RuntimeError("Locked NTCOY replan did not produce a writable plan.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in REQUIRED_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != current_plan.release.dataset_versions[dataset]
                or value.etag != current_plan.pointer_etags[dataset]
            ):
                raise RuntimeError(f"NTCOY pointer changed before apply: {dataset}")
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = (
            repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "us_ntco_ntcoy_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": current_plan.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value).decode("ascii")
                for dataset, value in old_pointers.items()
            },
            "planned_versions": dict(current_plan.planned_versions),
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            _persist_artifacts(
                repository,
                current_plan.archive_artifacts,
                current_plan.release.completed_session,
            )
            inject("after_artifacts")
            versions = dict(current_plan.release.dataset_versions)
            pin_map = _pin_map(validate_pin_contract(pins_path))
            for dataset in WRITE_DATASETS:
                parent = repository.manifest_for_version(
                    dataset, current_plan.release.dataset_versions[dataset]
                )
                metadata = dict(parent.metadata)
                metadata.update(
                    {
                        "operation": "repair_us_ntco_ntcoy_transition",
                        "ntco_security_id": SECURITY_ID,
                        "ntcoy_provider_symbol": PROVIDER_SYMBOL,
                        "ntcoy_ticker_change_date": TICKER_CHANGE_DATE,
                        "ntcoy_terminal_effective_date": TERMINAL_EFFECTIVE_DATE,
                        "ntcoy_terminal_cash_usd": str(TERMINAL_CASH_AMOUNT),
                        "ntco_overlap_price_records_sha256": CURRENT_OVERLAP_PRICE_SHA256,
                        "ntco_overlap_dividend_records_sha256": CURRENT_OVERLAP_DIVIDEND_SHA256,
                        "ntcoy_reviewed_raw_pins": pin_map,
                        "ntcoy_decision_mode": current_plan.summary.get(
                            "decision_mode"
                        ),
                        "ntcoy_provider_dividend_raw_decision": (
                            current_plan.summary.get(
                                "provider_dividend_raw_decision"
                            )
                        ),
                        "ntco_preserved_dividend_event_ids": list(
                            current_plan.summary.get(
                                "preserved_dividend_event_ids", ()
                            )
                        ),
                        "ntco_maximum_dividend_sensitivity_usd_per_ads": (
                            current_plan.summary.get(
                                "maximum_dividend_sensitivity_usd_per_ads"
                            )
                        ),
                        "ntco_release_scope_audit": current_plan.summary.get(
                            "release_scope_audit"
                        ),
                        "daily_price_version": current_plan.planned_versions[
                            "daily_price_raw"
                        ],
                        "corporate_action_version": current_plan.planned_versions[
                            "corporate_actions"
                        ],
                        "eodhd_http_attempts_this_run": 0,
                        "network_accessed": False,
                        "r2_accessed": False,
                    }
                )
                result = repository.write_frame(
                    dataset,
                    current_plan.frames[dataset],
                    completed_session=current_plan.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=metadata,
                    expected_pointer_etag=current_plan.pointer_etags[dataset],
                    version=current_plan.planned_versions[dataset],
                )
                if result.conflict:
                    raise RuntimeError(f"NTCOY write conflicted: {dataset}")
                versions[dataset] = result.manifest.version
                inject(f"after_dataset:{dataset}")
            for dataset in OUT_OF_SCOPE_CAS_DATASETS:
                pointer, etag = repository.current_pointer(dataset)
                if (
                    pointer is None
                    or pointer.version
                    != current_plan.release.dataset_versions[dataset]
                    or etag != current_plan.pointer_etags[dataset]
                ):
                    raise RuntimeError(
                        f"Out-of-scope pointer changed during NTCOY apply: {dataset}"
                    )
            written = {
                dataset: repository.read_frame(
                    dataset, current_plan.planned_versions[dataset]
                )
                for dataset in WRITE_DATASETS
            }
            bundle = _read_reviewed_bundle(
                repository.root,
                pins_path=pins_path,
                repository=repository,
            )
            if bundle is None or not _is_repaired(written, bundle, repository):
                raise RuntimeError("Written NTCOY snapshot failed exact invariant.")
            candidate = _CandidateRepository(repository, versions, written)
            allowed = tuple(
                _text(value)
                for value in current_plan.summary.get(
                    "prewrite_allowed_index_identity_gap_fingerprints",
                    (),
                )
            )
            validate_repository_snapshot(
                candidate,
                allowed_index_identity_gap_fingerprints=allowed,
            ).raise_for_errors()
            committed = repository.commit_release(
                current_plan.release.completed_session,
                versions,
                quality=current_plan.release.quality,
                warnings=(
                    tuple(current_plan.release.warnings)
                    + (
                        (PRICE_ONLY_RELEASE_WARNING,)
                        if current_plan.summary.get("decision_mode")
                        == PRICE_IDENTITY_TERMINAL_ONLY
                        and PRICE_ONLY_RELEASE_WARNING
                        not in current_plan.release.warnings
                        else ()
                    )
                ),
                expected_etag=current_plan.release_etag,
            )
            current_release, _ = repository.current_release()
            if (
                current_release is None
                or current_release.to_bytes() != committed.to_bytes()
            ):
                raise RuntimeError("Committed NTCOY release is not current.")
            for dataset in REQUIRED_DATASETS:
                pointer, _ = repository.current_pointer(dataset)
                if (
                    pointer is None
                    or pointer.version != committed.dataset_versions[dataset]
                ):
                    raise RuntimeError(
                        f"Committed NTCOY pointer mismatch: {dataset}."
                    )
            inject("after_release_commit")
            replay = prepare_repair(repository, pins_path=pins_path)
            if replay.summary["status"] != "already_repaired":
                raise RuntimeError("Committed NTCOY repair is not idempotent.")
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
                "transaction_id": transaction_id,
                "new_release_version": committed.version,
                "writes_performed": True,
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release=old_release.data,
                old_pointers=old_pointers,
                planned_versions=current_plan.planned_versions,
                committed_release=committed.to_bytes() if committed is not None else None,
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
                    f"NTCOY rollback failed; recovery marker created: {recovery}"
                ) from original
            raise


def readiness_plan(
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    supplemental_evidence_dir: Path = DEFAULT_SUPPLEMENTAL_EVIDENCE_DIR,
    supplemental_books_closed_evidence_dir: Path = (
        DEFAULT_SUPPLEMENTAL_BOOKS_CLOSED_EVIDENCE_DIR
    ),
    pins_path: Path = DEFAULT_PINS,
    repository: LocalDatasetRepository | None = None,
) -> dict[str, Any]:
    document = validate_pin_contract(pins_path)
    official_pins = document["official_sources"]
    supplemental_pins = document["supplemental_official_sources"]
    provider_pins = document["provider"]["requests"]
    official: dict[str, Any] = {}
    blockers: list[str] = []
    for key in OFFICIAL_URLS:
        staged = verify_staged_official(evidence_dir, key)
        pinned = _text(official_pins[key].get("source_sha256")).lower()
        if staged is None:
            state = "missing_cache"
            blockers.append(f"official_cache_missing:{key}")
        elif not pinned:
            state = "pending_reviewer_pin"
            blockers.append(f"official_pin_missing:{key}")
        elif staged.source_sha256 != pinned:
            state = "hash_mismatch"
            blockers.append(f"official_pin_mismatch:{key}")
        else:
            state = "pinned"
        official[key] = {
            "state": state,
            "source_url": OFFICIAL_URLS[key],
            "observed_sha256": staged.source_sha256 if staged else "",
            "pinned_sha256": pinned,
            "required_reviewed_claims": dict(OFFICIAL_REVIEWED_CLAIMS[key]),
        }
    supplemental_official: dict[str, Any] = {}
    supplemental_dirs = {
        "bny_termination": supplemental_evidence_dir,
        "bny_books_closed": supplemental_books_closed_evidence_dir,
    }
    for key in SUPPLEMENTAL_OFFICIAL_URLS:
        pinned = _text(supplemental_pins[key].get("source_sha256")).lower()
        staged: StagedOfficialEvidence | None = None
        semantic_error = ""
        try:
            staged = verify_staged_official(supplemental_dirs[key], key)
            if staged is not None and staged.source_sha256 == pinned:
                artifact = _supplemental_official_artifact(
                    supplemental_dirs[key],
                    source_key=key,
                )
                _validate_supplemental_artifact_pin(
                    artifact,
                    document,
                    source_key=key,
                )
                _validate_supplemental_official_semantics(
                    artifact,
                    source_key=key,
                )
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
        if semantic_error and staged is None:
            state = "invalid_cache"
            blockers.append(f"supplemental_official_cache_invalid:{key}")
        elif staged is None:
            state = "missing_cache"
            blockers.append(f"supplemental_official_cache_missing:{key}")
        elif staged.source_sha256 != pinned:
            state = "hash_mismatch"
            blockers.append(f"supplemental_official_pin_mismatch:{key}")
        elif semantic_error:
            state = "semantic_mismatch"
            blockers.append(f"supplemental_official_semantic_mismatch:{key}")
        else:
            state = "pinned_semantics_validated"
        supplemental_official[key] = {
            "state": state,
            "source_url": SUPPLEMENTAL_OFFICIAL_URLS[key],
            "observed_sha256": staged.source_sha256 if staged else "",
            "pinned_sha256": pinned,
            "required_reviewed_claims": dict(
                SUPPLEMENTAL_OFFICIAL_REVIEWED_CLAIMS[key]
            ),
            "semantic_error": semantic_error,
            "part_of_original_six_raw_signature": False,
        }
    provider_requests: list[dict[str, Any]] = []
    for position, endpoint in enumerate(EODHD_ENDPOINTS, start=1):
        digest = _text(provider_pins[endpoint].get("source_sha256")).lower()
        if not digest:
            blockers.append(f"provider_raw_pin_missing:{endpoint}")
        provider_requests.append(
            {
                "order": position,
                "endpoint": endpoint,
                "provider_symbol": PROVIDER_SYMBOL,
                "params": dict(EODHD_REQUEST_PARAMS),
                "source_url": EODHD_REQUEST_URLS[endpoint],
                "pinned_sha256": digest,
                "decision": (
                    "approved_exact_price_input"
                    if endpoint == "eod"
                    else "archived_exact_rejected_conflict_evidence"
                    if endpoint == "div"
                    else "approved_exact_empty_split_input"
                ),
                "retry_count": 0,
            }
        )
    cache_root = evidence_dir.parents[3]
    repository = repository or LocalDatasetRepository(cache_root)
    reviewed_path = _reviewed_cache_path(cache_root)
    if not reviewed_path.is_file():
        blockers.append("reviewed_provider_bundle_missing")
    quarantine_decision: dict[str, Any] | None = None
    observed_quarantine_path = _quarantine_path(
        cache_root,
        OBSERVED_UNREVIEWED_QUARANTINE_ID,
    )
    if observed_quarantine_path.is_file():
        try:
            quarantine_decision = assess_quarantine_decision(
                cache_root,
                OBSERVED_UNREVIEWED_QUARANTINE_ID,
                repository=repository,
                supplemental_evidence_dir=supplemental_evidence_dir,
                supplemental_books_closed_evidence_dir=(
                    supplemental_books_closed_evidence_dir
                ),
                pins_path=pins_path,
            )
        except Exception as exc:
            blockers.append(
                "observed_quarantine_decision_invalid:"
                f"{type(exc).__name__}:{exc}"
            )
        else:
            blockers.extend(quarantine_decision["blockers"])
    request_inventory = [
        {
            "order": position,
            "kind": "official",
            "source_key": key,
            "source_url": OFFICIAL_URLS[key],
            "execution_contract": "separate invocation, one attempt, no retry, no redirect",
        }
        for position, key in enumerate(OFFICIAL_URLS, start=1)
    ]
    request_inventory.extend(
        {
            "order": position,
            "kind": "eodhd",
            "source_key": row["endpoint"],
            "source_url": row["source_url"],
            "execution_contract": "one attempt, no retry, preserve raw response",
        }
        for position, row in enumerate(provider_requests, start=4)
    )
    base_result = {
        "schema": "us_ntco_ntcoy_transition_readiness/v1",
        "status": "blocked_pending_evidence",
        "apply_allowed": False,
        "blockers": sorted(blockers),
        "official_sources": official,
        "supplemental_official_sources": supplemental_official,
        "official_fetch_order": list(OFFICIAL_URLS),
        "official_fetch_contract": {
            "one_source_per_invocation": True,
            "max_http_attempts_per_run": MAX_OFFICIAL_HTTP_ATTEMPTS_PER_RUN,
            "automatic_redirects": False,
            "retries": 0,
        },
        "eodhd_acquisition_implemented": True,
        "eodhd_calls_this_run": 0,
        "eodhd_future_call_cap": MAX_EODHD_HTTP_ATTEMPTS,
        "eodhd_requests": provider_requests,
        "six_request_inventory": request_inventory,
        "independent_evidence_inventory": [
            {
                "kind": "supplemental_official",
                "source_key": key,
                "source_url": SUPPLEMENTAL_OFFICIAL_URLS[key],
                "state": supplemental_official[key]["state"],
                "part_of_original_six_raw_signature": False,
            }
            for key in SUPPLEMENTAL_OFFICIAL_URLS
        ],
        "unreviewed_quarantine_decision": quarantine_decision,
        "current_ntco_tail": {
            "state": "dividend_economics_preserved_under_price_only_policy",
            "price_rows": CURRENT_OVERLAP_PRICE_ROWS,
            "first_session": CURRENT_OVERLAP_FIRST_SESSION,
            "last_session": CURRENT_OVERLAP_LAST_SESSION,
            "price_records_sha256": CURRENT_OVERLAP_PRICE_SHA256,
            "raw_eod_archive_id": CURRENT_EOD_ARCHIVE_ID,
            "raw_eod_sha256": CURRENT_EOD_RAW_SHA256,
            "dividend_rows": CURRENT_OVERLAP_DIVIDEND_ROWS,
            "dividend_records_sha256": CURRENT_OVERLAP_DIVIDEND_SHA256,
            "raw_dividend_archive_id": CURRENT_DIVIDEND_ARCHIVE_ID,
            "raw_dividend_sha256": CURRENT_DIVIDEND_RAW_SHA256,
        },
        "decision_gates": [
            "reviewer pins all three official raw SHA-256 values",
            "a separately authorized collector preserves and pins exactly three "
            "NTCOY.US raw responses",
            "the exact code-pinned 43-row alias-difference inventory and all "
            "three provider raw hashes match",
            "BNY notice ad1140774 confirms ADR-facility termination at 5 PM ET on 2024-08-07",
            "the two NTCOY dividend values remain explicit rejected conflict evidence; "
            "the two existing NTCO action rows and values remain unchanged",
            "the current release proves zero NTCO index-anchor and membership-event rows",
            "the maximum two-event dividend sensitivity is recorded as USD 0.01585 per ADS",
            "any deviation from the exact observed profile blocks and quarantines both bundles",
            "transaction models same-security NTCO->NTCOY ticker_change on 2024-02-12",
            "transaction models BNY cash-settled delisting at USD 5.043659 on 2024-09-04",
            "transaction rollback, lifecycle, factors, and backtest ledger pass "
            "before release commit",
        ],
        "transition_model": transition_model(),
        "network_accessed": False,
        "writes_performed": False,
        "r2_accessed": False,
    }
    if reviewed_path.is_file():
        prepared = prepare_repair(repository, pins_path=pins_path)
        return {
            **base_result,
            **prepared.summary,
            "mode": "plan",
            "apply_allowed": prepared.summary["status"]
            in {"validated_offline_plan", "already_repaired"},
            "reviewed_bundle_path": str(reviewed_path),
            "network_accessed": False,
            "writes_performed": False,
        }
    return base_result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage official evidence or print the fail-closed NTCO/NTCOY plan."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument(
        "--supplemental-evidence-dir",
        type=Path,
        default=DEFAULT_SUPPLEMENTAL_EVIDENCE_DIR,
    )
    parser.add_argument(
        "--supplemental-books-closed-evidence-dir",
        type=Path,
        default=DEFAULT_SUPPLEMENTAL_BOOKS_CLOSED_EVIDENCE_DIR,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch-official", choices=tuple(ALL_OFFICIAL_URLS))
    mode.add_argument("--fetch-eodhd", action="store_true")
    mode.add_argument("--promote-quarantine", metavar="SHA256")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence_dir = (
        args.cache_root / "state/issuer_lifecycle/ntco_ntcoy_transition/official"
    )
    if args.fetch_official:
        load_env()
        target_evidence_dir = (
            args.supplemental_evidence_dir
            if args.fetch_official == "bny_termination"
            else args.supplemental_books_closed_evidence_dir
            if args.fetch_official == "bny_books_closed"
            else evidence_dir
        )
        result = fetch_official(
            target_evidence_dir,
            args.fetch_official,
            user_agent=os.getenv("SEC_USER_AGENT", ""),
        )
    elif args.fetch_eodhd:
        result = collect_eodhd_stage(
            args.cache_root,
            pins_path=args.pins,
        )
    elif args.promote_quarantine:
        result = promote_quarantine(
            args.cache_root,
            args.promote_quarantine,
            pins_path=args.pins,
            supplemental_evidence_dir=args.supplemental_evidence_dir,
            supplemental_books_closed_evidence_dir=(
                args.supplemental_books_closed_evidence_dir
            ),
        )
    elif args.apply:
        repository = LocalDatasetRepository(args.cache_root)
        prepared = prepare_repair(repository, pins_path=args.pins)
        result = apply_repair(repository, prepared, pins_path=args.pins)
    else:
        result = readiness_plan(
            evidence_dir=evidence_dir,
            supplemental_evidence_dir=args.supplemental_evidence_dir,
            supplemental_books_closed_evidence_dir=(
                args.supplemental_books_closed_evidence_dir
            ),
            pins_path=args.pins,
            repository=LocalDatasetRepository(args.cache_root),
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
