#!/usr/bin/env python3
"""Collect a small, audited set of missing EODHD successor histories.

The default mode performs provider calls and validates the complete candidate
snapshot without writing it.  ``--offline-plan`` performs no provider call,
and ``--apply`` is the only mode that advances dataset and release pointers.
"""

from __future__ import annotations

import argparse
import ast
import base64
import fcntl
import gzip
import html
import io
import json
import math
import re
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlparse

import exchange_calendars as xcals
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import (
    EodhdClient,
    EodhdDailySource,
    SourceArtifact,
)
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_EVIDENCE_REPORT = Path(
    "results/data_quality/us_lifecycle/sec_collection_v7.json"
)
FETCH_START = "2015-01-01"
DWDP_START = "2017-09-01"
EXPECTED_SUCCESSOR_CODES = 13
ENDPOINTS_PER_CODE = 3
MAX_EODHD_HTTP_ATTEMPTS = EXPECTED_SUCCESSOR_CODES * ENDPOINTS_PER_CODE
RECENT_PRICE_DAYS = 10
MISSING_PROVIDER_WARNING = "Market-data provider missing symbols: 1"
IDENTITY_REPAIR_MIGRATION_WARNING = (
    "Pending audited US index identity repairs: 11 security_ids"
)
# The successor transaction must run against the frozen v7 evidence report
# before the identity-repair transaction removes/rekeys several of its exact
# candidates.  These are the complete, independently audited price-coverage
# gaps in that frozen snapshot after COV itself is repaired.  The allow-list is
# transaction-local: ordinary validation and R2 publication remain strict.
IDENTITY_REPAIR_MIGRATION_GAP_IDS = frozenset(
    {
        "US:EODHD:1b6b9beb-42b0-5a06-81f3-23a49627565f",  # current LILA
        "US:EODHD:7fda02a3-10dd-51a3-96cb-41695fcff341",  # current LILAK
        "US:EODHD:bf3b55f9-c8af-5738-9772-2aa3f5b689b8",  # current FOX
        "US:EODHD:5c7fb4cf-793f-582b-9002-a5aa62819933",  # current FOXA
        "US:EODHD:cec57207-c56c-51c0-955f-204bca9b27c8",  # Sea / reused SE
        "US:EODHD:dc3f4283-a3cc-5bc7-916c-9ffdd71c9874",  # WYND/TNL
        "US:EODHD:67d3c3b7-8b0a-5475-91bc-6e7362300031",  # BHGE duplicate
        "US:EODHD:33cf5387-6cec-598e-84a9-563ca333b0f3",  # new ARNC
        "US:EODHD:6a76982a-782c-5b73-abd3-8c86f47d3a1f",  # new IR
        "US:EODHD:9f13974d-7f81-5aac-a3a7-ed1d184bd76b",  # legacy AGN
        "US:EODHD:2c15a3cb-4bdb-5b6f-b82e-8e17e286ee69",  # CoreSite / COR reuse
    }
)
COV_FETCH_START = "2015-01-01"
COV_LAST_TRADING_DATE = "2015-01-26"
COV_FIRST_SUCCESSOR_SESSION = "2015-01-27"
COV_MAX_EODHD_HTTP_ATTEMPTS = ENDPOINTS_PER_CODE
COV_EODHD_FAILURE_SESSION = "2015-01-02"
COV_EODHD_SYMBOL_FAILURE_URL = (
    "https://eodhd.com/api/eod-bulk-last-day/US"
    "?date=2015-01-02&symbols=COV&fmt=json"
)
COV_EODHD_SYMBOL_FAILURE_SHA256 = (
    "28e3f7baa0715126eacbfbb8b9ea32fcf2c10d36167e096a28cb07365c0f3806"
)
COV_EODHD_SYMBOL_FAILURE_RETURNED_DATE = "2012-06-15"
COV_EODHD_FULL_US_FAILURE_URL = (
    "https://eodhd.com/api/eod-bulk-last-day/US?date=2015-01-02&fmt=json"
)
COV_EODHD_FULL_US_FAILURE_SHA256 = (
    "d2556f73988e6488656232bbf31df609ef88d256f7ae5214ea9b36e5e2600b68"
)
COV_EODHD_FULL_US_FAILURE_ROWS = 42096
COV_EODHD_LEDGER_USAGE_AFTER_FAILURES = 8697
COV_PENDING_INDEPENDENT_VALIDATION_WARNING = (
    "COV terminal prices lack independent pinned-history cross-validation"
)
COV_WIKI_COMMIT = "0bcc715e2dd37b7ecec65c549be843574120bd58"
COV_WIKI_RAW_BASE = (
    "https://raw.githubusercontent.com/teddykoker/survivorship-free-spy/"
    f"{COV_WIKI_COMMIT}/survivorship-free"
)
COV_WIKI_URLS = {
    "cov_csv": f"{COV_WIKI_RAW_BASE}/data/COV.csv",
    "readme": f"{COV_WIKI_RAW_BASE}/README.md",
    "generate_py": f"{COV_WIKI_RAW_BASE}/generate.py",
}
COV_WIKI_SHA256 = {
    "cov_csv": "41bba9dde2282b8d69e5776c05af9b6b0b5026cd2664699974d98ed9e65b7731",
    "readme": "9d3bf24ed92378a992bc303f50b486efe02c43b6363ec7082cfbe99b3c794914",
    "generate_py": "992da8b5fdbe50808672b5de1f03c2cddbcbf0446e69b1f4725ec1ac67a5b32b",
}
COV_WIKI_CSV_SHA256 = COV_WIKI_SHA256["cov_csv"]
COV_WIKI_MAX_HTTP_ATTEMPTS = len(COV_WIKI_URLS)
COV_DIRECTINDEX_COMMIT = "8359b09ac8a00f1688ec9a1323a5533f0fc151d1"
COV_DIRECTINDEX_RAW_BASE = (
    "https://raw.githubusercontent.com/bdi2357/DirectIndexing/"
    f"{COV_DIRECTINDEX_COMMIT}"
)
COV_DIRECTINDEX_URLS = {
    "cov_csv": f"{COV_DIRECTINDEX_RAW_BASE}/data/PriceVolume/COV.csv",
    "readme": f"{COV_DIRECTINDEX_RAW_BASE}/README.md",
    "license": f"{COV_DIRECTINDEX_RAW_BASE}/LICENSE",
}
COV_DIRECTINDEX_SHA256 = {
    "cov_csv": "458e4272937f87f40330679d76bd6e9d6e3fd833fc22093c42f947ab3427cd03",
    "readme": "ab4ef229693f35ef0304cae8b3ef67b53e2a10f087e0a756035642f27f85d444",
    "license": "1a30ce35f47c785f749a3ed17ce4c6427ddd4f7eaafe6868bca14ee6dbe66694",
}
COV_DIRECTINDEX_MAX_HTTP_ATTEMPTS = len(COV_DIRECTINDEX_URLS)
COV_CROSS_OHLC_ABS_TOLERANCE = 0.001
COV_SEC_COMPLETION_URL = (
    "https://www.sec.gov/Archives/edgar/data/1385187/"
    "000119312515020714/d860155d8k.htm"
)
COV_ISSUER_RELEASE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1613103/"
    "000119312515020681/d859999dex991.htm"
)
COV_EXPECTED_SESSIONS = (
    "2015-01-02",
    "2015-01-05",
    "2015-01-06",
    "2015-01-07",
    "2015-01-08",
    "2015-01-09",
    "2015-01-12",
    "2015-01-13",
    "2015-01-14",
    "2015-01-15",
    "2015-01-16",
    "2015-01-20",
    "2015-01-21",
    "2015-01-22",
    "2015-01-23",
    "2015-01-26",
)
SPECTRA_SECURITY_ID = "US:EODHD:5fa7bd33-c752-57c7-873c-e9d812d90e05"
SPECTRA_LAST_TRADING_DATE = "2017-02-24"
SPECTRA_SYMBOL_HISTORY_END = "2017-02-26"
ENB_EFFECTIVE_DATE = "2017-02-27"
SPECTRA_MASTER_ACTIVE_TO = ENB_EFFECTIVE_DATE
ENB_RATIO = 0.984
ENB_MAX_EODHD_HTTP_ATTEMPTS = ENDPOINTS_PER_CODE
ENB_SEC_COMPLETION_URL = (
    "https://www.sec.gov/Archives/edgar/data/1373835/"
    "000119312517057856/d338638dex991.htm"
)
ENB_SEC_COMPLETION_SHA256 = (
    "4c9bed57c0ea0a194c163a857c724daf77a27fcb46a5441dd312c847141b983f"
)
ENB_SEC_TERMS_URL = (
    "https://www.sec.gov/Archives/edgar/data/1373835/"
    "000119312517057856/0001193125-17-057856.txt"
)
ENB_OFFICIAL_EVIDENCE_SCHEMA = "official_identity_evidence_raw/v1"


def _cov_xnys_sessions() -> tuple[str, ...]:
    sessions = xcals.get_calendar("XNYS").sessions_in_range(
        COV_FETCH_START,
        COV_LAST_TRADING_DATE,
    )
    return tuple(pd.Timestamp(value).date().isoformat() for value in sessions)


@dataclass(frozen=True)
class SuccessorSpec:
    symbol: str
    provider_code: str
    event_date: str
    history_start: str
    first_price_not_after: str
    catalog_kind: str
    name_tokens: tuple[str, ...]
    require_recent: bool = True

    @property
    def provider_symbol(self) -> str:
        return f"{self.provider_code}.US"

    @property
    def security_id(self) -> str:
        value = f"eodhd:US:{self.provider_code}:symbol:{self.symbol}"
        return f"US:EODHD:{uuid.uuid5(uuid.NAMESPACE_URL, value)}"


@dataclass(frozen=True)
class PurgeSpec:
    symbol: str
    cutoff: str
    name_tokens: tuple[str, ...] = ()
    history_active_to: str = ""


@dataclass(frozen=True)
class CatalogArchive:
    kind: str
    rows: tuple[dict[str, Any], ...]
    source_url: str
    retrieved_at: str
    source_hash: str


@dataclass(frozen=True)
class CatalogSelection:
    spec: SuccessorSpec
    row: dict[str, Any]
    archive: CatalogArchive


@dataclass(frozen=True)
class LoadedEvidenceReport:
    data: dict[str, Any]
    artifact: SourceArtifact


@dataclass(frozen=True)
class LocalPreflight:
    selections: dict[str, CatalogSelection]
    existing: dict[str, pd.DataFrame]
    purge_ids: dict[str, str]
    pointer_etags: dict[str, str | None]
    actavis_id: str
    dwdp_id: str
    repaired_identity_ids: dict[str, str]
    cov_selection: CatalogSelection
    cov_security_id: str
    mdt_security_id: str


@dataclass(frozen=True)
class EnbPreflight:
    selection: CatalogSelection
    existing: dict[str, pd.DataFrame]
    pointer_etags: dict[str, str | None]
    spectra_security_id: str
    official_artifact: SourceArtifact


@dataclass(frozen=True)
class FetchedBundle:
    prices: pd.DataFrame
    corporate_actions: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    missing_symbols: tuple[str, ...]


@dataclass(frozen=True)
class CovWikiEvidenceBundle:
    prices: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    http_attempts: int


@dataclass(frozen=True)
class CovDirectIndexBundle:
    prices: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    http_attempts: int


@dataclass
class PreparedCollection:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    artifacts: tuple[SourceArtifact, ...]
    archive_artifacts: tuple[SourceArtifact, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]
    cleared_release_warnings: tuple[str, ...] = ()


SUCCESSOR_SPECS = (
    SuccessorSpec("DINO", "DINO", "2022-03-15", "2022-03-15", "2022-03-22", "active", ("hf sinclair",)),
    SuccessorSpec("BTI", "BTI", "2017-07-25", "2015-01-01", "2015-01-09", "active", ("british american tobacco",)),
    SuccessorSpec("BFH", "BFH", "2022-04-04", "2022-04-04", "2022-04-11", "active", ("bread financial",)),
    SuccessorSpec("NTCO", "NTCO", "2020-01-03", "2020-01-03", "2020-01-10", "delisted", ("natura",), False),
    SuccessorSpec("ECA", "ECA", "2019-02-13", "2015-01-01", "2015-01-09", "delisted", ("ovintiv",), False),
    SuccessorSpec("FBIN", "FBIN", "2023-01-23", "2023-01-23", "2023-01-30", "active", ("fortune brands innovations",)),
    SuccessorSpec("GIL", "GIL", "2025-12-01", "2015-01-01", "2015-01-09", "active", ("gildan",)),
    SuccessorSpec("CP", "CP", "2021-12-14", "2015-01-01", "2015-01-09", "active", ("canadian pacific",)),
    SuccessorSpec("VEON", "VEON", "2017-03-30", "2017-03-30", "2017-04-06", "active", ("veon",)),
    SuccessorSpec("DVMT", "DVMT", "2016-09-07", "2016-09-07", "2016-09-14", "delisted", ("dell technologies",), False),
    SuccessorSpec("TAK", "TAK", "2019-01-08", "2019-01-08", "2019-01-08", "active", ("takeda",)),
    SuccessorSpec("GAP", "GAP", "2024-08-22", "2024-08-22", "2024-08-29", "active", ("gap",)),
    SuccessorSpec("QVCGA", "QVCGA", "2025-02-24", "2025-02-24", "2025-03-06", "delisted", ("qvc group",), False),
)


# COV is deliberately not part of SUCCESSOR_SPECS.  The frozen 13-code bundle
# and its 39-call signature must remain reusable byte-for-byte.  This one
# terminal identity is fetched and cached independently.
COV_SPEC = SuccessorSpec(
    "COV",
    "COV",
    COV_LAST_TRADING_DATE,
    COV_FETCH_START,
    "2015-01-02",
    "delisted",
    ("covidien",),
    False,
)


# ENB is intentionally isolated from the immutable 13-symbol/39-call bundle.
# Adding it to SUCCESSOR_SPECS would invalidate a successfully archived provider
# response set and spend 39 unnecessary EODHD calls.  This supplement therefore
# has its own signature, three-call cap, cache, preflight, and explicit CLI mode.
ENB_SPEC = SuccessorSpec(
    "ENB",
    "ENB",
    ENB_EFFECTIVE_DATE,
    FETCH_START,
    ENB_EFFECTIVE_DATE,
    "active",
    ("enbridge",),
)


PURGE_SPECS = (
    PurgeSpec("VIP", "2017-03-30"),
    PurgeSpec("VIAC", "2022-02-17"),
    PurgeSpec("FBHS", "2023-01-23"),
    PurgeSpec("MYL", "2020-11-16"),
    PurgeSpec("HFC", "2022-03-15"),
    PurgeSpec("ADS", "2022-04-04"),
    PurgeSpec("JEC", "2019-12-10"),
    PurgeSpec("TMK", "2019-08-09"),
    PurgeSpec("MMC", "2026-01-14"),
    PurgeSpec("FI", "2025-11-11"),
    PurgeSpec("ANTM", "2022-06-28"),
    # Keep old DOW's legal-completion close on 2017-08-31.  The first
    # tradable successor session, and therefore the trim boundary, is 9/1.
    PurgeSpec("DOW", "2017-09-01"),
    PurgeSpec("PX", "2018-10-31"),
    # Legacy Allergan stopped trading on 3/16, but remained the represented
    # S&P constituent until AAL replaced it before the 3/23 open.  Keep its
    # symbol resolvable through the day before that effective REMOVE without
    # manufacturing any extra price rows.
    PurgeSpec("AGN", "2015-03-17", ("allergan inc",), "2015-03-22"),
    PurgeSpec("SHPG", "2019-01-08"),
    PurgeSpec("GPS", "2024-08-22"),
    PurgeSpec("PKI", "2023-05-16"),
    PurgeSpec("CTL", "2020-09-18"),
    PurgeSpec("BK", "2026-05-21", ("bank of new york mellon",)),
    PurgeSpec("COR", "2021-12-28", ("coresite",)),
    PurgeSpec("QRTEA", "2025-02-24"),
)


WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "source_archive",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect and validate only the audited EODHD lifecycle successors. "
            "No dataset is written unless --apply is supplied."
        )
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--evidence-report", default=str(DEFAULT_EVIDENCE_REPORT))
    parser.add_argument("--workers", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument(
        "--offline-plan",
        action="store_true",
        help="Validate the cached catalog/evidence and print a plan without EODHD calls.",
    )
    parser.add_argument(
        "--fetch-cov-directindex",
        action="store_true",
        help=(
            "Explicitly allow three no-retry requests for the pinned DirectIndex "
            "COV CSV, README, and LICENSE artifacts."
        ),
    )
    parser.add_argument(
        "--fetch-cov-wiki",
        action="store_true",
        help=(
            "Explicitly allow three no-retry requests for the pinned COV CSV, "
            "README, and generate.py Quandl WIKI evidence."
        ),
    )
    parser.add_argument(
        "--cov-eodhd-full-us-failure-response",
        default="",
        help=(
            "Explicit local path to the already retrieved 2015-01-02 full-US "
            "EODHD failure response. It is hash/row validated and immutably imported; "
            "this option never performs a provider request."
        ),
    )
    parser.add_argument(
        "--enb-only",
        action="store_true",
        help=(
            "Run only the exact Spectra Energy successor supplement. This mode "
            "never replays or refetches the frozen 13-symbol/COV bundles."
        ),
    )
    parser.add_argument(
        "--fetch-enb",
        action="store_true",
        help=(
            "Explicitly allow at most three single-attempt EODHD calls for "
            "ENB.US (eod, div, splits). Without this flag only the immutable "
            "ENB cache may be replayed."
        ),
    )
    return parser.parse_args(argv)


def _read_release_frames(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, pd.DataFrame]:
    return {
        dataset: repository.read_frame(dataset, version)
        for dataset, version in release.dataset_versions.items()
    }


class CappedSingleAttemptEodhdClient(EodhdClient):
    """EODHD client with one HTTP attempt per endpoint and one run-wide cap."""

    def __init__(self, *args, max_attempts: int = MAX_EODHD_HTTP_ATTEMPTS, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_attempts = int(max_attempts)
        self._attempt_count = 0
        self._attempt_lock = threading.Lock()

    @property
    def attempt_count(self) -> int:
        with self._attempt_lock:
            return self._attempt_count

    def get_json(self, endpoint: str, *, params: dict[str, object] | None = None):
        safe_endpoint = "/" + endpoint.strip("/")
        query = {**(params or {}), "api_token": self.token, "fmt": "json"}
        with self._attempt_lock:
            if self._attempt_count >= self.max_attempts:
                raise RuntimeError(
                    "EODHD lifecycle call cap reached before HTTP request: "
                    f"attempts={self._attempt_count}, maximum={self.max_attempts}."
                )
            self.budget.claim()
            self._attempt_count += 1
        try:
            response = self.session.get(
                self.base_url + safe_endpoint,
                params=query,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f"HTTP {status}" if status else type(exc).__name__
            raise RuntimeError(
                f"EODHD single attempt failed for {safe_endpoint}: {detail}"
            ) from None


class CappedEodhdDailySource(EodhdDailySource):
    def __init__(
        self,
        *,
        workers: int = 8,
        max_attempts: int = MAX_EODHD_HTTP_ATTEMPTS,
    ):
        super().__init__(
            client=CappedSingleAttemptEodhdClient(max_attempts=max_attempts),
            workers=workers,
        )


@dataclass(frozen=True)
class CovEodhdFailureResponse:
    session: str
    source_url: str
    retrieved_at: str
    http_status: int
    content: bytes
    content_type: str

    @property
    def source_hash(self) -> str:
        return sha256_bytes(self.content)


def _validate_cov_eodhd_symbol_failure(
    response: CovEodhdFailureResponse,
) -> SourceArtifact:
    if response.session != COV_EODHD_FAILURE_SESSION:
        raise ValueError("COV EODHD symbol failure cache has an unexpected session.")
    if response.source_url != COV_EODHD_SYMBOL_FAILURE_URL:
        raise ValueError("COV EODHD symbol failure URL is not exact.")
    if response.http_status != 200 or "json" not in response.content_type.lower():
        raise ValueError("COV EODHD symbol failure response metadata is invalid.")
    if response.source_hash != COV_EODHD_SYMBOL_FAILURE_SHA256:
        raise ValueError("COV EODHD symbol failure SHA-256 is unexpected.")
    try:
        payload = json.loads(response.content)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("COV EODHD symbol failure is invalid JSON.") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("COV EODHD symbol failure must contain exactly one row.")
    row = payload[0]
    if str(row.get("code") or "").upper() != "COV":
        raise ValueError("COV EODHD symbol failure returned an unexpected code.")
    if str(row.get("date") or "") != COV_EODHD_SYMBOL_FAILURE_RETURNED_DATE:
        raise ValueError("COV EODHD symbol failure no longer proves the stale date.")
    if str(row.get("date") or "") == COV_EODHD_FAILURE_SESSION:
        raise ValueError("COV EODHD symbol failure unexpectedly became usable history.")
    return SourceArtifact(
        source="eodhd_cov_symbol_filter_failure",
        source_url=response.source_url,
        retrieved_at=response.retrieved_at,
        content=response.content,
        content_type=response.content_type,
    )


def _read_cov_eodhd_symbol_failure_cache(root: Path) -> SourceArtifact:
    path = Path(root) / "state/eodhd-cov-bulk/2015-01-02.json.gz"
    if not path.is_file():
        raise FileNotFoundError(
            "The preserved COV symbol-filter EODHD failure cache is missing."
        )
    wrapper = json.loads(gzip.decompress(path.read_bytes()))
    payload_bytes = base64.b64decode(wrapper["payload_base64"], validate=True)
    if sha256_bytes(payload_bytes) != str(wrapper.get("payload_sha256") or ""):
        raise ValueError("COV symbol-filter EODHD cache envelope hash mismatch.")
    payload = json.loads(payload_bytes)
    content = base64.b64decode(payload["content_base64"], validate=True)
    if sha256_bytes(content) != str(payload.get("content_sha256") or ""):
        raise ValueError("COV symbol-filter EODHD raw hash mismatch.")
    if int(payload.get("billing_units") or 0) != 101:
        raise ValueError("COV symbol-filter EODHD failure must record 101 billed units.")
    return _validate_cov_eodhd_symbol_failure(
        CovEodhdFailureResponse(
            session=str(payload["session"]),
            source_url=str(payload["source_url"]),
            retrieved_at=str(payload["retrieved_at"]),
            http_status=int(payload["http_status"]),
            content=content,
            content_type=str(payload["content_type"]),
        )
    )


def _validate_cov_eodhd_full_us_failure(content: bytes) -> None:
    if sha256_bytes(content) != COV_EODHD_FULL_US_FAILURE_SHA256:
        raise ValueError("COV full-US EODHD failure SHA-256 is unexpected.")
    try:
        payload = json.loads(content)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("COV full-US EODHD failure is invalid JSON.") from exc
    if not isinstance(payload, list) or len(payload) != COV_EODHD_FULL_US_FAILURE_ROWS:
        raise ValueError(
            "COV full-US EODHD failure row count is not exactly 42,096."
        )
    if any(
        isinstance(row, dict)
        and str(row.get("code") or "").upper() in {"COV", "COV.US"}
        for row in payload
    ):
        raise ValueError("COV unexpectedly appears in the full-US EODHD failure response.")


def _cov_eodhd_full_us_failure_cache_path(root: Path) -> Path:
    return Path(root) / "state/eodhd-cov-failures/full-us-2015-01-02.json.gz"


def _load_or_import_cov_eodhd_full_us_failure(
    root: Path,
    import_path: str | Path | None,
) -> SourceArtifact:
    cache_path = _cov_eodhd_full_us_failure_cache_path(root)
    imported = Path(import_path).expanduser() if str(import_path or "").strip() else None
    imported_content = None
    if imported is not None:
        if not imported.is_file():
            raise FileNotFoundError(
                f"Explicit COV full-US EODHD failure response is missing: {imported}"
            )
        imported_content = imported.read_bytes()
        _validate_cov_eodhd_full_us_failure(imported_content)
    if cache_path.is_file():
        wrapper = json.loads(gzip.decompress(cache_path.read_bytes()))
        payload_bytes = base64.b64decode(wrapper["payload_base64"], validate=True)
        if sha256_bytes(payload_bytes) != str(wrapper.get("payload_sha256") or ""):
            raise ValueError("COV full-US EODHD failure cache envelope hash mismatch.")
        payload = json.loads(payload_bytes)
        content = base64.b64decode(payload["content_base64"], validate=True)
        _validate_cov_eodhd_full_us_failure(content)
        if (
            str(payload.get("source_url") or "") != COV_EODHD_FULL_US_FAILURE_URL
            or str(payload.get("content_sha256") or "")
            != COV_EODHD_FULL_US_FAILURE_SHA256
            or str(payload.get("content_type") or "").lower()
            != "application/json"
            or int(payload.get("billed_units") or 0) != 100
            or int(payload.get("row_count") or 0)
            != COV_EODHD_FULL_US_FAILURE_ROWS
        ):
            raise ValueError("COV full-US EODHD failure cache metadata is invalid.")
        if imported_content is not None and imported_content != content:
            raise ValueError("Explicit COV full-US failure file differs from immutable cache.")
        retrieved_at = str(payload["imported_at"])
    else:
        if imported_content is None:
            raise FileNotFoundError(
                "COV full-US EODHD failure is not cached. Supply the explicit "
                "--cov-eodhd-full-us-failure-response path; no network fallback exists."
            )
        content = imported_content
        retrieved_at = utc_now_iso()
        payload_bytes = json.dumps(
            {
                "billed_units": 100,
                "content_base64": base64.b64encode(content).decode("ascii"),
                "content_sha256": COV_EODHD_FULL_US_FAILURE_SHA256,
                "content_type": "application/json",
                "imported_at": retrieved_at,
                "row_count": COV_EODHD_FULL_US_FAILURE_ROWS,
                "source_url": COV_EODHD_FULL_US_FAILURE_URL,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        wrapper = json.dumps(
            {
                "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
                "payload_sha256": sha256_bytes(payload_bytes),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        write_atomic(cache_path, gzip.compress(wrapper, mtime=0))
    return SourceArtifact(
        source="eodhd_cov_full_us_failure",
        source_url=COV_EODHD_FULL_US_FAILURE_URL,
        retrieved_at=retrieved_at,
        content=content,
        content_type="application/json",
    )


@dataclass(frozen=True)
class CovWikiCachedResponse:
    role: str
    source_url: str
    retrieved_at: str
    http_status: int
    content: bytes
    content_type: str

    @property
    def source_hash(self) -> str:
        return sha256_bytes(self.content)


def _validate_cov_wiki_url(role: str, url: str) -> None:
    expected = COV_WIKI_URLS.get(role)
    if expected is None or url != expected:
        raise ValueError(f"COV Quandl WIKI {role} URL is not the pinned URL.")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "raw.githubusercontent.com"
        or COV_WIKI_COMMIT not in parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"COV Quandl WIKI {role} URL lost its pinned commit.")


def _validate_cov_wiki_hash(role: str, content: bytes) -> None:
    expected = COV_WIKI_SHA256.get(role)
    if expected is None or sha256_bytes(content) != expected:
        raise ValueError(f"COV Quandl WIKI {role} SHA-256 is unexpected.")


def _normalized_source_text(value: str) -> str:
    output: list[str] = []
    previous_separator = False
    for character in value.lower():
        if character.isalnum():
            output.append(character)
            previous_separator = False
        elif not previous_separator:
            output.append("_")
            previous_separator = True
    return "".join(output).strip("_")


def _validate_cov_wiki_generation_evidence(
    readme: SourceArtifact,
    generate_py: SourceArtifact,
) -> None:
    _validate_cov_wiki_url("readme", readme.source_url)
    _validate_cov_wiki_url("generate_py", generate_py.source_url)
    _validate_cov_wiki_hash("readme", readme.content)
    _validate_cov_wiki_hash("generate_py", generate_py.content)
    try:
        readme_text = readme.content.decode("utf-8")
        generate_text = generate_py.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("COV Quandl WIKI provenance text is not UTF-8.") from exc
    readme_lower = readme_text.lower()
    if "quandl" not in readme_lower or "wiki" not in readme_lower:
        raise ValueError("Pinned README does not identify Quandl WIKI provenance.")
    try:
        tree = ast.parse(generate_text)
    except SyntaxError as exc:
        raise ValueError("Pinned generate.py is not valid Python source.") from exc
    quandl_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "quandl_data"
    ]
    if len(quandl_functions) != 1:
        raise ValueError("Pinned generate.py must define exactly one quandl_data function.")
    quandl_function = quandl_functions[0]
    quandl_text = _normalized_source_text(ast.unparse(quandl_function))
    wiki_membership_checks = sum(
        isinstance(node, ast.Compare)
        and any(isinstance(operator, ast.In) for operator in node.ops)
        and "wiki" in _normalized_source_text(ast.unparse(node))
        for node in ast.walk(quandl_function)
    )
    returns_none = any(
        isinstance(node, ast.Return)
        and (node.value is None or isinstance(node.value, ast.Constant) and node.value.value is None)
        for node in ast.walk(quandl_function)
    )
    required_adjusted = {
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_volume",
    }
    quandl_branch_valid = (
        wiki_membership_checks >= 2
        and "fix_ticker" in quandl_text
        and returns_none
        and all(value in quandl_text for value in required_adjusted)
        and "yahoo" not in quandl_text
    )
    yahoo_fallback_valid = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_text = _normalized_source_text(ast.unparse(node.test))
        body_text = _normalized_source_text(
            "\n".join(ast.unparse(item) for item in node.body)
        )
        if "df_is_none" in test_text and "yahoo_data" in body_text:
            yahoo_fallback_valid = True
            break
    module_text = _normalized_source_text(generate_text)
    if not (
        quandl_branch_valid
        and yahoo_fallback_valid
        and "df_quandl_data" in module_text
    ):
        raise ValueError(
            "Pinned generate.py no longer proves two-stage WIKI membership, "
            "adjusted OHLCV selection, and Yahoo only after quandl_data returns None."
        )


def _parse_cov_wiki_csv(artifact: SourceArtifact) -> pd.DataFrame:
    _validate_cov_wiki_url("cov_csv", artifact.source_url)
    _validate_cov_wiki_hash("cov_csv", artifact.content)
    if "csv" not in artifact.content_type.lower() and not artifact.source_url.endswith(
        ".csv"
    ):
        raise ValueError("Pinned COV Quandl WIKI response is not CSV.")
    try:
        raw = pd.read_csv(io.BytesIO(artifact.content))
    except Exception as exc:
        raise ValueError("Pinned COV Quandl WIKI CSV cannot be parsed.") from exc
    normalized = {
        _normalized_source_text(str(column)): column for column in raw.columns
    }
    date_column = normalized.get("date")
    if date_column is None:
        unnamed = [
            column
            for key, column in normalized.items()
            if key.startswith("unnamed")
        ]
        date_column = unnamed[0] if len(unnamed) == 1 else None
    if date_column is None:
        raise ValueError("Pinned COV Quandl WIKI CSV has no date column.")
    adjusted = {
        field: normalized.get(f"adj_{field}")
        for field in ("open", "high", "low", "close", "volume")
    }
    unadjusted = {
        field: normalized.get(field)
        for field in ("open", "high", "low", "close", "volume")
    }
    columns = adjusted if all(adjusted.values()) else unadjusted
    if not all(columns.values()):
        raise ValueError("Pinned COV Quandl WIKI CSV has no complete OHLCV columns.")
    frame = pd.DataFrame(
        {
            "session": pd.to_datetime(raw[date_column], errors="coerce"),
            **{
                field: pd.to_numeric(raw[column], errors="coerce")
                for field, column in columns.items()
            },
        }
    )
    if frame.isna().any().any():
        raise ValueError("Pinned COV Quandl WIKI CSV contains invalid OHLCV rows.")
    frame["session"] = frame["session"].dt.date.astype(str)
    frame = frame.loc[frame["session"].isin(COV_EXPECTED_SESSIONS)].copy()
    if frame["session"].duplicated().any():
        raise ValueError("Pinned COV Quandl WIKI CSV duplicates a session.")
    frame = frame.sort_values("session", kind="stable").reset_index(drop=True)
    if tuple(frame["session"]) != COV_EXPECTED_SESSIONS:
        raise ValueError("Pinned COV Quandl WIKI CSV is not the exact 16-session window.")
    for field in ("open", "high", "low", "close"):
        if not frame[field].map(lambda value: math.isfinite(float(value)) and value > 0).all():
            raise ValueError(f"Pinned COV Quandl WIKI CSV has invalid {field}.")
    if not frame["volume"].map(
        lambda value: math.isfinite(float(value)) and value >= 0
    ).all():
        raise ValueError("Pinned COV Quandl WIKI CSV has invalid volume.")
    frame["security_id"] = COV_SPEC.security_id
    frame["currency"] = "USD"
    frame["source"] = "quandl_wiki_pinned_csv"
    frame["source_url"] = artifact.source_url
    frame["retrieved_at"] = artifact.retrieved_at
    frame["source_hash"] = artifact.source_hash
    return frame.loc[:, list(dataset_spec("daily_price_raw").required_columns)].copy()


class CappedCovQuandlWikiSource:
    """Cache three pinned GitHub artifacts with no retries or mutable URLs."""

    def __init__(self, root: Path, *, allow_http: bool, session=None):
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.session = session
        self._attempt_count = 0
        self._attempted_roles: set[str] = set()

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    def _path(self, role: str) -> Path:
        return self.root / f"{role}.json.gz"

    def _read(self, role: str) -> CovWikiCachedResponse | None:
        path = self._path(role)
        if not path.is_file():
            return None
        wrapper = json.loads(gzip.decompress(path.read_bytes()))
        payload_bytes = base64.b64decode(wrapper["payload_base64"], validate=True)
        if sha256_bytes(payload_bytes) != str(wrapper.get("payload_sha256") or ""):
            raise ValueError(f"COV Quandl WIKI immutable cache hash mismatch: {path}")
        payload = json.loads(payload_bytes)
        if str(payload.get("role") or "") != role:
            raise ValueError(f"COV Quandl WIKI immutable cache role mismatch: {path}")
        content = base64.b64decode(payload["content_base64"], validate=True)
        if sha256_bytes(content) != str(payload.get("content_sha256") or ""):
            raise ValueError(f"COV Quandl WIKI raw content hash mismatch: {path}")
        response = CovWikiCachedResponse(
            role=role,
            source_url=str(payload["source_url"]),
            retrieved_at=str(payload["retrieved_at"]),
            http_status=int(payload["http_status"]),
            content=content,
            content_type=str(payload["content_type"]),
        )
        _validate_cov_wiki_url(role, response.source_url)
        _validate_cov_wiki_hash(role, response.content)
        return response

    def _write(self, response: CovWikiCachedResponse) -> None:
        payload_bytes = json.dumps(
            {
                "content_base64": base64.b64encode(response.content).decode("ascii"),
                "content_sha256": response.source_hash,
                "content_type": response.content_type,
                "http_status": response.http_status,
                "retrieved_at": response.retrieved_at,
                "role": response.role,
                "source_url": response.source_url,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        wrapper = json.dumps(
            {
                "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
                "payload_sha256": sha256_bytes(payload_bytes),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        compressed = gzip.compress(wrapper, mtime=0)
        path = self._path(response.role)
        if path.is_file():
            if path.read_bytes() != compressed:
                raise ValueError(
                    f"COV Quandl WIKI immutable cache already differs: {path}"
                )
            return
        write_atomic(path, compressed)

    def _fetch_one(self, role: str) -> CovWikiCachedResponse:
        if role in self._attempted_roles:
            raise RuntimeError(
                f"COV Quandl WIKI {role} already used its one-attempt cap."
            )
        if self._attempt_count >= COV_WIKI_MAX_HTTP_ATTEMPTS:
            raise RuntimeError("COV Quandl WIKI request cap reached before HTTP.")
        if self.session is None:
            import requests

            self.session = requests.Session()
        self._attempted_roles.add(role)
        self._attempt_count += 1
        url = COV_WIKI_URLS[role]
        try:
            http = self.session.get(url, timeout=120)
        except Exception as exc:
            raise RuntimeError(
                f"COV Quandl WIKI single attempt failed for {role}: "
                f"{type(exc).__name__}"
            ) from None
        response = CovWikiCachedResponse(
            role=role,
            source_url=url,
            retrieved_at=utc_now_iso(),
            http_status=int(getattr(http, "status_code", 0)),
            content=bytes(http.content),
            content_type=str(
                getattr(http, "headers", {}).get(
                    "Content-Type", "application/octet-stream"
                )
            ),
        )
        self._write(response)
        _validate_cov_wiki_hash(role, response.content)
        return response

    def fetch(self) -> CovWikiEvidenceBundle:
        cached = {
            role: self._read(role)
            for role in COV_WIKI_URLS
            if self._path(role).is_file()
        }
        missing = tuple(role for role in COV_WIKI_URLS if role not in cached)
        if missing and not self.allow_http:
            raise FileNotFoundError(
                "COV Quandl WIKI immutable cache is incomplete; explicitly allow "
                f"the three pinned requests. Missing: {list(missing)}."
            )
        artifacts: dict[str, SourceArtifact] = {}
        for role, url in COV_WIKI_URLS.items():
            response = cached.get(role)
            if response is None:
                response = self._fetch_one(role)
            if response.http_status != 200:
                raise RuntimeError(
                    f"COV Quandl WIKI {role} returned HTTP {response.http_status}."
                )
            _validate_cov_wiki_url(role, response.source_url)
            content_type = response.content_type
            if role == "cov_csv" and "html" in content_type.lower():
                raise ValueError("Pinned COV Quandl WIKI CSV returned HTML.")
            artifacts[role] = SourceArtifact(
                source=f"cov_quandl_wiki_{role}",
                source_url=url,
                retrieved_at=response.retrieved_at,
                content=response.content,
                content_type=content_type,
            )
        _validate_cov_wiki_generation_evidence(
            artifacts["readme"], artifacts["generate_py"]
        )
        prices = _parse_cov_wiki_csv(artifacts["cov_csv"])
        return CovWikiEvidenceBundle(
            prices=prices,
            artifacts=tuple(artifacts[role] for role in COV_WIKI_URLS),
            http_attempts=self.attempt_count,
        )


def _validate_cov_directindex_url(role: str, url: str) -> None:
    expected = COV_DIRECTINDEX_URLS.get(role)
    if expected is None or url != expected:
        raise ValueError(f"COV DirectIndex {role} URL is not the pinned URL.")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "raw.githubusercontent.com"
        or COV_DIRECTINDEX_COMMIT not in parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"COV DirectIndex {role} URL lost its pinned commit.")


def _validate_cov_directindex_hash(role: str, content: bytes) -> None:
    expected = COV_DIRECTINDEX_SHA256.get(role)
    if expected is None or sha256_bytes(content) != expected:
        raise ValueError(f"COV DirectIndex {role} SHA-256 is unexpected.")


def _parse_cov_directindex_csv(artifact: SourceArtifact) -> pd.DataFrame:
    _validate_cov_directindex_url("cov_csv", artifact.source_url)
    _validate_cov_directindex_hash("cov_csv", artifact.content)
    try:
        raw = pd.read_csv(io.BytesIO(artifact.content))
    except Exception as exc:
        raise ValueError("Pinned COV DirectIndex CSV cannot be parsed.") from exc
    normalized = {
        _normalized_source_text(str(column)): column for column in raw.columns
    }
    date_column = normalized.get("date")
    columns = {
        field: normalized.get(field)
        for field in ("open", "high", "low", "close", "volume")
    }
    if date_column is None or not all(columns.values()):
        raise ValueError("Pinned COV DirectIndex CSV has no complete raw OHLCV columns.")
    frame = pd.DataFrame(
        {
            "session": pd.to_datetime(raw[date_column], errors="coerce"),
            **{
                field: pd.to_numeric(raw[column], errors="coerce")
                for field, column in columns.items()
            },
        }
    )
    frame["session"] = frame["session"].dt.date.astype(str)
    frame = frame.loc[frame["session"].isin(COV_EXPECTED_SESSIONS)].copy()
    if frame.isna().any().any() or frame["session"].duplicated().any():
        raise ValueError("Pinned COV DirectIndex CSV has invalid or duplicate rows.")
    frame = frame.sort_values("session", kind="stable").reset_index(drop=True)
    if tuple(frame["session"]) != COV_EXPECTED_SESSIONS:
        raise ValueError("Pinned COV DirectIndex CSV is not the exact 16-session window.")
    for field in ("open", "high", "low", "close"):
        if not frame[field].map(
            lambda value: math.isfinite(float(value)) and value > 0
        ).all():
            raise ValueError(f"Pinned COV DirectIndex CSV has invalid {field}.")
    if not frame["volume"].map(
        lambda value: math.isfinite(float(value)) and value >= 0
    ).all():
        raise ValueError("Pinned COV DirectIndex CSV has invalid volume.")
    frame["security_id"] = COV_SPEC.security_id
    frame["currency"] = "USD"
    frame["source"] = "directindex_pinned_csv"
    frame["source_url"] = artifact.source_url
    frame["retrieved_at"] = artifact.retrieved_at
    frame["source_hash"] = artifact.source_hash
    return frame.loc[:, list(dataset_spec("daily_price_raw").required_columns)].copy()


class CappedCovDirectIndexSource:
    """Cache the three commit-pinned DirectIndex artifacts without retries."""

    def __init__(self, root: Path, *, allow_http: bool, session=None):
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.session = session
        self._attempt_count = 0
        self._attempted_roles: set[str] = set()

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    def _path(self, role: str) -> Path:
        return self.root / f"{role}.json.gz"

    def _read(self, role: str) -> CovWikiCachedResponse | None:
        path = self._path(role)
        if not path.is_file():
            return None
        wrapper = json.loads(gzip.decompress(path.read_bytes()))
        payload_bytes = base64.b64decode(wrapper["payload_base64"], validate=True)
        if sha256_bytes(payload_bytes) != str(wrapper.get("payload_sha256") or ""):
            raise ValueError(f"COV DirectIndex immutable cache hash mismatch: {path}")
        payload = json.loads(payload_bytes)
        if str(payload.get("role") or "") != role:
            raise ValueError(f"COV DirectIndex immutable cache role mismatch: {path}")
        content = base64.b64decode(payload["content_base64"], validate=True)
        if sha256_bytes(content) != str(payload.get("content_sha256") or ""):
            raise ValueError(f"COV DirectIndex raw content hash mismatch: {path}")
        response = CovWikiCachedResponse(
            role=role,
            source_url=str(payload["source_url"]),
            retrieved_at=str(payload["retrieved_at"]),
            http_status=int(payload["http_status"]),
            content=content,
            content_type=str(payload["content_type"]),
        )
        _validate_cov_directindex_url(role, response.source_url)
        _validate_cov_directindex_hash(role, response.content)
        return response

    def _write(self, response: CovWikiCachedResponse) -> None:
        payload_bytes = json.dumps(
            {
                "content_base64": base64.b64encode(response.content).decode("ascii"),
                "content_sha256": response.source_hash,
                "content_type": response.content_type,
                "http_status": response.http_status,
                "retrieved_at": response.retrieved_at,
                "role": response.role,
                "source_url": response.source_url,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        wrapper = json.dumps(
            {
                "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
                "payload_sha256": sha256_bytes(payload_bytes),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        compressed = gzip.compress(wrapper, mtime=0)
        path = self._path(response.role)
        if path.is_file():
            if path.read_bytes() != compressed:
                raise ValueError(f"COV DirectIndex immutable cache already differs: {path}")
            return
        write_atomic(path, compressed)

    def _fetch_one(self, role: str) -> CovWikiCachedResponse:
        if role in self._attempted_roles:
            raise RuntimeError(f"COV DirectIndex {role} already used its one-attempt cap.")
        if self._attempt_count >= COV_DIRECTINDEX_MAX_HTTP_ATTEMPTS:
            raise RuntimeError("COV DirectIndex request cap reached before HTTP.")
        if self.session is None:
            import requests

            self.session = requests.Session()
        self._attempted_roles.add(role)
        self._attempt_count += 1
        url = COV_DIRECTINDEX_URLS[role]
        try:
            http = self.session.get(url, timeout=120)
        except Exception as exc:
            raise RuntimeError(
                f"COV DirectIndex single attempt failed for {role}: "
                f"{type(exc).__name__}"
            ) from None
        response = CovWikiCachedResponse(
            role=role,
            source_url=url,
            retrieved_at=utc_now_iso(),
            http_status=int(getattr(http, "status_code", 0)),
            content=bytes(http.content),
            content_type=str(
                getattr(http, "headers", {}).get(
                    "Content-Type", "application/octet-stream"
                )
            ),
        )
        self._write(response)
        _validate_cov_directindex_hash(role, response.content)
        return response

    def fetch(self) -> CovDirectIndexBundle:
        cached = {
            role: self._read(role)
            for role in COV_DIRECTINDEX_URLS
            if self._path(role).is_file()
        }
        missing = tuple(role for role in COV_DIRECTINDEX_URLS if role not in cached)
        if missing and not self.allow_http:
            raise FileNotFoundError(
                "COV DirectIndex immutable cache is incomplete; explicitly allow "
                f"the three pinned requests. Missing: {list(missing)}."
            )
        artifacts: dict[str, SourceArtifact] = {}
        for role, url in COV_DIRECTINDEX_URLS.items():
            response = cached.get(role)
            if response is None:
                response = self._fetch_one(role)
            if response.http_status != 200:
                raise RuntimeError(
                    f"COV DirectIndex {role} returned HTTP {response.http_status}."
                )
            _validate_cov_directindex_url(role, response.source_url)
            artifacts[role] = SourceArtifact(
                source=f"cov_directindex_{role}",
                source_url=url,
                retrieved_at=response.retrieved_at,
                content=response.content,
                content_type=response.content_type,
            )
        prices = _parse_cov_directindex_csv(artifacts["cov_csv"])
        return CovDirectIndexBundle(
            prices=prices,
            artifacts=tuple(artifacts[role] for role in COV_DIRECTINDEX_URLS),
            http_attempts=self.attempt_count,
        )


def cross_validate_cov_directindex_with_wiki(
    direct: CovDirectIndexBundle,
    wiki: CovWikiEvidenceBundle,
) -> dict[str, Any]:
    """Require all 16 DirectIndex raw rows to agree with independent WIKI."""

    if len(direct.artifacts) != COV_DIRECTINDEX_MAX_HTTP_ATTEMPTS:
        raise ValueError("COV DirectIndex evidence must contain exactly three artifacts.")
    direct_roles = {
        item.source.removeprefix("cov_directindex_"): item
        for item in direct.artifacts
    }
    if set(direct_roles) != set(COV_DIRECTINDEX_URLS):
        raise ValueError("COV DirectIndex evidence roles are incomplete.")
    parsed_direct = _parse_cov_directindex_csv(direct_roles["cov_csv"])
    if not parsed_direct.equals(direct.prices.reset_index(drop=True)):
        raise ValueError("COV DirectIndex normalized rows differ from pinned raw CSV.")
    if len(wiki.artifacts) != COV_WIKI_MAX_HTTP_ATTEMPTS:
        raise ValueError("COV Quandl WIKI evidence must contain exactly three artifacts.")
    roles = {
        item.source.removeprefix("cov_quandl_wiki_"): item
        for item in wiki.artifacts
    }
    if set(roles) != set(COV_WIKI_URLS):
        raise ValueError("COV Quandl WIKI evidence roles are incomplete.")
    _validate_cov_wiki_generation_evidence(roles["readme"], roles["generate_py"])
    parsed_wiki = _parse_cov_wiki_csv(roles["cov_csv"])
    if not parsed_wiki.equals(wiki.prices.reset_index(drop=True)):
        raise ValueError("COV Quandl WIKI normalized rows differ from pinned raw CSV.")
    left = direct.prices.copy()
    right = wiki.prices.copy()
    if set(left["source"].astype(str)) != {"directindex_pinned_csv"}:
        raise ValueError("COV primary rows are not pinned DirectIndex raw data.")
    if set(right["source"].astype(str)) != {"quandl_wiki_pinned_csv"}:
        raise ValueError("COV independent rows are not pinned Quandl WIKI data.")
    joined = left.merge(
        right,
        on="session",
        suffixes=("_direct", "_wiki"),
        validate="one_to_one",
    ).sort_values("session", kind="stable")
    if tuple(joined["session"].astype(str)) != COV_EXPECTED_SESSIONS:
        raise ValueError("COV cross-validation does not cover all 16 XNYS sessions.")
    maximum_absolute_delta: dict[str, float] = {}
    for field in ("open", "high", "low", "close"):
        left_values = pd.to_numeric(joined[f"{field}_direct"], errors="coerce")
        right_values = pd.to_numeric(joined[f"{field}_wiki"], errors="coerce")
        deltas = (left_values - right_values).abs()
        if (
            left_values.isna().any()
            or right_values.isna().any()
            or (deltas > COV_CROSS_OHLC_ABS_TOLERANCE).any()
        ):
            bad = joined.loc[
                deltas > COV_CROSS_OHLC_ABS_TOLERANCE, "session"
            ].astype(str).tolist()
            raise ValueError(f"COV {field} cross-validation failed: {bad}.")
        maximum_absolute_delta[field] = float(deltas.max())
    direct_volume = pd.to_numeric(joined["volume_direct"], errors="coerce")
    wiki_volume = pd.to_numeric(joined["volume_wiki"], errors="coerce")
    volume_delta = (direct_volume - wiki_volume).abs()
    if (
        direct_volume.isna().any()
        or wiki_volume.isna().any()
        or (volume_delta != 0).any()
    ):
        bad = joined.loc[volume_delta != 0, "session"].astype(str).tolist()
        raise ValueError(f"COV volume cross-validation failed: {bad}.")
    maximum_absolute_delta["volume"] = float(volume_delta.max())
    return {
        "status": "passed",
        "cov_cross_validated": True,
        "sessions_compared": len(joined),
        "primary_source": "directindex_pinned_csv",
        "primary_commit": COV_DIRECTINDEX_COMMIT,
        "independent_source": "quandl_wiki_pinned_csv",
        "independent_commit": COV_WIKI_COMMIT,
        "maximum_absolute_delta": maximum_absolute_delta,
        "ohlc_absolute_tolerance": COV_CROSS_OHLC_ABS_TOLERANCE,
        "volume_tolerance": 0,
    }


def _load_evidence_report(path: Path) -> LoadedEvidenceReport:
    if not path.is_file():
        raise FileNotFoundError(f"Lifecycle evidence report is missing: {path}")
    content = path.read_bytes()
    value = json.loads(content)
    if not isinstance(value.get("records"), (dict, list)):
        raise ValueError("Lifecycle evidence report has no records collection.")
    return LoadedEvidenceReport(
        data=value,
        artifact=SourceArtifact(
            source="sec_lifecycle_evidence_report",
            source_url=path.resolve().as_uri(),
            retrieved_at=utc_now_iso(),
            content=content,
            content_type="application/json",
        ),
    )


def _request_archive_artifact(artifact: SourceArtifact) -> SourceArtifact:
    """Make request provenance unique while retaining the exact raw payload."""

    envelope = json.dumps(
        {
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
            "content_sha256": artifact.source_hash,
            "content_type": artifact.content_type,
            "source": artifact.source,
            "source_url": artifact.source_url,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SourceArtifact(
        source=artifact.source,
        source_url=artifact.source_url,
        retrieved_at=artifact.retrieved_at,
        content=envelope,
        content_type="application/vnd.supertrendquant.source-envelope+json",
    )


def _cov_official_evidence_artifact(*, retrieved_at: str) -> SourceArtifact:
    """Archive the reviewed facts without pretending to cache the SEC filing."""

    content = json.dumps(
        {
            "cash_amount": 35.19,
            "completion_date": COV_LAST_TRADING_DATE,
            "issuer_release_url": COV_ISSUER_RELEASE_URL,
            "last_trading_date": COV_LAST_TRADING_DATE,
            "new_symbol": "MDT",
            "ratio": 0.956,
            "sec_completion_url": COV_SEC_COMPLETION_URL,
            "verification": (
                "Covidien's SEC 8-K states the transaction completed on "
                "2015-01-26, each COV share converted into $35.19 cash plus "
                "0.956 MDT share, and COV trading was suspended before the "
                "2015-01-27 open. The SEC-filed issuer release states COV "
                "ceased trading at the 2015-01-26 close."
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SourceArtifact(
        source="cov_official_evidence_manifest",
        source_url=COV_SEC_COMPLETION_URL,
        retrieved_at=retrieved_at,
        content=content,
        content_type="application/json",
    )


def _load_enb_official_evidence(cache_root: Path) -> SourceArtifact:
    """Load the exact SEC-filed completion release without a network fallback.

    The identity-repair collector already archives this immutable EDGAR exhibit.
    Reusing that cache avoids a duplicate SEC request while still binding this
    supplement to the reviewed URL, raw SHA-256, schema, and completion text.
    """

    cache_key = sha256_bytes(ENB_SEC_COMPLETION_URL.encode("utf-8"))
    path = (
        Path(cache_root)
        / "state/official-us-index-identity"
        / f"{cache_key}.json.gz"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Exact Spectra/Enbridge SEC completion evidence is not cached: "
            f"{path}. Run the identity evidence collection first."
        )
    try:
        payload = json.loads(gzip.decompress(path.read_bytes()))
        content = base64.b64decode(payload["content_base64"], validate=True)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("Spectra/Enbridge official evidence cache is unreadable.") from exc
    if payload.get("schema") != ENB_OFFICIAL_EVIDENCE_SCHEMA:
        raise ValueError("Spectra/Enbridge official evidence schema is unexpected.")
    if str(payload.get("source_url") or "") != ENB_SEC_COMPLETION_URL:
        raise ValueError("Spectra/Enbridge official evidence URL is not exact.")
    if str(payload.get("source_hash") or "").lower() != ENB_SEC_COMPLETION_SHA256:
        raise ValueError("Spectra/Enbridge official evidence envelope hash is unexpected.")
    if sha256_bytes(content) != ENB_SEC_COMPLETION_SHA256:
        raise ValueError("Spectra/Enbridge official evidence raw SHA-256 is unexpected.")
    content_type = str(payload.get("content_type") or "").lower()
    if "html" not in content_type:
        raise ValueError("Spectra/Enbridge official evidence is not HTML.")
    normalized = re.sub(
        r"\s+",
        " ",
        html.unescape(
            re.sub(r"<[^>]+>", " ", content.decode("utf-8", errors="replace"))
        ),
    ).casefold()
    required_phrases = (
        "enbridge and spectra energy complete merger",
        "february 27, 2017",
        "stock-for-stock merger transaction",
        "trading in shares of spectra energy common stock",
        "will be suspended effective as of the opening of trading today",
        "will be delisted from the nyse",
        "under the symbol",
        "enb",
    )
    missing = tuple(phrase for phrase in required_phrases if phrase not in normalized)
    if missing:
        raise ValueError(
            "Spectra/Enbridge official evidence is missing reviewed text: "
            f"{missing}."
        )
    return SourceArtifact(
        source="sec_edgar_filing",
        source_url=ENB_SEC_COMPLETION_URL,
        retrieved_at=str(payload.get("retrieved_at") or ""),
        content=content,
        content_type=str(payload["content_type"]),
    )


def _cov_cross_validation_evidence_artifact(
    report: dict[str, Any],
    *,
    direct_artifacts: tuple[SourceArtifact, ...],
    wiki_artifacts: tuple[SourceArtifact, ...],
    retrieved_at: str,
) -> SourceArtifact:
    if report.get("cov_cross_validated") is not True:
        raise ValueError("Cannot archive a COV cross-validation report that did not pass.")
    content = json.dumps(
        {
            "report": report,
            "directindex_raw": [
                {
                    "source_hash": item.source_hash,
                    "source_url": item.source_url,
                }
                for item in direct_artifacts
            ],
            "quandl_wiki_raw": [
                {
                    "source_hash": item.source_hash,
                    "source_url": item.source_url,
                }
                for item in wiki_artifacts
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SourceArtifact(
        source="cov_directindex_quandl_wiki_cross_validation",
        source_url=COV_DIRECTINDEX_URLS["cov_csv"],
        retrieved_at=retrieved_at,
        content=content,
        content_type="application/json",
    )


def _cov_eodhd_failure_manifest_artifact(
    symbol_artifact: SourceArtifact,
    full_us_artifact: SourceArtifact,
    *,
    retrieved_at: str,
) -> SourceArtifact:
    """Archive why COV must never spend another EODHD request."""

    if (
        symbol_artifact.source != "eodhd_cov_symbol_filter_failure"
        or symbol_artifact.source_url != COV_EODHD_SYMBOL_FAILURE_URL
        or symbol_artifact.source_hash != COV_EODHD_SYMBOL_FAILURE_SHA256
    ):
        raise ValueError("COV symbol-filter EODHD failure evidence is not exact.")
    _validate_cov_eodhd_symbol_failure(
        CovEodhdFailureResponse(
            session=COV_EODHD_FAILURE_SESSION,
            source_url=symbol_artifact.source_url,
            retrieved_at=symbol_artifact.retrieved_at,
            http_status=200,
            content=symbol_artifact.content,
            content_type=symbol_artifact.content_type,
        )
    )
    if (
        full_us_artifact.source != "eodhd_cov_full_us_failure"
        or full_us_artifact.source_url != COV_EODHD_FULL_US_FAILURE_URL
        or full_us_artifact.source_hash != COV_EODHD_FULL_US_FAILURE_SHA256
    ):
        raise ValueError("COV full-US EODHD failure evidence is not exact.")
    _validate_cov_eodhd_full_us_failure(full_us_artifact.content)
    content = json.dumps(
        {
            "status": "known_unavailable_never_retry",
            "ledger_usage_after_failures": COV_EODHD_LEDGER_USAGE_AFTER_FAILURES,
            "symbol_filtered_request": {
                "requested_session": COV_EODHD_FAILURE_SESSION,
                "returned_session": COV_EODHD_SYMBOL_FAILURE_RETURNED_DATE,
                "billed_units": 101,
                "cache_path": "state/eodhd-cov-bulk/2015-01-02.json.gz",
                "source_hash": symbol_artifact.source_hash,
                "source_url": symbol_artifact.source_url,
            },
            "full_us_request": {
                "requested_session": COV_EODHD_FAILURE_SESSION,
                "row_count": COV_EODHD_FULL_US_FAILURE_ROWS,
                "contains_cov": False,
                "imported_raw_response": True,
                "billed_units": 100,
                "cache_path": "state/eodhd-cov-failures/full-us-2015-01-02.json.gz",
                "source_hash": full_us_artifact.source_hash,
                "source_url": full_us_artifact.source_url,
            },
            "policy": (
                "Both actual EODHD bulk paths failed for COV. The symbol-filtered "
                "path returned a stale 2012 row and the complete US response omitted "
                "COV, so this collector must never retry either path."
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SourceArtifact(
        source="cov_eodhd_known_unavailable_manifest",
        source_url=COV_EODHD_SYMBOL_FAILURE_URL,
        retrieved_at=retrieved_at,
        content=content,
        content_type="application/json",
    )


def _bundle_signature(release: DataRelease) -> dict[str, Any]:
    return {
        "release_version": release.version,
        "completed_session": release.completed_session,
        "fetch_start": FETCH_START,
        "successors": [
            {
                "security_id": spec.security_id,
                "provider_symbol": spec.provider_symbol,
            }
            for spec in SUCCESSOR_SPECS
        ],
    }


def _cov_bundle_signature(release: DataRelease) -> dict[str, Any]:
    return {
        "release_version": release.version,
        "completed_session": release.completed_session,
        "fetch_start": COV_FETCH_START,
        "fetch_end": COV_LAST_TRADING_DATE,
        "supplement": {
            "security_id": COV_SPEC.security_id,
            "provider_symbol": COV_SPEC.provider_symbol,
        },
    }


def _enb_bundle_signature(release: DataRelease) -> dict[str, Any]:
    return {
        "release_version": release.version,
        "completed_session": release.completed_session,
        "fetch_start": FETCH_START,
        "supplement": {
            "security_id": ENB_SPEC.security_id,
            "provider_symbol": ENB_SPEC.provider_symbol,
            "spectra_security_id": SPECTRA_SECURITY_ID,
            "effective_date": ENB_EFFECTIVE_DATE,
            "ratio": ENB_RATIO,
        },
    }


def _bundle_cache_path(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> Path:
    signature = json.dumps(
        _bundle_signature(release),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return (
        repository.root
        / "state/eodhd_lifecycle_successors"
        / f"{sha256_bytes(signature)}.json.gz"
    )


def _cov_bundle_cache_path(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> Path:
    signature = json.dumps(
        _cov_bundle_signature(release),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return (
        repository.root
        / "state/eodhd_lifecycle_successors_cov"
        / f"{sha256_bytes(signature)}.json.gz"
    )


def _enb_bundle_cache_path(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> Path:
    signature = json.dumps(
        _enb_bundle_signature(release),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return (
        repository.root
        / "state/eodhd_lifecycle_successors_enb"
        / f"{sha256_bytes(signature)}.json.gz"
    )


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _write_fetched_bundle_cache(
    path: Path,
    release: DataRelease,
    fetched,
    *,
    http_attempts: int,
    signature: dict[str, Any] | None = None,
) -> None:
    expected_signature = dict(signature or _bundle_signature(release))
    payload = {
        **expected_signature,
        "http_attempts": int(http_attempts),
        "prices": _frame_records(fetched.prices),
        "corporate_actions": _frame_records(fetched.corporate_actions),
        "missing_symbols": list(fetched.missing_symbols),
        "artifacts": [
            {
                "source": item.source,
                "source_url": item.source_url,
                "retrieved_at": item.retrieved_at,
                "content_type": item.content_type,
                "content_base64": base64.b64encode(item.content).decode("ascii"),
            }
            for item in fetched.artifacts
        ],
    }
    payload_bytes = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    wrapper = json.dumps(
        {
            "payload_sha256": sha256_bytes(payload_bytes),
            "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    write_atomic(path, gzip.compress(wrapper, mtime=0))


def _read_fetched_bundle_cache(
    path: Path,
    release: DataRelease,
    *,
    signature: dict[str, Any] | None = None,
) -> tuple[FetchedBundle, int] | None:
    if not path.is_file():
        return None
    wrapper = json.loads(gzip.decompress(path.read_bytes()))
    payload_bytes = base64.b64decode(wrapper["payload_base64"], validate=True)
    if sha256_bytes(payload_bytes) != str(wrapper.get("payload_sha256") or ""):
        raise ValueError(f"Fetched bundle cache hash mismatch: {path}")
    payload = json.loads(payload_bytes)
    expected = dict(signature or _bundle_signature(release))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Fetched bundle cache signature mismatch for {key}: {path}")
    artifacts = tuple(
        SourceArtifact(
            source=str(item["source"]),
            source_url=str(item["source_url"]),
            retrieved_at=str(item["retrieved_at"]),
            content=base64.b64decode(item["content_base64"], validate=True),
            content_type=str(item["content_type"]),
        )
        for item in payload["artifacts"]
    )
    return (
        FetchedBundle(
            prices=pd.DataFrame(payload["prices"]),
            corporate_actions=pd.DataFrame(payload["corporate_actions"]),
            artifacts=artifacts,
            missing_symbols=tuple(str(item) for item in payload["missing_symbols"]),
        ),
        int(payload["http_attempts"]),
    )


def _source_attempt_count(source, fetched) -> int:
    value = getattr(getattr(source, "client", None), "attempt_count", None)
    return int(value) if value is not None else len(fetched.artifacts)


def _catalog_archive_row(
    archive: pd.DataFrame,
    *,
    delisted: int,
) -> pd.Series:
    marker = f"delisted={delisted}"
    matches = archive.loc[
        archive["dataset"].astype(str).eq("eodhd_exchange_symbols")
        & archive["source_url"].astype(str).str.contains(marker, regex=False)
    ].copy()
    if matches.empty:
        raise ValueError(f"Cached EODHD catalog is missing {marker}.")
    matches["_retrieved"] = pd.to_datetime(matches["retrieved_at"], errors="coerce")
    return matches.sort_values("_retrieved", kind="stable").iloc[-1]


def load_archived_catalogs(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, CatalogArchive]:
    version = release.dataset_versions.get("source_archive")
    if not version:
        raise ValueError("The frozen release has no source_archive dataset.")
    archive = repository.read_frame("source_archive", version)
    output: dict[str, CatalogArchive] = {}
    root = repository.root.resolve()
    for kind, delisted in (("active", 0), ("delisted", 1)):
        row = _catalog_archive_row(archive, delisted=delisted)
        path = (repository.root / str(row["object_path"])).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"Catalog archive escapes cache root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Catalog archive payload is missing: {path}")
        content = gzip.decompress(path.read_bytes())
        expected_hash = str(row["source_hash"])
        if sha256_bytes(content) != expected_hash:
            raise ValueError(f"Catalog archive hash mismatch: {path}")
        payload = json.loads(content)
        if not isinstance(payload, list):
            raise ValueError(f"Catalog archive is not a JSON list: {path}")
        output[kind] = CatalogArchive(
            kind=kind,
            rows=tuple(dict(item) for item in payload if isinstance(item, dict)),
            source_url=str(row["source_url"]),
            retrieved_at=str(row["retrieved_at"]),
            source_hash=expected_hash,
        )
    return output


def select_catalog_entries(
    catalogs: dict[str, CatalogArchive],
    specs: tuple[SuccessorSpec, ...] = SUCCESSOR_SPECS,
) -> dict[str, CatalogSelection]:
    selected: dict[str, CatalogSelection] = {}
    for spec in specs:
        archive = catalogs.get(spec.catalog_kind)
        if archive is None:
            raise ValueError(f"Missing {spec.catalog_kind} EODHD catalog.")
        matches = [
            row
            for row in archive.rows
            if str(row.get("Code") or "").upper() == spec.provider_code.upper()
            and str(row.get("Type") or "").strip().lower() == "common stock"
            and all(
                token.lower() in str(row.get("Name") or "").lower()
                for token in spec.name_tokens
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one exact {spec.catalog_kind} catalog row for "
                f"{spec.provider_symbol}; found {len(matches)}."
            )
        selected[spec.symbol] = CatalogSelection(spec, matches[0], archive)
    return selected


def _report_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records = report["records"]
    values = records.values() if isinstance(records, dict) else records
    return [dict(item) for item in values if isinstance(item, dict)]


def resolve_purge_security_ids(
    report: dict[str, Any],
    master: pd.DataFrame,
    specs: tuple[PurgeSpec, ...] = PURGE_SPECS,
) -> dict[str, str]:
    records = _report_records(report)
    known_ids = set(master["security_id"].astype(str))
    output: dict[str, str] = {}
    for spec in specs:
        matches = []
        for record in records:
            candidate = record.get("candidate") or {}
            if str(candidate.get("symbol") or "").upper() != spec.symbol.upper():
                continue
            name = str(candidate.get("name") or "").lower()
            if not all(token.lower() in name for token in spec.name_tokens):
                continue
            security_id = str(candidate.get("security_id") or "")
            if security_id:
                matches.append(security_id)
        unique = tuple(dict.fromkeys(matches))
        if len(unique) != 1:
            raise ValueError(
                f"Expected one evidence candidate for {spec.symbol}; found {len(unique)}."
            )
        if unique[0] not in known_ids:
            raise ValueError(
                f"Evidence candidate for {spec.symbol} is absent from security_master: "
                f"{unique[0]}"
            )
        output[spec.symbol] = unique[0]
    return output


def _unique_master_id(
    master: pd.DataFrame,
    *,
    symbol: str,
    provider_symbol: str,
    name_token: str,
) -> str:
    matches = master.loc[
        master["primary_symbol"].astype(str).str.upper().eq(symbol.upper())
        & master.get("provider_symbol", pd.Series("", index=master.index))
        .astype(str)
        .str.upper()
        .eq(provider_symbol.upper())
        & master["name"].astype(str).str.lower().str.contains(name_token.lower(), regex=False)
    ]
    ids = matches["security_id"].astype(str).drop_duplicates()
    if len(ids) != 1:
        raise ValueError(
            f"Expected one {symbol}/{provider_symbol}/{name_token} identity; found {len(ids)}."
        )
    return str(ids.iloc[0])


def _artifact_rows(
    artifacts: tuple[SourceArtifact, ...],
    completed_session: str,
) -> pd.DataFrame:
    rows = []
    for artifact in artifacts:
        content_type = artifact.content_type.lower()
        extension = (
            "json"
            if "json" in content_type
            else "pdf"
            if "pdf" in content_type
            else "html"
            if "html" in content_type
            else "csv"
            if "csv" in content_type
            else "txt"
        )
        rows.append(
            {
                "archive_id": artifact.source_hash,
                "dataset": artifact.source,
                "object_path": (
                    f"archives/{completed_session}/{artifact.source_hash}.{extension}.gz"
                ),
                "content_type": artifact.content_type,
                "effective_date": completed_session,
                "source": artifact.source,
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["archive_id"].duplicated().any():
        raise ValueError("Archive artifacts must have unique request identities.")
    return frame


def _validate_artifact_coverage(
    artifacts: tuple[SourceArtifact, ...],
    specs: tuple[SuccessorSpec, ...],
) -> None:
    covered: set[tuple[str, str]] = set()
    for artifact in artifacts:
        path = urlparse(artifact.source_url).path.rstrip("/")
        for endpoint in ("eod", "div", "splits"):
            prefix = f"eodhd_{endpoint}"
            if artifact.source != prefix:
                continue
            for spec in specs:
                if path.endswith(f"/{endpoint}/{spec.provider_symbol}"):
                    covered.add((endpoint, spec.provider_symbol))
    expected = {
        (endpoint, spec.provider_symbol)
        for spec in specs
        for endpoint in ("eod", "div", "splits")
    }
    missing = sorted(expected - covered)
    if missing:
        raise ValueError(f"Fetched EODHD artifacts are incomplete: {missing}")
    if len(artifacts) != len(expected):
        raise ValueError(
            "Fetched EODHD artifact count is not exactly the audited request count: "
            f"expected={len(expected)}, actual={len(artifacts)}."
        )


def _validate_cov_artifact_coverage(fetched) -> None:
    """Bind DirectIndex primary rows to both pinned static histories."""

    artifacts = tuple(fetched.artifacts)
    eodhd = tuple(
        item
        for item in artifacts
        if item.source in {"eodhd_eod", "eodhd_div", "eodhd_splits"}
    )
    direct = tuple(
        item for item in artifacts if item.source.startswith("cov_directindex_")
    )
    wiki = tuple(
        item for item in artifacts if item.source.startswith("cov_quandl_wiki_")
    )
    unknown = tuple(item for item in artifacts if item not in (*eodhd, *direct, *wiki))
    if unknown:
        raise ValueError("COV supplemental bundle has unknown raw artifact sources.")
    _validate_artifact_coverage(eodhd, (COV_SPEC,))
    direct_roles = {
        item.source.removeprefix("cov_directindex_"): item for item in direct
    }
    wiki_roles = {
        item.source.removeprefix("cov_quandl_wiki_"): item for item in wiki
    }
    if set(direct_roles) != set(COV_DIRECTINDEX_URLS):
        raise ValueError("COV DirectIndex raw evidence roles are incomplete.")
    if set(wiki_roles) != set(COV_WIKI_URLS):
        raise ValueError("COV Quandl WIKI raw evidence roles are incomplete.")
    parsed_direct = _parse_cov_directindex_csv(direct_roles["cov_csv"])
    _validate_cov_wiki_generation_evidence(
        wiki_roles["readme"], wiki_roles["generate_py"]
    )
    _parse_cov_wiki_csv(wiki_roles["cov_csv"])
    if not parsed_direct.equals(fetched.prices.reset_index(drop=True)):
        raise ValueError("COV stored prices differ from pinned DirectIndex raw rows.")
    if set(fetched.prices["source"].astype(str)) != {"directindex_pinned_csv"}:
        raise ValueError("COV prices must use DirectIndex raw OHLCV only.")
    if len(artifacts) != 9:
        raise ValueError("COV supplement must preserve 3 EODHD and 6 pinned raw artifacts.")


def validate_fetched_result(
    fetched,
    *,
    completed_session: str,
    specs: tuple[SuccessorSpec, ...] = SUCCESSOR_SPECS,
) -> None:
    if fetched.missing_symbols:
        raise ValueError(
            "EODHD successor fetch is incomplete: "
            + ", ".join(sorted(map(str, fetched.missing_symbols)))
        )
    expected_ids = {spec.security_id for spec in specs}
    actual_ids = set(fetched.prices["security_id"].astype(str))
    unknown = actual_ids - expected_ids
    if unknown:
        raise ValueError(f"EODHD returned unexpected security_ids: {sorted(unknown)}")
    sessions = fetched.prices.loc[:, ["security_id", "session"]].copy()
    sessions["session"] = pd.to_datetime(sessions["session"], errors="coerce")
    completed = pd.Timestamp(completed_session).normalize()
    for spec in specs:
        own_sessions = sessions.loc[
            sessions["security_id"].astype(str).eq(spec.security_id), "session"
        ].dropna()
        if own_sessions.empty:
            raise ValueError(f"{spec.provider_symbol} has no valid price sessions.")
        first = own_sessions.min().normalize()
        first_limit = pd.Timestamp(spec.first_price_not_after).normalize()
        if first > first_limit:
            raise ValueError(
                f"{spec.provider_symbol} history is truncated: first={first.date()}, "
                f"required_not_after={first_limit.date()}."
            )
        if first < pd.Timestamp(FETCH_START):
            raise ValueError(
                f"{spec.provider_symbol} returned history before requested start {FETCH_START}."
            )
        if spec.require_recent:
            recent_floor = completed - pd.Timedelta(days=RECENT_PRICE_DAYS)
            if own_sessions.max().normalize() < recent_floor:
                raise ValueError(
                    f"{spec.provider_symbol} active history is stale: "
                    f"last={own_sessions.max().date()}, required_on_or_after={recent_floor.date()}."
                )
        event = pd.Timestamp(spec.event_date)
        covered = sessions.loc[
            sessions["security_id"].astype(str).eq(spec.security_id)
            & sessions["session"].between(event, event + pd.Timedelta(days=10))
        ]
        if covered.empty:
            raise ValueError(
                f"{spec.provider_symbol} has no price from {spec.event_date} "
                "through the following 10 calendar days."
            )
    action_ids = set(fetched.corporate_actions.get("security_id", pd.Series(dtype=str)).astype(str))
    if unexpected_actions := action_ids - expected_ids:
        raise ValueError(
            f"EODHD returned actions for unexpected security_ids: {sorted(unexpected_actions)}"
        )
    _validate_artifact_coverage(tuple(fetched.artifacts), specs)
    validate_dataset(
        "daily_price_raw",
        fetched.prices,
        completed_session=completed_session,
    ).raise_for_errors()
    if not fetched.corporate_actions.empty:
        validate_dataset(
            "corporate_actions",
            fetched.corporate_actions,
            incomplete_action_policy="warn",
        ).raise_for_errors()


def validate_cov_fetched_result(fetched) -> None:
    """Require the complete, legally bounded COV trading history."""

    if fetched.missing_symbols:
        raise ValueError(
            "EODHD COV supplemental fetch is incomplete: "
            + ", ".join(sorted(map(str, fetched.missing_symbols)))
        )
    actual_ids = set(fetched.prices.get("security_id", pd.Series(dtype=str)).astype(str))
    if actual_ids != {COV_SPEC.security_id}:
        raise ValueError(
            "COV supplemental prices have unexpected identities: "
            f"expected={[COV_SPEC.security_id]}, actual={sorted(actual_ids)}."
        )
    sessions = pd.to_datetime(fetched.prices["session"], errors="coerce")
    if sessions.isna().any():
        raise ValueError("COV supplemental prices contain an invalid session.")
    actual_sessions = tuple(sorted(sessions.dt.date.astype(str).unique()))
    if actual_sessions != COV_EXPECTED_SESSIONS:
        missing = sorted(set(COV_EXPECTED_SESSIONS) - set(actual_sessions))
        extra = sorted(set(actual_sessions) - set(COV_EXPECTED_SESSIONS))
        raise ValueError(
            "COV supplemental history is not the complete 2015-01-02 through "
            f"2015-01-26 NYSE session set: missing={missing}, extra={extra}."
        )
    action_ids = set(
        fetched.corporate_actions.get("security_id", pd.Series(dtype=str)).astype(str)
    )
    if action_ids - {COV_SPEC.security_id}:
        raise ValueError(
            "COV supplemental actions have unexpected identities: "
            f"{sorted(action_ids - {COV_SPEC.security_id})}."
        )
    if not fetched.corporate_actions.empty:
        action_types = set(
            fetched.corporate_actions["action_type"].astype(str).str.lower()
        )
        if unexpected := action_types - {"cash_dividend", "split"}:
            raise ValueError(
                f"COV supplemental actions contain unsupported types: {sorted(unexpected)}."
            )
        action_dates = pd.to_datetime(
            fetched.corporate_actions["effective_date"], errors="coerce"
        )
        if action_dates.isna().any() or not action_dates.between(
            pd.Timestamp(COV_FETCH_START),
            pd.Timestamp(COV_LAST_TRADING_DATE),
        ).all():
            raise ValueError("COV supplemental actions escape the audited history window.")
    _validate_cov_artifact_coverage(fetched)
    validate_dataset(
        "daily_price_raw",
        fetched.prices,
        completed_session=COV_LAST_TRADING_DATE,
    ).raise_for_errors()
    if not fetched.corporate_actions.empty:
        validate_dataset(
            "corporate_actions",
            fetched.corporate_actions,
            incomplete_action_policy="warn",
        ).raise_for_errors()


def _provider_action_mask(actions: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=actions.index)
    if "source_kind" in actions:
        mask |= actions["source_kind"].astype(str).str.lower().eq("provider")
    if "source" in actions:
        mask |= actions["source"].astype(str).str.lower().str.startswith("eodhd_")
    return mask


def _concat_deduplicated(
    existing: pd.DataFrame,
    addition: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    output = pd.concat([existing, addition], ignore_index=True, sort=False)
    return output.drop_duplicates(list(dataset_spec(dataset).primary_key), keep="last")


def _new_identity_frames(
    selections: dict[str, CatalogSelection],
    prices: pd.DataFrame,
    *,
    completed_session: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    completed = pd.Timestamp(completed_session)
    for symbol, selection in selections.items():
        spec = selection.spec
        own = prices.loc[prices["security_id"].astype(str).eq(spec.security_id)].copy()
        sessions = pd.to_datetime(own["session"], errors="coerce").dropna()
        if sessions.empty:
            raise ValueError(f"No fetched price history for {spec.provider_symbol}.")
        first = sessions.min().date().isoformat()
        last = sessions.max().date().isoformat()
        active_to = "" if pd.Timestamp(last) >= completed - pd.Timedelta(days=10) else last
        common = {
            "security_id": spec.security_id,
            "exchange": str(selection.row.get("Exchange") or "US"),
            "source": "eodhd_exchange_symbols",
            "source_url": selection.archive.source_url,
            "retrieved_at": selection.archive.retrieved_at,
            "source_hash": selection.archive.source_hash,
        }
        master_rows.append(
            {
                **common,
                "primary_symbol": spec.symbol,
                "provider_symbol": spec.provider_symbol,
                "action_provider_symbol": spec.provider_symbol,
                "name": str(selection.row.get("Name") or spec.symbol),
                "asset_type": "STOCK",
                "currency": str(selection.row.get("Currency") or "USD"),
                "country": "US",
                # Preserve the complete provider series for warm-up.  The
                # tradable symbol interval is represented separately below.
                "active_from": first,
                "active_to": active_to,
            }
        )
        history_rows.append(
            {
                **common,
                "symbol": spec.symbol,
                "effective_from": spec.history_start,
                "effective_to": active_to,
            }
        )
    return pd.DataFrame(master_rows), pd.DataFrame(history_rows)


def _trim_old_candidates(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    purge_ids: dict[str, str],
    specs: tuple[PurgeSpec, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    price_dates = pd.to_datetime(prices["session"], errors="coerce")
    action_dates = pd.to_datetime(actions["effective_date"], errors="coerce")
    provider_actions = _provider_action_mask(actions)
    drop_prices = pd.Series(False, index=prices.index)
    drop_actions = pd.Series(False, index=actions.index)
    for spec in specs:
        security_id = purge_ids[spec.symbol]
        cutoff = pd.Timestamp(spec.cutoff)
        drop_prices |= prices["security_id"].astype(str).eq(security_id) & price_dates.ge(cutoff)
        drop_actions |= (
            actions["security_id"].astype(str).eq(security_id)
            & action_dates.ge(cutoff)
            & provider_actions
        )
    return (
        prices.loc[~drop_prices].copy(),
        actions.loc[~drop_actions].copy(),
        {
            "old_price_rows_removed": int(drop_prices.sum()),
            "old_provider_action_rows_removed": int(drop_actions.sum()),
        },
    )


def _close_old_identities(
    master: pd.DataFrame,
    history: pd.DataFrame,
    prices: pd.DataFrame,
    purge_ids: dict[str, str],
    specs: tuple[PurgeSpec, ...],
    *,
    preserve_history_ids: Iterable[str] = IDENTITY_REPAIR_MIGRATION_GAP_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    output_master = master.copy()
    output_history = history.copy()
    last_prices: dict[str, str] = {}
    drop_history: list[int] = []
    specs_by_symbol = {spec.symbol: spec for spec in specs}
    preserved = {
        str(security_id).strip()
        for security_id in preserve_history_ids
        if str(security_id).strip()
    }
    for symbol, security_id in purge_ids.items():
        own = prices.loc[prices["security_id"].astype(str).eq(security_id)]
        sessions = pd.to_datetime(own["session"], errors="coerce").dropna()
        if sessions.empty:
            raise ValueError(f"Old candidate {symbol}/{security_id} has no price left after trim.")
        last = sessions.max().date().isoformat()
        last_prices[symbol] = last
        spec = specs_by_symbol.get(symbol)
        if spec is None:
            raise ValueError(f"Old candidate {symbol} has no matching purge specification.")
        history_end = spec.history_active_to or last
        if pd.Timestamp(history_end) < pd.Timestamp(last):
            raise ValueError(
                f"Old candidate {symbol} history cannot end before its last price: "
                f"history_end={history_end}, last_price={last}."
            )
        master_mask = output_master["security_id"].astype(str).eq(security_id)
        if int(master_mask.sum()) != 1:
            raise ValueError(f"Old candidate {symbol}/{security_id} is not unique in master.")
        output_master.loc[master_mask, "active_to"] = last
        indices = output_history.index[
            output_history["security_id"].astype(str).eq(security_id)
        ]
        if len(indices) == 0:
            raise ValueError(f"Old candidate {symbol}/{security_id} has no symbol history.")
        # AGN/ACT, COR and the other audited identity collisions are migrated by
        # the immediately following identity-repair release, which also rewrites
        # their index membership rows.  Closing their symbol interval here would
        # leave the still-unmigrated member ID without any active symbol on every
        # intervening replay date.  Keep only that transitional symbol interval
        # unchanged; prices and security_master are still closed fail-closed, and
        # the release remains degraded/publish-blocked until the identity repair.
        if security_id in preserved:
            continue
        for index in indices:
            start = pd.to_datetime(output_history.at[index, "effective_from"], errors="coerce")
            end = pd.to_datetime(output_history.at[index, "effective_to"], errors="coerce")
            if pd.notna(start) and start > pd.Timestamp(history_end):
                drop_history.append(index)
            elif pd.isna(end) or end > pd.Timestamp(history_end):
                output_history.at[index, "effective_to"] = history_end
    if drop_history:
        output_history = output_history.drop(index=drop_history)
    return output_master, output_history, last_prices


def _repair_actavis_history(
    master: pd.DataFrame,
    history: pd.DataFrame,
    *,
    stamp: str,
) -> tuple[pd.DataFrame, str]:
    security_id = _unique_master_id(
        master,
        symbol="AGN",
        provider_symbol="AGN.US",
        name_token="allergan plc",
    )
    base = master.loc[master["security_id"].astype(str).eq(security_id)].iloc[0]
    provenance = b"Actavis ACT to Allergan AGN symbol-history repair"
    source_hash = sha256_bytes(provenance)
    common = {
        "security_id": security_id,
        "exchange": str(base["exchange"]),
        "source": "derived_ticker_identity",
        "source_url": "local://us-lifecycle-successors/ACT-AGN",
        "retrieved_at": stamp,
        "source_hash": source_hash,
    }
    repaired = pd.DataFrame(
        [
            {
                **common,
                "symbol": "ACT",
                "effective_from": "2015-01-01",
                "effective_to": "2015-06-14",
            },
            {
                **common,
                "symbol": "AGN",
                "effective_from": "2015-06-15",
                "effective_to": "2020-05-08",
            },
        ]
    )
    output = history.loc[~history["security_id"].astype(str).eq(security_id)].copy()
    return _concat_deduplicated(output, repaired, dataset="symbol_history"), security_id


def _repair_existing_successor_histories(
    master: pd.DataFrame,
    history: pd.DataFrame,
    *,
    stamp: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    output_master = master.copy()
    output_history = history.copy()
    repairs = {
        "RVTY": (
            _unique_master_id(
                master,
                symbol="RVTY",
                provider_symbol="RVTY.US",
                name_token="revvity",
            ),
            "2023-05-16",
        ),
        "LUMN": (
            _unique_master_id(
                master,
                symbol="LUMN",
                provider_symbol="LUMN.US",
                name_token="lumen",
            ),
            "2020-09-18",
        ),
        "BNY": (
            _unique_master_id(
                master,
                symbol="BNY",
                provider_symbol="BNY.US",
                name_token="bank of new york mellon",
            ),
            "2026-05-21",
        ),
    }
    for symbol, (security_id, effective_from) in repairs.items():
        mask = (
            output_history["security_id"].astype(str).eq(security_id)
            & output_history["symbol"].astype(str).str.upper().eq(symbol)
        )
        if int(mask.sum()) != 1:
            raise ValueError(f"{symbol} symbol history must contain exactly one interval.")
        provenance = f"existing successor history repair|{symbol}|{effective_from}".encode()
        output_history.loc[mask, "effective_from"] = effective_from
        output_history.loc[mask, "source"] = "derived_ticker_identity"
        output_history.loc[mask, "source_url"] = (
            f"local://us-lifecycle-successors/{symbol}"
        )
        output_history.loc[mask, "retrieved_at"] = stamp
        output_history.loc[mask, "source_hash"] = sha256_bytes(provenance)

    bny_old_id = _unique_master_id(
        master,
        symbol="BNY",
        provider_symbol="BNY_old.US",
        name_token="blackrock",
    )
    old_master_mask = output_master["security_id"].astype(str).eq(bny_old_id)
    old_history_mask = output_history["security_id"].astype(str).eq(bny_old_id)
    if int(old_master_mask.sum()) != 1 or not old_history_mask.any():
        raise ValueError("BNY_old identity is incomplete.")
    output_master.loc[old_master_mask, "active_to"] = "2026-02-09"
    output_history.loc[old_history_mask, "effective_to"] = "2026-02-09"
    old_hash = sha256_bytes(b"BNY_old BlackRock identity closed 2026-02-09")
    for frame, mask in (
        (output_master, old_master_mask),
        (output_history, old_history_mask),
    ):
        frame.loc[mask, "source"] = "derived_ticker_identity"
        frame.loc[mask, "source_url"] = "local://us-lifecycle-successors/BNY-old"
        frame.loc[mask, "retrieved_at"] = stamp
        frame.loc[mask, "source_hash"] = old_hash
    return (
        output_master,
        output_history,
        {
            **{symbol: security_id for symbol, (security_id, _date) in repairs.items()},
            "BNY_old": bny_old_id,
        },
    )


def _repair_dwdp(
    master: pd.DataFrame,
    history: pd.DataFrame,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, dict[str, int]]:
    security_id = _unique_master_id(
        master,
        symbol="DWDP",
        provider_symbol="DWDP.US",
        name_token="dowdupont",
    )
    cutoff = pd.Timestamp(DWDP_START)
    price_dates = pd.to_datetime(prices["session"], errors="coerce")
    action_dates = pd.to_datetime(actions["effective_date"], errors="coerce")
    drop_prices = prices["security_id"].astype(str).eq(security_id) & price_dates.lt(cutoff)
    drop_actions = actions["security_id"].astype(str).eq(security_id) & action_dates.lt(cutoff)
    output_prices = prices.loc[~drop_prices].copy()
    output_actions = actions.loc[~drop_actions].copy()
    own_dates = pd.to_datetime(
        output_prices.loc[
            output_prices["security_id"].astype(str).eq(security_id), "session"
        ],
        errors="coerce",
    ).dropna()
    if own_dates.empty or own_dates.min().date().isoformat() != DWDP_START:
        raise ValueError("DWDP repair requires a retained 2017-09-01 first price.")
    output_master = master.copy()
    output_master.loc[
        output_master["security_id"].astype(str).eq(security_id), "active_from"
    ] = DWDP_START
    output_history = history.copy()
    history_mask = (
        output_history["security_id"].astype(str).eq(security_id)
        & output_history["symbol"].astype(str).str.upper().eq("DWDP")
    )
    if int(history_mask.sum()) != 1:
        raise ValueError("DWDP symbol history must contain exactly one interval.")
    output_history.loc[history_mask, "effective_from"] = DWDP_START
    return (
        output_master,
        output_history,
        output_prices,
        output_actions,
        security_id,
        {
            "dwdp_price_rows_removed": int(drop_prices.sum()),
            "dwdp_action_rows_removed": int(drop_actions.sum()),
        },
    )


def rewrite_market_frames(
    existing: dict[str, pd.DataFrame],
    selections: dict[str, CatalogSelection],
    purge_ids: dict[str, str],
    fetched,
    *,
    completed_session: str,
    stamp: str,
    successor_specs: tuple[SuccessorSpec, ...] = SUCCESSOR_SPECS,
    purge_specs: tuple[PurgeSpec, ...] = PURGE_SPECS,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    selected_ids = {spec.security_id for spec in successor_specs}
    master = existing["security_master"].copy()
    history = existing["symbol_history"].copy()
    prices = existing["daily_price_raw"].copy()
    actions = existing["corporate_actions"].copy()

    prices, actions, trim_stats = _trim_old_candidates(
        prices, actions, purge_ids, purge_specs
    )
    prices = prices.loc[~prices["security_id"].astype(str).isin(selected_ids)].copy()
    provider_mask = _provider_action_mask(actions)
    actions = actions.loc[
        ~(actions["security_id"].astype(str).isin(selected_ids) & provider_mask)
    ].copy()
    fetched_prices = fetched.prices.loc[
        pd.to_datetime(fetched.prices["session"], errors="coerce")
        <= pd.Timestamp(completed_session)
    ].copy()
    # Preserve every provider row from the requested 2015 backfill.  The
    # history interval, not this continuous series, controls symbol validity.
    prices = _concat_deduplicated(prices, fetched_prices, dataset="daily_price_raw")
    actions = _concat_deduplicated(
        actions, fetched.corporate_actions, dataset="corporate_actions"
    )

    master = master.loc[~master["security_id"].astype(str).isin(selected_ids)].copy()
    history = history.loc[~history["security_id"].astype(str).isin(selected_ids)].copy()
    new_master, new_history = _new_identity_frames(
        selections, fetched_prices, completed_session=completed_session
    )
    master = _concat_deduplicated(master, new_master, dataset="security_master")
    history = _concat_deduplicated(history, new_history, dataset="symbol_history")
    master, history, last_prices = _close_old_identities(
        master, history, prices, purge_ids, purge_specs
    )
    history, actavis_id = _repair_actavis_history(master, history, stamp=stamp)
    master, history, existing_repair_ids = _repair_existing_successor_histories(
        master,
        history,
        stamp=stamp,
    )
    (
        master,
        history,
        prices,
        actions,
        dwdp_id,
        dwdp_stats,
    ) = _repair_dwdp(master, history, prices, actions)
    stats = {
        **trim_stats,
        **dwdp_stats,
        "successor_price_rows": len(fetched_prices),
        "successor_provider_action_rows": len(fetched.corporate_actions),
        "old_last_prices": last_prices,
        "actavis_security_id": actavis_id,
        "dwdp_security_id": dwdp_id,
        "existing_successor_repair_ids": existing_repair_ids,
    }
    return {
        **existing,
        "security_master": master.reset_index(drop=True),
        "symbol_history": history.reset_index(drop=True),
        "daily_price_raw": prices.reset_index(drop=True),
        "corporate_actions": actions.reset_index(drop=True),
    }, stats


def apply_cov_supplement(
    frames: dict[str, pd.DataFrame],
    fetched,
    *,
    cov_security_id: str,
    stamp: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if cov_security_id != COV_SPEC.security_id:
        raise ValueError(
            f"COV identity changed after preflight: {cov_security_id}."
        )
    validate_cov_fetched_result(fetched)
    output = {name: frame.copy() for name, frame in frames.items()}
    master = output["security_master"]
    history = output["symbol_history"]
    prices = output["daily_price_raw"]
    actions = output["corporate_actions"]
    master_mask = master["security_id"].astype(str).eq(cov_security_id)
    history_mask = (
        history["security_id"].astype(str).eq(cov_security_id)
        & history["symbol"].astype(str).str.upper().eq("COV")
    )
    if int(master_mask.sum()) != 1 or int(history_mask.sum()) != 1:
        raise ValueError("COV master and symbol-history identities must remain unique.")

    evidence = _cov_official_evidence_artifact(retrieved_at=stamp)
    for frame, mask, end_column in (
        (master, master_mask, "active_to"),
        (history, history_mask, "effective_to"),
    ):
        frame.loc[mask, end_column] = COV_LAST_TRADING_DATE
        frame.loc[mask, "source"] = "sec_edgar+eodhd_terminal_price"
        frame.loc[mask, "source_url"] = COV_SEC_COMPLETION_URL
        frame.loc[mask, "retrieved_at"] = stamp
        frame.loc[mask, "source_hash"] = evidence.source_hash

    prices = prices.loc[
        ~prices["security_id"].astype(str).eq(cov_security_id)
    ].copy()
    provider_mask = _provider_action_mask(actions)
    actions = actions.loc[
        ~(
            actions["security_id"].astype(str).eq(cov_security_id)
            & provider_mask
        )
    ].copy()
    prices = _concat_deduplicated(
        prices,
        fetched.prices,
        dataset="daily_price_raw",
    )
    actions = _concat_deduplicated(
        actions,
        fetched.corporate_actions,
        dataset="corporate_actions",
    )
    output.update(
        {
            "security_master": master.reset_index(drop=True),
            "symbol_history": history.reset_index(drop=True),
            "daily_price_raw": prices.reset_index(drop=True),
            "corporate_actions": actions.reset_index(drop=True),
        }
    )
    return output, {
        "cov_security_id": cov_security_id,
        "cov_price_rows": len(fetched.prices),
        "cov_provider_action_rows": len(fetched.corporate_actions),
        "cov_first_session": COV_EXPECTED_SESSIONS[0],
        "cov_last_session": COV_LAST_TRADING_DATE,
        "cov_identity_closed_on": COV_LAST_TRADING_DATE,
    }


def _enb_economic_crosscheck(
    prices: pd.DataFrame,
    *,
    spectra_security_id: str,
    enb_security_id: str,
) -> dict[str, Any]:
    values = prices.loc[:, ["security_id", "session", "close"]].copy()
    values["session"] = pd.to_datetime(values["session"], errors="coerce")
    values["close"] = pd.to_numeric(values["close"], errors="coerce")
    values = values.dropna(subset=["session", "close"])
    old = values.loc[
        values["security_id"].astype(str).eq(spectra_security_id)
        & values["session"].le(pd.Timestamp(SPECTRA_LAST_TRADING_DATE))
    ].sort_values("session")
    new = values.loc[
        values["security_id"].astype(str).eq(enb_security_id)
        & values["session"].ge(pd.Timestamp(ENB_EFFECTIVE_DATE))
        & values["session"].le(pd.Timestamp(ENB_EFFECTIVE_DATE) + pd.Timedelta(days=10))
    ].sort_values("session")
    if old.empty or new.empty:
        raise ValueError("Spectra/ENB economic crosscheck lacks a terminal or successor close.")
    old_row = old.iloc[-1]
    new_row = new.iloc[0]
    if old_row["session"].date().isoformat() != SPECTRA_LAST_TRADING_DATE:
        raise ValueError("Spectra economic crosscheck did not use the exact terminal close.")
    if new_row["session"].date().isoformat() != ENB_EFFECTIVE_DATE:
        raise ValueError("ENB economic crosscheck requires the 2017-02-27 first close.")
    old_close = float(old_row["close"])
    enb_close = float(new_row["close"])
    implied = ENB_RATIO * enb_close
    deviation = abs(old_close - implied) / max(abs(old_close), abs(implied), 1e-12)
    if deviation > 0.20:
        raise ValueError(
            "Spectra/ENB 0.984-share economic crosscheck failed: "
            f"deviation={deviation:.6f}."
        )
    return {
        "spectra_session": SPECTRA_LAST_TRADING_DATE,
        "spectra_close": old_close,
        "enb_session": ENB_EFFECTIVE_DATE,
        "enb_close": enb_close,
        "ratio": ENB_RATIO,
        "implied_value": implied,
        "relative_deviation": deviation,
        "maximum_relative_deviation": 0.20,
        "passed": True,
    }


def apply_enb_supplement(
    existing: dict[str, pd.DataFrame],
    selection: CatalogSelection,
    fetched,
    *,
    completed_session: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    validate_fetched_result(
        fetched,
        completed_session=completed_session,
        specs=(ENB_SPEC,),
    )
    output = {name: frame.copy() for name, frame in existing.items()}
    selected_id = ENB_SPEC.security_id
    if output["security_master"]["security_id"].astype(str).eq(selected_id).any():
        raise ValueError("ENB security_id already exists before the supplement rewrite.")
    fetched_prices = fetched.prices.loc[
        pd.to_datetime(fetched.prices["session"], errors="coerce")
        <= pd.Timestamp(completed_session)
    ].copy()
    new_master, new_history = _new_identity_frames(
        {ENB_SPEC.symbol: selection},
        fetched_prices,
        completed_session=completed_session,
    )
    output["security_master"] = _concat_deduplicated(
        output["security_master"],
        new_master,
        dataset="security_master",
    ).reset_index(drop=True)
    output["symbol_history"] = _concat_deduplicated(
        output["symbol_history"],
        new_history,
        dataset="symbol_history",
    ).reset_index(drop=True)
    output["daily_price_raw"] = _concat_deduplicated(
        output["daily_price_raw"],
        fetched_prices,
        dataset="daily_price_raw",
    ).reset_index(drop=True)
    output["corporate_actions"] = _concat_deduplicated(
        output["corporate_actions"],
        fetched.corporate_actions,
        dataset="corporate_actions",
    ).reset_index(drop=True)
    crosscheck = _enb_economic_crosscheck(
        output["daily_price_raw"],
        spectra_security_id=SPECTRA_SECURITY_ID,
        enb_security_id=selected_id,
    )
    return output, {
        "spectra_security_id": SPECTRA_SECURITY_ID,
        "enb_security_id": selected_id,
        "enb_price_rows": len(fetched_prices),
        "enb_provider_action_rows": len(fetched.corporate_actions),
        "economic_crosscheck": crosscheck,
    }


def active_price_gaps(
    master: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    completed_session: str,
) -> tuple[dict[str, str], ...]:
    completed = pd.Timestamp(completed_session).normalize()
    starts = pd.to_datetime(master["active_from"], errors="coerce")
    ends = pd.to_datetime(master["active_to"], errors="coerce")
    active = master.loc[
        (starts.isna() | starts.le(completed))
        & (ends.isna() | ends.ge(completed))
    ].copy()
    last_prices = (
        prices.assign(_session=pd.to_datetime(prices["session"], errors="coerce"))
        .groupby(prices["security_id"].astype(str))["_session"]
        .max()
    )
    active["last_price_session"] = active["security_id"].astype(str).map(last_prices)
    floor = completed - pd.Timedelta(days=RECENT_PRICE_DAYS)
    gaps = active.loc[
        active["last_price_session"].isna()
        | active["last_price_session"].lt(floor)
    ]
    return tuple(
        {
            "security_id": str(row.security_id),
            "symbol": str(row.primary_symbol),
            "provider_symbol": str(row.provider_symbol),
            "last_price_session": (
                ""
                if pd.isna(row.last_price_session)
                else pd.Timestamp(row.last_price_session).date().isoformat()
            ),
        }
        for row in gaps.itertuples(index=False)
    )


class _FrameRepository:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def current_manifest(self, dataset: str):
        return object() if dataset in self.frames else None

    def read_frame(self, dataset: str):
        return self.frames[dataset].copy()


def validate_candidate_frames(
    frames: dict[str, pd.DataFrame],
    *,
    completed_session: str,
) -> tuple[str, ...]:
    warnings: list[str] = []
    for dataset in WRITE_DATASETS:
        report = validate_dataset(
            dataset,
            frames[dataset],
            completed_session=completed_session,
            incomplete_action_policy="warn",
        )
        report.raise_for_errors()
        warnings.extend(issue.message for issue in report.issues if issue.severity != "error")
    cross = validate_repository_snapshot(
        _FrameRepository(frames),
        allowed_index_price_gap_ids=IDENTITY_REPAIR_MIGRATION_GAP_IDS,
    )
    cross.raise_for_errors()
    warnings.extend(issue.message for issue in cross.issues if issue.severity != "error")
    warnings.append(IDENTITY_REPAIR_MIGRATION_WARNING)
    return tuple(dict.fromkeys(warnings))


def validate_enb_candidate_frames(
    frames: dict[str, pd.DataFrame],
    *,
    completed_session: str,
) -> tuple[str, ...]:
    """Validate the post-identity ENB supplement with no migration exceptions."""

    warnings: list[str] = []
    for dataset in WRITE_DATASETS:
        report = validate_dataset(
            dataset,
            frames[dataset],
            completed_session=completed_session,
            incomplete_action_policy="warn",
        )
        report.raise_for_errors()
        warnings.extend(
            issue.message for issue in report.issues if issue.severity != "error"
        )
    cross = validate_repository_snapshot(_FrameRepository(frames))
    cross.raise_for_errors()
    warnings.extend(
        issue.message for issue in cross.issues if issue.severity != "error"
    )
    return tuple(dict.fromkeys(warnings))


def _assert_release_unchanged(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
) -> None:
    current, current_etag = repository.current_release()
    if current is None or current.version != release.version or current_etag != release_etag:
        raise RuntimeError("Current release changed after the lifecycle collection began.")


def _capture_release_pointer_etags(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset, expected_version in release.dataset_versions.items():
        pointer, etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or not expected_version
            or pointer.version != expected_version
        ):
            actual = pointer.version if pointer is not None else "missing"
            raise RuntimeError(
                f"{dataset} current pointer is not the frozen release version: "
                f"expected={expected_version}, actual={actual}."
            )
        if dataset in WRITE_DATASETS:
            output[dataset] = etag
    missing = tuple(dataset for dataset in WRITE_DATASETS if dataset not in output)
    if missing:
        raise RuntimeError(
            "Frozen release is missing writable datasets: " + ", ".join(missing)
        )
    return output


def _validate_spec_invariants() -> None:
    if len(SUCCESSOR_SPECS) != EXPECTED_SUCCESSOR_CODES:
        raise RuntimeError(
            "Audited successor code count changed: "
            f"expected={EXPECTED_SUCCESSOR_CODES}, actual={len(SUCCESSOR_SPECS)}."
        )
    for label, values in (
        ("symbol", [item.symbol for item in SUCCESSOR_SPECS]),
        ("provider_code", [item.provider_code for item in SUCCESSOR_SPECS]),
        ("security_id", [item.security_id for item in SUCCESSOR_SPECS]),
    ):
        if len(values) != len(set(values)):
            raise RuntimeError(f"Audited successor {label} values are not unique.")
    if COV_SPEC in SUCCESSOR_SPECS or COV_SPEC.provider_code in {
        item.provider_code for item in SUCCESSOR_SPECS
    }:
        raise RuntimeError("COV must remain outside the frozen 13-code bundle.")
    if COV_MAX_EODHD_HTTP_ATTEMPTS != 3:
        raise RuntimeError("The preserved COV single-symbol evidence must remain three artifacts.")
    if ENB_SPEC in SUCCESSOR_SPECS or ENB_SPEC.provider_code in {
        item.provider_code for item in SUCCESSOR_SPECS
    }:
        raise RuntimeError("ENB must remain outside the frozen 13-code bundle.")
    if ENB_SPEC.security_id != "US:EODHD:8b62832f-27a7-5139-a199-62f9632c21bd":
        raise RuntimeError("The deterministic ENB successor identity changed.")
    if ENB_MAX_EODHD_HTTP_ATTEMPTS != 3:
        raise RuntimeError("The ENB supplement must remain capped at three EODHD calls.")
    if ENB_RATIO != 0.984 or ENB_EFFECTIVE_DATE != "2017-02-27":
        raise RuntimeError("The reviewed Spectra/Enbridge transaction terms changed.")
    if len(ENB_SEC_COMPLETION_SHA256) != 64:
        raise RuntimeError("The Spectra/Enbridge SEC completion SHA-256 is invalid.")
    xnys_sessions = _cov_xnys_sessions()
    if len(COV_EXPECTED_SESSIONS) != 16 or COV_EXPECTED_SESSIONS != xnys_sessions:
        raise RuntimeError(
            "COV terminal history must be exactly the 16 XNYS sessions from "
            f"{COV_EXPECTED_SESSIONS[0]} through {COV_LAST_TRADING_DATE}."
        )
    if COV_WIKI_MAX_HTTP_ATTEMPTS != 3 or set(COV_WIKI_URLS) != {
        "cov_csv",
        "readme",
        "generate_py",
    }:
        raise RuntimeError("COV independent evidence must remain exactly three artifacts.")
    if any(COV_WIKI_COMMIT not in url for url in COV_WIKI_URLS.values()):
        raise RuntimeError("COV independent evidence URLs must pin the audited commit.")
    if set(COV_WIKI_SHA256) != set(COV_WIKI_URLS) or any(
        len(value) != 64 for value in COV_WIKI_SHA256.values()
    ):
        raise RuntimeError("Every COV independent artifact must pin its SHA-256.")
    if COV_DIRECTINDEX_MAX_HTTP_ATTEMPTS != 3 or set(COV_DIRECTINDEX_URLS) != {
        "cov_csv",
        "readme",
        "license",
    }:
        raise RuntimeError("COV DirectIndex primary evidence must remain three artifacts.")
    if any(
        COV_DIRECTINDEX_COMMIT not in url for url in COV_DIRECTINDEX_URLS.values()
    ):
        raise RuntimeError("COV DirectIndex URLs must pin the audited commit.")
    if set(COV_DIRECTINDEX_SHA256) != set(COV_DIRECTINDEX_URLS) or any(
        len(value) != 64 for value in COV_DIRECTINDEX_SHA256.values()
    ):
        raise RuntimeError("Every COV DirectIndex artifact must pin its SHA-256.")
    if COV_EODHD_LEDGER_USAGE_AFTER_FAILURES != 8697:
        raise RuntimeError("COV EODHD failure ledger observation must remain 8,697.")


def _preflight_identity_ids(master: pd.DataFrame) -> dict[str, str]:
    return {
        "RVTY": _unique_master_id(
            master,
            symbol="RVTY",
            provider_symbol="RVTY.US",
            name_token="revvity",
        ),
        "LUMN": _unique_master_id(
            master,
            symbol="LUMN",
            provider_symbol="LUMN.US",
            name_token="lumen",
        ),
        "BNY": _unique_master_id(
            master,
            symbol="BNY",
            provider_symbol="BNY.US",
            name_token="bank of new york mellon",
        ),
        "BNY_old": _unique_master_id(
            master,
            symbol="BNY",
            provider_symbol="BNY_old.US",
            name_token="blackrock",
        ),
    }


def validate_local_preflight(
    existing: dict[str, pd.DataFrame],
    purge_ids: dict[str, str],
) -> tuple[str, str, dict[str, str]]:
    """Validate every local prerequisite before spending an EODHD request."""

    master = existing["security_master"]
    history = existing["symbol_history"]
    prices = existing["daily_price_raw"]
    price_dates = pd.to_datetime(prices["session"], errors="coerce")
    actavis_id = _unique_master_id(
        master,
        symbol="AGN",
        provider_symbol="AGN.US",
        name_token="allergan plc",
    )
    dwdp_id = _unique_master_id(
        master,
        symbol="DWDP",
        provider_symbol="DWDP.US",
        name_token="dowdupont",
    )
    repaired_ids = _preflight_identity_ids(master)

    for spec in PURGE_SPECS:
        security_id = purge_ids[spec.symbol]
        if int(master["security_id"].astype(str).eq(security_id).sum()) != 1:
            raise ValueError(
                f"Old candidate {spec.symbol}/{security_id} is not unique in master."
            )
        if not history["security_id"].astype(str).eq(security_id).any():
            raise ValueError(
                f"Old candidate {spec.symbol}/{security_id} has no symbol history."
            )
        cutoff = pd.Timestamp(spec.cutoff)
        retained = price_dates.loc[
            prices["security_id"].astype(str).eq(security_id) & price_dates.lt(cutoff)
        ].dropna()
        if retained.empty:
            raise ValueError(
                f"Old candidate {spec.symbol}/{security_id} has no price before {spec.cutoff}."
            )
        if retained.max() < cutoff - pd.Timedelta(days=10):
            raise ValueError(
                f"Old candidate {spec.symbol}/{security_id} has a pre-cutoff price gap: "
                f"last={retained.max().date()}, cutoff={spec.cutoff}."
            )

    dow_id = purge_ids["DOW"]
    dow_aug31 = prices.loc[
        prices["security_id"].astype(str).eq(dow_id)
        & price_dates.eq(pd.Timestamp("2017-08-31"))
    ]
    if len(dow_aug31) != 1:
        raise ValueError("Old DOW must retain exactly one 2017-08-31 close.")

    dwdp_dates = price_dates.loc[
        prices["security_id"].astype(str).eq(dwdp_id)
    ].dropna()
    if not dwdp_dates.eq(pd.Timestamp(DWDP_START)).any():
        raise ValueError("DWDP must contain its 2017-09-01 first tradable session.")
    dwdp_history = history.loc[
        history["security_id"].astype(str).eq(dwdp_id)
        & history["symbol"].astype(str).str.upper().eq("DWDP")
    ]
    if len(dwdp_history) != 1:
        raise ValueError("DWDP symbol history must contain exactly one interval.")
    if not prices["security_id"].astype(str).eq(actavis_id).any():
        raise ValueError("Actavis/AGN identity has no price history.")
    for label, security_id in repaired_ids.items():
        if not history["security_id"].astype(str).eq(security_id).any():
            raise ValueError(f"{label} repair identity has no symbol history.")
    return actavis_id, dwdp_id, repaired_ids


def validate_cov_local_preflight(
    existing: dict[str, pd.DataFrame],
    selection: CatalogSelection,
    *,
    completed_session: str,
    release_warnings: tuple[str, ...],
) -> tuple[str, str]:
    """Fail before any provider call unless the sole local gap is exact COV."""

    if selection.archive.kind != "delisted":
        raise ValueError("COV must resolve from the archived delisted catalog.")
    if str(selection.row.get("Isin") or "").upper() != "USG2554F1134":
        raise ValueError("The archived COV catalog row has an unexpected ISIN.")
    master = existing["security_master"]
    history = existing["symbol_history"]
    prices = existing["daily_price_raw"]
    actions = existing["corporate_actions"]
    cov_id = _unique_master_id(
        master,
        symbol="COV",
        provider_symbol="COV.US",
        name_token="covidien",
    )
    if cov_id != COV_SPEC.security_id:
        raise ValueError(
            f"COV existing identity does not match the deterministic catalog ID: {cov_id}."
        )
    cov_master = master.loc[master["security_id"].astype(str).eq(cov_id)]
    active_to = pd.to_datetime(cov_master.iloc[0]["active_to"], errors="coerce")
    if pd.notna(active_to) and active_to < pd.Timestamp(completed_session):
        raise ValueError("COV identity is already closed before the supplemental repair.")
    cov_history = history.loc[
        history["security_id"].astype(str).eq(cov_id)
        & history["symbol"].astype(str).str.upper().eq("COV")
    ]
    if len(cov_history) != 1:
        raise ValueError("COV must have exactly one open symbol-history interval.")
    cov_history_end = pd.to_datetime(
        cov_history.iloc[0]["effective_to"], errors="coerce"
    )
    if pd.notna(cov_history_end) and cov_history_end < pd.Timestamp(completed_session):
        raise ValueError("COV symbol history is already closed before repair.")
    if prices["security_id"].astype(str).eq(cov_id).any():
        raise ValueError("COV preflight expected zero stored price rows.")
    if actions["security_id"].astype(str).eq(cov_id).any():
        raise ValueError("COV preflight expected zero stored corporate actions.")

    mdt_id = _unique_master_id(
        master,
        symbol="MDT",
        provider_symbol="MDT.US",
        name_token="medtronic",
    )
    mdt_history = history.loc[
        history["security_id"].astype(str).eq(mdt_id)
        & history["symbol"].astype(str).str.upper().eq("MDT")
    ].copy()
    starts = pd.to_datetime(mdt_history["effective_from"], errors="coerce")
    ends = pd.to_datetime(mdt_history["effective_to"], errors="coerce")
    if not (
        (starts.isna() | starts.le(pd.Timestamp(COV_LAST_TRADING_DATE)))
        & (ends.isna() | ends.ge(pd.Timestamp(COV_LAST_TRADING_DATE)))
    ).any():
        raise ValueError("MDT symbol history does not cover the COV completion date.")
    mdt_sessions = set(
        pd.to_datetime(
            prices.loc[prices["security_id"].astype(str).eq(mdt_id), "session"],
            errors="coerce",
        )
        .dropna()
        .dt.date.astype(str)
    )
    required_mdt_sessions = {
        COV_LAST_TRADING_DATE,
        COV_FIRST_SUCCESSOR_SESSION,
    }
    if not required_mdt_sessions.issubset(mdt_sessions):
        raise ValueError(
            "MDT successor prices must contain the 2015-01-26 close and "
            "2015-01-27 first successor session."
        )

    gaps = active_price_gaps(
        master,
        prices,
        completed_session=completed_session,
    )
    if {item["security_id"] for item in gaps} != {cov_id} or len(gaps) != 1:
        raise ValueError(
            "Current active missing/stale identities are not exactly COV: "
            f"{gaps}."
        )
    if MISSING_PROVIDER_WARNING not in release_warnings:
        raise ValueError(
            "Current release does not carry the expected one-symbol provider warning."
        )
    return cov_id, mdt_id


def build_local_preflight(
    repository: LocalDatasetRepository,
    release: DataRelease,
    report: dict[str, Any],
) -> LocalPreflight:
    _validate_spec_invariants()
    catalogs = load_archived_catalogs(repository, release)
    selections = select_catalog_entries(catalogs)
    cov_selection = select_catalog_entries(catalogs, (COV_SPEC,))["COV"]
    existing = _read_release_frames(repository, release)
    purge_ids = resolve_purge_security_ids(report, existing["security_master"])
    pointer_etags = _capture_release_pointer_etags(repository, release)
    actavis_id, dwdp_id, repaired_ids = validate_local_preflight(
        existing,
        purge_ids,
    )
    cov_id, mdt_id = validate_cov_local_preflight(
        existing,
        cov_selection,
        completed_session=release.completed_session,
        release_warnings=tuple(release.warnings),
    )
    return LocalPreflight(
        selections=selections,
        existing=existing,
        purge_ids=purge_ids,
        pointer_etags=pointer_etags,
        actavis_id=actavis_id,
        dwdp_id=dwdp_id,
        repaired_identity_ids=repaired_ids,
        cov_selection=cov_selection,
        cov_security_id=cov_id,
        mdt_security_id=mdt_id,
    )


def validate_enb_local_preflight(
    existing: dict[str, pd.DataFrame],
    selection: CatalogSelection,
    *,
    completed_session: str,
    release_warnings: tuple[str, ...],
) -> None:
    """Require the exact repaired Spectra lineage before an ENB provider call."""

    if selection.spec != ENB_SPEC or selection.archive.kind != "active":
        raise ValueError("ENB must resolve from exactly one archived active catalog row.")
    if "enbridge" not in str(selection.row.get("Name") or "").casefold():
        raise ValueError("The archived ENB catalog row is not Enbridge.")
    if IDENTITY_REPAIR_MIGRATION_WARNING in release_warnings:
        raise ValueError(
            "ENB collection must run after the audited US identity repair is applied."
        )

    master = existing["security_master"]
    history = existing["symbol_history"]
    prices = existing["daily_price_raw"]
    archive = existing["source_archive"]
    spectra = master.loc[
        master["security_id"].astype(str).eq(SPECTRA_SECURITY_ID)
        & master["primary_symbol"].astype(str).str.upper().eq("SE")
        & master["name"].astype(str).str.casefold().str.contains("spectra energy", regex=False)
    ]
    if len(spectra) != 1:
        raise ValueError(
            "The exact repaired Spectra Energy identity is missing or ambiguous: "
            f"{SPECTRA_SECURITY_ID}."
        )
    if str(spectra.iloc[0].get("active_to") or "") != SPECTRA_MASTER_ACTIVE_TO:
        raise ValueError("Spectra security_master must close on the merger effective date.")

    spectra_history = history.loc[
        history["security_id"].astype(str).eq(SPECTRA_SECURITY_ID)
        & history["symbol"].astype(str).str.upper().eq("SE")
    ]
    if len(spectra_history) != 1 or str(
        spectra_history.iloc[0].get("effective_to") or ""
    ) != SPECTRA_SYMBOL_HISTORY_END:
        raise ValueError(
            "Spectra symbol history must end immediately before the merger effective date."
        )
    spectra_sessions = pd.to_datetime(
        prices.loc[
            prices["security_id"].astype(str).eq(SPECTRA_SECURITY_ID), "session"
        ],
        errors="coerce",
    ).dropna()
    if spectra_sessions.empty or spectra_sessions.max().date().isoformat() != (
        SPECTRA_LAST_TRADING_DATE
    ):
        raise ValueError("Spectra prices must end exactly on 2017-02-24.")

    enb_master_mask = (
        master["security_id"].astype(str).eq(ENB_SPEC.security_id)
        | master["primary_symbol"].astype(str).str.upper().eq("ENB")
        | master.get("provider_symbol", pd.Series("", index=master.index))
        .astype(str)
        .str.upper()
        .eq(ENB_SPEC.provider_symbol)
    )
    if enb_master_mask.any():
        raise ValueError("ENB successor identity is already present; refusing a duplicate insert.")
    if history["symbol"].astype(str).str.upper().eq("ENB").any():
        raise ValueError("ENB symbol history already exists without the audited successor insert.")
    if prices["security_id"].astype(str).eq(ENB_SPEC.security_id).any():
        raise ValueError("ENB prices already exist without the audited successor insert.")
    official_rows = archive.loc[
        archive["archive_id"].astype(str).eq(ENB_SEC_COMPLETION_SHA256)
        & archive["source_hash"].astype(str).eq(ENB_SEC_COMPLETION_SHA256)
        & archive["source_url"].astype(str).eq(ENB_SEC_COMPLETION_URL)
        & archive["dataset"].astype(str).eq("official_identity_evidence_raw")
    ]
    if len(official_rows) != 1:
        raise ValueError(
            "Post-identity source_archive must retain the exact Spectra SEC "
            "completion artifact."
        )
    gaps = active_price_gaps(
        master,
        prices,
        completed_session=completed_session,
    )
    if gaps:
        raise ValueError(
            "ENB collection requires a post-identity snapshot with no active price gaps: "
            f"{gaps}."
        )


def build_enb_preflight(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> EnbPreflight:
    _validate_spec_invariants()
    catalogs = load_archived_catalogs(repository, release)
    selection = select_catalog_entries(catalogs, (ENB_SPEC,))["ENB"]
    existing = _read_release_frames(repository, release)
    validate_enb_local_preflight(
        existing,
        selection,
        completed_session=release.completed_session,
        release_warnings=tuple(release.warnings),
    )
    return EnbPreflight(
        selection=selection,
        existing=existing,
        pointer_etags=_capture_release_pointer_etags(repository, release),
        spectra_security_id=SPECTRA_SECURITY_ID,
        official_artifact=_load_enb_official_evidence(repository.root),
    )


def build_enb_offline_plan(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    preflight = build_enb_preflight(repository, release)
    cache_path = _enb_bundle_cache_path(repository, release)
    return {
        "status": "offline_plan",
        "mode": "enb_only",
        "eodhd_accessed": False,
        "release_version": release.version,
        "completed_session": release.completed_session,
        "maximum_eodhd_http_attempts": ENB_MAX_EODHD_HTTP_ATTEMPTS,
        "maximum_eodhd_billing_units": ENB_MAX_EODHD_HTTP_ATTEMPTS,
        "fetch_flag": "--fetch-enb",
        "bundle_cache": str(cache_path),
        "bundle_cache_exists": cache_path.is_file(),
        "successor": {
            "symbol": ENB_SPEC.symbol,
            "security_id": ENB_SPEC.security_id,
            "provider_symbol": ENB_SPEC.provider_symbol,
            "catalog": preflight.selection.archive.kind,
            "catalog_name": preflight.selection.row.get("Name"),
            "history_start": FETCH_START,
        },
        "terminal_candidate": {
            "symbol": "SE",
            "security_id": preflight.spectra_security_id,
            "last_trading_date": SPECTRA_LAST_TRADING_DATE,
            "effective_date": ENB_EFFECTIVE_DATE,
            "ratio": ENB_RATIO,
        },
        "official_completion_evidence": {
            "source_url": preflight.official_artifact.source_url,
            "source_sha256": preflight.official_artifact.source_hash,
            "cache_verified": True,
        },
        "official_terms_evidence_url": ENB_SEC_TERMS_URL,
        "execution_order": [
            "apply_us_index_identity_repair",
            "fetch_or_replay_enb_three_endpoint_bundle",
            "dry_run_enb_only",
            "apply_enb_only",
            "collect_current_release_lifecycle_report",
        ],
    }


def build_offline_plan(
    repository: LocalDatasetRepository,
    release: DataRelease,
    report: dict[str, Any],
) -> dict[str, Any]:
    preflight = build_local_preflight(repository, release, report)
    selections = preflight.selections
    purge_ids = preflight.purge_ids
    return {
        "status": "offline_plan",
        "eodhd_accessed": False,
        "release_version": release.version,
        "completed_session": release.completed_session,
        "expected_eodhd_calls": MAX_EODHD_HTTP_ATTEMPTS,
        "maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "cov_supplemental_expected_eodhd_calls": 0,
        "cov_supplemental_maximum_eodhd_http_attempts": 0,
        "combined_maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "cov_stooq_failure_evidence_cache": str(
            repository.root / "state/stooq-cov-lifecycle"
        ),
        "cov_stooq_failure_evidence_cache_exists": bool(
            tuple(
                (repository.root / "state/stooq-cov-lifecycle").glob("*.json.gz")
            )
        ),
        "cov_stooq_execution_disabled": True,
        "cov_eodhd_status": "known_unavailable_never_retry",
        "cov_eodhd_ledger_usage_after_failures": COV_EODHD_LEDGER_USAGE_AFTER_FAILURES,
        "cov_eodhd_symbol_failure_cache": str(
            repository.root / "state/eodhd-cov-bulk/2015-01-02.json.gz"
        ),
        "cov_eodhd_symbol_failure_cache_exists": (
            repository.root / "state/eodhd-cov-bulk/2015-01-02.json.gz"
        ).is_file(),
        "cov_eodhd_full_us_failure_cache": str(
            _cov_eodhd_full_us_failure_cache_path(repository.root)
        ),
        "cov_eodhd_full_us_failure_cache_exists": (
            _cov_eodhd_full_us_failure_cache_path(repository.root).is_file()
        ),
        "cov_eodhd_full_us_failure_import_required": not (
            _cov_eodhd_full_us_failure_cache_path(repository.root).is_file()
        ),
        "cov_primary_provider": "directindex_pinned_csv",
        "cov_primary_commit": COV_DIRECTINDEX_COMMIT,
        "cov_primary_artifact_sha256": dict(COV_DIRECTINDEX_SHA256),
        "cov_primary_maximum_http_attempts": COV_DIRECTINDEX_MAX_HTTP_ATTEMPTS,
        "cov_primary_urls": dict(COV_DIRECTINDEX_URLS),
        "cov_primary_raw_cache": str(repository.root / "state/cov-directindex"),
        "cov_primary_raw_cache_count": len(
            tuple((repository.root / "state/cov-directindex").glob("*.json.gz"))
        ),
        "cov_independent_provider": "quandl_wiki_pinned_csv",
        "cov_independent_commit": COV_WIKI_COMMIT,
        "cov_independent_csv_sha256": COV_WIKI_CSV_SHA256,
        "cov_independent_artifact_sha256": dict(COV_WIKI_SHA256),
        "cov_independent_maximum_http_attempts": COV_WIKI_MAX_HTTP_ATTEMPTS,
        "cov_independent_urls": dict(COV_WIKI_URLS),
        "cov_independent_raw_cache": str(
            repository.root / "state/cov-quandl-wiki"
        ),
        "cov_independent_raw_cache_count": len(
            tuple((repository.root / "state/cov-quandl-wiki").glob("*.json.gz"))
        ),
        "primary_bundle_cache": str(_bundle_cache_path(repository, release)),
        "primary_bundle_cache_exists": _bundle_cache_path(repository, release).is_file(),
        "cov_supplemental_cache": str(
            _cov_bundle_cache_path(repository, release)
        ),
        "cov_supplemental_cache_exists": _cov_bundle_cache_path(
            repository, release
        ).is_file(),
        "cov_supplement": {
            "symbol": COV_SPEC.symbol,
            "security_id": preflight.cov_security_id,
            "provider_symbol": COV_SPEC.provider_symbol,
            "catalog": preflight.cov_selection.archive.kind,
            "catalog_name": preflight.cov_selection.row.get("Name"),
            "catalog_isin": preflight.cov_selection.row.get("Isin"),
            "fetch_start": COV_FETCH_START,
            "last_trading_date": COV_LAST_TRADING_DATE,
            "expected_session_count": len(COV_EXPECTED_SESSIONS),
            "successor_symbol": "MDT",
            "successor_security_id": preflight.mdt_security_id,
            "cash_amount": 35.19,
            "ratio": 0.956,
            "sec_completion_url": COV_SEC_COMPLETION_URL,
        },
        "successors": [
            {
                "symbol": spec.symbol,
                "security_id": spec.security_id,
                "provider_symbol": spec.provider_symbol,
                "event_date": spec.event_date,
                "history_start": spec.history_start,
                "catalog": selections[spec.symbol].archive.kind,
                "catalog_name": selections[spec.symbol].row.get("Name"),
            }
            for spec in SUCCESSOR_SPECS
        ],
        "purges": [
            {
                "symbol": spec.symbol,
                "security_id": purge_ids[spec.symbol],
                "cutoff": spec.cutoff,
            }
            for spec in PURGE_SPECS
        ],
        "actavis_history_repair_id": preflight.actavis_id,
        "dwdp_repair_id": preflight.dwdp_id,
        "existing_identity_repairs": preflight.repaired_identity_ids,
        "dwdp_start": DWDP_START,
    }


def prepare_collection(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
    report: dict[str, Any],
    evidence_artifact: SourceArtifact,
    source,
    cov_direct_source=None,
    cov_wiki_source=None,
    cov_full_us_failure_import: str | Path | None = None,
) -> PreparedCollection:
    preflight = build_local_preflight(repository, release, report)
    cache_path = _bundle_cache_path(repository, release)
    try:
        cached = _read_fetched_bundle_cache(cache_path, release)
        if cached is not None:
            validate_fetched_result(
                cached[0],
                completed_session=release.completed_session,
            )
    except Exception:
        if cache_path.exists():
            quarantine = cache_path.with_name(
                f"{cache_path.name}.invalid-{uuid.uuid4().hex}"
            )
            cache_path.replace(quarantine)
        cached = None
    if cached is None:
        fetched = source.fetch(
            {spec.security_id: spec.provider_symbol for spec in SUCCESSOR_SPECS},
            start=FETCH_START,
            end=release.completed_session,
        )
        attempts_this_run = _source_attempt_count(source, fetched)
        if attempts_this_run > MAX_EODHD_HTTP_ATTEMPTS:
            raise RuntimeError(
                "EODHD lifecycle HTTP attempt cap exceeded: "
                f"actual={attempts_this_run}, maximum={MAX_EODHD_HTTP_ATTEMPTS}."
            )
        validate_fetched_result(
            fetched,
            completed_session=release.completed_session,
        )
        _write_fetched_bundle_cache(
            cache_path,
            release,
            fetched,
            http_attempts=attempts_this_run,
        )
        bundle_attempts = attempts_this_run
        cache_reused = False
    else:
        fetched, bundle_attempts = cached
        attempts_this_run = 0
        cache_reused = True
    if bundle_attempts != MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError(
            "Fetched bundle does not represent exactly one audited endpoint attempt: "
            f"actual={bundle_attempts}, expected={MAX_EODHD_HTTP_ATTEMPTS}."
        )

    # COV has already exhausted its three single-symbol EODHD endpoint calls.
    # The immutable response bundle is retained as evidence and is never
    # fetched or repaired by this collector.
    cov_cache_path = _cov_bundle_cache_path(repository, release)
    cov_signature = _cov_bundle_signature(release)
    cov_cached = _read_fetched_bundle_cache(
        cov_cache_path,
        release,
        signature=cov_signature,
    )
    if cov_cached is None:
        raise FileNotFoundError(
            "The preserved three-response COV EODHD cache is required; "
            "COV EODHD is known unavailable and is never retried."
        )
    cov_eodhd_fetched, cov_bundle_attempts = cov_cached
    cov_attempts_this_run = 0
    cov_cache_reused = True
    if cov_bundle_attempts != COV_MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError(
            "COV supplemental bundle does not represent exactly one audited "
            "attempt per endpoint: "
            f"actual={cov_bundle_attempts}, "
            f"expected={COV_MAX_EODHD_HTTP_ATTEMPTS}."
        )
    _validate_artifact_coverage(tuple(cov_eodhd_fetched.artifacts), (COV_SPEC,))

    cov_symbol_failure_artifact = _read_cov_eodhd_symbol_failure_cache(
        repository.root
    )
    cov_full_us_failure_artifact = _load_or_import_cov_eodhd_full_us_failure(
        repository.root,
        cov_full_us_failure_import,
    )
    if cov_direct_source is None:
        raise RuntimeError("The pinned DirectIndex COV primary source is required.")
    if cov_wiki_source is None:
        raise RuntimeError("The pinned Quandl WIKI COV cross-check is required.")
    direct_fetched = cov_direct_source.fetch()
    cov_direct_attempts_this_run = int(
        getattr(cov_direct_source, "attempt_count", direct_fetched.http_attempts)
    )
    if cov_direct_attempts_this_run > COV_DIRECTINDEX_MAX_HTTP_ATTEMPTS:
        raise RuntimeError("COV DirectIndex request cap was exceeded.")
    wiki_fetched = cov_wiki_source.fetch()
    cov_wiki_attempts_this_run = int(
        getattr(cov_wiki_source, "attempt_count", wiki_fetched.http_attempts)
    )
    if cov_wiki_attempts_this_run > COV_WIKI_MAX_HTTP_ATTEMPTS:
        raise RuntimeError("COV Quandl WIKI request cap was exceeded.")
    cov_cross_validation = cross_validate_cov_directindex_with_wiki(
        direct_fetched,
        wiki_fetched,
    )
    cov_direct_artifacts = tuple(direct_fetched.artifacts)
    cov_wiki_artifacts = tuple(wiki_fetched.artifacts)
    cov_fetched = FetchedBundle(
        prices=direct_fetched.prices.copy(),
        corporate_actions=cov_eodhd_fetched.corporate_actions.copy(),
        artifacts=tuple(
            (
                *cov_eodhd_fetched.artifacts,
                *cov_direct_artifacts,
                *cov_wiki_artifacts,
            )
        ),
        missing_symbols=(),
    )
    validate_cov_fetched_result(cov_fetched)
    cov_price_source = "directindex_pinned_csv"

    stamp = utc_now_iso()
    frames, stats = rewrite_market_frames(
        preflight.existing,
        preflight.selections,
        preflight.purge_ids,
        fetched,
        completed_session=release.completed_session,
        stamp=stamp,
    )
    frames, cov_stats = apply_cov_supplement(
        frames,
        cov_fetched,
        cov_security_id=preflight.cov_security_id,
        stamp=stamp,
    )
    stats.update(cov_stats)
    source_version = f"lifecycle-successors:{release.version}"
    frames["adjustment_factors"] = build_adjustment_factors(
        frames["daily_price_raw"],
        frames["corporate_actions"],
        source_version=source_version,
    )
    raw_artifacts = (
        *tuple(fetched.artifacts),
        *tuple(cov_fetched.artifacts),
        cov_symbol_failure_artifact,
        cov_full_us_failure_artifact,
    )
    request_artifacts = tuple(
        _request_archive_artifact(item) for item in raw_artifacts
    )
    cov_official_artifact = _cov_official_evidence_artifact(retrieved_at=stamp)
    cov_cross_artifacts: tuple[SourceArtifact, ...] = ()
    if cov_cross_validation.get("cov_cross_validated") is True:
        cov_cross_artifacts = (
            _cov_cross_validation_evidence_artifact(
                cov_cross_validation,
                direct_artifacts=cov_direct_artifacts,
                wiki_artifacts=cov_wiki_artifacts,
                retrieved_at=stamp,
            ),
        )
    cov_eodhd_failure_manifest = _cov_eodhd_failure_manifest_artifact(
        cov_symbol_failure_artifact,
        cov_full_us_failure_artifact,
        retrieved_at=stamp,
    )
    archive_artifacts = (
        *request_artifacts,
        evidence_artifact,
        cov_official_artifact,
        *cov_cross_artifacts,
        cov_eodhd_failure_manifest,
    )
    archive_delta = _artifact_rows(archive_artifacts, release.completed_session)
    frames["source_archive"] = _concat_deduplicated(
        preflight.existing["source_archive"],
        archive_delta,
        dataset="source_archive",
    )
    candidate_warnings = validate_candidate_frames(
        frames, completed_session=release.completed_session
    )
    warnings = tuple(
        dict.fromkeys(
            (
                *candidate_warnings,
                *(
                    ()
                    if cov_cross_validation.get("cov_cross_validated") is True
                    else (COV_PENDING_INDEPENDENT_VALIDATION_WARNING,)
                ),
            )
        )
    )
    remaining_active_gaps = active_price_gaps(
        frames["security_master"],
        frames["daily_price_raw"],
        completed_session=release.completed_session,
    )
    if remaining_active_gaps:
        raise ValueError(
            "COV supplement did not clear every active missing/stale security: "
            f"{remaining_active_gaps}."
        )
    _assert_release_unchanged(repository, release, release_etag)
    total_attempts_this_run = attempts_this_run
    total_billing_units_this_run = attempts_this_run
    cleared_release_warnings = tuple(
        warning
        for warning in (
            MISSING_PROVIDER_WARNING,
            COV_PENDING_INDEPENDENT_VALIDATION_WARNING,
        )
        if warning in release.warnings
        and (
            warning == MISSING_PROVIDER_WARNING
            or cov_cross_validation.get("cov_cross_validated") is True
        )
    )
    summary = {
        "status": "validated_dry_run",
        "eodhd_accessed": bool(total_attempts_this_run),
        "release_version": release.version,
        "completed_session": release.completed_session,
        "expected_eodhd_calls": MAX_EODHD_HTTP_ATTEMPTS,
        "maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "actual_eodhd_http_attempts_this_run": attempts_this_run,
        "bundle_eodhd_http_attempts": bundle_attempts,
        "fetched_bundle_cache": str(cache_path),
        "fetched_bundle_reused": cache_reused,
        "cov_supplemental_maximum_eodhd_http_attempts": 0,
        "cov_supplemental_actual_eodhd_http_attempts_this_run": cov_attempts_this_run,
        "cov_supplemental_bundle_eodhd_http_attempts": cov_bundle_attempts,
        "cov_supplemental_cache": str(cov_cache_path),
        "cov_supplemental_cache_reused": cov_cache_reused,
        "cov_price_source": cov_price_source,
        "cov_eodhd_status": "known_unavailable_never_retry",
        "cov_eodhd_ledger_usage_after_failures": COV_EODHD_LEDGER_USAGE_AFTER_FAILURES,
        "cov_eodhd_symbol_failure_url": COV_EODHD_SYMBOL_FAILURE_URL,
        "cov_eodhd_symbol_failure_sha256": cov_symbol_failure_artifact.source_hash,
        "cov_eodhd_symbol_failure_cache": str(
            repository.root / "state/eodhd-cov-bulk/2015-01-02.json.gz"
        ),
        "cov_eodhd_full_us_failure_url": COV_EODHD_FULL_US_FAILURE_URL,
        "cov_eodhd_full_us_failure_sha256": cov_full_us_failure_artifact.source_hash,
        "cov_eodhd_full_us_failure_rows": COV_EODHD_FULL_US_FAILURE_ROWS,
        "cov_eodhd_full_us_failure_cache": str(
            _cov_eodhd_full_us_failure_cache_path(repository.root)
        ),
        "cov_eodhd_failure_manifest_hash": cov_eodhd_failure_manifest.source_hash,
        "cov_stooq_execution_disabled": True,
        "cov_primary_maximum_http_attempts": COV_DIRECTINDEX_MAX_HTTP_ATTEMPTS,
        "cov_primary_actual_http_attempts_this_run": cov_direct_attempts_this_run,
        "cov_primary_commit": COV_DIRECTINDEX_COMMIT,
        "cov_primary_csv_sha256": COV_DIRECTINDEX_SHA256["cov_csv"],
        "cov_primary_artifact_sha256": dict(COV_DIRECTINDEX_SHA256),
        "cov_independent_maximum_http_attempts": COV_WIKI_MAX_HTTP_ATTEMPTS,
        "cov_independent_actual_http_attempts_this_run": cov_wiki_attempts_this_run,
        "cov_independent_commit": COV_WIKI_COMMIT,
        "cov_independent_csv_sha256": COV_WIKI_CSV_SHA256,
        "cov_independent_artifact_sha256": dict(COV_WIKI_SHA256),
        "cov_cross_validated": bool(
            cov_cross_validation.get("cov_cross_validated") is True
        ),
        "cov_cross_validation": cov_cross_validation,
        "cov_cross_validation_evidence_hash": (
            cov_cross_artifacts[0].source_hash if cov_cross_artifacts else ""
        ),
        "network_accessed": bool(
            total_attempts_this_run
            or cov_direct_attempts_this_run
            or cov_wiki_attempts_this_run
        ),
        "combined_maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "combined_actual_eodhd_http_attempts_this_run": total_attempts_this_run,
        "combined_maximum_eodhd_billing_units": MAX_EODHD_HTTP_ATTEMPTS,
        "combined_actual_eodhd_billing_units_this_run": (
            total_billing_units_this_run
        ),
        "artifact_count": len(raw_artifacts),
        "archived_artifact_count": len(archive_artifacts),
        "archive_index_delta_rows": len(archive_delta),
        "row_counts": {
            dataset: len(frames[dataset]) for dataset in WRITE_DATASETS
        },
        "repairs": stats,
        "active_missing_or_stale_after_supplement": len(remaining_active_gaps),
        "cleared_release_warnings": list(cleared_release_warnings),
        "warnings": list(warnings),
    }
    return PreparedCollection(
        release=release,
        release_etag=release_etag,
        pointer_etags=preflight.pointer_etags,
        frames=frames,
        artifacts=raw_artifacts,
        archive_artifacts=archive_artifacts,
        warnings=warnings,
        summary=summary,
        cleared_release_warnings=cleared_release_warnings,
    )


def prepare_enb_collection(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
    source=None,
) -> PreparedCollection:
    """Prepare only the exact ENB successor insert; never touch frozen bundles."""

    preflight = build_enb_preflight(repository, release)
    signature = _enb_bundle_signature(release)
    cache_path = _enb_bundle_cache_path(repository, release)
    try:
        cached = _read_fetched_bundle_cache(
            cache_path,
            release,
            signature=signature,
        )
        if cached is not None:
            validate_fetched_result(
                cached[0],
                completed_session=release.completed_session,
                specs=(ENB_SPEC,),
            )
    except Exception:
        # An offline replay is fail-closed and leaves the suspect evidence in
        # place.  A user-authorized refetch may quarantine it before exactly
        # one new three-endpoint attempt.
        if source is None:
            raise
        if cache_path.exists():
            quarantine = cache_path.with_name(
                f"{cache_path.name}.invalid-{uuid.uuid4().hex}"
            )
            cache_path.replace(quarantine)
        cached = None
    if cached is None:
        if source is None:
            raise FileNotFoundError(
                "The immutable ENB EODHD bundle is missing. Re-run with "
                "--enb-only --fetch-enb to authorize at most three calls."
            )
        fetched = source.fetch(
            {ENB_SPEC.security_id: ENB_SPEC.provider_symbol},
            start=FETCH_START,
            end=release.completed_session,
        )
        attempts_this_run = _source_attempt_count(source, fetched)
        if attempts_this_run > ENB_MAX_EODHD_HTTP_ATTEMPTS:
            raise RuntimeError(
                "ENB EODHD HTTP attempt cap exceeded: "
                f"actual={attempts_this_run}, maximum={ENB_MAX_EODHD_HTTP_ATTEMPTS}."
            )
        validate_fetched_result(
            fetched,
            completed_session=release.completed_session,
            specs=(ENB_SPEC,),
        )
        _write_fetched_bundle_cache(
            cache_path,
            release,
            fetched,
            http_attempts=attempts_this_run,
            signature=signature,
        )
        bundle_attempts = attempts_this_run
        cache_reused = False
    else:
        fetched, bundle_attempts = cached
        attempts_this_run = 0
        cache_reused = True
    if bundle_attempts != ENB_MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError(
            "ENB bundle must contain exactly one audited attempt per endpoint: "
            f"actual={bundle_attempts}, expected={ENB_MAX_EODHD_HTTP_ATTEMPTS}."
        )

    frames, stats = apply_enb_supplement(
        preflight.existing,
        preflight.selection,
        fetched,
        completed_session=release.completed_session,
    )
    source_version = f"lifecycle-successors-enb:{release.version}"
    frames["adjustment_factors"] = build_adjustment_factors(
        frames["daily_price_raw"],
        frames["corporate_actions"],
        source_version=source_version,
    )
    raw_artifacts = tuple(fetched.artifacts)
    request_artifacts = tuple(
        _request_archive_artifact(item) for item in raw_artifacts
    )
    archive_artifacts = (*request_artifacts, preflight.official_artifact)
    existing_archive_ids = set(
        preflight.existing["source_archive"]["archive_id"].astype(str)
    )
    archive_index_artifacts = tuple(
        item
        for item in archive_artifacts
        if item.source_hash not in existing_archive_ids
    )
    archive_delta = _artifact_rows(
        archive_index_artifacts,
        release.completed_session,
    )
    frames["source_archive"] = _concat_deduplicated(
        preflight.existing["source_archive"],
        archive_delta,
        dataset="source_archive",
    ).reset_index(drop=True)
    warnings = validate_enb_candidate_frames(
        frames,
        completed_session=release.completed_session,
    )
    remaining_active_gaps = active_price_gaps(
        frames["security_master"],
        frames["daily_price_raw"],
        completed_session=release.completed_session,
    )
    if remaining_active_gaps:
        raise ValueError(
            "ENB supplement left active missing/stale identities: "
            f"{remaining_active_gaps}."
        )
    _assert_release_unchanged(repository, release, release_etag)
    summary = {
        "status": "validated_dry_run",
        "mode": "enb_only",
        "eodhd_accessed": bool(attempts_this_run),
        "network_accessed": bool(attempts_this_run),
        "release_version": release.version,
        "completed_session": release.completed_session,
        "expected_eodhd_calls": ENB_MAX_EODHD_HTTP_ATTEMPTS,
        "maximum_eodhd_http_attempts": ENB_MAX_EODHD_HTTP_ATTEMPTS,
        "actual_eodhd_http_attempts_this_run": attempts_this_run,
        "bundle_eodhd_http_attempts": bundle_attempts,
        "maximum_eodhd_billing_units": ENB_MAX_EODHD_HTTP_ATTEMPTS,
        "actual_eodhd_billing_units_this_run": attempts_this_run,
        "fetched_bundle_cache": str(cache_path),
        "fetched_bundle_reused": cache_reused,
        "spectra_security_id": SPECTRA_SECURITY_ID,
        "spectra_last_trading_date": SPECTRA_LAST_TRADING_DATE,
        "enb_security_id": ENB_SPEC.security_id,
        "enb_provider_symbol": ENB_SPEC.provider_symbol,
        "effective_date": ENB_EFFECTIVE_DATE,
        "ratio": ENB_RATIO,
        "official_completion_evidence_url": preflight.official_artifact.source_url,
        "official_completion_evidence_sha256": preflight.official_artifact.source_hash,
        "official_completion_evidence_cache_verified": True,
        "official_terms_evidence_url": ENB_SEC_TERMS_URL,
        "official_terms_evidence_status": (
            "exact_url_bound_for_current_release_lifecycle_collection"
        ),
        "artifact_count": len(raw_artifacts),
        "archived_artifact_count": len(archive_artifacts),
        "archive_index_delta_rows": len(archive_delta),
        "row_counts": {
            dataset: len(frames[dataset]) for dataset in WRITE_DATASETS
        },
        "repairs": stats,
        "active_missing_or_stale_after_supplement": 0,
        "cleared_release_warnings": [],
        "warnings": list(warnings),
    }
    return PreparedCollection(
        release=release,
        release_etag=release_etag,
        pointer_etags=preflight.pointer_etags,
        frames=frames,
        artifacts=raw_artifacts,
        archive_artifacts=archive_artifacts,
        warnings=warnings,
        summary=summary,
        cleared_release_warnings=(),
    )


def _artifact_extension(artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower()
    if "json" in content_type:
        return "json"
    if "pdf" in content_type:
        return "pdf"
    if "html" in content_type:
        return "html"
    if "csv" in content_type:
        return "csv"
    return "txt"


def _persist_archive_payloads(
    repository: LocalDatasetRepository,
    artifacts: tuple[SourceArtifact, ...],
    completed_session: str,
) -> None:
    for artifact in artifacts:
        object_path = (
            repository.root
            / "archives"
            / completed_session
            / f"{artifact.source_hash}.{_artifact_extension(artifact)}.gz"
        )
        if object_path.is_file():
            try:
                existing = gzip.decompress(object_path.read_bytes())
            except Exception as exc:
                raise RuntimeError(
                    f"Existing archive payload is unreadable: {object_path}"
                ) from exc
            if existing != artifact.content:
                raise RuntimeError(
                    f"Existing archive payload conflicts with content hash: {object_path}"
                )
            continue
        write_atomic(object_path, gzip.compress(artifact.content, mtime=0))
        if gzip.decompress(object_path.read_bytes()) != artifact.content:
            raise RuntimeError(f"Archive payload verification failed: {object_path}")


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    lock_path = repository.root / ".locks/market-store-write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery_root = repository.root / "recovery/lifecycle-successors"
        pending = tuple(recovery_root.glob("*.json")) if recovery_root.exists() else ()
        if pending:
            raise RuntimeError(
                "Lifecycle-successor recovery marker blocks writes: "
                + ", ".join(str(path) for path in pending)
            )
        transaction_root = repository.root / "transactions/lifecycle-successors"
        interrupted: list[Path] = []
        if transaction_root.exists():
            for path in transaction_root.glob("*.json"):
                try:
                    status = str(json.loads(path.read_bytes()).get("status") or "")
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    interrupted.append(path)
        if interrupted:
            raise RuntimeError(
                "Interrupted lifecycle-successor transaction requires recovery: "
                + ", ".join(str(path) for path in interrupted)
            )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_transaction_record(path: Path, value: dict[str, Any]) -> None:
    write_atomic(
        path,
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode(),
    )


def _restore_transaction_state(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: dict[str, bytes],
    planned_versions: dict[str, str],
    committed_release_version: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    release_key = "releases/current.json"
    try:
        current = repository.objects.get(release_key)
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            is_transaction_release = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or all(
                observed.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not is_transaction_release:
                raise RuntimeError(
                    f"unexpected release during rollback: {observed.version}"
                )
            repository.objects.put(
                release_key,
                old_release_bytes,
                if_match=current.etag,
            )
        if repository.objects.get(release_key).data != old_release_bytes:
            raise RuntimeError("release preimage verification failed")
    except Exception as exc:
        errors.append(f"{release_key}: {type(exc).__name__}: {exc}")

    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        old_bytes = old_pointer_bytes[dataset]
        try:
            current = repository.objects.get(key)
            if current.data != old_bytes:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected pointer version during rollback: {pointer.version}"
                    )
                repository.objects.put(key, old_bytes, if_match=current.etag)
            if repository.objects.get(key).data != old_bytes:
                raise RuntimeError("pointer preimage verification failed")
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release_invariant(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> None:
    current, _etag = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes():
        raise RuntimeError("Committed lifecycle release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, _pointer_etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            actual = pointer.version if pointer is not None else "missing"
            raise RuntimeError(
                f"Applied release pointer mismatch for {dataset}: "
                f"expected={version}, actual={actual}."
            )
    post = validate_repository_snapshot(
        repository,
        allowed_index_price_gap_ids=IDENTITY_REPAIR_MIGRATION_GAP_IDS,
    )
    post.raise_for_errors()


def apply_collection(
    repository: LocalDatasetRepository,
    prepared: PreparedCollection,
) -> dict[str, Any]:
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(
            repository,
            prepared.release,
            prepared.release_etag,
        )
        old_release_value = repository.objects.get("releases/current.json")
        old_pointer_bytes: dict[str, bytes] = {}
        for dataset, expected_etag in prepared.pointer_etags.items():
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions.get(dataset)
                or value.etag != expected_etag
            ):
                raise RuntimeError(f"{dataset} pointer changed before apply.")
            old_pointer_bytes[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        planned_versions = {
            dataset: (
                "lifecycle-successors-"
                f"{prepared.release.completed_session.replace('-', '')}-"
                f"{transaction_id}-{dataset}"
            )
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/lifecycle-successors"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release_value.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value).decode("ascii")
                for dataset, value in old_pointer_bytes.items()
            },
            "planned_versions": planned_versions,
            "created_at": utc_now_iso(),
        }
        _write_transaction_record(journal_path, journal)

        committed_release: DataRelease | None = None
        try:
            _persist_archive_payloads(
                repository,
                prepared.archive_artifacts,
                prepared.release.completed_session,
            )
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="warn",
                    metadata={"operation": "collect_eodhd_lifecycle_successors"},
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned_versions[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version

            post = validate_repository_snapshot(
                repository,
                allowed_index_price_gap_ids=IDENTITY_REPAIR_MIGRATION_GAP_IDS,
            )
            post.raise_for_errors()
            cleared = set(prepared.cleared_release_warnings)
            if cleared:
                allowed_clears = {
                    MISSING_PROVIDER_WARNING,
                    COV_PENDING_INDEPENDENT_VALIDATION_WARNING,
                }
                if not cleared.issubset(allowed_clears):
                    raise RuntimeError(
                        f"Unsupported release warnings requested for clearing: {sorted(cleared)}."
                    )
                if MISSING_PROVIDER_WARNING in cleared and (
                    prepared.summary.get("active_missing_or_stale_after_supplement")
                    != 0
                    or prepared.summary.get("repairs", {}).get(
                        "cov_identity_closed_on"
                    )
                    != COV_LAST_TRADING_DATE
                ):
                    raise RuntimeError(
                        "The provider warning can only be cleared after a complete COV supplement."
                    )
                if (
                    COV_PENDING_INDEPENDENT_VALIDATION_WARNING in cleared
                    and prepared.summary.get("cov_cross_validated") is not True
                ):
                    raise RuntimeError(
                        "The COV external-validation warning can only be cleared "
                        "after the pinned independent cross-validation passes."
                    )
            inherited_warnings = tuple(
                warning
                for warning in prepared.release.warnings
                if warning not in cleared
            )
            warnings = tuple(
                dict.fromkeys(
                    (
                        *inherited_warnings,
                        *(
                            warning
                            for warning in prepared.warnings
                            if warning not in cleared
                        ),
                        *(
                            issue.message
                            for issue in post.issues
                            if issue.severity != "error"
                            and issue.message not in cleared
                        ),
                    )
                )
            )
            committed_release = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=DataQuality.DEGRADED if warnings else DataQuality.VALID,
                warnings=warnings,
                expected_etag=prepared.release_etag,
            )
            _assert_applied_release_invariant(repository, committed_release)
            journal["status"] = "committed"
            journal["committed_release_version"] = committed_release.version
            journal["completed_at"] = utc_now_iso()
            _write_transaction_record(journal_path, journal)
            return {
                **prepared.summary,
                "status": "applied",
                "new_release_version": committed_release.version,
                "quality": committed_release.quality,
                "warnings": list(committed_release.warnings),
                "transaction_id": transaction_id,
            }
        except BaseException as original:
            rollback_errors = _restore_transaction_state(
                repository,
                old_release_bytes=old_release_value.data,
                old_pointer_bytes=old_pointer_bytes,
                planned_versions=planned_versions,
                committed_release_version=(
                    committed_release.version if committed_release is not None else ""
                ),
            )
            journal["status"] = "rollback_failed" if rollback_errors else "rolled_back"
            journal["original_error"] = f"{type(original).__name__}: {original}"
            journal["rollback_errors"] = list(rollback_errors)
            journal["completed_at"] = utc_now_iso()
            _write_transaction_record(journal_path, journal)
            if rollback_errors:
                recovery_path = (
                    repository.root
                    / "recovery/lifecycle-successors"
                    / f"{transaction_id}.json"
                )
                _write_transaction_record(recovery_path, journal)
                raise RuntimeError(
                    "Lifecycle-successor rollback failed; recovery marker blocks "
                    f"further writes: {recovery_path}; errors={rollback_errors}"
                ) from original
            raise


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[[str | Path], LocalDatasetRepository] = LocalDatasetRepository,
    source_factory: Callable[..., Any] = CappedEodhdDailySource,
    cov_direct_source_factory: Callable[..., Any] = CappedCovDirectIndexSource,
    cov_wiki_source_factory: Callable[..., Any] = CappedCovQuandlWikiSource,
    enb_source_factory: Callable[..., Any] = CappedEodhdDailySource,
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local data release is required.")
    enb_only = bool(getattr(args, "enb_only", False))
    fetch_enb = bool(getattr(args, "fetch_enb", False))
    if fetch_enb and not enb_only:
        raise ValueError("--fetch-enb is only valid together with --enb-only.")
    if enb_only:
        if any(
            (
                bool(getattr(args, "fetch_cov_directindex", False)),
                bool(getattr(args, "fetch_cov_wiki", False)),
                bool(
                    str(
                        getattr(args, "cov_eodhd_full_us_failure_response", "")
                    ).strip()
                ),
            )
        ):
            raise ValueError("COV options cannot be combined with --enb-only.")
        if args.offline_plan:
            return build_enb_offline_plan(repository, release)
        enb_source = (
            enb_source_factory(
                workers=max(1, int(args.workers)),
                max_attempts=ENB_MAX_EODHD_HTTP_ATTEMPTS,
            )
            if fetch_enb
            else None
        )
        prepared = prepare_enb_collection(
            repository,
            release,
            release_etag,
            enb_source,
        )
        return apply_collection(repository, prepared) if args.apply else prepared.summary
    loaded_report = _load_evidence_report(Path(args.evidence_report))
    report = loaded_report.data
    report_release = str(report.get("release_version") or "")
    if report_release != release.version:
        raise RuntimeError(
            "Lifecycle evidence report is not for the current release: "
            f"report={report_release or 'missing'}, current={release.version}."
        )
    if args.offline_plan:
        return build_offline_plan(repository, release, report)
    source = source_factory(workers=max(1, int(args.workers)))
    cov_direct_source = cov_direct_source_factory(
        Path(args.cache_root) / "state/cov-directindex",
        allow_http=bool(getattr(args, "fetch_cov_directindex", False)),
    )
    cov_wiki_source = cov_wiki_source_factory(
        Path(args.cache_root) / "state/cov-quandl-wiki",
        allow_http=bool(getattr(args, "fetch_cov_wiki", False)),
    )
    prepared = prepare_collection(
        repository,
        release,
        release_etag,
        report,
        loaded_report.artifact,
        source,
        cov_direct_source,
        cov_wiki_source,
        getattr(args, "cov_eodhd_full_us_failure_response", ""),
    )
    return apply_collection(repository, prepared) if args.apply else prepared.summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
