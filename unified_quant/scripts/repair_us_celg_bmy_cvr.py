#!/usr/bin/env python3
"""Collect and atomically install the exact CELG/BMY/BMYRT lifecycle.

The Celgene acquisition cannot be represented by one legacy action row.  A
Celgene holder received three economically separate pieces of consideration:
one BMY share, USD 50, and one exchange-traded BMYRT contingent value right.
This repair therefore records, in this order:

1. a 1:1 BMYRT distribution on 2019-11-21;
2. a 1:1 CELG-to-BMY stock merger plus USD 50 on the same date; and
3. BMYRT termination for USD 0 on 2021-01-01.

The default command is read-only.  ``--fetch`` is the only EODHD network mode.
It never calls the search API: identity is selected from two hash-pinned frozen
US catalogs and the official SEC submitter ticker.  Fetch is hard-capped at
EOD/div/splits (three HTTP attempts total, no retries), and div/splits are
forbidden until the EOD response passes the exact reviewed price gate.
``--apply`` is offline-only and requires the hash-wrapped cached bundle produced
by a prior fetch.  R2 is never accessed.

The separate ``--official-exit-mark`` mode never uses or claims provider
OHLCV.  It records the hash-pinned BMY 2020 10-K first-trade close of USD 2.30
as one retrospective valuation/exit mark for the non-index BMYRT child, makes
the liquidation cash available on the next session, and keeps only a USD 0
residual termination guard on 2021-01-01.  The unsupported 280-session path is
therefore released only with an explicit degraded-quality warning.
All release writes use repository and pointer CAS, an exclusive writer lock,
an on-disk transaction journal, validation before commit, and rollback.
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
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import (
    EodhdCallBudget,
    EodhdClient,
    SourceArtifact,
)
from supertrend_quant.market_store.lifecycle import (
    build_lifecycle_candidates,
    canonical_lifecycle_event_id,
)
from supertrend_quant.market_store.lifecycle_coverage import (
    LifecycleCoverageReport,
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
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_EVIDENCE_DIR = DEFAULT_CACHE_ROOT / "state/celg_cvr_evidence"
DEFAULT_BUNDLE_PATH = DEFAULT_CACHE_ROOT / "state/celg_cvr/bmyrt.json.gz"

CELG_SECURITY_ID = "US:EODHD:0337dd23-67ad-5354-b972-50babd1ae5a0"
BMY_SECURITY_ID = "US:EODHD:25d16784-a5a9-5eee-bf6e-81519b64ef0b"
CELG_SYMBOL = "CELG"
BMY_SYMBOL = "BMY"
CVR_SYMBOL = "BMYRT"
CVR_PROVIDER_CODE = "CELG-RI"
CVR_PROVIDER_SYMBOL = f"{CVR_PROVIDER_CODE}.US"

CELG_LAST_SESSION = "2019-11-20"
MERGER_SESSION = "2019-11-21"
CVR_LAST_SESSION = "2020-12-31"
CVR_TERMINATION_DATE = "2021-01-01"
FETCH_START = MERGER_SESSION
FETCH_END = CVR_LAST_SESSION
EXPECTED_CVR_SESSIONS = 280
EXPECTED_EODHD_CALLS = 3
PRIOR_FAILED_SEARCH_CALLS = 1
ENDPOINTS = ("eod", "div", "splits")
REQUEST_PARAMS = {"from": FETCH_START, "to": FETCH_END}

ACTIVE_CATALOG_URL = (
    "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
)
ACTIVE_CATALOG_SHA256 = (
    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
)
DELISTED_CATALOG_URL = (
    "https://eodhd.com/api/exchange-symbol-list/US?delisted=1"
)
DELISTED_CATALOG_SHA256 = (
    "8a64e65e316b71e5d165265db2796b68a31f821812f74b63367435b8fcb2ed13"
)
EXACT_CATALOG_ROW = {
    "Code": "CELG-RI",
    "Country": "USA",
    "Currency": "USD",
    "Exchange": "NYSE",
    "Isin": "US1101221406",
    "Name": "Bristol-Myers Squibb Company Ce",
    "Type": "Common Stock",
}
SECONDARY_CATALOG_CODES = ("BMY-R", "BMY-RI")
SEC_SUBMITTER_SYMBOL = "CELG-RI"
SEC_SUBMITTER_DISPLAY_NAME = (
    "BRISTOL MYERS SQUIBB CO  (BMY, BMYMP, CELG-RI)  (CIK 0000014272)"
)
SEC_SUBMITTER_SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index?q=%22Celgene+Corporation%22"
    "&forms=8-K&startdt=2019-09-06&enddt=2020-01-04&from=0&size=50"
)
SEC_SUBMITTER_SEARCH_SHA256 = (
    "96515e8434724844d5bb0ab1bc824176396d78d212d486df06196bc42caabf19"
)
SEC_SUBMITTER_CACHE_OBJECT = (
    "state/sec_lifecycle/"
    "d0dac91335fd5b309c3405dc74a61260a47a027dba1e4502f9e0bdfbfc2cc809.bin"
)
SEC_CLOSING_SEARCH_HIT_ID = "0001140361-19-021048:form8k.htm"
DIAGNOSTIC_BUDGET_NOTE = {
    "prior_failed_search_calls": PRIOR_FAILED_SEARCH_CALLS,
    "endpoint": "search/BMYRT",
    "included_in_bundle_budget_delta": False,
    "note": "Historical failed discovery call; v2 forbids search and excludes it from the bundle.",
}
OFFICIAL_EXIT_MODE = "official_exit_mark"
OFFICIAL_EXIT_REVIEWED_BY = "celg_bmy_cvr_official_exit_mark/v1"
OFFICIAL_EXIT_POLICY_URL = (
    "supertrendquant://policy/celg-bmyrt-official-exit-mark/v1"
)
OFFICIAL_EXIT_WARNING = (
    "CELG/BMYRT official_exit_mark uses only the hash-pinned BMY 2020 10-K "
    "first-trade close of USD 2.30 as a retrospective first-session close exit "
    "mark for a non-index child, with cash available next session; the "
    "2019-11-21..2020-12-31 280-session trading path is unsupported."
)
PRIOR_FAILED_EOD_NOTE = {
    "endpoint": "eod/CELG-RI.US",
    "budget_used_before": 8835,
    "budget_used_after": 8836,
    "result": "empty_payload_exact_gate_stopped",
    "raw_response_preserved": False,
    "included_in_official_exit_evidence": False,
    "alias_fallback_forbidden": True,
}

MERGER_TERMS_URL = (
    "https://www.sec.gov/Archives/edgar/data/14272/"
    "000114036119021048/0001140361-19-021048.txt"
)
MERGER_TERMS_SHA256 = (
    "157cae6dae5486f16c63a51e61d79aab2ce2f37d0e8584337fb21d7d0ec6f211"
)
MERGER_TERMS_BYTES = 1_257_384

TERMINATION_URL = (
    "https://www.sec.gov/Archives/edgar/data/14272/"
    "000001427221000066/bmy-20201231.htm"
)
TERMINATION_SHA256 = (
    "a86e198381d31eacf1fd4b17e93e7a09c8b7f191c1941bec43467729b4f8b055"
)
TERMINATION_BYTES = 5_654_027
TERMINATION_RETRIEVED_AT = "2026-07-18T08:39:00Z"

MERGER_CASH_PER_SHARE = 50.0
BMY_REFERENCE_CLOSE = 56.48
CVR_REFERENCE_CLOSE = 2.30
TOTAL_REFERENCE_CONSIDERATION = round(
    MERGER_CASH_PER_SHARE + BMY_REFERENCE_CLOSE + CVR_REFERENCE_CLOSE,
    2,
)
CVR_ECONOMIC_BASIS_FRACTION = CVR_REFERENCE_CLOSE / TOTAL_REFERENCE_CONSIDERATION
REVIEWED_BY = "celg_bmy_cvr_exact_model/v1"
REVIEWED_AT = TERMINATION_RETRIEVED_AT
PLANNED_FACTOR_SOURCE = "__planned_celg_bmy_cvr_factor_source__"

CVR_DISTRIBUTION_EVENT_ID = canonical_lifecycle_event_id(
    CELG_SECURITY_ID, "spinoff", MERGER_SESSION
)
CELG_STOCK_MERGER_EVENT_ID = canonical_lifecycle_event_id(
    CELG_SECURITY_ID, "stock_merger", MERGER_SESSION
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
REQUIRED_DATASETS = WRITE_DATASETS


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _canonical_json(value: Any) -> str:
    return _canonical_json_bytes(value).decode()


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _date_text(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date: {value!r}")
    return pd.Timestamp(parsed).date().isoformat()


def _normalized_document_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    decoded = html.unescape(decoded)
    decoded = re.sub(r"<[^>]+>", " ", decoded)
    return re.sub(r"\s+", " ", decoded).strip().lower()


def _security_id(provider_code: str) -> str:
    value = f"eodhd:US:{provider_code}:symbol:{CVR_SYMBOL}"
    return f"US:EODHD:{uuid.uuid5(uuid.NAMESPACE_URL, value)}"


OFFICIAL_EXIT_SECURITY_ID = _security_id(CVR_PROVIDER_CODE)
OFFICIAL_EXIT_EVENT_ID = canonical_lifecycle_event_id(
    OFFICIAL_EXIT_SECURITY_ID, "delisting", MERGER_SESSION
)
OFFICIAL_RESIDUAL_TERMINATION_EVENT_ID = canonical_lifecycle_event_id(
    OFFICIAL_EXIT_SECURITY_ID, "delisting", CVR_TERMINATION_DATE
)


def _expected_sessions() -> tuple[str, ...]:
    import exchange_calendars as xcals

    sessions = xcals.get_calendar("XNYS").sessions_in_range(FETCH_START, FETCH_END)
    values = tuple(pd.Timestamp(value).date().isoformat() for value in sessions)
    if len(values) != EXPECTED_CVR_SESSIONS:
        raise RuntimeError("Pinned XNYS BMYRT session count changed.")
    return values


@dataclass(frozen=True)
class CvrBundle:
    provider_code: str
    security_id: str
    catalog_source_url: str
    catalog_source_hash: str
    catalog_row: Mapping[str, Any]
    secondary_catalog_evidence: tuple[Mapping[str, Any], ...]
    prices: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    http_attempts: int
    fetched_against_release: str
    budget_used_before: int | None = None
    budget_used_after: int | None = None


@dataclass(frozen=True)
class FrozenCatalogSelection:
    provider_code: str
    source_url: str
    source_hash: str
    row: Mapping[str, Any]
    secondary_evidence: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class OfficialExitMarkModel:
    provider_code: str
    security_id: str
    catalog_source_url: str
    catalog_source_hash: str
    catalog_row: Mapping[str, Any]
    secondary_catalog_evidence: tuple[Mapping[str, Any], ...]
    mode: str = OFFICIAL_EXIT_MODE


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    frames: Mapping[str, pd.DataFrame]
    artifacts: tuple[SourceArtifact, ...]
    coverage: LifecycleCoverageReport | None
    transaction_id: str
    planned_versions: Mapping[str, str]
    summary: Mapping[str, Any]


def _source_artifact(
    source: str,
    url: str,
    payload: Any,
    retrieved_at: str,
) -> SourceArtifact:
    return SourceArtifact(
        source=source,
        source_url=url,
        retrieved_at=retrieved_at,
        content=_canonical_json_bytes(payload),
        content_type="application/json",
    )


def _endpoint_url(endpoint: str, provider_code: str) -> str:
    return (
        f"https://eodhd.com/api/{endpoint}/{provider_code}.US?"
        f"from={FETCH_START}&to={FETCH_END}"
    )


def _validate_catalog_selection(selection: FrozenCatalogSelection) -> None:
    if selection.provider_code != CVR_PROVIDER_CODE:
        raise ValueError("Frozen catalog selection is not the exact CELG-RI code.")
    if selection.source_url != ACTIVE_CATALOG_URL:
        raise ValueError("Exact CELG-RI catalog URL is not the pinned active catalog.")
    if selection.source_hash != ACTIVE_CATALOG_SHA256:
        raise ValueError("Exact CELG-RI catalog hash is not pinned.")
    if dict(selection.row) != EXACT_CATALOG_ROW:
        raise ValueError("Exact CELG-RI catalog row changed.")
    if _text(selection.row.get("Code")) != SEC_SUBMITTER_SYMBOL:
        raise ValueError("EODHD exact code disagrees with the official SEC submitter ticker.")
    secondary = tuple(selection.secondary_evidence)
    codes: list[str] = []
    for item in secondary:
        if _text(item.get("role")) != "secondary_ambiguous":
            raise ValueError("BMY-R catalog evidence must remain secondary/ambiguous.")
        if _text(item.get("source_url")) != DELISTED_CATALOG_URL:
            raise ValueError("Secondary catalog URL is not the pinned delisted catalog.")
        if _text(item.get("source_hash")) != DELISTED_CATALOG_SHA256:
            raise ValueError("Secondary catalog hash is not pinned.")
        row = item.get("row")
        if not isinstance(row, Mapping):
            raise ValueError("Secondary catalog evidence lacks its raw row.")
        codes.append(_text(row.get("Code")))
    if tuple(sorted(codes)) != tuple(sorted(SECONDARY_CATALOG_CODES)):
        raise ValueError("Pinned BMY-R secondary catalog evidence is incomplete.")


def _artifact_rows(artifact: SourceArtifact, endpoint: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(artifact.content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"BMYRT {endpoint} raw response is not JSON.") from exc
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"BMYRT {endpoint} response must be a JSON row list.")
    return payload


def _prices_from_eod_artifact(
    artifact: SourceArtifact,
    *,
    security_id: str,
) -> pd.DataFrame:
    rows = _artifact_rows(artifact, "eod")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        session = _text(row.get("date"))
        if not session or row.get("close") is None:
            raise ValueError("BMYRT EOD row is missing date or close.")
        row_currency = _text(row.get("currency")).upper()
        if row_currency and row_currency != "USD":
            raise ValueError("BMYRT EOD response contains a non-USD row.")
        normalized.append(
            {
                "security_id": security_id,
                "session": session,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume", 0),
                "currency": "USD",
                "source": artifact.source,
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    return pd.DataFrame(normalized)


def _validate_exact_prices(prices: pd.DataFrame) -> dict[str, Any]:
    if prices.empty or prices.duplicated(["security_id", "session"], keep=False).any():
        raise ValueError("BMYRT prices are empty or contain duplicate sessions.")
    if not prices["currency"].astype(str).eq("USD").all():
        raise ValueError("BMYRT prices must be coherently denominated in USD.")
    sessions = tuple(prices["session"].astype(str).sort_values())
    expected_sessions = _expected_sessions()
    if sessions != expected_sessions:
        missing = sorted(set(expected_sessions) - set(sessions))
        extra = sorted(set(sessions) - set(expected_sessions))
        raise ValueError(
            "BMYRT session coverage is not exact: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    numeric = prices[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("BMYRT prices contain non-finite values.")
    coherent = (
        numeric[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & numeric["volume"].ge(0)
        & numeric["high"].ge(numeric[["open", "low", "close"]].max(axis=1))
        & numeric["low"].le(numeric[["open", "high", "close"]].min(axis=1))
    )
    if not bool(coherent.all()):
        raise ValueError("BMYRT OHLCV values are incoherent.")
    first = prices.sort_values("session", kind="stable").iloc[0]
    if not math.isclose(
        float(first["close"]), CVR_REFERENCE_CLOSE, rel_tol=0, abs_tol=1e-6
    ):
        raise ValueError("BMYRT first close disagrees with the official BMS 10-K.")
    return {
        "price_rows": len(prices),
        "first_session": sessions[0],
        "last_session": sessions[-1],
        "first_close": float(first["close"]),
    }


def _bundle_from_artifacts(
    artifacts: Iterable[SourceArtifact],
    *,
    catalog_selection: FrozenCatalogSelection,
    http_attempts: int,
    fetched_against_release: str,
    budget_used_before: int | None,
    budget_used_after: int | None,
) -> CvrBundle:
    _validate_catalog_selection(catalog_selection)
    materialized = tuple(artifacts)
    by_source = {artifact.source: artifact for artifact in materialized}
    required = {f"eodhd_{name}" for name in ENDPOINTS}
    if len(materialized) != len(required) or set(by_source) != required:
        raise ValueError("BMYRT bundle has an incomplete or duplicate artifact set.")
    provider_code = catalog_selection.provider_code
    security_id = _security_id(provider_code)
    for endpoint in ENDPOINTS:
        artifact = by_source[f"eodhd_{endpoint}"]
        if artifact.source_url != _endpoint_url(endpoint, provider_code):
            raise ValueError(f"BMYRT {endpoint} artifact URL is not exact.")
        _artifact_rows(artifact, endpoint)
    eod = by_source["eodhd_eod"]
    prices = _prices_from_eod_artifact(eod, security_id=security_id)
    if _artifact_rows(by_source["eodhd_div"], "div") or _artifact_rows(
        by_source["eodhd_splits"], "splits"
    ):
        raise ValueError("BMYRT unexpectedly has provider dividends or splits.")
    bundle = CvrBundle(
        provider_code=provider_code,
        security_id=security_id,
        catalog_source_url=catalog_selection.source_url,
        catalog_source_hash=catalog_selection.source_hash,
        catalog_row=dict(catalog_selection.row),
        secondary_catalog_evidence=tuple(catalog_selection.secondary_evidence),
        prices=prices,
        artifacts=tuple(by_source[name] for name in sorted(by_source)),
        http_attempts=int(http_attempts),
        fetched_against_release=fetched_against_release,
        budget_used_before=budget_used_before,
        budget_used_after=budget_used_after,
    )
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: CvrBundle) -> dict[str, Any]:
    selection = FrozenCatalogSelection(
        provider_code=bundle.provider_code,
        source_url=bundle.catalog_source_url,
        source_hash=bundle.catalog_source_hash,
        row=bundle.catalog_row,
        secondary_evidence=bundle.secondary_catalog_evidence,
    )
    _validate_catalog_selection(selection)
    if bundle.http_attempts != EXPECTED_EODHD_CALLS:
        raise ValueError("BMYRT evidence must originate from exactly three EODHD calls.")
    if (
        bundle.budget_used_before is not None
        and bundle.budget_used_after is not None
        and bundle.budget_used_after - bundle.budget_used_before != EXPECTED_EODHD_CALLS
    ):
        raise ValueError("Persistent EODHD usage did not advance by exactly three calls.")
    prices = bundle.prices.copy()
    price_summary = _validate_exact_prices(prices)
    return {
        "provider_symbol": f"{bundle.provider_code}.US",
        "security_id": bundle.security_id,
        **price_summary,
        "actual_eodhd_calls": bundle.http_attempts,
        "maximum_eodhd_calls": EXPECTED_EODHD_CALLS,
    }


class ExactThreeEodhdClient(EodhdClient):
    """No-retry raw client restricted to the reviewed CELG-RI history path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempted_endpoints: list[str] = []
        self.provider_code = CVR_PROVIDER_CODE
        self.validated_eod_hash = ""

    def get_json(self, endpoint: str, *, params: dict[str, object] | None = None):
        del endpoint, params
        raise RuntimeError(
            "BMYRT v2 forbids get_json/search; use the reviewed raw artifact path."
        )

    def fetch_artifact(
        self,
        endpoint: str,
        *,
        params: dict[str, object] | None = None,
    ) -> SourceArtifact:
        normalized = endpoint.strip("/")
        position = len(self.attempted_endpoints)
        if position >= EXPECTED_EODHD_CALLS:
            raise RuntimeError("BMYRT client refused a fourth EODHD request.")
        expected_name = ENDPOINTS[position]
        expected = f"{expected_name}/{CVR_PROVIDER_SYMBOL}"
        expected_params = REQUEST_PARAMS
        if normalized != expected or dict(params or {}) != dict(expected_params):
            raise RuntimeError(
                "BMYRT client refused a non-reviewed request: "
                f"expected={expected}/{expected_params}, observed={normalized}/{params}."
            )
        if position > 0 and not self.validated_eod_hash:
            raise RuntimeError(
                "BMYRT client refused div/splits before the exact EOD gate passed."
            )
        self.budget.claim()
        self.attempted_endpoints.append(normalized)
        response = self.session.get(
            f"{self.base_url}/{normalized}",
            params={**expected_params, "api_token": self.token, "fmt": "json"},
            timeout=120,
        )
        response.raise_for_status()
        content = bytes(response.content)
        artifact = SourceArtifact(
            source=f"eodhd_{expected_name}",
            source_url=_endpoint_url(expected_name, CVR_PROVIDER_CODE),
            retrieved_at=utc_now_iso(),
            content=content,
            content_type=response.headers.get("Content-Type", "application/json"),
        )
        _artifact_rows(artifact, expected_name)
        return artifact

    def validate_and_authorize_eod(
        self,
        artifact: SourceArtifact,
        *,
        security_id: str,
    ) -> pd.DataFrame:
        if self.attempted_endpoints != [f"eod/{CVR_PROVIDER_SYMBOL}"]:
            raise RuntimeError("BMYRT EOD authorization requires exactly one reviewed call.")
        if artifact.source != "eodhd_eod" or artifact.source_url != _endpoint_url(
            "eod", CVR_PROVIDER_CODE
        ):
            raise ValueError("BMYRT EOD authorization artifact identity is not exact.")
        prices = _prices_from_eod_artifact(artifact, security_id=security_id)
        _validate_exact_prices(prices)
        self.validated_eod_hash = artifact.source_hash
        return prices


def _budget_used(budget: EodhdCallBudget) -> int:
    used = int(budget.seed_used)
    if budget.state_path.is_file():
        try:
            value = json.loads(budget.state_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            value = {}
        if _text(value.get("period")) == budget.period:
            used = max(used, int(value.get("used", 0)))
    return used


def fetch_exact_bundle(
    client: ExactThreeEodhdClient,
    *,
    catalog_selection: FrozenCatalogSelection,
    release_version: str,
    budget_used_before: int,
) -> CvrBundle:
    _validate_catalog_selection(catalog_selection)
    if client.provider_code != catalog_selection.provider_code:
        raise ValueError("Reviewed client code disagrees with the frozen catalog selection.")
    eod = client.fetch_artifact(
        f"eod/{CVR_PROVIDER_SYMBOL}", params=dict(REQUEST_PARAMS)
    )
    client.validate_and_authorize_eod(
        eod,
        security_id=_security_id(catalog_selection.provider_code),
    )
    artifacts = [eod]
    for endpoint in ("div", "splits"):
        artifacts.append(
            client.fetch_artifact(
                f"{endpoint}/{CVR_PROVIDER_SYMBOL}", params=dict(REQUEST_PARAMS)
            )
        )
    bundle = _bundle_from_artifacts(
        artifacts,
        catalog_selection=catalog_selection,
        http_attempts=len(client.attempted_endpoints),
        fetched_against_release=release_version,
        budget_used_before=budget_used_before,
        budget_used_after=_budget_used(client.budget),
    )
    return bundle


def _bundle_signature() -> dict[str, Any]:
    return {
        "schema": "us_celg_bmy_cvr_eodhd_bundle/v2",
        "provider_code": CVR_PROVIDER_CODE,
        "search_forbidden": True,
        "fetch_start": FETCH_START,
        "fetch_end": FETCH_END,
        "expected_sessions": EXPECTED_CVR_SESSIONS,
        "expected_http_attempts": EXPECTED_EODHD_CALLS,
        "active_catalog_sha256": ACTIVE_CATALOG_SHA256,
        "delisted_catalog_sha256": DELISTED_CATALOG_SHA256,
        "sec_submitter_search_sha256": SEC_SUBMITTER_SEARCH_SHA256,
        "official_terms_sha256": MERGER_TERMS_SHA256,
        "termination_sha256": TERMINATION_SHA256,
    }


def write_bundle_cache(path: Path, bundle: CvrBundle) -> None:
    validate_bundle(bundle)
    payload = {
        **_bundle_signature(),
        "provider_code": bundle.provider_code,
        "security_id": bundle.security_id,
        "catalog_source_url": bundle.catalog_source_url,
        "catalog_source_hash": bundle.catalog_source_hash,
        "catalog_row": dict(bundle.catalog_row),
        "secondary_catalog_evidence": [
            dict(item) for item in bundle.secondary_catalog_evidence
        ],
        "fetched_against_release": bundle.fetched_against_release,
        "http_attempts": bundle.http_attempts,
        "budget_used_before": bundle.budget_used_before,
        "budget_used_after": bundle.budget_used_after,
        "artifacts": [
            {
                "source": artifact.source,
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "content_type": artifact.content_type,
                "content_base64": base64.b64encode(artifact.content).decode("ascii"),
                "content_sha256": artifact.source_hash,
            }
            for artifact in bundle.artifacts
        ],
    }
    payload_bytes = _canonical_json_bytes(payload)
    wrapper = _canonical_json_bytes(
        {
            "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
            "payload_sha256": sha256_bytes(payload_bytes),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, gzip.compress(wrapper, mtime=0))


def read_bundle_cache(path: Path) -> CvrBundle | None:
    if not path.is_file():
        return None
    try:
        wrapper = json.loads(gzip.decompress(path.read_bytes()))
        payload_bytes = base64.b64decode(wrapper["payload_base64"], validate=True)
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"BMYRT bundle cache is unreadable: {path}.") from exc
    if sha256_bytes(payload_bytes) != _text(wrapper.get("payload_sha256")):
        raise ValueError("BMYRT bundle cache wrapper hash mismatch.")
    payload = json.loads(payload_bytes)
    for key, expected in _bundle_signature().items():
        if payload.get(key) != expected:
            raise ValueError(f"BMYRT bundle signature mismatch: {key}.")
    artifacts = []
    for item in payload.get("artifacts", ()):
        content = base64.b64decode(item["content_base64"], validate=True)
        if sha256_bytes(content) != _text(item.get("content_sha256")):
            raise ValueError("BMYRT cached artifact hash mismatch.")
        artifacts.append(
            SourceArtifact(
                source=_text(item.get("source")),
                source_url=_text(item.get("source_url")),
                retrieved_at=_text(item.get("retrieved_at")),
                content=content,
                content_type=_text(item.get("content_type")),
            )
        )
    bundle = _bundle_from_artifacts(
        artifacts,
        catalog_selection=FrozenCatalogSelection(
            provider_code=_text(payload.get("provider_code")),
            source_url=_text(payload.get("catalog_source_url")),
            source_hash=_text(payload.get("catalog_source_hash")),
            row=dict(payload.get("catalog_row") or {}),
            secondary_evidence=tuple(
                dict(item) for item in payload.get("secondary_catalog_evidence", ())
            ),
        ),
        http_attempts=int(payload.get("http_attempts", -1)),
        fetched_against_release=_text(payload.get("fetched_against_release")),
        budget_used_before=(
            int(payload["budget_used_before"])
            if payload.get("budget_used_before") is not None
            else None
        ),
        budget_used_after=(
            int(payload["budget_used_after"])
            if payload.get("budget_used_after") is not None
            else None
        ),
    )
    if (
        bundle.provider_code != _text(payload.get("provider_code"))
        or bundle.security_id != _text(payload.get("security_id"))
        or bundle.catalog_source_url != _text(payload.get("catalog_source_url"))
        or bundle.catalog_source_hash != _text(payload.get("catalog_source_hash"))
        or dict(bundle.catalog_row) != dict(payload.get("catalog_row") or {})
        or list(bundle.secondary_catalog_evidence)
        != list(payload.get("secondary_catalog_evidence") or [])
    ):
        raise ValueError("BMYRT cached identity disagrees with pinned catalog evidence.")
    return bundle


def _safe_archived_content(repository: LocalDatasetRepository, object_path: str) -> bytes:
    root = repository.root.resolve()
    path = (repository.root / object_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Archived object escapes the cache root.")
    if not path.is_file():
        raise FileNotFoundError(f"Archived object is missing: {path}.")
    try:
        return gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError(f"Archived object is not valid gzip: {path}.") from exc


def _load_merger_terms(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> SourceArtifact:
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    matches = archive.loc[
        archive["source_url"].astype(str).eq(MERGER_TERMS_URL)
        & archive["source_hash"].astype(str).eq(MERGER_TERMS_SHA256)
    ]
    if len(matches) != 1:
        raise ValueError("Pinned official CELG merger filing is not uniquely archived.")
    row = matches.iloc[0]
    content = _safe_archived_content(repository, _text(row["object_path"]))
    if len(content) != MERGER_TERMS_BYTES or sha256_bytes(content) != MERGER_TERMS_SHA256:
        raise ValueError("Pinned official CELG merger filing hash/size changed.")
    text = _normalized_document_text(content)
    required_groups = (
        ("november 20, 2019",),
        ("$50.00",),
        (
            "1.00 share of bristol-myers squibb common stock",
            "one share of bms common stock",
        ),
        ("one tradeable contingent value right", "one cvr"),
        ("cvrs trading under the symbol “bmyrt.”",),
    )
    for group in required_groups:
        if not any(value in text for value in group):
            raise ValueError(
                "Official CELG merger filing lacks reviewed text: "
                + " | ".join(group)
            )
    return SourceArtifact(
        source="sec_edgar_filing",
        source_url=MERGER_TERMS_URL,
        retrieved_at=_text(row.get("retrieved_at")),
        content=content,
        content_type=_text(row.get("content_type")) or "text/plain",
    )


def _load_termination_evidence(evidence_dir: Path) -> SourceArtifact:
    path = evidence_dir / f"{TERMINATION_SHA256}.html"
    if not path.is_file():
        raise FileNotFoundError(f"Pinned official BMYRT termination evidence is missing: {path}.")
    content = path.read_bytes()
    if len(content) != TERMINATION_BYTES or sha256_bytes(content) != TERMINATION_SHA256:
        raise ValueError("Pinned official BMYRT termination evidence hash/size changed.")
    text = _normalized_document_text(content)
    required_groups = (
        ("on january 1, 2021", "january 1, 2021"),
        ("terminated automatically",),
        ("no longer eligible for payment",),
        ("closing price of bms common stock on november 19, 2019",),
        ("$ 56.48", "$56.48"),
        ("closing price of cvr",),
        ("$ 2.30", "$2.30"),
        ("first trade on november 21, 2019",),
    )
    for group in required_groups:
        if not any(value in text for value in group):
            raise ValueError(
                "Official BMYRT termination/valuation evidence lacks reviewed text: "
                + " | ".join(group)
            )
    return SourceArtifact(
        source="sec_bmy_2020_10k",
        source_url=TERMINATION_URL,
        retrieved_at=TERMINATION_RETRIEVED_AT,
        content=content,
        content_type="text/html",
    )


def _load_sec_submitter_evidence(
    repository: LocalDatasetRepository,
) -> dict[str, Any]:
    path = repository.root / SEC_SUBMITTER_CACHE_OBJECT
    if not path.is_file():
        raise FileNotFoundError(
            f"Pinned local SEC submitter search evidence is missing: {path}."
        )
    content = path.read_bytes()
    if sha256_bytes(content) != SEC_SUBMITTER_SEARCH_SHA256:
        raise ValueError("Pinned SEC submitter search evidence hash changed.")
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Pinned SEC submitter search evidence is not JSON.") from exc
    hits = payload.get("hits", {}).get("hits", [])
    exact = [item for item in hits if _text(item.get("_id")) == SEC_CLOSING_SEARCH_HIT_ID]
    if len(exact) != 1:
        raise ValueError("Pinned SEC closing filing search hit is not unique.")
    names = exact[0].get("_source", {}).get("display_names", [])
    if names != [SEC_SUBMITTER_DISPLAY_NAME]:
        raise ValueError("Official SEC submitter display name/ticker changed.")
    return {
        "source_url": SEC_SUBMITTER_SEARCH_URL,
        "source_hash": SEC_SUBMITTER_SEARCH_SHA256,
        "display_name": SEC_SUBMITTER_DISPLAY_NAME,
        "matched_symbol": SEC_SUBMITTER_SYMBOL,
        "closing_hit_id": SEC_CLOSING_SEARCH_HIT_ID,
    }


def _frozen_catalog_selection(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> FrozenCatalogSelection:
    """Select CELG-RI exactly; retain BMY-R variants only as ambiguity evidence."""

    _load_sec_submitter_evidence(repository)
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    catalogs: dict[str, list[dict[str, Any]]] = {}
    for url, pinned_hash in (
        (ACTIVE_CATALOG_URL, ACTIVE_CATALOG_SHA256),
        (DELISTED_CATALOG_URL, DELISTED_CATALOG_SHA256),
    ):
        rows = archive.loc[
            archive["dataset"].astype(str).eq("eodhd_exchange_symbols")
            & archive["source_url"].astype(str).eq(url)
            & archive["source_hash"].astype(str).eq(pinned_hash)
        ]
        if len(rows) != 1:
            raise ValueError(f"Pinned frozen EODHD catalog is not unique: {url}.")
        row = rows.iloc[0]
        content = _safe_archived_content(repository, _text(row["object_path"]))
        if sha256_bytes(content) != pinned_hash:
            raise ValueError("Frozen EODHD catalog archive hash changed.")
        try:
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Frozen EODHD catalog is not JSON.") from exc
        if not isinstance(payload, list) or any(
            not isinstance(item, dict) for item in payload
        ):
            raise ValueError("Frozen EODHD catalog must be a JSON row list.")
        catalogs[url] = payload

    exact = [
        dict(item)
        for item in catalogs[ACTIVE_CATALOG_URL]
        if _text(item.get("Code")) == SEC_SUBMITTER_SYMBOL
    ]
    if exact != [EXACT_CATALOG_ROW]:
        raise ValueError("Pinned active catalog does not contain one exact CELG-RI row.")
    secondary: list[dict[str, Any]] = []
    for code in SECONDARY_CATALOG_CODES:
        matches = [
            dict(item)
            for item in catalogs[DELISTED_CATALOG_URL]
            if _text(item.get("Code")) == code
        ]
        if len(matches) != 1:
            raise ValueError(f"Pinned secondary catalog code is not unique: {code}.")
        secondary.append(
            {
                "role": "secondary_ambiguous",
                "reason": "Provider alias lacks the SEC submitter ticker and ISIN binding.",
                "source_url": DELISTED_CATALOG_URL,
                "source_hash": DELISTED_CATALOG_SHA256,
                "row": matches[0],
            }
        )
    selection = FrozenCatalogSelection(
        provider_code=CVR_PROVIDER_CODE,
        source_url=ACTIVE_CATALOG_URL,
        source_hash=ACTIVE_CATALOG_SHA256,
        row=exact[0],
        secondary_evidence=tuple(secondary),
    )
    _validate_catalog_selection(selection)
    return selection


def _official_exit_model(
    selection: FrozenCatalogSelection,
) -> OfficialExitMarkModel:
    _validate_catalog_selection(selection)
    model = OfficialExitMarkModel(
        provider_code=selection.provider_code,
        security_id=_security_id(selection.provider_code),
        catalog_source_url=selection.source_url,
        catalog_source_hash=selection.source_hash,
        catalog_row=dict(selection.row),
        secondary_catalog_evidence=tuple(selection.secondary_evidence),
    )
    if model.security_id != OFFICIAL_EXIT_SECURITY_ID:
        raise ValueError("Official exit-mark BMYRT identity changed.")
    return model


def call_plan(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    selection = _frozen_catalog_selection(repository, release)
    return {
        "exact_catalog_candidate": {
            "provider_symbol": CVR_PROVIDER_SYMBOL,
            "source_url": selection.source_url,
            "source_hash": selection.source_hash,
            "row": dict(selection.row),
        },
        "secondary_catalog_evidence": [
            dict(item) for item in selection.secondary_evidence
        ],
        "selection_basis": {
            "rule": "hash_pinned_catalog_row_matches_official_sec_submitter_symbol",
            "sec_submitter": _load_sec_submitter_evidence(repository),
        },
        "catalog_note": "CELGZ is an older Abraxis/Celgene CVR and is excluded.",
        "search_required": False,
        "search_forbidden": True,
        "minimum_eodhd_calls": 1,
        "maximum_eodhd_calls": EXPECTED_EODHD_CALLS,
        "diagnostic_budget_note": dict(DIAGNOSTIC_BUDGET_NOTE),
        "request_order": [
            {
                "endpoint": f"eod/{CVR_PROVIDER_SYMBOL}",
                "call_number": 1,
                "gate": (
                    "exact 280 XNYS sessions, 2019-11-21..2020-12-31, "
                    "first close 2.30, coherent USD OHLCV"
                ),
            },
            {
                "endpoint": f"div/{CVR_PROVIDER_SYMBOL}",
                "call_number": 2,
                "only_after": "EOD gate passes",
            },
            {
                "endpoint": f"splits/{CVR_PROVIDER_SYMBOL}",
                "call_number": 3,
                "only_after": "EOD gate passes",
            },
        ],
        "failure_policy": "If call 1 fails the exact gate, stop without alias probes.",
    }


def _archive_extension(artifact: SourceArtifact) -> str:
    if artifact.content_type == "application/json":
        return "json"
    if artifact.content_type == "text/html":
        return "html"
    return "txt"


def _append_source_archive(
    source_archive: pd.DataFrame,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> tuple[pd.DataFrame, int]:
    output = source_archive.copy()
    added = 0
    for artifact in artifacts:
        exact = (
            output["source_hash"].astype(str).eq(artifact.source_hash)
            & output.get("source_url", pd.Series("", index=output.index))
            .astype(str)
            .eq(artifact.source_url)
            & output["source"].astype(str).eq(artifact.source)
        )
        if exact.any():
            if int(exact.sum()) != 1:
                raise ValueError(f"Source archive pair is duplicated: {artifact.source_hash}.")
            continue
        archive_id = artifact.source_hash
        if output["archive_id"].astype(str).eq(archive_id).any():
            archive_id = sha256_bytes(
                _canonical_json_bytes(
                    {
                        "source": artifact.source,
                        "source_url": artifact.source_url,
                        "source_hash": artifact.source_hash,
                    }
                )
            )
        collision = output["archive_id"].astype(str).eq(archive_id)
        if collision.any():
            raise ValueError(f"Source archive identity collision: {archive_id}.")
        row = {
            "archive_id": archive_id,
            "dataset": artifact.source,
            "object_path": (
                f"archives/{completed_session}/{archive_id}."
                f"{_archive_extension(artifact)}.gz"
            ),
            "content_type": artifact.content_type,
            "effective_date": completed_session,
            "source": artifact.source,
            "source_url": artifact.source_url,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
        }
        output = pd.concat([output, pd.DataFrame([row])], ignore_index=True, sort=False)
        added += 1
    return output.reset_index(drop=True), added


def _official_actions(
    bundle: CvrBundle,
    terms: SourceArtifact,
    termination: SourceArtifact,
) -> pd.DataFrame:
    basis_metadata = {
        "asset_kind": "exchange_traded_contingent_value_right",
        "cost_basis_fraction": CVR_ECONOMIC_BASIS_FRACTION,
        "basis_kind": "economic_relative_fair_value_including_cash",
        "cash_consideration": MERGER_CASH_PER_SHARE,
        "bmy_reference_close": BMY_REFERENCE_CLOSE,
        "cvr_reference_close": CVR_REFERENCE_CLOSE,
        "reference_total_consideration": TOTAL_REFERENCE_CONSIDERATION,
        "reference_source_url": TERMINATION_URL,
        "reference_source_hash": TERMINATION_SHA256,
        "trade_start": MERGER_SESSION,
        "termination_date": CVR_TERMINATION_DATE,
        "terminal_payment": 0.0,
    }
    common = {
        "announcement_date": "2019-11-20",
        "record_date": "",
        "currency": "USD",
        "official": True,
        "source_url": terms.source_url,
        "source_kind": "official_crosscheck",
        "source": "official_celg_bmy_cvr",
        "retrieved_at": terms.retrieved_at,
        "source_hash": terms.source_hash,
    }
    rows = [
        {
            **common,
            "event_id": CVR_DISTRIBUTION_EVENT_ID,
            "security_id": CELG_SECURITY_ID,
            "action_type": "spinoff",
            "effective_date": MERGER_SESSION,
            "ex_date": MERGER_SESSION,
            "payment_date": MERGER_SESSION,
            "cash_amount": None,
            "ratio": 1.0,
            "new_security_id": bundle.security_id,
            "new_symbol": CVR_SYMBOL,
            "metadata": _canonical_json(basis_metadata),
        },
        {
            **common,
            "event_id": CELG_STOCK_MERGER_EVENT_ID,
            "security_id": CELG_SECURITY_ID,
            "action_type": "stock_merger",
            "effective_date": MERGER_SESSION,
            "ex_date": MERGER_SESSION,
            "payment_date": MERGER_SESSION,
            "cash_amount": MERGER_CASH_PER_SHARE,
            "ratio": 1.0,
            "new_security_id": BMY_SECURITY_ID,
            "new_symbol": BMY_SYMBOL,
            "metadata": _canonical_json(
                {
                    "additional_security_event_id": CVR_DISTRIBUTION_EVENT_ID,
                    "consideration_sequence": ["BMYRT", "BMY", "USD"],
                    "cvr_security_id": bundle.security_id,
                    "cvr_symbol": CVR_SYMBOL,
                    "source_url": terms.source_url,
                    "source_hash": terms.source_hash,
                }
            ),
        },
        {
            "event_id": canonical_lifecycle_event_id(
                bundle.security_id, "delisting", CVR_TERMINATION_DATE
            ),
            "security_id": bundle.security_id,
            "action_type": "delisting",
            "effective_date": CVR_TERMINATION_DATE,
            "ex_date": CVR_TERMINATION_DATE,
            "announcement_date": CVR_TERMINATION_DATE,
            "record_date": "",
            "payment_date": CVR_TERMINATION_DATE,
            "cash_amount": 0.0,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "official": True,
            "source_url": termination.source_url,
            "source_kind": "official_filing",
            "source": termination.source,
            "retrieved_at": termination.retrieved_at,
            "source_hash": termination.source_hash,
            "metadata": _canonical_json(
                {
                    "contract_terminated_automatically": True,
                    "last_trading_session": CVR_LAST_SESSION,
                    "milestone_not_met": "liso-cel FDA approval by 2020-12-31",
                    "payout_per_right": 0.0,
                }
            ),
        },
    ]
    return pd.DataFrame(rows)


def _official_exit_basis_metadata() -> dict[str, Any]:
    return {
        "asset_kind": "exchange_traded_contingent_value_right",
        "cost_basis_fraction": CVR_ECONOMIC_BASIS_FRACTION,
        "basis_kind": "economic_relative_fair_value_including_cash",
        "cash_consideration": MERGER_CASH_PER_SHARE,
        "bmy_reference_close": BMY_REFERENCE_CLOSE,
        "cvr_reference_close": CVR_REFERENCE_CLOSE,
        "reference_total_consideration": TOTAL_REFERENCE_CONSIDERATION,
        "reference_source_url": TERMINATION_URL,
        "reference_source_hash": TERMINATION_SHA256,
        "trade_start": MERGER_SESSION,
        "exit_policy": OFFICIAL_EXIT_MODE,
        "non_index_child": True,
        "trading_path_supported": False,
        "termination_date": CVR_TERMINATION_DATE,
        "terminal_payment": 0.0,
    }


def _official_exit_liquidation_metadata() -> dict[str, Any]:
    return {
        "mode": OFFICIAL_EXIT_MODE,
        "exit_only": True,
        "non_index_child": True,
        "first_tradable_session": MERGER_SESSION,
        "official_first_trade_close": CVR_REFERENCE_CLOSE,
        "price_row_kind": "official_valuation_mark_not_provider_ohlcv",
        "execution_timing": "first_tradable_session_close",
        "cash_available_session": "2019-11-22",
        "retrospective_official_evidence": True,
        "trading_path_supported": False,
        "reference_source_url": TERMINATION_URL,
        "reference_source_hash": TERMINATION_SHA256,
    }


def _official_residual_termination_metadata() -> dict[str, Any]:
    return {
        "contract_terminated_automatically": True,
        "last_trading_session": CVR_LAST_SESSION,
        "milestone_not_met": "liso-cel FDA approval by 2020-12-31",
        "payout_per_right": 0.0,
        "residual_only": True,
        "only_if_position_remains": True,
        "trading_path_supported": False,
    }


def _official_exit_actions(
    model: OfficialExitMarkModel,
    terms: SourceArtifact,
    termination: SourceArtifact,
) -> pd.DataFrame:
    common = {
        "announcement_date": "2019-11-20",
        "record_date": "",
        "currency": "USD",
        "official": True,
        "source_url": terms.source_url,
        "source_kind": "official_crosscheck",
        "source": "official_celg_bmy_cvr",
        "retrieved_at": terms.retrieved_at,
        "source_hash": terms.source_hash,
    }
    rows = [
        {
            **common,
            "event_id": CVR_DISTRIBUTION_EVENT_ID,
            "security_id": CELG_SECURITY_ID,
            "action_type": "spinoff",
            "effective_date": MERGER_SESSION,
            "ex_date": MERGER_SESSION,
            "payment_date": MERGER_SESSION,
            "cash_amount": None,
            "ratio": 1.0,
            "new_security_id": model.security_id,
            "new_symbol": CVR_SYMBOL,
            "metadata": _canonical_json(_official_exit_basis_metadata()),
        },
        {
            **common,
            "event_id": CELG_STOCK_MERGER_EVENT_ID,
            "security_id": CELG_SECURITY_ID,
            "action_type": "stock_merger",
            "effective_date": MERGER_SESSION,
            "ex_date": MERGER_SESSION,
            "payment_date": MERGER_SESSION,
            "cash_amount": MERGER_CASH_PER_SHARE,
            "ratio": 1.0,
            "new_security_id": BMY_SECURITY_ID,
            "new_symbol": BMY_SYMBOL,
            "metadata": _canonical_json(
                {
                    "additional_security_event_id": CVR_DISTRIBUTION_EVENT_ID,
                    "consideration_sequence": ["BMYRT", "BMY", "USD"],
                    "cvr_security_id": model.security_id,
                    "cvr_symbol": CVR_SYMBOL,
                    "cvr_exit_event_id": OFFICIAL_EXIT_EVENT_ID,
                    "cvr_exit_policy": OFFICIAL_EXIT_MODE,
                    "source_url": terms.source_url,
                    "source_hash": terms.source_hash,
                }
            ),
        },
        {
            "event_id": OFFICIAL_EXIT_EVENT_ID,
            "security_id": model.security_id,
            "action_type": "delisting",
            "effective_date": MERGER_SESSION,
            "ex_date": MERGER_SESSION,
            "announcement_date": "",
            "record_date": "",
            "payment_date": "2019-11-22",
            "cash_amount": CVR_REFERENCE_CLOSE,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "official": True,
            "source_url": termination.source_url,
            "source_kind": "official_filing_exit_mark",
            "source": termination.source,
            "retrieved_at": termination.retrieved_at,
            "source_hash": termination.source_hash,
            "metadata": _canonical_json(_official_exit_liquidation_metadata()),
        },
        {
            "event_id": OFFICIAL_RESIDUAL_TERMINATION_EVENT_ID,
            "security_id": model.security_id,
            "action_type": "delisting",
            "effective_date": CVR_TERMINATION_DATE,
            "ex_date": CVR_TERMINATION_DATE,
            "announcement_date": CVR_TERMINATION_DATE,
            "record_date": "",
            "payment_date": CVR_TERMINATION_DATE,
            "cash_amount": 0.0,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "official": True,
            "source_url": termination.source_url,
            "source_kind": "official_filing",
            "source": termination.source,
            "retrieved_at": termination.retrieved_at,
            "source_hash": termination.source_hash,
            "metadata": _canonical_json(_official_residual_termination_metadata()),
        },
    ]
    return pd.DataFrame(rows)


def _official_exit_policy_artifact(
    model: OfficialExitMarkModel,
    termination: SourceArtifact,
) -> SourceArtifact:
    payload = {
        "schema": "celg_bmyrt_official_exit_mark_policy/v1",
        "mode": OFFICIAL_EXIT_MODE,
        "security_id": model.security_id,
        "symbol": CVR_SYMBOL,
        "session": MERGER_SESSION,
        "mark_usd": CVR_REFERENCE_CLOSE,
        "row_encoding": {
            "open": CVR_REFERENCE_CLOSE,
            "high": CVR_REFERENCE_CLOSE,
            "low": CVR_REFERENCE_CLOSE,
            "close": CVR_REFERENCE_CLOSE,
            "volume": 0.0,
            "meaning": "valuation_mark_not_observed_provider_ohlcv",
        },
        "execution_timing": "first_tradable_session_close",
        "cash_available_session": "2019-11-22",
        "non_index_child": True,
        "retrospective_official_evidence": True,
        "trading_path_supported": False,
        "official_source_url": termination.source_url,
        "official_source_hash": termination.source_hash,
    }
    return _source_artifact(
        "official_exit_mark_policy",
        OFFICIAL_EXIT_POLICY_URL,
        payload,
        termination.retrieved_at,
    )


def _official_exit_price(
    model: OfficialExitMarkModel,
    policy: SourceArtifact,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": model.security_id,
                "session": MERGER_SESSION,
                "open": CVR_REFERENCE_CLOSE,
                "high": CVR_REFERENCE_CLOSE,
                "low": CVR_REFERENCE_CLOSE,
                "close": CVR_REFERENCE_CLOSE,
                "volume": 0.0,
                "currency": "USD",
                "source": policy.source,
                "source_url": policy.source_url,
                "retrieved_at": policy.retrieved_at,
                "source_hash": policy.source_hash,
            }
        ]
    )


def _official_exit_identity_artifact(
    model: OfficialExitMarkModel,
    terms: SourceArtifact,
    termination: SourceArtifact,
    policy: SourceArtifact,
) -> SourceArtifact:
    payload = {
        "schema": "celg_bmyrt_identity_resolution/official_exit_mark/v1",
        "mode": OFFICIAL_EXIT_MODE,
        "security_id": model.security_id,
        "symbol": CVR_SYMBOL,
        "provider_code": model.provider_code,
        "provider_symbol": f"{model.provider_code}.US",
        "exchange": "NYSE",
        "active_from": MERGER_SESSION,
        "active_to": CVR_LAST_SESSION,
        "catalog_source_url": model.catalog_source_url,
        "catalog_source_hash": model.catalog_source_hash,
        "catalog_row": dict(model.catalog_row),
        "secondary_catalog_evidence": [
            dict(item) for item in model.secondary_catalog_evidence
        ],
        "official_exit_mark": CVR_REFERENCE_CLOSE,
        "official_exit_session": MERGER_SESSION,
        "official_exit_source_url": termination.source_url,
        "official_exit_source_hash": termination.source_hash,
        "official_exit_policy_url": policy.source_url,
        "official_exit_policy_hash": policy.source_hash,
        "trading_path_supported": False,
        "provider_price_artifact_claimed": False,
        "official_merger_url": terms.source_url,
        "official_merger_sha256": terms.source_hash,
        "official_termination_url": termination.source_url,
        "official_termination_sha256": termination.source_hash,
    }
    return _source_artifact(
        "celg_bmyrt_identity_resolution",
        termination.source_url,
        payload,
        termination.retrieved_at,
    )


def _merge_actions(existing: pd.DataFrame, additions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    output = existing.copy()
    by_event = {
        _text(row.event_id): row
        for row in output.itertuples(index=False)
        if _text(row.event_id)
    }
    semantic = {
        (
            _text(row.security_id),
            _text(row.action_type).lower(),
            _date_text(row.effective_date),
        ): row
        for row in output.itertuples(index=False)
    }
    added = 0
    for row in additions.to_dict(orient="records"):
        event_id = _text(row["event_id"])
        event_collision = by_event.get(event_id)
        if event_collision is not None:
            prior_record = event_collision._asdict()
            if (
                _text(prior_record.get("security_id")) != _text(row.get("security_id"))
                or _text(prior_record.get("action_type")) != _text(row.get("action_type"))
                or _date_text(prior_record.get("effective_date"))
                != _date_text(row.get("effective_date"))
            ):
                raise ValueError(f"Corporate-action event_id collision: {event_id}.")
        key = (
            _text(row["security_id"]),
            _text(row["action_type"]).lower(),
            _date_text(row["effective_date"]),
        )
        prior = semantic.get(key)
        if prior is not None:
            prior_record = prior._asdict()
            columns = (
                "event_id",
                "security_id",
                "action_type",
                "effective_date",
                "ex_date",
                "cash_amount",
                "ratio",
                "new_security_id",
                "new_symbol",
                "source_url",
                "source_hash",
                "metadata",
            )
            def normalized(value: Any) -> Any:
                if _text(value) == "":
                    return ""
                if isinstance(value, (int, float, np.number)):
                    return float(value)
                return _text(value)
            if any(normalized(prior_record.get(col)) != normalized(row.get(col)) for col in columns):
                raise ValueError(f"Conflicting CELG/BMYRT corporate action: {key}.")
            continue
        output = pd.concat([output, pd.DataFrame([row])], ignore_index=True, sort=False)
        semantic[key] = next(pd.DataFrame([row]).itertuples(index=False))
        by_event[event_id] = semantic[key]
        added += 1
    output = output.drop_duplicates(
        list(dataset_spec("corporate_actions").primary_key), keep="last"
    )
    return output.sort_values(
        ["security_id", "effective_date", "event_id"], kind="stable"
    ).reset_index(drop=True), added


def _identity_artifact(
    bundle: CvrBundle,
    terms: SourceArtifact,
    termination: SourceArtifact,
) -> SourceArtifact:
    eod = next(artifact for artifact in bundle.artifacts if artifact.source == "eodhd_eod")
    payload = {
        "schema": "celg_bmyrt_identity_resolution/v2",
        "security_id": bundle.security_id,
        "symbol": CVR_SYMBOL,
        "provider_code": bundle.provider_code,
        "provider_symbol": f"{bundle.provider_code}.US",
        "exchange": "NYSE",
        "active_from": MERGER_SESSION,
        "active_to": CVR_LAST_SESSION,
        "eodhd_catalog_url": bundle.catalog_source_url,
        "eodhd_catalog_sha256": bundle.catalog_source_hash,
        "eodhd_catalog_row": dict(bundle.catalog_row),
        "eodhd_secondary_catalog_evidence": [
            dict(item) for item in bundle.secondary_catalog_evidence
        ],
        "eodhd_eod_url": eod.source_url,
        "eodhd_eod_sha256": eod.source_hash,
        "official_merger_url": terms.source_url,
        "official_merger_sha256": terms.source_hash,
        "official_termination_url": termination.source_url,
        "official_termination_sha256": termination.source_hash,
    }
    return _source_artifact(
        "celg_bmyrt_identity_resolution",
        terms.source_url,
        payload,
        eod.retrieved_at,
    )


def _expected_cvr_master(
    bundle: CvrBundle,
    identity: SourceArtifact,
) -> dict[str, Any]:
    name = _text(bundle.catalog_row.get("Name") or bundle.catalog_row.get("name"))
    if not name:
        name = "Bristol-Myers Squibb Contingent Value Rights"
    return {
        "security_id": bundle.security_id,
        "primary_symbol": CVR_SYMBOL,
        "provider_symbol": f"{bundle.provider_code}.US",
        "action_provider_symbol": f"{bundle.provider_code}.US",
        "name": name,
        "exchange": "NYSE",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": MERGER_SESSION,
        "active_to": CVR_LAST_SESSION,
        "isin": "",
        "source": identity.source,
        "source_url": identity.source_url,
        "retrieved_at": identity.retrieved_at,
        "source_hash": identity.source_hash,
    }


def _expected_cvr_history(
    bundle: CvrBundle,
    identity: SourceArtifact,
) -> dict[str, Any]:
    return {
        "security_id": bundle.security_id,
        "symbol": CVR_SYMBOL,
        "exchange": "NYSE",
        "effective_from": MERGER_SESSION,
        "effective_to": CVR_LAST_SESSION,
        "source": identity.source,
        "source_url": identity.source_url,
        "retrieved_at": identity.retrieved_at,
        "source_hash": identity.source_hash,
    }


def _exact_text_fields(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    fields: Iterable[str],
) -> bool:
    return all(_text(row.get(field)) == _text(expected.get(field)) for field in fields)


def _rewrite_master_history(
    master: pd.DataFrame,
    history: pd.DataFrame,
    bundle: CvrBundle,
    terms: SourceArtifact,
    identity: SourceArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    output_master = master.copy()
    output_history = history.copy()
    celg_master = output_master["security_id"].astype(str).eq(CELG_SECURITY_ID)
    bmy_master = output_master["security_id"].astype(str).eq(BMY_SECURITY_ID)
    if int(celg_master.sum()) != 1 or int(bmy_master.sum()) != 1:
        raise ValueError("CELG/BMY security-master identities are missing or duplicated.")
    if _text(output_master.loc[celg_master, "primary_symbol"].iloc[0]).upper() != CELG_SYMBOL:
        raise ValueError("Pinned CELG security_id no longer maps to CELG.")
    if _text(output_master.loc[bmy_master, "primary_symbol"].iloc[0]).upper() != BMY_SYMBOL:
        raise ValueError("Pinned BMY security_id no longer maps to BMY.")
    master_boundary = _date_text(output_master.loc[celg_master, "active_to"].iloc[0])
    if master_boundary not in {"", CELG_LAST_SESSION}:
        raise ValueError("CELG security-master terminal boundary conflicts with filing.")
    master_changed = 0
    if not master_boundary:
        index = output_master.index[celg_master][0]
        output_master.at[index, "active_to"] = CELG_LAST_SESSION
        for field, value in {
            "source": "sec_edgar_filing",
            "source_url": terms.source_url,
            "retrieved_at": terms.retrieved_at,
            "source_hash": terms.source_hash,
        }.items():
            output_master.at[index, field] = value
        master_changed += 1

    expected_master = _expected_cvr_master(bundle, identity)
    cvr_id = output_master["security_id"].astype(str).eq(bundle.security_id)
    cvr_symbol = output_master["primary_symbol"].astype(str).str.upper().eq(CVR_SYMBOL)
    provider_symbol = (
        output_master.get("provider_symbol", pd.Series("", index=output_master.index))
        .astype(str)
        .str.upper()
        .eq(f"{bundle.provider_code}.US".upper())
    )
    conflicting = output_master.loc[(cvr_symbol | provider_symbol) & ~cvr_id]
    if not conflicting.empty:
        raise ValueError("BMYRT symbol/provider code is already bound to another identity.")
    if int(cvr_id.sum()) > 1:
        raise ValueError("BMYRT security-master identity is duplicated.")
    if cvr_id.any():
        existing = output_master.loc[cvr_id].iloc[0].to_dict()
        if not _exact_text_fields(existing, expected_master, expected_master):
            raise ValueError("Existing BMYRT security-master row is partial or conflicting.")
    else:
        output_master = pd.concat(
            [output_master, pd.DataFrame([expected_master])],
            ignore_index=True,
            sort=False,
        )
        master_changed += 1

    celg_history = (
        output_history["security_id"].astype(str).eq(CELG_SECURITY_ID)
        & output_history["symbol"].astype(str).str.upper().eq(CELG_SYMBOL)
    )
    if int(celg_history.sum()) != 1:
        raise ValueError("CELG symbol-history row is missing or duplicated.")
    history_boundary = _date_text(
        output_history.loc[celg_history, "effective_to"].iloc[0]
    )
    if history_boundary not in {"", CELG_LAST_SESSION}:
        raise ValueError("CELG symbol-history terminal boundary conflicts with filing.")
    history_changed = 0
    if not history_boundary:
        index = output_history.index[celg_history][0]
        output_history.at[index, "effective_to"] = CELG_LAST_SESSION
        for field, value in {
            "source": "sec_edgar_filing",
            "source_url": terms.source_url,
            "retrieved_at": terms.retrieved_at,
            "source_hash": terms.source_hash,
        }.items():
            output_history.at[index, field] = value
        history_changed += 1

    expected_history = _expected_cvr_history(bundle, identity)
    history_id = output_history["security_id"].astype(str).eq(bundle.security_id)
    history_symbol = output_history["symbol"].astype(str).str.upper().eq(CVR_SYMBOL)
    if bool((history_symbol & ~history_id).any()):
        raise ValueError("BMYRT symbol history is already bound to another identity.")
    if int(history_id.sum()) > 1:
        raise ValueError("BMYRT symbol-history identity is duplicated.")
    if history_id.any():
        existing = output_history.loc[history_id].iloc[0].to_dict()
        if not _exact_text_fields(existing, expected_history, expected_history):
            raise ValueError("Existing BMYRT symbol-history row is partial or conflicting.")
    else:
        output_history = pd.concat(
            [output_history, pd.DataFrame([expected_history])],
            ignore_index=True,
            sort=False,
        )
        history_changed += 1
    output_master = output_master.sort_values(
        ["security_id"], kind="stable"
    ).reset_index(drop=True)
    output_history = output_history.sort_values(
        ["security_id", "effective_from", "symbol"], kind="stable"
    ).reset_index(drop=True)
    return output_master, output_history, {
        "security_master_rows_changed": master_changed,
        "symbol_history_rows_changed": history_changed,
    }


def _normalized_bundle_prices(bundle: CvrBundle) -> pd.DataFrame:
    output = bundle.prices.copy()
    output["session"] = output["session"].map(_date_text)
    for field in ("open", "high", "low", "close", "volume"):
        output[field] = pd.to_numeric(output[field], errors="raise")
    output["volume"] = output["volume"].astype(float)
    return output.sort_values("session", kind="stable").reset_index(drop=True)


def _price_rows_are_exact(existing: pd.DataFrame, expected: pd.DataFrame) -> bool:
    if len(existing) != len(expected):
        return False
    left = existing.copy().sort_values("session", kind="stable").reset_index(drop=True)
    right = expected.copy().sort_values("session", kind="stable").reset_index(drop=True)
    if tuple(left["session"].map(_date_text)) != tuple(right["session"].map(_date_text)):
        return False
    for field in ("open", "high", "low", "close", "volume"):
        a = pd.to_numeric(left[field], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right[field], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=False):
            return False
    return _exact_text_fields(
        left.iloc[0].to_dict(),
        right.iloc[0].to_dict(),
        ("security_id", "currency", "source", "source_url", "retrieved_at", "source_hash"),
    ) and all(
        left[field].fillna("").astype(str).eq(right[field].fillna("").astype(str)).all()
        for field in ("security_id", "currency", "source", "source_url", "retrieved_at", "source_hash")
    )


def _merge_prices(
    existing: pd.DataFrame,
    bundle: CvrBundle,
) -> tuple[pd.DataFrame, int]:
    output = existing.copy()
    expected = _normalized_bundle_prices(bundle)
    target = output["security_id"].astype(str).eq(bundle.security_id)
    if target.any():
        if not _price_rows_are_exact(output.loc[target], expected):
            raise ValueError("Existing BMYRT price history is partial or conflicting.")
        return output.reset_index(drop=True), 0
    output = pd.concat([output, expected], ignore_index=True, sort=False)
    output = output.drop_duplicates(
        list(dataset_spec("daily_price_raw").primary_key), keep="last"
    )
    return output.sort_values(
        ["security_id", "session"], kind="stable"
    ).reset_index(drop=True), len(expected)


def _resolution_is_exact(row: Mapping[str, Any]) -> bool:
    return bool(
        _text(row.get("candidate_id"))
        == lifecycle_candidate_id(CELG_SECURITY_ID, CELG_LAST_SESSION)
        and _text(row.get("security_id")) == CELG_SECURITY_ID
        and _text(row.get("symbol")).upper() == CELG_SYMBOL
        and _date_text(row.get("last_price_date")) == CELG_LAST_SESSION
        and _text(row.get("resolution")) == "applied"
        and _text(row.get("event_id")) == CELG_STOCK_MERGER_EVENT_ID
        and all(
            not _text(row.get(field))
            for field in ("exception_code", "exception_reason", "recheck_after")
        )
        and _text(row.get("successor_security_id")) == BMY_SECURITY_ID
        and _text(row.get("successor_symbol")).upper() == BMY_SYMBOL
        and _text(row.get("source_url")) == MERGER_TERMS_URL
        and _text(row.get("source_hash")) == MERGER_TERMS_SHA256
    )


def _rewrite_resolution(
    existing: pd.DataFrame,
    *,
    terms: SourceArtifact,
    action_present: bool,
) -> tuple[pd.DataFrame, int]:
    output = existing.copy()
    matches = output["security_id"].astype(str).eq(CELG_SECURITY_ID)
    if int(matches.sum()) != 1:
        raise ValueError("CELG lifecycle resolution is missing or duplicated.")
    index = output.index[matches][0]
    row = output.loc[index].to_dict()
    if _resolution_is_exact(row):
        if not action_present:
            raise ValueError("CELG applied resolution exists without exact merger action.")
        return output.reset_index(drop=True), 0
    reviewed_exception = bool(
        _text(row.get("candidate_id"))
        == lifecycle_candidate_id(CELG_SECURITY_ID, CELG_LAST_SESSION)
        and _text(row.get("symbol")).upper() == CELG_SYMBOL
        and _date_text(row.get("last_price_date")) == CELG_LAST_SESSION
        and _text(row.get("resolution")) == "exception"
        and not _text(row.get("event_id"))
        and _text(row.get("exception_code")) == "unsupported_consideration"
        and _text(row.get("source_url")) == terms.source_url
        and _text(row.get("source_hash")) == terms.source_hash
    )
    if not reviewed_exception:
        raise ValueError("CELG lifecycle resolution is not the reviewed CVR exception.")
    if not action_present:
        raise ValueError("CELG exact merger action must exist before resolution rewrite.")
    values = {
        "resolution": "applied",
        "event_id": CELG_STOCK_MERGER_EVENT_ID,
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": REVIEWED_BY,
        "reviewed_at": REVIEWED_AT,
        "recheck_after": "",
        "successor_security_id": BMY_SECURITY_ID,
        "successor_symbol": BMY_SYMBOL,
        "source_url": terms.source_url,
        "source": "celg_bmy_cvr_exact_repair",
        "retrieved_at": terms.retrieved_at,
        "source_hash": terms.source_hash,
    }
    for field, value in values.items():
        output.at[index, field] = value
    return output.reset_index(drop=True), 1


def _rewrite_factors(
    existing: pd.DataFrame,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    bundle: CvrBundle,
    *,
    source_version: str,
) -> tuple[pd.DataFrame, int]:
    output = existing.copy()
    target = output["security_id"].astype(str).eq(bundle.security_id)
    if target.any():
        own = output.loc[target].copy()
        expected_sessions = tuple(_normalized_bundle_prices(bundle)["session"])
        observed_sessions = tuple(
            own["session"].map(_date_text).sort_values(kind="stable")
        )
        numeric = own[["split_factor", "total_return_factor"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if (
            len(own) != EXPECTED_CVR_SESSIONS
            or observed_sessions != expected_sessions
            or not np.allclose(numeric.to_numpy(dtype=float), 1.0, rtol=0, atol=0)
        ):
            raise ValueError("Existing BMYRT adjustment factors are partial or conflicting.")
        added = 0
    else:
        target_prices = prices.loc[
            prices["security_id"].astype(str).eq(bundle.security_id)
        ].copy()
        target_actions = actions.loc[
            actions["security_id"].astype(str).eq(bundle.security_id)
        ].copy()
        built = build_adjustment_factors(
            target_prices,
            target_actions,
            source_version=source_version,
        )
        output = pd.concat([output, built], ignore_index=True, sort=False)
        added = len(built)
    output["source_version"] = source_version
    output["calculated_at"] = REVIEWED_AT
    output["source"] = "derived"
    output["retrieved_at"] = REVIEWED_AT
    output["source_hash"] = source_version
    output = output.drop_duplicates(
        list(dataset_spec("adjustment_factors").primary_key), keep="last"
    )
    return output.sort_values(
        ["security_id", "session"], kind="stable"
    ).reset_index(drop=True), added


def _merge_official_exit_prices(
    existing: pd.DataFrame,
    model: OfficialExitMarkModel,
    policy: SourceArtifact,
) -> tuple[pd.DataFrame, int]:
    output = existing.copy()
    expected = _official_exit_price(model, policy)
    target = output["security_id"].astype(str).eq(model.security_id)
    if target.any():
        if not _price_rows_are_exact(output.loc[target], expected):
            raise ValueError("Existing BMYRT official exit mark is partial or conflicting.")
        return output.reset_index(drop=True), 0
    output = pd.concat([output, expected], ignore_index=True, sort=False)
    output = output.drop_duplicates(
        list(dataset_spec("daily_price_raw").primary_key), keep="last"
    )
    return output.sort_values(
        ["security_id", "session"], kind="stable"
    ).reset_index(drop=True), 1


def _official_exit_resolution_is_exact(row: Mapping[str, Any]) -> bool:
    return bool(
        _text(row.get("candidate_id"))
        == lifecycle_candidate_id(CELG_SECURITY_ID, CELG_LAST_SESSION)
        and _text(row.get("security_id")) == CELG_SECURITY_ID
        and _text(row.get("symbol")).upper() == CELG_SYMBOL
        and _date_text(row.get("last_price_date")) == CELG_LAST_SESSION
        and _text(row.get("resolution")) == "applied"
        and _text(row.get("event_id")) == CELG_STOCK_MERGER_EVENT_ID
        and all(
            not _text(row.get(field))
            for field in ("exception_code", "exception_reason", "recheck_after")
        )
        and _text(row.get("reviewed_by")) == OFFICIAL_EXIT_REVIEWED_BY
        and _text(row.get("reviewed_at")) == REVIEWED_AT
        and _text(row.get("successor_security_id")) == BMY_SECURITY_ID
        and _text(row.get("successor_symbol")).upper() == BMY_SYMBOL
        and _text(row.get("source_url")) == MERGER_TERMS_URL
        and _text(row.get("source")) == "celg_bmy_cvr_official_exit_mark_repair"
        and _text(row.get("source_hash")) == MERGER_TERMS_SHA256
    )


def _official_exit_child_resolution(
    termination: SourceArtifact,
) -> dict[str, Any]:
    return {
        "candidate_id": lifecycle_candidate_id(
            OFFICIAL_EXIT_SECURITY_ID, MERGER_SESSION
        ),
        "security_id": OFFICIAL_EXIT_SECURITY_ID,
        "symbol": CVR_SYMBOL,
        "last_price_date": MERGER_SESSION,
        "resolution": "applied",
        "event_id": OFFICIAL_EXIT_EVENT_ID,
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": OFFICIAL_EXIT_REVIEWED_BY,
        "reviewed_at": REVIEWED_AT,
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        "source_url": TERMINATION_URL,
        "source": "sec_bmy_2020_10k",
        "retrieved_at": termination.retrieved_at,
        "source_hash": TERMINATION_SHA256,
    }


def _official_exit_child_resolution_is_exact(
    row: Mapping[str, Any],
    termination: SourceArtifact,
) -> bool:
    expected = _official_exit_child_resolution(termination)
    return all(
        _text(row.get(field)) == _text(value)
        for field, value in expected.items()
    )


def _rewrite_official_exit_resolution(
    existing: pd.DataFrame,
    *,
    terms: SourceArtifact,
    termination: SourceArtifact,
    merger_action_present: bool,
    exit_action_present: bool,
) -> tuple[pd.DataFrame, int]:
    output = existing.copy()
    matches = output["security_id"].astype(str).eq(CELG_SECURITY_ID)
    if int(matches.sum()) != 1:
        raise ValueError("CELG lifecycle resolution is missing or duplicated.")
    index = output.index[matches][0]
    row = output.loc[index].to_dict()
    changed = 0
    if _official_exit_resolution_is_exact(row):
        if not merger_action_present:
            raise ValueError("CELG applied resolution exists without exact merger action.")
    else:
        reviewed_exception = bool(
            _text(row.get("candidate_id"))
            == lifecycle_candidate_id(CELG_SECURITY_ID, CELG_LAST_SESSION)
            and _text(row.get("symbol")).upper() == CELG_SYMBOL
            and _date_text(row.get("last_price_date")) == CELG_LAST_SESSION
            and _text(row.get("resolution")) == "exception"
            and not _text(row.get("event_id"))
            and _text(row.get("exception_code")) == "unsupported_consideration"
            and _text(row.get("source_url")) == terms.source_url
            and _text(row.get("source_hash")) == terms.source_hash
        )
        if not reviewed_exception:
            raise ValueError("CELG lifecycle resolution is not the reviewed CVR exception.")
        if not merger_action_present:
            raise ValueError("CELG exact merger action must exist before resolution rewrite.")
        values = {
            "resolution": "applied",
            "event_id": CELG_STOCK_MERGER_EVENT_ID,
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": OFFICIAL_EXIT_REVIEWED_BY,
            "reviewed_at": REVIEWED_AT,
            "recheck_after": "",
            "successor_security_id": BMY_SECURITY_ID,
            "successor_symbol": BMY_SYMBOL,
            "source_url": terms.source_url,
            "source": "celg_bmy_cvr_official_exit_mark_repair",
            "retrieved_at": terms.retrieved_at,
            "source_hash": terms.source_hash,
        }
        for field, value in values.items():
            output.at[index, field] = value
        changed += 1

    if not exit_action_present:
        raise ValueError(
            "BMYRT exact official exit action must exist before resolution rewrite."
        )
    child_matches = output["security_id"].astype(str).eq(
        OFFICIAL_EXIT_SECURITY_ID
    )
    if int(child_matches.sum()) > 1:
        raise ValueError("BMYRT lifecycle resolution is duplicated.")
    if child_matches.any():
        child = output.loc[child_matches].iloc[0].to_dict()
        if not _official_exit_child_resolution_is_exact(child, termination):
            raise ValueError(
                "BMYRT lifecycle resolution is partial or conflicting."
            )
    else:
        output = pd.concat(
            [output, pd.DataFrame([_official_exit_child_resolution(termination)])],
            ignore_index=True,
            sort=False,
        )
        changed += 1
    return (
        output.sort_values("candidate_id", kind="stable").reset_index(drop=True),
        changed,
    )


def _rewrite_official_exit_factors(
    existing: pd.DataFrame,
    model: OfficialExitMarkModel,
    *,
    source_version: str,
) -> tuple[pd.DataFrame, int]:
    output = existing.copy()
    # Current Parquet snapshots load factor sessions as pandas.Timestamp.
    # Appending MERGER_SESSION as a plain string produces an object column
    # containing Timestamp + str, which PyArrow cannot serialize.  Normalize
    # before both the exact/idempotent check and append path.
    output["session"] = pd.to_datetime(output["session"], errors="raise")
    target = output["security_id"].astype(str).eq(model.security_id)
    expected = pd.DataFrame(
        [
            {
                "security_id": model.security_id,
                "session": pd.Timestamp(MERGER_SESSION),
                "split_factor": 1.0,
                "total_return_factor": 1.0,
                "source_version": source_version,
                "calculated_at": REVIEWED_AT,
                "source": "derived",
                "retrieved_at": REVIEWED_AT,
                "source_hash": source_version,
            }
        ]
    )
    if target.any():
        own = output.loc[target].copy().reset_index(drop=True)
        comparable = (
            len(own) == 1
            and _date_text(own.iloc[0].get("session")) == MERGER_SESSION
            and math.isclose(float(own.iloc[0].get("split_factor")), 1.0)
            and math.isclose(float(own.iloc[0].get("total_return_factor")), 1.0)
            and _exact_text_fields(
                own.iloc[0].to_dict(),
                expected.iloc[0].to_dict(),
                (
                    "security_id",
                    "source_version",
                    "calculated_at",
                    "source",
                    "retrieved_at",
                    "source_hash",
                ),
            )
        )
        if not comparable:
            raise ValueError(
                "Existing BMYRT official exit-mark factor is partial or conflicting."
            )
        return output.reset_index(drop=True), 0
    output = pd.concat([output, expected], ignore_index=True, sort=False)
    output["source_version"] = source_version
    output["calculated_at"] = REVIEWED_AT
    output["source"] = "derived"
    output["retrieved_at"] = REVIEWED_AT
    output["source_hash"] = source_version
    output = output.drop_duplicates(
        list(dataset_spec("adjustment_factors").primary_key), keep="last"
    )
    return output.sort_values(
        ["security_id", "session"], kind="stable"
    ).reset_index(drop=True), 1


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
            return self.overrides[dataset]
        version = self.versions.get(dataset)
        if not version:
            return pd.DataFrame(columns=dataset_spec(dataset).required_columns)
        return self.base.read_frame(dataset, version)


def _validate_exact_actions(
    actions: pd.DataFrame,
    bundle: CvrBundle,
    terms: SourceArtifact,
    termination: SourceArtifact,
) -> None:
    expected = _official_actions(bundle, terms, termination)
    for row in expected.to_dict(orient="records"):
        matches = actions["event_id"].astype(str).eq(_text(row["event_id"]))
        if int(matches.sum()) != 1:
            raise ValueError(f"Exact CELG/BMYRT event is missing: {row['event_id']}.")
        observed = actions.loc[matches].iloc[0].to_dict()
        for field in (
            "event_id",
            "security_id",
            "action_type",
            "effective_date",
            "ex_date",
            "announcement_date",
            "record_date",
            "payment_date",
            "currency",
            "new_security_id",
            "new_symbol",
            "source_url",
            "source_kind",
            "source",
            "retrieved_at",
            "source_hash",
            "metadata",
        ):
            if _text(observed.get(field)) != _text(row.get(field)):
                raise ValueError(f"Exact CELG/BMYRT event field changed: {field}.")
        for field in ("cash_amount", "ratio"):
            a = pd.to_numeric(pd.Series([observed.get(field)]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
            if pd.isna(a) != pd.isna(b) or (
                not pd.isna(a) and not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-15)
            ):
                raise ValueError(f"Exact CELG/BMYRT event economic field changed: {field}.")


def _validate_official_exit_actions(
    actions: pd.DataFrame,
    model: OfficialExitMarkModel,
    terms: SourceArtifact,
    termination: SourceArtifact,
) -> None:
    expected = _official_exit_actions(model, terms, termination)
    for row in expected.to_dict(orient="records"):
        matches = actions["event_id"].astype(str).eq(_text(row["event_id"]))
        if int(matches.sum()) != 1:
            raise ValueError(
                f"Exact CELG/BMYRT official exit event is missing: {row['event_id']}."
            )
        observed = actions.loc[matches].iloc[0].to_dict()
        for field in (
            "event_id",
            "security_id",
            "action_type",
            "effective_date",
            "ex_date",
            "announcement_date",
            "record_date",
            "payment_date",
            "currency",
            "new_security_id",
            "new_symbol",
            "source_url",
            "source_kind",
            "source",
            "retrieved_at",
            "source_hash",
            "metadata",
        ):
            if _text(observed.get(field)) != _text(row.get(field)):
                raise ValueError(
                    f"Exact CELG/BMYRT official exit field changed: {field}."
                )
        for field in ("cash_amount", "ratio"):
            a = pd.to_numeric(pd.Series([observed.get(field)]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
            if pd.isna(a) != pd.isna(b) or (
                not pd.isna(a)
                and not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-15)
            ):
                raise ValueError(
                    f"Exact CELG/BMYRT official exit economic field changed: {field}."
                )


def prepare_frames(
    frames: Mapping[str, pd.DataFrame],
    bundle: CvrBundle,
    terms: SourceArtifact,
    termination: SourceArtifact,
    *,
    completed_session: str,
    factor_source_version: str = PLANNED_FACTOR_SOURCE,
) -> tuple[dict[str, pd.DataFrame], tuple[SourceArtifact, ...], dict[str, Any]]:
    missing = sorted(set(REQUIRED_DATASETS) - set(frames))
    if missing:
        raise ValueError("CELG/BMYRT repair is missing datasets: " + ", ".join(missing))
    validate_bundle(bundle)
    identity = _identity_artifact(bundle, terms, termination)
    master, history, identity_counts = _rewrite_master_history(
        frames["security_master"],
        frames["symbol_history"],
        bundle,
        terms,
        identity,
    )
    prices, price_rows_added = _merge_prices(frames["daily_price_raw"], bundle)
    additions = _official_actions(bundle, terms, termination)
    actions, action_rows_added = _merge_actions(
        frames["corporate_actions"], additions
    )
    _validate_exact_actions(actions, bundle, terms, termination)
    action_present = bool(
        actions["event_id"].astype(str).eq(CELG_STOCK_MERGER_EVENT_ID).sum() == 1
    )
    resolutions, resolution_rows_changed = _rewrite_resolution(
        frames["lifecycle_resolutions"],
        terms=terms,
        action_present=action_present,
    )
    factors, factor_rows_added = _rewrite_factors(
        frames["adjustment_factors"],
        prices,
        actions,
        bundle,
        source_version=factor_source_version,
    )
    artifacts = tuple((*bundle.artifacts, termination, identity))
    archive, archive_rows_added = _append_source_archive(
        frames["source_archive"],
        artifacts,
        completed_session=completed_session,
    )
    output = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "adjustment_factors": factors,
        "source_archive": archive,
    }
    for dataset, frame in output.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    return output, artifacts, {
        **identity_counts,
        "price_rows_added": price_rows_added,
        "corporate_action_rows_added": action_rows_added,
        "lifecycle_resolution_rows_changed": resolution_rows_changed,
        "adjustment_factor_rows_added": factor_rows_added,
        "source_archive_rows_added": archive_rows_added,
    }


def prepare_official_exit_frames(
    frames: Mapping[str, pd.DataFrame],
    model: OfficialExitMarkModel,
    terms: SourceArtifact,
    termination: SourceArtifact,
    *,
    completed_session: str,
    factor_source_version: str = PLANNED_FACTOR_SOURCE,
) -> tuple[dict[str, pd.DataFrame], tuple[SourceArtifact, ...], dict[str, Any]]:
    missing = sorted(set(REQUIRED_DATASETS) - set(frames))
    if missing:
        raise ValueError("CELG/BMYRT repair is missing datasets: " + ", ".join(missing))
    if model.mode != OFFICIAL_EXIT_MODE or model.security_id != OFFICIAL_EXIT_SECURITY_ID:
        raise ValueError("Official exit-mark model identity/mode changed.")
    _validate_catalog_selection(
        FrozenCatalogSelection(
            provider_code=model.provider_code,
            source_url=model.catalog_source_url,
            source_hash=model.catalog_source_hash,
            row=model.catalog_row,
            secondary_evidence=model.secondary_catalog_evidence,
        )
    )
    policy = _official_exit_policy_artifact(model, termination)
    identity = _official_exit_identity_artifact(model, terms, termination, policy)
    master, history, identity_counts = _rewrite_master_history(
        frames["security_master"],
        frames["symbol_history"],
        model,
        terms,
        identity,
    )
    prices, price_rows_added = _merge_official_exit_prices(
        frames["daily_price_raw"], model, policy
    )
    additions = _official_exit_actions(model, terms, termination)
    actions, action_rows_added = _merge_actions(
        frames["corporate_actions"], additions
    )
    _validate_official_exit_actions(actions, model, terms, termination)
    merger_action_present = bool(
        actions["event_id"].astype(str).eq(CELG_STOCK_MERGER_EVENT_ID).sum() == 1
    )
    exit_action_present = bool(
        actions["event_id"].astype(str).eq(OFFICIAL_EXIT_EVENT_ID).sum() == 1
    )
    resolutions, resolution_rows_changed = _rewrite_official_exit_resolution(
        frames["lifecycle_resolutions"],
        terms=terms,
        termination=termination,
        merger_action_present=merger_action_present,
        exit_action_present=exit_action_present,
    )
    factors, factor_rows_added = _rewrite_official_exit_factors(
        frames["adjustment_factors"],
        model,
        source_version=factor_source_version,
    )
    artifacts = (termination, policy, identity)
    archive, archive_rows_added = _append_source_archive(
        frames["source_archive"],
        artifacts,
        completed_session=completed_session,
    )
    output = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "adjustment_factors": factors,
        "source_archive": archive,
    }
    for dataset, frame in output.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    _validate_official_exit_frame_delta(
        frames,
        output,
        model,
        artifacts=artifacts,
        factor_source_version=factor_source_version,
    )
    return output, artifacts, {
        **identity_counts,
        "price_rows_added": price_rows_added,
        "corporate_action_rows_added": action_rows_added,
        "lifecycle_resolution_rows_changed": resolution_rows_changed,
        "adjustment_factor_rows_added": factor_rows_added,
        "source_archive_rows_added": archive_rows_added,
    }


def _sorted_frame_for_exact_compare(
    frame: pd.DataFrame,
    dataset: str,
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    selected = list(columns) if columns is not None else sorted(frame.columns)
    output = frame.reindex(columns=selected).copy()
    for column in dataset_spec(dataset).date_columns:
        if column not in output.columns:
            continue
        present = output[column].map(_text).ne("")
        normalized = pd.Series("", index=output.index, dtype="object")
        if present.any():
            parsed = pd.to_datetime(output.loc[present, column], errors="raise")
            normalized.loc[present] = parsed.dt.strftime("%Y-%m-%d")
        output[column] = normalized
    primary = [
        column
        for column in dataset_spec(dataset).primary_key
        if column in output.columns
    ]
    if primary and not output.empty:
        output = output.sort_values(primary, kind="stable")
    return output.reset_index(drop=True)


def _assert_filtered_frame_unchanged(
    before: pd.DataFrame,
    after: pd.DataFrame,
    dataset: str,
    *,
    before_mask: pd.Series,
    after_mask: pd.Series,
    columns: Iterable[str] | None = None,
) -> None:
    selected_columns = (
        tuple(columns)
        if columns is not None
        else tuple(sorted(set(before.columns) | set(after.columns)))
    )
    left = _sorted_frame_for_exact_compare(
        before.loc[before_mask], dataset, columns=selected_columns
    )
    right = _sorted_frame_for_exact_compare(
        after.loc[after_mask], dataset, columns=selected_columns
    )
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=True,
            check_categorical=False,
        )
    except AssertionError as exc:
        raise ValueError(
            f"CELG official_exit_mark changed unrelated {dataset} rows."
        ) from exc


def _validate_official_exit_frame_delta(
    before: Mapping[str, pd.DataFrame],
    after: Mapping[str, pd.DataFrame],
    model: OfficialExitMarkModel,
    *,
    artifacts: Iterable[SourceArtifact],
    factor_source_version: str,
) -> None:
    target_security_ids = {CELG_SECURITY_ID, model.security_id}
    for dataset in ("security_master", "symbol_history"):
        left = before[dataset]
        right = after[dataset]
        _assert_filtered_frame_unchanged(
            left,
            right,
            dataset,
            before_mask=~left["security_id"].astype(str).isin(target_security_ids),
            after_mask=~right["security_id"].astype(str).isin(target_security_ids),
        )

    left_prices = before["daily_price_raw"]
    right_prices = after["daily_price_raw"]
    _assert_filtered_frame_unchanged(
        left_prices,
        right_prices,
        "daily_price_raw",
        before_mask=~left_prices["security_id"].astype(str).eq(model.security_id),
        after_mask=~right_prices["security_id"].astype(str).eq(model.security_id),
    )

    target_event_ids = {
        CVR_DISTRIBUTION_EVENT_ID,
        CELG_STOCK_MERGER_EVENT_ID,
        OFFICIAL_EXIT_EVENT_ID,
        OFFICIAL_RESIDUAL_TERMINATION_EVENT_ID,
    }
    left_actions = before["corporate_actions"]
    right_actions = after["corporate_actions"]
    _assert_filtered_frame_unchanged(
        left_actions,
        right_actions,
        "corporate_actions",
        before_mask=~left_actions["event_id"].astype(str).isin(target_event_ids),
        after_mask=~right_actions["event_id"].astype(str).isin(target_event_ids),
    )

    left_resolutions = before["lifecycle_resolutions"]
    right_resolutions = after["lifecycle_resolutions"]
    _assert_filtered_frame_unchanged(
        left_resolutions,
        right_resolutions,
        "lifecycle_resolutions",
        before_mask=~left_resolutions["security_id"]
        .astype(str)
        .isin(target_security_ids),
        after_mask=~right_resolutions["security_id"]
        .astype(str)
        .isin(target_security_ids),
    )

    artifact_hashes = {artifact.source_hash for artifact in artifacts}
    left_archive = before["source_archive"]
    right_archive = after["source_archive"]
    _assert_filtered_frame_unchanged(
        left_archive,
        right_archive,
        "source_archive",
        before_mask=~left_archive["source_hash"].astype(str).isin(artifact_hashes),
        after_mask=~right_archive["source_hash"].astype(str).isin(artifact_hashes),
    )

    factor_economics = (
        "security_id",
        "session",
        "split_factor",
        "total_return_factor",
    )
    left_factors = before["adjustment_factors"]
    right_factors = after["adjustment_factors"]
    _assert_filtered_frame_unchanged(
        left_factors,
        right_factors,
        "adjustment_factors",
        before_mask=~left_factors["security_id"].astype(str).eq(model.security_id),
        after_mask=~right_factors["security_id"].astype(str).eq(model.security_id),
        columns=factor_economics,
    )
    expected_provenance = {
        "source_version": factor_source_version,
        "calculated_at": REVIEWED_AT,
        "source": "derived",
        "retrieved_at": REVIEWED_AT,
        "source_hash": factor_source_version,
    }
    for field, expected in expected_provenance.items():
        if not right_factors[field].astype(str).eq(expected).all():
            raise ValueError(
                "CELG official_exit_mark factor provenance rewrite is incomplete: "
                f"{field}."
            )


def _structural_target_present(
    frames: Mapping[str, pd.DataFrame], bundle: CvrBundle
) -> bool:
    event_ids = {
        CVR_DISTRIBUTION_EVENT_ID,
        CELG_STOCK_MERGER_EVENT_ID,
        canonical_lifecycle_event_id(
            bundle.security_id, "delisting", CVR_TERMINATION_DATE
        ),
    }
    return bool(
        frames["security_master"]["security_id"].astype(str).eq(bundle.security_id).any()
        or frames["symbol_history"]["security_id"].astype(str).eq(bundle.security_id).any()
        or frames["daily_price_raw"]["security_id"].astype(str).eq(bundle.security_id).any()
        or frames["adjustment_factors"]["security_id"].astype(str).eq(bundle.security_id).any()
        or frames["corporate_actions"]["event_id"].astype(str).isin(event_ids).any()
        or frames["lifecycle_resolutions"]["event_id"]
        .astype(str)
        .eq(CELG_STOCK_MERGER_EVENT_ID)
        .any()
    )


def _target_is_exact(
    frames: Mapping[str, pd.DataFrame],
    bundle: CvrBundle,
    terms: SourceArtifact,
    termination: SourceArtifact,
) -> bool:
    try:
        identity = _identity_artifact(bundle, terms, termination)
        expected_master = _expected_cvr_master(bundle, identity)
        expected_history = _expected_cvr_history(bundle, identity)
        master = frames["security_master"]
        history = frames["symbol_history"]
        master_row = master.loc[
            master["security_id"].astype(str).eq(bundle.security_id)
        ]
        history_row = history.loc[
            history["security_id"].astype(str).eq(bundle.security_id)
        ]
        if (
            len(master_row) != 1
            or len(history_row) != 1
            or not _exact_text_fields(
                master_row.iloc[0].to_dict(), expected_master, expected_master
            )
            or not _exact_text_fields(
                history_row.iloc[0].to_dict(), expected_history, expected_history
            )
        ):
            return False
        celg_master = master.loc[
            master["security_id"].astype(str).eq(CELG_SECURITY_ID)
        ]
        celg_history = history.loc[
            history["security_id"].astype(str).eq(CELG_SECURITY_ID)
            & history["symbol"].astype(str).str.upper().eq(CELG_SYMBOL)
        ]
        if (
            len(celg_master) != 1
            or len(celg_history) != 1
            or _date_text(celg_master.iloc[0].get("active_to")) != CELG_LAST_SESSION
            or _date_text(celg_history.iloc[0].get("effective_to"))
            != CELG_LAST_SESSION
        ):
            return False
        celg_master = master.loc[
            master["security_id"].astype(str).eq(CELG_SECURITY_ID)
        ]
        celg_history = history.loc[
            history["security_id"].astype(str).eq(CELG_SECURITY_ID)
            & history["symbol"].astype(str).str.upper().eq(CELG_SYMBOL)
        ]
        if (
            len(celg_master) != 1
            or len(celg_history) != 1
            or _date_text(celg_master.iloc[0].get("active_to")) != CELG_LAST_SESSION
            or _date_text(celg_history.iloc[0].get("effective_to"))
            != CELG_LAST_SESSION
        ):
            return False
        prices = frames["daily_price_raw"].loc[
            frames["daily_price_raw"]["security_id"]
            .astype(str)
            .eq(bundle.security_id)
        ]
        if not _price_rows_are_exact(prices, _normalized_bundle_prices(bundle)):
            return False
        _validate_exact_actions(
            frames["corporate_actions"], bundle, terms, termination
        )
        resolutions = frames["lifecycle_resolutions"].loc[
            frames["lifecycle_resolutions"]["security_id"]
            .astype(str)
            .eq(CELG_SECURITY_ID)
        ]
        if len(resolutions) != 1 or not _resolution_is_exact(
            resolutions.iloc[0].to_dict()
        ):
            return False
        factors = frames["adjustment_factors"].loc[
            frames["adjustment_factors"]["security_id"]
            .astype(str)
            .eq(bundle.security_id)
        ]
        if len(factors) != EXPECTED_CVR_SESSIONS:
            return False
        factor_sessions = tuple(
            factors["session"].map(_date_text).sort_values(kind="stable")
        )
        factor_values = factors[["split_factor", "total_return_factor"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if factor_sessions != _expected_sessions() or not np.allclose(
            factor_values.to_numpy(dtype=float), 1.0, rtol=0, atol=0
        ):
            return False
        archive = frames["source_archive"]
        for artifact in (*bundle.artifacts, termination, identity):
            match = (
                archive["source_hash"].astype(str).eq(artifact.source_hash)
                & archive.get("source_url", pd.Series("", index=archive.index))
                .astype(str)
                .eq(artifact.source_url)
                & archive["source"].astype(str).eq(artifact.source)
            )
            if int(match.sum()) != 1:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _official_exit_target_is_exact(
    frames: Mapping[str, pd.DataFrame],
    model: OfficialExitMarkModel,
    terms: SourceArtifact,
    termination: SourceArtifact,
    *,
    release_warnings: Iterable[str],
) -> bool:
    try:
        if OFFICIAL_EXIT_WARNING not in tuple(release_warnings):
            return False
        policy = _official_exit_policy_artifact(model, termination)
        identity = _official_exit_identity_artifact(
            model, terms, termination, policy
        )
        expected_master = _expected_cvr_master(model, identity)
        expected_history = _expected_cvr_history(model, identity)
        master = frames["security_master"]
        history = frames["symbol_history"]
        master_row = master.loc[
            master["security_id"].astype(str).eq(model.security_id)
        ]
        history_row = history.loc[
            history["security_id"].astype(str).eq(model.security_id)
        ]
        if (
            len(master_row) != 1
            or len(history_row) != 1
            or not _exact_text_fields(
                master_row.iloc[0].to_dict(), expected_master, expected_master
            )
            or not _exact_text_fields(
                history_row.iloc[0].to_dict(), expected_history, expected_history
            )
        ):
            return False
        prices = frames["daily_price_raw"].loc[
            frames["daily_price_raw"]["security_id"].astype(str).eq(model.security_id)
        ]
        if not _price_rows_are_exact(prices, _official_exit_price(model, policy)):
            return False
        _validate_official_exit_actions(
            frames["corporate_actions"], model, terms, termination
        )
        resolutions = frames["lifecycle_resolutions"]
        celg_resolution = resolutions.loc[
            resolutions["security_id"].astype(str).eq(CELG_SECURITY_ID)
        ]
        child_resolution = resolutions.loc[
            resolutions["security_id"].astype(str).eq(model.security_id)
        ]
        if (
            len(celg_resolution) != 1
            or not _official_exit_resolution_is_exact(
                celg_resolution.iloc[0].to_dict()
            )
            or len(child_resolution) != 1
            or not _official_exit_child_resolution_is_exact(
                child_resolution.iloc[0].to_dict(), termination
            )
        ):
            return False
        factors = frames["adjustment_factors"].loc[
            frames["adjustment_factors"]["security_id"].astype(str).eq(model.security_id)
        ]
        if len(factors) != 1:
            return False
        factor = factors.iloc[0]
        if (
            _date_text(factor.get("session")) != MERGER_SESSION
            or not math.isclose(float(factor.get("split_factor")), 1.0)
            or not math.isclose(float(factor.get("total_return_factor")), 1.0)
            or _text(factor.get("source")) != "derived"
            or not _text(factor.get("source_version"))
            or _text(factor.get("source_hash")) != _text(factor.get("source_version"))
        ):
            return False
        archive = frames["source_archive"]
        for artifact in (termination, policy, identity):
            match = (
                archive["source_hash"].astype(str).eq(artifact.source_hash)
                & archive.get("source_url", pd.Series("", index=archive.index))
                .astype(str)
                .eq(artifact.source_url)
                & archive["source"].astype(str).eq(artifact.source)
            )
            if int(match.sum()) != 1:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _capture_pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}.")
        output[dataset] = etag
    return output


def _lifecycle_coverage(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> LifecycleCoverageReport:
    report, _ = _lifecycle_coverage_report(repository, release, frames)
    report.raise_for_errors()
    if not report.valid or report.open_count:
        raise ValueError("CELG/BMYRT repair did not close lifecycle coverage.")
    return report


def _lifecycle_coverage_report(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> tuple[LifecycleCoverageReport, pd.DataFrame]:
    candidate_repository = _CandidateRepository(
        repository, release.dataset_versions, frames
    )
    candidates = tuple(
        build_lifecycle_candidates(candidate_repository, release=release)
    )
    candidate_frame = (
        pd.DataFrame([asdict(item) for item in candidates])
        if candidates
        else pd.DataFrame(columns=("security_id", "last_price_date"))
    )
    report = validate_lifecycle_coverage(
        candidate_frame,
        frames["lifecycle_resolutions"],
        frames["corporate_actions"],
        completed_session=release.completed_session,
    )
    return report, candidate_frame


def _coverage_issues_payload(report: LifecycleCoverageReport) -> list[dict[str, Any]]:
    return [
        {
            "code": issue.code,
            "message": issue.message,
            "candidate_ids": sorted(issue.candidate_ids),
        }
        for issue in sorted(
            report.issues,
            key=lambda item: (
                item.code,
                item.message,
                tuple(sorted(item.candidate_ids)),
            ),
        )
    ]


def _coverage_issue_tokens(report: LifecycleCoverageReport) -> set[tuple[str, str, str]]:
    output: set[tuple[str, str, str]] = set()
    for issue in report.issues:
        if issue.candidate_ids:
            output.update(
                (issue.code, candidate_id, "")
                for candidate_id in issue.candidate_ids
            )
        else:
            output.add((issue.code, "", issue.message))
    return output


def _official_exit_coverage_delta(
    repository: LocalDatasetRepository,
    release: DataRelease,
    before: Mapping[str, pd.DataFrame],
    after: Mapping[str, pd.DataFrame],
    *,
    expected_transition: bool,
) -> tuple[LifecycleCoverageReport, LifecycleCoverageReport, dict[str, Any]]:
    baseline, baseline_candidates = _lifecycle_coverage_report(
        repository, release, before
    )
    prepared, prepared_candidates = _lifecycle_coverage_report(
        repository, release, after
    )
    child_candidate_id = lifecycle_candidate_id(
        OFFICIAL_EXIT_SECURITY_ID, MERGER_SESSION
    )
    baseline_candidate_ids = {
        lifecycle_candidate_id(row.security_id, row.last_price_date)
        for row in baseline_candidates.itertuples(index=False)
    }
    prepared_candidate_ids = {
        lifecycle_candidate_id(row.security_id, row.last_price_date)
        for row in prepared_candidates.itertuples(index=False)
    }
    added_candidates = prepared_candidate_ids - baseline_candidate_ids
    removed_candidates = baseline_candidate_ids - prepared_candidate_ids
    if expected_transition:
        if (
            prepared.candidate_count != baseline.candidate_count + 1
            or added_candidates != {child_candidate_id}
            or removed_candidates
        ):
            raise ValueError(
                "CELG official_exit_mark candidate expansion is not exactly BMYRT."
            )
        child_candidate = prepared_candidates.loc[
            prepared_candidates["security_id"].astype(str).eq(
                OFFICIAL_EXIT_SECURITY_ID
            )
            & prepared_candidates["last_price_date"].map(_date_text).eq(
                MERGER_SESSION
            )
        ]
        if (
            len(child_candidate) != 1
            or _text(child_candidate.iloc[0].get("symbol")).upper() != CVR_SYMBOL
            or _text(child_candidate.iloc[0].get("exchange")).upper() != "NYSE"
            or _date_text(child_candidate.iloc[0].get("active_to"))
            != CVR_LAST_SESSION
        ):
            raise ValueError(
                "CELG official_exit_mark BMYRT candidate identity is not exact."
            )
        baseline_rows = baseline_candidates.sort_values(
            ["security_id", "last_price_date"], kind="stable"
        ).reset_index(drop=True)
        prepared_existing_rows = prepared_candidates.loc[
            ~prepared_candidates["security_id"].astype(str).eq(
                OFFICIAL_EXIT_SECURITY_ID
            )
        ].sort_values(
            ["security_id", "last_price_date"], kind="stable"
        ).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                baseline_rows,
                prepared_existing_rows,
                check_dtype=False,
                check_like=False,
            )
        except AssertionError as exc:
            raise ValueError(
                "CELG official_exit_mark changed an existing lifecycle candidate."
            ) from exc
        if prepared.resolution_count != baseline.resolution_count + 1:
            raise ValueError(
                "CELG official_exit_mark did not add exactly one BMYRT resolution."
            )
    else:
        if (
            prepared.candidate_count != baseline.candidate_count
            or prepared.candidate_set_sha256 != baseline.candidate_set_sha256
            or added_candidates
            or removed_candidates
        ):
            raise ValueError(
                "Already-applied CELG official_exit_mark changed the global "
                "lifecycle candidate set."
            )
        if prepared.resolution_count != baseline.resolution_count:
            raise ValueError(
                "Already-applied CELG official_exit_mark changed the global "
                "lifecycle resolution count."
            )
    new_issue_tokens = _coverage_issue_tokens(prepared) - _coverage_issue_tokens(
        baseline
    )
    if new_issue_tokens:
        raise ValueError(
            "CELG official_exit_mark introduced lifecycle coverage issues: "
            + repr(sorted(new_issue_tokens))
        )
    if (
        prepared.open_count > baseline.open_count
        or prepared.closed_count < baseline.closed_count
    ):
        raise ValueError(
            "CELG official_exit_mark regressed global lifecycle closure."
        )
    if expected_transition:
        if not (
            prepared.applied_count == baseline.applied_count + 2
            and prepared.exception_count == baseline.exception_count - 1
            and prepared.closed_count == baseline.closed_count + 1
            and prepared.open_count == baseline.open_count
        ):
            raise ValueError(
                "CELG official_exit_mark did not perform the exact CELG "
                "exception transition plus BMYRT applied closure."
            )
    elif not (
        prepared.applied_count == baseline.applied_count
        and prepared.exception_count == baseline.exception_count
    ):
        raise ValueError(
            "Already-applied CELG official_exit_mark changed lifecycle counts."
        )

    target_candidate_id = lifecycle_candidate_id(
        CELG_SECURITY_ID, CELG_LAST_SESSION
    )
    scoped_candidate_ids = {target_candidate_id, child_candidate_id}
    scoped_candidates = prepared_candidates.loc[
        (
            prepared_candidates["security_id"].astype(str).eq(CELG_SECURITY_ID)
            & prepared_candidates["last_price_date"].map(_date_text).eq(
                CELG_LAST_SESSION
            )
        )
        | (
            prepared_candidates["security_id"].astype(str).eq(
                OFFICIAL_EXIT_SECURITY_ID
            )
            & prepared_candidates["last_price_date"].map(_date_text).eq(
                MERGER_SESSION
            )
        )
    ]
    scoped_resolutions = after["lifecycle_resolutions"].loc[
        after["lifecycle_resolutions"]["candidate_id"]
        .astype(str)
        .isin(scoped_candidate_ids)
    ]
    if len(scoped_candidates) != 2 or len(scoped_resolutions) != 2:
        raise ValueError(
            "CELG/BMYRT scoped lifecycle candidates/resolutions are not exact."
        )
    scoped = validate_lifecycle_coverage(
        scoped_candidates,
        scoped_resolutions,
        after["corporate_actions"],
        completed_session=release.completed_session,
    )
    scoped.raise_for_errors()
    if not scoped.valid or scoped.open_count:
        raise ValueError("CELG/BMYRT scoped lifecycle coverage is not closed.")

    baseline_issues = _coverage_issues_payload(baseline)
    prepared_issues = _coverage_issues_payload(prepared)
    baseline_issue_fingerprint = sha256_bytes(
        _canonical_json_bytes(baseline_issues)
    )
    return prepared, scoped, {
        # Reaching this point proves the repair added no issue and closed the
        # exact CELG/BMYRT scope. Pre-existing global drift is preserved visibly
        # for the subsequent report regeneration/finalizer, but is not caused
        # by (or a reason to prevent) this baseline-delta transaction.
        "apply_ready": True,
        "apply_blockers": [],
        "global_lifecycle_valid": prepared.valid,
        "inherited_global_lifecycle_drift": not prepared.valid,
        "required_follow_up": (
            []
            if prepared.valid
            else [
                "regenerate_current_release_sec_lifecycle_report",
                "run_lifecycle_finalizer",
            ]
        ),
        "baseline_global_coverage": baseline.manifest_metadata(),
        "prepared_global_coverage": prepared.manifest_metadata(),
        "celg_scoped_coverage": scoped.manifest_metadata(),
        "inherited_global_lifecycle_issues": prepared_issues,
        "baseline_issue_fingerprint": baseline_issue_fingerprint,
        "prepared_issue_fingerprint": sha256_bytes(
            _canonical_json_bytes(prepared_issues)
        ),
        "candidate_set_unchanged": not expected_transition,
        "expected_successor_candidate_added": expected_transition,
        "new_global_issue_count": 0,
    }


def prepare_repair(
    repository: LocalDatasetRepository,
    bundle: CvrBundle,
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    terms = _load_merger_terms(repository, release)
    termination = _load_termination_evidence(evidence_dir)
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    if _structural_target_present(frames, bundle):
        if not _target_is_exact(frames, bundle, terms, termination):
            raise ValueError(
                "Partial or conflicting CELG/BMYRT structural state blocks repair."
            )
        coverage = _lifecycle_coverage(repository, release, frames)
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags={},
            frames={},
            artifacts=(),
            coverage=coverage,
            transaction_id="",
            planned_versions={},
            summary={
                "status": "already_applied",
                "base_release_version": release.version,
                "provider_symbol": f"{bundle.provider_code}.US",
                "security_id": bundle.security_id,
                "price_rows": EXPECTED_CVR_SESSIONS,
                "network_accessed": False,
                "eodhd_calls": 0,
                "r2_accessed": False,
            },
        )
    if bundle.fetched_against_release != release.version:
        raise RuntimeError(
            "BMYRT bundle was fetched against another release; rerun --fetch: "
            f"bundle={bundle.fetched_against_release}, current={release.version}."
        )
    transaction_id = uuid.uuid4().hex
    planned = {
        dataset: (
            "celg-bmy-cvr-"
            f"{release.completed_session.replace('-', '')}-"
            f"{transaction_id}-{dataset}"
        )
        for dataset in WRITE_DATASETS
    }
    factor_source = (
        planned["daily_price_raw"] + "+" + planned["corporate_actions"]
    )
    prepared_frames, artifacts, counts = prepare_frames(
        frames,
        bundle,
        terms,
        termination,
        completed_session=release.completed_session,
        factor_source_version=factor_source,
    )
    candidate_repository = _CandidateRepository(
        repository, release.dataset_versions, prepared_frames
    )
    validate_repository_snapshot(candidate_repository).raise_for_errors()
    coverage = _lifecycle_coverage(repository, release, prepared_frames)
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=_capture_pointer_etags(repository, release),
        frames=prepared_frames,
        artifacts=artifacts,
        coverage=coverage,
        transaction_id=transaction_id,
        planned_versions=planned,
        summary={
            "status": "validated_offline_plan",
            "base_release_version": release.version,
            "completed_session": release.completed_session,
            "provider_symbol": f"{bundle.provider_code}.US",
            "security_id": bundle.security_id,
            "price_rows": EXPECTED_CVR_SESSIONS,
            "first_session": MERGER_SESSION,
            "last_session": CVR_LAST_SESSION,
            "official_event_ids": [
                CVR_DISTRIBUTION_EVENT_ID,
                CELG_STOCK_MERGER_EVENT_ID,
                canonical_lifecycle_event_id(
                    bundle.security_id, "delisting", CVR_TERMINATION_DATE
                ),
            ],
            "consideration": {
                "bmy_ratio": 1.0,
                "cash_per_celg_share": MERGER_CASH_PER_SHARE,
                "bmyrt_ratio": 1.0,
                "bmyrt_terminal_payment": 0.0,
            },
            "economic_basis_fraction": CVR_ECONOMIC_BASIS_FRACTION,
            "economic_basis_kind": (
                "economic_relative_fair_value_including_cash_not_tax_basis"
            ),
            "official_source_hashes": [
                MERGER_TERMS_SHA256,
                TERMINATION_SHA256,
            ],
            "provider_artifact_hashes": sorted(
                artifact.source_hash for artifact in bundle.artifacts
            ),
            "coverage": coverage.manifest_metadata(),
            **counts,
            "fetch_http_attempts": bundle.http_attempts,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


def prepare_official_exit_repair(
    repository: LocalDatasetRepository,
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    terms = _load_merger_terms(repository, release)
    termination = _load_termination_evidence(evidence_dir)
    selection = _frozen_catalog_selection(repository, release)
    model = _official_exit_model(selection)
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    if _structural_target_present(frames, model):
        if not _official_exit_target_is_exact(
            frames,
            model,
            terms,
            termination,
            release_warnings=release.warnings,
        ):
            raise ValueError(
                "Partial or conflicting CELG/BMYRT official_exit_mark state blocks repair."
            )
        coverage, _scoped, coverage_delta = _official_exit_coverage_delta(
            repository,
            release,
            frames,
            frames,
            expected_transition=False,
        )
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags={},
            frames={},
            artifacts=(),
            coverage=coverage,
            transaction_id="",
            planned_versions={},
            summary={
                "status": "already_applied",
                "operation": "repair_us_celg_bmy_cvr_official_exit_mark",
                "model_mode": OFFICIAL_EXIT_MODE,
                "base_release_version": release.version,
                "provider_symbol": CVR_PROVIDER_SYMBOL,
                "security_id": model.security_id,
                "price_rows": 1,
                "trading_path_supported": False,
                "release_warnings": [OFFICIAL_EXIT_WARNING],
                "prior_failed_eod_note": dict(PRIOR_FAILED_EOD_NOTE),
                "coverage": coverage.manifest_metadata(),
                **coverage_delta,
                "network_accessed": False,
                "eodhd_calls": 0,
                "r2_accessed": False,
            },
        )
    transaction_id = uuid.uuid4().hex
    planned = {
        dataset: (
            "celg-bmy-cvr-official-exit-"
            f"{release.completed_session.replace('-', '')}-"
            f"{transaction_id}-{dataset}"
        )
        for dataset in WRITE_DATASETS
    }
    factor_source = planned["daily_price_raw"] + "+" + planned["corporate_actions"]
    prepared_frames, artifacts, counts = prepare_official_exit_frames(
        frames,
        model,
        terms,
        termination,
        completed_session=release.completed_session,
        factor_source_version=factor_source,
    )
    candidate_repository = _CandidateRepository(
        repository, release.dataset_versions, prepared_frames
    )
    validate_repository_snapshot(candidate_repository).raise_for_errors()
    coverage, _scoped, coverage_delta = _official_exit_coverage_delta(
        repository,
        release,
        frames,
        prepared_frames,
        expected_transition=True,
    )
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=_capture_pointer_etags(repository, release),
        frames=prepared_frames,
        artifacts=artifacts,
        coverage=coverage,
        transaction_id=transaction_id,
        planned_versions=planned,
        summary={
            "status": "validated_offline_plan",
            "operation": "repair_us_celg_bmy_cvr_official_exit_mark",
            "model_mode": OFFICIAL_EXIT_MODE,
            "base_release_version": release.version,
            "completed_session": release.completed_session,
            "provider_symbol": CVR_PROVIDER_SYMBOL,
            "security_id": model.security_id,
            "price_rows": 1,
            "price_row_kind": "official_valuation_mark_not_provider_ohlcv",
            "first_session": MERGER_SESSION,
            "last_session": MERGER_SESSION,
            "trading_path_supported": False,
            "unsupported_trading_path": {
                "from": MERGER_SESSION,
                "to": CVR_LAST_SESSION,
                "expected_sessions_if_supported": EXPECTED_CVR_SESSIONS,
            },
            "official_event_ids": [
                CVR_DISTRIBUTION_EVENT_ID,
                CELG_STOCK_MERGER_EVENT_ID,
                OFFICIAL_EXIT_EVENT_ID,
                OFFICIAL_RESIDUAL_TERMINATION_EVENT_ID,
            ],
            "consideration": {
                "bmy_ratio": 1.0,
                "cash_per_celg_share": MERGER_CASH_PER_SHARE,
                "bmyrt_ratio": 1.0,
                "bmyrt_immediate_exit_mark": CVR_REFERENCE_CLOSE,
                "bmyrt_residual_terminal_payment": 0.0,
            },
            "official_source_hashes": [
                MERGER_TERMS_SHA256,
                TERMINATION_SHA256,
            ],
            "provider_price_artifact_claimed": False,
            "prior_failed_eod_note": dict(PRIOR_FAILED_EOD_NOTE),
            "release_warnings": [OFFICIAL_EXIT_WARNING],
            "planned_quality": str(DataQuality.DEGRADED),
            "coverage": coverage.manifest_metadata(),
            **coverage_delta,
            **counts,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


def _persist_archive_payloads(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    archive = prepared.frames["source_archive"]
    for artifact in prepared.artifacts:
        matches = archive.loc[
            archive["source_hash"].astype(str).eq(artifact.source_hash)
            & archive.get("source_url", pd.Series("", index=archive.index))
            .astype(str)
            .eq(artifact.source_url)
            & archive["source"].astype(str).eq(artifact.source)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Prepared archive row is not exact: {artifact.source}/{artifact.source_hash}."
            )
        path = repository.root / _text(matches.iloc[0]["object_path"])
        if path.is_file():
            try:
                current = gzip.decompress(path.read_bytes())
            except (OSError, EOFError) as exc:
                raise RuntimeError(f"Immutable archive is invalid gzip: {path}.") from exc
            if current != artifact.content:
                raise RuntimeError(f"Conflicting immutable archive payload: {path}.")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, gzip.compress(artifact.content, mtime=0))
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"Immutable archive verification failed: {path}.")


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / "recovery/us-celg-bmy-cvr"
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved CELG/BMYRT recovery marker blocks writes.")
        transactions = repository.root / "transactions/us-celg-bmy-cvr"
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted CELG/BMYRT transaction blocks writes: {journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_release_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, etag = repository.current_release()
    if (
        release is None
        or release.version != prepared.release.version
        or etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed during CELG/BMYRT validation.")


def _restore_transaction_state(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            release = DataRelease.from_bytes(current.data)
            belongs = (
                bool(committed_release_version)
                and release.version == committed_release_version
            ) or all(
                release.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not belongs:
                raise RuntimeError(f"unexpected release during rollback: {release.version}")
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
                        f"unexpected pointer during rollback: {pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _release_quality(
    release: DataRelease,
    warnings: Iterable[str] = (),
) -> DataQuality:
    if str(release.quality) == str(DataQuality.BLOCKED):
        return DataQuality.BLOCKED
    return DataQuality.DEGRADED if tuple(warnings) else DataQuality.VALID


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
) -> dict[str, Any]:
    current = repository.manifest_for_version(
        dataset, prepared.release.dataset_versions[dataset]
    )
    metadata = dict(current.metadata)
    metadata.update(
        {
            "operation": prepared.summary.get(
                "operation", "repair_us_celg_bmy_cvr"
            ),
            "official_merger_sha256": MERGER_TERMS_SHA256,
            "official_termination_sha256": TERMINATION_SHA256,
            "provider_symbol": prepared.summary["provider_symbol"],
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    release_warnings = tuple(prepared.summary.get("release_warnings", ()))
    if release_warnings:
        metadata["_logical_warnings"] = list(release_warnings)
        metadata["_logical_quality"] = str(DataQuality.DEGRADED)
        metadata["model_mode"] = prepared.summary.get("model_mode", "")
        metadata["trading_path_supported"] = bool(
            prepared.summary.get("trading_path_supported", True)
        )
        metadata["provider_price_artifact_claimed"] = bool(
            prepared.summary.get("provider_price_artifact_claimed", False)
        )
    if dataset == "corporate_actions":
        metadata["celg_bmyrt_event_ids"] = list(
            prepared.summary["official_event_ids"]
        )
    elif dataset == "adjustment_factors":
        source_version = (
            prepared.planned_versions["daily_price_raw"]
            + "+"
            + prepared.planned_versions["corporate_actions"]
        )
        metadata.update(
            {
                "source_daily_price_version": prepared.planned_versions[
                    "daily_price_raw"
                ],
                "source_corporate_actions_version": prepared.planned_versions[
                    "corporate_actions"
                ],
                "source_version": source_version,
            }
        )
    elif dataset == "lifecycle_resolutions":
        if prepared.coverage is None:
            raise RuntimeError("Lifecycle coverage metadata is missing.")
        metadata.update(prepared.coverage.manifest_metadata())
        if prepared.summary.get("model_mode") == OFFICIAL_EXIT_MODE:
            metadata.update(
                {
                    "global_lifecycle_valid": bool(
                        prepared.summary.get("global_lifecycle_valid", False)
                    ),
                    "inherited_global_lifecycle_drift": bool(
                        prepared.summary.get(
                            "inherited_global_lifecycle_drift", False
                        )
                    ),
                    "baseline_issue_fingerprint": prepared.summary.get(
                        "baseline_issue_fingerprint", ""
                    ),
                    "prepared_issue_fingerprint": prepared.summary.get(
                        "prepared_issue_fingerprint", ""
                    ),
                    "inherited_global_lifecycle_issues": list(
                        prepared.summary.get(
                            "inherited_global_lifecycle_issues", ()
                        )
                    ),
                    "celg_scoped_coverage": dict(
                        prepared.summary.get("celg_scoped_coverage", {})
                    ),
                    "required_follow_up": list(
                        prepared.summary.get("required_follow_up", ())
                    ),
                }
            )
        metadata["output_versions"] = {
            **dict(metadata.get("output_versions") or {}),
            **dict(prepared.planned_versions),
        }
    elif dataset == "source_archive":
        metadata["celg_bmyrt_source_hashes"] = sorted(
            artifact.source_hash for artifact in prepared.artifacts
        )
    return metadata


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_applied":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    if prepared.summary.get("apply_ready") is False:
        blockers = tuple(prepared.summary.get("apply_blockers", ()))
        raise RuntimeError(
            "CELG/BMYRT apply is blocked by inherited global lifecycle drift: "
            + ", ".join(blockers)
        )
    inject_failure = inject_failure or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(repository, prepared)
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before CELG/BMYRT apply.")
            old_pointers[dataset] = value.data
        journal_path = (
            repository.root
            / "transactions/us-celg-bmy-cvr"
            / f"{prepared.transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "us_celg_bmy_cvr_transaction/v1",
            "transaction_id": prepared.transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": dict(prepared.planned_versions),
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            _persist_archive_payloads(repository, prepared)
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(repository, prepared, dataset),
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=prepared.planned_versions[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                versions[dataset] = result.manifest.version
                inject_failure(f"after_write:{dataset}")
            release_warnings = tuple(
                dict.fromkeys(
                    (
                        *prepared.release.warnings,
                        *tuple(prepared.summary.get("release_warnings", ())),
                    )
                )
            )
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=_release_quality(prepared.release, release_warnings),
                warnings=release_warnings,
                expected_etag=prepared.release_etag,
            )
            inject_failure("after_release_commit")
            validate_repository_snapshot(repository).raise_for_errors()
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
                "new_release_version": committed.version,
                "transaction_id": prepared.transaction_id,
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
                "writes_performed": True,
            }
        except BaseException as original:
            errors = _restore_transaction_state(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=prepared.planned_versions,
                committed_release_version=committed.version if committed else "",
            )
            journal.update(
                {
                    "status": "rollback_failed" if errors else "rolled_back",
                    "original_error": f"{type(original).__name__}: {original}",
                    "rollback_errors": list(errors),
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            if errors:
                recovery = (
                    repository.root
                    / "recovery/us-celg-bmy-cvr"
                    / f"{prepared.transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "CELG/BMYRT rollback was incomplete; recovery marker blocks "
                    f"writes: {recovery}; errors={errors}."
                ) from original
            raise


def _read_only_plan(
    repository: LocalDatasetRepository,
    *,
    evidence_dir: Path,
    bundle_path: Path,
) -> dict[str, Any]:
    release, _ = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    terms = _load_merger_terms(repository, release)
    termination = _load_termination_evidence(evidence_dir)
    plan = call_plan(repository, release)
    bundle = read_bundle_cache(bundle_path)
    result: dict[str, Any] = {
        "mode": "plan",
        "status": "fetch_required" if bundle is None else "bundle_cached",
        "base_release_version": release.version,
        "completed_session": release.completed_session,
        "official_evidence": {
            "merger_url": terms.source_url,
            "merger_sha256": terms.source_hash,
            "termination_url": termination.source_url,
            "termination_sha256": termination.source_hash,
        },
        "eodhd_call_plan": plan,
        "bundle_path": str(bundle_path),
        "network_accessed": False,
        "eodhd_calls": 0,
        "r2_accessed": False,
        "writes_performed": False,
    }
    if bundle is not None:
        result.update(
            {
                "bundle_fetched_against_release": bundle.fetched_against_release,
                "bundle_matches_current_release": (
                    bundle.fetched_against_release == release.version
                ),
                "provider_symbol": f"{bundle.provider_code}.US",
                "price_rows": len(bundle.prices),
            }
        )
        if bundle.fetched_against_release != release.version:
            result["status"] = "stale_bundle_refetch_required"
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan, fetch, or atomically install the exact CELG/BMY/BMYRT lifecycle."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--official-exit-mark", action="store_true")
    mode.add_argument("--apply-official-exit-mark", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    if args.official_exit_mark:
        prepared = prepare_official_exit_repair(
            repository,
            evidence_dir=args.evidence_dir,
        )
        result = {
            **prepared.summary,
            "mode": "official_exit_mark_plan",
            "status": prepared.summary["status"],
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
            "writes_performed": False,
        }
    elif args.apply_official_exit_mark:
        prepared = prepare_official_exit_repair(
            repository,
            evidence_dir=args.evidence_dir,
        )
        result = apply_repair(repository, prepared)
        result["mode"] = "apply_official_exit_mark"
    elif args.fetch:
        release, _ = repository.current_release()
        if release is None:
            raise RuntimeError("A current local release is required.")
        _load_merger_terms(repository, release)
        _load_termination_evidence(args.evidence_dir)
        catalog_selection = _frozen_catalog_selection(repository, release)
        cached = read_bundle_cache(args.bundle_path)
        if cached is not None and cached.fetched_against_release == release.version:
            result = {
                "mode": "fetch",
                "status": "already_fetched",
                "base_release_version": release.version,
                **validate_bundle(cached),
                "network_accessed": False,
                "eodhd_calls": 0,
                "diagnostic_budget_note": dict(DIAGNOSTIC_BUDGET_NOTE),
                "r2_accessed": False,
                "writes_performed": False,
            }
        else:
            client = ExactThreeEodhdClient()
            before = _budget_used(client.budget)
            bundle = fetch_exact_bundle(
                client,
                catalog_selection=catalog_selection,
                release_version=release.version,
                budget_used_before=before,
            )
            write_bundle_cache(args.bundle_path, bundle)
            result = {
                "mode": "fetch",
                "status": "fetched",
                "base_release_version": release.version,
                **validate_bundle(bundle),
                "network_accessed": True,
                "eodhd_calls": bundle.http_attempts,
                "diagnostic_budget_note": dict(DIAGNOSTIC_BUDGET_NOTE),
                "r2_accessed": False,
                "bundle_path": str(args.bundle_path),
                "writes_performed": True,
            }
    elif args.apply:
        bundle = read_bundle_cache(args.bundle_path)
        if bundle is None:
            raise RuntimeError("BMYRT bundle cache is missing; run --fetch first.")
        prepared = prepare_repair(
            repository, bundle, evidence_dir=args.evidence_dir
        )
        result = apply_repair(repository, prepared)
    else:
        result = _read_only_plan(
            repository,
            evidence_dir=args.evidence_dir,
            bundle_path=args.bundle_path,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
