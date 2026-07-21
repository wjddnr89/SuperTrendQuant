#!/usr/bin/env python3
"""Collect and plan the exact legacy-DuPont -> Chemours repair.

The transaction is deliberately narrow and fail-closed:

* the frozen EODHD US exchange list supplies the CC identity with zero calls;
* network collection is capped at four one-attempt EODHD requests;
* the fourth request independently verifies all eleven ordinary legacy-DD
  dividends before the spin-off can enter the same candidate;
* three hash-pinned SEC/issuer documents establish the 1-for-5 distribution,
  first regular-way session, valuation, and parent/child basis allocation;
* the 2015-07-01 WIKI ``3.2`` value is never represented as cash;
* all 672 existing legacy-DD raw OHLCV rows must remain byte-for-byte equal;
* default and ``--offline-plan`` modes do not advance release pointers;
* ``--apply`` exists for a later reviewed decision and is never implied by
  collection or planning; the script has no R2 code path.

The exact provider requests are:

1. ``eod/CC.US`` for 2015-07-01..2026-07-15;
2. ``div/CC.US`` for the same range;
3. ``splits/CC.US`` for the same range; and
4. ``div/DD_old.US`` for 2015-01-01..2017-09-01.

Every failed attempt consumes its budget claim and there are no retries.  API
tokens are passed only in the HTTP request and are absent from every persisted
URL, exception, report, and source-archive row.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import html
import io
import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from supertrend_quant.env import load_env
from supertrend_quant.indicators import add_triple_supertrend
from supertrend_quant.ledger import PortfolioLedger
from supertrend_quant.market_store.adjustments import (
    SPINOFF_PRICE_ADJUSTMENT_CONTRACT,
    apply_adjustment_factors,
    build_adjustment_factors,
)
from supertrend_quant.market_store.ingest import (
    EodhdCallBudget,
    EodhdClient,
    SourceArtifact,
)
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
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
from supertrend_quant.portfolio import Position


DEFAULT_CACHE_ROOT = Path("data/cache")
EXPECTED_BASE_COMPLETED_SESSION = "2026-07-15"
CC_FETCH_START = "2015-07-01"
CC_FETCH_END = EXPECTED_BASE_COMPLETED_SESSION
DD_DIV_FETCH_START = "2015-01-01"
DD_DIV_FETCH_END = "2017-09-01"

LEGACY_DD_SECURITY_ID = "US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1"
CC_SECURITY_ID = "US:EODHD:a710ce0e-a20e-5558-ba2b-f2719e2981cf"
CC_SYMBOL = "CC"
CC_PROVIDER_SYMBOL = "CC.US"
DD_PROVIDER_SYMBOL = "DD_old.US"
SPINOFF_DATE = "2015-07-01"
SPINOFF_RECORD_DATE = "2015-06-23"
SPINOFF_RATIO = 1.0 / 5.0
SPINOFF_EVENT_ID = canonical_lifecycle_event_id(
    LEGACY_DD_SECURITY_ID, "spinoff", SPINOFF_DATE
)

CATALOG_URL = "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
CATALOG_SHA256 = (
    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
)
EXPECTED_CATALOG_ROW = {
    "Code": "CC",
    "Country": "USA",
    "Currency": "USD",
    "Exchange": "NYSE",
    "Isin": "US1638511089",
    "Name": "Chemours Co",
    "Type": "Common Stock",
}

LEGACY_DD_RAW_SOURCE_SHA256 = (
    "36bc4a610a64882f576cecff4f73fb9022e19083664e8c3383d75b25e40bc77a"
)
EXPECTED_LEGACY_DD_PRICE_ROWS = 672
EXPECTED_BUDGET_USED_BEFORE = 8853
EXPECTED_BUDGET_USED_AFTER = 8857
MAX_EODHD_ATTEMPTS = 4


@dataclass(frozen=True)
class OfficialEvidenceSpec:
    key: str
    url: str
    sha256: str
    size: int
    content_type: str
    filename: str
    required_text: tuple[str, ...] = ()


OFFICIAL_EVIDENCE_SPECS = (
    OfficialEvidenceSpec(
        key="sec_distribution_terms",
        url=(
            "https://www.sec.gov/Archives/edgar/data/1627223/"
            "000119312515215110/d832629dex991.htm"
        ),
        sha256=(
            "46a4febfc3e55eb86798f94bc537bad081a4590c71f208c3c50c14df133d647d"
        ),
        size=2_365_044,
        content_type="text/html",
        filename="dd_chemours_terms.htm",
        required_text=(
            "for every five shares of dupont common stock",
            "you will receive one share of chemours common stock",
            "july 1 2015 prior to the opening of trading",
            "under the symbol cc",
        ),
    ),
    OfficialEvidenceSpec(
        key="sec_regular_way_confirmation",
        url=(
            "https://www.sec.gov/Archives/edgar/data/1627223/"
            "000162722315000023/cc-2015630x10q.htm"
        ),
        sha256=(
            "3462a65e3695aff0b481578568ed89206429fe46d1fabbb44ff60d23812176ac"
        ),
        size=1_429_547,
        content_type="text/html",
        filename="dd_chemours_10q.htm",
        required_text=(
            "common stock began regular way trading on the nyse on july 1 2015",
            "under the symbol cc",
        ),
    ),
    OfficialEvidenceSpec(
        key="issuer_basis_allocation",
        url=(
            "https://s23.q4cdn.com/116192123/files/doc_downloads/"
            "Tax-Cost-Basis-Allocation.pdf"
        ),
        sha256=(
            "dea8c3a9852d484a59052d03c61c7dfb61c702ce99510c0d8c4cbc45cc20ad88"
        ),
        size=703_430,
        content_type="application/pdf",
        filename="dd_chemours_basis.pdf",
    ),
)

TERMS_SPEC = OFFICIAL_EVIDENCE_SPECS[0]
CONFIRMATION_SPEC = OFFICIAL_EVIDENCE_SPECS[1]
BASIS_SPEC = OFFICIAL_EVIDENCE_SPECS[2]

SPINOFF_METADATA_TEMPLATE: dict[str, Any] = {
    "allow_fractional": False,
    "basis_source_hash": BASIS_SPEC.sha256,
    "basis_source_url": BASIS_SPEC.url,
    "child_fair_market_value_per_share": 16.21,
    "cost_basis_fraction": 0.05085,
    "distribution_ratio": SPINOFF_RATIO,
    "distributed_value_per_parent_share": 3.242,
    "fractional_settlement_status": "cash_in_lieu_price_not_source_pinned",
    "parent_cost_basis_fraction": 0.94915,
    "parent_fair_market_value_per_share": 60.51,
    "price_adjustment_contract": SPINOFF_PRICE_ADJUSTMENT_CONTRACT,
    "terms_source_hash": TERMS_SPEC.sha256,
    "terms_source_url": TERMS_SPEC.url,
    "confirmation_source_hash": CONFIRMATION_SPEC.sha256,
    "confirmation_source_url": CONFIRMATION_SPEC.url,
    "valuation_date": "2015-07-02",
    "valuation_method": "issuer_average_high_low",
}

EXPECTED_DD_ORDINARY_DIVIDENDS = {
    "2015-02-11": 0.47,
    "2015-05-13": 0.49,
    "2015-08-12": 0.38,
    "2015-11-10": 0.38,
    "2016-02-10": 0.38,
    "2016-05-11": 0.38,
    "2016-08-11": 0.38,
    "2016-11-10": 0.38,
    "2017-02-13": 0.38,
    "2017-05-11": 0.38,
    "2017-07-27": 0.38,
}
OPTIONAL_PROVIDER_SPIN_PROXY = {SPINOFF_DATE: 3.2}

REQUEST_SPECS = (
    ("cc_eod", "eod", CC_PROVIDER_SYMBOL, CC_FETCH_START, CC_FETCH_END),
    ("cc_div", "div", CC_PROVIDER_SYMBOL, CC_FETCH_START, CC_FETCH_END),
    ("cc_splits", "splits", CC_PROVIDER_SYMBOL, CC_FETCH_START, CC_FETCH_END),
    (
        "dd_div",
        "div",
        DD_PROVIDER_SYMBOL,
        DD_DIV_FETCH_START,
        DD_DIV_FETCH_END,
    ),
)


def _request_url(endpoint: str, symbol: str, start: str, end: str) -> str:
    return f"https://eodhd.com/api/{endpoint}/{symbol}?from={start}&to={end}"


REQUEST_URLS = {
    key: _request_url(endpoint, symbol, start, end)
    for key, endpoint, symbol, start, end in REQUEST_SPECS
}

WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "source_archive",
)
REPORT_SOURCE_URL = "supertrendquant://reviewed/us-dd-chemours/v1"
TRANSACTION_DIR = "transactions/us-dd-chemours"
RECOVERY_DIR = "recovery/us-dd-chemours"
OPERATION = "repair_us_dd_chemours"


@dataclass(frozen=True)
class FrozenCatalogSelection:
    row: Mapping[str, Any]
    source_url: str
    retrieved_at: str
    source_hash: str
    object_path: str


@dataclass(frozen=True)
class EvidenceBundle:
    official_artifacts: tuple[SourceArtifact, ...]
    provider_artifacts: tuple[SourceArtifact, ...]
    cc_prices: pd.DataFrame
    cc_actions: pd.DataFrame
    dd_dividend_actions: pd.DataFrame
    eodhd_http_attempts: int
    official_http_attempts: int
    budget_used_before: int
    budget_used_after: int
    fetched_against_release: str


@dataclass
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    archive_artifacts: tuple[SourceArtifact, ...]
    summary: dict[str, Any]


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


def _date(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).date().isoformat()


def _normalized_markup(content: bytes) -> str:
    decoded = html.unescape(content.decode("utf-8", errors="replace"))
    decoded = re.sub(r"<[^>]+>", " ", decoded)
    decoded = re.sub(r"[^a-zA-Z0-9]+", " ", decoded).lower()
    return re.sub(r"\s+", " ", decoded).strip()


def _validate_official_artifact(
    spec: OfficialEvidenceSpec, artifact: SourceArtifact
) -> None:
    if (
        artifact.source_url != spec.url
        or artifact.source_hash != spec.sha256
        or len(artifact.content) != spec.size
        or artifact.content_type != spec.content_type
    ):
        raise ValueError(f"Official Chemours evidence changed: {spec.key}.")
    if spec.required_text:
        normalized = _normalized_markup(artifact.content)
        missing = [value for value in spec.required_text if value not in normalized]
        if missing:
            raise ValueError(
                f"Official Chemours evidence lacks reviewed terms: {spec.key}/{missing}."
            )


def _official_artifact(
    spec: OfficialEvidenceSpec, content: bytes, retrieved_at: str
) -> SourceArtifact:
    artifact = SourceArtifact(
        source=f"official_dd_chemours_{spec.key}",
        source_url=spec.url,
        retrieved_at=retrieved_at,
        content=content,
        content_type=spec.content_type,
    )
    _validate_official_artifact(spec, artifact)
    return artifact


def _load_official_from_directory(
    directory: Path, *, retrieved_at: str
) -> tuple[SourceArtifact, ...]:
    artifacts = []
    for spec in OFFICIAL_EVIDENCE_SPECS:
        path = directory / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"Official seed file is missing: {path}")
        artifacts.append(_official_artifact(spec, path.read_bytes(), retrieved_at))
    return tuple(artifacts)


def _fetch_official_artifacts(
    session: Any,
    *,
    user_agent: str,
    retrieved_at: str,
) -> tuple[tuple[SourceArtifact, ...], int]:
    if not user_agent:
        raise RuntimeError("SEC_USER_AGENT is required for exact official collection.")
    artifacts: list[SourceArtifact] = []
    attempts = 0
    for spec in OFFICIAL_EVIDENCE_SPECS:
        attempts += 1
        try:
            response = session.get(
                spec.url,
                timeout=120,
                headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Official Chemours request failed: {spec.key}/{type(exc).__name__}."
            ) from None
        if int(getattr(response, "status_code", 0)) != 200:
            raise RuntimeError(
                f"Official Chemours request failed: {spec.key}/HTTP "
                f"{getattr(response, 'status_code', 'unknown')}."
            )
        artifacts.append(_official_artifact(spec, bytes(response.content), retrieved_at))
    return tuple(artifacts), attempts


def _parse_json_rows(artifact: SourceArtifact, label: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(artifact.content)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"{label} response is not JSON.") from exc
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{label} response must be a JSON row list.")
    return value


def _provider_artifact(
    key: str, content: bytes, retrieved_at: str
) -> SourceArtifact:
    return SourceArtifact(
        source=f"eodhd_{key}",
        source_url=REQUEST_URLS[key],
        retrieved_at=retrieved_at,
        content=content,
        content_type="application/json",
    )


def _parse_split_ratio(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            ratio = float(numerator) / denominator_value if denominator_value else math.nan
        else:
            ratio = float(text)
    except (TypeError, ValueError):
        return None
    return ratio if math.isfinite(ratio) and ratio > 0 else None


def _provider_event_id(
    source: str, security_id: str, action_type: str, date: str
) -> str:
    return hashlib.sha256(
        f"{source}|{security_id}|{action_type}|{date}".encode()
    ).hexdigest()


def _provider_action(
    *,
    artifact: SourceArtifact,
    security_id: str,
    action_type: str,
    effective_date: str,
    cash_amount: Any = None,
    ratio: Any = None,
    announcement_date: str = "",
    record_date: str = "",
    payment_date: str = "",
    currency: str = "USD",
) -> dict[str, Any]:
    return {
        "event_id": _provider_event_id(
            artifact.source, security_id, action_type, effective_date
        ),
        "security_id": security_id,
        "action_type": action_type,
        "effective_date": effective_date,
        "ex_date": effective_date,
        "announcement_date": announcement_date,
        "record_date": record_date,
        "payment_date": payment_date,
        "cash_amount": cash_amount,
        "ratio": ratio,
        "currency": currency or "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_url": artifact.source_url,
        "source_kind": "provider",
        "source": artifact.source,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
        "metadata": "",
    }


def _dividend_amount(row: Mapping[str, Any]) -> float:
    raw = row.get("unadjustedValue")
    if raw is None or _text(raw) == "":
        raw = row.get("value")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("EODHD dividend lacks a numeric unadjusted value.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("EODHD dividend value must be finite and positive.")
    return value


def _expected_cc_sessions() -> tuple[str, ...]:
    sessions = xcals.get_calendar("XNYS").sessions_in_range(
        CC_FETCH_START, CC_FETCH_END
    )
    return tuple(pd.Timestamp(value).date().isoformat() for value in sessions)


def _prices_from_cc_eod(artifact: SourceArtifact) -> pd.DataFrame:
    rows = _parse_json_rows(artifact, "CC eod")
    output = pd.DataFrame(
        [
            {
                "security_id": CC_SECURITY_ID,
                "session": _date(row.get("date")),
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
            for row in rows
            if _date(row.get("date")) and row.get("close") is not None
        ]
    )
    if output.empty or output.duplicated(["security_id", "session"]).any():
        raise ValueError("CC EOD is empty or has duplicate sessions.")
    numeric = output[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("CC EOD has non-finite OHLCV values.")
    coherent = (
        numeric[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & numeric["volume"].ge(0)
        & numeric["high"].ge(numeric[["open", "low", "close"]].max(axis=1))
        & numeric["low"].le(numeric[["open", "high", "close"]].min(axis=1))
    )
    if not bool(coherent.all()):
        raise ValueError("CC EOD violates OHLCV coherence.")
    observed = tuple(sorted(output["session"].astype(str)))
    expected = _expected_cc_sessions()
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(
            "CC EOD session coverage is not exact: "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )
    return output.sort_values("session", ignore_index=True)


def _cc_actions(
    dividend_artifact: SourceArtifact, split_artifact: SourceArtifact
) -> pd.DataFrame:
    actions: list[dict[str, Any]] = []
    for row in _parse_json_rows(dividend_artifact, "CC div"):
        effective = _date(row.get("date"))
        if not effective:
            raise ValueError("CC dividend has no exact ex-date.")
        if not CC_FETCH_START <= effective <= CC_FETCH_END:
            raise ValueError("CC dividend escapes the bounded request.")
        currency = _text(row.get("currency")) or "USD"
        if currency.upper() != "USD":
            raise ValueError("CC dividend currency is not USD.")
        actions.append(
            _provider_action(
                artifact=dividend_artifact,
                security_id=CC_SECURITY_ID,
                action_type="cash_dividend",
                effective_date=effective,
                cash_amount=_dividend_amount(row),
                announcement_date=_date(row.get("declarationDate")),
                record_date=_date(row.get("recordDate")),
                payment_date=_date(row.get("paymentDate")),
                currency="USD",
            )
        )
    for row in _parse_json_rows(split_artifact, "CC splits"):
        effective = _date(row.get("date"))
        ratio = _parse_split_ratio(row.get("split"))
        if not effective or ratio is None:
            raise ValueError("CC split lacks an exact date or ratio.")
        if not CC_FETCH_START <= effective <= CC_FETCH_END:
            raise ValueError("CC split escapes the bounded request.")
        actions.append(
            _provider_action(
                artifact=split_artifact,
                security_id=CC_SECURITY_ID,
                action_type="split",
                effective_date=effective,
                ratio=ratio,
            )
        )
    frame = pd.DataFrame(actions)
    if not frame.empty and frame.duplicated("event_id").any():
        raise ValueError("CC provider actions contain duplicate event IDs.")
    return frame


def _dd_dividend_actions(artifact: SourceArtifact) -> pd.DataFrame:
    rows = _parse_json_rows(artifact, "legacy DD div")
    observed: dict[str, float] = {}
    row_by_date: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        effective = _date(row.get("date"))
        if not effective:
            raise ValueError("Legacy DD dividend has no exact ex-date.")
        if effective in observed:
            raise ValueError(f"Legacy DD dividend date is duplicated: {effective}.")
        observed[effective] = _dividend_amount(row)
        row_by_date[effective] = row
    allowed = set(EXPECTED_DD_ORDINARY_DIVIDENDS) | set(OPTIONAL_PROVIDER_SPIN_PROXY)
    if set(observed) - allowed:
        raise ValueError(
            "Legacy DD dividend response contains unexpected dates: "
            f"{sorted(set(observed) - allowed)}."
        )
    for date, expected in EXPECTED_DD_ORDINARY_DIVIDENDS.items():
        if date not in observed or not math.isclose(
            observed[date], expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "Legacy DD ordinary dividends do not independently match WIKI: "
                f"{date}/expected={expected}/observed={observed.get(date)}."
            )
    if SPINOFF_DATE in observed and not math.isclose(
        observed[SPINOFF_DATE],
        OPTIONAL_PROVIDER_SPIN_PROXY[SPINOFF_DATE],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Legacy DD 2015-07-01 provider row is not the 3.2 spin proxy.")

    actions: list[dict[str, Any]] = []
    for date, amount in EXPECTED_DD_ORDINARY_DIVIDENDS.items():
        row = row_by_date[date]
        currency = _text(row.get("currency")) or "USD"
        if currency.upper() != "USD":
            raise ValueError("Legacy DD ordinary dividend currency is not USD.")
        actions.append(
            _provider_action(
                artifact=artifact,
                security_id=LEGACY_DD_SECURITY_ID,
                action_type="cash_dividend",
                effective_date=date,
                cash_amount=amount,
                announcement_date=_date(row.get("declarationDate")),
                record_date=_date(row.get("recordDate")),
                payment_date=_date(row.get("paymentDate")),
                currency="USD",
            )
        )
    return pd.DataFrame(actions)


def _bundle_from_artifacts(
    official_artifacts: Iterable[SourceArtifact],
    provider_artifacts: Iterable[SourceArtifact],
    *,
    eodhd_http_attempts: int,
    official_http_attempts: int,
    budget_used_before: int,
    budget_used_after: int,
    fetched_against_release: str,
) -> EvidenceBundle:
    official = tuple(official_artifacts)
    by_official = {artifact.source_url: artifact for artifact in official}
    if len(by_official) != len(OFFICIAL_EVIDENCE_SPECS):
        raise ValueError("Official Chemours evidence is incomplete or duplicated.")
    for spec in OFFICIAL_EVIDENCE_SPECS:
        artifact = by_official.get(spec.url)
        if artifact is None:
            raise ValueError(f"Official Chemours evidence is absent: {spec.key}.")
        _validate_official_artifact(spec, artifact)

    provider = tuple(provider_artifacts)
    by_key = {artifact.source.removeprefix("eodhd_"): artifact for artifact in provider}
    if set(by_key) != {item[0] for item in REQUEST_SPECS} or len(provider) != 4:
        raise ValueError("DD/CC EODHD artifact set is not exactly four responses.")
    for key, artifact in by_key.items():
        if artifact.source_url != REQUEST_URLS[key]:
            raise ValueError(f"EODHD artifact URL is not exact: {key}.")
        _parse_json_rows(artifact, key)

    bundle = EvidenceBundle(
        official_artifacts=tuple(by_official[spec.url] for spec in OFFICIAL_EVIDENCE_SPECS),
        provider_artifacts=tuple(by_key[key] for key, *_ in REQUEST_SPECS),
        cc_prices=_prices_from_cc_eod(by_key["cc_eod"]),
        cc_actions=_cc_actions(by_key["cc_div"], by_key["cc_splits"]),
        dd_dividend_actions=_dd_dividend_actions(by_key["dd_div"]),
        eodhd_http_attempts=int(eodhd_http_attempts),
        official_http_attempts=int(official_http_attempts),
        budget_used_before=int(budget_used_before),
        budget_used_after=int(budget_used_after),
        fetched_against_release=fetched_against_release,
    )
    validate_evidence_bundle(bundle)
    return bundle


def validate_evidence_bundle(bundle: EvidenceBundle) -> dict[str, Any]:
    if bundle.eodhd_http_attempts != MAX_EODHD_ATTEMPTS:
        raise ValueError("DD/CC bundle must originate from exactly four EODHD attempts.")
    if (
        bundle.budget_used_before != EXPECTED_BUDGET_USED_BEFORE
        or bundle.budget_used_after != EXPECTED_BUDGET_USED_AFTER
        or bundle.budget_used_after - bundle.budget_used_before != MAX_EODHD_ATTEMPTS
    ):
        raise ValueError("DD/CC bundle budget pins are not exactly 8853 -> 8857.")
    if len(bundle.dd_dividend_actions) != len(EXPECTED_DD_ORDINARY_DIVIDENDS):
        raise ValueError("Legacy DD ordinary-dividend action count is not eleven.")
    if bundle.dd_dividend_actions["effective_date"].astype(str).eq(SPINOFF_DATE).any():
        raise ValueError("The 3.2 spin proxy was misclassified as cash.")
    if len(bundle.cc_prices) != len(_expected_cc_sessions()):
        raise ValueError("CC price inventory is not exact.")
    return {
        "eodhd_http_attempts": bundle.eodhd_http_attempts,
        "official_http_attempts": bundle.official_http_attempts,
        "budget_used_before": bundle.budget_used_before,
        "budget_used_after": bundle.budget_used_after,
        "cc_price_rows": len(bundle.cc_prices),
        "cc_first_session": str(bundle.cc_prices["session"].min()),
        "cc_last_session": str(bundle.cc_prices["session"].max()),
        "cc_dividend_rows": int(
            bundle.cc_actions.get("action_type", pd.Series(dtype=str))
            .astype(str)
            .eq("cash_dividend")
            .sum()
        ),
        "cc_split_rows": int(
            bundle.cc_actions.get("action_type", pd.Series(dtype=str))
            .astype(str)
            .eq("split")
            .sum()
        ),
        "dd_ordinary_dividend_rows": len(bundle.dd_dividend_actions),
        "dd_spin_proxy_booked_as_cash": False,
    }


class ExactFourEodhdClient(EodhdClient):
    """No-retry client restricted to the four reviewed DD/CC requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempted_keys: list[str] = []

    def fetch_artifact(self, key: str, *, retrieved_at: str) -> SourceArtifact:
        position = len(self.attempted_keys)
        if position >= MAX_EODHD_ATTEMPTS:
            raise RuntimeError("DD/CC client refused a fifth EODHD request.")
        expected_key, endpoint, symbol, start, end = REQUEST_SPECS[position]
        if key != expected_key:
            raise RuntimeError(
                "DD/CC client refused an out-of-order request: "
                f"expected={expected_key}, observed={key}."
            )
        self.budget.claim()
        self.attempted_keys.append(key)
        try:
            response = self.session.get(
                f"{self.base_url}/{endpoint}/{symbol}",
                params={
                    "from": start,
                    "to": end,
                    "api_token": self.token,
                    "fmt": "json",
                },
                timeout=120,
            )
        except Exception as exc:
            raise RuntimeError(
                f"EODHD exact request failed: {key}/{type(exc).__name__}."
            ) from None
        status = int(getattr(response, "status_code", 0))
        if status != 200:
            raise RuntimeError(f"EODHD exact request failed: {key}/HTTP {status}.")
        content = bytes(response.content)
        artifact = _provider_artifact(key, content, retrieved_at)
        _parse_json_rows(artifact, key)
        if self.token.encode() in content or "api_token" in artifact.source_url:
            raise RuntimeError(f"EODHD credential leaked into persisted evidence: {key}.")
        return artifact


def _budget_used(budget: EodhdCallBudget) -> int:
    used = int(budget.seed_used)
    if budget.state_path.is_file():
        try:
            value = json.loads(budget.state_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError("EODHD budget state is unreadable.") from exc
        if _text(value.get("period")) == budget.period:
            used = max(used, int(value.get("used", 0)))
    return used


def _conservative_existing_budget() -> EodhdCallBudget:
    """Continue the existing 8853 ledger even after the UTC date boundary.

    The provider may have reset its daily usage, but continuing the prior local
    period is deliberately more conservative and makes the requested
    8853 -> 8857 transition auditable.
    """

    load_env()
    state_path = Path(
        os.getenv(
            "EODHD_API_CALL_STATE_FILE",
            "data/cache/state/eodhd_call_budget.json",
        )
    )
    if not state_path.is_file():
        raise RuntimeError("Existing EODHD budget ledger is missing; collection refused.")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("Existing EODHD budget ledger is unreadable.") from exc
    if int(state.get("used", -1)) != EXPECTED_BUDGET_USED_BEFORE:
        raise RuntimeError(
            "EODHD budget changed before DD/CC collection: "
            f"expected={EXPECTED_BUDGET_USED_BEFORE}, "
            f"observed={state.get('used')}."
        )
    return EodhdCallBudget(
        state_path=state_path,
        limit=int(state.get("daily_limit", 100000)),
        reserve=int(state.get("reserve", 0)),
        seed_used=EXPECTED_BUDGET_USED_BEFORE,
        period=_text(state.get("period")),
    )


def fetch_exact_provider_artifacts(
    client: ExactFourEodhdClient,
    *,
    retrieved_at: str,
) -> tuple[SourceArtifact, ...]:
    artifacts: list[SourceArtifact] = []
    for key, *_ in REQUEST_SPECS:
        artifact = client.fetch_artifact(key, retrieved_at=retrieved_at)
        artifacts.append(artifact)
        if key == "cc_eod":
            # Spend no action calls unless the one irreplaceable price request
            # first proves exact, complete CC coverage.
            _prices_from_cc_eod(artifact)
        elif key == "dd_div":
            _dd_dividend_actions(artifact)
    if tuple(client.attempted_keys) != tuple(item[0] for item in REQUEST_SPECS):
        raise RuntimeError("DD/CC EODHD request inventory is not exactly four.")
    return tuple(artifacts)


def _bundle_signature(catalog: FrozenCatalogSelection) -> dict[str, Any]:
    return {
        "schema": "us_dd_chemours_evidence_bundle/v1",
        "base_completed_session": EXPECTED_BASE_COMPLETED_SESSION,
        "catalog_sha256": catalog.source_hash,
        "cc_security_id": CC_SECURITY_ID,
        "legacy_dd_security_id": LEGACY_DD_SECURITY_ID,
        "request_urls": REQUEST_URLS,
        "official_sha256": {
            spec.key: spec.sha256 for spec in OFFICIAL_EVIDENCE_SPECS
        },
        "expected_eodhd_attempts": MAX_EODHD_ATTEMPTS,
        "expected_budget_used_before": EXPECTED_BUDGET_USED_BEFORE,
        "expected_budget_used_after": EXPECTED_BUDGET_USED_AFTER,
    }


def _bundle_cache_path(
    cache_root: Path, catalog: FrozenCatalogSelection
) -> Path:
    digest = sha256_bytes(_canonical_json_bytes(_bundle_signature(catalog)))
    return cache_root / "state/us_dd_chemours" / f"{digest}.json.gz"


def _artifact_record(artifact: SourceArtifact) -> dict[str, Any]:
    return {
        "source": artifact.source,
        "source_url": artifact.source_url,
        "retrieved_at": artifact.retrieved_at,
        "content_type": artifact.content_type,
        "content_sha256": artifact.source_hash,
        "content_base64": base64.b64encode(artifact.content).decode("ascii"),
    }


def _write_bundle_cache(
    path: Path,
    catalog: FrozenCatalogSelection,
    bundle: EvidenceBundle,
) -> None:
    validate_evidence_bundle(bundle)
    payload = {
        **_bundle_signature(catalog),
        "fetched_against_release": bundle.fetched_against_release,
        "eodhd_http_attempts": bundle.eodhd_http_attempts,
        "official_http_attempts": bundle.official_http_attempts,
        "budget_used_before": bundle.budget_used_before,
        "budget_used_after": bundle.budget_used_after,
        "official_artifacts": [
            _artifact_record(item) for item in bundle.official_artifacts
        ],
        "provider_artifacts": [
            _artifact_record(item) for item in bundle.provider_artifacts
        ],
    }
    payload_bytes = _canonical_json_bytes(payload)
    if b"api_token" in payload_bytes:
        raise RuntimeError("DD/CC cache payload contains a credential parameter.")
    wrapper = _canonical_json_bytes(
        {
            "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
            "payload_sha256": sha256_bytes(payload_bytes),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, gzip.compress(wrapper, mtime=0))


def _decode_artifact_records(
    values: Iterable[Mapping[str, Any]], label: str
) -> tuple[SourceArtifact, ...]:
    artifacts: list[SourceArtifact] = []
    for value in values:
        try:
            content = base64.b64decode(value["content_base64"], validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cached {label} artifact is unreadable.") from exc
        if sha256_bytes(content) != _text(value.get("content_sha256")):
            raise ValueError(f"Cached {label} artifact hash changed.")
        artifacts.append(
            SourceArtifact(
                source=_text(value.get("source")),
                source_url=_text(value.get("source_url")),
                retrieved_at=_text(value.get("retrieved_at")),
                content=content,
                content_type=_text(value.get("content_type")),
            )
        )
    return tuple(artifacts)


def _read_bundle_cache(
    path: Path, catalog: FrozenCatalogSelection
) -> EvidenceBundle | None:
    if not path.is_file():
        return None
    try:
        wrapper = json.loads(gzip.decompress(path.read_bytes()))
        payload_bytes = base64.b64decode(wrapper["payload_base64"], validate=True)
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"DD/CC evidence bundle is unreadable: {path}") from exc
    if sha256_bytes(payload_bytes) != _text(wrapper.get("payload_sha256")):
        raise ValueError("DD/CC evidence bundle wrapper hash changed.")
    if b"api_token" in payload_bytes:
        raise ValueError("DD/CC evidence bundle contains a credential parameter.")
    payload = json.loads(payload_bytes)
    for key, expected in _bundle_signature(catalog).items():
        if payload.get(key) != expected:
            raise ValueError(f"DD/CC evidence bundle signature changed: {key}.")
    return _bundle_from_artifacts(
        _decode_artifact_records(payload.get("official_artifacts", ()), "official"),
        _decode_artifact_records(payload.get("provider_artifacts", ()), "provider"),
        eodhd_http_attempts=int(payload.get("eodhd_http_attempts", -1)),
        official_http_attempts=int(payload.get("official_http_attempts", -1)),
        budget_used_before=int(payload.get("budget_used_before", -1)),
        budget_used_after=int(payload.get("budget_used_after", -1)),
        fetched_against_release=_text(payload.get("fetched_against_release")),
    )


def _safe_archived_content(
    repository: LocalDatasetRepository, object_path: str
) -> bytes:
    root = repository.root.resolve()
    path = (repository.root / object_path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Archived object escapes cache root: {object_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Archived object is missing: {path}")
    try:
        return gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError(f"Archived object is not valid gzip: {path}") from exc


def load_frozen_catalog(
    repository: LocalDatasetRepository, release: DataRelease
) -> FrozenCatalogSelection:
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    matches = archive.loc[
        archive["dataset"].astype(str).eq("eodhd_exchange_symbols")
        & archive["source_url"].astype(str).eq(CATALOG_URL)
        & archive["source_hash"].astype(str).eq(CATALOG_SHA256)
    ]
    if len(matches) != 1:
        raise ValueError("Frozen EODHD active catalog is not uniquely pinned.")
    row = matches.iloc[0]
    content = _safe_archived_content(repository, _text(row.get("object_path")))
    if sha256_bytes(content) != CATALOG_SHA256:
        raise ValueError("Frozen EODHD active catalog payload hash changed.")
    payload = json.loads(content)
    candidates = [
        item for item in payload if _text(item.get("Code")).upper() == CC_SYMBOL
    ]
    if len(candidates) != 1 or dict(candidates[0]) != EXPECTED_CATALOG_ROW:
        raise ValueError("Frozen EODHD catalog lacks the exact reviewed CC row.")
    derived = f"US:EODHD:{uuid.uuid5(uuid.NAMESPACE_URL, 'eodhd:US:CC:symbol:CC')}"
    if derived != CC_SECURITY_ID:
        raise RuntimeError("Pinned CC security-id derivation changed.")
    return FrozenCatalogSelection(
        row=dict(candidates[0]),
        source_url=CATALOG_URL,
        retrieved_at=_text(row.get("retrieved_at")),
        source_hash=CATALOG_SHA256,
        object_path=_text(row.get("object_path")),
    )
