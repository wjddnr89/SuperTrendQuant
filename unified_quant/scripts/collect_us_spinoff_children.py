#!/usr/bin/env python3
"""Collect and atomically install the audited FBIN -> MBC spin-off child.

Safety properties are deliberately narrow:

* the default command is a read-only, fail-closed plan;
* only ``--fetch-missing`` may make the three exact EODHD requests;
* ``--offline-plan`` and ``--apply`` replay a hash-checked local bundle;
* only ``--apply`` advances dataset/release pointers, under CAS + journal;
* the frozen EODHD US exchange list supplies identity metadata (zero catalog calls);
* the transaction is rejected until the FBHS/FBIN identity merge is complete;
* the issuer Form 8937 is hash-pinned and supplies canonical cost-basis metadata;
* this script has no R2 code path.

The three provider requests are exactly ``eod``, ``div``, and ``splits`` for
``MBC.US`` over 2022-12-15..2026-07-15.  Network collection and dataset apply
must be separate invocations.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
from html import unescape as html_unescape
import json
import math
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import exchange_calendars as xcals
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
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


DEFAULT_CACHE_ROOT = Path("data/cache")
FETCH_START = "2022-12-15"
FETCH_END = "2026-07-15"
PROVIDER_SYMBOL = "MBC.US"
MBC_SYMBOL = "MBC"
MBC_SECURITY_ID = (
    "US:EODHD:3f53d1d0-8990-5ec5-bb4b-4b72946ee66e"
)
FBIN_SECURITY_ID = (
    "US:EODHD:89fe6d28-737c-5b16-82e6-c1207561311c"
)
RETIRED_FBHS_SECURITY_ID = (
    "US:EODHD:724457bc-0eaf-5959-8c93-f0c2a03c80de"
)
FBHS_SYMBOL = "FBHS"
FBIN_SYMBOL = "FBIN"
FBHS_HISTORY_START = "2015-01-01"
FBHS_HISTORY_END = "2022-12-14"
FBIN_HISTORY_START = "2022-12-15"

CATALOG_URL = "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
CATALOG_SHA256 = (
    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
)
EXPECTED_CATALOG_ROW = {
    "Code": "MBC",
    "Country": "USA",
    "Currency": "USD",
    "Exchange": "NYSE",
    "Isin": "US57638P1049",
    "Name": "MasterBrand Inc.",
    "Type": "Common Stock",
}

SEC_URL = (
    "https://www.sec.gov/Archives/edgar/data/1519751/"
    "000119312522306146/0001193125-22-306146.txt"
)
SEC_SHA256 = (
    "2c2703ed8949f1d72ceea49e655005cd39165a8020b1750c71894d185d987135"
)
SEC_EXACT_BYTES = 1_430_923
SEC_RETRIEVED_AT = "2026-07-17T20:04:15.373205Z"
SEC_CACHE_KEY = sha256_bytes(f"{SEC_URL}?".encode())
SEC_REQUIRED_TEXT = (
    "one share of masterbrand common stock for every one share",
    "distribution of masterbrand shares was completed at 5:00 p.m.",
    "wednesday, december 14, 2022",
    "regular way",
    "symbol \u201cmbc\u201d",
)

FORM_8937_URL = (
    "https://ir.fbin.com/static-files/"
    "a2cbb4cc-e288-4caa-9114-89605ab89b6d"
)
FORM_8937_SHA256 = (
    "c44a12b1185910921d69549d7d8a67bd055e48844ed9abd0cc6879d50afcc96c"
)
FORM_8937_EXACT_BYTES = 9_681_136
FORM_8937_RETRIEVED_AT = "2026-07-18T06:32:19Z"
FORM_8937_CACHE_RELATIVE_PATH = Path(
    "state/issuer_lifecycle/"
    f"{FORM_8937_SHA256}.pdf"
)

SPINOFF_EFFECTIVE_DATE = "2022-12-15"
SPINOFF_ANNOUNCEMENT_DATE = "2022-12-01"
SPINOFF_RECORD_DATE = "2022-12-02"
SPINOFF_PAYMENT_DATE = "2022-12-14"
SPINOFF_RATIO = 1.0
SPINOFF_COST_BASIS_METADATA = {
    "cost_basis_fraction": 0.123028,
    "currency": "USD",
    "method": "relative_fair_market_value_based_on_vwap",
    "source_hash": FORM_8937_SHA256,
    "source_url": FORM_8937_URL,
    "valuation_date": "2022-12-15",
    "vwaps": {"FBIN": 55.9795, "MBC": 7.8532},
}
SPINOFF_EVENT_ID = canonical_lifecycle_event_id(
    FBIN_SECURITY_ID, "spinoff", SPINOFF_EFFECTIVE_DATE
)
FBIN_TICKER_EVENT_ID = canonical_lifecycle_event_id(
    FBIN_SECURITY_ID, "ticker_change", FBIN_HISTORY_START
)
PSEUDO_SPLIT_EVENT_ID = (
    "42bbc1a172d3a03bff59d361361c715e38ec85b5e51b0115a6b9cf4e6b7ad176"
)
PSEUDO_SPLIT_URL = (
    "https://eodhd.com/api/splits/FBIN.US?from=2015-01-01&to=2026-07-15"
)
PSEUDO_SPLIT_RAW_SHA256 = (
    "948db5813a8bcc3f51963ea162309f2253ddb01a9297cd59c477809d6f32fdc3"
)

ENDPOINTS = ("eod", "div", "splits")
EXPECTED_EODHD_CALLS = len(ENDPOINTS)
PENDING_MBC_FOLLOWUP_WARNING = (
    "MBC 1:1 spinoff collection required before final validation/publication"
)
REQUEST_PARAMS = {"from": FETCH_START, "to": FETCH_END}
REQUEST_URLS = {
    endpoint: (
        f"https://eodhd.com/api/{endpoint}/{PROVIDER_SYMBOL}"
        f"?from={FETCH_START}&to={FETCH_END}"
    )
    for endpoint in ENDPOINTS
}
REQUEST_ENVELOPE_CONTENT_TYPE = (
    "application/vnd.supertrendquant.source-envelope+json"
)

WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "source_archive",
)
FBHS_IDENTITY_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "index_constituent_anchors",
    "index_membership_events",
)


@dataclass(frozen=True)
class FrozenCatalogSelection:
    row: Mapping[str, Any]
    source_url: str
    retrieved_at: str
    source_hash: str
    object_path: str


@dataclass(frozen=True)
class FetchedBundle:
    prices: pd.DataFrame
    corporate_actions: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    http_attempts: int
    fetched_against_release: str = ""
    budget_used_before: int | None = None
    budget_used_after: int | None = None


@dataclass
class PreparedCollection:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    archive_artifacts: tuple[SourceArtifact, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _spinoff_metadata(form_8937: SourceArtifact) -> str:
    metadata = dict(SPINOFF_COST_BASIS_METADATA)
    metadata["source_url"] = form_8937.source_url
    metadata["source_hash"] = form_8937.source_hash
    return _canonical_json_bytes(metadata).decode("utf-8")


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _warnings_after_mbc_completion(
    warnings: Iterable[str],
) -> tuple[str, ...]:
    """Remove only the FBHS repair's completed MBC follow-up warning."""

    return tuple(
        warning
        for warning in warnings
        if warning != PENDING_MBC_FOLLOWUP_WARNING
    )


def _quality_for_release_warnings(warnings: Iterable[str]) -> DataQuality:
    return DataQuality.DEGRADED if tuple(warnings) else DataQuality.VALID


def _date(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _event_id(source: str, security_id: str, action_type: str, date: str) -> str:
    return hashlib.sha256(
        f"{source}|{security_id}|{action_type}|{date}".encode()
    ).hexdigest()


def _parse_split_ratio(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            divisor = float(denominator)
            ratio = float(numerator) / divisor if divisor else math.nan
        else:
            ratio = float(text)
    except (TypeError, ValueError):
        return None
    return ratio if math.isfinite(ratio) and ratio > 0 else None


def _expected_sessions() -> tuple[str, ...]:
    sessions = xcals.get_calendar("XNYS").sessions_in_range(FETCH_START, FETCH_END)
    return tuple(pd.Timestamp(value).date().isoformat() for value in sessions)


def _source_artifact(endpoint: str, rows: list[dict[str, Any]], retrieved_at: str) -> SourceArtifact:
    return SourceArtifact(
        source=f"eodhd_{endpoint}",
        source_url=REQUEST_URLS[endpoint],
        retrieved_at=retrieved_at,
        content=_canonical_json_bytes(rows),
        content_type="application/json",
    )


def _provider_action(
    *,
    artifact: SourceArtifact,
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
        "event_id": _event_id(
            artifact.source,
            MBC_SECURITY_ID,
            action_type,
            effective_date,
        ),
        "security_id": MBC_SECURITY_ID,
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
    }


def _bundle_from_artifacts(
    artifacts: Iterable[SourceArtifact],
    *,
    http_attempts: int,
    fetched_against_release: str = "",
    budget_used_before: int | None = None,
    budget_used_after: int | None = None,
) -> FetchedBundle:
    by_endpoint: dict[str, SourceArtifact] = {}
    for artifact in artifacts:
        expected_endpoint = artifact.source.removeprefix("eodhd_")
        if expected_endpoint not in ENDPOINTS:
            raise ValueError(f"Unexpected MBC artifact source: {artifact.source}")
        if expected_endpoint in by_endpoint:
            raise ValueError(f"Duplicate MBC artifact: {expected_endpoint}")
        if artifact.source_url != REQUEST_URLS[expected_endpoint]:
            raise ValueError(f"MBC {expected_endpoint} artifact URL is not exact.")
        by_endpoint[expected_endpoint] = artifact
    if set(by_endpoint) != set(ENDPOINTS):
        missing = sorted(set(ENDPOINTS) - set(by_endpoint))
        raise ValueError("MBC bundle lacks exact endpoints: " + ", ".join(missing))

    payloads: dict[str, list[dict[str, Any]]] = {}
    for endpoint, artifact in by_endpoint.items():
        try:
            value = json.loads(artifact.content)
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"MBC {endpoint} payload is not JSON.") from exc
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise ValueError(f"MBC {endpoint} payload must be a JSON row list.")
        if _canonical_json_bytes(value) != artifact.content:
            raise ValueError(f"MBC {endpoint} payload is not canonical raw evidence.")
        payloads[endpoint] = value

    eod = by_endpoint["eod"]
    prices = pd.DataFrame(
        [
            {
                "security_id": MBC_SECURITY_ID,
                "session": _text(row.get("date")),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume", 0),
                "currency": "USD",
                "source": eod.source,
                "source_url": eod.source_url,
                "retrieved_at": eod.retrieved_at,
                "source_hash": eod.source_hash,
            }
            for row in payloads["eod"]
            if _text(row.get("date")) and row.get("close") is not None
        ]
    )

    actions: list[dict[str, Any]] = []
    dividends = by_endpoint["div"]
    for row in payloads["div"]:
        effective = _text(row.get("date"))
        if not effective:
            continue
        actions.append(
            _provider_action(
                artifact=dividends,
                action_type="cash_dividend",
                effective_date=effective,
                cash_amount=row.get("unadjustedValue", row.get("value")),
                announcement_date=_text(row.get("declarationDate")),
                record_date=_text(row.get("recordDate")),
                payment_date=_text(row.get("paymentDate")),
                currency=_text(row.get("currency")) or "USD",
            )
        )
    splits = by_endpoint["splits"]
    for row in payloads["splits"]:
        effective = _text(row.get("date"))
        ratio = _parse_split_ratio(row.get("split"))
        if not effective or ratio is None:
            continue
        actions.append(
            _provider_action(
                artifact=splits,
                action_type="split",
                effective_date=effective,
                ratio=ratio,
            )
        )
    action_frame = pd.DataFrame(actions)
    return FetchedBundle(
        prices=prices,
        corporate_actions=action_frame,
        artifacts=tuple(by_endpoint[item] for item in ENDPOINTS),
        http_attempts=int(http_attempts),
        fetched_against_release=fetched_against_release,
        budget_used_before=budget_used_before,
        budget_used_after=budget_used_after,
    )


def validate_fetched_bundle(bundle: FetchedBundle) -> dict[str, Any]:
    if bundle.http_attempts != EXPECTED_EODHD_CALLS:
        raise ValueError(
            "MBC evidence must originate from exactly three EODHD calls: "
            f"observed={bundle.http_attempts}."
        )
    if (
        bundle.budget_used_before is not None
        and bundle.budget_used_after is not None
        and bundle.budget_used_after - bundle.budget_used_before
        != EXPECTED_EODHD_CALLS
    ):
        raise ValueError("Persistent EODHD budget did not advance by exactly three calls.")
    prices = bundle.prices.copy()
    if prices.empty:
        raise ValueError("MBC EOD payload contains no prices.")
    if prices.duplicated(["security_id", "session"], keep=False).any():
        raise ValueError("MBC EOD payload contains duplicate sessions.")
    numeric = prices[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or not numeric.gt(0).all().all():
        raise ValueError("MBC EOD payload contains invalid OHLC values.")
    if not (numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1)).all():
        raise ValueError("MBC EOD payload violates low <= OHLC.")
    if not (numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1)).all():
        raise ValueError("MBC EOD payload violates high >= OHLC.")
    observed = tuple(sorted(prices["session"].astype(str)))
    expected = _expected_sessions()
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(
            "MBC EOD session coverage is not exact: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    actions = bundle.corporate_actions
    if not actions.empty:
        if actions.duplicated(["event_id"], keep=False).any():
            raise ValueError("MBC provider actions contain duplicate event IDs.")
        dates = pd.to_datetime(actions["effective_date"], errors="coerce")
        if dates.isna().any() or not dates.between(FETCH_START, FETCH_END).all():
            raise ValueError("MBC provider actions escape the exact fetch range.")
        cash = actions["action_type"].astype(str).eq("cash_dividend")
        amounts = pd.to_numeric(actions.loc[cash, "cash_amount"], errors="coerce")
        if amounts.isna().any() or (amounts < 0).any():
            raise ValueError("MBC dividends contain invalid cash amounts.")
        split = actions["action_type"].astype(str).eq("split")
        ratios = pd.to_numeric(actions.loc[split, "ratio"], errors="coerce")
        if ratios.isna().any() or (ratios <= 0).any():
            raise ValueError("MBC splits contain invalid ratios.")
    return {
        "mbc_price_rows": len(prices),
        "mbc_first_session": observed[0],
        "mbc_last_session": observed[-1],
        "mbc_dividend_rows": int(
            actions.get("action_type", pd.Series(dtype=str)).astype(str).eq("cash_dividend").sum()
        ),
        "mbc_split_rows": int(
            actions.get("action_type", pd.Series(dtype=str)).astype(str).eq("split").sum()
        ),
        "expected_eodhd_calls": EXPECTED_EODHD_CALLS,
        "actual_eodhd_calls": bundle.http_attempts,
    }


class ExactThreeEodhdClient(EodhdClient):
    """One-attempt client restricted to the three reviewed MBC endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempted_endpoints: list[str] = []

    def get_json(self, endpoint: str, *, params: dict[str, object] | None = None):
        normalized = endpoint.strip("/")
        position = len(self.attempted_endpoints)
        if position >= EXPECTED_EODHD_CALLS:
            raise RuntimeError("MBC client refused a fourth EODHD request.")
        expected = f"{ENDPOINTS[position]}/{PROVIDER_SYMBOL}"
        if normalized != expected or dict(params or {}) != REQUEST_PARAMS:
            raise RuntimeError(
                "MBC client refused a non-reviewed request: "
                f"expected={expected}/{REQUEST_PARAMS}, observed={normalized}/{params}."
            )
        self.budget.claim()
        self.attempted_endpoints.append(normalized)
        response = self.session.get(
            f"{self.base_url}/{normalized}",
            params={**REQUEST_PARAMS, "api_token": self.token, "fmt": "json"},
            timeout=120,
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, list):
            raise RuntimeError(f"EODHD {normalized} did not return a row list.")
        return value


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
    release_version: str,
    budget_used_before: int,
) -> FetchedBundle:
    retrieved_at = utc_now_iso()
    artifacts: list[SourceArtifact] = []
    for endpoint in ENDPOINTS:
        rows = client.get_json(
            f"{endpoint}/{PROVIDER_SYMBOL}", params=dict(REQUEST_PARAMS)
        )
        artifacts.append(_source_artifact(endpoint, rows, retrieved_at))
    attempts = len(client.attempted_endpoints)
    if attempts != EXPECTED_EODHD_CALLS:
        raise RuntimeError(
            f"MBC collection made {attempts} calls; exactly three are required."
        )
    budget_after = _budget_used(client.budget)
    bundle = _bundle_from_artifacts(
        artifacts,
        http_attempts=attempts,
        fetched_against_release=release_version,
        budget_used_before=budget_used_before,
        budget_used_after=budget_after,
    )
    validate_fetched_bundle(bundle)
    return bundle


def _request_archive_artifact(artifact: SourceArtifact) -> SourceArtifact:
    envelope = _canonical_json_bytes(
        {
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
            "content_sha256": artifact.source_hash,
            "content_type": artifact.content_type,
            "source": artifact.source,
            "source_url": artifact.source_url,
        }
    )
    return SourceArtifact(
        source=artifact.source,
        source_url=artifact.source_url,
        retrieved_at=artifact.retrieved_at,
        content=envelope,
        content_type=REQUEST_ENVELOPE_CONTENT_TYPE,
    )


def _bundle_signature(catalog: FrozenCatalogSelection) -> dict[str, Any]:
    return {
        "schema": "us_spinoff_child_eodhd_bundle/v1",
        "security_id": MBC_SECURITY_ID,
        "provider_symbol": PROVIDER_SYMBOL,
        "fetch_start": FETCH_START,
        "fetch_end": FETCH_END,
        "request_urls": REQUEST_URLS,
        "catalog_source_hash": catalog.source_hash,
        "official_terms_sha256": SEC_SHA256,
        "expected_http_attempts": EXPECTED_EODHD_CALLS,
    }


def _bundle_cache_path(
    cache_root: Path,
    catalog: FrozenCatalogSelection,
) -> Path:
    digest = sha256_bytes(_canonical_json_bytes(_bundle_signature(catalog)))
    return cache_root / "state/us_spinoff_children" / f"{digest}.json.gz"


def _write_bundle_cache(
    path: Path,
    catalog: FrozenCatalogSelection,
    bundle: FetchedBundle,
) -> None:
    validate_fetched_bundle(bundle)
    payload = {
        **_bundle_signature(catalog),
        "fetched_against_release": bundle.fetched_against_release,
        "http_attempts": bundle.http_attempts,
        "budget_used_before": bundle.budget_used_before,
        "budget_used_after": bundle.budget_used_after,
        "artifacts": [
            {
                "source": item.source,
                "source_url": item.source_url,
                "retrieved_at": item.retrieved_at,
                "content_type": item.content_type,
                "content_base64": base64.b64encode(item.content).decode("ascii"),
                "content_sha256": item.source_hash,
            }
            for item in bundle.artifacts
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


def _read_bundle_cache(
    path: Path,
    catalog: FrozenCatalogSelection,
) -> FetchedBundle | None:
    if not path.is_file():
        return None
    try:
        wrapper = json.loads(gzip.decompress(path.read_bytes()))
        payload_bytes = base64.b64decode(
            wrapper["payload_base64"], validate=True
        )
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"MBC bundle cache is unreadable: {path}") from exc
    if sha256_bytes(payload_bytes) != _text(wrapper.get("payload_sha256")):
        raise ValueError(f"MBC bundle cache wrapper hash mismatch: {path}")
    payload = json.loads(payload_bytes)
    for key, expected in _bundle_signature(catalog).items():
        if payload.get(key) != expected:
            raise ValueError(f"MBC bundle signature mismatch for {key}: {path}")
    artifacts: list[SourceArtifact] = []
    for item in payload.get("artifacts", ()):
        content = base64.b64decode(item["content_base64"], validate=True)
        if sha256_bytes(content) != _text(item.get("content_sha256")):
            raise ValueError(f"MBC cached artifact hash mismatch: {item.get('source')}")
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
    validate_fetched_bundle(bundle)
    return bundle


def _safe_archived_content(repository: LocalDatasetRepository, object_path: str) -> bytes:
    root = repository.root.resolve()
    path = (repository.root / object_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Archived object escapes cache root: {object_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Archived object is missing: {path}")
    try:
        return gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError(f"Archived object is not valid gzip: {path}") from exc


def load_frozen_catalog(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> FrozenCatalogSelection:
    version = release.dataset_versions.get("source_archive")
    if not version:
        raise RuntimeError("Current release lacks source_archive.")
    archive = repository.read_frame("source_archive", version)
    matches = archive.loc[
        archive["dataset"].astype(str).eq("eodhd_exchange_symbols")
        & archive["source_url"].astype(str).eq(CATALOG_URL)
        & archive["source_hash"].astype(str).eq(CATALOG_SHA256)
    ].copy()
    if len(matches) != 1:
        raise ValueError(
            "Frozen active EODHD catalog is not uniquely pinned: "
            f"rows={len(matches)}."
        )
    source_row = matches.iloc[0]
    content = _safe_archived_content(repository, _text(source_row["object_path"]))
    if sha256_bytes(content) != CATALOG_SHA256:
        raise ValueError("Frozen EODHD catalog payload hash mismatch.")
    payload = json.loads(content)
    rows = [row for row in payload if _text(row.get("Code")).upper() == MBC_SYMBOL]
    if len(rows) != 1 or any(
        _text(rows[0].get(key)) != value
        for key, value in EXPECTED_CATALOG_ROW.items()
    ):
        raise ValueError("Frozen EODHD catalog lacks the exact reviewed MBC identity.")
    return FrozenCatalogSelection(
        row=dict(rows[0]),
        source_url=CATALOG_URL,
        retrieved_at=_text(source_row["retrieved_at"]),
        source_hash=CATALOG_SHA256,
        object_path=_text(source_row["object_path"]),
    )


def load_sec_evidence(cache_root: Path) -> SourceArtifact:
    path = cache_root / "state/sec_lifecycle" / f"{SEC_CACHE_KEY}.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Pinned FBIN/MBC SEC filing is missing: {path}")
    content = path.read_bytes()
    if len(content) != SEC_EXACT_BYTES or sha256_bytes(content) != SEC_SHA256:
        raise ValueError("Pinned FBIN/MBC SEC filing hash/size mismatch.")
    # SEC inline filings mix named HTML entities with the legacy numeric
    # Windows-1252 quote codes 145-148. Decode both before checking the
    # reviewed legal terms; the immutable raw bytes remain hash-pinned above.
    normalized = " ".join(
        html_unescape(content.decode("utf-8", errors="replace"))
        .translate(
            str.maketrans(
                {"\x91": "‘", "\x92": "’", "\x93": "“", "\x94": "”"}
            )
        )
        .lower()
        .split()
    )
    if any(value not in normalized for value in SEC_REQUIRED_TEXT):
        raise ValueError("Pinned SEC filing lacks the exact reviewed spin-off terms.")
    return SourceArtifact(
        source="sec_edgar_filing",
        source_url=SEC_URL,
        retrieved_at=SEC_RETRIEVED_AT,
        content=content,
        content_type="text/plain",
    )


def load_form_8937_evidence(cache_root: Path) -> SourceArtifact:
    path = cache_root / FORM_8937_CACHE_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Pinned issuer Form 8937 PDF is missing: {path}"
        )
    content = path.read_bytes()
    if (
        len(content) != FORM_8937_EXACT_BYTES
        or sha256_bytes(content) != FORM_8937_SHA256
    ):
        raise ValueError("Pinned issuer Form 8937 PDF hash/size mismatch.")
    return SourceArtifact(
        source="fbin_investor_relations_form_8937",
        source_url=FORM_8937_URL,
        retrieved_at=FORM_8937_RETRIEVED_AT,
        content=content,
        content_type="application/pdf",
    )


def _read_release_frames(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, pd.DataFrame]:
    return {
        dataset: repository.read_frame(dataset, version)
        for dataset, version in release.dataset_versions.items()
    }


def _fbhs_merge_error(message: str) -> RuntimeError:
    return RuntimeError(
        "FBHS/FBIN identity merge must be applied before MBC collection: "
        f"{message}. Run repair_us_fbhs_fbin_identity.py offline validation/apply first."
    )


def assert_fbhs_identity_merged(frames: Mapping[str, pd.DataFrame]) -> None:
    for dataset in FBHS_IDENTITY_DATASETS:
        frame = frames.get(dataset)
        if frame is not None and not frame.empty and "security_id" in frame:
            if frame["security_id"].astype(str).eq(RETIRED_FBHS_SECURITY_ID).any():
                raise _fbhs_merge_error(f"retired ID remains in {dataset}")
    master = frames["security_master"]
    canonical = master.loc[
        master["security_id"].astype(str).eq(FBIN_SECURITY_ID)
    ]
    if len(canonical) != 1:
        raise _fbhs_merge_error("canonical FBIN master row is not unique")
    row = canonical.iloc[0]
    if not (
        _text(row.get("primary_symbol")).upper() == FBIN_SYMBOL
        and _text(row.get("provider_symbol")).upper() == "FBIN.US"
        and not _date(row.get("active_to"))
    ):
        raise _fbhs_merge_error("canonical FBIN master row is not active/exact")
    history = frames["symbol_history"]
    observed = {
        (
            _text(item.symbol).upper(),
            _date(item.effective_from),
            _date(item.effective_to),
        )
        for item in history.loc[
            history["security_id"].astype(str).eq(FBIN_SECURITY_ID)
        ].itertuples(index=False)
    }
    expected = {
        (FBHS_SYMBOL, FBHS_HISTORY_START, FBHS_HISTORY_END),
        (FBIN_SYMBOL, FBIN_HISTORY_START, ""),
    }
    if observed != expected:
        raise _fbhs_merge_error(
            f"canonical symbol history differs: observed={sorted(observed)}"
        )
    actions = frames["corporate_actions"]
    ticker = actions.loc[
        actions["event_id"].astype(str).eq(FBIN_TICKER_EVENT_ID)
    ]
    if len(ticker) != 1:
        raise _fbhs_merge_error("official 2022-12-15 ticker action is missing")
    ticker_row = ticker.iloc[0]
    if not (
        _text(ticker_row.get("action_type")) == "ticker_change"
        and _text(ticker_row.get("new_security_id")) == FBIN_SECURITY_ID
        and _text(ticker_row.get("new_symbol")).upper() == FBIN_SYMBOL
        and bool(ticker_row.get("official"))
    ):
        raise _fbhs_merge_error("official ticker action is not exact")


def _pseudo_split_mask(actions: pd.DataFrame) -> pd.Series:
    ratio = pd.to_numeric(actions["ratio"], errors="coerce")
    return actions["event_id"].astype(str).eq(PSEUDO_SPLIT_EVENT_ID) | (
        actions["security_id"].astype(str).eq(FBIN_SECURITY_ID)
        & actions["action_type"].astype(str).eq("split")
        & actions["effective_date"].astype(str).eq(SPINOFF_EFFECTIVE_DATE)
        & ratio.sub(1.17).abs().le(1e-12)
        & actions["source"].astype(str).eq("eodhd_splits")
        & actions["source_hash"].astype(str).eq(PSEUDO_SPLIT_RAW_SHA256)
    )


def _official_spinoff_action(
    sec: SourceArtifact,
    form_8937: SourceArtifact,
) -> dict[str, Any]:
    return {
        "event_id": SPINOFF_EVENT_ID,
        "security_id": FBIN_SECURITY_ID,
        "action_type": "spinoff",
        "effective_date": SPINOFF_EFFECTIVE_DATE,
        "ex_date": SPINOFF_EFFECTIVE_DATE,
        "announcement_date": SPINOFF_ANNOUNCEMENT_DATE,
        "record_date": SPINOFF_RECORD_DATE,
        "payment_date": SPINOFF_PAYMENT_DATE,
        "cash_amount": None,
        "ratio": SPINOFF_RATIO,
        "currency": "USD",
        "new_security_id": MBC_SECURITY_ID,
        "new_symbol": MBC_SYMBOL,
        "official": True,
        "source_url": sec.source_url,
        "source_kind": "official_filing",
        "source": "official_fbin_mbc_spinoff",
        "retrieved_at": sec.retrieved_at,
        "source_hash": sec.source_hash,
        "metadata": _spinoff_metadata(form_8937),
    }


def _append_master_history(
    frames: Mapping[str, pd.DataFrame],
    catalog: FrozenCatalogSelection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = frames["security_master"].copy()
    history = frames["symbol_history"].copy()
    symbol_collision = master["primary_symbol"].astype(str).str.upper().eq(MBC_SYMBOL)
    id_collision = master["security_id"].astype(str).eq(MBC_SECURITY_ID)
    if symbol_collision.any() or id_collision.any():
        raise ValueError("MBC identity already exists but the spin-off is not complete.")
    row = {
        "security_id": MBC_SECURITY_ID,
        "primary_symbol": MBC_SYMBOL,
        "provider_symbol": PROVIDER_SYMBOL,
        "action_provider_symbol": PROVIDER_SYMBOL,
        "name": _text(catalog.row.get("Name")),
        "exchange": _text(catalog.row.get("Exchange")),
        "asset_type": "STOCK",
        "currency": _text(catalog.row.get("Currency")),
        "country": "US",
        "active_from": FETCH_START,
        "active_to": "",
        "isin": _text(catalog.row.get("Isin")),
        "source": "eodhd_exchange_symbols",
        "source_url": catalog.source_url,
        "retrieved_at": catalog.retrieved_at,
        "source_hash": catalog.source_hash,
    }
    master = pd.concat([master, pd.DataFrame([row])], ignore_index=True, sort=False)
    history_row = {
        "security_id": MBC_SECURITY_ID,
        "symbol": MBC_SYMBOL,
        "exchange": _text(catalog.row.get("Exchange")),
        "effective_from": FETCH_START,
        "effective_to": "",
        "source": "eodhd_exchange_symbols",
        "source_url": catalog.source_url,
        "retrieved_at": catalog.retrieved_at,
        "source_hash": catalog.source_hash,
    }
    history = pd.concat(
        [history, pd.DataFrame([history_row])], ignore_index=True, sort=False
    )
    return master, history


def _archive_extension(artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower().split(";", 1)[0]
    if "json" in content_type:
        return "json"
    if content_type == "text/plain":
        return "txt"
    if content_type == "application/pdf":
        return "pdf"
    return "bin"


def _append_source_archive(
    source_archive: pd.DataFrame,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> pd.DataFrame:
    output = source_archive.copy()
    for artifact in artifacts:
        row = {
            "archive_id": artifact.source_hash,
            "dataset": artifact.source,
            "object_path": (
                f"archives/{completed_session}/{artifact.source_hash}."
                f"{_archive_extension(artifact)}.gz"
            ),
            "content_type": artifact.content_type,
            "effective_date": completed_session,
            "source": artifact.source,
            "source_url": artifact.source_url,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
        }
        existing = output.loc[
            output["archive_id"].astype(str).eq(artifact.source_hash)
        ]
        if not existing.empty:
            exact = existing["source_url"].astype(str).eq(artifact.source_url)
            if len(existing) != 1 or not bool(exact.iloc[0]):
                raise ValueError(
                    f"Archive ID collision for {artifact.source_hash}."
                )
            continue
        output = pd.concat([output, pd.DataFrame([row])], ignore_index=True, sort=False)
    return output.reset_index(drop=True)


def _rebuild_factors(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    existing: pd.DataFrame,
    *,
    source_version: str,
) -> pd.DataFrame:
    affected = {FBIN_SECURITY_ID, MBC_SECURITY_ID}
    output = existing.loc[
        ~existing["security_id"].astype(str).isin(affected)
    ].copy()
    additions: list[pd.DataFrame] = []
    for security_id in sorted(affected):
        security_prices = prices.loc[
            prices["security_id"].astype(str).eq(security_id)
        ].copy()
        security_actions = actions.loc[
            actions["security_id"].astype(str).eq(security_id)
        ].copy()
        if security_prices.empty:
            raise ValueError(f"Cannot rebuild factors without prices: {security_id}")
        additions.append(
            build_adjustment_factors(
                security_prices,
                security_actions,
                source_version=source_version,
            )
        )
    return (
        pd.concat([output, *additions], ignore_index=True, sort=False)
        .drop_duplicates(
            list(dataset_spec("adjustment_factors").primary_key), keep="last"
        )
        .sort_values(["security_id", "session"])
        .reset_index(drop=True)
    )


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

    def read_frame(self, dataset: str, version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        resolved = version or self.versions.get(dataset)
        if not resolved:
            return pd.DataFrame(columns=dataset_spec(dataset).required_columns)
        return self.base.read_frame(dataset, resolved)


def _looks_applied(frames: Mapping[str, pd.DataFrame]) -> bool:
    master = frames["security_master"]
    actions = frames["corporate_actions"]
    return bool(
        master["security_id"].astype(str).eq(MBC_SECURITY_ID).any()
        and actions["event_id"].astype(str).eq(SPINOFF_EVENT_ID).any()
        and not _pseudo_split_mask(actions).any()
    )


def _prepare_already_applied(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
    frames: Mapping[str, pd.DataFrame],
    catalog: FrozenCatalogSelection,
    bundle: FetchedBundle | None,
    sec: SourceArtifact,
    form_8937: SourceArtifact,
) -> PreparedCollection:
    summary = validate_candidate_frames(
        frames,
        bundle,
        catalog,
        sec,
        form_8937,
        completed_session=release.completed_session,
        base_repository=repository,
        base_versions=release.dataset_versions,
    )
    remaining_warnings = _warnings_after_mbc_completion(release.warnings)
    return PreparedCollection(
        release=release,
        release_etag=release_etag,
        pointer_etags=_capture_pointer_etags(repository, release),
        frames={dataset: frames[dataset].copy() for dataset in WRITE_DATASETS},
        archive_artifacts=(),
        warnings=remaining_warnings,
        summary={
            **summary,
            "status": "already_applied",
            "base_release_version": release.version,
            "network_accessed": False,
            "r2_accessed": False,
            "warning_cleanup_required": (
                remaining_warnings != tuple(release.warnings)
            ),
            "remaining_release_warnings": list(remaining_warnings),
        },
    )


def validate_candidate_frames(
    frames: Mapping[str, pd.DataFrame],
    bundle: FetchedBundle | None,
    catalog: FrozenCatalogSelection,
    sec: SourceArtifact,
    form_8937: SourceArtifact,
    *,
    completed_session: str,
    base_repository: LocalDatasetRepository | None = None,
    base_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    assert_fbhs_identity_merged(frames)
    for dataset in WRITE_DATASETS:
        validate_dataset(
            dataset,
            frames[dataset],
            incomplete_action_policy="warn",
            completed_session=completed_session,
        ).raise_for_errors()
    master = frames["security_master"]
    mbc = master.loc[master["security_id"].astype(str).eq(MBC_SECURITY_ID)]
    if len(mbc) != 1:
        raise ValueError("MBC security_master row is not unique.")
    row = mbc.iloc[0]
    if not (
        _text(row.get("primary_symbol")) == MBC_SYMBOL
        and _text(row.get("provider_symbol")) == PROVIDER_SYMBOL
        and _text(row.get("action_provider_symbol")) == PROVIDER_SYMBOL
        and _text(row.get("name")) == EXPECTED_CATALOG_ROW["Name"]
        and _text(row.get("exchange")) == EXPECTED_CATALOG_ROW["Exchange"]
        and _text(row.get("isin")) == EXPECTED_CATALOG_ROW["Isin"]
        and _date(row.get("active_from")) == FETCH_START
        and not _date(row.get("active_to"))
        and _text(row.get("source_hash")) == catalog.source_hash
    ):
        raise ValueError("MBC security_master row is not exact.")
    history = frames["symbol_history"].loc[
        frames["symbol_history"]["security_id"].astype(str).eq(MBC_SECURITY_ID)
    ]
    if len(history) != 1 or not (
        _text(history.iloc[0].get("symbol")) == MBC_SYMBOL
        and _date(history.iloc[0].get("effective_from")) == FETCH_START
        and not _date(history.iloc[0].get("effective_to"))
    ):
        raise ValueError("MBC symbol_history row is not exact.")
    prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"].astype(str).eq(MBC_SECURITY_ID)
    ].copy()
    if bundle is not None:
        fetched_summary = validate_fetched_bundle(bundle)
        expected_keys = set(
            zip(bundle.prices["security_id"].astype(str), bundle.prices["session"].astype(str))
        )
        observed_keys = set(zip(prices["security_id"].astype(str), prices["session"].astype(str)))
        if observed_keys != expected_keys:
            raise ValueError("Installed MBC prices differ from the fetched bundle.")
        eod = bundle.artifacts[0]
        if set(prices["source_hash"].astype(str)) != {eod.source_hash}:
            raise ValueError("Installed MBC prices lost raw EODHD provenance.")
    else:
        fetched_summary = {
            "mbc_price_rows": len(prices),
            "mbc_first_session": min(prices["session"].astype(str)),
            "mbc_last_session": max(prices["session"].astype(str)),
        }
    if tuple(sorted(prices["session"].astype(str))) != _expected_sessions():
        raise ValueError("Installed MBC price coverage is not exact.")
    actions = frames["corporate_actions"]
    if _pseudo_split_mask(actions).any():
        raise ValueError("The fake FBIN 1.17 split remains installed.")
    spinoff = actions.loc[actions["event_id"].astype(str).eq(SPINOFF_EVENT_ID)]
    if len(spinoff) != 1:
        raise ValueError("Official FBIN/MBC spinoff action is not unique.")
    action = spinoff.iloc[0]
    expected_metadata = _spinoff_metadata(form_8937)
    if not (
        _text(action.get("security_id")) == FBIN_SECURITY_ID
        and _text(action.get("action_type")) == "spinoff"
        and _date(action.get("effective_date")) == SPINOFF_EFFECTIVE_DATE
        and _date(action.get("ex_date")) == SPINOFF_EFFECTIVE_DATE
        and _date(action.get("record_date")) == SPINOFF_RECORD_DATE
        and _date(action.get("payment_date")) == SPINOFF_PAYMENT_DATE
        and float(action.get("ratio")) == SPINOFF_RATIO
        and _text(action.get("new_security_id")) == MBC_SECURITY_ID
        and _text(action.get("new_symbol")) == MBC_SYMBOL
        and bool(action.get("official"))
        and _text(action.get("source_url")) == sec.source_url
        and _text(action.get("source_hash")) == sec.source_hash
    ):
        raise ValueError("Official FBIN/MBC spinoff terms are not exact.")
    if _text(action.get("metadata")) != expected_metadata:
        raise ValueError(
            "Official FBIN/MBC spinoff cost-basis metadata is not exact."
        )
    factors = frames["adjustment_factors"]
    for security_id in (FBIN_SECURITY_ID, MBC_SECURITY_ID):
        price_sessions = set(
            frames["daily_price_raw"].loc[
                frames["daily_price_raw"]["security_id"].astype(str).eq(security_id),
                "session",
            ].astype(str)
        )
        factor_rows = factors.loc[
            factors["security_id"].astype(str).eq(security_id)
        ]
        if set(factor_rows["session"].astype(str)) != price_sessions:
            raise ValueError(f"Adjustment factors do not cover {security_id} prices.")
    fbin_factors = factors.loc[
        factors["security_id"].astype(str).eq(FBIN_SECURITY_ID)
    ]
    if not pd.to_numeric(fbin_factors["split_factor"], errors="coerce").eq(1.0).all():
        raise ValueError("Removed FBIN pseudo-split still changes split factors.")
    archive = frames["source_archive"]
    expected_artifacts = (
        tuple(_request_archive_artifact(item) for item in bundle.artifacts)
        if bundle is not None
        else ()
    ) + (sec, form_8937)
    for artifact in expected_artifacts:
        found = archive["archive_id"].astype(str).eq(artifact.source_hash) & archive[
            "source_url"
        ].astype(str).eq(artifact.source_url)
        if int(found.sum()) != 1:
            raise ValueError(
                f"Source archive lacks exact evidence: {artifact.source_url}."
            )
    if base_repository is not None and base_versions is not None:
        candidate = _CandidateRepository(base_repository, base_versions, frames)
        validate_repository_snapshot(candidate).raise_for_errors()
    return {
        **fetched_summary,
        "mbc_security_id": MBC_SECURITY_ID,
        "spinoff_event_id": SPINOFF_EVENT_ID,
        "official_spinoff_rows": 1,
        "spinoff_cost_basis_fraction": 0.123028,
        "form_8937_archive_id": form_8937.source_hash,
        "pseudo_split_rows": 0,
        "catalog_calls": 0,
        "r2_accessed": False,
    }


def prepare_collection(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
    frames: Mapping[str, pd.DataFrame],
    catalog: FrozenCatalogSelection,
    bundle: FetchedBundle,
    sec: SourceArtifact,
    form_8937: SourceArtifact,
) -> PreparedCollection:
    assert_fbhs_identity_merged(frames)
    if _looks_applied(frames):
        return _prepare_already_applied(
            repository,
            release,
            release_etag,
            frames,
            catalog,
            bundle,
            sec,
            form_8937,
        )
    validate_fetched_bundle(bundle)
    master, history = _append_master_history(frames, catalog)
    prices = pd.concat(
        [frames["daily_price_raw"], bundle.prices], ignore_index=True, sort=False
    ).drop_duplicates(
        list(dataset_spec("daily_price_raw").primary_key), keep="last"
    )
    actions = frames["corporate_actions"].copy()
    pseudo = _pseudo_split_mask(actions)
    if int(pseudo.sum()) not in {0, 1}:
        raise ValueError("FBIN pseudo-split is not unique before replacement.")
    actions = actions.loc[~pseudo].copy()
    actions = pd.concat(
        [
            actions,
            bundle.corporate_actions,
            pd.DataFrame([_official_spinoff_action(sec, form_8937)]),
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(list(dataset_spec("corporate_actions").primary_key), keep="last")
    factors = _rebuild_factors(
        prices,
        actions,
        frames["adjustment_factors"],
        source_version=f"us-spinoff-children:{release.version}",
    )
    archive_artifacts = tuple(
        _request_archive_artifact(item) for item in bundle.artifacts
    ) + (sec, form_8937)
    archive = _append_source_archive(
        frames["source_archive"],
        archive_artifacts,
        completed_session=release.completed_session,
    )
    rewritten = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices.sort_values(
            ["security_id", "session"]
        ).reset_index(drop=True),
        "corporate_actions": actions.sort_values(
            ["security_id", "effective_date", "event_id"]
        ).reset_index(drop=True),
        "adjustment_factors": factors,
        "source_archive": archive,
    }
    summary = validate_candidate_frames(
        rewritten,
        bundle,
        catalog,
        sec,
        form_8937,
        completed_session=release.completed_session,
        base_repository=repository,
        base_versions=release.dataset_versions,
    )
    remaining_warnings = _warnings_after_mbc_completion(release.warnings)
    return PreparedCollection(
        release=release,
        release_etag=release_etag,
        pointer_etags=_capture_pointer_etags(repository, release),
        frames=rewritten,
        archive_artifacts=archive_artifacts,
        warnings=remaining_warnings,
        summary={
            **summary,
            "status": "validated_offline_plan",
            "base_release_version": release.version,
            "fake_split_rows_removed": int(pseudo.sum()),
            "network_accessed": False,
            "warning_cleanup_required": (
                remaining_warnings != tuple(release.warnings)
            ),
            "remaining_release_warnings": list(remaining_warnings),
        },
    )


def _capture_pointer_etags(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}")
        output[dataset] = etag
    return output


def _persist_archive_payloads(
    repository: LocalDatasetRepository,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> None:
    for artifact in artifacts:
        path = (
            repository.root
            / "archives"
            / completed_session
            / f"{artifact.source_hash}.{_archive_extension(artifact)}.gz"
        )
        if path.is_file():
            if gzip.decompress(path.read_bytes()) != artifact.content:
                raise RuntimeError(f"Conflicting immutable archive payload: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, gzip.compress(artifact.content, mtime=0))
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"Immutable archive verification failed: {path}")


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / "recovery/us-spinoff-children"
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved spin-off recovery marker blocks writes.")
        transactions = repository.root / "transactions/us-spinoff-children"
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted spin-off transaction blocks writes: {journal}"
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, indent=2
    ).encode() + b"\n")


def _assert_release_unchanged(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
) -> None:
    current, etag = repository.current_release()
    if current is None or current.version != release.version or etag != release_etag:
        raise RuntimeError("Current release changed during MBC offline validation.")


def _restore_transaction_state(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    release_key = "releases/current.json"
    try:
        current = repository.objects.get(release_key)
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            belongs = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or all(
                observed.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not belongs:
                raise RuntimeError(
                    f"unexpected release during rollback: {observed.version}"
                )
            repository.objects.put(release_key, old_release_bytes, if_match=current.etag)
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
                        f"unexpected pointer during rollback: {pointer.version}"
                    )
                repository.objects.put(key, old_bytes, if_match=current.etag)
            if repository.objects.get(key).data != old_bytes:
                raise RuntimeError("pointer preimage verification failed")
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> None:
    current, _ = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes():
        raise RuntimeError("Committed MBC release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, _ = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"Committed MBC pointer mismatch: {dataset}")
    validate_repository_snapshot(repository).raise_for_errors()


def _apply_already_applied_warning_cleanup(
    repository: LocalDatasetRepository,
    prepared: PreparedCollection,
) -> dict[str, Any]:
    """CAS-repair stale release metadata without touching any dataset pointer."""

    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(
            repository, prepared.release, prepared.release_etag
        )
        remaining_warnings = _warnings_after_mbc_completion(
            prepared.release.warnings
        )
        if tuple(prepared.warnings) != remaining_warnings:
            raise RuntimeError(
                "Prepared MBC warning cleanup changed before metadata apply."
            )
        if remaining_warnings == tuple(prepared.release.warnings):
            return {
                **prepared.summary,
                "status": "already_applied",
                "mode": "apply",
                "warning_cleanup_required": False,
                "warning_removed": False,
                "metadata_only_release_repair": False,
                "dataset_writes_performed": False,
                "writes_performed": False,
                "release_version": prepared.release.version,
                "quality": prepared.release.quality,
                "warnings": list(prepared.release.warnings),
            }

        for dataset, version in prepared.release.dataset_versions.items():
            pointer, _etag = repository.current_pointer(dataset)
            if pointer is None or pointer.version != version:
                raise RuntimeError(
                    f"MBC metadata-only repair pointer mismatch: {dataset}"
                )
        validate_repository_snapshot(repository).raise_for_errors()
        committed = repository.commit_release(
            prepared.release.completed_session,
            dict(prepared.release.dataset_versions),
            quality=_quality_for_release_warnings(remaining_warnings),
            warnings=remaining_warnings,
            expected_etag=prepared.release_etag,
        )
        _assert_applied_release(repository, committed)
        return {
            **prepared.summary,
            "status": "applied_metadata_only",
            "mode": "apply",
            "warning_cleanup_required": False,
            "warning_removed": True,
            "metadata_only_release_repair": True,
            "dataset_writes_performed": False,
            "writes_performed": True,
            "old_release_version": prepared.release.version,
            "new_release_version": committed.version,
            "new_dataset_versions": dict(committed.dataset_versions),
            "quality": committed.quality,
            "warnings": list(committed.warnings),
            "network_accessed": False,
            "r2_accessed": False,
        }


def apply_collection(
    repository: LocalDatasetRepository,
    prepared: PreparedCollection,
) -> dict[str, Any]:
    if prepared.summary.get("status") == "already_applied":
        return _apply_already_applied_warning_cleanup(repository, prepared)
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(
            repository, prepared.release, prepared.release_etag
        )
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before MBC apply.")
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        planned = {
            dataset: (
                f"us-spinoff-children-"
                f"{prepared.release.completed_session.replace('-', '')}-"
                f"{transaction_id}-{dataset}"
            )
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/us-spinoff-children"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "us_spinoff_children_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": planned,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            _persist_archive_payloads(
                repository,
                prepared.archive_artifacts,
                completed_session=prepared.release.completed_session,
            )
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="warn",
                    metadata={
                        "operation": "collect_us_spinoff_children",
                        "network_accessed": False,
                        "eodhd_calls_from_cache": EXPECTED_EODHD_CALLS,
                        "catalog_calls": 0,
                        "r2_accessed": False,
                    },
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version
            post = validate_repository_snapshot(repository)
            post.raise_for_errors()
            warnings = _warnings_after_mbc_completion(
                dict.fromkeys(
                    (
                        *prepared.warnings,
                        *(
                            issue.message
                            for issue in post.issues
                            if issue.severity != "error"
                        ),
                    )
                )
            )
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=_quality_for_release_warnings(warnings),
                warnings=warnings,
                expected_etag=prepared.release_etag,
            )
            _assert_applied_release(repository, committed)
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
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
                "transaction_id": transaction_id,
            }
        except BaseException as original:
            rollback_errors = _restore_transaction_state(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=committed.version if committed else "",
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
                recovery = (
                    repository.root
                    / "recovery/us-spinoff-children"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "MBC rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}"
                ) from original
            raise


def _base_context(
    repository: LocalDatasetRepository,
) -> tuple[DataRelease, str | None, dict[str, pd.DataFrame], FrozenCatalogSelection]:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    if release.completed_session != FETCH_END:
        raise RuntimeError(
            "MBC transaction is pinned to the reviewed completed session: "
            f"expected={FETCH_END}, current={release.completed_session}."
        )
    missing = sorted(set(WRITE_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    frames = _read_release_frames(repository, release)
    assert_fbhs_identity_merged(frames)
    catalog = load_frozen_catalog(repository, release)
    return release, release_etag, frames, catalog


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[[str | Path], LocalDatasetRepository] = LocalDatasetRepository,
    budget_factory: Callable[[], EodhdCallBudget] = EodhdCallBudget,
    client_factory: Callable[..., ExactThreeEodhdClient] = ExactThreeEodhdClient,
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    release, release_etag, frames, catalog = _base_context(repository)
    sec = load_sec_evidence(repository.root)
    form_8937 = load_form_8937_evidence(repository.root)
    path = _bundle_cache_path(repository.root, catalog)
    mode = _text(getattr(args, "mode", "plan")) or "plan"
    if _looks_applied(frames):
        prepared = _prepare_already_applied(
            repository,
            release,
            release_etag,
            frames,
            catalog,
            None,
            sec,
            form_8937,
        )
        if mode == "apply":
            return apply_collection(repository, prepared)
        return {
            **prepared.summary,
            "would_write": False,
        }
    cached = _read_bundle_cache(path, catalog)
    if mode == "plan":
        return {
            "status": "ready_offline_replay" if cached else "blocked_cache_missing",
            "base_release_version": release.version,
            "cache_path": str(path),
            "cache_present": cached is not None,
            "expected_eodhd_calls": EXPECTED_EODHD_CALLS,
            "request_urls": REQUEST_URLS,
            "form_8937_cache_path": str(
                repository.root / FORM_8937_CACHE_RELATIVE_PATH
            ),
            "form_8937_sha256": form_8937.source_hash,
            "catalog_calls": 0,
            "network_accessed": False,
            "would_write": False,
            "r2_accessed": False,
        }
    if mode == "fetch_missing":
        if cached is not None:
            return {
                **validate_fetched_bundle(cached),
                "status": "cache_already_complete",
                "cache_path": str(path),
                "network_accessed": False,
                "would_write": False,
                "r2_accessed": False,
            }
        budget = budget_factory()
        before = _budget_used(budget)
        if before + EXPECTED_EODHD_CALLS > budget.ceiling:
            raise RuntimeError(
                "EODHD budget cannot reserve the exact three-call MBC bundle: "
                f"used={before}, ceiling={budget.ceiling}."
            )
        client = client_factory(budget=budget)
        bundle = fetch_exact_bundle(
            client,
            release_version=release.version,
            budget_used_before=before,
        )
        _write_bundle_cache(path, catalog, bundle)
        replay = _read_bundle_cache(path, catalog)
        if replay is None:
            raise RuntimeError("MBC bundle cache disappeared after write.")
        return {
            **validate_fetched_bundle(replay),
            "status": "fetched_cache_only",
            "cache_path": str(path),
            "budget_used_before": replay.budget_used_before,
            "budget_used_after": replay.budget_used_after,
            "network_accessed": True,
            "would_write": False,
            "r2_accessed": False,
        }
    if cached is None:
        raise RuntimeError(
            "MBC immutable bundle is missing. Run the default plan, then explicitly "
            "use --fetch-missing once before offline replay/apply."
        )
    prepared = prepare_collection(
        repository,
        release,
        release_etag,
        frames,
        catalog,
        cached,
        sec,
        form_8937,
    )
    if mode == "offline_plan":
        return {
            **prepared.summary,
            "cache_path": str(path),
            "would_write": False,
        }
    if mode == "apply":
        return apply_collection(repository, prepared)
    raise ValueError(f"Unsupported mode: {mode}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect exactly three MBC EODHD endpoints and atomically replace "
            "the synthetic FBIN 1.17 split with the official 1:1 spin-off."
        )
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fetch-missing",
        action="store_const",
        const="fetch_missing",
        dest="mode",
        help="Make the exact three one-attempt EODHD calls and write only a local bundle cache.",
    )
    mode.add_argument(
        "--offline-plan",
        action="store_const",
        const="offline_plan",
        dest="mode",
        help="Require and replay the local bundle without network or dataset writes.",
    )
    mode.add_argument(
        "--apply",
        action="store_const",
        const="apply",
        dest="mode",
        help="Replay the local bundle and perform the CAS/journal transaction.",
    )
    parser.set_defaults(mode="plan")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
