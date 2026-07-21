#!/usr/bin/env python3
"""Repair the two audited Alcoa-lineage spin-offs as one local transaction.

The current US snapshot already owns the Old Alcoa -> Arconic -> Howmet
identity and the 2020 ARNC child price segment.  It does not own the 2016
Alcoa Corporation child, its two issuer cost-basis allocations, or a clean
parent action stream.  EODHD also reports a synthetic 667/500 split on the
2016 separation date; that ratio is not a legal share split and must never be
applied to the parent position.

The default command is read-only.  ``--fetch-official-evidence`` makes at
most two one-shot requests to the two hash-pinned Howmet tax-basis PDFs.
``--fetch-missing`` makes exactly three one-shot EODHD requests (AA eod,
div, splits) and writes only an immutable local replay bundle.  ``--offline-
plan`` replays both caches without network or dataset writes.  ``--apply`` is
the only mode allowed to move dataset and release pointers; it uses one
writer lock, compare-and-swap pointers, a durable journal, verified rollback,
and an idempotent already-applied path.  No mode accesses R2.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import json
import math
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
FETCH_START = "2016-11-01"
FETCH_END = "2026-07-15"
PROVIDER_SYMBOL = "AA.US"
AA_SYMBOL = "AA"
HWM_SYMBOL = "HWM"
ARNC_SYMBOL = "ARNC"

HWM_SECURITY_ID = "US:EODHD:f5daeed5-d1a2-5279-aa49-8c06c902b97f"
AA_SECURITY_ID = "US:EODHD:a0eebd04-b9d4-54bc-8682-899643216993"
ARNC_SECURITY_ID = "US:EODHD:33cf5387-6cec-598e-84a9-563ca333b0f3"

SPINOFF_2016_EVENT_ID = (
    "fd8e8f6bc37de73342db04090624676b30b75765657bd2867bd41fd62fffd187"
)
SPINOFF_2016_EFFECTIVE_DATE = "2016-11-01"
SPINOFF_2016_RECORD_DATE = "2016-10-20"
SPINOFF_2016_PAYMENT_DATE = "2016-11-01"
SPINOFF_2016_RATIO = 1.0 / 3.0
SPINOFF_2020_EVENT_ID = (
    "15d7f68b627981221f55f109696085382809e34d4abfb98163704d1b47a45b04"
)
SPINOFF_2020_EFFECTIVE_DATE = "2020-04-01"
SPINOFF_2020_RECORD_DATE = "2020-03-19"
SPINOFF_2020_PAYMENT_DATE = "2020-04-01"
SPINOFF_2020_RATIO = 0.25
LEGAL_REVERSE_SPLIT_EVENT_ID = (
    "3e171cfde0ecf70fa5458c4c0e0f32f9c3f11aaada2c2bf9722b643ceb63a1c0"
)
LEGAL_REVERSE_SPLIT_DATE = "2016-10-06"
PSEUDO_SPLIT_EVENT_ID = (
    "c104151f5fc1f48840e00400f764d56c5588699e6a5c8cc0acf322b9f49239d8"
)
PSEUDO_SPLIT_EFFECTIVE_DATE = "2016-11-01"
PSEUDO_SPLIT_RATIO = 667.0 / 500.0
PSEUDO_SPLIT_SOURCE_URL = (
    "https://eodhd.com/api/splits/HWM.US?from=2015-01-01&to=2026-07-15"
)
PSEUDO_SPLIT_RAW_SHA256 = (
    "f062be90e011e57b238265c59e514238e82ec97616d1d2047acdbd5885578d2b"
)

SPINOFF_2016_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/4281/"
    "000119312516731663/d249430dex991.htm"
)
SPINOFF_2016_SOURCE_SHA256 = (
    "3e79667b1b4efd2981b0aa2137ec9e83239661960b8ed88d7ecca4f594d18e49"
)
SPINOFF_2020_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1790982/"
    "000179098221000096/arnc-20210630.htm"
)
SPINOFF_2020_SOURCE_SHA256 = (
    "6b88966f49fb2a2c7a2bbb832873aad7db2f396148ccf63484849729009adf70"
)

TAX_BASIS_2016_URL = (
    "https://www.howmet.com/wp-content/uploads/sites/3/2023/05/"
    "Tax-Basis-Information-for-Shares-after-the-Separation.pdf"
)
TAX_BASIS_2016_SHA256 = (
    "527dfea5f6529bd0154cc351c7fb9066c7b6e7b7db5fac39456c2d372593b9ec"
)
TAX_BASIS_2016_SIZE = 133_224
TAX_BASIS_2020_URL = (
    "https://www.howmet.com/wp-content/uploads/sites/3/2023/05/"
    "Howmet-Tax-Basis-Information-for-Shares-after-the-Separation.pdf"
)
TAX_BASIS_2020_SHA256 = (
    "6aab045e25c436e1b8a2bccd68edd907fe34761e1c24ba7e51d2c3bd0899e6eb"
)
TAX_BASIS_2020_SIZE = 130_103
OFFICIAL_RETRIEVED_AT = "2026-07-18T07:47:00Z"

OFFICIAL_SPECS = {
    "2016": {
        "url": TAX_BASIS_2016_URL,
        "sha256": TAX_BASIS_2016_SHA256,
        "size": TAX_BASIS_2016_SIZE,
        "source": "howmet_tax_basis_alcoa_2016",
    },
    "2020": {
        "url": TAX_BASIS_2020_URL,
        "sha256": TAX_BASIS_2020_SHA256,
        "size": TAX_BASIS_2020_SIZE,
        "source": "howmet_tax_basis_arconic_2020",
    },
}

COST_BASIS_2016 = {
    "average_prices": {"AA": 22.67, "ARNC": 20.59},
    "cost_basis_fraction": 0.2686,
    "currency": "USD",
    "method": "relative_fair_market_value_average_high_low",
    "parent_cost_basis_fraction": 0.7314,
    "source_hash": TAX_BASIS_2016_SHA256,
    "source_url": TAX_BASIS_2016_URL,
    "valuation_date": "2016-11-01",
}
COST_BASIS_2020 = {
    "cost_basis_fraction": 0.114,
    "currency": "USD",
    "method": "relative_fair_market_value_based_on_vwap",
    "parent_cost_basis_fraction": 0.886,
    "source_hash": TAX_BASIS_2020_SHA256,
    "source_url": TAX_BASIS_2020_URL,
    "valuation_date": "2020-04-02",
    "vwaps": {"ARNC": 6.93, "HWM": 13.40},
}

EXPECTED_CATALOG_ROW = {
    "Code": "AA",
    "Country": "USA",
    "Currency": "USD",
    "Exchange": "NYSE",
    "Isin": "US0138721065",
    "Name": "Alcoa Corp",
    "Type": "Common Stock",
}

ENDPOINTS = ("eod", "div", "splits")
EXPECTED_EODHD_CALLS = len(ENDPOINTS)
REQUEST_PARAMS = {"from": FETCH_START, "to": FETCH_END}
REQUEST_URLS = {
    endpoint: (
        f"https://eodhd.com/api/{endpoint}/{PROVIDER_SYMBOL}"
        f"?from={FETCH_START}&to={FETCH_END}"
    )
    for endpoint in ENDPOINTS
}

WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "source_archive",
)
TRANSACTION_NAME = "us-alcoa-spinoff-repair"
REQUEST_ENVELOPE_CONTENT_TYPE = (
    "application/vnd.supertrendquant.raw-request+json"
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
    inception_pseudo_split_rows: int
    budget_used_before: int | None = None
    budget_used_after: int | None = None


@dataclass
class PreparedRepair:
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


def _text(value: Any) -> str:
    if value is None or (
        not isinstance(value, (dict, list)) and pd.isna(value)
    ):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _quality_for_warnings(warnings: Iterable[str]) -> DataQuality:
    return DataQuality.DEGRADED if tuple(warnings) else DataQuality.VALID


def _expected_sessions() -> tuple[str, ...]:
    sessions = xcals.get_calendar("XNYS").sessions_in_range(FETCH_START, FETCH_END)
    output = tuple(pd.Timestamp(value).date().isoformat() for value in sessions)
    if len(output) != 2_437:
        raise RuntimeError(
            "Pinned AA XNYS session inventory changed: "
            f"expected=2437, actual={len(output)}."
        )
    return output


def _parse_split_ratio(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            divisor = float(denominator)
            ratio = float(numerator) / divisor if divisor else math.nan
        elif ":" in text:
            numerator, denominator = text.split(":", 1)
            divisor = float(denominator)
            ratio = float(numerator) / divisor if divisor else math.nan
        else:
            ratio = float(text)
    except (TypeError, ValueError):
        return None
    return ratio if math.isfinite(ratio) and ratio > 0 else None


def _event_id(source: str, action_type: str, date: str) -> str:
    return hashlib.sha256(
        f"{source}|{AA_SECURITY_ID}|{action_type}|{date}".encode()
    ).hexdigest()


def _official_cache_path(cache_root: Path, spec: Mapping[str, Any]) -> Path:
    return cache_root / "state/issuer_lifecycle" / f"{spec['sha256']}.pdf"


def _load_one_official(
    cache_root: Path,
    spec: Mapping[str, Any],
) -> SourceArtifact:
    path = _official_cache_path(cache_root, spec)
    if not path.is_file():
        raise FileNotFoundError(f"Pinned issuer tax-basis PDF is missing: {path}")
    content = path.read_bytes()
    if len(content) != int(spec["size"]) or sha256_bytes(content) != spec["sha256"]:
        raise ValueError(f"Pinned issuer tax-basis PDF hash/size mismatch: {path}")
    return SourceArtifact(
        source=str(spec["source"]),
        source_url=str(spec["url"]),
        retrieved_at=OFFICIAL_RETRIEVED_AT,
        content=content,
        content_type="application/pdf",
    )


def load_official_evidence(cache_root: Path) -> dict[str, SourceArtifact]:
    return {
        label: _load_one_official(cache_root, spec)
        for label, spec in OFFICIAL_SPECS.items()
    }


class ExactOfficialEvidenceClient:
    """At most one request per reviewed Howmet PDF and never a third request."""

    def __init__(self, session=None):
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.attempted_urls: list[str] = []

    def fetch(self, label: str) -> SourceArtifact:
        if label not in OFFICIAL_SPECS:
            raise RuntimeError("Official-evidence client refused an unreviewed request.")
        spec = OFFICIAL_SPECS[label]
        url = str(spec["url"])
        if len(self.attempted_urls) >= len(OFFICIAL_SPECS) or url in self.attempted_urls:
            raise RuntimeError("Official-evidence client refused an unreviewed request.")
        self.attempted_urls.append(url)
        response = self.session.get(
            url,
            headers={"User-Agent": "SuperTrendQuant/1.0 data-validation"},
            timeout=120,
        )
        response.raise_for_status()
        content = bytes(response.content)
        if len(content) != int(spec["size"]) or sha256_bytes(content) != spec["sha256"]:
            raise RuntimeError(f"Official evidence changed for {label}.")
        return SourceArtifact(
            source=str(spec["source"]),
            source_url=url,
            retrieved_at=utc_now_iso(),
            content=content,
            content_type="application/pdf",
        )


def fetch_missing_official_evidence(
    cache_root: Path,
    *,
    client: ExactOfficialEvidenceClient,
) -> dict[str, Any]:
    fetched: list[str] = []
    ordered_missing: list[str] = []
    for label, spec in OFFICIAL_SPECS.items():
        path = _official_cache_path(cache_root, spec)
        if path.is_file():
            _load_one_official(cache_root, spec)
        else:
            ordered_missing.append(label)
    for label in ordered_missing:
        artifact = client.fetch(label)
        spec = OFFICIAL_SPECS[label]
        path = _official_cache_path(cache_root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, artifact.content)
        _load_one_official(cache_root, spec)
        fetched.append(label)
    return {
        "status": "official_evidence_fetched" if fetched else "official_evidence_ready",
        "official_http_attempts": len(client.attempted_urls),
        "fetched": fetched,
        "network_accessed": bool(fetched),
        "would_write_datasets": False,
        "r2_accessed": False,
    }


def _read_archived_content(
    repository: LocalDatasetRepository,
    row: pd.Series,
) -> bytes:
    path = repository.root / _text(row.get("object_path"))
    if not path.is_file():
        raise FileNotFoundError(f"Archived source is missing: {path}")
    content = gzip.decompress(path.read_bytes())
    if sha256_bytes(content) != _text(row.get("source_hash")):
        raise ValueError(f"Archived source hash mismatch: {path}")
    return content


def load_frozen_aa_catalog(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
) -> FrozenCatalogSelection:
    rows = source_archive.loc[
        source_archive["dataset"].astype(str).eq("eodhd_exchange_symbols")
        & source_archive["source_url"].astype(str).str.contains(
            "delisted=0", regex=False
        )
    ]
    if rows.empty:
        raise RuntimeError("Frozen active EODHD exchange catalog is missing.")
    artifacts: dict[str, tuple[pd.Series, bytes]] = {}
    for _index, row in rows.iterrows():
        content = _read_archived_content(repository, row)
        artifacts[_text(row.get("source_hash"))] = (row, content)
    if len(artifacts) != 1:
        raise RuntimeError("Frozen active EODHD exchange catalog is ambiguous.")
    row, content = next(iter(artifacts.values()))
    value = json.loads(content)
    matches = [
        dict(item)
        for item in value
        if _text(item.get("Code")).upper() == AA_SYMBOL
    ]
    if len(matches) != 1 or matches[0] != EXPECTED_CATALOG_ROW:
        raise ValueError(
            "Frozen AA catalog identity changed: "
            + json.dumps(matches, sort_keys=True)
        )
    return FrozenCatalogSelection(
        row=matches[0],
        source_url=_text(row.get("source_url")),
        retrieved_at=_text(row.get("retrieved_at")),
        source_hash=_text(row.get("source_hash")),
        object_path=_text(row.get("object_path")),
    )


def _source_artifact(
    endpoint: str,
    rows: list[dict[str, Any]],
    retrieved_at: str,
) -> SourceArtifact:
    return SourceArtifact(
        source=f"eodhd_{endpoint}",
        source_url=REQUEST_URLS[endpoint],
        retrieved_at=retrieved_at,
        content=_canonical_json_bytes(rows),
        content_type="application/json",
    )


def _request_archive_artifact(artifact: SourceArtifact) -> SourceArtifact:
    """Bind raw response bytes to their safe request identity in the archive."""

    envelope = _canonical_json_bytes(
        {
            "schema": "supertrendquant_raw_request/v1",
            "source": artifact.source,
            "source_url": artifact.source_url,
            "content_type": artifact.content_type,
            "content_sha256": artifact.source_hash,
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
        }
    )
    return SourceArtifact(
        source=artifact.source,
        source_url=artifact.source_url,
        retrieved_at=artifact.retrieved_at,
        content=envelope,
        content_type=REQUEST_ENVELOPE_CONTENT_TYPE,
    )


def _price_records(
    rows: Any,
    artifact: SourceArtifact,
) -> pd.DataFrame:
    if not isinstance(rows, list):
        raise ValueError("AA EOD payload is not a list.")
    records = []
    for row in rows:
        session = _date(row.get("date"))
        if not session or row.get("close") is None:
            continue
        records.append(
            {
                "security_id": AA_SECURITY_ID,
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
    return pd.DataFrame(
        records,
        columns=tuple(
            dict.fromkeys(
                (*dataset_spec("daily_price_raw").required_columns, "source_url")
            )
        ),
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
        "event_id": _event_id(artifact.source, action_type, effective_date),
        "security_id": AA_SECURITY_ID,
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
        "metadata": None,
    }


def _action_records(
    endpoint: str,
    rows: Any,
    artifact: SourceArtifact,
) -> tuple[pd.DataFrame, int]:
    if not isinstance(rows, list):
        raise ValueError(f"AA {endpoint} payload is not a list.")
    records: list[dict[str, Any]] = []
    inception_pseudo = 0
    for row in rows:
        effective = _date(row.get("date"))
        if not effective:
            continue
        if effective < FETCH_START or effective > FETCH_END:
            raise ValueError(f"AA {endpoint} action is outside the frozen range.")
        if endpoint == "div":
            amount = row.get("unadjustedValue", row.get("value"))
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                raise ValueError(f"AA dividend amount is invalid on {effective}.")
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError(f"AA dividend amount is invalid on {effective}.")
            records.append(
                _provider_action(
                    artifact=artifact,
                    action_type="cash_dividend",
                    effective_date=effective,
                    cash_amount=amount,
                    announcement_date=_date(row.get("declarationDate")),
                    record_date=_date(row.get("recordDate")),
                    payment_date=_date(row.get("paymentDate")),
                    currency=_text(row.get("currency")) or "USD",
                )
            )
        else:
            ratio = _parse_split_ratio(row.get("split"))
            if ratio is None:
                raise ValueError(f"AA split ratio is invalid on {effective}.")
            # A newly issued spin company cannot legally split its own shares
            # on its first regular-way session.  Preserve the raw endpoint in
            # source_archive, but do not publish any inception pseudo-split.
            if effective == FETCH_START:
                inception_pseudo += 1
                continue
            records.append(
                _provider_action(
                    artifact=artifact,
                    action_type="split",
                    effective_date=effective,
                    ratio=ratio,
                )
            )
    return (
        pd.DataFrame(
            records,
            columns=tuple(
                dict.fromkeys(
                    (*dataset_spec("corporate_actions").required_columns, "metadata")
                )
            ),
        ),
        inception_pseudo,
    )


def _bundle_from_artifacts(
    artifacts: Iterable[SourceArtifact],
    *,
    http_attempts: int,
    budget_used_before: int | None = None,
    budget_used_after: int | None = None,
) -> FetchedBundle:
    by_endpoint = {
        artifact.source.removeprefix("eodhd_"): artifact
        for artifact in artifacts
    }
    if set(by_endpoint) != set(ENDPOINTS):
        raise ValueError("AA replay bundle does not contain the exact three endpoints.")
    parsed = {
        endpoint: json.loads(by_endpoint[endpoint].content)
        for endpoint in ENDPOINTS
    }
    prices = _price_records(parsed["eod"], by_endpoint["eod"])
    dividend_actions, dividend_pseudo = _action_records(
        "div", parsed["div"], by_endpoint["div"]
    )
    split_actions, split_pseudo = _action_records(
        "splits", parsed["splits"], by_endpoint["splits"]
    )
    actions = pd.concat(
        [dividend_actions, split_actions], ignore_index=True, sort=False
    )
    return FetchedBundle(
        prices=prices,
        corporate_actions=actions,
        artifacts=tuple(by_endpoint[endpoint] for endpoint in ENDPOINTS),
        http_attempts=int(http_attempts),
        inception_pseudo_split_rows=dividend_pseudo + split_pseudo,
        budget_used_before=budget_used_before,
        budget_used_after=budget_used_after,
    )


def validate_fetched_bundle(bundle: FetchedBundle) -> dict[str, Any]:
    if bundle.http_attempts != EXPECTED_EODHD_CALLS:
        raise ValueError(
            "AA bundle must record exactly three EODHD attempts: "
            f"actual={bundle.http_attempts}."
        )
    if (
        bundle.budget_used_before is not None
        and bundle.budget_used_after is not None
        and bundle.budget_used_after - bundle.budget_used_before
        != EXPECTED_EODHD_CALLS
    ):
        raise ValueError("AA bundle budget delta is not exactly three.")
    if tuple(bundle.prices["session"].astype(str)) != _expected_sessions():
        raise ValueError("AA EOD payload does not exactly cover 2,437 XNYS sessions.")
    if bundle.prices["security_id"].astype(str).ne(AA_SECURITY_ID).any():
        raise ValueError("AA bundle contains another security identity.")
    numeric = bundle.prices[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not numeric.map(lambda value: math.isfinite(float(value))).all().all():
        raise ValueError("AA EOD payload contains non-finite OHLCV.")
    invalid_ohlc = (
        numeric[["open", "high", "low", "close"]].le(0).any(axis=1)
        | numeric["volume"].lt(0)
        | numeric["high"].lt(numeric[["open", "low", "close"]].max(axis=1))
        | numeric["low"].gt(numeric[["open", "high", "close"]].min(axis=1))
    )
    if invalid_ohlc.any():
        raise ValueError("AA EOD payload contains invalid OHLCV relationships.")
    actions = bundle.corporate_actions
    if actions["event_id"].astype(str).duplicated().any():
        raise ValueError("AA provider actions contain duplicate event IDs.")
    if not actions.empty:
        if actions["security_id"].astype(str).ne(AA_SECURITY_ID).any():
            raise ValueError("AA provider actions contain another identity.")
        if (
            actions["effective_date"].astype(str).eq(FETCH_START)
            & actions["action_type"].astype(str).eq("split")
        ).any():
            raise ValueError("AA inception pseudo-split escaped the replay filter.")
    for artifact, endpoint in zip(bundle.artifacts, ENDPOINTS, strict=True):
        if artifact.source != f"eodhd_{endpoint}":
            raise ValueError("AA bundle artifact order changed.")
        if artifact.source_url != REQUEST_URLS[endpoint]:
            raise ValueError("AA bundle artifact URL changed.")
        if sha256_bytes(artifact.content) != artifact.source_hash:
            raise ValueError("AA bundle artifact hash is not self-consistent.")
    return {
        "aa_price_rows": len(bundle.prices),
        "aa_dividend_rows": int(
            actions["action_type"].astype(str).eq("cash_dividend").sum()
        ),
        "aa_split_rows": int(actions["action_type"].astype(str).eq("split").sum()),
        "aa_inception_pseudo_split_rows_archived_only": int(
            bundle.inception_pseudo_split_rows
        ),
        "expected_eodhd_calls": EXPECTED_EODHD_CALLS,
        "actual_eodhd_calls": bundle.http_attempts,
    }


def _bundle_signature(catalog: FrozenCatalogSelection) -> dict[str, Any]:
    return {
        "schema": "us_alcoa_spinoff_eodhd_bundle/v1",
        "security_id": AA_SECURITY_ID,
        "provider_symbol": PROVIDER_SYMBOL,
        "fetch_start": FETCH_START,
        "fetch_end": FETCH_END,
        "request_urls": REQUEST_URLS,
        "catalog_source_hash": catalog.source_hash,
        "expected_http_attempts": EXPECTED_EODHD_CALLS,
    }


def _bundle_cache_path(
    cache_root: Path,
    catalog: FrozenCatalogSelection,
) -> Path:
    digest = sha256_bytes(_canonical_json_bytes(_bundle_signature(catalog)))
    return cache_root / "state/us-alcoa-spinoff-repair" / f"{digest}.json.gz"


def _bundle_envelope(
    bundle: FetchedBundle,
    catalog: FrozenCatalogSelection,
) -> dict[str, Any]:
    return {
        "signature": _bundle_signature(catalog),
        "http_attempts": bundle.http_attempts,
        "inception_pseudo_split_rows": bundle.inception_pseudo_split_rows,
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


def _write_bundle_cache(
    path: Path,
    catalog: FrozenCatalogSelection,
    bundle: FetchedBundle,
) -> None:
    validate_fetched_bundle(bundle)
    content = _canonical_json_bytes(_bundle_envelope(bundle, catalog))
    encoded = gzip.compress(content, mtime=0)
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Conflicting immutable AA bundle cache: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, encoded)


def _read_bundle_cache(
    path: Path,
    catalog: FrozenCatalogSelection,
) -> FetchedBundle | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(gzip.decompress(path.read_bytes()))
    except Exception as exc:
        raise ValueError(f"AA bundle cache is unreadable: {path}") from exc
    if value.get("signature") != _bundle_signature(catalog):
        raise ValueError("AA bundle cache signature changed.")
    artifacts: list[SourceArtifact] = []
    for item in value.get("artifacts", []):
        content = base64.b64decode(item["content_base64"], validate=True)
        if sha256_bytes(content) != item.get("content_sha256"):
            raise ValueError("AA bundle cached artifact hash mismatch.")
        artifacts.append(
            SourceArtifact(
                source=str(item["source"]),
                source_url=str(item["source_url"]),
                retrieved_at=str(item["retrieved_at"]),
                content=content,
                content_type=str(item["content_type"]),
            )
        )
    bundle = _bundle_from_artifacts(
        artifacts,
        http_attempts=int(value.get("http_attempts", 0)),
        budget_used_before=value.get("budget_used_before"),
        budget_used_after=value.get("budget_used_after"),
    )
    if bundle.inception_pseudo_split_rows != int(
        value.get("inception_pseudo_split_rows", -1)
    ):
        raise ValueError("AA bundle pseudo-split inventory changed.")
    validate_fetched_bundle(bundle)
    return bundle


class ExactThreeAaEodhdClient(EodhdClient):
    """One attempt for each reviewed AA endpoint and never a fourth request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempted_endpoints: list[str] = []

    def get_json(self, endpoint: str, *, params: dict[str, object] | None = None):
        normalized = endpoint.strip("/")
        position = len(self.attempted_endpoints)
        if position >= EXPECTED_EODHD_CALLS:
            raise RuntimeError("AA client refused a fourth EODHD request.")
        expected = f"{ENDPOINTS[position]}/{PROVIDER_SYMBOL}"
        if normalized != expected or dict(params or {}) != REQUEST_PARAMS:
            raise RuntimeError(
                "AA client refused a non-reviewed request: "
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
    client: ExactThreeAaEodhdClient,
    *,
    budget_used_before: int,
) -> FetchedBundle:
    artifacts: list[SourceArtifact] = []
    retrieved_at = utc_now_iso()
    for endpoint in ENDPOINTS:
        rows = client.get_json(
            f"{endpoint}/{PROVIDER_SYMBOL}", params=REQUEST_PARAMS
        )
        artifacts.append(_source_artifact(endpoint, rows, retrieved_at))
    if len(client.attempted_endpoints) != EXPECTED_EODHD_CALLS:
        raise RuntimeError("AA collection did not make exactly three requests.")
    budget_after = _budget_used(client.budget)
    bundle = _bundle_from_artifacts(
        artifacts,
        http_attempts=len(client.attempted_endpoints),
        budget_used_before=budget_used_before,
        budget_used_after=budget_after,
    )
    validate_fetched_bundle(bundle)
    return bundle


def _read_release_frames(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, pd.DataFrame]:
    return {
        dataset: repository.read_frame(dataset, version)
        for dataset, version in release.dataset_versions.items()
    }


def _metadata(value: Mapping[str, Any]) -> str:
    return _canonical_json_bytes(dict(value)).decode("utf-8")


def _spinoff_metadata(
    template: Mapping[str, Any],
    artifact: SourceArtifact,
) -> str:
    value = dict(template)
    value["source_url"] = artifact.source_url
    value["source_hash"] = artifact.source_hash
    return _metadata(value)


def _float_equal(value: Any, expected: float, *, tolerance: float = 1e-12) -> bool:
    try:
        observed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(observed) and abs(observed - expected) <= tolerance


def _legal_reverse_split_row(actions: pd.DataFrame) -> pd.Series:
    rows = actions.loc[
        actions["event_id"].astype(str).eq(LEGAL_REVERSE_SPLIT_EVENT_ID)
    ]
    if len(rows) != 1:
        raise ValueError("The official 2016 Alcoa 1-for-3 reverse split is missing.")
    row = rows.iloc[0]
    if not (
        _text(row.get("security_id")) == HWM_SECURITY_ID
        and _text(row.get("action_type")) == "split"
        and _date(row.get("effective_date")) == LEGAL_REVERSE_SPLIT_DATE
        and _date(row.get("ex_date")) == LEGAL_REVERSE_SPLIT_DATE
        and _float_equal(row.get("ratio"), 1.0 / 3.0)
        and bool(row.get("official"))
    ):
        raise ValueError("The official 2016 Alcoa 1-for-3 split changed.")
    return row


def _pseudo_split_candidates(actions: pd.DataFrame) -> pd.DataFrame:
    ratio = pd.to_numeric(actions["ratio"], errors="coerce")
    return actions.loc[
        actions["event_id"].astype(str).eq(PSEUDO_SPLIT_EVENT_ID)
        | (
            actions["security_id"].astype(str).eq(HWM_SECURITY_ID)
            & actions["action_type"].astype(str).eq("split")
            & actions["effective_date"].astype(str).eq(PSEUDO_SPLIT_EFFECTIVE_DATE)
            & ratio.sub(PSEUDO_SPLIT_RATIO).abs().le(1e-12)
        )
    ]


def _validated_pseudo_split_mask(actions: pd.DataFrame) -> pd.Series:
    same_date = actions.loc[
        actions["security_id"].astype(str).eq(HWM_SECURITY_ID)
        & actions["action_type"].astype(str).eq("split")
        & actions["effective_date"].astype(str).eq(PSEUDO_SPLIT_EFFECTIVE_DATE)
    ]
    candidates = _pseudo_split_candidates(actions)
    if len(same_date) != 1 or len(candidates) != 1:
        raise ValueError(
            "The HWM 2016-11-01 pseudo-split is missing or ambiguous; refusing repair."
        )
    row = candidates.iloc[0]
    if not (
        _text(row.get("event_id")) == PSEUDO_SPLIT_EVENT_ID
        and _text(row.get("security_id")) == HWM_SECURITY_ID
        and _text(row.get("action_type")) == "split"
        and _date(row.get("effective_date")) == PSEUDO_SPLIT_EFFECTIVE_DATE
        and _date(row.get("ex_date")) == PSEUDO_SPLIT_EFFECTIVE_DATE
        and _float_equal(row.get("ratio"), PSEUDO_SPLIT_RATIO)
        and not bool(row.get("official"))
        and _text(row.get("source")) == "eodhd_splits"
        and _text(row.get("source_kind")) == "provider"
        and _text(row.get("source_url")) == PSEUDO_SPLIT_SOURCE_URL
        and _text(row.get("source_hash")) == PSEUDO_SPLIT_RAW_SHA256
    ):
        raise ValueError(
            "The HWM 2016-11-01 pseudo-split provenance changed; refusing repair."
        )
    return actions["event_id"].astype(str).eq(PSEUDO_SPLIT_EVENT_ID)


def _assert_no_pseudo_split(actions: pd.DataFrame) -> None:
    at_date = (
        actions["security_id"].astype(str).eq(HWM_SECURITY_ID)
        & actions["action_type"].astype(str).eq("split")
        & actions["effective_date"].astype(str).eq(PSEUDO_SPLIT_EFFECTIVE_DATE)
    )
    if at_date.any() or actions["event_id"].astype(str).eq(
        PSEUDO_SPLIT_EVENT_ID
    ).any():
        raise ValueError("The synthetic HWM 667/500 split remains installed.")


def _source_row_exact(row: pd.Series, *, url: str, source_hash: str) -> bool:
    return (
        _text(row.get("source_url")) == url
        and _text(row.get("source_hash")) == source_hash
        and bool(row.get("official"))
        and _text(row.get("source_kind")) == "official_filing"
    )


def _assert_base_spinoffs(actions: pd.DataFrame) -> None:
    rows_2016 = actions.loc[
        actions["event_id"].astype(str).eq(SPINOFF_2016_EVENT_ID)
    ]
    rows_2020 = actions.loc[
        actions["event_id"].astype(str).eq(SPINOFF_2020_EVENT_ID)
    ]
    if len(rows_2016) != 1 or len(rows_2020) != 1:
        raise ValueError("The two reviewed Alcoa lineage spin-offs are not unique.")
    first, second = rows_2016.iloc[0], rows_2020.iloc[0]
    if not (
        _text(first.get("security_id")) == HWM_SECURITY_ID
        and _text(first.get("action_type")) == "spinoff"
        and _date(first.get("effective_date")) == SPINOFF_2016_EFFECTIVE_DATE
        and _date(first.get("ex_date")) == SPINOFF_2016_EFFECTIVE_DATE
        and not _date(first.get("record_date"))
        and not _date(first.get("payment_date"))
        and _float_equal(first.get("ratio"), SPINOFF_2016_RATIO)
        and _text(first.get("currency")) == "USD"
        and pd.isna(first.get("cash_amount"))
        and _text(first.get("new_symbol")) == AA_SYMBOL
        and not _text(first.get("new_security_id"))
        and not _text(first.get("metadata"))
        and _source_row_exact(
            first,
            url=SPINOFF_2016_SOURCE_URL,
            source_hash=SPINOFF_2016_SOURCE_SHA256,
        )
    ):
        raise ValueError("The base 2016 Alcoa spin-off row changed.")
    if not (
        _text(second.get("security_id")) == HWM_SECURITY_ID
        and _text(second.get("action_type")) == "spinoff"
        and _date(second.get("effective_date")) == SPINOFF_2020_EFFECTIVE_DATE
        and _date(second.get("ex_date")) == SPINOFF_2020_EFFECTIVE_DATE
        and not _date(second.get("record_date"))
        and not _date(second.get("payment_date"))
        and _float_equal(second.get("ratio"), SPINOFF_2020_RATIO)
        and _text(second.get("currency")) == "USD"
        and pd.isna(second.get("cash_amount"))
        and _text(second.get("new_symbol")) == ARNC_SYMBOL
        and _text(second.get("new_security_id")) == ARNC_SECURITY_ID
        and not _text(second.get("metadata"))
        and _source_row_exact(
            second,
            url=SPINOFF_2020_SOURCE_URL,
            source_hash=SPINOFF_2020_SOURCE_SHA256,
        )
    ):
        raise ValueError("The base 2020 Arconic spin-off row changed.")


def _assert_base_lineage(frames: Mapping[str, pd.DataFrame]) -> None:
    master = frames["security_master"]
    hwm = master.loc[master["security_id"].astype(str).eq(HWM_SECURITY_ID)]
    arnc = master.loc[master["security_id"].astype(str).eq(ARNC_SECURITY_ID)]
    if len(hwm) != 1 or not (
        _text(hwm.iloc[0].get("primary_symbol")) == HWM_SYMBOL
        and _text(hwm.iloc[0].get("provider_symbol")) == "HWM.US"
        and _date(hwm.iloc[0].get("active_from")) == "2015-01-02"
        and not _date(hwm.iloc[0].get("active_to"))
    ):
        raise ValueError("The reviewed Old Alcoa/Howmet parent identity changed.")
    if len(arnc) != 1 or not (
        _text(arnc.iloc[0].get("primary_symbol")) == ARNC_SYMBOL
        and _text(arnc.iloc[0].get("provider_symbol")) == "ARNC.US"
        and _date(arnc.iloc[0].get("active_from")) == "2020-04-01"
        and _date(arnc.iloc[0].get("active_to")) == "2023-08-17"
    ):
        raise ValueError("The reviewed 2020 Arconic child identity changed.")
    history = frames["symbol_history"]
    hwm_history = {
        (_text(row.symbol), _date(row.effective_from), _date(row.effective_to))
        for row in history.loc[
            history["security_id"].astype(str).eq(HWM_SECURITY_ID)
        ].itertuples(index=False)
    }
    if hwm_history != {
        ("AA", "2015-01-01", "2016-10-31"),
        ("ARNC", "2016-11-01", "2020-03-31"),
        ("HWM", "2020-04-01", ""),
    }:
        raise ValueError("The reviewed Old Alcoa/Howmet symbol history changed.")
    arnc_history = history.loc[
        history["security_id"].astype(str).eq(ARNC_SECURITY_ID)
    ]
    if len(arnc_history) != 1 or not (
        _text(arnc_history.iloc[0].get("symbol")) == ARNC_SYMBOL
        and _date(arnc_history.iloc[0].get("effective_from")) == "2020-04-01"
        and _date(arnc_history.iloc[0].get("effective_to")) == "2023-08-17"
    ):
        raise ValueError("The reviewed 2020 Arconic symbol history changed.")
    prices = frames["daily_price_raw"]
    hwm_prices = prices.loc[
        prices["security_id"].astype(str).eq(HWM_SECURITY_ID)
    ]
    hwm_sessions = tuple(sorted(hwm_prices["session"].astype(str)))
    expected_hwm_sessions = tuple(
        pd.Timestamp(value).date().isoformat()
        for value in xcals.get_calendar("XNYS").sessions_in_range(
            "2015-01-02", FETCH_END
        )
    )
    if hwm_sessions != expected_hwm_sessions or len(hwm_sessions) != 2_899:
        raise ValueError("The reviewed Old Alcoa/Howmet price segment changed.")
    arnc_prices = prices.loc[
        prices["security_id"].astype(str).eq(ARNC_SECURITY_ID)
    ]
    sessions = tuple(sorted(arnc_prices["session"].astype(str)))
    expected_arnc_sessions = tuple(
        pd.Timestamp(value).date().isoformat()
        for value in xcals.get_calendar("XNYS").sessions_in_range(
            "2020-04-01", "2023-08-17"
        )
    )
    if sessions != expected_arnc_sessions or len(sessions) != 851:
        raise ValueError("The reviewed 2020 Arconic price segment changed.")
    _legal_reverse_split_row(frames["corporate_actions"])


def _aa_is_present(frames: Mapping[str, pd.DataFrame]) -> bool:
    for dataset in (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "adjustment_factors",
    ):
        frame = frames[dataset]
        if frame["security_id"].astype(str).eq(AA_SECURITY_ID).any():
            return True
    actions = frames["corporate_actions"]
    return bool(
        actions["security_id"].astype(str).eq(AA_SECURITY_ID).any()
        or actions["new_security_id"].astype(str).eq(AA_SECURITY_ID).any()
    )


def _looks_applied(frames: Mapping[str, pd.DataFrame]) -> bool:
    master = frames["security_master"]
    actions = frames["corporate_actions"]
    return bool(
        master["security_id"].astype(str).eq(AA_SECURITY_ID).any()
        and actions["event_id"].astype(str).eq(SPINOFF_2016_EVENT_ID).any()
        and not actions["event_id"].astype(str).eq(PSEUDO_SPLIT_EVENT_ID).any()
    )


def _updated_spinoff_rows(
    actions: pd.DataFrame,
    official: Mapping[str, SourceArtifact],
) -> pd.DataFrame:
    output = actions.copy()
    updates = {
        SPINOFF_2016_EVENT_ID: {
            "record_date": SPINOFF_2016_RECORD_DATE,
            "payment_date": SPINOFF_2016_PAYMENT_DATE,
            "new_security_id": AA_SECURITY_ID,
            "new_symbol": AA_SYMBOL,
            "metadata": _spinoff_metadata(COST_BASIS_2016, official["2016"]),
        },
        SPINOFF_2020_EVENT_ID: {
            "record_date": SPINOFF_2020_RECORD_DATE,
            "payment_date": SPINOFF_2020_PAYMENT_DATE,
            "new_security_id": ARNC_SECURITY_ID,
            "new_symbol": ARNC_SYMBOL,
            "metadata": _spinoff_metadata(COST_BASIS_2020, official["2020"]),
        },
    }
    for event_id, values in updates.items():
        mask = output["event_id"].astype(str).eq(event_id)
        if int(mask.sum()) != 1:
            raise ValueError(f"Spin-off action is not unique: {event_id}")
        for column, value in values.items():
            output.loc[mask, column] = value
    return output


def _append_aa_identity(
    frames: Mapping[str, pd.DataFrame],
    catalog: FrozenCatalogSelection,
    official_2016: SourceArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = frames["security_master"].copy()
    history = frames["symbol_history"].copy()
    if master["security_id"].astype(str).eq(AA_SECURITY_ID).any():
        raise ValueError("AA child identity already exists in an incomplete repair.")
    active_symbol_collision = master["primary_symbol"].astype(str).str.upper().eq(
        AA_SYMBOL
    ) & master["active_to"].map(_date).eq("")
    if active_symbol_collision.any():
        raise ValueError("Another active security owns the AA primary symbol.")
    master_row = {
        "security_id": AA_SECURITY_ID,
        "primary_symbol": AA_SYMBOL,
        "provider_symbol": PROVIDER_SYMBOL,
        "action_provider_symbol": PROVIDER_SYMBOL,
        "name": EXPECTED_CATALOG_ROW["Name"],
        "exchange": EXPECTED_CATALOG_ROW["Exchange"],
        "asset_type": "STOCK",
        "currency": EXPECTED_CATALOG_ROW["Currency"],
        "country": "US",
        "active_from": FETCH_START,
        "active_to": "",
        "isin": EXPECTED_CATALOG_ROW["Isin"],
        "source": "eodhd_exchange_symbols",
        "source_url": catalog.source_url,
        "retrieved_at": catalog.retrieved_at,
        "source_hash": catalog.source_hash,
    }
    history_row = {
        "security_id": AA_SECURITY_ID,
        "symbol": AA_SYMBOL,
        "exchange": EXPECTED_CATALOG_ROW["Exchange"],
        "effective_from": FETCH_START,
        "effective_to": "",
        "source": official_2016.source,
        "source_url": official_2016.source_url,
        "retrieved_at": official_2016.retrieved_at,
        "source_hash": official_2016.source_hash,
    }
    return (
        pd.concat([master, pd.DataFrame([master_row])], ignore_index=True, sort=False),
        pd.concat([history, pd.DataFrame([history_row])], ignore_index=True, sort=False),
    )


def _archive_extension(artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower().split(";", 1)[0]
    if "json" in content_type:
        return "json"
    if content_type == "application/pdf":
        return "pdf"
    if "html" in content_type:
        return "html"
    if content_type.startswith("text/"):
        return "txt"
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
            current = existing.iloc[0]
            if len(existing) != 1 or not (
                _text(current.get("source_url")) == artifact.source_url
                and _text(current.get("content_type")) == artifact.content_type
            ):
                raise ValueError(f"Archive ID collision: {artifact.source_hash}")
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
    affected = {HWM_SECURITY_ID, AA_SECURITY_ID}
    output = existing.loc[
        ~existing["security_id"].astype(str).isin(affected)
    ].copy()
    rebuilt: list[pd.DataFrame] = []
    for security_id in sorted(affected):
        security_prices = prices.loc[
            prices["security_id"].astype(str).eq(security_id)
        ].copy()
        security_actions = actions.loc[
            actions["security_id"].astype(str).eq(security_id)
        ].copy()
        if security_prices.empty:
            raise ValueError(f"Cannot rebuild factors without prices: {security_id}")
        rebuilt.append(
            build_adjustment_factors(
                security_prices,
                security_actions,
                source_version=source_version,
            )
        )
    return (
        pd.concat([output, *rebuilt], ignore_index=True, sort=False)
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


def _assert_exact_spinoff(
    actions: pd.DataFrame,
    *,
    event_id: str,
    effective_date: str,
    record_date: str,
    payment_date: str,
    ratio: float,
    child_security_id: str,
    child_symbol: str,
    source_url: str,
    source_hash: str,
    metadata: str,
) -> None:
    rows = actions.loc[actions["event_id"].astype(str).eq(event_id)]
    if len(rows) != 1:
        raise ValueError(f"Official spin-off action is not unique: {event_id}")
    row = rows.iloc[0]
    if not (
        _text(row.get("security_id")) == HWM_SECURITY_ID
        and _text(row.get("action_type")) == "spinoff"
        and _date(row.get("effective_date")) == effective_date
        and _date(row.get("ex_date")) == effective_date
        and _date(row.get("record_date")) == record_date
        and _date(row.get("payment_date")) == payment_date
        and _float_equal(row.get("ratio"), ratio)
        and _text(row.get("currency")) == "USD"
        and pd.isna(row.get("cash_amount"))
        and _text(row.get("new_security_id")) == child_security_id
        and _text(row.get("new_symbol")) == child_symbol
        and _source_row_exact(row, url=source_url, source_hash=source_hash)
        and _text(row.get("metadata")) == metadata
    ):
        raise ValueError(f"Official spin-off terms are not exact: {event_id}")


def _assert_archive_pair(
    archive: pd.DataFrame,
    *,
    source_url: str,
    source_hash: str | None = None,
) -> pd.Series:
    found = archive.loc[archive["source_url"].astype(str).eq(source_url)]
    if len(found) != 1:
        raise ValueError(f"Source archive lacks exact evidence: {source_url}")
    row = found.iloc[0]
    if source_hash is not None and _text(row.get("source_hash")) != source_hash:
        raise ValueError(f"Source archive evidence hash changed: {source_url}")
    if _text(row.get("archive_id")) != _text(row.get("source_hash")):
        raise ValueError(f"Source archive identity/hash mismatch: {source_url}")
    return row


def _verify_installed_archive_payload(
    repository: LocalDatasetRepository,
    row: pd.Series,
) -> bytes:
    path = repository.root / _text(row.get("object_path"))
    if not path.is_file():
        raise FileNotFoundError(f"Installed raw evidence payload is missing: {path}")
    try:
        content = gzip.decompress(path.read_bytes())
    except Exception as exc:
        raise ValueError(f"Installed raw evidence payload is corrupt: {path}") from exc
    if sha256_bytes(content) != _text(row.get("source_hash")):
        raise ValueError(f"Installed raw evidence payload hash mismatch: {path}")
    return content


def _bundle_from_installed_archive(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> FetchedBundle:
    artifacts: list[SourceArtifact] = []
    for endpoint in ENDPOINTS:
        row = _assert_archive_pair(
            archive,
            source_url=REQUEST_URLS[endpoint],
        )
        envelope_content = _verify_installed_archive_payload(repository, row)
        if _text(row.get("content_type")) != REQUEST_ENVELOPE_CONTENT_TYPE:
            raise ValueError(
                f"Installed AA raw request envelope type changed: {REQUEST_URLS[endpoint]}"
            )
        try:
            envelope = json.loads(envelope_content)
            raw_content = base64.b64decode(
                envelope["content_base64"], validate=True
            )
        except Exception as exc:
            raise ValueError(
                f"Installed AA raw request envelope is invalid: {REQUEST_URLS[endpoint]}"
            ) from exc
        if not (
            envelope.get("schema") == "supertrendquant_raw_request/v1"
            and envelope.get("source") == f"eodhd_{endpoint}"
            and envelope.get("source_url") == REQUEST_URLS[endpoint]
            and envelope.get("content_type") == "application/json"
            and envelope.get("content_sha256") == sha256_bytes(raw_content)
        ):
            raise ValueError(
                f"Installed AA raw request envelope changed: {REQUEST_URLS[endpoint]}"
            )
        artifacts.append(
            SourceArtifact(
                source=f"eodhd_{endpoint}",
                source_url=REQUEST_URLS[endpoint],
                retrieved_at=_text(row.get("retrieved_at")),
                content=raw_content,
                content_type="application/json",
            )
        )
    replay = _bundle_from_artifacts(artifacts, http_attempts=EXPECTED_EODHD_CALLS)
    validate_fetched_bundle(replay)
    return replay


def validate_candidate_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    catalog: FrozenCatalogSelection,
    bundle: FetchedBundle | None,
    official: Mapping[str, SourceArtifact] | None,
    completed_session: str,
    base_repository: LocalDatasetRepository | None = None,
    base_versions: Mapping[str, str] | None = None,
    verify_archive_files: bool = False,
) -> dict[str, Any]:
    _assert_base_lineage(frames)
    for dataset in WRITE_DATASETS:
        validate_dataset(
            dataset,
            frames[dataset],
            incomplete_action_policy="warn",
            completed_session=completed_session,
        ).raise_for_errors()

    archive = frames["source_archive"]
    if bundle is None and verify_archive_files:
        if base_repository is None:
            raise ValueError("Archive replay verification requires a repository.")
        bundle = _bundle_from_installed_archive(base_repository, archive)

    master = frames["security_master"]
    aa = master.loc[master["security_id"].astype(str).eq(AA_SECURITY_ID)]
    if len(aa) != 1:
        raise ValueError("AA security_master row is not unique.")
    aa_row = aa.iloc[0]
    if not (
        _text(aa_row.get("primary_symbol")) == AA_SYMBOL
        and _text(aa_row.get("provider_symbol")) == PROVIDER_SYMBOL
        and _text(aa_row.get("action_provider_symbol")) == PROVIDER_SYMBOL
        and _text(aa_row.get("name")) == EXPECTED_CATALOG_ROW["Name"]
        and _text(aa_row.get("exchange")) == EXPECTED_CATALOG_ROW["Exchange"]
        and _text(aa_row.get("currency")) == EXPECTED_CATALOG_ROW["Currency"]
        and _text(aa_row.get("country")) == "US"
        and _text(aa_row.get("isin")) == EXPECTED_CATALOG_ROW["Isin"]
        and _date(aa_row.get("active_from")) == FETCH_START
        and not _date(aa_row.get("active_to"))
        and _text(aa_row.get("source_hash")) == catalog.source_hash
    ):
        raise ValueError("AA security_master row is not exact.")
    history = frames["symbol_history"].loc[
        frames["symbol_history"]["security_id"].astype(str).eq(AA_SECURITY_ID)
    ]
    if len(history) != 1:
        raise ValueError("AA symbol_history row is not unique.")
    history_row = history.iloc[0]
    if not (
        _text(history_row.get("symbol")) == AA_SYMBOL
        and _text(history_row.get("exchange")) == "NYSE"
        and _date(history_row.get("effective_from")) == FETCH_START
        and not _date(history_row.get("effective_to"))
        and _text(history_row.get("source_url")) == TAX_BASIS_2016_URL
        and _text(history_row.get("source_hash")) == TAX_BASIS_2016_SHA256
    ):
        raise ValueError("AA symbol_history row is not exact.")

    prices_all = frames["daily_price_raw"]
    aa_prices = prices_all.loc[
        prices_all["security_id"].astype(str).eq(AA_SECURITY_ID)
    ].copy()
    observed_sessions = tuple(sorted(aa_prices["session"].astype(str)))
    if observed_sessions != _expected_sessions():
        raise ValueError("Installed AA prices do not cover the exact 2,437 sessions.")
    fetched_summary: dict[str, Any]
    if bundle is not None:
        fetched_summary = validate_fetched_bundle(bundle)
        expected = bundle.prices.sort_values("session").reset_index(drop=True)
        observed = aa_prices.sort_values("session").reset_index(drop=True)
        columns = (
            "security_id",
            "session",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "currency",
            "source_url",
            "source_hash",
        )
        try:
            pd.testing.assert_frame_equal(
                observed.loc[:, columns],
                expected.loc[:, columns],
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            raise ValueError("Installed AA prices differ from the replay bundle.") from exc
    else:
        fetched_summary = {
            "aa_price_rows": len(aa_prices),
            "aa_first_session": observed_sessions[0],
            "aa_last_session": observed_sessions[-1],
        }

    actions = frames["corporate_actions"]
    _assert_no_pseudo_split(actions)
    _legal_reverse_split_row(actions)
    expected_meta_2016 = _metadata(COST_BASIS_2016)
    expected_meta_2020 = _metadata(COST_BASIS_2020)
    _assert_exact_spinoff(
        actions,
        event_id=SPINOFF_2016_EVENT_ID,
        effective_date=SPINOFF_2016_EFFECTIVE_DATE,
        record_date=SPINOFF_2016_RECORD_DATE,
        payment_date=SPINOFF_2016_PAYMENT_DATE,
        ratio=SPINOFF_2016_RATIO,
        child_security_id=AA_SECURITY_ID,
        child_symbol=AA_SYMBOL,
        source_url=SPINOFF_2016_SOURCE_URL,
        source_hash=SPINOFF_2016_SOURCE_SHA256,
        metadata=expected_meta_2016,
    )
    _assert_exact_spinoff(
        actions,
        event_id=SPINOFF_2020_EVENT_ID,
        effective_date=SPINOFF_2020_EFFECTIVE_DATE,
        record_date=SPINOFF_2020_RECORD_DATE,
        payment_date=SPINOFF_2020_PAYMENT_DATE,
        ratio=SPINOFF_2020_RATIO,
        child_security_id=ARNC_SECURITY_ID,
        child_symbol=ARNC_SYMBOL,
        source_url=SPINOFF_2020_SOURCE_URL,
        source_hash=SPINOFF_2020_SOURCE_SHA256,
        metadata=expected_meta_2020,
    )
    aa_actions = actions.loc[
        actions["security_id"].astype(str).eq(AA_SECURITY_ID)
    ]
    inception_split = (
        aa_actions["action_type"].astype(str).eq("split")
        & aa_actions["effective_date"].astype(str).eq(FETCH_START)
    )
    if inception_split.any():
        raise ValueError("AA inception pseudo-split is installed.")
    if bundle is not None:
        expected_action_ids = set(bundle.corporate_actions["event_id"].astype(str))
        observed_action_ids = set(aa_actions["event_id"].astype(str))
        if observed_action_ids != expected_action_ids:
            raise ValueError("Installed AA provider actions differ from replay bundle.")

    factors = frames["adjustment_factors"]
    for security_id in (HWM_SECURITY_ID, AA_SECURITY_ID, ARNC_SECURITY_ID):
        price_sessions = set(
            prices_all.loc[
                prices_all["security_id"].astype(str).eq(security_id), "session"
            ].astype(str)
        )
        factor_rows = factors.loc[
            factors["security_id"].astype(str).eq(security_id)
        ]
        factor_sessions = set(
            pd.to_datetime(factor_rows["session"], errors="coerce")
            .dt.date.astype(str)
        )
        if factor_sessions != price_sessions:
            raise ValueError(f"Adjustment factors do not cover prices: {security_id}")
    hwm_factors = factors.loc[
        factors["security_id"].astype(str).eq(HWM_SECURITY_ID)
    ]
    early_hwm = hwm_factors.loc[
        pd.to_datetime(hwm_factors["session"], errors="coerce")
        < pd.Timestamp(LEGAL_REVERSE_SPLIT_DATE)
    ]
    if early_hwm.empty or not pd.to_numeric(
        early_hwm["split_factor"], errors="coerce"
    ).map(lambda value: _float_equal(value, 3.0)).all():
        raise ValueError("HWM factors still encode the removed 667/500 pseudo-split.")

    archive_rows: list[pd.Series] = []
    archive_rows.append(
        _assert_archive_pair(
            archive,
            source_url=SPINOFF_2016_SOURCE_URL,
            source_hash=SPINOFF_2016_SOURCE_SHA256,
        )
    )
    archive_rows.append(
        _assert_archive_pair(
            archive,
            source_url=SPINOFF_2020_SOURCE_URL,
            source_hash=SPINOFF_2020_SOURCE_SHA256,
        )
    )
    archive_rows.append(
        _assert_archive_pair(
            archive,
            source_url=TAX_BASIS_2016_URL,
            source_hash=TAX_BASIS_2016_SHA256,
        )
    )
    archive_rows.append(
        _assert_archive_pair(
            archive,
            source_url=TAX_BASIS_2020_URL,
            source_hash=TAX_BASIS_2020_SHA256,
        )
    )
    bundle_hashes: dict[str, str] = {}
    for endpoint in ENDPOINTS:
        source_hash = None
        if bundle is not None:
            raw_artifact = bundle.artifacts[ENDPOINTS.index(endpoint)]
            archived_artifact = _request_archive_artifact(raw_artifact)
            source_hash = archived_artifact.source_hash
            bundle_hashes[endpoint] = raw_artifact.source_hash
        archive_rows.append(
            _assert_archive_pair(
                archive,
                source_url=REQUEST_URLS[endpoint],
                source_hash=source_hash,
            )
        )
    if verify_archive_files:
        if base_repository is None:
            raise ValueError("Archive payload verification requires a repository.")
        for row in archive_rows:
            _verify_installed_archive_payload(base_repository, row)

    if base_repository is not None and base_versions is not None:
        candidate = _CandidateRepository(base_repository, base_versions, frames)
        validate_repository_snapshot(candidate).raise_for_errors()
    return {
        **fetched_summary,
        "aa_security_id": AA_SECURITY_ID,
        "hwm_security_id": HWM_SECURITY_ID,
        "arnc_security_id": ARNC_SECURITY_ID,
        "official_spinoff_rows": 2,
        "spinoff_2016_cost_basis_fraction": COST_BASIS_2016[
            "cost_basis_fraction"
        ],
        "spinoff_2020_cost_basis_fraction": COST_BASIS_2020[
            "cost_basis_fraction"
        ],
        "pseudo_split_rows": 0,
        "aa_raw_hashes": bundle_hashes,
        "catalog_calls": 0,
        "r2_accessed": False,
    }


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


def prepare_repair(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
    frames: Mapping[str, pd.DataFrame],
    catalog: FrozenCatalogSelection,
    bundle: FetchedBundle | None,
    official: Mapping[str, SourceArtifact] | None,
) -> PreparedRepair:
    if _looks_applied(frames):
        summary = validate_candidate_frames(
            frames,
            catalog=catalog,
            bundle=bundle,
            official=official,
            completed_session=release.completed_session,
            base_repository=repository,
            base_versions=release.dataset_versions,
            verify_archive_files=True,
        )
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=_capture_pointer_etags(repository, release),
            frames={dataset: frames[dataset].copy() for dataset in WRITE_DATASETS},
            archive_artifacts=(),
            warnings=tuple(release.warnings),
            summary={
                **summary,
                "status": "already_applied",
                "base_release_version": release.version,
                "network_accessed": False,
                "writes_performed": False,
                "r2_accessed": False,
            },
        )
    if bundle is None or official is None:
        raise RuntimeError("Offline replay requires the AA bundle and both official PDFs.")
    _assert_base_lineage(frames)
    _assert_base_spinoffs(frames["corporate_actions"])
    if _aa_is_present(frames):
        raise ValueError("AA repair state is partial; refusing to merge it.")
    validate_fetched_bundle(bundle)
    pseudo = _validated_pseudo_split_mask(frames["corporate_actions"])
    master, history = _append_aa_identity(frames, catalog, official["2016"])
    prices = pd.concat(
        [frames["daily_price_raw"], bundle.prices], ignore_index=True, sort=False
    ).drop_duplicates(
        list(dataset_spec("daily_price_raw").primary_key), keep="last"
    )
    base_actions = frames["corporate_actions"].loc[~pseudo].copy()
    actions = _updated_spinoff_rows(base_actions, official)
    collisions = set(actions["event_id"].astype(str)) & set(
        bundle.corporate_actions["event_id"].astype(str)
    )
    if collisions:
        raise ValueError(f"AA provider action ID collision: {sorted(collisions)}")
    actions = pd.concat(
        [actions, bundle.corporate_actions], ignore_index=True, sort=False
    ).drop_duplicates(
        list(dataset_spec("corporate_actions").primary_key), keep="last"
    )
    factors = _rebuild_factors(
        prices,
        actions,
        frames["adjustment_factors"],
        source_version=f"us-alcoa-spinoff-repair:{release.version}",
    )
    archive_artifacts = tuple(
        _request_archive_artifact(artifact) for artifact in bundle.artifacts
    ) + tuple(
        official[label] for label in OFFICIAL_SPECS
    )
    archive = _append_source_archive(
        frames["source_archive"],
        archive_artifacts,
        completed_session=release.completed_session,
    )
    rewritten = {
        "security_master": master.sort_values("security_id").reset_index(drop=True),
        "symbol_history": history.sort_values(
            ["security_id", "effective_from", "symbol"]
        ).reset_index(drop=True),
        "daily_price_raw": prices.sort_values(
            ["security_id", "session"]
        ).reset_index(drop=True),
        "corporate_actions": actions.sort_values(
            ["security_id", "effective_date", "event_id"]
        ).reset_index(drop=True),
        "adjustment_factors": factors,
        "source_archive": archive.sort_values("archive_id").reset_index(drop=True),
    }
    summary = validate_candidate_frames(
        rewritten,
        catalog=catalog,
        bundle=bundle,
        official=official,
        completed_session=release.completed_session,
        base_repository=repository,
        base_versions=release.dataset_versions,
    )
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=_capture_pointer_etags(repository, release),
        frames=rewritten,
        archive_artifacts=archive_artifacts,
        warnings=tuple(release.warnings),
        summary={
            **summary,
            "status": "validated_offline_plan",
            "base_release_version": release.version,
            "fake_split_rows_removed": int(pseudo.sum()),
            "network_accessed": False,
            "writes_performed": False,
            "r2_accessed": False,
        },
    )


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
        encoded = gzip.compress(artifact.content, mtime=0)
        if path.is_file():
            try:
                current = gzip.decompress(path.read_bytes())
            except Exception as exc:
                raise RuntimeError(f"Conflicting immutable archive payload: {path}") from exc
            if current != artifact.content:
                raise RuntimeError(f"Conflicting immutable archive payload: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(path, encoded)
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"Immutable archive verification failed: {path}")
        if sha256_bytes(gzip.decompress(path.read_bytes())) != artifact.source_hash:
            raise RuntimeError(f"Immutable archive hash verification failed: {path}")


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / f"recovery/{TRANSACTION_NAME}"
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved Alcoa repair recovery marker blocks writes.")
        transactions = repository.root / f"transactions/{TRANSACTION_NAME}"
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted Alcoa repair transaction blocks writes: {journal}"
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    write_atomic(
        path,
        json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, indent=2
        ).encode()
        + b"\n",
    )


def _assert_release_unchanged(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
) -> None:
    current, etag = repository.current_release()
    if current is None or current.version != release.version or etag != release_etag:
        raise RuntimeError(
            "Current release changed during Alcoa offline validation; rerun the plan."
        )


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
            repository.objects.put(
                release_key, old_release_bytes, if_match=current.etag
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
        raise RuntimeError("Committed Alcoa repair release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, _ = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"Committed Alcoa repair pointer mismatch: {dataset}")
    validate_repository_snapshot(repository).raise_for_errors()


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> dict[str, Any]:
    if prepared.summary.get("status") == "already_applied":
        _assert_release_unchanged(
            repository, prepared.release, prepared.release_etag
        )
        return {
            **prepared.summary,
            "mode": "apply",
            "status": "already_applied",
            "release_version": prepared.release.version,
            "dataset_writes_performed": False,
            "writes_performed": False,
            "network_accessed": False,
            "r2_accessed": False,
        }
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
                raise RuntimeError(
                    f"{dataset} pointer changed before Alcoa repair apply."
                )
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        planned = {
            dataset: (
                f"{TRANSACTION_NAME}-"
                f"{prepared.release.completed_session.replace('-', '')}-"
                f"{transaction_id}-{dataset}"
            )
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / f"transactions/{TRANSACTION_NAME}"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "us_alcoa_spinoff_repair_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "base_release_etag": prepared.release_etag,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value).decode("ascii")
                for dataset, value in old_pointers.items()
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
                        "operation": TRANSACTION_NAME,
                        "network_accessed": False,
                        "eodhd_calls_from_cache": EXPECTED_EODHD_CALLS,
                        "catalog_calls": 0,
                        "r2_accessed": False,
                        "base_release_version": prepared.release.version,
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
            warnings = tuple(
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
                quality=_quality_for_warnings(warnings),
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
                "old_release_version": prepared.release.version,
                "new_release_version": committed.version,
                "new_dataset_versions": dict(committed.dataset_versions),
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
                "transaction_id": transaction_id,
                "dataset_writes_performed": True,
                "writes_performed": True,
                "network_accessed": False,
                "r2_accessed": False,
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
                    / f"recovery/{TRANSACTION_NAME}"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "Alcoa rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}"
                ) from original
            raise


def _base_context(
    repository: LocalDatasetRepository,
) -> tuple[
    DataRelease,
    str | None,
    dict[str, pd.DataFrame],
    FrozenCatalogSelection,
]:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    if release.completed_session != FETCH_END:
        raise RuntimeError(
            "Alcoa repair is pinned to the reviewed completed session: "
            f"expected={FETCH_END}, current={release.completed_session}."
        )
    missing = sorted(set(WRITE_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    frames = _read_release_frames(repository, release)
    _assert_base_lineage(frames)
    if not _looks_applied(frames):
        _assert_base_spinoffs(frames["corporate_actions"])
        _validated_pseudo_split_mask(frames["corporate_actions"])
        if _aa_is_present(frames):
            raise ValueError("AA repair state is partial; refusing to continue.")
    catalog = load_frozen_aa_catalog(repository, frames["source_archive"])
    return release, release_etag, frames, catalog


def _official_cache_status(cache_root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, spec in OFFICIAL_SPECS.items():
        path = _official_cache_path(cache_root, spec)
        present = path.is_file()
        if present:
            _load_one_official(cache_root, spec)
        output[label] = {
            "path": str(path),
            "present": present,
            "source_url": spec["url"],
            "expected_sha256": spec["sha256"],
            "expected_size": spec["size"],
        }
    return output


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[
        [str | Path], LocalDatasetRepository
    ] = LocalDatasetRepository,
    budget_factory: Callable[[], EodhdCallBudget] = EodhdCallBudget,
    client_factory: Callable[..., ExactThreeAaEodhdClient] = ExactThreeAaEodhdClient,
    official_client_factory: Callable[..., ExactOfficialEvidenceClient] = (
        ExactOfficialEvidenceClient
    ),
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    release, release_etag, frames, catalog = _base_context(repository)
    mode = _text(getattr(args, "mode", "plan")) or "plan"
    bundle_path = _bundle_cache_path(repository.root, catalog)
    cached = _read_bundle_cache(bundle_path, catalog)
    official_status = _official_cache_status(repository.root)

    if _looks_applied(frames):
        official = (
            load_official_evidence(repository.root)
            if all(item["present"] for item in official_status.values())
            else None
        )
        prepared = prepare_repair(
            repository,
            release,
            release_etag,
            frames,
            catalog,
            cached,
            official,
        )
        if mode == "apply":
            return apply_repair(repository, prepared)
        return {
            **prepared.summary,
            "mode": mode,
            "cache_path": str(bundle_path),
            "would_write": False,
        }

    if mode == "plan":
        return {
            "status": (
                "ready_offline_replay"
                if cached is not None
                and all(item["present"] for item in official_status.values())
                else "blocked_cache_missing"
            ),
            "base_release_version": release.version,
            "base_release_etag": release_etag,
            "completed_session": release.completed_session,
            "bundle_cache_path": str(bundle_path),
            "bundle_cache_present": cached is not None,
            "official_evidence": official_status,
            "expected_eodhd_calls": EXPECTED_EODHD_CALLS,
            "request_order": list(ENDPOINTS),
            "request_urls": REQUEST_URLS,
            "catalog_calls": 0,
            "network_accessed": False,
            "would_write": False,
            "writes_performed": False,
            "r2_accessed": False,
        }

    if mode == "fetch_official_evidence":
        client = official_client_factory()
        result = fetch_missing_official_evidence(
            repository.root,
            client=client,
        )
        return {
            **result,
            "base_release_version": release.version,
            "bundle_cache_path": str(bundle_path),
            "expected_eodhd_calls": 0,
        }

    if mode == "fetch_missing":
        if cached is not None:
            return {
                **validate_fetched_bundle(cached),
                "status": "cache_already_complete",
                "base_release_version": release.version,
                "cache_path": str(bundle_path),
                "network_accessed": False,
                "would_write": False,
                "writes_performed": False,
                "r2_accessed": False,
            }
        budget = budget_factory()
        before = _budget_used(budget)
        if before + EXPECTED_EODHD_CALLS > budget.ceiling:
            raise RuntimeError(
                "EODHD budget cannot reserve the exact three-call AA bundle: "
                f"used={before}, ceiling={budget.ceiling}."
            )
        client = client_factory(budget=budget)
        bundle = fetch_exact_bundle(client, budget_used_before=before)
        _write_bundle_cache(bundle_path, catalog, bundle)
        replay = _read_bundle_cache(bundle_path, catalog)
        if replay is None:
            raise RuntimeError("AA bundle cache disappeared after write.")
        return {
            **validate_fetched_bundle(replay),
            "status": "fetched_cache_only",
            "base_release_version": release.version,
            "cache_path": str(bundle_path),
            "budget_used_before": replay.budget_used_before,
            "budget_used_after": replay.budget_used_after,
            "network_accessed": True,
            "would_write": False,
            "writes_performed": False,
            "r2_accessed": False,
        }

    if cached is None:
        raise RuntimeError(
            "AA immutable replay bundle is missing. Run the default plan, then "
            "explicitly use --fetch-missing once."
        )
    if not all(item["present"] for item in official_status.values()):
        raise RuntimeError(
            "Pinned Howmet tax-basis PDFs are missing. Explicitly use "
            "--fetch-official-evidence before offline replay/apply."
        )
    official = load_official_evidence(repository.root)
    prepared = prepare_repair(
        repository,
        release,
        release_etag,
        frames,
        catalog,
        cached,
        official,
    )
    if mode == "offline_plan":
        return {
            **prepared.summary,
            "mode": mode,
            "cache_path": str(bundle_path),
            "would_write": False,
            "writes_performed": False,
        }
    if mode == "apply":
        return apply_repair(repository, prepared)
    raise ValueError(f"Unsupported mode: {mode}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect exactly three AA EODHD endpoints and atomically install "
            "both audited Alcoa lineage spin-off cost-basis allocations."
        )
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fetch-official-evidence",
        action="store_const",
        const="fetch_official_evidence",
        dest="mode",
        help="Fetch only missing hash-pinned Howmet PDFs; make no EODHD calls.",
    )
    mode.add_argument(
        "--fetch-missing",
        action="store_const",
        const="fetch_missing",
        dest="mode",
        help="Make exactly three one-attempt AA EODHD calls into a local cache.",
    )
    mode.add_argument(
        "--offline-plan",
        action="store_const",
        const="offline_plan",
        dest="mode",
        help="Replay both caches with no network or dataset writes.",
    )
    mode.add_argument(
        "--apply",
        action="store_const",
        const="apply",
        dest="mode",
        help="Replay local caches and perform the CAS/journal transaction.",
    )
    parser.set_defaults(mode="plan")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
