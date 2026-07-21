#!/usr/bin/env python3
"""Repair the reused BBBY ticker without conflating two legal issuers.

The bootstrap snapshot currently owns two ``BBBY`` security ids, but both
price streams are Overstock/Beyond history.  This tool keeps the continuous
Overstock -> Beyond -> Bed Bath & Beyond issuer under the current id and
rebuilds the bankrupt legacy Bed Bath & Beyond issuer under the delisted id.

The network inventory is deliberately frozen and opt-in:

* EODHD ``BBBYQ.US``: eod, div and splits -- three one-shot attempts.
* A commit-pinned Quandl WIKI ``BBBY`` adjusted-price CSV on GitHub -- one
  one-shot attempt with immutable raw caching.
* Three SEC filings proving the old delisting and both current-issuer ticker
  transitions -- three one-shot attempts with immutable raw caching.

``--offline-plan`` constructs none of those clients.  A dry run validates a
complete candidate in memory; ``--apply`` is the only write mode.  Apply uses
one repository lock, compare-and-swap pointers and an explicit rollback
journal.  EODHD must contain every expected session and is the only permitted
primary source.  The pinned WIKI file is adjusted data, so it is used only as
an independent overlap check of intraday bar shape, volume, non-dividend
returns and action-aware dividend returns.  It can never fill a gap or become
the repaired price source.
"""

from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import gzip
import hashlib
import html
import io
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.index_membership import IndexEventReplayer
from supertrend_quant.market_store.ingest import EodhdClient, SourceArtifact
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

# These ids and catalog identities were audited against the frozen 2026-07-15
# EODHD active and delisted exchange-symbol archives.  The first id was
# created from BBBY_old.US and is retained as the canonical legacy issuer id;
# the second is the live BBBY.US listing and remains the current issuer id.
OLD_SECURITY_ID = "US:EODHD:dbd287da-b35f-5aaa-873c-76f941b4d93b"
CURRENT_SECURITY_ID = "US:EODHD:f7fefc0d-d331-5638-9934-09178df3981b"
OLD_ISIN = "US0758961009"
CURRENT_ISIN = "US6903701018"

OLD_START = "2015-01-02"
OLD_LAST_TRADING_DATE = "2023-05-02"
OSTK_TO_BYON = "2023-11-06"
BYON_TO_BBBY = "2025-08-29"

EODHD_CODE = "BBBYQ"
EODHD_ENDPOINTS = ("eod", "div", "splits")
MAX_EODHD_HTTP_ATTEMPTS = len(EODHD_ENDPOINTS)
WIKI_COMMIT = "0bcc715e2dd37b7ecec65c549be843574120bd58"
WIKI_URL = (
    "https://raw.githubusercontent.com/teddykoker/survivorship-free-spy/"
    f"{WIKI_COMMIT}/survivorship-free/data/BBBY.csv"
)
WIKI_SHA256 = "70af68cd8cf848525d3d8611ececf06c9f2f0a81a7ad2f10f0b1438fc3cf4328"
WIKI_FIRST_DATE = "2013-02-28"
WIKI_LAST_DATE = "2017-07-25"
WIKI_EXPECTED_ROWS = 1_110
MAX_WIKI_HTTP_ATTEMPTS = 1
# Consolidated daily volume is routinely restated differently by vendors.
# Keep price-shape and return checks fail-closed on every overlap session, but
# accept the independent volume cross-check only when at least 98% of all
# sessions agree within the row-level 2% tolerance.  The actual mismatch dates
# and maximum deviation remain part of the immutable validation summary.
MINIMUM_VOLUME_MATCH_RATIO = 0.98

OLD_DELISTING_URL = (
    "https://www.sec.gov/Archives/edgar/data/886158/"
    "000119312523115523/d89202d8k.htm"
)
OSTK_TO_BYON_URL = (
    "https://www.sec.gov/Archives/edgar/data/1130713/"
    "000113071323000074/ostk-20231024.htm"
)
BYON_TO_BBBY_URL = (
    "https://www.sec.gov/Archives/edgar/data/1130713/"
    "000114036125032363/ef20054350_8k.htm"
)
OFFICIAL_URLS = (OLD_DELISTING_URL, OSTK_TO_BYON_URL, BYON_TO_BBBY_URL)
MAX_OFFICIAL_HTTP_ATTEMPTS = len(OFFICIAL_URLS)
MAX_TOTAL_HTTP_ATTEMPTS = (
    MAX_EODHD_HTTP_ATTEMPTS
    + MAX_WIKI_HTTP_ATTEMPTS
    + MAX_OFFICIAL_HTTP_ATTEMPTS
)
if MAX_TOTAL_HTTP_ATTEMPTS != 7:
    raise RuntimeError("BBBY repair HTTP inventory changed without a cap audit.")

ACTIVE_CATALOG_URL = "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
DELISTED_CATALOG_URL = "https://eodhd.com/api/exchange-symbol-list/US?delisted=1"

WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "index_constituent_anchors",
    "index_membership_events",
    "source_archive",
)

EXPECTED_REMOVALS = {
    "nasdaq100": "2016-12-19",
    "sp500": "2017-07-26",
}

@dataclass(frozen=True)
class CatalogProof:
    active_row: dict[str, Any]
    old_row: dict[str, Any]
    bbbyq_row: dict[str, Any]
    active_archive_hash: str
    delisted_archive_hash: str


@dataclass(frozen=True)
class LocalPreflight:
    existing: dict[str, pd.DataFrame]
    pointer_etags: dict[str, str | None]
    catalog: CatalogProof
    contaminated_price_hashes: frozenset[str]
    contamination_overlap_rows: int
    already_repaired: bool = False


@dataclass(frozen=True)
class PriceSourceBundle:
    prices: pd.DataFrame
    actions: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    http_attempts: int


@dataclass(frozen=True)
class OfficialEvidenceBundle:
    artifacts: tuple[SourceArtifact, ...]

    def artifact(self, url: str) -> SourceArtifact:
        matches = [item for item in self.artifacts if item.source_url == url]
        if len(matches) != 1:
            raise ValueError(f"Official BBBY evidence is not unique: {url}")
        return matches[0]


@dataclass
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    archive_artifacts: tuple[SourceArtifact, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair the legacy/current BBBY issuer collision."
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument(
        "--fetch-eodhd-bbbyq",
        action="store_true",
        help="Allow at most three one-shot BBBYQ EODHD requests.",
    )
    parser.add_argument(
        "--fetch-wiki",
        action="store_true",
        help="Allow the one missing immutable commit-pinned WIKI BBBY request.",
    )
    parser.add_argument(
        "--fetch-official-evidence",
        action="store_true",
        help="Allow one request for each missing SEC filing (maximum three).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--offline-plan", action="store_true")
    return parser.parse_args(argv)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _expected_sessions(start: str, end: str) -> tuple[str, ...]:
    import exchange_calendars as xcals

    values = xcals.get_calendar("XNYS").sessions_in_range(start, end)
    return tuple(pd.Timestamp(value).date().isoformat() for value in values)


def _concat_unique(
    frames: Iterable[pd.DataFrame], *, keys: tuple[str, ...]
) -> pd.DataFrame:
    values = [frame for frame in frames if frame is not None and not frame.empty]
    if not values:
        columns: list[str] = []
        for frame in frames:
            columns.extend(str(column) for column in frame.columns)
        return pd.DataFrame(columns=tuple(dict.fromkeys(columns)))
    output = pd.concat(values, ignore_index=True, sort=False)
    return output.drop_duplicates(list(keys), keep="last").reset_index(drop=True)


def _safe_archive_payload(
    repository: LocalDatasetRepository,
    archive_row: pd.Series,
) -> bytes:
    root = repository.root.resolve()
    path = (root / str(archive_row["object_path"])).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Catalog archive path escapes repository: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Catalog archive payload is missing: {path}")
    encoded = path.read_bytes()
    try:
        content = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    except Exception as exc:
        raise ValueError(f"Catalog archive payload is unreadable: {path}") from exc
    expected = str(archive_row["source_hash"])
    if expected != str(archive_row["archive_id"]) or sha256_bytes(content) != expected:
        raise ValueError(f"Catalog archive hash mismatch: {path}")
    return content


def _catalog_rows(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    source_url: str,
) -> tuple[list[dict[str, Any]], str]:
    matches = archive.loc[archive["source_url"].astype(str).eq(source_url)]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen catalog archive for {source_url}")
    row = matches.iloc[0]
    try:
        parsed = json.loads(_safe_archive_payload(repository, row))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Frozen catalog is invalid JSON: {source_url}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError(f"Frozen catalog has the wrong shape: {source_url}")
    return parsed, str(row["source_hash"])


def _one_catalog_row(
    rows: Iterable[dict[str, Any]], *, code: str, isin: str
) -> dict[str, Any]:
    matches = [
        item
        for item in rows
        if str(item.get("Code", "")).upper() == code.upper()
        and str(item.get("Isin", "")).upper() == isin.upper()
    ]
    if len(matches) != 1:
        raise ValueError(f"Frozen catalog identity is not unique: {code}/{isin}")
    return dict(matches[0])


def load_catalog_proof(
    repository: LocalDatasetRepository, archive: pd.DataFrame
) -> CatalogProof:
    active, active_hash = _catalog_rows(repository, archive, ACTIVE_CATALOG_URL)
    delisted, delisted_hash = _catalog_rows(
        repository, archive, DELISTED_CATALOG_URL
    )
    active_row = _one_catalog_row(active, code="BBBY", isin=CURRENT_ISIN)
    old_row = _one_catalog_row(delisted, code="BBBY_old", isin=OLD_ISIN)
    bbbyq_row = _one_catalog_row(delisted, code="BBBYQ", isin=OLD_ISIN)
    # These historical aliases prove that the current issuer is the OSTK/BYON
    # lineage, not the old bankrupt Bed Bath issuer.
    _one_catalog_row(delisted, code="OSTK", isin=CURRENT_ISIN)
    _one_catalog_row(delisted, code="BYON", isin=CURRENT_ISIN)
    return CatalogProof(
        active_row=active_row,
        old_row=old_row,
        bbbyq_row=bbbyq_row,
        active_archive_hash=active_hash,
        delisted_archive_hash=delisted_hash,
    )


def _is_forbidden_old_eodhd_url(value: Any) -> bool:
    parsed = urlparse(str(value or ""))
    if (parsed.hostname or "").lower() != "eodhd.com":
        return False
    path = parsed.path.lower()
    return path.endswith("/bbby.us") or path.endswith("/bbby_old.us")


def _history_is_repaired(history: pd.DataFrame) -> bool:
    rows = history.loc[
        history["security_id"].astype(str).isin(
            (OLD_SECURITY_ID, CURRENT_SECURITY_ID)
        )
    ].copy()
    actual = {
        (
            str(row.security_id),
            str(row.symbol),
            str(row.exchange),
            str(row.effective_from),
            str(row.effective_to or ""),
        )
        for row in rows.itertuples(index=False)
    }
    expected = {
        (OLD_SECURITY_ID, "BBBY", "NASDAQ", "2015-01-01", OLD_LAST_TRADING_DATE),
        (CURRENT_SECURITY_ID, "OSTK", "NASDAQ", "2015-01-01", "2023-11-05"),
        (CURRENT_SECURITY_ID, "BYON", "NYSE", OSTK_TO_BYON, "2025-08-28"),
        (CURRENT_SECURITY_ID, "BBBY", "NYSE", BYON_TO_BBBY, ""),
    }
    return actual == expected


def _snapshot_is_repaired(existing: dict[str, pd.DataFrame]) -> bool:
    prices = existing["daily_price_raw"]
    old = prices.loc[prices["security_id"].astype(str).eq(OLD_SECURITY_ID)].copy()
    if old.empty:
        return False
    sessions = pd.to_datetime(old["session"], errors="coerce")
    if sessions.isna().any() or sessions.max() != pd.Timestamp(OLD_LAST_TRADING_DATE):
        return False
    if "source_url" in old and old["source_url"].map(_is_forbidden_old_eodhd_url).any():
        return False
    if not _history_is_repaired(existing["symbol_history"]):
        return False
    for dataset in ("index_constituent_anchors", "index_membership_events"):
        frame = existing[dataset]
        date_column = "anchor_date" if dataset.endswith("anchors") else "effective_date"
        early = pd.to_datetime(frame[date_column], errors="coerce").le(
            pd.Timestamp(OLD_LAST_TRADING_DATE)
        )
        if (
            early
            & frame["security_id"].astype(str).eq(CURRENT_SECURITY_ID)
        ).any():
            return False
    return True


def _contamination_overlap(old: pd.DataFrame, current: pd.DataFrame) -> int:
    columns = ["session", "open", "high", "low", "close"]
    left = old.loc[:, columns].copy()
    right = current.loc[:, columns].copy()
    joined = left.merge(right, on="session", suffixes=("_old", "_current"))
    if len(joined) < 3:
        raise ValueError("The two BBBY ids lack enough overlap to prove contamination.")
    equal = np.ones(len(joined), dtype=bool)
    for column in ("open", "high", "low", "close"):
        equal &= np.isclose(
            pd.to_numeric(joined[f"{column}_old"]),
            pd.to_numeric(joined[f"{column}_current"]),
            rtol=0.001,
            atol=0.01,
            equal_nan=False,
        )
    ratio = float(equal.mean())
    if ratio < 0.98:
        raise ValueError(
            "The frozen BBBY rows no longer match the audited OSTK contamination "
            f"fingerprint: equal_ratio={ratio:.6f}."
        )
    return int(equal.sum())


def _capture_pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        expected = release.dataset_versions.get(dataset)
        if pointer is None or pointer.version != expected:
            raise RuntimeError(f"Release/pointer mismatch for {dataset}.")
        result[dataset] = etag
    return result


def build_local_preflight(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> LocalPreflight:
    missing = [name for name in WRITE_DATASETS if name not in release.dataset_versions]
    if missing:
        raise RuntimeError("BBBY repair datasets are missing: " + ", ".join(missing))
    existing = {
        name: repository.read_frame(name, release.dataset_versions[name])
        for name in WRITE_DATASETS
    }
    master = existing["security_master"]
    for security_id in (OLD_SECURITY_ID, CURRENT_SECURITY_ID):
        matches = master.loc[master["security_id"].astype(str).eq(security_id)]
        if len(matches) != 1:
            raise ValueError(f"Expected one BBBY master row: {security_id}")
    catalog = load_catalog_proof(repository, existing["source_archive"])
    if _snapshot_is_repaired(existing):
        return LocalPreflight(
            existing=existing,
            pointer_etags=_capture_pointer_etags(repository, release),
            catalog=catalog,
            contaminated_price_hashes=frozenset(),
            contamination_overlap_rows=0,
            already_repaired=True,
        )

    prices = existing["daily_price_raw"]
    old = prices.loc[prices["security_id"].astype(str).eq(OLD_SECURITY_ID)].copy()
    current = prices.loc[
        prices["security_id"].astype(str).eq(CURRENT_SECURITY_ID)
    ].copy()
    if old.empty or current.empty:
        raise ValueError("Both contaminated BBBY price streams must exist before repair.")
    if pd.to_datetime(old["session"]).max() <= pd.Timestamp(OLD_LAST_TRADING_DATE):
        raise ValueError("Legacy BBBY stream no longer has the audited post-delisting leak.")
    overlap = _contamination_overlap(old, current)
    hashes = frozenset(old["source_hash"].dropna().astype(str))
    if not hashes:
        raise ValueError("Legacy contaminated BBBY rows lack provenance hashes.")
    return LocalPreflight(
        existing=existing,
        pointer_etags=_capture_pointer_etags(repository, release),
        catalog=catalog,
        contaminated_price_hashes=hashes,
        contamination_overlap_rows=overlap,
    )


class CappedSingleAttemptEodhdClient(EodhdClient):
    """One attempt per endpoint and a three-request run-wide hard cap."""

    def __init__(self, *args, max_attempts: int = MAX_EODHD_HTTP_ATTEMPTS, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_attempts = int(max_attempts)
        self._attempt_count = 0
        self._lock = threading.Lock()

    @property
    def attempt_count(self) -> int:
        with self._lock:
            return self._attempt_count

    def get_json(self, endpoint: str, *, params: dict[str, object] | None = None):
        safe_endpoint = "/" + endpoint.strip("/")
        query = {**(params or {}), "api_token": self.token, "fmt": "json"}
        with self._lock:
            if self._attempt_count >= self.max_attempts:
                raise RuntimeError("BBBYQ EODHD request cap reached before HTTP.")
            self.budget.claim()
            self._attempt_count += 1
        try:
            response = self.session.get(
                self.base_url + safe_endpoint, params=query, timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f"HTTP {status}" if status else type(exc).__name__
            raise RuntimeError(
                f"BBBYQ EODHD single attempt failed for {safe_endpoint}: {detail}"
            ) from None


def _eodhd_params() -> dict[str, str]:
    return {"from": "2015-01-01", "to": OLD_LAST_TRADING_DATE}


def _eodhd_url(client: Any, endpoint: str) -> str:
    return client.safe_url(f"{endpoint}/{EODHD_CODE}.US", params=_eodhd_params())


class EodhdBbbyqSource:
    """Three immutable endpoint caches; partial/no-data responses are evidence."""

    def __init__(
        self,
        root: Path,
        *,
        allow_http: bool,
        client_factory: Callable[[], Any] = CappedSingleAttemptEodhdClient,
    ):
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.client_factory = client_factory
        self.http_attempts = 0

    @staticmethod
    def _public_url(endpoint: str) -> str:
        from urllib.parse import urlencode

        return (
            f"https://eodhd.com/api/{endpoint}/{EODHD_CODE}.US?"
            + urlencode(_eodhd_params())
        )

    def path(self, endpoint: str) -> Path:
        return self.root / f"{sha256_bytes(self._public_url(endpoint).encode())}.json.gz"

    def _decode(self, endpoint: str, payload: bytes) -> SourceArtifact:
        path = self.path(endpoint)
        try:
            envelope = json.loads(gzip.decompress(payload))
            content = base64.b64decode(envelope["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError(f"Unreadable BBBYQ endpoint cache: {path}") from exc
        expected_url = self._public_url(endpoint)
        if envelope.get("schema") != "bbbyq_eodhd_raw/v1":
            raise ValueError("BBBYQ endpoint cache schema mismatch.")
        if envelope.get("endpoint") != endpoint or envelope.get("source_url") != expected_url:
            raise ValueError("BBBYQ endpoint cache identity mismatch.")
        if envelope.get("source_hash") != sha256_bytes(content):
            raise ValueError("BBBYQ endpoint cache hash mismatch.")
        return SourceArtifact(
            source=f"eodhd_{endpoint}",
            source_url=expected_url,
            retrieved_at=str(envelope["retrieved_at"]),
            content=content,
            content_type="application/json",
        )

    def get(self, endpoint: str) -> SourceArtifact | None:
        path = self.path(endpoint)
        return self._decode(endpoint, path.read_bytes()) if path.is_file() else None

    def _store(self, endpoint: str, artifact: SourceArtifact) -> SourceArtifact:
        envelope = {
            "schema": "bbbyq_eodhd_raw/v1",
            "endpoint": endpoint,
            "source_url": artifact.source_url,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
        }
        path = self.path(endpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(_canonical_json_bytes(envelope), mtime=0)
        if path.is_file():
            existing = self._decode(endpoint, path.read_bytes())
            if existing.content != artifact.content:
                raise RuntimeError(f"Immutable BBBYQ cache changed: {path}")
            return existing
        write_atomic(path, encoded)
        return self._decode(endpoint, path.read_bytes())

    def fetch(self) -> PriceSourceBundle:
        cached = {endpoint: self.get(endpoint) for endpoint in EODHD_ENDPOINTS}
        missing = [endpoint for endpoint, artifact in cached.items() if artifact is None]
        if missing and not self.allow_http:
            raise FileNotFoundError(
                "BBBYQ endpoint cache is incomplete; explicitly allow EODHD fetch: "
                + ", ".join(missing)
            )
        if len(missing) > MAX_EODHD_HTTP_ATTEMPTS:
            raise RuntimeError("BBBYQ missing request set exceeds the frozen cap.")
        client = self.client_factory() if missing else None
        for endpoint in missing:
            rows = client.get_json(
                f"{endpoint}/{EODHD_CODE}.US", params=_eodhd_params()
            )
            artifact = SourceArtifact(
                source=f"eodhd_{endpoint}",
                source_url=_eodhd_url(client, endpoint),
                retrieved_at=utc_now_iso(),
                content=_canonical_json_bytes(rows),
                content_type="application/json",
            )
            # A custom test client may format query ordering differently.  The
            # immutable cache key is the canonical token-free public URL.
            artifact = SourceArtifact(
                source=artifact.source,
                source_url=self._public_url(endpoint),
                retrieved_at=artifact.retrieved_at,
                content=artifact.content,
                content_type=artifact.content_type,
            )
            cached[endpoint] = self._store(endpoint, artifact)
        self.http_attempts = int(getattr(client, "attempt_count", len(missing))) if client else 0
        if self.http_attempts > MAX_EODHD_HTTP_ATTEMPTS:
            raise RuntimeError("BBBYQ EODHD source exceeded its hard cap.")
        artifacts = tuple(cached[endpoint] for endpoint in EODHD_ENDPOINTS)
        prices = _eodhd_price_frame(artifacts[0])
        actions = _eodhd_action_frame(artifacts[1], artifacts[2])
        return PriceSourceBundle(prices, actions, artifacts, self.http_attempts)


def _artifact_json_rows(artifact: SourceArtifact) -> list[dict[str, Any]]:
    try:
        value = json.loads(artifact.content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Provider artifact is invalid JSON: {artifact.source_url}") from exc
    if value in (None, {}, []):
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Provider artifact has unexpected payload: {artifact.source_url}")
    return value


def _eodhd_price_frame(artifact: SourceArtifact) -> pd.DataFrame:
    rows = []
    for item in _artifact_json_rows(artifact):
        if not item.get("date") or item.get("close") is None:
            continue
        rows.append(
            {
                "security_id": OLD_SECURITY_ID,
                "session": str(item["date"]),
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "close": item.get("close"),
                "volume": item.get("volume", 0),
                "currency": "USD",
                "source": "eodhd_eod",
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    columns = tuple(dict.fromkeys((*dataset_spec("daily_price_raw").required_columns, "source_url")))
    return pd.DataFrame(rows, columns=columns)


def _parse_split(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    for separator in ("/", ":"):
        if separator in text:
            left, right = text.split(separator, 1)
            try:
                denominator = float(right)
                return float(left) / denominator if denominator else None
            except ValueError:
                return None
    try:
        return float(text)
    except ValueError:
        return None


def _action_id(*values: Any) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode()).hexdigest()


def _eodhd_action_frame(
    dividends: SourceArtifact, splits: SourceArtifact
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for endpoint, artifact in (("div", dividends), ("splits", splits)):
        for item in _artifact_json_rows(artifact):
            effective = str(item.get("date") or "")
            if not effective:
                continue
            if endpoint == "div":
                action_type = "cash_dividend"
                cash = item.get("unadjustedValue", item.get("value"))
                ratio = None
            else:
                action_type = "split"
                cash = None
                ratio = _parse_split(item.get("split"))
                if ratio is None:
                    continue
            rows.append(
                {
                    "event_id": _action_id(
                        artifact.source,
                        OLD_SECURITY_ID,
                        action_type,
                        effective,
                        cash,
                        ratio,
                    ),
                    "security_id": OLD_SECURITY_ID,
                    "action_type": action_type,
                    "effective_date": effective,
                    "ex_date": effective,
                    "announcement_date": str(item.get("declarationDate") or ""),
                    "record_date": str(item.get("recordDate") or ""),
                    "payment_date": str(item.get("paymentDate") or ""),
                    "cash_amount": cash,
                    "ratio": ratio,
                    "currency": str(item.get("currency") or "USD"),
                    "new_security_id": "",
                    "new_symbol": "",
                    "official": False,
                    "source_url": artifact.source_url,
                    "source_kind": "provider",
                    "source": artifact.source,
                    "retrieved_at": artifact.retrieved_at,
                    "source_hash": artifact.source_hash,
                }
            )
    return pd.DataFrame(rows, columns=dataset_spec("corporate_actions").required_columns)


def _validate_wiki_artifact_identity(artifact: SourceArtifact) -> None:
    if artifact.source != "quandl_wiki_adjusted_git_csv":
        raise ValueError("Pinned WIKI BBBY artifact source label is wrong.")
    if artifact.source_url != WIKI_URL:
        raise ValueError("Pinned WIKI BBBY artifact URL is wrong.")
    if artifact.source_hash != WIKI_SHA256:
        raise ValueError(
            "Pinned WIKI BBBY artifact hash differs from the audited commit file."
        )


def _wiki_price_frame(artifact: SourceArtifact) -> pd.DataFrame:
    """Parse the commit-pinned adjusted WIKI file as validation-only OHLCV."""

    _validate_wiki_artifact_identity(artifact)
    try:
        text = artifact.content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Pinned WIKI BBBY response is not UTF-8 CSV.") from exc
    reader = csv.DictReader(io.StringIO(text))
    expected_header = ("date", "open", "high", "low", "close", "volume")
    if tuple(reader.fieldnames or ()) != expected_header:
        raise ValueError(
            "Pinned WIKI BBBY CSV header changed: "
            f"expected={expected_header}, actual={tuple(reader.fieldnames or ())}."
        )
    rows = list(reader)
    if len(rows) != WIKI_EXPECTED_ROWS:
        raise ValueError(
            "Pinned WIKI BBBY CSV row count changed: "
            f"expected={WIKI_EXPECTED_ROWS}, actual={len(rows)}."
        )
    frame = pd.DataFrame(rows).rename(columns={"date": "session"})
    frame["session"] = pd.to_datetime(frame["session"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["session", "open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("Pinned WIKI BBBY CSV contains invalid values.")
    if frame["session"].duplicated().any():
        raise ValueError("Pinned WIKI BBBY CSV contains duplicate sessions.")
    numeric = frame[["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=float
    )
    if not bool(np.isfinite(numeric).all()):
        raise ValueError("Pinned WIKI BBBY CSV contains non-finite values.")
    positive = frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    coherent = frame["high"].ge(frame[["open", "close"]].max(axis=1)) & frame[
        "low"
    ].le(frame[["open", "close"]].min(axis=1))
    if not bool((positive & coherent & frame["volume"].ge(0)).all()):
        raise ValueError("Pinned WIKI BBBY CSV contains invalid adjusted OHLCV bars.")
    frame = frame.sort_values("session", kind="stable").reset_index(drop=True)
    frame["session"] = frame["session"].dt.date.astype(str)
    if (
        frame.iloc[0]["session"] != WIKI_FIRST_DATE
        or frame.iloc[-1]["session"] != WIKI_LAST_DATE
    ):
        raise ValueError(
            "Pinned WIKI BBBY CSV date boundary changed: "
            f"actual={frame.iloc[0]['session']}..{frame.iloc[-1]['session']}."
        )
    _sessions_exact(
        frame,
        _expected_sessions(WIKI_FIRST_DATE, WIKI_LAST_DATE),
        "Pinned WIKI BBBY adjusted history",
    )
    frame["security_id"] = OLD_SECURITY_ID
    frame["currency"] = "USD"
    frame["source"] = artifact.source
    frame["source_url"] = artifact.source_url
    frame["retrieved_at"] = artifact.retrieved_at
    frame["source_hash"] = artifact.source_hash
    columns = tuple(
        dict.fromkeys((*dataset_spec("daily_price_raw").required_columns, "source_url"))
    )
    return frame.loc[:, columns].reset_index(drop=True)


class PinnedWikiBbbySource:
    """One-attempt immutable cache for the audited adjusted WIKI BBBY CSV."""

    def __init__(
        self,
        root: Path,
        *,
        allow_http: bool,
        opener: Callable[..., Any] | None = None,
    ):
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.opener = opener or urlopen
        self.http_attempts = 0

    def path(self) -> Path:
        return self.root / f"{sha256_bytes(WIKI_URL.encode())}.json.gz"

    def _decode(self, payload: bytes) -> SourceArtifact:
        try:
            envelope = json.loads(gzip.decompress(payload))
            content = base64.b64decode(envelope["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError(f"Unreadable pinned WIKI BBBY cache: {self.path()}") from exc
        if envelope.get("schema") != "bbby_wiki_git_raw/v1":
            raise ValueError("Pinned WIKI BBBY cache schema mismatch.")
        if envelope.get("source_url") != WIKI_URL:
            raise ValueError("Pinned WIKI BBBY cache URL mismatch.")
        if envelope.get("source_hash") != sha256_bytes(content):
            raise ValueError("Pinned WIKI BBBY cache content hash mismatch.")
        artifact = SourceArtifact(
            source="quandl_wiki_adjusted_git_csv",
            source_url=WIKI_URL,
            retrieved_at=str(envelope["retrieved_at"]),
            content=content,
            content_type=str(envelope.get("content_type") or "text/csv"),
        )
        _wiki_price_frame(artifact)
        return artifact

    def get(self) -> SourceArtifact | None:
        path = self.path()
        return self._decode(path.read_bytes()) if path.is_file() else None

    def _store(self, artifact: SourceArtifact) -> SourceArtifact:
        _wiki_price_frame(artifact)
        envelope = {
            "schema": "bbby_wiki_git_raw/v1",
            "source_url": WIKI_URL,
            "retrieved_at": artifact.retrieved_at,
            "content_type": artifact.content_type,
            "source_hash": artifact.source_hash,
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
        }
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(_canonical_json_bytes(envelope), mtime=0)
        if path.is_file():
            existing = self._decode(path.read_bytes())
            if existing.content != artifact.content:
                raise RuntimeError(f"Immutable pinned WIKI BBBY cache changed: {path}")
            return existing
        write_atomic(path, encoded)
        return self._decode(path.read_bytes())

    def _fetch_once(self) -> SourceArtifact:
        if self.http_attempts >= MAX_WIKI_HTTP_ATTEMPTS:
            raise RuntimeError("Pinned WIKI BBBY request cap reached before HTTP.")
        self.http_attempts += 1
        request = Request(
            WIKI_URL,
            headers={
                "User-Agent": "SuperTrendQuant BBBY identity repair/1.0",
                "Accept": "text/csv,text/plain;q=0.9",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with self.opener(request, timeout=60) as response:
                status = int(getattr(response, "status", 200))
                content_type = str(response.headers.get("Content-Type") or "text/csv")
                content = response.read(5 * 1024 * 1024 + 1)
        except HTTPError as exc:
            raise RuntimeError(
                f"Pinned WIKI BBBY single request failed: HTTP {exc.code}"
            ) from None
        except URLError as exc:
            raise RuntimeError(
                f"Pinned WIKI BBBY single request failed: {exc.reason}"
            ) from None
        if status != 200 or len(content) > 5 * 1024 * 1024:
            raise RuntimeError(
                "Pinned WIKI BBBY response rejected: "
                f"status={status}, bytes={len(content)}."
            )
        artifact = SourceArtifact(
            source="quandl_wiki_adjusted_git_csv",
            source_url=WIKI_URL,
            retrieved_at=utc_now_iso(),
            content=content,
            content_type=content_type,
        )
        return self._store(artifact)

    def fetch(self) -> PriceSourceBundle:
        artifact = self.get()
        if artifact is None and not self.allow_http:
            raise FileNotFoundError(
                "Pinned WIKI BBBY immutable cache is missing; explicitly allow fetch."
            )
        if artifact is None:
            artifact = self._fetch_once()
        return PriceSourceBundle(
            prices=_wiki_price_frame(artifact),
            actions=pd.DataFrame(columns=dataset_spec("corporate_actions").required_columns),
            artifacts=(artifact,),
            http_attempts=self.http_attempts,
        )


def _assert_logical_frame_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    label: str,
) -> None:
    columns = sorted(set(actual.columns) | set(expected.columns))
    left = actual.copy()
    right = expected.copy()
    for column in columns:
        if column not in left:
            left[column] = ""
        if column not in right:
            right[column] = ""
    left = left.loc[:, columns].sort_values(list(keys), kind="stable").reset_index(drop=True)
    right = right.loc[:, columns].sort_values(list(keys), kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(f"{label} rows do not match their archived raw response.") from exc


def validate_price_bundle_provenance(
    eodhd: PriceSourceBundle, wiki: PriceSourceBundle
) -> None:
    if not 0 <= eodhd.http_attempts <= MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError("BBBYQ bundle reports an invalid HTTP-attempt count.")
    if not 0 <= wiki.http_attempts <= MAX_WIKI_HTTP_ATTEMPTS:
        raise ValueError("Pinned WIKI BBBY bundle reports an invalid HTTP-attempt count.")
    by_source = {item.source: item for item in eodhd.artifacts}
    required = {"eodhd_eod", "eodhd_div", "eodhd_splits"}
    if set(by_source) != required or len(eodhd.artifacts) != 3:
        raise ValueError("BBBYQ bundle must contain exactly eod/div/splits raw artifacts.")
    for endpoint, source in (
        ("eod", "eodhd_eod"),
        ("div", "eodhd_div"),
        ("splits", "eodhd_splits"),
    ):
        artifact = by_source[source]
        parsed = urlparse(artifact.source_url)
        if (
            (parsed.hostname or "").lower() != "eodhd.com"
            or not parsed.path.lower().endswith(f"/{endpoint}/bbbyq.us")
        ):
            raise ValueError(f"BBBYQ raw artifact URL is wrong: {artifact.source_url}")
    parsed_prices = _eodhd_price_frame(by_source["eodhd_eod"])
    parsed_actions = _eodhd_action_frame(
        by_source["eodhd_div"], by_source["eodhd_splits"]
    )
    _assert_logical_frame_equal(
        eodhd.prices,
        parsed_prices,
        keys=dataset_spec("daily_price_raw").primary_key,
        label="BBBYQ price",
    )
    _assert_logical_frame_equal(
        eodhd.actions,
        parsed_actions,
        keys=dataset_spec("corporate_actions").primary_key,
        label="BBBYQ action",
    )
    if len(wiki.artifacts) != 1:
        raise ValueError("Pinned WIKI BBBY bundle must contain one raw artifact.")
    wiki_artifact = wiki.artifacts[0]
    _validate_wiki_artifact_identity(wiki_artifact)
    _assert_logical_frame_equal(
        wiki.prices,
        _wiki_price_frame(wiki_artifact),
        keys=dataset_spec("daily_price_raw").primary_key,
        label="Pinned WIKI BBBY adjusted price",
    )


class OfficialEvidenceSource:
    """Immutable SEC raw-byte cache with no retries and three URLs total."""

    def __init__(self, root: Path, *, allow_http: bool):
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.http_attempts = 0

    def path(self, url: str) -> Path:
        return self.root / f"{sha256_bytes(url.encode())}.json.gz"

    def _decode(self, url: str, payload: bytes) -> SourceArtifact:
        try:
            envelope = json.loads(gzip.decompress(payload))
            content = base64.b64decode(envelope["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError(f"Unreadable SEC BBBY cache: {self.path(url)}") from exc
        if envelope.get("schema") != "bbby_sec_raw/v1":
            raise ValueError("SEC BBBY cache schema mismatch.")
        if envelope.get("source_url") != url:
            raise ValueError("SEC BBBY cache URL mismatch.")
        if envelope.get("source_hash") != sha256_bytes(content):
            raise ValueError("SEC BBBY cache hash mismatch.")
        return SourceArtifact(
            source="sec_bbby_identity_evidence",
            source_url=url,
            retrieved_at=str(envelope["retrieved_at"]),
            content=content,
            content_type=str(envelope.get("content_type") or "text/html"),
        )

    def get(self, url: str) -> SourceArtifact | None:
        path = self.path(url)
        return self._decode(url, path.read_bytes()) if path.is_file() else None

    def _fetch(self, url: str) -> SourceArtifact:
        if self.http_attempts >= MAX_OFFICIAL_HTTP_ATTEMPTS:
            raise RuntimeError("SEC BBBY request cap reached before HTTP.")
        self.http_attempts += 1
        user_agent = os.getenv(
            "SEC_USER_AGENT", "SuperTrendQuant BBBY repair contact-required"
        )
        request = Request(
            url,
            headers={"User-Agent": user_agent, "Accept-Encoding": "identity"},
        )
        try:
            with urlopen(request, timeout=60) as response:
                status = int(getattr(response, "status", 200))
                content_type = str(response.headers.get("Content-Type") or "text/html")
                content = response.read(50 * 1024 * 1024 + 1)
        except HTTPError as exc:
            raise RuntimeError(f"SEC BBBY single request failed: HTTP {exc.code}") from None
        except URLError as exc:
            raise RuntimeError(f"SEC BBBY single request failed: {exc.reason}") from None
        if status != 200 or len(content) > 50 * 1024 * 1024:
            raise RuntimeError(
                f"SEC BBBY response rejected: status={status}, bytes={len(content)}"
            )
        artifact = SourceArtifact(
            source="sec_bbby_identity_evidence",
            source_url=url,
            retrieved_at=utc_now_iso(),
            content=content,
            content_type=content_type,
        )
        envelope = {
            "schema": "bbby_sec_raw/v1",
            "source_url": url,
            "retrieved_at": artifact.retrieved_at,
            "content_type": content_type,
            "source_hash": artifact.source_hash,
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
        path = self.path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(_canonical_json_bytes(envelope), mtime=0)
        if path.is_file():
            existing = self._decode(url, path.read_bytes())
            if existing.content != content:
                raise RuntimeError(f"Immutable SEC BBBY cache changed: {path}")
            return existing
        write_atomic(path, encoded)
        return self._decode(url, path.read_bytes())

    def load(self) -> OfficialEvidenceBundle:
        cached = {url: self.get(url) for url in OFFICIAL_URLS}
        missing = [url for url, artifact in cached.items() if artifact is None]
        if missing and not self.allow_http:
            raise FileNotFoundError(
                "SEC BBBY evidence cache is incomplete; explicitly allow fetch: "
                + ", ".join(missing)
            )
        if len(missing) > MAX_OFFICIAL_HTTP_ATTEMPTS:
            raise RuntimeError("SEC BBBY request set exceeds the frozen cap.")
        for url in missing:
            cached[url] = self._fetch(url)
        bundle = OfficialEvidenceBundle(tuple(cached[url] for url in OFFICIAL_URLS))
        validate_official_evidence(bundle)
        return bundle


def _document_text(content: bytes) -> str:
    text = content.decode("utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip().lower()


def _require_document_tokens(
    artifact: SourceArtifact, groups: Iterable[Iterable[str]]
) -> None:
    text = _document_text(artifact.content)
    missing = [
        " | ".join(group)
        for group in groups
        if not any(token.lower() in text for token in group)
    ]
    if missing:
        raise ValueError(
            f"SEC BBBY document lacks audited facts ({artifact.source_url}): "
            + "; ".join(missing)
        )


def validate_official_evidence(bundle: OfficialEvidenceBundle) -> None:
    old = bundle.artifact(OLD_DELISTING_URL)
    _require_document_tokens(
        old,
        (
            ("bed bath & beyond inc", "bed bath and beyond, inc"),
            ("suspended at the opening of business",),
            ("may 3, 2023", "may 3 2023"),
        ),
    )
    ostk = bundle.artifact(OSTK_TO_BYON_URL)
    _require_document_tokens(
        ostk,
        (
            ("overstock.com",),
            ("november 3, 2023", "november 3 2023"),
            ("november 6, 2023", "november 6 2023"),
            ("ostk",),
            ("byon",),
        ),
    )
    current = bundle.artifact(BYON_TO_BBBY_URL)
    _require_document_tokens(
        current,
        (
            ("beyond, inc",),
            ("august 28, 2025", "august 28 2025"),
            ("august 29, 2025", "august 29 2025"),
            ("byon",),
            ("bbby",),
        ),
    )
    for artifact in bundle.artifacts:
        if len(artifact.source_hash) != 64:
            raise ValueError("SEC BBBY evidence SHA-256 is invalid.")


def _sessions_exact(frame: pd.DataFrame, expected: tuple[str, ...], label: str) -> None:
    actual = tuple(
        sorted(pd.to_datetime(frame["session"], errors="coerce").dt.date.astype(str))
    )
    if actual != tuple(expected):
        actual_set = set(actual)
        expected_set = set(expected)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"{label} full-history gate failed: rows={len(actual)}, "
            f"expected={len(expected)}, missing={missing[:3]}, extra={extra[:3]}."
        )


def _cross_validate_prices(
    eodhd: pd.DataFrame,
    wiki: pd.DataFrame,
    actions: pd.DataFrame,
    expected: tuple[str, ...],
) -> dict[str, Any]:
    _sessions_exact(eodhd, expected, "EODHD BBBYQ")
    overlap_start = max(OLD_START, WIKI_FIRST_DATE)
    overlap_end = min(OLD_LAST_TRADING_DATE, WIKI_LAST_DATE)
    if overlap_start > overlap_end:
        raise ValueError("Pinned WIKI BBBY has no overlap with EODHD BBBYQ.")
    overlap_sessions = _expected_sessions(overlap_start, overlap_end)
    if len(overlap_sessions) < 3:
        raise ValueError("Pinned WIKI BBBY overlap is too short for validation.")
    overlap_set = set(overlap_sessions)
    eodhd_overlap = eodhd.loc[
        eodhd["session"].astype(str).isin(overlap_set)
    ].copy()
    wiki_overlap = wiki.loc[wiki["session"].astype(str).isin(overlap_set)].copy()
    _sessions_exact(eodhd_overlap, overlap_sessions, "EODHD BBBYQ overlap")
    _sessions_exact(wiki_overlap, overlap_sessions, "Pinned WIKI BBBY overlap")
    columns = ["session", "open", "high", "low", "close", "volume"]
    joined = eodhd_overlap.loc[:, columns].merge(
        wiki_overlap.loc[:, columns],
        on="session",
        suffixes=("_eodhd", "_wiki"),
        validate="one_to_one",
    )
    joined = joined.sort_values("session", kind="stable").reset_index(drop=True)
    if len(joined) != len(overlap_sessions):
        raise ValueError("EODHD/WIKI did not compare every overlap session.")

    mismatch_rows: set[int] = set()
    shape_stats: dict[str, float] = {}
    for numerator in ("high", "low", "close"):
        raw = pd.to_numeric(joined[f"{numerator}_eodhd"]).to_numpy(dtype=float) / pd.to_numeric(
            joined["open_eodhd"]
        ).to_numpy(dtype=float)
        adjusted = pd.to_numeric(joined[f"{numerator}_wiki"]).to_numpy(dtype=float) / pd.to_numeric(
            joined["open_wiki"]
        ).to_numpy(dtype=float)
        differences = np.abs(raw - adjusted)
        passed = np.isclose(raw, adjusted, rtol=0.0025, atol=0.0005)
        mismatch_rows.update(np.flatnonzero(~passed).tolist())
        shape_stats[f"maximum_{numerator}_to_open_absolute_deviation"] = float(
            differences.max()
        )

    raw_volume = pd.to_numeric(joined["volume_eodhd"]).to_numpy(dtype=float)
    wiki_volume = pd.to_numeric(joined["volume_wiki"]).to_numpy(dtype=float)
    volume_passed = np.isclose(raw_volume, wiki_volume, rtol=0.02, atol=1.0)
    volume_mismatch_rows = np.flatnonzero(~volume_passed)
    volume_match_ratio = float(volume_passed.mean())
    volume_denominator = np.maximum(np.maximum(np.abs(raw_volume), np.abs(wiki_volume)), 1.0)
    volume_relative_deviation = np.abs(raw_volume - wiki_volume) / volume_denominator
    if volume_match_ratio < MINIMUM_VOLUME_MATCH_RATIO:
        sample = joined.loc[
            volume_mismatch_rows[:3], "session"
        ].astype(str).tolist()
        raise ValueError(
            "EODHD BBBYQ/pinned WIKI adjusted overlap comparison failed: "
            "independent volume match ratio below the audited floor; "
            f"ratio={volume_match_ratio:.6f}, "
            f"required={MINIMUM_VOLUME_MATCH_RATIO:.6f}, "
            f"mismatch_sessions={len(volume_mismatch_rows)}, sample={sample}."
        )

    action_dates: set[str] = set()
    dividend_amounts: dict[str, float] = {}
    if not actions.empty:
        action_type = actions["action_type"].astype(str)
        effective = actions["effective_date"].astype(str)
        if "ex_date" in actions:
            ex_dates = actions["ex_date"].fillna("").astype(str)
            effective = ex_dates.where(ex_dates.ne(""), effective)
        for index, date in effective.items():
            if date not in overlap_set:
                continue
            kind = str(action_type.loc[index])
            if kind in {"cash_dividend", "split"}:
                action_dates.add(date)
            if kind == "split":
                raise ValueError(
                    "Pinned WIKI BBBY volume comparison assumes the audited "
                    f"split-free overlap, but a split was found on {date}."
                )
            if kind == "cash_dividend":
                amount = pd.to_numeric(
                    pd.Series([actions.loc[index, "cash_amount"]]), errors="coerce"
                ).iloc[0]
                if pd.isna(amount) or float(amount) < 0:
                    raise ValueError(f"BBBY dividend amount is invalid on {date}.")
                dividend_amounts[date] = dividend_amounts.get(date, 0.0) + float(amount)

    raw_close = pd.to_numeric(joined["close_eodhd"]).to_numpy(dtype=float)
    wiki_close = pd.to_numeric(joined["close_wiki"]).to_numpy(dtype=float)
    raw_returns = raw_close[1:] / raw_close[:-1] - 1.0
    wiki_returns = wiki_close[1:] / wiki_close[:-1] - 1.0
    return_sessions = joined["session"].astype(str).iloc[1:].tolist()
    non_action_positions = np.array(
        [index for index, session in enumerate(return_sessions) if session not in action_dates],
        dtype=int,
    )
    if len(non_action_positions) < 2:
        raise ValueError("Pinned WIKI BBBY lacks enough non-action returns to compare.")
    non_action_passed = np.isclose(
        raw_returns[non_action_positions],
        wiki_returns[non_action_positions],
        rtol=0.01,
        atol=0.0005,
    )
    mismatch_rows.update((non_action_positions[~non_action_passed] + 1).tolist())

    dividend_checked = 0
    for position, session in enumerate(return_sessions):
        if session not in dividend_amounts:
            continue
        expected_total_return = (
            (raw_close[position + 1] + dividend_amounts[session]) / raw_close[position]
            - 1.0
        )
        if not np.isclose(
            expected_total_return,
            wiki_returns[position],
            rtol=0.01,
            atol=0.00075,
        ):
            mismatch_rows.add(position + 1)
        dividend_checked += 1

    if mismatch_rows:
        sample = joined.loc[sorted(mismatch_rows)[:3], "session"].astype(str).tolist()
        raise ValueError(
            "EODHD BBBYQ/pinned WIKI adjusted overlap comparison failed: "
            f"mismatch_sessions={len(mismatch_rows)}, sample={sample}."
        )
    return {
        "status": "passed",
        "eodhd_rows": len(eodhd),
        "wiki_rows": len(wiki),
        "wiki_sha256": WIKI_SHA256,
        "wiki_commit": WIKI_COMMIT,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "overlap_session_count": len(joined),
        "all_overlap_sessions_compared": True,
        "mismatch_sessions": 0,
        "shape_checks": shape_stats,
        "volume_sessions_checked": len(joined),
        "volume_match_ratio": volume_match_ratio,
        "minimum_volume_match_ratio": MINIMUM_VOLUME_MATCH_RATIO,
        "volume_mismatch_sessions": int(len(volume_mismatch_rows)),
        "volume_mismatch_sample": joined.loc[
            volume_mismatch_rows[:10], "session"
        ].astype(str).tolist(),
        "maximum_volume_relative_deviation": float(
            volume_relative_deviation.max()
        ),
        "non_action_return_sessions_checked": len(non_action_positions),
        "dividend_return_sessions_checked": dividend_checked,
    }


def _metadata_row(
    base: dict[str, Any],
    *,
    source_url: str,
    source_hash: str,
    retrieved_at: str,
) -> dict[str, Any]:
    value = dict(base)
    value.update(
        {
            "source": "bbby_identity_repair",
            "source_url": source_url,
            "retrieved_at": retrieved_at,
            "source_hash": source_hash,
        }
    )
    return value


def rewrite_security_identity(
    master: pd.DataFrame,
    history: pd.DataFrame,
    *,
    catalog: CatalogProof,
    official: OfficialEvidenceBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_doc = official.artifact(OLD_DELISTING_URL)
    ostk_doc = official.artifact(OSTK_TO_BYON_URL)
    current_doc = official.artifact(BYON_TO_BBBY_URL)
    output_master = master.copy()
    if "isin" not in output_master:
        output_master["isin"] = ""
    old_index = output_master.index[
        output_master["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ]
    current_index = output_master.index[
        output_master["security_id"].astype(str).eq(CURRENT_SECURITY_ID)
    ]
    if len(old_index) != 1 or len(current_index) != 1:
        raise ValueError("BBBY master identities changed after preflight.")
    old_values = {
        "primary_symbol": "BBBY",
        "provider_symbol": "BBBY_old.US",
        "action_provider_symbol": "BBBYQ.US",
        "name": "Bed Bath & Beyond Inc",
        "exchange": "NASDAQ",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": OLD_START,
        "active_to": OLD_LAST_TRADING_DATE,
        "isin": OLD_ISIN,
        "source": "bbby_identity_repair",
        "source_url": OLD_DELISTING_URL,
        "retrieved_at": old_doc.retrieved_at,
        "source_hash": old_doc.source_hash,
    }
    current_values = {
        "primary_symbol": "BBBY",
        "provider_symbol": "BBBY.US",
        "action_provider_symbol": "BBBY.US",
        "name": "Bed Bath & Beyond, Inc.",
        "exchange": "NYSE",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": OLD_START,
        "active_to": "",
        "isin": CURRENT_ISIN,
        "source": "eodhd_exchange_symbols",
        "source_url": ACTIVE_CATALOG_URL,
        "retrieved_at": str(output_master.loc[current_index[0], "retrieved_at"]),
        "source_hash": catalog.active_archive_hash,
    }
    for column, value in old_values.items():
        output_master.loc[old_index, column] = value
    for column, value in current_values.items():
        output_master.loc[current_index, column] = value

    keep = ~history["security_id"].astype(str).isin(
        (OLD_SECURITY_ID, CURRENT_SECURITY_ID)
    )
    base_columns = list(history.columns)
    rows = [
        _metadata_row(
            {
                "security_id": OLD_SECURITY_ID,
                "symbol": "BBBY",
                "exchange": "NASDAQ",
                "effective_from": "2015-01-01",
                "effective_to": OLD_LAST_TRADING_DATE,
            },
            source_url=OLD_DELISTING_URL,
            source_hash=old_doc.source_hash,
            retrieved_at=old_doc.retrieved_at,
        ),
        _metadata_row(
            {
                "security_id": CURRENT_SECURITY_ID,
                "symbol": "OSTK",
                "exchange": "NASDAQ",
                "effective_from": "2015-01-01",
                "effective_to": "2023-11-05",
            },
            source_url=OSTK_TO_BYON_URL,
            source_hash=ostk_doc.source_hash,
            retrieved_at=ostk_doc.retrieved_at,
        ),
        _metadata_row(
            {
                "security_id": CURRENT_SECURITY_ID,
                "symbol": "BYON",
                "exchange": "NYSE",
                "effective_from": OSTK_TO_BYON,
                "effective_to": "2025-08-28",
            },
            source_url=BYON_TO_BBBY_URL,
            source_hash=current_doc.source_hash,
            retrieved_at=current_doc.retrieved_at,
        ),
        _metadata_row(
            {
                "security_id": CURRENT_SECURITY_ID,
                "symbol": "BBBY",
                "exchange": "NYSE",
                "effective_from": BYON_TO_BBBY,
                "effective_to": "",
            },
            source_url=BYON_TO_BBBY_URL,
            source_hash=current_doc.source_hash,
            retrieved_at=current_doc.retrieved_at,
        ),
    ]
    additions = pd.DataFrame(rows)
    for column in base_columns:
        if column not in additions:
            additions[column] = ""
    output_history = pd.concat(
        [history.loc[keep], additions.loc[:, base_columns]], ignore_index=True
    )
    return output_master.reset_index(drop=True), output_history.reset_index(drop=True)


def rewrite_prices_actions_factors(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    primary_prices: pd.DataFrame,
    provider_actions: pd.DataFrame,
    source_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    affected = (OLD_SECURITY_ID, CURRENT_SECURITY_ID)
    current_prices = prices.loc[
        prices["security_id"].astype(str).eq(CURRENT_SECURITY_ID)
    ].copy()
    output_prices = pd.concat(
        [
            prices.loc[~prices["security_id"].astype(str).isin(affected)],
            current_prices,
            primary_prices,
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(["security_id", "session"], keep="last")

    output_actions = pd.concat(
        [
            actions.loc[~actions["security_id"].astype(str).isin(affected)],
            provider_actions,
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(["event_id"], keep="last")

    affected_prices = output_prices.loc[
        output_prices["security_id"].astype(str).isin(affected)
    ].copy()
    affected_actions = output_actions.loc[
        output_actions["security_id"].astype(str).isin(affected)
    ].copy()
    rebuilt = build_adjustment_factors(
        affected_prices,
        affected_actions,
        source_version=source_version,
    )
    output_factors = pd.concat(
        [
            factors.loc[~factors["security_id"].astype(str).isin(affected)],
            rebuilt,
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(["security_id", "session"], keep="last")
    return (
        output_prices.reset_index(drop=True),
        output_actions.reset_index(drop=True),
        output_factors.reset_index(drop=True),
    )


def select_legacy_actions(
    existing_actions: pd.DataFrame, provider_actions: pd.DataFrame
) -> pd.DataFrame:
    """Keep legitimate old-issuer actions, preferring the audited BBBYQ probe.

    The collision is in price ownership.  The existing ``BBBY_old`` dividend
    rows are valid legacy-company events and must not disappear merely because
    BBBYQ's delisted endpoint returns no dividend history.  Current-issuer
    duplicates are never used.  When both sources describe the same event,
    their economic terms must agree and the BBBYQ row wins.
    """

    old = existing_actions.loc[
        existing_actions["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ].copy()
    provider = provider_actions.copy()
    for frame, label in ((old, "existing"), (provider, "BBBYQ")):
        if frame.empty:
            continue
        dates = pd.to_datetime(frame["effective_date"], errors="coerce")
        invalid = dates.isna() | dates.gt(pd.Timestamp(OLD_LAST_TRADING_DATE))
        if invalid.any():
            raise ValueError(f"{label} legacy BBBY actions cross the legal boundary.")
    semantic = ("action_type", "effective_date")
    old_by_key = {
        (str(row.action_type), str(row.effective_date)): row
        for row in old.itertuples(index=False)
    }
    for row in provider.itertuples(index=False):
        key = (str(row.action_type), str(row.effective_date))
        prior = old_by_key.get(key)
        if prior is None:
            continue
        for column in ("cash_amount", "ratio"):
            left = getattr(prior, column)
            right = getattr(row, column)
            if pd.isna(left) and pd.isna(right):
                continue
            if pd.isna(left) != pd.isna(right) or not np.isclose(
                float(left), float(right), rtol=1e-8, atol=1e-10
            ):
                raise ValueError(
                    "BBBYQ action terms conflict with retained legacy evidence: "
                    f"{key}/{column}."
                )
    provider_keys = set(
        zip(provider.get("action_type", pd.Series(dtype=str)).astype(str),
            provider.get("effective_date", pd.Series(dtype=str)).astype(str))
    )
    retained = old.loc[
        ~pd.Series(
            list(zip(old.get("action_type", pd.Series(dtype=str)).astype(str),
                     old.get("effective_date", pd.Series(dtype=str)).astype(str))),
            index=old.index,
        ).isin(provider_keys)
    ]
    return pd.concat([retained, provider], ignore_index=True, sort=False).drop_duplicates(
        list(semantic), keep="last"
    ).reset_index(drop=True)


def rewrite_index_references(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    *,
    official: OfficialEvidenceBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    old_doc = official.artifact(OLD_DELISTING_URL)
    output_anchors = anchors.copy()
    early_anchors = (
        output_anchors["security_id"].astype(str).eq(CURRENT_SECURITY_ID)
        & pd.to_datetime(output_anchors["anchor_date"], errors="coerce").le(
            pd.Timestamp(OLD_LAST_TRADING_DATE)
        )
    )
    output_anchors.loc[early_anchors, "security_id"] = OLD_SECURITY_ID
    output_anchors.loc[early_anchors, "source"] = "bbby_identity_repair"
    output_anchors.loc[early_anchors, "source_url"] = OLD_DELISTING_URL
    output_anchors.loc[early_anchors, "source_kind"] = "derived_identity"
    output_anchors.loc[early_anchors, "retrieved_at"] = old_doc.retrieved_at
    output_anchors.loc[early_anchors, "source_hash"] = old_doc.source_hash
    output_anchors = output_anchors.drop_duplicates(
        ["index_id", "anchor_date", "security_id"], keep="last"
    )

    output_events = events.copy()
    early_events = (
        output_events["security_id"].astype(str).eq(CURRENT_SECURITY_ID)
        & pd.to_datetime(output_events["effective_date"], errors="coerce").le(
            pd.Timestamp(OLD_LAST_TRADING_DATE)
        )
    )
    output_events.loc[early_events, "security_id"] = OLD_SECURITY_ID
    output_events.loc[early_events, "source"] = "bbby_identity_repair"
    output_events.loc[early_events, "source_url"] = OLD_DELISTING_URL
    output_events.loc[early_events, "source_kind"] = "derived_identity"
    output_events.loc[early_events, "retrieved_at"] = old_doc.retrieved_at
    output_events.loc[early_events, "source_hash"] = old_doc.source_hash
    return (
        output_anchors.reset_index(drop=True),
        output_events.reset_index(drop=True),
        {
            "anchors_remapped": int(early_anchors.sum()),
            "events_remapped": int(early_events.sum()),
        },
    )


def _artifact_extension(artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower()
    if "json" in content_type:
        return "json"
    if "csv" in content_type:
        return "csv"
    if "html" in content_type:
        return "html"
    if "pdf" in content_type:
        return "pdf"
    return "bin"


def _request_archive_artifact(artifact: SourceArtifact) -> SourceArtifact:
    """Preserve raw bytes while making equal no-data responses URL-unique.

    Dividend and split endpoints commonly both return ``[]``.  ``source_archive``
    is content-addressed by one primary key, so storing the bare bytes would
    discard one request URL.  This envelope keeps the exact bytes and their
    raw SHA-256 while its own hash is unique to the request provenance.
    """

    content = _canonical_json_bytes(
        {
            "schema": "bbby_request_archive/v1",
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
        content=content,
        content_type="application/vnd.supertrendquant.source-envelope+json",
    )


def _validate_request_envelopes(
    raw_artifacts: Iterable[SourceArtifact],
    archived_artifacts: Iterable[SourceArtifact],
) -> None:
    archived_by_url = {item.source_url: item for item in archived_artifacts}
    for raw in raw_artifacts:
        archived = archived_by_url.get(raw.source_url)
        if archived is None:
            raise ValueError(f"BBBY request envelope is absent: {raw.source_url}")
        try:
            value = json.loads(archived.content)
            decoded = base64.b64decode(value["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError(f"BBBY request envelope is invalid: {raw.source_url}") from exc
        if (
            value.get("schema") != "bbby_request_archive/v1"
            or value.get("source_url") != raw.source_url
            or value.get("content_sha256") != raw.source_hash
            or decoded != raw.content
        ):
            raise ValueError(f"BBBY request envelope does not preserve raw bytes: {raw.source_url}")


def append_source_archive(
    frame: pd.DataFrame,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> pd.DataFrame:
    rows = []
    for artifact in artifacts:
        rows.append(
            {
                "archive_id": artifact.source_hash,
                "dataset": artifact.source,
                "object_path": (
                    f"archives/{completed_session}/{artifact.source_hash}."
                    f"{_artifact_extension(artifact)}.gz"
                ),
                "content_type": artifact.content_type,
                "effective_date": completed_session,
                "source": artifact.source,
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    return _concat_unique(
        (frame, pd.DataFrame(rows)), keys=dataset_spec("source_archive").primary_key
    )


class _FrameRepository:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def current_manifest(self, dataset: str):
        return object() if dataset in self.frames else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        return self.frames[dataset].copy()


def _issue_counts(frames: dict[str, pd.DataFrame]) -> dict[tuple[str, str], int]:
    report = validate_repository_snapshot(_FrameRepository(frames))
    result: dict[tuple[str, str], int] = {}
    for issue in report.issues:
        key = (issue.code, issue.severity)
        result[key] = result.get(key, 0) + int(issue.row_count or 1)
    return result


def _validate_nonregression(
    previous: dict[str, pd.DataFrame], candidate: dict[str, pd.DataFrame]
) -> None:
    before = _issue_counts(previous)
    after = _issue_counts(candidate)
    regressions = {
        key: (before.get(key, 0), value)
        for key, value in after.items()
        if value > before.get(key, 0)
    }
    if regressions:
        raise ValueError(f"BBBY repair introduced repository validation issues: {regressions}")


def validate_index_replay_gate(
    previous: dict[str, pd.DataFrame],
    candidate: dict[str, pd.DataFrame],
    *,
    completed_session: str,
) -> dict[str, Any]:
    before = IndexEventReplayer(
        previous["index_constituent_anchors"],
        previous["index_membership_events"],
    )
    after = IndexEventReplayer(
        candidate["index_constituent_anchors"],
        candidate["index_membership_events"],
    )
    anchors = candidate["index_constituent_anchors"]
    events = candidate["index_membership_events"]
    checked = 0
    for index_id, removal in EXPECTED_REMOVALS.items():
        relevant_anchor = anchors.loc[
            anchors["index_id"].astype(str).eq(index_id)
            & anchors["security_id"].astype(str).eq(OLD_SECURITY_ID)
        ]
        if len(relevant_anchor) != 1:
            raise ValueError(f"{index_id} must have one canonical old BBBY anchor.")
        relevant_removal = events.loc[
            events["index_id"].astype(str).eq(index_id)
            & events["security_id"].astype(str).eq(OLD_SECURITY_ID)
            & events["operation"].astype(str).str.upper().eq("REMOVE")
            & events["effective_date"].astype(str).eq(removal)
        ]
        if len(relevant_removal) != 1:
            raise ValueError(f"{index_id} old BBBY removal is absent or duplicated.")
        start = str(relevant_anchor.iloc[0]["anchor_date"])
        dates = (
            start,
            (pd.Timestamp(removal) - pd.Timedelta(days=1)).date().isoformat(),
            removal,
            completed_session,
        )
        for date in dates:
            prior = before.members_on(index_id, date)
            current = after.members_on(index_id, date)
            if len(prior.security_ids) != len(current.security_ids):
                raise ValueError(f"BBBY remap changed {index_id} cardinality on {date}.")
            if any(
                OLD_SECURITY_ID in warning or CURRENT_SECURITY_ID in warning
                for warning in current.warnings
            ):
                raise ValueError(f"BBBY replay warning remains for {index_id} on {date}.")
            members = set(current.security_ids)
            should_exist = pd.Timestamp(date) < pd.Timestamp(removal)
            if (OLD_SECURITY_ID in members) != should_exist:
                raise ValueError(f"Old BBBY membership boundary is wrong: {index_id}/{date}.")
            if CURRENT_SECURITY_ID in members and pd.Timestamp(date) <= pd.Timestamp(removal):
                raise ValueError(f"Current BBBY issuer leaked into historical {index_id}.")
            checked += 1
    return {"index_snapshots_checked": checked, "removals": dict(EXPECTED_REMOVALS)}


def validate_full_history_gate(
    preflight: LocalPreflight,
    frames: dict[str, pd.DataFrame],
    *,
    completed_session: str,
    expected_old: tuple[str, ...],
) -> dict[str, Any]:
    prices = frames["daily_price_raw"]
    old = prices.loc[prices["security_id"].astype(str).eq(OLD_SECURITY_ID)].copy()
    current = prices.loc[
        prices["security_id"].astype(str).eq(CURRENT_SECURITY_ID)
    ].copy()
    _sessions_exact(old, expected_old, "Legacy BBBY")
    expected_current = _expected_sessions(OLD_START, completed_session)
    _sessions_exact(current, expected_current, "Current OSTK/BYON/BBBY")
    if set(old["source_hash"].astype(str)) & set(preflight.contaminated_price_hashes):
        raise ValueError("Contaminated legacy BBBY EODHD rows survived replacement.")
    if "source_url" in old and old["source_url"].map(_is_forbidden_old_eodhd_url).any():
        raise ValueError("A contaminated BBBY.US/BBBY_old.US row survived replacement.")
    original_current = preflight.existing["daily_price_raw"].loc[
        preflight.existing["daily_price_raw"]["security_id"]
        .astype(str)
        .eq(CURRENT_SECURITY_ID)
    ].copy()
    common = sorted(set(original_current.columns) & set(current.columns))
    left = original_current.loc[:, common].sort_values("session").reset_index(drop=True)
    right = current.loc[:, common].sort_values("session").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError("Current OSTK/BYON/BBBY price series was modified.") from exc

    master = frames["security_master"]
    isin = {
        str(row.security_id): str(row.isin)
        for row in master.loc[
            master["security_id"].astype(str).isin(
                (OLD_SECURITY_ID, CURRENT_SECURITY_ID)
            )
        ].itertuples(index=False)
    }
    if isin != {OLD_SECURITY_ID: OLD_ISIN, CURRENT_SECURITY_ID: CURRENT_ISIN}:
        raise ValueError(f"BBBY legal identity/ISIN split is wrong: {isin}")
    if not _history_is_repaired(frames["symbol_history"]):
        raise ValueError("OSTK -> BYON -> BBBY symbol history gate failed.")
    actions = frames["corporate_actions"]
    if actions["security_id"].astype(str).eq(CURRENT_SECURITY_ID).any():
        raise ValueError("Old Bed Bath actions remain assigned to the current issuer.")
    old_actions = actions.loc[
        actions["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ]
    if (
        pd.to_datetime(old_actions["effective_date"], errors="coerce")
        .gt(pd.Timestamp(OLD_LAST_TRADING_DATE))
        .any()
    ):
        raise ValueError("Legacy BBBY action history exceeds the official boundary.")
    return {
        "old_price_rows": len(old),
        "old_first_session": min(expected_old),
        "old_last_session": max(expected_old),
        "current_price_rows": len(current),
        "old_action_rows": len(old_actions),
        "contaminated_rows_remaining": 0,
        "old_isin": OLD_ISIN,
        "current_isin": CURRENT_ISIN,
    }


def validate_archive_gate(
    frames: dict[str, pd.DataFrame], artifacts: Iterable[SourceArtifact]
) -> None:
    archive = frames["source_archive"]
    pairs = set(zip(archive["source_url"].astype(str), archive["source_hash"].astype(str)))
    for artifact in artifacts:
        if (artifact.source_url, artifact.source_hash) not in pairs:
            raise ValueError(f"BBBY raw evidence is not represented in source_archive: {artifact.source_url}")
    for url in OFFICIAL_URLS:
        matches = archive.loc[archive["source_url"].astype(str).eq(url)]
        if len(matches) != 1 or len(str(matches.iloc[0]["source_hash"])) != 64:
            raise ValueError(f"Official SEC raw bytes/hash are not uniquely archived: {url}")


def validate_candidate_frames(
    preflight: LocalPreflight,
    frames: dict[str, pd.DataFrame],
    artifacts: tuple[SourceArtifact, ...],
    *,
    completed_session: str,
    expected_old: tuple[str, ...],
) -> dict[str, Any]:
    for dataset in WRITE_DATASETS:
        report = validate_dataset(
            dataset,
            frames[dataset],
            incomplete_action_policy="warn",
            completed_session=completed_session,
        )
        report.raise_for_errors()
    _validate_nonregression(preflight.existing, frames)
    replay = validate_index_replay_gate(
        preflight.existing, frames, completed_session=completed_session
    )
    history = validate_full_history_gate(
        preflight,
        frames,
        completed_session=completed_session,
        expected_old=expected_old,
    )
    validate_archive_gate(frames, artifacts)
    return {"index_replay": replay, "full_history": history, "archive": "passed"}


def _repair_manifest_artifact(
    *,
    release: DataRelease,
    selected_primary: str,
    crosscheck: dict[str, Any],
    catalog: CatalogProof,
    eodhd: PriceSourceBundle,
    wiki: PriceSourceBundle,
    official: OfficialEvidenceBundle,
) -> SourceArtifact:
    retrieved_at = utc_now_iso()
    content = _canonical_json_bytes(
        {
            "schema": "us_bbby_identity_repair/v1",
            "base_release_version": release.version,
            "old_security_id": OLD_SECURITY_ID,
            "old_isin": OLD_ISIN,
            "current_security_id": CURRENT_SECURITY_ID,
            "current_isin": CURRENT_ISIN,
            "legacy_price_window": [OLD_START, OLD_LAST_TRADING_DATE],
            "selected_primary": selected_primary,
            "independent_cross_validation": crosscheck,
            "catalog_hashes": {
                "active": catalog.active_archive_hash,
                "delisted": catalog.delisted_archive_hash,
            },
            "eodhd_raw": [
                {"url": item.source_url, "sha256": item.source_hash}
                for item in eodhd.artifacts
            ],
            "wiki_adjusted_raw": [
                {"url": item.source_url, "sha256": item.source_hash}
                for item in wiki.artifacts
            ],
            "official_raw": [
                {"url": item.source_url, "sha256": item.source_hash}
                for item in official.artifacts
            ],
            "wiki_validation_mode": (
                "adjusted_bar_shape_volume_non_action_return_and_"
                "action_aware_dividend_return"
            ),
            "wiki_never_primary_or_gap_fill": True,
            "self_validation_excluded": False,
        }
    )
    return SourceArtifact(
        source="bbby_identity_repair_manifest",
        source_url="local://bbby-identity-repair/manifest",
        retrieved_at=retrieved_at,
        content=content,
        content_type="application/json",
    )


def prepare_repair(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
    preflight: LocalPreflight,
    *,
    eodhd: PriceSourceBundle,
    wiki: PriceSourceBundle,
    official: OfficialEvidenceBundle,
) -> PreparedRepair:
    validate_price_bundle_provenance(eodhd, wiki)
    expected_old = _expected_sessions(OLD_START, OLD_LAST_TRADING_DATE)
    _sessions_exact(eodhd.prices, expected_old, "EODHD BBBYQ legacy history")
    legacy_actions = select_legacy_actions(
        preflight.existing["corporate_actions"], eodhd.actions
    )
    crosscheck = _cross_validate_prices(
        eodhd.prices, wiki.prices, legacy_actions, expected_old
    )
    if crosscheck["status"] != "passed":
        raise ValueError("Pinned WIKI BBBY independent validation did not pass.")
    selected_primary = "eodhd_bbbyq"
    primary_prices = eodhd.prices.copy()

    master, history = rewrite_security_identity(
        preflight.existing["security_master"],
        preflight.existing["symbol_history"],
        catalog=preflight.catalog,
        official=official,
    )
    source_version = sha256_bytes(
        _canonical_json_bytes(
            {
                "old": OLD_SECURITY_ID,
                "current": CURRENT_SECURITY_ID,
                "primary": selected_primary,
                "primary_hashes": sorted(set(primary_prices["source_hash"].astype(str))),
                "action_hashes": sorted(
                    set(
                        legacy_actions.get("source_hash", pd.Series(dtype=str)).astype(str)
                    )
                ),
            }
        )
    )
    prices, actions, factors = rewrite_prices_actions_factors(
        preflight.existing["daily_price_raw"],
        preflight.existing["corporate_actions"],
        preflight.existing["adjustment_factors"],
        primary_prices=primary_prices,
        provider_actions=legacy_actions,
        source_version=source_version,
    )
    anchors, events, index_stats = rewrite_index_references(
        preflight.existing["index_constituent_anchors"],
        preflight.existing["index_membership_events"],
        official=official,
    )
    manifest_artifact = _repair_manifest_artifact(
        release=release,
        selected_primary=selected_primary,
        crosscheck=crosscheck,
        catalog=preflight.catalog,
        eodhd=eodhd,
        wiki=wiki,
        official=official,
    )
    eodhd_archives = tuple(_request_archive_artifact(item) for item in eodhd.artifacts)
    _validate_request_envelopes(eodhd.artifacts, eodhd_archives)
    artifacts = tuple(
        dict.fromkeys(
            (*eodhd_archives, *wiki.artifacts, *official.artifacts, manifest_artifact)
        )
    )
    archive = append_source_archive(
        preflight.existing["source_archive"],
        artifacts,
        completed_session=release.completed_session,
    )
    frames = {
        **preflight.existing,
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": archive,
    }
    gates = validate_candidate_frames(
        preflight,
        frames,
        artifacts,
        completed_session=release.completed_session,
        expected_old=expected_old,
    )
    current, current_etag = repository.current_release()
    if current is None or current.version != release.version or current_etag != release_etag:
        raise RuntimeError("Current release changed during BBBY repair preparation.")
    warnings = tuple(release.warnings)
    summary = {
        "status": "validated_dry_run",
        "base_release_version": release.version,
        "completed_session": release.completed_session,
        "old_security_id": OLD_SECURITY_ID,
        "current_security_id": CURRENT_SECURITY_ID,
        "old_isin": OLD_ISIN,
        "current_isin": CURRENT_ISIN,
        "selected_primary": selected_primary,
        "independent_cross_validation": crosscheck,
        "contamination_overlap_rows": preflight.contamination_overlap_rows,
        "index_rewrite": index_stats,
        "maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "eodhd_http_attempts_this_run": eodhd.http_attempts,
        "maximum_wiki_http_attempts": MAX_WIKI_HTTP_ATTEMPTS,
        "wiki_http_attempts_this_run": wiki.http_attempts,
        "maximum_official_http_attempts": MAX_OFFICIAL_HTTP_ATTEMPTS,
        "maximum_total_http_attempts": MAX_TOTAL_HTTP_ATTEMPTS,
        "gates": gates,
        "warnings": list(warnings),
        "write_datasets": list(WRITE_DATASETS),
    }
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=preflight.pointer_etags,
        frames=frames,
        archive_artifacts=artifacts,
        warnings=warnings,
        summary=summary,
    )


def _cache_inventory(root: Path) -> dict[str, Any]:
    eodhd_root = root / "state/bbby-identity/eodhd"
    eodhd = EodhdBbbyqSource(eodhd_root, allow_http=False)
    wiki_path = (
        root
        / "state/bbby-identity/wiki"
        / f"{sha256_bytes(WIKI_URL.encode())}.json.gz"
    )
    official_root = root / "state/bbby-identity/sec"
    return {
        "eodhd": {
            endpoint: eodhd.path(endpoint).is_file() for endpoint in EODHD_ENDPOINTS
        },
        "wiki": {WIKI_URL: wiki_path.is_file()},
        "official": {
            url: (
                official_root / f"{sha256_bytes(url.encode())}.json.gz"
            ).is_file()
            for url in OFFICIAL_URLS
        },
    }


def build_offline_plan(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, Any]:
    preflight = build_local_preflight(repository, release)
    inventory = _cache_inventory(repository.root)
    missing_eodhd = sum(not value for value in inventory["eodhd"].values())
    missing_wiki = sum(not value for value in inventory["wiki"].values())
    missing_official = sum(not value for value in inventory["official"].values())
    return {
        "status": "already_repaired" if preflight.already_repaired else "offline_plan",
        "release_version": release.version,
        "network_clients_constructed": 0,
        "http_attempts": 0,
        "maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "maximum_wiki_http_attempts": MAX_WIKI_HTTP_ATTEMPTS,
        "maximum_official_http_attempts": MAX_OFFICIAL_HTTP_ATTEMPTS,
        "maximum_total_http_attempts": MAX_TOTAL_HTTP_ATTEMPTS,
        "next_run_maximum_http_attempts": (
            missing_eodhd + missing_wiki + missing_official
        ),
        "cache_inventory": inventory,
        "old_security_id": OLD_SECURITY_ID,
        "old_isin": OLD_ISIN,
        "current_security_id": CURRENT_SECURITY_ID,
        "current_isin": CURRENT_ISIN,
        "would_write": False,
    }


def _persist_archive_payloads(
    repository: LocalDatasetRepository,
    artifacts: Iterable[SourceArtifact],
    completed_session: str,
) -> None:
    for artifact in artifacts:
        path = (
            repository.root
            / "archives"
            / completed_session
            / f"{artifact.source_hash}.{_artifact_extension(artifact)}.gz"
        )
        if path.is_file():
            try:
                existing = gzip.decompress(path.read_bytes())
            except Exception as exc:
                raise RuntimeError(f"Unreadable BBBY archive payload: {path}") from exc
            if existing != artifact.content:
                raise RuntimeError(f"Conflicting BBBY archive payload: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, gzip.compress(artifact.content, mtime=0))
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"BBBY archive verification failed: {path}")


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / "recovery"
        pending = tuple(recovery.rglob("*.json")) if recovery.exists() else ()
        if pending:
            raise RuntimeError(
                "A market-store recovery marker blocks BBBY writes: "
                + ", ".join(str(item) for item in pending)
            )
        transactions = repository.root / "transactions"
        interrupted = []
        if transactions.exists():
            for item in transactions.rglob("*.json"):
                try:
                    status = str(json.loads(item.read_bytes()).get("status") or "")
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    interrupted.append(item)
        if interrupted:
            raise RuntimeError(
                "Interrupted BBBY transaction requires recovery: "
                + ", ".join(str(item) for item in interrupted)
            )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_journal(path: Path, value: dict[str, Any]) -> None:
    write_atomic(path, _canonical_json_bytes(value))


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: dict[str, bytes],
    planned_versions: dict[str, str],
    committed_release_version: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            ours = observed.version == committed_release_version or all(
                observed.dataset_versions.get(name) == version
                for name, version in planned_versions.items()
            )
            if not ours:
                raise RuntimeError(f"Unexpected release during rollback: {observed.version}")
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
        if repository.objects.get("releases/current.json").data != old_release_bytes:
            raise RuntimeError("Release rollback verification failed.")
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            if current.data != old_pointer_bytes[dataset]:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"Unexpected pointer during rollback: {pointer.version}"
                    )
                repository.objects.put(
                    key, old_pointer_bytes[dataset], if_match=current.etag
                )
            if repository.objects.get(key).data != old_pointer_bytes[dataset]:
                raise RuntimeError("Pointer rollback verification failed.")
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_release_unchanged(
    repository: LocalDatasetRepository,
    release: DataRelease,
    etag: str | None,
) -> None:
    current, current_etag = repository.current_release()
    if current is None or current.version != release.version or current_etag != etag:
        raise RuntimeError("Current release changed after BBBY preflight.")


def apply_repair(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> dict[str, Any]:
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
                raise RuntimeError(f"{dataset} pointer changed before BBBY apply.")
            old_pointers[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        session = prepared.release.completed_session.replace("-", "")
        planned = {
            dataset: f"bbby-identity-{session}-{transaction_id}-{dataset}"
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/bbby-identity-repair"
            / f"{transaction_id}.json"
        )
        journal = {
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                name: base64.b64encode(value).decode("ascii")
                for name, value in old_pointers.items()
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
                prepared.release.completed_session,
            )
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="warn",
                    metadata={
                        "operation": "repair_us_bbby_identity",
                        "old_isin": OLD_ISIN,
                        "current_isin": CURRENT_ISIN,
                        "strict_bbby_gate": "passed",
                    },
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version

            # Re-read the written current pointers before making a new release.
            written = {
                name: repository.read_frame(name, planned[name])
                for name in WRITE_DATASETS
            }
            validate_candidate_frames(
                LocalPreflight(
                    existing=prepared.frames,
                    pointer_etags=prepared.pointer_etags,
                    catalog=CatalogProof({}, {}, {}, "", ""),
                    contaminated_price_hashes=frozenset(),
                    contamination_overlap_rows=0,
                    already_repaired=True,
                ),
                written,
                prepared.archive_artifacts,
                completed_session=prepared.release.completed_session,
                expected_old=_expected_sessions(OLD_START, OLD_LAST_TRADING_DATE),
            )
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=(DataQuality.DEGRADED if prepared.warnings else DataQuality.VALID),
                warnings=prepared.warnings,
                expected_etag=prepared.release_etag,
            )
            current, _ = repository.current_release()
            if current is None or current.to_bytes() != committed.to_bytes():
                raise RuntimeError("Committed BBBY release is not current.")
            for dataset, version in committed.dataset_versions.items():
                pointer, _ = repository.current_pointer(dataset)
                if pointer is None or pointer.version != version:
                    raise RuntimeError(f"BBBY committed pointer mismatch: {dataset}")
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
                "transaction_id": transaction_id,
                "new_release_version": committed.version,
                "quality": committed.quality,
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=(committed.version if committed else ""),
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
                    / "recovery/bbby-identity-repair"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    f"BBBY rollback failed; recovery marker blocks writes: {recovery}; "
                    f"errors={rollback_errors}"
                ) from original
            raise


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[[str | Path], LocalDatasetRepository] = LocalDatasetRepository,
    eodhd_source_factory: Callable[..., EodhdBbbyqSource] = EodhdBbbyqSource,
    wiki_source_factory: Callable[..., PinnedWikiBbbySource] = PinnedWikiBbbySource,
    official_source_factory: Callable[..., OfficialEvidenceSource] = OfficialEvidenceSource,
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required for BBBY repair.")
    if args.offline_plan:
        return build_offline_plan(repository, release)
    preflight = build_local_preflight(repository, release)
    if preflight.already_repaired:
        return {
            "status": "already_repaired",
            "release_version": release.version,
            "http_attempts": 0,
            "old_security_id": OLD_SECURITY_ID,
            "current_security_id": CURRENT_SECURITY_ID,
        }
    eodhd_source = eodhd_source_factory(
        repository.root / "state/bbby-identity/eodhd",
        allow_http=bool(args.fetch_eodhd_bbbyq),
    )
    wiki_source = wiki_source_factory(
        repository.root / "state/bbby-identity/wiki",
        allow_http=bool(args.fetch_wiki),
    )
    official_source = official_source_factory(
        repository.root / "state/bbby-identity/sec",
        allow_http=bool(args.fetch_official_evidence),
    )
    prepared = prepare_repair(
        repository,
        release,
        release_etag,
        preflight,
        eodhd=eodhd_source.fetch(),
        wiki=wiki_source.fetch(),
        official=official_source.load(),
    )
    return apply_repair(repository, prepared) if args.apply else prepared.summary


def main(argv: list[str] | None = None) -> int:
    result = run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
