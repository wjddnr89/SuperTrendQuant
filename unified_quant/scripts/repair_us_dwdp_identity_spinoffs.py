#!/usr/bin/env python3
"""Repair the 2017-2019 DuPont/DowDuPont identity and distribution chain.

The repair itself is offline.  Three already-cached SEC filings are pinned by
exact URL and SHA-256 and prove:

* legacy E. I. du Pont -> DowDuPont at 1.282 shares on 2017-09-01;
* the 1-for-3 Dow distribution completed on 2019-04-01; and
* the 1-for-3 Corteva distribution, 1-for-3 reverse split, and DWDP -> DD
  symbol transition completed on 2019-06-01/03.

The current EODHD ``DD.US`` endpoint is a modern-DD back-adjusted lineage: its
pre-2019 rows are *not* the raw legacy DuPont listing.  Those rows are always
discarded.  A separate immutable ``DD_old.US`` three-endpoint cache is required
and is cross-checked against the commit-pinned raw Quandl WIKI DD history before
any plan can succeed.  The known 699-byte Boris/Kaggle ``dd.us.txt`` pseudo
history is explicitly rejected.  Raw request bytes remain in ``source_archive``.
Three provider pseudo-bars left on the retired DWDP endpoint are removed from
the published price/factor tables.
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
import math
import os
import re
import shutil
import threading
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlencode

import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.ingest import EodhdClient
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
from supertrend_quant.market_store.yahoo_chart import parse_yahoo_chart_json


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_SEC_CACHE = DEFAULT_CACHE_ROOT / "state/sec_lifecycle"
DEFAULT_LEGACY_DD_CACHE = DEFAULT_CACHE_ROOT / "state/dwdp-legacy-dd"

LEGACY_DD_ID = "US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1"
CURRENT_DD_ID = "US:EODHD:06c72bf7-3123-53cf-8d38-edaa8d5ecf63"
DOW_ID = "US:EODHD:97d908f3-f2ea-52f4-b179-a5fc616014b6"
CTVA_ID = "US:EODHD:80f98754-6fdc-5892-bd6f-3361a23e5fc8"
DWDP_ID = "US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3"

LEGACY_DD_FIRST = "2015-01-02"
LEGACY_DD_LAST = "2017-08-31"
DWDP_FIRST = "2017-09-01"
DOW_FIRST_WHEN_ISSUED = "2019-03-20"
DOW_DISTRIBUTION = "2019-04-01"
CTVA_FIRST_WHEN_ISSUED = "2019-05-24"
DWDP_LAST = "2019-05-31"
DWDP_SYMBOL_LAST = "2019-06-02"
CTVA_DISTRIBUTION = "2019-06-01"
DD_FIRST_REGULAR_WAY = "2019-06-03"
DWDP_PSEUDO_SESSIONS = ("2019-06-03", "2019-06-04", "2019-06-28")

DD_EOD_URL = "https://eodhd.com/api/eod/DD.US?from=2015-01-01&to=2026-07-15"
DD_EOD_SHA256 = "95718837bfc6d353c7da70eb46fb9ec5b939b2fa2a4a8213ec92cd42c5a836f1"
DWDP_EOD_URL = "https://eodhd.com/api/eod/DWDP.US?from=2015-01-01&to=2026-07-15"
DWDP_EOD_SHA256 = "b4b90b31bd87fbdde9b5e1c22698c775fa3282fd29475e576a49ca16abd85209"

LEGACY_DD_PROVIDER_SYMBOL = "DD_old.US"
LEGACY_DD_REQUEST_START = "2015-01-01"
LEGACY_DD_REQUEST_END = LEGACY_DD_LAST
LEGACY_DD_ENDPOINTS = ("eod", "div", "splits")
MAX_LEGACY_DD_HTTP_ATTEMPTS = len(LEGACY_DD_ENDPOINTS)

WIKI_DD_SOURCE = "quandl_wiki_arnc_commit_snapshot"
WIKI_DD_COMMIT = "ce85e08888de5b8c4f6fd8c2d03bba85a9034f64"
WIKI_DD_URL = (
    "https://media.githubusercontent.com/media/kmfranz/trading_pairs/"
    f"{WIKI_DD_COMMIT}/WIKI_PRICES.csv"
)
WIKI_DD_SHA256 = "dd5127aae478d270150904fcbad6e96a42e461e13c3d48a1587edb9b89cea43e"
WIKI_DD_SIZE = 235_562_224
WIKI_DD_FIRST = LEGACY_DD_FIRST
WIKI_DD_LAST = "2016-12-19"
WIKI_DD_EXPECTED_ROWS = 490
# The pinned WIKI table has six documented DD omissions inside the otherwise
# valid XNYS interval.  They are permitted only for independent overlap; the
# DD_old.US primary series must still contain all 672 exchange sessions.
WIKI_DD_MISSING_SESSIONS = (
    "2015-01-07",
    "2015-01-09",
    "2015-01-16",
    "2015-01-27",
    "2015-07-13",
    "2016-03-10",
)
WIKI_DD_KNOWN_INCOHERENT_BAR = {
    "session": "2015-07-16",
    "open": 59.16,
    "high": 60.0,
    "low": 60.0,
    "close": 59.77,
    "volume": 6_462_524.0,
}

BORIS_DD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/"
    "price-volume-data-for-all-us-stocks-etfs/Data%2FStocks%2Fdd.us.txt"
    "?datasetVersionNumber=3"
)
BORIS_DD_REJECTED_SHA256 = (
    "51245e1fbafe994229175bad5860449d79e21a65117e7936ad25d7fd0b14397f"
)
BORIS_DD_REJECTED_SIZE = 699

STOOQ_DD_SOURCE = "stooq_legacy_dd_raw_csv"
STOOQ_DD_URL = (
    "https://stooq.com/q/d/l/?s=dd.us&d1=20150101&d2=20170831&i=d"
)
MAX_STOOQ_DD_HTTP_ATTEMPTS = 1
STOOQ_DD_FETCH_DISABLED_AFTER_AUDIT = True
STOOQ_DD_AUDITED_HTTP_STATUS = 404

YAHOO_DD_SOURCE = "yahoo_chart_legacy_dd_raw_quote"
YAHOO_DD_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/DD"
    "?period1=1420070400&period2=1504224000&interval=1d"
    "&events=history&includeAdjustedClose=true"
)
MAX_YAHOO_DD_HTTP_ATTEMPTS = 1

KAGGLE_WIKI_DD_SOURCE = "kaggle_frozen_quandl_wiki_mirror"
KAGGLE_WIKI_DD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/marketneutral/"
    "quandl-wiki-prices-us-equites/WIKI_PRICES.csv"
)
# Filled only after the one exact full response has been fetched and audited.
# A blank pin deliberately keeps plan/apply fail-closed while still allowing
# the probe command to report the immutable response hash and size.
KAGGLE_WIKI_FULL_SHA256 = (
    "36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae"
)
KAGGLE_WIKI_FULL_SIZE = 463_184_323
KAGGLE_WIKI_DD_SEGMENT_SHA256 = (
    "36bc4a610a64882f576cecff4f73fb9022e19083664e8c3383d75b25e40bc77a"
)
KAGGLE_WIKI_DD_SEGMENT_ROWS = 672
KAGGLE_WIKI_DD_TOTAL_ROWS = 14_014
KAGGLE_WIKI_DD_FIRST_AVAILABLE = "1962-01-02"
KAGGLE_WIKI_DD_LAST_AVAILABLE = LEGACY_DD_LAST
KAGGLE_WIKI_DD_TERMINAL = {
    "open": 83.54,
    "high": 85.16,
    "low": 83.31,
    "close": 83.93,
    "volume": 33_154_128.0,
}
MAX_KAGGLE_WIKI_HTTP_ATTEMPTS = 1
MAX_KAGGLE_WIKI_RESPONSE_BYTES = 5 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class EvidenceSpec:
    key: str
    url: str
    sha256: str
    retrieved_at: str
    required_phrases: tuple[str, ...]


OFFICIAL_EVIDENCE = (
    EvidenceSpec(
        key="legacy_dd_merger",
        url=(
            "https://www.sec.gov/Archives/edgar/data/30554/"
            "000119312517274840/0001193125-17-274840.txt"
        ),
        sha256="098828aa2714df3fdd52a18b1fffb91d6a72865ff8dd4e94e84f7bc079cf0e64",
        retrieved_at="2026-07-17T18:13:39.099668Z",
        required_phrases=(
            "converted into the right to receive 1.2820 fully paid",
            "suspended from trading on the nyse prior to the open of trading on september 1, 2017",
        ),
    ),
    EvidenceSpec(
        key="dow_distribution",
        url=(
            "https://www.sec.gov/Archives/edgar/data/1751788/"
            "000175178819000013/0001751788-19-000013.txt"
        ),
        sha256="d2b6f7fb864a447dbfedb007219123aa0ab020115df51908ff44d07a521398c3",
        retrieved_at="2026-07-17T18:14:07.927535Z",
        required_phrases=(
            "on april 1, 2019, dowdupont inc.",
            "received one share of dow inc. common stock",
            "for every three shares of dowdupont common stock",
        ),
    ),
    EvidenceSpec(
        key="corteva_dd_completion",
        url=(
            "https://www.sec.gov/Archives/edgar/data/1666700/"
            "000119312519163322/0001193125-19-163322.txt"
        ),
        sha256="ae9343609e64dcd8421f11462b8782cc8db38a130e03c983714f3c10ba8db311",
        retrieved_at="2026-07-17T20:03:16.620717Z",
        required_phrases=(
            "on june 1, 2019, the company effected the corteva distribution",
            "one (1) share of corteva common stock for every three (3) shares",
            "one-for-three reverse stock split",
            "under the new symbol \u201cdd\u201d beginning on june 3, 2019",
        ),
    ),
)

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
REQUIRED_DATASETS = WRITE_DATASETS
AFFECTED_FACTOR_IDS = frozenset({LEGACY_DD_ID, CURRENT_DD_ID, DWDP_ID})


@dataclass(frozen=True)
class EvidenceBundle:
    artifacts: tuple[SourceArtifact, ...]
    specs: tuple[EvidenceSpec, ...]

    def artifact(self, key: str) -> SourceArtifact:
        matches = [
            artifact
            for artifact, spec in zip(self.artifacts, self.specs, strict=True)
            if spec.key == key
        ]
        if len(matches) != 1:
            raise ValueError(f"Official DWDP evidence is not unique: {key}")
        return matches[0]


@dataclass(frozen=True)
class LegacyDdEvidence:
    """Validated raw legacy-DD rows and the exact provider artifacts behind them."""

    prices: pd.DataFrame
    actions: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    wiki_url: str
    wiki_hash: str
    overlap_rows: int
    http_attempts: int = 0
    file_artifacts: tuple["FileSourceArtifact", ...] = ()
    negative_artifact_count: int = 0


@dataclass(frozen=True)
class FileSourceArtifact:
    """Large exact response retained as a deterministic gzip file, not RAM bytes."""

    source: str
    source_url: str
    retrieved_at: str
    source_hash: str
    content_type: str
    content_size: int
    gzip_path: Path


@dataclass(frozen=True)
class PreparedDwdpRepair:
    frames: dict[str, pd.DataFrame]
    artifacts: tuple[SourceArtifact, ...]
    official_evidence: EvidenceBundle
    legacy_evidence: LegacyDdEvidence
    file_artifacts: tuple[FileSourceArtifact, ...]
    summary: dict[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _event_id(*values: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(list(values)))


def _normalized_filing_text(content: bytes) -> str:
    value = content.decode("utf-8", errors="replace")
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value).replace("\u00a0", " ")
    return " ".join(value.lower().split())


def _sec_cache_path(cache_dir: Path, url: str) -> Path:
    # SecEdgarLifecycleSource hashes the exact encoded URL with the empty-query
    # suffix.  Keeping this derivation here makes cache-only behavior explicit.
    return cache_dir / f"{hashlib.sha256((url + '?').encode()).hexdigest()}.bin"


def load_official_evidence(
    cache_dir: str | Path,
    *,
    specs: tuple[EvidenceSpec, ...] = OFFICIAL_EVIDENCE,
) -> EvidenceBundle:
    root = Path(cache_dir)
    artifacts: list[SourceArtifact] = []
    for spec in specs:
        path = _sec_cache_path(root, spec.url)
        if not path.is_file():
            raise FileNotFoundError(
                f"Pinned SEC cache payload is missing for {spec.key}: {path}"
            )
        content = path.read_bytes()
        actual = sha256_bytes(content)
        if actual != spec.sha256:
            raise ValueError(
                f"Pinned SEC payload hash changed for {spec.key}: "
                f"expected={spec.sha256}, actual={actual}"
            )
        text = _normalized_filing_text(content)
        missing = [phrase for phrase in spec.required_phrases if phrase not in text]
        if missing:
            raise ValueError(
                f"Pinned SEC payload no longer proves {spec.key}: {missing}"
            )
        artifacts.append(
            SourceArtifact(
                source="sec_edgar_filing",
                source_url=spec.url,
                retrieved_at=spec.retrieved_at,
                content=content,
                content_type="text/plain",
            )
        )
    return EvidenceBundle(tuple(artifacts), specs)


def _legacy_dd_public_url(endpoint: str) -> str:
    if endpoint not in LEGACY_DD_ENDPOINTS:
        raise ValueError(f"Unsupported legacy DD endpoint: {endpoint}")
    return (
        f"https://eodhd.com/api/{endpoint}/{LEGACY_DD_PROVIDER_SYMBOL}?"
        + urlencode(
            {
                "from": LEGACY_DD_REQUEST_START,
                "to": LEGACY_DD_REQUEST_END,
            }
        )
    )


class CappedLegacyDdClient(EodhdClient):
    """Exactly one HTTP attempt per endpoint and three attempts per run."""

    def __init__(
        self,
        *args: Any,
        max_attempts: int = MAX_LEGACY_DD_HTTP_ATTEMPTS,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if int(max_attempts) != MAX_LEGACY_DD_HTTP_ATTEMPTS:
            raise ValueError("Legacy DD EODHD cap must remain exactly three calls.")
        self.max_attempts = int(max_attempts)
        self._attempt_count = 0
        self._lock = threading.Lock()

    @property
    def attempt_count(self) -> int:
        with self._lock:
            return self._attempt_count

    def get_json(
        self, endpoint: str, *, params: dict[str, object] | None = None
    ) -> Any:
        safe_endpoint = "/" + endpoint.strip("/")
        query = {**(params or {}), "api_token": self.token, "fmt": "json"}
        with self._lock:
            if self._attempt_count >= self.max_attempts:
                raise RuntimeError("Legacy DD EODHD three-call cap reached before HTTP.")
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
                f"Legacy DD EODHD single attempt failed for {safe_endpoint}: {detail}"
            ) from None


class LegacyDdEndpointCache:
    """Immutable token-free cache for the three exact ``DD_old.US`` calls."""

    SCHEMA = "dwdp_legacy_dd_eodhd_raw/v1"

    def __init__(
        self,
        root: str | Path,
        *,
        allow_http: bool,
        client_factory: Callable[[], Any] = CappedLegacyDdClient,
    ):
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.client_factory = client_factory
        self.http_attempts = 0

    def path(self, endpoint: str) -> Path:
        return self.root / f"{sha256_bytes(_legacy_dd_public_url(endpoint).encode())}.json.gz"

    def _decode(self, endpoint: str, encoded: bytes) -> SourceArtifact:
        path = self.path(endpoint)
        try:
            envelope = json.loads(gzip.decompress(encoded))
            content = base64.b64decode(
                str(envelope["content_base64"]), validate=True
            )
        except Exception as exc:
            raise ValueError(f"Unreadable legacy DD endpoint cache: {path}") from exc
        expected_url = _legacy_dd_public_url(endpoint)
        if (
            envelope.get("schema") != self.SCHEMA
            or envelope.get("endpoint") != endpoint
            or envelope.get("provider_symbol") != LEGACY_DD_PROVIDER_SYMBOL
            or envelope.get("source_url") != expected_url
            or envelope.get("source_hash") != sha256_bytes(content)
        ):
            raise ValueError(f"Legacy DD endpoint cache identity mismatch: {path}")
        if (
            envelope.get("source_url") == BORIS_DD_URL
            or sha256_bytes(content) == BORIS_DD_REJECTED_SHA256
            or len(content) == BORIS_DD_REJECTED_SIZE
        ):
            raise ValueError("Known Boris DD pseudo-history is forbidden as legacy DD evidence.")
        return SourceArtifact(
            source=f"eodhd_{endpoint}",
            source_url=expected_url,
            retrieved_at=str(envelope.get("retrieved_at") or ""),
            content=content,
            content_type="application/json",
        )

    def get(self, endpoint: str) -> SourceArtifact | None:
        path = self.path(endpoint)
        return self._decode(endpoint, path.read_bytes()) if path.is_file() else None

    def _store(self, endpoint: str, artifact: SourceArtifact) -> SourceArtifact:
        if artifact.source_url != _legacy_dd_public_url(endpoint):
            raise ValueError("Legacy DD artifact URL is not the exact token-free URL.")
        envelope = {
            "schema": self.SCHEMA,
            "endpoint": endpoint,
            "provider_symbol": LEGACY_DD_PROVIDER_SYMBOL,
            "source_url": artifact.source_url,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
        }
        encoded = gzip.compress(_canonical_json_bytes(envelope), mtime=0)
        path = self.path(endpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            existing = self._decode(endpoint, path.read_bytes())
            if existing.content != artifact.content:
                raise RuntimeError(f"Immutable legacy DD cache changed: {path}")
            return existing
        write_atomic(path, encoded)
        return self._decode(endpoint, path.read_bytes())

    def load(self) -> tuple[tuple[SourceArtifact, ...], int]:
        cached = {endpoint: self.get(endpoint) for endpoint in LEGACY_DD_ENDPOINTS}
        missing = [endpoint for endpoint, artifact in cached.items() if artifact is None]
        if missing and not self.allow_http:
            calls = ", ".join(_legacy_dd_public_url(endpoint) for endpoint in missing)
            raise FileNotFoundError(
                "Safe legacy DD cache is absent; repair is fail-closed. "
                "Run once with --fetch-legacy-dd to make at most three exact calls: "
                + calls
            )
        if len(missing) > MAX_LEGACY_DD_HTTP_ATTEMPTS:
            raise RuntimeError("Legacy DD missing request count exceeds frozen cap.")
        client = self.client_factory() if missing else None
        params = {
            "from": LEGACY_DD_REQUEST_START,
            "to": LEGACY_DD_REQUEST_END,
        }
        for endpoint in missing:
            rows = client.get_json(
                f"{endpoint}/{LEGACY_DD_PROVIDER_SYMBOL}", params=params
            )
            content = json.dumps(
                rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
            artifact = SourceArtifact(
                source=f"eodhd_{endpoint}",
                source_url=_legacy_dd_public_url(endpoint),
                retrieved_at=utc_now_iso(),
                content=content,
                content_type="application/json",
            )
            cached[endpoint] = self._store(endpoint, artifact)
        self.http_attempts += (
            int(getattr(client, "attempt_count", len(missing))) if client else 0
        )
        if self.http_attempts > MAX_LEGACY_DD_HTTP_ATTEMPTS:
            raise RuntimeError("Legacy DD EODHD three-call cap was exceeded.")
        artifacts = tuple(cached[endpoint] for endpoint in LEGACY_DD_ENDPOINTS)
        if any(artifact is None for artifact in artifacts):
            raise RuntimeError("Legacy DD endpoint cache did not fill completely.")
        return tuple(artifact for artifact in artifacts if artifact is not None), self.http_attempts

    def probe_eod(self) -> tuple[SourceArtifact, int]:
        """Fetch only the EOD endpoint once; actions remain deliberately absent."""

        existing = self.get("eod")
        if existing is not None:
            return existing, 0
        if not self.allow_http:
            raise FileNotFoundError(
                "Legacy DD EOD probe cache is absent; explicitly use "
                "--probe-legacy-dd-eod for the one exact call."
            )
        client = self.client_factory()
        rows = client.get_json(
            f"eod/{LEGACY_DD_PROVIDER_SYMBOL}",
            params={
                "from": LEGACY_DD_REQUEST_START,
                "to": LEGACY_DD_REQUEST_END,
            },
        )
        artifact = SourceArtifact(
            source="eodhd_eod",
            source_url=_legacy_dd_public_url("eod"),
            retrieved_at=utc_now_iso(),
            content=json.dumps(
                rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode(),
            content_type="application/json",
        )
        stored = self._store("eod", artifact)
        attempts = int(getattr(client, "attempt_count", 1))
        if attempts != 1:
            raise RuntimeError(f"Legacy DD EOD probe used {attempts} calls; expected one.")
        self.http_attempts = attempts
        return stored, attempts


@dataclass(frozen=True)
class StooqLegacyDdArtifact:
    artifact: SourceArtifact
    http_status: int
    content_type: str


class StooqLegacyDdCache:
    """One-attempt immutable cache for the exact Stooq legacy-DD CSV request."""

    SCHEMA = "dwdp_legacy_dd_stooq_raw/v1"

    def __init__(
        self,
        root: str | Path,
        *,
        allow_http: bool,
        session_factory: Callable[[], Any] | None = None,
    ):
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.session_factory = session_factory
        self.http_attempts = 0

    def path(self) -> Path:
        return self.root / f"{sha256_bytes(STOOQ_DD_URL.encode())}.json.gz"

    def _decode(self, encoded: bytes) -> StooqLegacyDdArtifact:
        path = self.path()
        try:
            envelope = json.loads(gzip.decompress(encoded))
            content = base64.b64decode(
                str(envelope["content_base64"]), validate=True
            )
            http_status = int(envelope["http_status"])
        except Exception as exc:
            raise ValueError(f"Unreadable Stooq legacy DD cache: {path}") from exc
        if (
            envelope.get("schema") != self.SCHEMA
            or envelope.get("source_url") != STOOQ_DD_URL
            or envelope.get("source_hash") != sha256_bytes(content)
        ):
            raise ValueError(f"Stooq legacy DD cache identity mismatch: {path}")
        if (
            sha256_bytes(content) == BORIS_DD_REJECTED_SHA256
            or len(content) == BORIS_DD_REJECTED_SIZE
        ):
            raise ValueError("Known Boris DD pseudo-history cannot enter the Stooq cache.")
        content_type = str(envelope.get("content_type") or "text/csv")
        return StooqLegacyDdArtifact(
            artifact=SourceArtifact(
                source=STOOQ_DD_SOURCE,
                source_url=STOOQ_DD_URL,
                retrieved_at=str(envelope.get("retrieved_at") or ""),
                content=content,
                content_type=content_type,
            ),
            http_status=http_status,
            content_type=content_type,
        )

    def get(self) -> StooqLegacyDdArtifact | None:
        path = self.path()
        return self._decode(path.read_bytes()) if path.is_file() else None

    def _store(
        self,
        artifact: SourceArtifact,
        *,
        http_status: int,
        content_type: str,
    ) -> StooqLegacyDdArtifact:
        if artifact.source_url != STOOQ_DD_URL:
            raise ValueError("Stooq legacy DD artifact URL is not exact.")
        envelope = {
            "schema": self.SCHEMA,
            "source_url": STOOQ_DD_URL,
            "retrieved_at": artifact.retrieved_at,
            "http_status": int(http_status),
            "content_type": content_type,
            "source_hash": artifact.source_hash,
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
        }
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(_canonical_json_bytes(envelope), mtime=0)
        if path.is_file():
            existing = self._decode(path.read_bytes())
            if (
                existing.artifact.content != artifact.content
                or existing.http_status != int(http_status)
            ):
                raise RuntimeError(f"Immutable Stooq legacy DD cache changed: {path}")
            return existing
        write_atomic(path, encoded)
        return self._decode(path.read_bytes())

    def load(self) -> tuple[StooqLegacyDdArtifact, int]:
        existing = self.get()
        if existing is not None:
            return existing, 0
        if not self.allow_http:
            raise FileNotFoundError(
                "Safe Stooq legacy DD cache is absent; explicitly use "
                "--probe-legacy-dd-stooq for the one exact CSV call."
            )
        if STOOQ_DD_FETCH_DISABLED_AFTER_AUDIT:
            raise RuntimeError(
                "Stooq legacy DD exact URL already returned audited HTTP 404; "
                "a retry is forbidden."
            )
        if self.http_attempts >= MAX_STOOQ_DD_HTTP_ATTEMPTS:
            raise RuntimeError("Stooq legacy DD one-call cap reached before HTTP.")
        if self.session_factory is None:
            import requests

            session = requests.Session()
        else:
            session = self.session_factory()
        self.http_attempts += 1
        try:
            response = session.get(STOOQ_DD_URL, timeout=120)
            content = bytes(response.content)
            status = int(getattr(response, "status_code", 200))
            content_type = str(
                getattr(response, "headers", {}).get("Content-Type", "text/csv")
            )
        except Exception as exc:
            raise RuntimeError(
                f"Stooq legacy DD single attempt failed: {type(exc).__name__}"
            ) from None
        artifact = SourceArtifact(
            source=STOOQ_DD_SOURCE,
            source_url=STOOQ_DD_URL,
            retrieved_at=utc_now_iso(),
            content=content,
            content_type=content_type,
        )
        return (
            self._store(
                artifact, http_status=status, content_type=content_type
            ),
            self.http_attempts,
        )


@dataclass(frozen=True)
class YahooLegacyDdArtifact:
    artifact: SourceArtifact
    http_status: int
    content_type: str


class YahooLegacyDdCache:
    """One-attempt exact-byte cache for the bounded Yahoo DD chart request."""

    SCHEMA = "dwdp_legacy_dd_yahoo_chart_raw/v1"

    def __init__(
        self,
        root: str | Path,
        *,
        allow_http: bool,
        session_factory: Callable[[], Any] | None = None,
    ):
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.session_factory = session_factory
        self.http_attempts = 0

    def path(self) -> Path:
        return self.root / f"{sha256_bytes(YAHOO_DD_URL.encode())}.json.gz"

    def _decode(self, encoded: bytes) -> YahooLegacyDdArtifact:
        path = self.path()
        try:
            envelope = json.loads(gzip.decompress(encoded))
            content = base64.b64decode(
                str(envelope["content_base64"]), validate=True
            )
            http_status = int(envelope["http_status"])
        except Exception as exc:
            raise ValueError(f"Unreadable Yahoo legacy DD cache: {path}") from exc
        if (
            envelope.get("schema") != self.SCHEMA
            or envelope.get("source_url") != YAHOO_DD_URL
            or envelope.get("source_hash") != sha256_bytes(content)
        ):
            raise ValueError(f"Yahoo legacy DD cache identity mismatch: {path}")
        if (
            sha256_bytes(content) == BORIS_DD_REJECTED_SHA256
            or len(content) == BORIS_DD_REJECTED_SIZE
        ):
            raise ValueError("Known Boris DD pseudo-history cannot enter Yahoo cache.")
        return YahooLegacyDdArtifact(
            artifact=SourceArtifact(
                source=YAHOO_DD_SOURCE,
                source_url=YAHOO_DD_URL,
                retrieved_at=str(envelope.get("retrieved_at") or ""),
                content=content,
                content_type=str(envelope.get("content_type") or "application/json"),
            ),
            http_status=http_status,
            content_type=str(envelope.get("content_type") or "application/json"),
        )

    def get(self) -> YahooLegacyDdArtifact | None:
        path = self.path()
        return self._decode(path.read_bytes()) if path.is_file() else None

    def _store(
        self,
        artifact: SourceArtifact,
        *,
        http_status: int,
        content_type: str,
    ) -> YahooLegacyDdArtifact:
        envelope = {
            "schema": self.SCHEMA,
            "source_url": YAHOO_DD_URL,
            "retrieved_at": artifact.retrieved_at,
            "http_status": int(http_status),
            "content_type": content_type,
            "source_hash": artifact.source_hash,
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
        }
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(_canonical_json_bytes(envelope), mtime=0)
        if path.is_file():
            existing = self._decode(path.read_bytes())
            if (
                existing.artifact.content != artifact.content
                or existing.http_status != int(http_status)
            ):
                raise RuntimeError(f"Immutable Yahoo legacy DD cache changed: {path}")
            return existing
        write_atomic(path, encoded)
        return self._decode(path.read_bytes())

    def load(self) -> tuple[YahooLegacyDdArtifact, int]:
        existing = self.get()
        if existing is not None:
            return existing, 0
        if not self.allow_http:
            raise FileNotFoundError(
                "Safe Yahoo legacy DD cache is absent; explicitly use "
                "--probe-legacy-dd-yahoo for the one exact chart call."
            )
        if self.http_attempts >= MAX_YAHOO_DD_HTTP_ATTEMPTS:
            raise RuntimeError("Yahoo legacy DD one-call cap reached before HTTP.")
        if self.session_factory is None:
            import requests

            session = requests.Session()
        else:
            session = self.session_factory()
        self.http_attempts += 1
        try:
            response = session.get(
                YAHOO_DD_URL,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "SuperTrendQuant personal legacy validation/1.0",
                },
                timeout=120,
            )
            content = bytes(response.content)
            status = int(getattr(response, "status_code", 200))
            content_type = str(
                getattr(response, "headers", {}).get(
                    "Content-Type", "application/json"
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Yahoo legacy DD single attempt failed: {type(exc).__name__}"
            ) from None
        artifact = SourceArtifact(
            source=YAHOO_DD_SOURCE,
            source_url=YAHOO_DD_URL,
            retrieved_at=utc_now_iso(),
            content=content,
            content_type=content_type,
        )
        return (
            self._store(
                artifact, http_status=status, content_type=content_type
            ),
            self.http_attempts,
        )


@dataclass(frozen=True)
class KaggleWikiCachedFile:
    artifact: FileSourceArtifact
    http_status: int
    content_type: str


class KaggleWikiLegacyDdCache:
    """One-attempt streaming cache for the large frozen final WIKI CSV."""

    SCHEMA = "dwdp_legacy_dd_kaggle_quandl_wiki_raw/v1"

    def __init__(
        self,
        root: str | Path,
        *,
        allow_http: bool,
        session_factory: Callable[[], Any] | None = None,
    ):
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.session_factory = session_factory
        self.http_attempts = 0

    def blob_path(self) -> Path:
        return self.root / f"{sha256_bytes(KAGGLE_WIKI_DD_URL.encode())}.csv.gz"

    def metadata_path(self) -> Path:
        return self.root / f"{sha256_bytes(KAGGLE_WIKI_DD_URL.encode())}.json"

    @staticmethod
    def _stream_hash(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        try:
            with gzip.open(path, "rb") as handle:
                while chunk := handle.read(4 * 1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    if size > MAX_KAGGLE_WIKI_RESPONSE_BYTES:
                        raise ValueError("Kaggle WIKI response exceeds the frozen size cap.")
        except (EOFError, OSError) as exc:
            raise ValueError(f"Kaggle WIKI gzip cache is truncated: {path}") from exc
        return digest.hexdigest(), size

    def _decode(self) -> KaggleWikiCachedFile:
        blob = self.blob_path()
        metadata_path = self.metadata_path()
        if blob.is_file() != metadata_path.is_file():
            raise ValueError("Kaggle WIKI cache is partial; blob and metadata are both required.")
        if not blob.is_file():
            raise FileNotFoundError("Kaggle WIKI full-response cache is absent.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            http_status = int(metadata["http_status"])
            content_size = int(metadata["content_size"])
        except Exception as exc:
            raise ValueError("Kaggle WIKI cache metadata is unreadable.") from exc
        if (
            metadata.get("schema") != self.SCHEMA
            or metadata.get("source_url") != KAGGLE_WIKI_DD_URL
        ):
            raise ValueError("Kaggle WIKI cache metadata identity changed.")
        actual_hash, actual_size = self._stream_hash(blob)
        if (
            actual_hash != str(metadata.get("source_hash") or "")
            or actual_size != content_size
        ):
            raise ValueError("Kaggle WIKI complete response hash/size changed.")
        artifact = FileSourceArtifact(
            source=KAGGLE_WIKI_DD_SOURCE,
            source_url=KAGGLE_WIKI_DD_URL,
            retrieved_at=str(metadata.get("retrieved_at") or ""),
            source_hash=actual_hash,
            content_type=str(metadata.get("content_type") or "text/csv"),
            content_size=actual_size,
            gzip_path=blob,
        )
        return KaggleWikiCachedFile(
            artifact=artifact,
            http_status=http_status,
            content_type=artifact.content_type,
        )

    def get(self) -> KaggleWikiCachedFile | None:
        if not self.blob_path().exists() and not self.metadata_path().exists():
            return None
        return self._decode()

    def load(self) -> tuple[KaggleWikiCachedFile, int]:
        existing = self.get()
        if existing is not None:
            return existing, 0
        if not self.allow_http:
            raise FileNotFoundError(
                "Safe frozen Kaggle WIKI cache is absent; explicitly use "
                "--probe-legacy-dd-kaggle-wiki for the one complete streaming call."
            )
        if self.http_attempts >= MAX_KAGGLE_WIKI_HTTP_ATTEMPTS:
            raise RuntimeError("Kaggle WIKI one-call cap reached before HTTP.")
        if self.session_factory is None:
            import requests

            session = requests.Session()
        else:
            session = self.session_factory()
        self.http_attempts += 1
        try:
            response = session.get(
                KAGGLE_WIKI_DD_URL,
                headers={
                    "Accept": "text/csv,application/octet-stream",
                    "User-Agent": "SuperTrendQuant personal provenance archive/1.0",
                },
                stream=True,
                timeout=(30, 600),
            )
            status = int(getattr(response, "status_code", 200))
            content_type = str(
                getattr(response, "headers", {}).get(
                    "Content-Type", "application/octet-stream"
                )
            )
            chunks = response.iter_content(chunk_size=4 * 1024 * 1024)
            temporary = self.blob_path().with_name(
                f".{self.blob_path().name}.{uuid.uuid4().hex}.tmp"
            )
            self.root.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with temporary.open("wb") as raw_handle:
                with gzip.GzipFile(
                    fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
                ) as compressed:
                    for chunk in chunks:
                        if not chunk:
                            continue
                        payload = bytes(chunk)
                        size += len(payload)
                        if size > MAX_KAGGLE_WIKI_RESPONSE_BYTES:
                            raise RuntimeError(
                                "Kaggle WIKI response exceeded the 5 GiB cap."
                            )
                        digest.update(payload)
                        compressed.write(payload)
        except Exception as exc:
            if "temporary" in locals():
                temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Kaggle WIKI single streaming attempt failed: {type(exc).__name__}"
            ) from None
        if self.blob_path().exists():
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Kaggle WIKI immutable blob appeared concurrently.")
        os.replace(temporary, self.blob_path())
        metadata = {
            "schema": self.SCHEMA,
            "source_url": KAGGLE_WIKI_DD_URL,
            "retrieved_at": utc_now_iso(),
            "http_status": status,
            "content_type": content_type,
            "source_hash": digest.hexdigest(),
            "content_size": size,
        }
        write_atomic(self.metadata_path(), _canonical_json_bytes(metadata))
        return self._decode(), self.http_attempts


def _expected_xnys_sessions(start: str, end: str) -> tuple[str, ...]:
    import exchange_calendars as xcals

    return tuple(
        pd.Timestamp(value).date().isoformat()
        for value in xcals.get_calendar("XNYS").sessions_in_range(start, end)
    )


def _provider_archive_path(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    *,
    url: str,
    source_hash: str,
) -> Path:
    row = _one_row(
        source_archive,
        source_archive["source_url"].astype(str).eq(url)
        & source_archive["source_hash"].astype(str).eq(source_hash),
        f"source archive row for {url}",
    )
    root = repository.root.resolve()
    path = (root / str(row["object_path"])).resolve()
    if path == root or root not in path.parents or not path.is_file():
        raise ValueError(f"Pinned archive path is unsafe or absent: {path}")
    return path


def _load_wiki_dd_rows(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
) -> pd.DataFrame:
    path = _provider_archive_path(
        repository,
        source_archive,
        url=WIKI_DD_URL,
        source_hash=WIKI_DD_SHA256,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with gzip.open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except Exception as exc:
        raise ValueError(f"Pinned WIKI archive is unreadable: {path}") from exc
    if digest.hexdigest() != WIKI_DD_SHA256 or size != WIKI_DD_SIZE:
        raise ValueError("Pinned WIKI full snapshot bytes changed.")

    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"ticker", "date", "open", "high", "low", "close", "volume"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError("Pinned WIKI CSV header changed.")
            for row in reader:
                if str(row.get("ticker") or "").upper() != "DD":
                    continue
                session = str(row.get("date") or "")
                if WIKI_DD_FIRST <= session <= WIKI_DD_LAST:
                    rows.append(
                        {
                            "session": session,
                            "open": row.get("open"),
                            "high": row.get("high"),
                            "low": row.get("low"),
                            "close": row.get("close"),
                            "volume": row.get("volume"),
                        }
                    )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Pinned WIKI DD rows cannot be parsed.") from exc
    frame = pd.DataFrame(rows)
    if len(frame) != WIKI_DD_EXPECTED_ROWS:
        raise ValueError(
            f"Pinned WIKI DD row count changed: {len(frame)} != {WIKI_DD_EXPECTED_ROWS}."
        )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any() or frame["session"].duplicated().any():
        raise ValueError("Pinned WIKI DD rows contain invalid or duplicate values.")
    expected_wiki_sessions = tuple(
        session
        for session in _expected_xnys_sessions(WIKI_DD_FIRST, WIKI_DD_LAST)
        if session not in WIKI_DD_MISSING_SESSIONS
    )
    if tuple(frame["session"].astype(str)) != expected_wiki_sessions:
        raise ValueError("Pinned WIKI DD exchange-session coverage changed.")
    coherent = (
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame["volume"].ge(0)
        & frame["high"].ge(frame[["open", "low", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "high", "close"]].min(axis=1))
    )
    incoherent = frame.loc[~coherent]
    if len(incoherent) != 1:
        raise ValueError("Pinned WIKI DD incoherent-bar inventory changed.")
    observed_bad_bar = incoherent.iloc[0]
    for column, expected in WIKI_DD_KNOWN_INCOHERENT_BAR.items():
        observed = (
            str(observed_bad_bar[column])
            if column == "session"
            else float(observed_bad_bar[column])
        )
        if observed != expected:
            raise ValueError("Pinned WIKI DD known bad bar changed.")
    return frame.reset_index(drop=True)


def _json_rows(artifact: SourceArtifact, *, endpoint: str) -> list[dict[str, Any]]:
    if artifact.source_url != _legacy_dd_public_url(endpoint):
        raise ValueError(f"Legacy DD {endpoint} URL is not exact.")
    if artifact.source_hash in {DD_EOD_SHA256, BORIS_DD_REJECTED_SHA256}:
        raise ValueError("Current DD.US or Boris DD bytes are forbidden for legacy DD.")
    try:
        value = json.loads(artifact.content)
    except Exception as exc:
        raise ValueError(f"Legacy DD {endpoint} payload is not JSON.") from exc
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Legacy DD {endpoint} payload is not a JSON row list.")
    return value


def _legacy_dd_prices(artifact: SourceArtifact) -> pd.DataFrame:
    rows = _json_rows(artifact, endpoint="eod")
    frame = pd.DataFrame(
        [
            {
                "security_id": LEGACY_DD_ID,
                "session": str(row.get("date") or ""),
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
            if row.get("date") and row.get("close") is not None
        ]
    )
    required = list(dataset_spec("daily_price_raw").required_columns)
    if "source_url" not in required:
        required.append("source_url")
    if frame.empty:
        raise ValueError("DD_old.US returned no legacy price rows.")
    frame = frame.loc[
        frame["session"].astype(str).between(LEGACY_DD_FIRST, LEGACY_DD_LAST)
    ].copy()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[required].isna().any().any() or frame["session"].duplicated().any():
        raise ValueError("DD_old.US contains invalid or duplicate legacy rows.")
    frame = frame.sort_values("session").reset_index(drop=True)
    if tuple(frame["session"].astype(str)) != _expected_xnys_sessions(
        LEGACY_DD_FIRST, LEGACY_DD_LAST
    ):
        raise ValueError("DD_old.US does not exactly cover the legacy DD sessions.")
    coherent = (
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame["volume"].ge(0)
        & frame["high"].ge(frame[["open", "low", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "high", "close"]].min(axis=1))
    )
    if not bool(coherent.all()):
        raise ValueError("DD_old.US contains invalid legacy OHLCV bars.")
    return frame.loc[:, required]


def _stooq_legacy_dd_prices(value: StooqLegacyDdArtifact) -> pd.DataFrame:
    artifact = value.artifact
    if artifact.source_url != STOOQ_DD_URL or artifact.source != STOOQ_DD_SOURCE:
        raise ValueError("Stooq legacy DD artifact identity is not exact.")
    if value.http_status != 200:
        raise ValueError(f"Stooq legacy DD returned HTTP {value.http_status}.")
    if (
        artifact.source_hash in {DD_EOD_SHA256, BORIS_DD_REJECTED_SHA256}
        or len(artifact.content) == BORIS_DD_REJECTED_SIZE
    ):
        raise ValueError("Current DD.US or Boris DD bytes are forbidden as Stooq evidence.")
    stripped = artifact.content.lstrip().lower()
    if not stripped or stripped.startswith((b"<", b"<!")) or b"no data" in stripped[:200]:
        raise ValueError("Stooq legacy DD response is empty, HTML, or no-data text.")
    try:
        raw = pd.read_csv(io.BytesIO(artifact.content))
    except Exception as exc:
        raise ValueError("Stooq legacy DD response is not a readable CSV.") from exc
    normalized = {str(column).strip().lower(): column for column in raw.columns}
    required_input = ("date", "open", "high", "low", "close", "volume")
    if any(column not in normalized for column in required_input):
        raise ValueError("Stooq legacy DD CSV header changed.")
    frame = pd.DataFrame(
        {
            "security_id": LEGACY_DD_ID,
            "session": raw[normalized["date"]].astype(str),
            "open": raw[normalized["open"]],
            "high": raw[normalized["high"]],
            "low": raw[normalized["low"]],
            "close": raw[normalized["close"]],
            "volume": raw[normalized["volume"]],
            "currency": "USD",
            "source": artifact.source,
            "source_url": artifact.source_url,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
        }
    )
    frame = frame.loc[
        frame["session"].astype(str).between(LEGACY_DD_FIRST, LEGACY_DD_LAST)
    ].copy()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    columns = list(dataset_spec("daily_price_raw").required_columns)
    if "source_url" not in columns:
        columns.append("source_url")
    if frame.empty or frame[columns].isna().any().any() or frame["session"].duplicated().any():
        raise ValueError("Stooq legacy DD CSV contains invalid or duplicate rows.")
    frame = frame.sort_values("session").reset_index(drop=True)
    if tuple(frame["session"].astype(str)) != _expected_xnys_sessions(
        LEGACY_DD_FIRST, LEGACY_DD_LAST
    ):
        raise ValueError(
            "Stooq legacy DD does not exactly cover 2015-01-02 through 2017-08-31."
        )
    coherent = (
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame["volume"].ge(0)
        & frame["high"].ge(frame[["open", "low", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "high", "close"]].min(axis=1))
    )
    if not bool(coherent.all()):
        raise ValueError("Stooq legacy DD contains invalid OHLCV bars.")
    return frame.loc[:, columns]


def _yahoo_legacy_dd_prices(value: YahooLegacyDdArtifact) -> pd.DataFrame:
    artifact = value.artifact
    if artifact.source_url != YAHOO_DD_URL or artifact.source != YAHOO_DD_SOURCE:
        raise ValueError("Yahoo legacy DD artifact identity is not exact.")
    if value.http_status != 200:
        raise ValueError(f"Yahoo legacy DD returned HTTP {value.http_status}.")
    if "json" not in value.content_type.lower() or artifact.content.lstrip().startswith(
        (b"<", b"<!")
    ):
        raise ValueError("Yahoo legacy DD response is HTML or non-JSON.")
    if artifact.source_hash in {DD_EOD_SHA256, BORIS_DD_REJECTED_SHA256}:
        raise ValueError("Current DD.US or Boris DD bytes are forbidden as Yahoo evidence.")
    try:
        payload = json.loads(artifact.content)
        result = payload["chart"]["result"]
        metadata = result[0]["meta"]
    except Exception as exc:
        raise ValueError("Yahoo legacy DD chart metadata is malformed.") from exc
    if (
        not isinstance(result, list)
        or len(result) != 1
        or str(metadata.get("dataGranularity") or "") != "1d"
    ):
        raise ValueError("Yahoo legacy DD response was not delivered at daily granularity.")
    parsed = parse_yahoo_chart_json(artifact.content, "DD")
    raw = parsed.bars.copy()
    if raw.empty:
        raise ValueError("Yahoo legacy DD chart has no raw quote bars.")
    raw["session"] = pd.to_datetime(raw["session"], errors="coerce").dt.date.astype(str)
    raw = raw.loc[
        raw["session"].between(LEGACY_DD_FIRST, LEGACY_DD_LAST)
    ].copy()
    frame = pd.DataFrame(
        {
            "security_id": LEGACY_DD_ID,
            "session": raw["session"],
            "open": raw["open"],
            "high": raw["high"],
            "low": raw["low"],
            "close": raw["close"],
            "volume": raw["volume"],
            "currency": "USD",
            "source": artifact.source,
            "source_url": artifact.source_url,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
        }
    ).sort_values("session").reset_index(drop=True)
    if tuple(frame["session"].astype(str)) != _expected_xnys_sessions(
        LEGACY_DD_FIRST, LEGACY_DD_LAST
    ):
        raise ValueError(
            "Yahoo raw DD quote does not exactly cover the legacy 2015-2017 tail."
        )
    columns = list(dataset_spec("daily_price_raw").required_columns)
    if "source_url" not in columns:
        columns.append("source_url")
    return frame.loc[:, columns]


@contextmanager
def _kaggle_wiki_csv_handle(value: KaggleWikiCachedFile):
    """Yield the exact CSV stream whether Kaggle returned CSV or a one-file ZIP."""

    temporary_zip: Path | None = None
    with gzip.open(value.artifact.gzip_path, "rb") as probe:
        magic = probe.read(4)
    if magic.startswith(b"PK\x03\x04"):
        temporary_zip = Path("/tmp") / f"dwdp-kaggle-wiki-{uuid.uuid4().hex}.zip"
        try:
            with gzip.open(value.artifact.gzip_path, "rb") as source, temporary_zip.open(
                "wb"
            ) as destination:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
            with zipfile.ZipFile(temporary_zip) as archive:
                files = [item for item in archive.infolist() if not item.is_dir()]
                if len(files) != 1:
                    raise ValueError("Kaggle WIKI ZIP must contain exactly one CSV file.")
                member = files[0]
                if Path(member.filename).name != "WIKI_PRICES.csv":
                    raise ValueError("Kaggle WIKI ZIP member identity changed.")
                if member.file_size > MAX_KAGGLE_WIKI_RESPONSE_BYTES:
                    raise ValueError("Kaggle WIKI CSV member exceeds the 5 GiB cap.")
                with archive.open(member, "r") as handle:
                    yield handle
        finally:
            if temporary_zip is not None:
                temporary_zip.unlink(missing_ok=True)
        return
    with gzip.open(value.artifact.gzip_path, "rb") as handle:
        yield handle


def _kaggle_wiki_legacy_dd_prices(
    value: KaggleWikiCachedFile,
    *,
    require_full_pin: bool,
) -> tuple[pd.DataFrame, SourceArtifact, dict[str, Any]]:
    if value.http_status != 200:
        raise ValueError(f"Kaggle WIKI mirror returned HTTP {value.http_status}.")
    if value.artifact.source_url != KAGGLE_WIKI_DD_URL:
        raise ValueError("Kaggle WIKI mirror URL is not exact.")
    if require_full_pin:
        if not KAGGLE_WIKI_FULL_SHA256 or KAGGLE_WIKI_FULL_SIZE <= 0:
            raise ValueError(
                "Kaggle WIKI complete-response hash/size await operator review; "
                "plan/apply remain blocked."
            )
        if (
            value.artifact.source_hash != KAGGLE_WIKI_FULL_SHA256
            or value.artifact.content_size != KAGGLE_WIKI_FULL_SIZE
        ):
            raise ValueError("Kaggle WIKI complete-response pin changed.")

    target_lines: list[bytes] = []
    all_dd_rows = 0
    all_first = ""
    all_last = ""
    with _kaggle_wiki_csv_handle(value) as handle:
        header = handle.readline()
        try:
            header_fields = next(csv.reader([header.decode("utf-8-sig")]))
        except Exception as exc:
            raise ValueError("Kaggle WIKI CSV header is unreadable.") from exc
        normalized_header = tuple(str(item).strip().lower() for item in header_fields)
        required_header = (
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        )
        if normalized_header[: len(required_header)] != required_header:
            raise ValueError("Kaggle WIKI CSV header/order changed.")
        for line in handle:
            if not line.startswith(b"DD,"):
                continue
            parts = line.split(b",", 2)
            if len(parts) < 3:
                raise ValueError("Kaggle WIKI DD line is malformed.")
            try:
                session = parts[1].decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError("Kaggle WIKI DD date is not ASCII.") from exc
            all_dd_rows += 1
            all_first = all_first or session
            all_last = session
            if LEGACY_DD_FIRST <= session <= LEGACY_DD_LAST:
                target_lines.append(line)
    if (
        all_dd_rows != KAGGLE_WIKI_DD_TOTAL_ROWS
        or all_first != KAGGLE_WIKI_DD_FIRST_AVAILABLE
        or all_last != KAGGLE_WIKI_DD_LAST_AVAILABLE
    ):
        raise ValueError(
            "Kaggle WIKI full DD inventory changed: "
            f"rows={all_dd_rows}, first={all_first}, last={all_last}."
        )
    segment_content = b"".join(target_lines)
    segment_hash = sha256_bytes(segment_content)
    if (
        len(target_lines) != KAGGLE_WIKI_DD_SEGMENT_ROWS
        or segment_hash != KAGGLE_WIKI_DD_SEGMENT_SHA256
    ):
        raise ValueError(
            "Kaggle WIKI exact DD line segment changed: "
            f"rows={len(target_lines)}, sha256={segment_hash}."
        )
    try:
        reader = csv.DictReader(
            io.StringIO((header + segment_content).decode("utf-8-sig"))
        )
        rows = list(reader)
    except Exception as exc:
        raise ValueError("Kaggle WIKI DD target segment is not valid CSV.") from exc
    frame = pd.DataFrame(rows)
    renamed = {column: str(column).strip().lower() for column in frame.columns}
    frame = frame.rename(columns=renamed)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.rename(columns={"date": "session"})
    if tuple(frame["session"].astype(str)) != _expected_xnys_sessions(
        LEGACY_DD_FIRST, LEGACY_DD_LAST
    ):
        raise ValueError("Kaggle WIKI DD target sessions are not exactly complete.")
    coherent = (
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame["volume"].ge(0)
        & frame["high"].ge(frame[["open", "low", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "high", "close"]].min(axis=1))
    )
    if frame.isna().any().any() or not bool(coherent.all()):
        raise ValueError("Kaggle WIKI DD target contains invalid OHLCV rows.")
    terminal = frame.iloc[-1]
    for column, expected in KAGGLE_WIKI_DD_TERMINAL.items():
        if not math.isclose(float(terminal[column]), expected, abs_tol=1e-12):
            raise ValueError("Kaggle WIKI DD exact terminal row changed.")
    segment_artifact = SourceArtifact(
        source=KAGGLE_WIKI_DD_SOURCE,
        source_url=KAGGLE_WIKI_DD_URL,
        retrieved_at=value.artifact.retrieved_at,
        content=segment_content,
        content_type="text/csv",
    )
    if segment_artifact.source_hash != KAGGLE_WIKI_DD_SEGMENT_SHA256:
        raise ValueError("Kaggle WIKI DD segment artifact hash changed.")
    output = pd.DataFrame(
        {
            "security_id": LEGACY_DD_ID,
            "session": frame["session"].astype(str),
            "open": frame["open"],
            "high": frame["high"],
            "low": frame["low"],
            "close": frame["close"],
            "volume": frame["volume"],
            "currency": "USD",
            "source": KAGGLE_WIKI_DD_SOURCE,
            "source_url": KAGGLE_WIKI_DD_URL,
            "retrieved_at": value.artifact.retrieved_at,
            "source_hash": KAGGLE_WIKI_DD_SEGMENT_SHA256,
        }
    )
    columns = list(dataset_spec("daily_price_raw").required_columns)
    if "source_url" not in columns:
        columns.append("source_url")
    return output.loc[:, columns], segment_artifact, {
        "full_response_sha256": value.artifact.source_hash,
        "full_response_size": value.artifact.content_size,
        "full_dd_rows": all_dd_rows,
        "segment_sha256": segment_hash,
        "segment_rows": len(target_lines),
    }


def _parse_split_ratio(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    for separator in ("/", ":"):
        if separator in text:
            left, right = text.split(separator, 1)
            try:
                denominator = float(right)
                ratio = float(left) / denominator
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            return ratio if math.isfinite(ratio) and ratio > 0 else None
    try:
        ratio = float(text)
    except ValueError:
        return None
    return ratio if math.isfinite(ratio) and ratio > 0 else None


def _legacy_provider_action(
    artifact: SourceArtifact,
    *,
    action_type: str,
    effective_date: str,
    cash_amount: float | None = None,
    ratio: float | None = None,
    announcement_date: str = "",
    record_date: str = "",
    payment_date: str = "",
    currency: str = "USD",
) -> dict[str, Any]:
    return {
        "event_id": _event_id(
            "legacy_dd_provider_action/v1",
            LEGACY_DD_ID,
            action_type,
            effective_date,
            cash_amount,
            ratio,
            artifact.source_hash,
        ),
        "security_id": LEGACY_DD_ID,
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


def _legacy_dd_actions(artifacts: Mapping[str, SourceArtifact]) -> pd.DataFrame:
    actions: list[dict[str, Any]] = []
    dividend_artifact = artifacts["div"]
    for row in _json_rows(dividend_artifact, endpoint="div"):
        effective = str(row.get("date") or "")
        if not (LEGACY_DD_FIRST <= effective <= LEGACY_DD_LAST):
            continue
        raw_amount = row.get("unadjustedValue", row.get("value"))
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(amount) or amount <= 0:
            continue
        actions.append(
            _legacy_provider_action(
                dividend_artifact,
                action_type="cash_dividend",
                effective_date=effective,
                cash_amount=amount,
                announcement_date=str(row.get("declarationDate") or ""),
                record_date=str(row.get("recordDate") or ""),
                payment_date=str(row.get("paymentDate") or ""),
                currency=str(row.get("currency") or "USD"),
            )
        )
    split_artifact = artifacts["splits"]
    for row in _json_rows(split_artifact, endpoint="splits"):
        effective = str(row.get("date") or "")
        ratio = _parse_split_ratio(row.get("split"))
        if not (LEGACY_DD_FIRST <= effective <= LEGACY_DD_LAST) or ratio is None:
            continue
        actions.append(
            _legacy_provider_action(
                split_artifact,
                action_type="split",
                effective_date=effective,
                ratio=ratio,
            )
        )
    return pd.DataFrame(
        actions, columns=dataset_spec("corporate_actions").required_columns
    )


def _validate_legacy_dd_overlap(
    provider: pd.DataFrame, wiki: pd.DataFrame
) -> dict[str, Any]:
    overlap = provider.loc[
        provider["session"].astype(str).between(WIKI_DD_FIRST, WIKI_DD_LAST),
        ["session", "open", "high", "low", "close", "volume"],
    ].copy()
    merged = overlap.merge(wiki, on="session", suffixes=("_provider", "_wiki"))
    if (
        len(merged) != WIKI_DD_EXPECTED_ROWS
        or tuple(merged["session"].astype(str)) != tuple(wiki["session"].astype(str))
    ):
        raise ValueError("DD_old.US/WIKI overlap is not one-to-one on every pinned WIKI session.")
    relative_errors: list[float] = []
    for column in ("open", "high", "low", "close"):
        left = pd.to_numeric(merged[f"{column}_provider"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_wiki"], errors="coerce")
        relative = (left - right).abs() / right.abs()
        if relative.isna().any():
            raise ValueError("DD_old.US/WIKI overlap contains non-numeric prices.")
        relative_errors.extend(float(value) for value in relative)
    error_series = pd.Series(relative_errors, dtype=float)
    provider_close = pd.to_numeric(merged["close_provider"], errors="coerce")
    wiki_close = pd.to_numeric(merged["close_wiki"], errors="coerce")
    level_ratio = float((provider_close / wiki_close).median())
    provider_returns = provider_close.pct_change().dropna()
    wiki_returns = wiki_close.pct_change().dropna()
    return_correlation = float(provider_returns.corr(wiki_returns))
    provider_volume = pd.to_numeric(merged["volume_provider"], errors="coerce")
    wiki_volume = pd.to_numeric(merged["volume_wiki"], errors="coerce")
    volume_correlation = float(provider_volume.corr(wiki_volume))
    if not (
        0.98 <= level_ratio <= 1.02
        and float(error_series.quantile(0.95)) <= 0.02
        and return_correlation >= 0.995
        and volume_correlation >= 0.99
    ):
        raise ValueError(
            "DD_old.US fails raw WIKI price-level/return cross-validation; "
            "modern DD.US backcast is forbidden."
        )
    terminal_close = float(provider.loc[provider["session"].eq(LEGACY_DD_LAST), "close"].iloc[0])
    if not 40.0 <= terminal_close <= 150.0:
        raise ValueError("Legacy DD terminal price is implausibly back-adjusted.")
    return {
        "overlap_rows": len(merged),
        "median_level_ratio": level_ratio,
        "p95_relative_ohlc_error": float(error_series.quantile(0.95)),
        "return_correlation": return_correlation,
        "volume_correlation": volume_correlation,
    }


def load_legacy_dd_evidence(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    cache_dir: str | Path,
    *,
    allow_http: bool = False,
    allow_stooq_http: bool = False,
    allow_yahoo_http: bool = False,
    allow_kaggle_wiki_http: bool = False,
    client_factory: Callable[[], Any] = CappedLegacyDdClient,
    stooq_session_factory: Callable[[], Any] | None = None,
    yahoo_session_factory: Callable[[], Any] | None = None,
    kaggle_session_factory: Callable[[], Any] | None = None,
) -> LegacyDdEvidence:
    wiki = _load_wiki_dd_rows(repository, source_archive)
    eodhd_cache = LegacyDdEndpointCache(
        cache_dir, allow_http=allow_http, client_factory=client_factory
    )
    errors: list[str] = []
    negative_artifacts: list[SourceArtifact] = []
    eod_artifact = eodhd_cache.get("eod")
    if eod_artifact is None and allow_http:
        eod_artifact, _ = eodhd_cache.probe_eod()
    if eod_artifact is not None:
        try:
            eod_prices = _legacy_dd_prices(eod_artifact)
            overlap = _validate_legacy_dd_overlap(eod_prices, wiki)
            artifact_tuple, _ = eodhd_cache.load()
            artifacts = {
                endpoint: artifact
                for endpoint, artifact in zip(
                    LEGACY_DD_ENDPOINTS, artifact_tuple, strict=True
                )
            }
            return LegacyDdEvidence(
                prices=eod_prices,
                actions=_legacy_dd_actions(artifacts),
                artifacts=artifact_tuple,
                wiki_url=WIKI_DD_URL,
                wiki_hash=WIKI_DD_SHA256,
                overlap_rows=int(overlap["overlap_rows"]),
                http_attempts=eodhd_cache.http_attempts,
            )
        except (FileNotFoundError, ValueError) as exc:
            negative_artifacts.append(eod_artifact)
            errors.append(f"DD_old.US rejected: {exc}")
    else:
        errors.append("DD_old.US EOD cache absent")

    stooq_cache = StooqLegacyDdCache(
        cache_dir,
        allow_http=allow_stooq_http,
        session_factory=stooq_session_factory,
    )
    try:
        stooq_value, attempts = stooq_cache.load()
        stooq_prices = _stooq_legacy_dd_prices(stooq_value)
        overlap = _validate_legacy_dd_overlap(stooq_prices, wiki)
        return LegacyDdEvidence(
            prices=stooq_prices,
            actions=pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=(stooq_value.artifact,),
            wiki_url=WIKI_DD_URL,
            wiki_hash=WIKI_DD_SHA256,
            overlap_rows=int(overlap["overlap_rows"]),
            http_attempts=eodhd_cache.http_attempts + attempts,
        )
    except (FileNotFoundError, ValueError) as exc:
        if "stooq_value" in locals():
            negative_artifacts.append(stooq_value.artifact)
        errors.append(f"Stooq rejected/unavailable: {exc}")
    yahoo_cache = YahooLegacyDdCache(
        cache_dir,
        allow_http=allow_yahoo_http,
        session_factory=yahoo_session_factory,
    )
    try:
        yahoo_value, attempts = yahoo_cache.load()
        yahoo_prices = _yahoo_legacy_dd_prices(yahoo_value)
        overlap = _validate_legacy_dd_overlap(yahoo_prices, wiki)
        return LegacyDdEvidence(
            prices=yahoo_prices,
            actions=pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=(yahoo_value.artifact,),
            wiki_url=WIKI_DD_URL,
            wiki_hash=WIKI_DD_SHA256,
            overlap_rows=int(overlap["overlap_rows"]),
            http_attempts=(
                eodhd_cache.http_attempts + stooq_cache.http_attempts + attempts
            ),
        )
    except (FileNotFoundError, ValueError) as exc:
        if "yahoo_value" in locals():
            negative_artifacts.append(yahoo_value.artifact)
        errors.append(f"Yahoo rejected/unavailable: {exc}")
    kaggle_cache = KaggleWikiLegacyDdCache(
        cache_dir,
        allow_http=allow_kaggle_wiki_http,
        session_factory=kaggle_session_factory,
    )
    try:
        kaggle_value, attempts = kaggle_cache.load()
        kaggle_prices, segment_artifact, _ = _kaggle_wiki_legacy_dd_prices(
            kaggle_value, require_full_pin=True
        )
        overlap = _validate_legacy_dd_overlap(kaggle_prices, wiki)
        return LegacyDdEvidence(
            prices=kaggle_prices,
            actions=pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=(segment_artifact, *negative_artifacts),
            wiki_url=WIKI_DD_URL,
            wiki_hash=WIKI_DD_SHA256,
            overlap_rows=int(overlap["overlap_rows"]),
            http_attempts=(
                eodhd_cache.http_attempts
                + stooq_cache.http_attempts
                + yahoo_cache.http_attempts
                + attempts
            ),
            file_artifacts=(kaggle_value.artifact,),
            negative_artifact_count=len(negative_artifacts),
        )
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"Kaggle final WIKI rejected/unavailable: {exc}")
    raise FileNotFoundError(
        "No safe legacy DD primary history passed fail-closed validation. "
        + " | ".join(errors)
    )


def probe_legacy_dd_eod(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    cache_dir: str | Path,
    *,
    client_factory: Callable[[], Any] = CappedLegacyDdClient,
) -> dict[str, Any]:
    wiki = _load_wiki_dd_rows(repository, source_archive)
    cache = LegacyDdEndpointCache(
        cache_dir, allow_http=True, client_factory=client_factory
    )
    artifact, attempts = cache.probe_eod()
    prices = _legacy_dd_prices(artifact)
    overlap = _validate_legacy_dd_overlap(prices, wiki)
    return {
        "status": "legacy_dd_eod_probe_validated",
        "provider_symbol": LEGACY_DD_PROVIDER_SYMBOL,
        "source_url": artifact.source_url,
        "source_hash": artifact.source_hash,
        "price_rows": len(prices),
        "first_session": str(prices["session"].iloc[0]),
        "last_session": str(prices["session"].iloc[-1]),
        "terminal_close": float(prices["close"].iloc[-1]),
        "wiki_overlap_rows": int(overlap["overlap_rows"]),
        "median_level_ratio": float(overlap["median_level_ratio"]),
        "p95_relative_ohlc_error": float(overlap["p95_relative_ohlc_error"]),
        "return_correlation": float(overlap["return_correlation"]),
        "http_attempts": attempts,
        "maximum_http_attempts": 1,
    }


def probe_legacy_dd_stooq(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    cache_dir: str | Path,
    *,
    session_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    wiki = _load_wiki_dd_rows(repository, source_archive)
    cache = StooqLegacyDdCache(
        cache_dir, allow_http=True, session_factory=session_factory
    )
    value, attempts = cache.load()
    prices = _stooq_legacy_dd_prices(value)
    overlap = _validate_legacy_dd_overlap(prices, wiki)
    return {
        "status": "legacy_dd_stooq_probe_validated",
        "source_url": value.artifact.source_url,
        "source_hash": value.artifact.source_hash,
        "price_rows": len(prices),
        "first_session": str(prices["session"].iloc[0]),
        "last_session": str(prices["session"].iloc[-1]),
        "terminal_close": float(prices["close"].iloc[-1]),
        "wiki_overlap_rows": int(overlap["overlap_rows"]),
        "median_level_ratio": float(overlap["median_level_ratio"]),
        "p95_relative_ohlc_error": float(overlap["p95_relative_ohlc_error"]),
        "return_correlation": float(overlap["return_correlation"]),
        "http_attempts": attempts,
        "maximum_http_attempts": MAX_STOOQ_DD_HTTP_ATTEMPTS,
    }


def probe_legacy_dd_yahoo(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    cache_dir: str | Path,
    *,
    session_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    wiki = _load_wiki_dd_rows(repository, source_archive)
    cache = YahooLegacyDdCache(
        cache_dir, allow_http=True, session_factory=session_factory
    )
    value, attempts = cache.load()
    prices = _yahoo_legacy_dd_prices(value)
    overlap = _validate_legacy_dd_overlap(prices, wiki)
    return {
        "status": "legacy_dd_yahoo_probe_validated",
        "source_url": value.artifact.source_url,
        "source_hash": value.artifact.source_hash,
        "price_rows": len(prices),
        "first_session": str(prices["session"].iloc[0]),
        "last_session": str(prices["session"].iloc[-1]),
        "terminal_close": float(prices["close"].iloc[-1]),
        "wiki_overlap_rows": int(overlap["overlap_rows"]),
        "median_level_ratio": float(overlap["median_level_ratio"]),
        "p95_relative_ohlc_error": float(overlap["p95_relative_ohlc_error"]),
        "return_correlation": float(overlap["return_correlation"]),
        "http_attempts": attempts,
        "maximum_http_attempts": MAX_YAHOO_DD_HTTP_ATTEMPTS,
    }


def probe_legacy_dd_kaggle_wiki(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    cache_dir: str | Path,
    *,
    session_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    wiki = _load_wiki_dd_rows(repository, source_archive)
    cache = KaggleWikiLegacyDdCache(
        cache_dir, allow_http=True, session_factory=session_factory
    )
    value, attempts = cache.load()
    prices, _segment_artifact, audit = _kaggle_wiki_legacy_dd_prices(
        value, require_full_pin=False
    )
    overlap = _validate_legacy_dd_overlap(prices, wiki)
    full_pin_matches = bool(
        KAGGLE_WIKI_FULL_SHA256
        and KAGGLE_WIKI_FULL_SIZE > 0
        and value.artifact.source_hash == KAGGLE_WIKI_FULL_SHA256
        and value.artifact.content_size == KAGGLE_WIKI_FULL_SIZE
    )
    return {
        "status": (
            "legacy_dd_kaggle_wiki_probe_validated"
            if full_pin_matches
            else "legacy_dd_kaggle_wiki_full_pin_review_required"
        ),
        "source_url": value.artifact.source_url,
        **audit,
        "first_session": str(prices["session"].iloc[0]),
        "last_session": str(prices["session"].iloc[-1]),
        "terminal_close": float(prices["close"].iloc[-1]),
        "wiki_overlap_rows": int(overlap["overlap_rows"]),
        "median_level_ratio": float(overlap["median_level_ratio"]),
        "p95_relative_ohlc_error": float(overlap["p95_relative_ohlc_error"]),
        "return_correlation": float(overlap["return_correlation"]),
        "volume_correlation": float(overlap["volume_correlation"]),
        "http_attempts": attempts,
        "maximum_http_attempts": MAX_KAGGLE_WIKI_HTTP_ATTEMPTS,
        "full_pin_matches": full_pin_matches,
        "plan_apply_allowed": full_pin_matches,
    }


def _dates(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"DWDP repair encountered invalid {column} values.")
    return values.dt.normalize()


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"DWDP repair requires exactly one {label}; got {len(rows)}.")
    return rows.iloc[0]


def _archive_pair_exists(frame: pd.DataFrame, url: str, source_hash: str) -> bool:
    return bool(
        (
            frame["source_url"].astype(str).eq(url)
            & frame["source_hash"].astype(str).eq(source_hash)
        ).sum()
        == 1
    )


def _validate_provider_raw_archive(frame: pd.DataFrame) -> None:
    for url, source_hash, label in (
        (DD_EOD_URL, DD_EOD_SHA256, "DD EOD"),
        (DWDP_EOD_URL, DWDP_EOD_SHA256, "DWDP EOD"),
    ):
        if not _archive_pair_exists(frame, url, source_hash):
            raise ValueError(f"Pinned {label} raw bytes are not unique in source_archive.")


def verify_existing_archive_payloads(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
) -> None:
    """Verify the two provider request blobs before rewriting published rows."""

    root = repository.root.resolve()
    for url, source_hash in (
        (DD_EOD_URL, DD_EOD_SHA256),
        (DWDP_EOD_URL, DWDP_EOD_SHA256),
    ):
        row = _one_row(
            source_archive,
            source_archive["source_url"].astype(str).eq(url)
            & source_archive["source_hash"].astype(str).eq(source_hash),
            f"source archive row for {url}",
        )
        path = (root / str(row["object_path"])).resolve()
        if path == root or root not in path.parents or not path.is_file():
            raise ValueError(f"Pinned provider archive path is unsafe or absent: {path}")
        try:
            content = gzip.decompress(path.read_bytes())
        except Exception as exc:
            raise ValueError(f"Pinned provider archive is unreadable: {path}") from exc
        if sha256_bytes(content) != source_hash:
            raise ValueError(f"Pinned provider archive bytes changed: {path}")


def _identity_preflight(frames: Mapping[str, pd.DataFrame]) -> None:
    master = frames["security_master"]
    history = frames["symbol_history"]
    expected = {
        CURRENT_DD_ID: ("DD", "2015-01-01", ""),
        DWDP_ID: ("DWDP", DWDP_FIRST, "2019-06-28"),
        DOW_ID: ("DOW", DOW_FIRST_WHEN_ISSUED, ""),
        CTVA_ID: ("CTVA", "2015-01-01", ""),
    }
    for security_id, (symbol, active_from, active_to) in expected.items():
        row = _one_row(
            master,
            master["security_id"].astype(str).eq(security_id),
            f"{symbol} master row",
        )
        observed = (
            str(row["primary_symbol"]).upper(),
            str(row["active_from"]),
            str(row["active_to"]),
        )
        if observed != (symbol, active_from, active_to):
            raise ValueError(
                f"{symbol} pre-repair master boundary changed: {observed}"
            )
        history_row = _one_row(
            history,
            history["security_id"].astype(str).eq(security_id)
            & history["symbol"].astype(str).str.upper().eq(symbol),
            f"{symbol} history row",
        )
        expected_history_from = DWDP_FIRST if symbol == "DWDP" else "2015-01-01"
        if str(history_row["effective_from"]) != expected_history_from:
            raise ValueError(f"{symbol} pre-repair symbol boundary changed.")
    if master["security_id"].astype(str).eq(LEGACY_DD_ID).any():
        raise ValueError("Legacy DD identity is partially present before repair.")


def _price_preflight(frames: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    prices = frames["daily_price_raw"]
    sessions = _dates(prices, "session")
    archive = frames["source_archive"]
    _validate_provider_raw_archive(archive)

    dwdp_mask = prices["security_id"].astype(str).eq(DWDP_ID)
    dwdp_overrun = dwdp_mask & sessions.gt(pd.Timestamp(DWDP_LAST))
    overrun = prices.loc[dwdp_overrun].copy()
    if not (
        len(overrun) == 3
        and tuple(sorted(_dates(overrun, "session").dt.date.astype(str)))
        == DWDP_PSEUDO_SESSIONS
        and set(overrun["source_hash"].astype(str)) == {DWDP_EOD_SHA256}
    ):
        raise ValueError("DWDP provider pseudo-bar inventory changed.")

    terminal = _one_row(
        prices,
        dwdp_mask & sessions.eq(pd.Timestamp(DWDP_LAST)),
        "DWDP 2019-05-31 terminal price",
    )
    repeated = _one_row(
        prices,
        dwdp_mask & sessions.eq(pd.Timestamp("2019-06-28")),
        "DWDP 2019-06-28 synthetic repeat",
    )
    for column in ("open", "high", "low", "close"):
        if not math.isclose(
            float(terminal[column]), float(repeated[column]), abs_tol=1e-12
        ):
            raise ValueError("DWDP 2019-06-28 is no longer the terminal repeat.")

    dd_mask = prices["security_id"].astype(str).eq(CURRENT_DD_ID)
    unsafe_backcast = dd_mask & sessions.le(pd.Timestamp(LEGACY_DD_LAST))
    bridge = dd_mask & sessions.between(
        pd.Timestamp(DWDP_FIRST), pd.Timestamp(DWDP_LAST)
    )
    current = dd_mask & sessions.ge(pd.Timestamp(DD_FIRST_REGULAR_WAY))
    if not unsafe_backcast.any() or not bridge.any() or not current.any():
        raise ValueError("DD endpoint no longer contains all three audited backcast eras.")
    if sessions.loc[unsafe_backcast].min().date().isoformat() != LEGACY_DD_FIRST:
        raise ValueError("Unsafe DD.US backcast first session changed.")
    dwdp_valid_dates = set(
        sessions.loc[
            dwdp_mask
            & sessions.between(pd.Timestamp(DWDP_FIRST), pd.Timestamp(DWDP_LAST))
        ]
    )
    if set(sessions.loc[bridge]) != dwdp_valid_dates:
        raise ValueError("Adjusted DD/DWDP bridge session inventory changed.")
    if (unsafe_backcast | bridge | current).sum() != int(dd_mask.sum()):
        raise ValueError("DD endpoint contains an unaudited identity-date gap.")

    factor_sessions = _dates(frames["adjustment_factors"], "session")
    factor_ids = frames["adjustment_factors"]["security_id"].astype(str)
    dwdp_factor_overrun = factor_ids.eq(DWDP_ID) & factor_sessions.gt(
        pd.Timestamp(DWDP_LAST)
    )
    if tuple(sorted(factor_sessions.loc[dwdp_factor_overrun].dt.date.astype(str))) != (
        DWDP_PSEUDO_SESSIONS
    ):
        raise ValueError("DWDP adjustment-factor pseudo-row inventory changed.")
    return {
        "unsafe_dd_us_backcast_price_rows_removed": int(unsafe_backcast.sum()),
        "adjusted_bridge_price_rows_removed": int(bridge.sum()),
        "current_dd_price_rows": int(current.sum()),
        "dwdp_pseudo_price_rows_removed": int(dwdp_overrun.sum()),
        "dwdp_pseudo_factor_rows_removed": int(dwdp_factor_overrun.sum()),
    }


def _provider_action_mask(actions: pd.DataFrame) -> pd.Series:
    return actions["source_kind"].astype(str).eq("provider") | actions[
        "source"
    ].astype(str).str.startswith("eodhd_")


def _action_preflight(frames: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    actions = frames["corporate_actions"]
    dates = _dates(actions, "effective_date")
    provider = _provider_action_mask(actions)
    dd = actions["security_id"].astype(str).eq(CURRENT_DD_ID) & provider
    dwdp = actions["security_id"].astype(str).eq(DWDP_ID) & provider
    dd_prelisting = dd & dates.lt(pd.Timestamp(DD_FIRST_REGULAR_WAY))
    dd_bridge = dd & dates.between(pd.Timestamp(DWDP_FIRST), pd.Timestamp(DWDP_LAST))
    dd_pseudo_split = (
        dd
        & dates.eq(pd.Timestamp(DD_FIRST_REGULAR_WAY))
        & actions["action_type"].astype(str).eq("split")
    )
    dwdp_post = dwdp & dates.gt(pd.Timestamp(DWDP_LAST))
    dwdp_pseudo_split = (
        dwdp
        & dates.eq(pd.Timestamp("2019-04-02"))
        & actions["action_type"].astype(str).eq("split")
    )
    if not dd_prelisting.any() or not dd_bridge.any():
        raise ValueError("Adjusted DD bridge actions are absent before repair.")
    split_row = _one_row(actions, dd_pseudo_split, "DD 2019-06-03 pseudo split")
    if not math.isclose(
        float(split_row["ratio"]), 0.4725190839694656, abs_tol=1e-15
    ):
        raise ValueError("DD 2019 pseudo split ratio changed.")
    april_row = _one_row(actions, dwdp_pseudo_split, "DWDP 2019-04-02 pseudo split")
    if not math.isclose(float(april_row["ratio"]), 1.487, abs_tol=1e-12):
        raise ValueError("DWDP April provider pseudo split ratio changed.")
    if not dwdp_post.any():
        raise ValueError("DWDP post-identity provider actions are absent before repair.")
    return {
        "unsafe_dd_us_prelisting_actions_removed": int(dd_prelisting.sum()),
        "dd_bridge_actions_removed": int(dd_bridge.sum()),
        "dd_pseudo_splits_removed": int(dd_pseudo_split.sum()),
        "dwdp_post_identity_actions_removed": int(dwdp_post.sum()),
        "dwdp_pseudo_spinoff_splits_removed": int(dwdp_pseudo_split.sum()),
    }


def _official_action(
    *,
    evidence: EvidenceBundle,
    evidence_key: str,
    security_id: str,
    action_type: str,
    effective_date: str,
    ex_date: str,
    ratio: float | None = None,
    new_security_id: str = "",
    new_symbol: str = "",
    record_date: str = "",
) -> dict[str, Any]:
    artifact = evidence.artifact(evidence_key)
    return {
        "event_id": _event_id(
            "official_dwdp_identity_repair/v1",
            security_id,
            action_type,
            effective_date,
            ex_date,
            ratio,
            new_security_id,
            new_symbol,
        ),
        "security_id": security_id,
        "action_type": action_type,
        "effective_date": effective_date,
        "ex_date": ex_date,
        "announcement_date": "",
        "record_date": record_date,
        "payment_date": "",
        "cash_amount": None,
        "ratio": ratio,
        "currency": "USD",
        "new_security_id": new_security_id,
        "new_symbol": new_symbol,
        "official": True,
        "source_url": artifact.source_url,
        "source_kind": "official_filing",
        "source": "official_dwdp_identity_repair",
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
    }


def _rewrite_identities(
    frames: Mapping[str, pd.DataFrame], evidence: EvidenceBundle
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = frames["security_master"].copy()
    history = frames["symbol_history"].copy()
    old_doc = evidence.artifact("legacy_dd_merger")
    completion = evidence.artifact("corteva_dd_completion")
    dow_doc = evidence.artifact("dow_distribution")

    current_template = _one_row(
        master,
        master["security_id"].astype(str).eq(CURRENT_DD_ID),
        "current DD master template",
    ).copy()
    legacy = current_template.copy()
    legacy["security_id"] = LEGACY_DD_ID
    legacy["primary_symbol"] = "DD"
    if "provider_symbol" in legacy.index:
        legacy["provider_symbol"] = "DD_old.US"
    if "action_provider_symbol" in legacy.index:
        legacy["action_provider_symbol"] = "DD_old.US"
    legacy["name"] = "E. I. du Pont de Nemours and Company (pre-merger)"
    legacy["active_from"] = LEGACY_DD_FIRST
    legacy["active_to"] = LEGACY_DD_LAST
    legacy["source"] = "official_dwdp_identity_repair"
    legacy["source_url"] = old_doc.source_url
    legacy["retrieved_at"] = old_doc.retrieved_at
    legacy["source_hash"] = old_doc.source_hash
    # The active-catalog DD id is a duplicate endpoint identity.  DowDuPont is
    # the legal lineage that renamed to DuPont, so DWDP_ID remains canonical.
    master = master.loc[
        ~master["security_id"].astype(str).eq(CURRENT_DD_ID)
    ].copy()
    master = pd.concat([master, legacy.to_frame().T], ignore_index=True, sort=False)

    updates = {
        DWDP_ID: (DWDP_FIRST, "", completion),
        DOW_ID: (DOW_FIRST_WHEN_ISSUED, "", dow_doc),
        CTVA_ID: (CTVA_FIRST_WHEN_ISSUED, "", completion),
    }
    for security_id, (start, end, artifact) in updates.items():
        mask = master["security_id"].astype(str).eq(security_id)
        if int(mask.sum()) != 1:
            raise ValueError(f"Target identity changed while rewriting: {security_id}")
        master.loc[mask, "active_from"] = start
        master.loc[mask, "active_to"] = end
        master.loc[mask, "source"] = "official_dwdp_identity_repair"
        master.loc[mask, "source_url"] = artifact.source_url
        master.loc[mask, "retrieved_at"] = artifact.retrieved_at
        master.loc[mask, "source_hash"] = artifact.source_hash
    canonical = master["security_id"].astype(str).eq(DWDP_ID)
    master.loc[canonical, "primary_symbol"] = "DD"
    master.loc[canonical, "name"] = "DuPont de Nemours, Inc. (formerly DowDuPont Inc.)"
    if "provider_symbol" in master:
        master.loc[canonical, "provider_symbol"] = "DD.US"
    if "action_provider_symbol" in master:
        master.loc[canonical, "action_provider_symbol"] = "DD.US"

    current_history = _one_row(
        history,
        history["security_id"].astype(str).eq(CURRENT_DD_ID)
        & history["symbol"].astype(str).str.upper().eq("DD"),
        "current DD history template",
    ).copy()
    legacy_history = current_history.copy()
    legacy_history["security_id"] = LEGACY_DD_ID
    legacy_history["effective_from"] = LEGACY_DD_FIRST
    legacy_history["effective_to"] = LEGACY_DD_LAST
    legacy_history["source"] = "official_dwdp_identity_repair"
    legacy_history["source_url"] = old_doc.source_url
    legacy_history["retrieved_at"] = old_doc.retrieved_at
    legacy_history["source_hash"] = old_doc.source_hash
    current_dd_history = current_history.copy()
    current_dd_history["security_id"] = DWDP_ID
    current_dd_history["effective_from"] = DD_FIRST_REGULAR_WAY
    current_dd_history["effective_to"] = ""
    current_dd_history["source"] = "official_dwdp_identity_repair"
    current_dd_history["source_url"] = completion.source_url
    current_dd_history["retrieved_at"] = completion.retrieved_at
    current_dd_history["source_hash"] = completion.source_hash
    history = history.loc[
        ~history["security_id"].astype(str).eq(CURRENT_DD_ID)
    ].copy()
    history = pd.concat(
        [history, legacy_history.to_frame().T, current_dd_history.to_frame().T],
        ignore_index=True,
        sort=False,
    )
    history_updates = {
        # The final exchange session was May 31, while the DWDP symbol remained
        # the legal symbol through Sunday June 2 before DD began on June 3.
        DWDP_ID: ("DWDP", DWDP_FIRST, DWDP_SYMBOL_LAST, completion),
        DOW_ID: ("DOW", DOW_FIRST_WHEN_ISSUED, "", dow_doc),
        CTVA_ID: ("CTVA", CTVA_FIRST_WHEN_ISSUED, "", completion),
    }
    for security_id, (symbol, start, end, artifact) in history_updates.items():
        mask = history["security_id"].astype(str).eq(security_id) & history[
            "symbol"
        ].astype(str).str.upper().eq(symbol)
        if int(mask.sum()) != 1:
            raise ValueError(f"Target symbol history changed: {symbol}/{security_id}")
        history.loc[mask, "effective_from"] = start
        history.loc[mask, "effective_to"] = end
        history.loc[mask, "source"] = "official_dwdp_identity_repair"
        history.loc[mask, "source_url"] = artifact.source_url
        history.loc[mask, "retrieved_at"] = artifact.retrieved_at
        history.loc[mask, "source_hash"] = artifact.source_hash
    return master.reset_index(drop=True), history.reset_index(drop=True)


def _rewrite_prices(
    frames: Mapping[str, pd.DataFrame], legacy: LegacyDdEvidence
) -> pd.DataFrame:
    prices = frames["daily_price_raw"].copy()
    sessions = _dates(prices, "session")
    ids = prices["security_id"].astype(str)
    unsafe_prelisting = ids.eq(CURRENT_DD_ID) & sessions.lt(
        pd.Timestamp(DD_FIRST_REGULAR_WAY)
    )
    current = ids.eq(CURRENT_DD_ID) & sessions.ge(
        pd.Timestamp(DD_FIRST_REGULAR_WAY)
    )
    dwdp_overrun = ids.eq(DWDP_ID) & sessions.gt(pd.Timestamp(DWDP_LAST))
    output = prices.loc[~unsafe_prelisting & ~dwdp_overrun].copy()
    output.loc[current.loc[output.index], "security_id"] = DWDP_ID
    output = pd.concat([output, legacy.prices], ignore_index=True, sort=False)
    return output.drop_duplicates(
        list(dataset_spec("daily_price_raw").primary_key), keep="last"
    ).reset_index(drop=True)


def _remapped_event_id(row: pd.Series, *, namespace: str) -> str:
    return _event_id(
        namespace,
        str(row.get("event_id") or ""),
        str(row.get("security_id") or ""),
        str(row.get("effective_date") or ""),
        str(row.get("operation") or ""),
    )


def _rewrite_actions(
    frames: Mapping[str, pd.DataFrame],
    evidence: EvidenceBundle,
    legacy_evidence: LegacyDdEvidence,
) -> pd.DataFrame:
    actions = frames["corporate_actions"].copy()
    dates = _dates(actions, "effective_date")
    ids = actions["security_id"].astype(str)
    provider = _provider_action_mask(actions)

    unsafe_prelisting = ids.eq(CURRENT_DD_ID) & dates.lt(
        pd.Timestamp(DD_FIRST_REGULAR_WAY)
    )
    dd_pseudo_split = (
        ids.eq(CURRENT_DD_ID)
        & dates.eq(pd.Timestamp(DD_FIRST_REGULAR_WAY))
        & actions["action_type"].astype(str).eq("split")
        & provider
    )
    dwdp_post = ids.eq(DWDP_ID) & dates.gt(pd.Timestamp(DWDP_LAST)) & provider
    dwdp_april_pseudo = (
        ids.eq(DWDP_ID)
        & dates.eq(pd.Timestamp("2019-04-02"))
        & actions["action_type"].astype(str).eq("split")
        & provider
    )
    output = actions.loc[
        ~unsafe_prelisting & ~dd_pseudo_split & ~dwdp_post & ~dwdp_april_pseudo
    ].copy()
    current = (
        ids.eq(CURRENT_DD_ID)
        & dates.ge(pd.Timestamp(DD_FIRST_REGULAR_WAY))
        & ~dd_pseudo_split
    )
    output_current = current.loc[output.index]
    for index in output.index[output_current]:
        output.loc[index, "security_id"] = DWDP_ID
        output.loc[index, "event_id"] = _remapped_event_id(
            output.loc[index], namespace="canonical_dd_provider_action/v1"
        )

    official = pd.DataFrame(
        [
            _official_action(
                evidence=evidence,
                evidence_key="legacy_dd_merger",
                security_id=LEGACY_DD_ID,
                action_type="stock_merger",
                effective_date=DWDP_FIRST,
                ex_date=DWDP_FIRST,
                ratio=1.282,
                new_security_id=DWDP_ID,
                new_symbol="DWDP",
            ),
            _official_action(
                evidence=evidence,
                evidence_key="dow_distribution",
                security_id=DWDP_ID,
                action_type="spinoff",
                effective_date=DOW_DISTRIBUTION,
                ex_date=DOW_DISTRIBUTION,
                ratio=1 / 3,
                new_security_id=DOW_ID,
                new_symbol="DOW",
                record_date="2019-03-21",
            ),
            _official_action(
                evidence=evidence,
                evidence_key="corteva_dd_completion",
                security_id=DWDP_ID,
                action_type="spinoff",
                effective_date=CTVA_DISTRIBUTION,
                ex_date=CTVA_DISTRIBUTION,
                ratio=1 / 3,
                new_security_id=CTVA_ID,
                new_symbol="CTVA",
                record_date="2019-05-24",
            ),
            _official_action(
                evidence=evidence,
                evidence_key="corteva_dd_completion",
                security_id=DWDP_ID,
                action_type="split",
                effective_date=CTVA_DISTRIBUTION,
                # Legally effective Saturday June 1; the first split-adjusted
                # regular-way session was Monday June 3 under DD.
                ex_date=DD_FIRST_REGULAR_WAY,
                ratio=1 / 3,
            ),
            _official_action(
                evidence=evidence,
                evidence_key="corteva_dd_completion",
                security_id=DWDP_ID,
                action_type="ticker_change",
                effective_date=DD_FIRST_REGULAR_WAY,
                ex_date=DD_FIRST_REGULAR_WAY,
                new_security_id=DWDP_ID,
                new_symbol="DD",
            ),
        ]
    )
    output = pd.concat(
        [output, legacy_evidence.actions, official], ignore_index=True, sort=False
    )
    return output.drop_duplicates(
        list(dataset_spec("corporate_actions").primary_key), keep="last"
    ).reset_index(drop=True)


def _rewrite_index_identity(
    frames: Mapping[str, pd.DataFrame], evidence: EvidenceBundle
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    old_doc = evidence.artifact("legacy_dd_merger")
    anchors = frames["index_constituent_anchors"].copy()
    anchor_dates = _dates(anchors, "anchor_date")
    anchor_mask = anchors["security_id"].astype(str).eq(CURRENT_DD_ID) & anchor_dates.le(
        pd.Timestamp(LEGACY_DD_LAST)
    )
    anchors.loc[anchor_mask, "security_id"] = LEGACY_DD_ID
    for column, value in (
        ("source", "official_dwdp_identity_repair"),
        ("source_url", old_doc.source_url),
        ("source_kind", "derived_identity"),
        ("retrieved_at", old_doc.retrieved_at),
        ("source_hash", old_doc.source_hash),
    ):
        anchors.loc[anchor_mask, column] = value
    anchors = anchors.drop_duplicates(
        list(dataset_spec("index_constituent_anchors").primary_key), keep="last"
    ).reset_index(drop=True)

    events = frames["index_membership_events"].copy()
    event_dates = _dates(events, "effective_date")
    event_mask = events["security_id"].astype(str).eq(CURRENT_DD_ID) & event_dates.le(
        pd.Timestamp(DWDP_FIRST)
    )
    for index in events.index[event_mask]:
        events.loc[index, "security_id"] = LEGACY_DD_ID
        events.loc[index, "event_id"] = _remapped_event_id(
            events.loc[index], namespace="legacy_dd_index_event/v1"
        )
    for column, value in (
        ("source", "official_dwdp_identity_repair"),
        ("source_url", old_doc.source_url),
        ("source_kind", "derived_identity"),
        ("retrieved_at", old_doc.retrieved_at),
        ("source_hash", old_doc.source_hash),
    ):
        events.loc[event_mask, column] = value
    # The community history expresses the same-lineage rename as DWDP REMOVE
    # plus DD ADD.  At security-id level membership is continuous, so both
    # transitions must disappear rather than becoming a remove/add round trip.
    operations = events["operation"].astype(str).str.upper()
    june_transition = event_dates.eq(pd.Timestamp(DD_FIRST_REGULAR_WAY)) & (
        (
            events["security_id"].astype(str).eq(DWDP_ID)
            & operations.eq("REMOVE")
        )
        | (
            events["security_id"].astype(str).eq(CURRENT_DD_ID)
            & operations.eq("ADD")
        )
    )
    if int(june_transition.sum()) != 2:
        raise ValueError("DWDP/DD index transition inventory changed.")
    events = events.loc[~june_transition].copy()
    later_current = events["security_id"].astype(str).eq(CURRENT_DD_ID)
    for index in events.index[later_current]:
        events.loc[index, "security_id"] = DWDP_ID
        events.loc[index, "event_id"] = _remapped_event_id(
            events.loc[index], namespace="canonical_dd_index_event/v1"
        )
    events = events.drop_duplicates(
        list(dataset_spec("index_membership_events").primary_key), keep="last"
    ).reset_index(drop=True)
    return anchors, events, {
        "legacy_dd_anchors_remapped": int(anchor_mask.sum()),
        "legacy_dd_membership_events_remapped": int(event_mask.sum()),
        "same_lineage_index_transition_rows_removed": int(june_transition.sum()),
    }


def _archive_extension(artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower().split(";", 1)[0].strip()
    if content_type == "text/plain":
        return "txt"
    if content_type == "text/csv":
        return "csv"
    if content_type == "application/json":
        return "json"
    return "bin"


def _file_archive_extension(artifact: FileSourceArtifact) -> str:
    content_type = artifact.content_type.lower().split(";", 1)[0].strip()
    if content_type == "text/csv":
        return "csv"
    if "zip" in content_type or artifact.gzip_path.name.endswith(".zip.gz"):
        return "zip"
    return "bin"


def _append_source_archive(
    frame: pd.DataFrame,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> pd.DataFrame:
    rows = [
        {
            "archive_id": artifact.source_hash,
            "dataset": artifact.source,
            "object_path": (
                f"archives/{completed_session}/{artifact.source_hash}."
                f"{_archive_extension(artifact)}.gz"
            ),
            "content_type": artifact.content_type,
            "effective_date": completed_session,
            "source": artifact.source,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
            "source_url": artifact.source_url,
        }
        for artifact in artifacts
    ]
    output = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return output.drop_duplicates(
        list(dataset_spec("source_archive").primary_key), keep="last"
    ).reset_index(drop=True)


def _append_file_source_archive(
    frame: pd.DataFrame,
    artifacts: Iterable[FileSourceArtifact],
    *,
    completed_session: str,
) -> pd.DataFrame:
    rows = [
        {
            "archive_id": artifact.source_hash,
            "dataset": artifact.source,
            "object_path": (
                f"archives/{completed_session}/{artifact.source_hash}."
                f"{_file_archive_extension(artifact)}.gz"
            ),
            "content_type": artifact.content_type,
            "effective_date": completed_session,
            "source": artifact.source,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
            "source_url": artifact.source_url,
        }
        for artifact in artifacts
    ]
    if not rows:
        return frame.copy().reset_index(drop=True)
    output = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return output.drop_duplicates(
        list(dataset_spec("source_archive").primary_key), keep="last"
    ).reset_index(drop=True)


def _rewrite_factors(
    frames: Mapping[str, pd.DataFrame],
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
) -> pd.DataFrame:
    old = frames["adjustment_factors"]
    retained = old.loc[
        ~old["security_id"].astype(str).isin(AFFECTED_FACTOR_IDS)
    ].copy()
    affected_prices = prices.loc[
        prices["security_id"].astype(str).isin(AFFECTED_FACTOR_IDS)
    ].copy()
    affected_actions = actions.loc[
        actions["security_id"].astype(str).isin(AFFECTED_FACTOR_IDS)
    ].copy()
    rebuilt = build_adjustment_factors(
        affected_prices,
        affected_actions,
        source_version=source_version,
    )
    output = pd.concat([retained, rebuilt], ignore_index=True, sort=False)
    return output.drop_duplicates(
        list(dataset_spec("adjustment_factors").primary_key), keep="last"
    ).reset_index(drop=True)


class _FrameRepository:
    def __init__(self, frames: Mapping[str, pd.DataFrame]):
        self.frames = dict(frames)

    def current_manifest(self, dataset: str):
        return object() if dataset in self.frames else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        return self.frames[dataset].copy()


def _official_events(actions: pd.DataFrame) -> pd.DataFrame:
    return actions.loc[
        actions["source"].astype(str).eq("official_dwdp_identity_repair")
    ]


def validate_repaired_frames(
    frames: Mapping[str, pd.DataFrame],
    evidence: EvidenceBundle,
    legacy_evidence: LegacyDdEvidence,
    *,
    completed_session: str,
) -> dict[str, Any]:
    for dataset in WRITE_DATASETS:
        report = validate_dataset(
            dataset,
            frames[dataset],
            completed_session=completed_session,
            incomplete_action_policy="warn",
        )
        report.raise_for_errors()
    validate_repository_snapshot(_FrameRepository(frames)).raise_for_errors()

    master = frames["security_master"]
    history = frames["symbol_history"]
    master_expectations = {
        LEGACY_DD_ID: ("DD", LEGACY_DD_FIRST, LEGACY_DD_LAST),
        DWDP_ID: ("DD", DWDP_FIRST, ""),
        DOW_ID: ("DOW", DOW_FIRST_WHEN_ISSUED, ""),
        CTVA_ID: ("CTVA", CTVA_FIRST_WHEN_ISSUED, ""),
    }
    for security_id, (symbol, start, end) in master_expectations.items():
        master_row = _one_row(
            master,
            master["security_id"].astype(str).eq(security_id),
            f"repaired {symbol} master row",
        )
        if (
            str(master_row["primary_symbol"]).upper(),
            str(master_row["active_from"]),
            str(master_row["active_to"]),
        ) != (symbol, start, end):
            raise ValueError(f"Repaired {symbol} master boundary is not exact.")
    history_expectations = (
        (LEGACY_DD_ID, "DD", LEGACY_DD_FIRST, LEGACY_DD_LAST),
        (DWDP_ID, "DWDP", DWDP_FIRST, DWDP_SYMBOL_LAST),
        (DWDP_ID, "DD", DD_FIRST_REGULAR_WAY, ""),
        (DOW_ID, "DOW", DOW_FIRST_WHEN_ISSUED, ""),
        (CTVA_ID, "CTVA", CTVA_FIRST_WHEN_ISSUED, ""),
    )
    for security_id, symbol, start, end in history_expectations:
        history_row = _one_row(
            history,
            history["security_id"].astype(str).eq(security_id)
            & history["symbol"].astype(str).str.upper().eq(symbol),
            f"repaired {symbol} symbol interval",
        )
        if (
            str(history_row["effective_from"]),
            str(history_row["effective_to"]),
        ) != (start, end):
            raise ValueError(f"Repaired {symbol} symbol boundary is not exact.")
    if master["security_id"].astype(str).eq(CURRENT_DD_ID).any():
        raise ValueError("Duplicate current DD identity remains in security_master.")
    if history["security_id"].astype(str).eq(CURRENT_DD_ID).any():
        raise ValueError("Duplicate current DD identity remains in symbol_history.")

    prices = frames["daily_price_raw"]
    sessions = _dates(prices, "session")
    ids = prices["security_id"].astype(str)
    ranges = {
        LEGACY_DD_ID: (LEGACY_DD_FIRST, LEGACY_DD_LAST),
        DWDP_ID: (DWDP_FIRST, completed_session),
        DOW_ID: (DOW_FIRST_WHEN_ISSUED, completed_session),
        CTVA_ID: (CTVA_FIRST_WHEN_ISSUED, completed_session),
    }
    for security_id, (start, end) in ranges.items():
        own = sessions.loc[ids.eq(security_id)]
        if own.empty or own.min().date().isoformat() != start:
            raise ValueError(f"Repaired price start is not exact: {security_id}")
        if security_id in {LEGACY_DD_ID, DWDP_ID} and own.max().date().isoformat() != end:
            raise ValueError(f"Repaired terminal price is not exact: {security_id}")
    if ids.eq(CURRENT_DD_ID).any():
        raise ValueError("Duplicate current DD identity remains in published prices.")
    legacy_prices = prices.loc[ids.eq(LEGACY_DD_ID)]
    primary_urls = set(legacy_prices["source_url"].astype(str))
    primary_hashes = set(legacy_prices["source_hash"].astype(str))
    artifact_pairs = {
        (artifact.source_url, artifact.source_hash)
        for artifact in legacy_evidence.artifacts
    }
    if (
        len(primary_urls) != 1
        or len(primary_hashes) != 1
        or next(iter(zip(primary_urls, primary_hashes))) not in artifact_pairs
        or DD_EOD_URL in primary_urls
        or BORIS_DD_URL in primary_urls
        or BORIS_DD_REJECTED_SHA256 in primary_hashes
    ):
        raise ValueError("Legacy DD prices are not a validated independent artifact.")

    factors = frames["adjustment_factors"]
    factor_pairs = set(
        zip(
            factors["security_id"].astype(str),
            _dates(factors, "session").dt.date.astype(str),
        )
    )
    price_pairs = set(
        zip(prices["security_id"].astype(str), sessions.dt.date.astype(str))
    )
    for pair in price_pairs | factor_pairs:
        if pair[0] in AFFECTED_FACTOR_IDS and (
            (pair in price_pairs) != (pair in factor_pairs)
        ):
            raise ValueError("Affected adjustment-factor coverage differs from prices.")

    actions = _official_events(frames["corporate_actions"])
    expected = {
        (LEGACY_DD_ID, "stock_merger", DWDP_FIRST): (1.282, DWDP_ID, "DWDP"),
        (DWDP_ID, "spinoff", DOW_DISTRIBUTION): (1 / 3, DOW_ID, "DOW"),
        (DWDP_ID, "spinoff", CTVA_DISTRIBUTION): (1 / 3, CTVA_ID, "CTVA"),
        (DWDP_ID, "split", CTVA_DISTRIBUTION): (1 / 3, "", ""),
        (DWDP_ID, "ticker_change", DD_FIRST_REGULAR_WAY): (
            None,
            DWDP_ID,
            "DD",
        ),
    }
    if len(actions) != len(expected):
        raise ValueError("Official DWDP action inventory is not exact.")
    for key, (ratio, new_id, new_symbol) in expected.items():
        row = _one_row(
            actions,
            actions["security_id"].astype(str).eq(key[0])
            & actions["action_type"].astype(str).eq(key[1])
            & actions["effective_date"].astype(str).eq(key[2]),
            f"official action {key}",
        )
        if ratio is not None and not math.isclose(
            float(row["ratio"]), ratio, abs_tol=1e-15
        ):
            raise ValueError(f"Official ratio changed: {key}")
        if str(row["new_security_id"]) != new_id or str(row["new_symbol"]) != new_symbol:
            raise ValueError(f"Official successor changed: {key}")
        if row["official"] is not True and bool(row["official"]) is not True:
            raise ValueError(f"Official action flag changed: {key}")

    archive = frames["source_archive"]
    _validate_provider_raw_archive(archive)
    for artifact in evidence.artifacts:
        if not _archive_pair_exists(archive, artifact.source_url, artifact.source_hash):
            raise ValueError(f"Official raw evidence is not archived: {artifact.source_url}")
    for artifact in legacy_evidence.artifacts:
        if not _archive_pair_exists(archive, artifact.source_url, artifact.source_hash):
            raise ValueError(f"Legacy DD raw evidence is not archived: {artifact.source_url}")
    for artifact in legacy_evidence.file_artifacts:
        if not _archive_pair_exists(archive, artifact.source_url, artifact.source_hash):
            raise ValueError(
                f"Legacy DD complete file evidence is not archived: {artifact.source_url}"
            )
    if not _archive_pair_exists(archive, legacy_evidence.wiki_url, legacy_evidence.wiki_hash):
        raise ValueError("Pinned WIKI DD cross-validation source is not archived.")

    anchors = frames["index_constituent_anchors"]
    if anchors["security_id"].astype(str).eq(CURRENT_DD_ID).any():
        raise ValueError("Duplicate current DD identity remains in index anchors.")
    events = frames["index_membership_events"]
    if events["security_id"].astype(str).eq(CURRENT_DD_ID).any():
        raise ValueError("Duplicate current DD identity remains in membership events.")
    if frames["corporate_actions"]["security_id"].astype(str).eq(CURRENT_DD_ID).any():
        raise ValueError("Duplicate current DD identity remains in corporate actions.")
    if frames["adjustment_factors"]["security_id"].astype(str).eq(CURRENT_DD_ID).any():
        raise ValueError("Duplicate current DD identity remains in adjustment factors.")
    return {
        "official_event_count": len(actions),
        "official_archive_count": len(evidence.artifacts),
        "legacy_dd_archive_count": len(legacy_evidence.artifacts),
        "legacy_dd_negative_archive_count": legacy_evidence.negative_artifact_count,
        "legacy_dd_file_archive_count": len(legacy_evidence.file_artifacts),
        "legacy_dd_wiki_overlap_rows": legacy_evidence.overlap_rows,
        "legacy_dd_security_id": LEGACY_DD_ID,
        "network_accessed": bool(legacy_evidence.http_attempts),
    }


def _looks_repaired(frames: Mapping[str, pd.DataFrame]) -> bool:
    return frames["security_master"]["security_id"].astype(str).eq(LEGACY_DD_ID).any()


def prepare_dwdp_repair(
    frames: Mapping[str, pd.DataFrame],
    evidence: EvidenceBundle,
    legacy_evidence: LegacyDdEvidence,
    *,
    completed_session: str,
    source_version: str,
) -> PreparedDwdpRepair:
    missing = sorted(set(REQUIRED_DATASETS) - set(frames))
    if missing:
        raise ValueError("DWDP repair is missing datasets: " + ", ".join(missing))
    if _looks_repaired(frames):
        summary = validate_repaired_frames(
            frames,
            evidence,
            legacy_evidence,
            completed_session=completed_session,
        )
        return PreparedDwdpRepair(
            frames={name: frames[name].copy() for name in WRITE_DATASETS},
            artifacts=(*evidence.artifacts, *legacy_evidence.artifacts),
            official_evidence=evidence,
            legacy_evidence=legacy_evidence,
            file_artifacts=legacy_evidence.file_artifacts,
            summary={**summary, "status": "already_repaired"},
        )

    _identity_preflight(frames)
    price_stats = _price_preflight(frames)
    action_stats = _action_preflight(frames)
    master, history = _rewrite_identities(frames, evidence)
    prices = _rewrite_prices(frames, legacy_evidence)
    actions = _rewrite_actions(frames, evidence, legacy_evidence)
    anchors, events, index_stats = _rewrite_index_identity(frames, evidence)
    archive = _append_source_archive(
        frames["source_archive"],
        (*evidence.artifacts, *legacy_evidence.artifacts),
        completed_session=completed_session,
    )
    archive = _append_file_source_archive(
        archive,
        legacy_evidence.file_artifacts,
        completed_session=completed_session,
    )
    rewritten: dict[str, pd.DataFrame] = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": archive,
    }
    rewritten["adjustment_factors"] = _rewrite_factors(
        frames,
        prices,
        actions,
        source_version=source_version,
    )
    summary = validate_repaired_frames(
        rewritten,
        evidence,
        legacy_evidence,
        completed_session=completed_session,
    )
    return PreparedDwdpRepair(
        frames=rewritten,
        artifacts=(*evidence.artifacts, *legacy_evidence.artifacts),
        official_evidence=evidence,
        legacy_evidence=legacy_evidence,
        file_artifacts=legacy_evidence.file_artifacts,
        summary={
            **summary,
            **price_stats,
            **action_stats,
            **index_stats,
            "status": "validated_dry_run",
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
        if path.is_file():
            if gzip.decompress(path.read_bytes()) != artifact.content:
                raise RuntimeError(f"Conflicting official archive payload: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, gzip.compress(artifact.content, mtime=0))
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"Official archive verification failed: {path}")


def _verify_gzip_file_artifact(path: Path, artifact: FileSourceArtifact) -> None:
    digest = hashlib.sha256()
    size = 0
    try:
        with gzip.open(path, "rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except (EOFError, OSError) as exc:
        raise RuntimeError(f"Large archive payload is truncated: {path}") from exc
    if digest.hexdigest() != artifact.source_hash or size != artifact.content_size:
        raise RuntimeError(f"Large archive payload hash/size mismatch: {path}")


def _persist_file_archive_payloads(
    repository: LocalDatasetRepository,
    artifacts: Iterable[FileSourceArtifact],
    *,
    completed_session: str,
) -> None:
    for artifact in artifacts:
        _verify_gzip_file_artifact(artifact.gzip_path, artifact)
        path = (
            repository.root
            / "archives"
            / completed_session
            / (
                f"{artifact.source_hash}."
                f"{_file_archive_extension(artifact)}.gz"
            )
        )
        if path.is_file():
            _verify_gzip_file_artifact(path, artifact)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(artifact.gzip_path, temporary)
            _verify_gzip_file_artifact(temporary, artifact)
            try:
                os.link(temporary, path)
            except FileExistsError:
                _verify_gzip_file_artifact(path, artifact)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        yield


def _assert_release_unchanged(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
) -> None:
    current, current_etag = repository.current_release()
    if (
        current is None
        or current.version != release.version
        or current_etag != release_etag
    ):
        raise RuntimeError("Current release changed during DWDP repair validation.")


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, _canonical_json_bytes(dict(value)))


def _rollback_pointers(
    repository: LocalDatasetRepository,
    old_pointers: Mapping[str, bytes],
) -> list[str]:
    errors: list[str] = []
    for dataset, payload in old_pointers.items():
        try:
            current = repository.objects.get(repository.current_key(dataset))
            repository.objects.put(
                repository.current_key(dataset), payload, if_match=current.etag
            )
        except Exception as exc:  # pragma: no cover - catastrophic IO path
            errors.append(f"{dataset}: {type(exc).__name__}: {exc}")
    return errors


def apply_dwdp_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedDwdpRepair,
    *,
    release: DataRelease,
    release_etag: str | None,
    pointer_etags: Mapping[str, str | None],
) -> DataRelease:
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(repository, release, release_etag)
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != release.dataset_versions[dataset]
                or value.etag != pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before DWDP apply.")
            old_pointers[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        planned = {
            dataset: (
                f"dwdp-identity-{release.completed_session.replace('-', '')}-"
                f"{transaction_id}-{dataset}"
            )
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/dwdp-identity-repair"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "dwdp_identity_repair_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": release.version,
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": planned,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        try:
            _persist_archive_payloads(
                repository,
                prepared.artifacts,
                completed_session=release.completed_session,
            )
            _persist_file_archive_payloads(
                repository,
                prepared.file_artifacts,
                completed_session=release.completed_session,
            )
            versions = dict(release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=release.completed_session,
                    incomplete_action_policy="warn",
                    metadata={
                        "operation": "repair_us_dwdp_identity_spinoffs",
                        "repair_evidence_sha256": [
                            artifact.source_hash for artifact in prepared.artifacts
                        ]
                        + [
                            artifact.source_hash
                            for artifact in prepared.file_artifacts
                        ],
                        "network_accessed": bool(
                            prepared.legacy_evidence.http_attempts
                        ),
                    },
                    expected_pointer_etag=pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version

            written = {
                dataset: repository.read_frame(dataset, planned[dataset])
                for dataset in WRITE_DATASETS
            }
            validate_repaired_frames(
                written,
                prepared.official_evidence,
                prepared.legacy_evidence,
                completed_session=release.completed_session,
            )
            committed = repository.commit_release(
                release.completed_session,
                versions,
                quality=(
                    DataQuality.DEGRADED if release.warnings else DataQuality.VALID
                ),
                warnings=release.warnings,
                expected_etag=release_etag,
            )
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": committed.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            return committed
        except Exception:
            errors = _rollback_pointers(repository, old_pointers)
            journal.update(
                {
                    "status": "rollback_failed" if errors else "rolled_back",
                    "rollback_errors": errors,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            if errors:
                raise RuntimeError(
                    "DWDP repair failed and pointer rollback was incomplete: "
                    + " | ".join(errors)
                )
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair DD/DWDP/DOW/CTVA identities and distributions offline."
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--sec-cache-dir", default=str(DEFAULT_SEC_CACHE))
    parser.add_argument(
        "--legacy-dd-cache-dir", default=str(DEFAULT_LEGACY_DD_CACHE)
    )
    parser.add_argument(
        "--fetch-legacy-dd",
        action="store_true",
        help=(
            "Explicitly fill missing DD_old.US eod/div/splits caches with at most "
            "three one-attempt EODHD calls. Never combine with --apply."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument(
        "--probe-legacy-dd-eod",
        action="store_true",
        help="Make at most one exact DD_old.US EOD call and cross-check it against WIKI.",
    )
    mode.add_argument(
        "--probe-legacy-dd-stooq",
        action="store_true",
        help="Make at most one exact Stooq DD CSV call and cross-check it against WIKI.",
    )
    mode.add_argument(
        "--probe-legacy-dd-yahoo",
        action="store_true",
        help="Make at most one exact bounded Yahoo DD daily-chart call and cross-check WIKI.",
    )
    mode.add_argument(
        "--probe-legacy-dd-kaggle-wiki",
        action="store_true",
        help=(
            "Make one complete streaming request for the frozen final Quandl WIKI "
            "mirror, audit the DD segment, and print the full response hash/size."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.apply and args.fetch_legacy_dd:
        raise ValueError(
            "Network collection and atomic apply must be separate runs; fetch first, "
            "then replay --plan offline, then use --apply."
        )
    repository = LocalDatasetRepository(args.cache_root)
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1] for dataset in WRITE_DATASETS
    }
    verify_existing_archive_payloads(repository, frames["source_archive"])
    if args.probe_legacy_dd_eod:
        summary = probe_legacy_dd_eod(
            repository,
            frames["source_archive"],
            args.legacy_dd_cache_dir,
        )
        print(
            json.dumps(
                {
                    **summary,
                    "base_release_version": release.version,
                    "completed_session": release.completed_session,
                    "eodhd_call_budget_before_run": os.getenv(
                        "EODHD_CALL_BUDGET", ""
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.probe_legacy_dd_stooq:
        summary = probe_legacy_dd_stooq(
            repository,
            frames["source_archive"],
            args.legacy_dd_cache_dir,
        )
        print(
            json.dumps(
                {
                    **summary,
                    "base_release_version": release.version,
                    "completed_session": release.completed_session,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.probe_legacy_dd_yahoo:
        summary = probe_legacy_dd_yahoo(
            repository,
            frames["source_archive"],
            args.legacy_dd_cache_dir,
        )
        print(
            json.dumps(
                {
                    **summary,
                    "base_release_version": release.version,
                    "completed_session": release.completed_session,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.probe_legacy_dd_kaggle_wiki:
        summary = probe_legacy_dd_kaggle_wiki(
            repository,
            frames["source_archive"],
            args.legacy_dd_cache_dir,
        )
        print(
            json.dumps(
                {
                    **summary,
                    "base_release_version": release.version,
                    "completed_session": release.completed_session,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    evidence = load_official_evidence(args.sec_cache_dir)
    legacy_evidence = load_legacy_dd_evidence(
        repository,
        frames["source_archive"],
        args.legacy_dd_cache_dir,
        allow_http=args.fetch_legacy_dd,
    )
    prepared = prepare_dwdp_repair(
        frames,
        evidence,
        legacy_evidence,
        completed_session=release.completed_session,
        source_version=f"dwdp-identity-repair:{release.version}",
    )
    summary = {
        **prepared.summary,
        "base_release_version": release.version,
        "completed_session": release.completed_session,
        "would_write": bool(args.apply),
        "http_attempts": legacy_evidence.http_attempts,
        "maximum_legacy_dd_http_attempts": MAX_LEGACY_DD_HTTP_ATTEMPTS,
    }
    if not args.apply or prepared.summary["status"] == "already_repaired":
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    committed = apply_dwdp_repair(
        repository,
        prepared,
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
    )
    validate_repository_snapshot(repository).raise_for_errors()
    print(
        json.dumps(
            {
                **summary,
                "status": "applied",
                "new_release_version": committed.version,
                "quality": str(committed.quality),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
