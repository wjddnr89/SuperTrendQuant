#!/usr/bin/env python3
"""Repair the duplicated VIP -> VEON same-security lineage.

The bootstrap snapshot currently contains two identities for one continuous
NASDAQ-listed ADS.  The retired ``VIP.US`` endpoint is contaminated (its
terminal prices are on a false scale), while the active ``VEON.US`` endpoint
contains a continuous raw-USD history across the official 2017-03-31 symbol
transition.  This tool therefore never copies a price or action from
``VIP.US``.

Evidence is deliberately fail-closed and offline during plan/apply:

* one exact cached SEC 6-K/A identifies VimpelCom as ``NASDAQ: VIP`` and says
  that its NASDAQ ticker will change to VEON;
* one exact cached SEC 6-K proves shareholder approval on 2017-03-30 and that
  the ADSs trade under ``VEON`` from 2017-03-31;
* one exact bounded Yahoo chart response independently checks every XNYS
  session from 2015-01-02 through 2017-03-31 against stored EODHD VEON bars;
* the already archived final Quandl WIKI mirror is hash/size checked and
  scanned to preserve the negative audit that it has zero VIP/VEON rows.

``--fetch-yahoo-evidence`` is acquisition-only and permits at most one
no-retry HTTP attempt.  ``--offline-plan`` and ``--apply`` construct the cache
reader but never call ``fetch``.  Apply uses a repository-wide writer lock,
compare-and-swap release/pointer checks, an explicit rollback journal and a
post-write idempotence invariant.  The tool never uploads to R2.
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
import shutil
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import SourceArtifact
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
from supertrend_quant.market_store.yahoo_chart import (
    YahooChartCache,
    YahooChartCachedResponse,
    YahooChartData,
    parse_yahoo_chart_json,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_YAHOO_CACHE = DEFAULT_CACHE_ROOT / "state/vip-veon-identity/yahoo"

OLD_SECURITY_ID = "US:EODHD:e2c710f1-f687-511b-93ff-233a8b8e40a7"
CANONICAL_SECURITY_ID = "US:EODHD:8102c8e2-e5d1-5331-a987-4692d29da477"
OLD_SYMBOL = "VIP"
NEW_SYMBOL = "VEON"
HISTORY_START = "2015-01-01"
PRICE_START = "2015-01-02"
LEGAL_NAME_APPROVAL_DATE = "2017-03-30"
OLD_LAST_SESSION = "2017-03-30"
TRANSITION_DATE = "2017-03-31"
# Provider bootstrap boundaries being corrected by this repair.
SOURCE_OLD_LAST_SESSION = "2017-03-29"
SOURCE_NEW_FIRST_SESSION = "2017-03-30"

OLD_SYMBOL_SEC_URL = (
    "https://www.sec.gov/Archives/edgar/data/1468091/"
    "000119312517061033/0001193125-17-061033.txt"
)
OLD_SYMBOL_SEC_SHA256 = (
    "2930324ccdb54ad67108c4497bec7a614bc80d82bc3ff81760a192ba8d5bf43a"
)
OLD_SYMBOL_SEC_EXACT_BYTES = 754_544
OLD_SYMBOL_SEC_RETRIEVED_AT = "2026-07-17T19:25:21.635969Z"
OLD_SYMBOL_SEC_CACHE_KEY = sha256_bytes(f"{OLD_SYMBOL_SEC_URL}?".encode())
OLD_SYMBOL_SEC_REQUIRED_TEXT_GROUPS = (
    ("accession number: 0001193125-17-061033",),
    ("conformed submission type: 6-k/a",),
    ("central index key: 0001468091",),
    ("vimpelcom ltd. (nasdaq: vip)",),
    (
        "the ticker for the company's listing on nasdaq will also change to veon",
    ),
    ("this amendment does not otherwise amend, modify or update any disclosures",),
)

SEC_URL = (
    "https://www.sec.gov/Archives/edgar/data/1468091/"
    "000119312517103808/0001193125-17-103808.txt"
)
SEC_SHA256 = "cb257624d286b531e891aed1a9c21f0a2c1fef92023331433cdbb8b0434416aa"
SEC_EXACT_BYTES = 29_006
SEC_RETRIEVED_AT = "2026-07-17T19:25:21.608346Z"
SEC_CACHE_KEY = sha256_bytes(f"{SEC_URL}?".encode())
SEC_REQUIRED_TEXT_GROUPS = (
    ("accession number: 0001193125-17-103808",),
    ("central index key: 0001468091",),
    ("veon ltd. (formerly vimpelcom ltd.)",),
    (
        "veon ltd. announces today it has secured shareholder approval to change its name from vimpelcom ltd. to veon ltd.",
    ),
    (
        "from march 31, 2017, the company's american depositary shares will trade on the nasdaq global select market under the symbol \"veon.\"",
    ),
)

YAHOO_PERIOD1 = 1_420_156_800  # 2015-01-02T00:00:00Z
YAHOO_PERIOD2 = 1_491_004_800  # 2017-04-01T00:00:00Z, exclusive
YAHOO_EXPECTED_ROWS = 566
MAX_YAHOO_HTTP_ATTEMPTS = 1
# The former 565-session request stopped at 2017-03-30. Keep its pins as
# immutable audit history; the 566-session URL has a different cache key and
# must be acquired once and reviewed before the new pins are filled below.
LEGACY_YAHOO_PERIOD2 = 1_490_918_400
LEGACY_YAHOO_EXPECTED_ROWS = 565
LEGACY_YAHOO_SHA256 = (
    "fabb8003319fd6b48a9486523eda02889ea5d9cf298c28d435fc9862805700ab"
)
LEGACY_YAHOO_WRAPPER_SHA256 = (
    "97d0c3b80a1fd913cbc9e4cc462d41394529839f90ea6f282a4a26e517e1830b"
)
YAHOO_SHA256 = "1fc7063eba99c9ab5ee24855b21e9765fdbb5beb7055dfc0e0a746ecee8c7715"
YAHOO_WRAPPER_SHA256 = (
    "754b793be930378ee07b58bc963d602703554ba69b48c745d2154b7e571e1559"
)

EODHD_VEON_EOD_SHA256 = (
    "0716fde93069e518d6225dabb6c58fc24e6f53d4e97f0f32dd91ee9424e35d0d"
)
EODHD_VIP_REJECTED_EOD_SHA256 = (
    "922bb6440aa84ec9743e8d77f53bc95eac93398980c7670a2a90f71c31109be6"
)

WIKI_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/marketneutral/"
    "quandl-wiki-prices-us-equites/WIKI_PRICES.csv"
)
WIKI_FULL_SHA256 = (
    "36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae"
)
WIKI_FULL_SIZE = 463_184_323
WIKI_TOTAL_DATA_ROWS = 15_389_314
WIKI_MEMBER_NAME = "WIKI_PRICES.csv"
WIKI_REJECTED_TICKERS = ("VIP", "VEON")
WIKI_AUDIT_SCHEMA = "vip_veon_rejected_wiki_source_audit/v1"

MIN_RETURN_CORRELATION = 0.995
MAX_SCALE_DEVIATION = 0.01
CLOSE_RELATIVE_TOLERANCE = 0.005
OHL_RELATIVE_TOLERANCE = 0.01
ABSOLUTE_PRICE_TOLERANCE = 0.02
MAX_BOUNDARY_RETURN = 0.20

OFFICIAL_SOURCE = "official_vip_veon_identity_repair"
OFFICIAL_SOURCE_KIND = "official_filing"
YAHOO_SOURCE = "yahoo_chart_vip_veon_crosscheck"
WIKI_AUDIT_SOURCE = "vip_veon_rejected_wiki_audit"
NASDAQ100_HISTORY_SOURCE = "community_nasdaq100_history"
NASDAQ100_HISTORY_SHA256 = (
    "83465af4e2f80f45ea239068ee41ba2069db990720896380c6ef8df4c1c9cb97"
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


@dataclass(frozen=True)
class WikiRejectedAudit:
    artifact: SourceArtifact
    full_archive_path: Path
    full_response_hash: str
    full_response_size: int
    total_data_rows: int
    ticker_rows: Mapping[str, int]


@dataclass(frozen=True)
class EvidenceBundle:
    sec_old_symbol: SourceArtifact
    sec: SourceArtifact
    yahoo: SourceArtifact
    yahoo_response: YahooChartCachedResponse
    yahoo_data: YahooChartData
    wiki: WikiRejectedAudit
    metrics: Mapping[str, Any]

    @property
    def archive_artifacts(self) -> tuple[SourceArtifact, ...]:
        return (self.sec_old_symbol, self.sec, self.yahoo, self.wiki.artifact)


@dataclass
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    evidence: EvidenceBundle
    warnings: tuple[str, ...]
    summary: dict[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


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
    if not _text(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid VIP/VEON date: {value!r}")
    return pd.Timestamp(parsed).date().isoformat()


def _normalized_document_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    decoded = html.unescape(decoded).replace("\xa0", " ")
    decoded = re.sub(r"<[^>]+>", " ", decoded)
    decoded = re.sub(r"\s+", " ", decoded).strip().casefold()
    return decoded.replace("’", "'").replace("“", '"').replace("”", '"')


def _expected_sessions() -> tuple[str, ...]:
    import exchange_calendars as xcals

    values = xcals.get_calendar("XNYS").sessions_in_range(
        PRICE_START, TRANSITION_DATE
    )
    output = tuple(pd.Timestamp(value).date().isoformat() for value in values)
    if len(output) != YAHOO_EXPECTED_ROWS:
        raise RuntimeError(
            "Pinned VIP/VEON XNYS inventory changed: "
            f"expected={YAHOO_EXPECTED_ROWS}, observed={len(output)}"
        )
    return output


def _sec_cache_path(cache_root: Path, cache_key: str = SEC_CACHE_KEY) -> Path:
    return cache_root / "state/sec_lifecycle" / f"{cache_key}.bin"


def _load_pinned_sec_artifact(
    cache_root: Path,
    *,
    cache_key: str,
    source_url: str,
    source_hash: str,
    exact_bytes: int,
    retrieved_at: str,
    required_text_groups: tuple[tuple[str, ...], ...],
    label: str,
) -> SourceArtifact:
    path = _sec_cache_path(cache_root, cache_key)
    if not path.is_file():
        raise FileNotFoundError(f"Pinned VIP/VEON {label} SEC cache is absent: {path}")
    content = path.read_bytes()
    if len(content) != exact_bytes or sha256_bytes(content) != source_hash:
        raise ValueError(f"Pinned VIP/VEON {label} SEC filing hash/size changed.")
    normalized = _normalized_document_text(content)
    missing = [
        group
        for group in required_text_groups
        if not any(phrase.casefold() in normalized for phrase in group)
    ]
    if missing:
        raise ValueError(
            f"Pinned VIP/VEON {label} SEC filing no longer proves reviewed claims: "
            + repr(missing)
        )
    return SourceArtifact(
        source="sec_edgar_filing",
        source_url=source_url,
        retrieved_at=retrieved_at,
        content=content,
        content_type="text/plain",
    )


def load_sec_evidence(cache_root: Path) -> tuple[SourceArtifact, SourceArtifact]:
    """Load the reviewed two-filing chain: old VIP identity, then live VEON date."""

    old_symbol = _load_pinned_sec_artifact(
        cache_root,
        cache_key=OLD_SYMBOL_SEC_CACHE_KEY,
        source_url=OLD_SYMBOL_SEC_URL,
        source_hash=OLD_SYMBOL_SEC_SHA256,
        exact_bytes=OLD_SYMBOL_SEC_EXACT_BYTES,
        retrieved_at=OLD_SYMBOL_SEC_RETRIEVED_AT,
        required_text_groups=OLD_SYMBOL_SEC_REQUIRED_TEXT_GROUPS,
        label="old-symbol",
    )
    transition = _load_pinned_sec_artifact(
        cache_root,
        cache_key=SEC_CACHE_KEY,
        source_url=SEC_URL,
        source_hash=SEC_SHA256,
        exact_bytes=SEC_EXACT_BYTES,
        retrieved_at=SEC_RETRIEVED_AT,
        required_text_groups=SEC_REQUIRED_TEXT_GROUPS,
        label="trading-effective",
    )
    if old_symbol.source_hash == transition.source_hash:
        raise ValueError("VIP/VEON two-filing identity evidence unexpectedly collapsed.")
    return old_symbol, transition


def _one_source_archive_row(
    source_archive: pd.DataFrame,
    *,
    source_url: str,
    source_hash: str,
) -> pd.Series:
    matches = source_archive.loc[
        source_archive["source_url"].astype(str).eq(source_url)
        & source_archive["source_hash"].astype(str).eq(source_hash)
        & source_archive["archive_id"].astype(str).eq(source_hash)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one exact source_archive row for "
            f"{source_url}/{source_hash}; observed={len(matches)}"
        )
    return matches.iloc[0]


def _safe_archive_path(
    repository: LocalDatasetRepository,
    row: pd.Series,
) -> Path:
    relative = Path(_text(row.get("object_path")))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".gz":
        raise ValueError(f"Unsafe VIP/VEON archive object path: {relative}")
    root = repository.root.resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents or not path.is_file():
        raise ValueError(f"Missing/escaping VIP/VEON archive object: {relative}")
    return path


def audit_rejected_wiki_source(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
) -> WikiRejectedAudit:
    """Hash the full frozen mirror and prove it has no exact VIP/VEON rows."""

    row = _one_source_archive_row(
        source_archive,
        source_url=WIKI_URL,
        source_hash=WIKI_FULL_SHA256,
    )
    if _text(row.get("content_type")).lower() != "application/zip":
        raise ValueError("Frozen WIKI source_archive row is not the exact ZIP response.")
    path = _safe_archive_path(repository, row)
    temporary = Path("/tmp") / f"vip-veon-wiki-{uuid.uuid4().hex}.zip"
    digest = hashlib.sha256()
    size = 0
    try:
        try:
            with gzip.open(path, "rb") as source, temporary.open("wb") as target:
                while chunk := source.read(4 * 1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    if size > WIKI_FULL_SIZE:
                        raise ValueError("Frozen WIKI response exceeds its exact size pin.")
                    target.write(chunk)
        except (EOFError, OSError) as exc:
            raise ValueError("Frozen WIKI gzip archive is unreadable/truncated.") from exc
        observed_hash = digest.hexdigest()
        if observed_hash != WIKI_FULL_SHA256 or size != WIKI_FULL_SIZE:
            raise ValueError(
                "Frozen WIKI full response changed: "
                f"sha256={observed_hash}, bytes={size}"
            )
        with zipfile.ZipFile(temporary) as archive:
            files = [item for item in archive.infolist() if not item.is_dir()]
            if len(files) != 1 or Path(files[0].filename).name != WIKI_MEMBER_NAME:
                raise ValueError("Frozen WIKI ZIP member inventory changed.")
            ticker_rows = {ticker: 0 for ticker in WIKI_REJECTED_TICKERS}
            total_rows = 0
            with archive.open(files[0], "r") as handle:
                header = handle.readline().decode("utf-8-sig", errors="strict")
                if not header.startswith("ticker,date,open,high,low,close,volume,"):
                    raise ValueError("Frozen WIKI CSV header changed.")
                for line in handle:
                    total_rows += 1
                    ticker = line.split(b",", 1)[0].decode("ascii", errors="strict")
                    if ticker in ticker_rows:
                        ticker_rows[ticker] += 1
        if total_rows != WIKI_TOTAL_DATA_ROWS:
            raise ValueError(
                "Frozen WIKI total row count changed: "
                f"expected={WIKI_TOTAL_DATA_ROWS}, observed={total_rows}"
            )
        if any(ticker_rows.values()):
            raise ValueError(
                "Frozen WIKI is no longer a rejected zero-row VIP/VEON source: "
                f"{ticker_rows}"
            )
        audit_content = _canonical_json_bytes(
            {
                "schema": WIKI_AUDIT_SCHEMA,
                "source_url": WIKI_URL,
                "full_response_sha256": WIKI_FULL_SHA256,
                "full_response_bytes": WIKI_FULL_SIZE,
                "total_data_rows": WIKI_TOTAL_DATA_ROWS,
                "searched_exact_tickers": list(WIKI_REJECTED_TICKERS),
                "exact_ticker_rows": ticker_rows,
                "disposition": "rejected_no_target_rows",
                "permitted_as_price_source": False,
            }
        )
        artifact = SourceArtifact(
            source=WIKI_AUDIT_SOURCE,
            source_url=WIKI_URL,
            retrieved_at=_text(row.get("retrieved_at")),
            content=audit_content,
            content_type="application/json",
        )
        return WikiRejectedAudit(
            artifact=artifact,
            full_archive_path=path,
            full_response_hash=observed_hash,
            full_response_size=size,
            total_data_rows=total_rows,
            ticker_rows=dict(ticker_rows),
        )
    finally:
        temporary.unlink(missing_ok=True)


def _yahoo_cache(
    cache_root: Path,
    *,
    factory: Callable[..., YahooChartCache] = YahooChartCache,
) -> YahooChartCache:
    return factory(
        cache_root,
        max_http_attempts=MAX_YAHOO_HTTP_ATTEMPTS,
        timeout_seconds=30.0,
        max_response_bytes=50 * 1024 * 1024,
    )


def _expected_yahoo_url(cache: YahooChartCache) -> str:
    return cache.url(
        NEW_SYMBOL,
        period1=YAHOO_PERIOD1,
        period2=YAHOO_PERIOD2,
    )


def _canonical_eodhd_window(prices: pd.DataFrame) -> pd.DataFrame:
    sessions = pd.to_datetime(prices["session"], errors="coerce")
    target = prices.loc[
        prices["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
        & sessions.ge(pd.Timestamp(PRICE_START))
        & sessions.le(pd.Timestamp(TRANSITION_DATE))
    ].copy()
    target["session"] = pd.to_datetime(target["session"], errors="coerce").dt.normalize()
    target = target.sort_values("session").reset_index(drop=True)
    expected = _expected_sessions()
    actual = tuple(target["session"].dt.date.astype(str))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            "Stored EODHD VEON history does not exactly cover the bounded XNYS "
            f"inventory: rows={len(actual)}, missing={missing}, extra={extra}"
        )
    if target["session"].duplicated().any():
        raise ValueError("Stored EODHD VEON bounded history contains duplicate sessions.")
    if set(target["currency"].astype(str).str.upper()) != {"USD"}:
        raise ValueError("Stored EODHD VEON bounded history is not exactly USD.")
    if set(target["source"].astype(str)) != {"eodhd_eod"}:
        raise ValueError("Stored VEON bounded history is not exclusively EODHD EOD.")
    if set(target["source_hash"].astype(str)) != {EODHD_VEON_EOD_SHA256}:
        raise ValueError("Stored EODHD VEON raw response hash changed.")
    return target


def validate_yahoo_crosscheck(
    response: YahooChartCachedResponse,
    prices: pd.DataFrame,
    *,
    require_pin: bool,
    cache: YahooChartCache,
) -> tuple[YahooChartData, dict[str, Any]]:
    expected_url = _expected_yahoo_url(cache)
    identity_matches = (
        response.symbol == NEW_SYMBOL
        and response.source_url == expected_url
        and response.request_period1 == YAHOO_PERIOD1
        and response.request_period2 == YAHOO_PERIOD2
    )
    if not identity_matches:
        raise ValueError("Yahoo VIP/VEON response URL/symbol/bounds changed.")
    if response.http_status != 200:
        raise ValueError(f"Yahoo VEON bounded response returned HTTP {response.http_status}.")
    if response.content_type.lower().split(";", 1)[0].strip() != "application/json":
        raise ValueError("Yahoo VEON bounded response is not JSON.")
    if require_pin:
        if not YAHOO_SHA256 or not YAHOO_WRAPPER_SHA256:
            raise ValueError(
                "Yahoo VEON exact content/wrapper hashes await reviewed code pins; "
                "offline plan/apply remain blocked."
            )
        if (
            response.source_hash != YAHOO_SHA256
            or response.wrapper_hash != YAHOO_WRAPPER_SHA256
        ):
            raise ValueError("Yahoo VEON exact content/wrapper pin changed.")

    parsed = parse_yahoo_chart_json(response.content, NEW_SYMBOL)
    provider = parsed.bars.copy()
    provider["session"] = pd.to_datetime(provider["session"]).dt.normalize()
    expected = _expected_sessions()
    actual = tuple(provider["session"].dt.date.astype(str))
    if actual != expected or provider["session"].duplicated().any():
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            "Yahoo VEON response failed exact bounded XNYS inventory: "
            f"rows={len(actual)}, missing={missing}, extra={extra}"
        )

    eodhd = _canonical_eodhd_window(prices)
    merged = eodhd.merge(
        provider,
        on="session",
        suffixes=("_eodhd", "_yahoo"),
        validate="one_to_one",
    )
    if len(merged) != YAHOO_EXPECTED_ROWS:
        raise ValueError("Yahoo/EODHD VEON overlap is not the exact bounded inventory.")
    for column in ("open", "high", "low", "close", "volume"):
        merged[f"{column}_eodhd"] = pd.to_numeric(
            merged[f"{column}_eodhd"], errors="coerce"
        )
        merged[f"{column}_yahoo"] = pd.to_numeric(
            merged[f"{column}_yahoo"], errors="coerce"
        )
    if merged.filter(regex="_(eodhd|yahoo)$").isna().any().any():
        raise ValueError("Yahoo/EODHD VEON overlap contains non-numeric values.")

    close_ratio = merged["close_yahoo"] / merged["close_eodhd"]
    median_scale = float(close_ratio.median())
    scale_deviation = ((close_ratio / median_scale) - 1.0).abs()
    maximum_scale_deviation = float(scale_deviation.max())
    if (
        not math.isfinite(median_scale)
        or median_scale <= 0
        or maximum_scale_deviation > MAX_SCALE_DEVIATION
    ):
        raise ValueError(
            "Yahoo/EODHD VEON scale is unstable: "
            f"median={median_scale}, max_deviation={maximum_scale_deviation}"
        )

    mismatch_counts: dict[str, int] = {}
    maximum_relative_errors: dict[str, float] = {}
    for column in ("open", "high", "low", "close"):
        normalized = merged[f"{column}_yahoo"] / median_scale
        expected_values = merged[f"{column}_eodhd"]
        absolute = (normalized - expected_values).abs()
        relative = absolute / expected_values.abs().clip(lower=1e-12)
        tolerance = (
            CLOSE_RELATIVE_TOLERANCE
            if column == "close"
            else OHL_RELATIVE_TOLERANCE
        )
        passed = absolute.le(
            np.maximum(ABSOLUTE_PRICE_TOLERANCE, expected_values.abs() * tolerance)
        )
        mismatch_counts[column] = int((~passed).sum())
        maximum_relative_errors[column] = float(relative.max())
    if sum(mismatch_counts.values()):
        raise ValueError(
            "Yahoo/EODHD VEON OHLC tolerance mismatch: " + repr(mismatch_counts)
        )

    eod_return = merged["close_eodhd"].pct_change()
    yahoo_return = merged["close_yahoo"].pct_change()
    valid = eod_return.notna() & yahoo_return.notna()
    return_correlation = float(eod_return[valid].corr(yahoo_return[valid]))
    return_error = (eod_return[valid] - yahoo_return[valid]).abs()
    if not math.isfinite(return_correlation) or return_correlation < MIN_RETURN_CORRELATION:
        raise ValueError(
            "Yahoo/EODHD VEON return correlation is too low: "
            f"{return_correlation}"
        )

    boundary = merged.loc[
        merged["session"].isin(
            [pd.Timestamp(OLD_LAST_SESSION), pd.Timestamp(TRANSITION_DATE)]
        )
    ].sort_values("session")
    if tuple(boundary["session"].dt.date.astype(str)) != (
        OLD_LAST_SESSION,
        TRANSITION_DATE,
    ):
        raise ValueError("VIP/VEON adjacent transition sessions are missing.")
    eod_boundary_return = float(
        boundary.iloc[1]["close_eodhd"] / boundary.iloc[0]["close_eodhd"] - 1.0
    )
    yahoo_boundary_return = float(
        boundary.iloc[1]["close_yahoo"] / boundary.iloc[0]["close_yahoo"] - 1.0
    )
    if (
        abs(eod_boundary_return) > MAX_BOUNDARY_RETURN
        or abs(yahoo_boundary_return) > MAX_BOUNDARY_RETURN
        or abs(eod_boundary_return - yahoo_boundary_return) > 0.01
    ):
        raise ValueError(
            "VIP/VEON adjacent-session continuity failed: "
            f"eodhd={eod_boundary_return}, yahoo={yahoo_boundary_return}"
        )

    volume_ratio = merged["volume_yahoo"] / merged["volume_eodhd"].replace(0, np.nan)
    return parsed, {
        "request_url": expected_url,
        "request_period1": YAHOO_PERIOD1,
        "request_period2": YAHOO_PERIOD2,
        "request_start": PRICE_START,
        "request_end": TRANSITION_DATE,
        "request_period2_is_exclusive": True,
        "yahoo_response_sha256": response.source_hash,
        "yahoo_wrapper_sha256": response.wrapper_hash,
        "yahoo_http_status": response.http_status,
        "yahoo_data_granularity": "1d",
        "xnys_session_count": len(expected),
        "yahoo_session_count": len(provider),
        "eodhd_session_count": len(eodhd),
        "overlap_session_count": len(merged),
        "all_sessions_compared": True,
        "median_yahoo_to_eodhd_close_scale": median_scale,
        "maximum_close_scale_deviation": maximum_scale_deviation,
        "ohlc_mismatch_counts": mismatch_counts,
        "maximum_relative_errors": maximum_relative_errors,
        "return_observation_count": int(valid.sum()),
        "close_return_correlation": return_correlation,
        "maximum_close_return_absolute_error": float(return_error.max()),
        "p99_close_return_absolute_error": float(return_error.quantile(0.99)),
        "boundary_old_symbol_last_session": OLD_LAST_SESSION,
        "boundary_new_symbol_first_session": TRANSITION_DATE,
        "eodhd_old_symbol_last_close": float(boundary.iloc[0]["close_eodhd"]),
        "eodhd_new_symbol_first_close": float(boundary.iloc[1]["close_eodhd"]),
        "yahoo_old_symbol_last_close": float(boundary.iloc[0]["close_yahoo"]),
        "yahoo_new_symbol_first_close": float(boundary.iloc[1]["close_yahoo"]),
        "eodhd_boundary_return": eod_boundary_return,
        "yahoo_boundary_return": yahoo_boundary_return,
        "median_volume_ratio_record_only": float(volume_ratio.median()),
        "volume_compared_for_gate": False,
        "price_validation_passed": True,
    }


def _yahoo_artifact(response: YahooChartCachedResponse) -> SourceArtifact:
    return SourceArtifact(
        source=YAHOO_SOURCE,
        source_url=response.source_url,
        retrieved_at=response.retrieved_at,
        content=response.content,
        content_type=response.content_type,
    )


def load_evidence(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    yahoo_cache_root: Path,
    require_yahoo_pin: bool,
    yahoo_factory: Callable[..., YahooChartCache] = YahooChartCache,
) -> EvidenceBundle:
    sec_old_symbol, sec = load_sec_evidence(repository.root)
    wiki = audit_rejected_wiki_source(repository, source_archive)
    cache = _yahoo_cache(yahoo_cache_root, factory=yahoo_factory)
    response = cache.get(
        NEW_SYMBOL,
        period1=YAHOO_PERIOD1,
        period2=YAHOO_PERIOD2,
    )
    if response is None:
        raise FileNotFoundError(
            "Bounded Yahoo VEON cache is absent; run acquisition-only "
            "--fetch-yahoo-evidence once, then review and code-pin its hashes."
        )
    yahoo_data, metrics = validate_yahoo_crosscheck(
        response,
        prices,
        require_pin=require_yahoo_pin,
        cache=cache,
    )
    return EvidenceBundle(
        sec_old_symbol=sec_old_symbol,
        sec=sec,
        yahoo=_yahoo_artifact(response),
        yahoo_response=response,
        yahoo_data=yahoo_data,
        wiki=wiki,
        metrics=metrics,
    )


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    matches = frame.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"Expected one {label}; observed={len(matches)}")
    return matches.iloc[0]


def _identity_preflight(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    master = frames["security_master"]
    history = frames["symbol_history"]
    prices = frames["daily_price_raw"]
    actions = frames["corporate_actions"]

    old_master = _one_row(
        master,
        master["security_id"].astype(str).eq(OLD_SECURITY_ID),
        "legacy VIP master row",
    )
    canonical_master = _one_row(
        master,
        master["security_id"].astype(str).eq(CANONICAL_SECURITY_ID),
        "canonical VEON master row",
    )
    if (
        _text(old_master.get("primary_symbol")).upper() != OLD_SYMBOL
        or _text(canonical_master.get("primary_symbol")).upper() != NEW_SYMBOL
        or _date(old_master.get("active_to")) != SOURCE_OLD_LAST_SESSION
        or _date(canonical_master.get("active_to"))
    ):
        raise ValueError("VIP/VEON master identity boundaries changed.")
    old_history = _one_row(
        history,
        history["security_id"].astype(str).eq(OLD_SECURITY_ID)
        & history["symbol"].astype(str).str.upper().eq(OLD_SYMBOL),
        "legacy VIP symbol history",
    )
    canonical_history = _one_row(
        history,
        history["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
        & history["symbol"].astype(str).str.upper().eq(NEW_SYMBOL),
        "canonical VEON symbol history",
    )
    if (
        _date(old_history.get("effective_from")) != HISTORY_START
        or _date(old_history.get("effective_to")) != SOURCE_OLD_LAST_SESSION
        or _date(canonical_history.get("effective_from")) != SOURCE_NEW_FIRST_SESSION
        or _date(canonical_history.get("effective_to"))
    ):
        raise ValueError("VIP/VEON symbol history boundaries changed.")

    old_prices = prices.loc[
        prices["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ].copy()
    canonical_window = _canonical_eodhd_window(prices)
    if old_prices.empty:
        raise ValueError("Expected contaminated VIP endpoint history before repair.")
    if set(old_prices["source_hash"].astype(str)) != {
        EODHD_VIP_REJECTED_EOD_SHA256
    }:
        raise ValueError("Rejected VIP endpoint response hash changed.")
    old_sessions = pd.to_datetime(old_prices["session"], errors="coerce")
    if (
        old_sessions.isna().any()
        or old_sessions.max().date().isoformat() != SOURCE_OLD_LAST_SESSION
    ):
        raise ValueError("Rejected VIP endpoint terminal boundary changed.")
    old_terminal = float(
        old_prices.loc[
            old_sessions.eq(pd.Timestamp(SOURCE_OLD_LAST_SESSION)), "close"
        ].iloc[0]
    )
    canonical_terminal = float(
        canonical_window.loc[
            canonical_window["session"].eq(pd.Timestamp(SOURCE_OLD_LAST_SESSION)),
            "close",
        ].iloc[0]
    )
    contamination_scale = old_terminal / canonical_terminal
    if old_terminal < 50 or canonical_terminal > 20 or contamination_scale < 10:
        raise ValueError("Rejected VIP price contamination signature changed.")

    old_actions = actions.loc[
        actions["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ]
    canonical_actions = actions.loc[
        actions["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ]
    if old_actions.empty or canonical_actions.empty:
        raise ValueError("Expected both duplicated VIP and canonical VEON actions.")
    return {
        "contaminated_vip_price_rows_removed": len(old_prices),
        "contaminated_vip_action_rows_removed": len(old_actions),
        "contaminated_vip_terminal_close": old_terminal,
        "canonical_veon_same_day_close": canonical_terminal,
        "contamination_close_scale": contamination_scale,
    }


def _official_action(evidence: SourceArtifact) -> dict[str, Any]:
    return {
        "event_id": canonical_lifecycle_event_id(
            CANONICAL_SECURITY_ID, "ticker_change", TRANSITION_DATE
        ),
        "security_id": CANONICAL_SECURITY_ID,
        "action_type": "ticker_change",
        "effective_date": TRANSITION_DATE,
        "ex_date": TRANSITION_DATE,
        "announcement_date": LEGAL_NAME_APPROVAL_DATE,
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "ratio": None,
        "currency": "USD",
        "new_security_id": CANONICAL_SECURITY_ID,
        "new_symbol": NEW_SYMBOL,
        "official": True,
        "source_url": evidence.source_url,
        "source_kind": OFFICIAL_SOURCE_KIND,
        "source": OFFICIAL_SOURCE,
        "retrieved_at": evidence.retrieved_at,
        "source_hash": evidence.source_hash,
    }


# Code-side handoff for the reviewed-nonterminal registry.  The shared YAML and
# its trusted fingerprint are updated only after this exact repair has passed
# cache-only review and writer coordination.
REVIEWED_NONTERMINAL_EXTRACTION = {
    key: _official_action(
        SourceArtifact(
            source="sec_edgar_filing",
            source_url=SEC_URL,
            retrieved_at=SEC_RETRIEVED_AT,
            content=b"",
            content_type="text/plain",
        )
    )[key]
    for key in (
        "event_id",
        "security_id",
        "action_type",
        "effective_date",
        "new_security_id",
        "new_symbol",
        "ratio",
        "cash_amount",
        "currency",
        "source_kind",
        "source_url",
    )
}
REVIEWED_NONTERMINAL_EXTRACTION["source_hash"] = SEC_SHA256


def _rewrite_master_history(
    frames: Mapping[str, pd.DataFrame],
    evidence: SourceArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = frames["security_master"].copy()
    history = frames["symbol_history"].copy()
    ids = master["security_id"].astype(str)
    master = master.loc[~ids.eq(OLD_SECURITY_ID)].copy()
    canonical = master["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    if int(canonical.sum()) != 1:
        raise ValueError("Canonical VEON identity disappeared during rewrite.")
    updates = {
        "primary_symbol": NEW_SYMBOL,
        "name": "VEON Ltd. (formerly VimpelCom Ltd.)",
        "active_from": PRICE_START,
        "active_to": "",
        "source": OFFICIAL_SOURCE,
        "source_url": evidence.source_url,
        "retrieved_at": evidence.retrieved_at,
        "source_hash": evidence.source_hash,
    }
    if "provider_symbol" in master:
        updates["provider_symbol"] = f"{NEW_SYMBOL}.US"
    if "action_provider_symbol" in master:
        updates["action_provider_symbol"] = f"{NEW_SYMBOL}.US"
    for column, value in updates.items():
        master.loc[canonical, column] = value

    old_template = _one_row(
        history,
        history["security_id"].astype(str).eq(OLD_SECURITY_ID)
        & history["symbol"].astype(str).str.upper().eq(OLD_SYMBOL),
        "VIP history template",
    ).copy()
    new_template = _one_row(
        history,
        history["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
        & history["symbol"].astype(str).str.upper().eq(NEW_SYMBOL),
        "VEON history template",
    ).copy()
    base_columns = list(history.columns)
    rows = []
    for template, symbol, start, end in (
        (old_template, OLD_SYMBOL, HISTORY_START, OLD_LAST_SESSION),
        (new_template, NEW_SYMBOL, TRANSITION_DATE, ""),
    ):
        row = template.copy()
        row["security_id"] = CANONICAL_SECURITY_ID
        row["symbol"] = symbol
        row["effective_from"] = start
        row["effective_to"] = end
        row["source"] = OFFICIAL_SOURCE
        row["source_url"] = evidence.source_url
        row["retrieved_at"] = evidence.retrieved_at
        row["source_hash"] = evidence.source_hash
        rows.append(row)
    affected = history["security_id"].astype(str).isin(
        {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
    )
    additions = pd.DataFrame(rows)
    history = pd.concat(
        [history.loc[~affected], additions.loc[:, base_columns]],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(
        list(dataset_spec("symbol_history").primary_key), keep="last"
    )
    return master.reset_index(drop=True), history.reset_index(drop=True)


def _rewrite_prices_actions_factors(
    frames: Mapping[str, pd.DataFrame],
    evidence: SourceArtifact,
    *,
    source_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = frames["daily_price_raw"].copy()
    actions = frames["corporate_actions"].copy()
    factors = frames["adjustment_factors"].copy()

    prices = prices.loc[
        ~prices["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ].copy()
    prices = prices.drop_duplicates(
        list(dataset_spec("daily_price_raw").primary_key), keep="last"
    )
    actions = actions.loc[
        ~actions["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ].copy()
    action = pd.DataFrame([_official_action(evidence)])
    actions = pd.concat([actions, action], ignore_index=True, sort=False).drop_duplicates(
        list(dataset_spec("corporate_actions").primary_key), keep="last"
    )

    target_prices = prices.loc[
        prices["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ].copy()
    target_actions = actions.loc[
        actions["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ].copy()
    rebuilt = build_adjustment_factors(
        target_prices,
        target_actions,
        source_version=source_version,
    )
    factors = pd.concat(
        [
            factors.loc[
                ~factors["security_id"].astype(str).isin(
                    {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
                )
            ],
            rebuilt,
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(
        list(dataset_spec("adjustment_factors").primary_key), keep="last"
    )
    return (
        prices.sort_values(["security_id", "session"]).reset_index(drop=True),
        actions.sort_values(["security_id", "effective_date", "event_id"]).reset_index(
            drop=True
        ),
        factors.sort_values(["security_id", "session"]).reset_index(drop=True),
    )


def _remapped_index_event_id(row: Mapping[str, Any]) -> str:
    return sha256_bytes(
        _canonical_json_bytes(
            {
                "operation": "vip_veon_index_identity_remap/v1",
                "prior_event_id": _text(row.get("event_id")),
                "index_id": _text(row.get("index_id")),
                "effective_date": _date(row.get("effective_date")),
                "membership_operation": _text(row.get("operation")).upper(),
                "security_id": CANONICAL_SECURITY_ID,
            }
        )
    )


def _rewrite_index_references(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    anchors = frames["index_constituent_anchors"].copy()
    events = frames["index_membership_events"].copy()
    anchor_mask = anchors["security_id"].astype(str).eq(OLD_SECURITY_ID)
    provenance_columns = tuple(
        column
        for column in (
            "official",
            "source",
            "source_url",
            "source_kind",
            "retrieved_at",
            "source_hash",
        )
        if column in anchors.columns
    )
    prior_anchor_provenance = anchors.loc[anchor_mask, provenance_columns].copy()
    anchors.loc[anchor_mask, "security_id"] = CANONICAL_SECURITY_ID
    event_mask = events["security_id"].astype(str).eq(OLD_SECURITY_ID)
    event_provenance_columns = tuple(
        column
        for column in (
            "official",
            "source",
            "source_url",
            "source_kind",
            "retrieved_at",
            "source_hash",
        )
        if column in events.columns
    )
    prior_event_provenance = events.loc[event_mask, event_provenance_columns].copy()
    for index in events.index[event_mask]:
        prior = events.loc[index].to_dict()
        events.loc[index, "security_id"] = CANONICAL_SECURITY_ID
        events.loc[index, "event_id"] = _remapped_index_event_id(prior)
    if not anchors.loc[anchor_mask, provenance_columns].equals(
        prior_anchor_provenance
    ):
        raise ValueError("VIP/VEON index anchor provenance changed during rekey.")
    if not events.loc[event_mask, event_provenance_columns].equals(
        prior_event_provenance
    ):
        raise ValueError("VIP/VEON index event provenance changed during rekey.")
    anchor_key = list(dataset_spec("index_constituent_anchors").primary_key)
    event_key = list(dataset_spec("index_membership_events").primary_key)
    if anchors.duplicated(anchor_key, keep=False).any():
        raise ValueError("VIP/VEON index anchor rekey collides with an existing row.")
    if events.duplicated(event_key, keep=False).any():
        raise ValueError("VIP/VEON index event rekey collides with an existing row.")
    anchors = anchors.reset_index(drop=True)
    events = events.reset_index(drop=True)
    return anchors, events, {
        "index_anchor_rows_rekeyed": int(anchor_mask.sum()),
        "index_event_rows_rekeyed": int(event_mask.sum()),
    }


def _archive_extension(artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower().split(";", 1)[0].strip()
    if content_type == "application/json":
        return "json"
    if content_type == "text/plain":
        return "txt"
    if content_type in {"text/html", "application/xhtml+xml"}:
        return "html"
    return "bin"


def _append_source_archive(
    source_archive: pd.DataFrame,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> pd.DataFrame:
    rows = []
    for artifact in artifacts:
        extension = _archive_extension(artifact)
        rows.append(
            {
                "archive_id": artifact.source_hash,
                "dataset": artifact.source,
                "object_path": (
                    f"archives/{completed_session}/{artifact.source_hash}."
                    f"{extension}.gz"
                ),
                "content_type": artifact.content_type,
                "effective_date": completed_session,
                "source": artifact.source,
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    output = pd.concat(
        [source_archive, pd.DataFrame(rows)], ignore_index=True, sort=False
    )
    return output.drop_duplicates(
        list(dataset_spec("source_archive").primary_key), keep="last"
    ).reset_index(drop=True)


def _archive_pair_exists(
    archive: pd.DataFrame,
    source_url: str,
    source_hash: str,
) -> bool:
    return bool(
        (
            archive["source_url"].astype(str).eq(source_url)
            & archive["source_hash"].astype(str).eq(source_hash)
            & archive["archive_id"].astype(str).eq(source_hash)
        ).any()
    )


def _hash_gzip_payload(
    path: Path,
    *,
    expected_content: bytes | None = None,
    expected_size: int | None = None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with gzip.open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                if expected_size is not None and size + len(chunk) > expected_size:
                    raise ValueError(
                        f"VIP/VEON archive exceeds exact size pin: {path}"
                    )
                if expected_content is not None:
                    expected_chunk = expected_content[size : size + len(chunk)]
                    if chunk != expected_chunk:
                        raise ValueError(
                            f"VIP/VEON archive payload differs from evidence: {path}"
                        )
                digest.update(chunk)
                size += len(chunk)
    except (EOFError, OSError) as exc:
        raise ValueError(f"VIP/VEON archive is unreadable/truncated: {path}") from exc
    if expected_size is not None and size != expected_size:
        raise ValueError(
            f"VIP/VEON archive size changed: expected={expected_size}, observed={size}"
        )
    if expected_content is not None and size != len(expected_content):
        raise ValueError(
            f"VIP/VEON archive payload length differs from evidence: {path}"
        )
    return digest.hexdigest(), size


def _verify_persisted_archive_artifact(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    artifact: SourceArtifact,
) -> tuple[str, int]:
    row = _one_source_archive_row(
        archive,
        source_url=artifact.source_url,
        source_hash=artifact.source_hash,
    )
    path = _safe_archive_path(repository, row)
    if not path.name.startswith(f"{artifact.source_hash}."):
        raise ValueError(f"VIP/VEON archive filename is not content-addressed: {path}")
    if _text(row.get("content_type")) != artifact.content_type:
        raise ValueError(f"VIP/VEON archive content type changed: {path}")
    observed_hash, observed_size = _hash_gzip_payload(
        path,
        expected_content=artifact.content,
        expected_size=len(artifact.content),
    )
    if observed_hash != artifact.source_hash:
        raise ValueError(
            "VIP/VEON archived evidence hash changed: "
            f"expected={artifact.source_hash}, observed={observed_hash}"
        )
    return observed_hash, observed_size


def _verify_prevalidated_wiki_archive(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    wiki: WikiRejectedAudit,
) -> None:
    """Bind the source_archive row to the full WIKI bytes audited in this run."""

    row = _one_source_archive_row(
        archive,
        source_url=WIKI_URL,
        source_hash=WIKI_FULL_SHA256,
    )
    path = _safe_archive_path(repository, row)
    if not path.name.startswith(f"{WIKI_FULL_SHA256}."):
        raise ValueError("Frozen WIKI archive filename is not content-addressed.")
    if path.resolve() != wiki.full_archive_path.resolve():
        raise ValueError("Frozen WIKI source_archive path changed after full audit.")
    if (
        wiki.full_response_hash != WIKI_FULL_SHA256
        or wiki.full_response_size != WIKI_FULL_SIZE
    ):
        raise ValueError("Frozen WIKI full-audit result changed before validation.")


def _validate_index_membership_provenance(
    repository: LocalDatasetRepository,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Prove and retain the exact preexisting Nasdaq-100 VIP membership source."""

    anchors = frames["index_constituent_anchors"]
    events = frames["index_membership_events"]
    allowed_ids = {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
    anchor = _one_row(
        anchors,
        anchors["index_id"].astype(str).str.lower().eq("nasdaq100")
        & pd.to_datetime(anchors["anchor_date"], errors="coerce").eq(
            pd.Timestamp(HISTORY_START)
        )
        & anchors["security_id"].astype(str).isin(allowed_ids)
        & anchors["source"].astype(str).eq(NASDAQ100_HISTORY_SOURCE)
        & anchors["source_hash"].astype(str).eq(NASDAQ100_HISTORY_SHA256),
        "exact Nasdaq-100 VIP anchor provenance",
    )
    event = _one_row(
        events,
        events["index_id"].astype(str).str.lower().eq("nasdaq100")
        & pd.to_datetime(events["effective_date"], errors="coerce").eq(
            pd.Timestamp("2015-12-21")
        )
        & events["operation"].astype(str).str.upper().eq("REMOVE")
        & events["security_id"].astype(str).isin(allowed_ids)
        & events["source"].astype(str).eq(NASDAQ100_HISTORY_SOURCE)
        & events["source_hash"].astype(str).eq(NASDAQ100_HISTORY_SHA256),
        "exact Nasdaq-100 VIP removal provenance",
    )
    for column in ("source_url", "source_kind", "source_hash"):
        if not _text(anchor.get(column)) or _text(anchor.get(column)) != _text(
            event.get(column)
        ):
            raise ValueError(
                f"VIP Nasdaq-100 anchor/event provenance diverged: {column}"
            )
    archive = frames["source_archive"]
    row = _one_source_archive_row(
        archive,
        source_url=_text(anchor.get("source_url")),
        source_hash=NASDAQ100_HISTORY_SHA256,
    )
    path = _safe_archive_path(repository, row)
    if not path.name.startswith(f"{NASDAQ100_HISTORY_SHA256}."):
        raise ValueError("Nasdaq-100 membership archive is not content-addressed.")
    observed_hash, observed_size = _hash_gzip_payload(path)
    if observed_hash != NASDAQ100_HISTORY_SHA256:
        raise ValueError("Nasdaq-100 VIP membership archive hash changed.")
    try:
        content = gzip.decompress(path.read_bytes()).decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Nasdaq-100 VIP membership archive is unreadable.") from exc
    if not re.search(r"(?m)^\s*year:\s*2015\s*$", content):
        raise ValueError("Nasdaq-100 VIP membership archive lacks its 2015 year.")
    if not re.search(r"(?m)^\s*-\s+VIP\s*$", content):
        raise ValueError("Nasdaq-100 VIP membership archive lacks exact VIP ticker.")
    change = content.split("'2015-12-21':", 1)
    if len(change) != 2 or not re.search(
        r"(?ms)^\s*difference:\s*$.*?^\s*-\s+VIP\s*$", change[1]
    ):
        raise ValueError("Nasdaq-100 VIP removal provenance changed.")
    return {
        "index_membership_provenance_preserved": True,
        "index_membership_source_sha256": observed_hash,
        "index_membership_source_bytes": observed_size,
    }


def validate_repaired_frames(
    frames: Mapping[str, pd.DataFrame],
    evidence: EvidenceBundle,
    *,
    completed_session: str,
    repository: LocalDatasetRepository | None = None,
    require_persisted_archives: bool = False,
) -> dict[str, Any]:
    for dataset in WRITE_DATASETS:
        validate_dataset(
            dataset,
            frames[dataset],
            completed_session=completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()

    for dataset in (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "index_constituent_anchors",
        "index_membership_events",
    ):
        if frames[dataset]["security_id"].astype(str).eq(OLD_SECURITY_ID).any():
            raise ValueError(f"Retired VIP security_id remains in {dataset}.")

    master = frames["security_master"]
    row = _one_row(
        master,
        master["security_id"].astype(str).eq(CANONICAL_SECURITY_ID),
        "repaired VEON master row",
    )
    if (
        _text(row.get("primary_symbol")).upper() != NEW_SYMBOL
        or _text(row.get("provider_symbol")).upper() != f"{NEW_SYMBOL}.US"
        or _date(row.get("active_from")) != PRICE_START
        or _date(row.get("active_to"))
    ):
        raise ValueError("Repaired VEON master identity is not exact.")

    history = frames["symbol_history"].loc[
        frames["symbol_history"]["security_id"].astype(str).eq(
            CANONICAL_SECURITY_ID
        )
    ]
    intervals = {
        (_text(value.symbol).upper(), _date(value.effective_from), _date(value.effective_to))
        for value in history.itertuples(index=False)
        if _text(value.symbol).upper() in {OLD_SYMBOL, NEW_SYMBOL}
    }
    if intervals != {
        (OLD_SYMBOL, HISTORY_START, OLD_LAST_SESSION),
        (NEW_SYMBOL, TRANSITION_DATE, ""),
    }:
        raise ValueError(f"Repaired VIP/VEON symbol intervals changed: {intervals}")

    prices = frames["daily_price_raw"]
    if prices["source_hash"].astype(str).eq(EODHD_VIP_REJECTED_EOD_SHA256).any():
        raise ValueError("Contaminated VIP raw response remains in repaired prices.")
    canonical_price_mask = prices["security_id"].astype(str).eq(
        CANONICAL_SECURITY_ID
    )
    if prices.loc[
        canonical_price_mask,
        "source_url",
    ].astype(str).str.contains("query1.finance.yahoo.com", regex=False).any():
        raise ValueError("Yahoo cross-check bytes were incorrectly used as primary prices.")
    canonical_window = _canonical_eodhd_window(prices)
    _, replay_metrics = validate_yahoo_crosscheck(
        evidence.yahoo_response,
        prices,
        require_pin=True,
        cache=_yahoo_cache(Path("unused")),
    )
    if replay_metrics["overlap_session_count"] != YAHOO_EXPECTED_ROWS:
        raise ValueError("Repaired VEON Yahoo replay coverage changed.")

    actions = frames["corporate_actions"]
    event_id = canonical_lifecycle_event_id(
        CANONICAL_SECURITY_ID, "ticker_change", TRANSITION_DATE
    )
    action = _one_row(
        actions,
        actions["event_id"].astype(str).eq(event_id),
        "official VIP/VEON ticker change",
    )
    if not (
        _text(action.get("security_id")) == CANONICAL_SECURITY_ID
        and _text(action.get("action_type")) == "ticker_change"
        and _date(action.get("effective_date")) == TRANSITION_DATE
        and _text(action.get("new_security_id")) == CANONICAL_SECURITY_ID
        and _text(action.get("new_symbol")).upper() == NEW_SYMBOL
        and not _text(action.get("ratio"))
        and _text(action.get("source_url")) == SEC_URL
        and _text(action.get("source_hash")) == SEC_SHA256
        and bool(action.get("official"))
    ):
        raise ValueError("Official VIP/VEON ticker-change row is not exact.")

    factors = frames["adjustment_factors"]
    price_pairs = set(
        zip(
            canonical_window["security_id"].astype(str),
            canonical_window["session"].dt.date.astype(str),
        )
    )
    target_factors = factors.loc[
        factors["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ].copy()
    target_factors["session"] = pd.to_datetime(target_factors["session"])
    factor_pairs = set(
        zip(
            target_factors["security_id"].astype(str),
            target_factors["session"].dt.date.astype(str),
        )
    )
    if not price_pairs.issubset(factor_pairs):
        raise ValueError("Repaired VEON prices lack adjustment factors.")

    archive = frames["source_archive"]
    required_pairs = (
        (OLD_SYMBOL_SEC_URL, OLD_SYMBOL_SEC_SHA256),
        (SEC_URL, SEC_SHA256),
        (evidence.yahoo.source_url, evidence.yahoo.source_hash),
        (WIKI_URL, WIKI_FULL_SHA256),
        (evidence.wiki.artifact.source_url, evidence.wiki.artifact.source_hash),
    )
    for source_url, source_hash in required_pairs:
        if not _archive_pair_exists(archive, source_url, source_hash):
            raise ValueError(
                f"VIP/VEON evidence is not archived: {source_url}/{source_hash}"
            )
    if require_persisted_archives:
        if repository is None:
            raise ValueError(
                "Persisted VIP/VEON archive verification requires a repository."
            )
        _verify_prevalidated_wiki_archive(repository, archive, evidence.wiki)
        for artifact in evidence.archive_artifacts:
            _verify_persisted_archive_artifact(repository, archive, artifact)
    return {
        "canonical_security_id": CANONICAL_SECURITY_ID,
        "retired_security_id": OLD_SECURITY_ID,
        "canonical_bounded_price_rows": len(canonical_window),
        "official_ticker_change_rows": 1,
        "old_symbol_sec_evidence_sha256": OLD_SYMBOL_SEC_SHA256,
        "sec_evidence_sha256": SEC_SHA256,
        "yahoo_evidence_sha256": evidence.yahoo.source_hash,
        "yahoo_wrapper_sha256": evidence.yahoo_response.wrapper_hash,
        "wiki_full_response_sha256": evidence.wiki.full_response_hash,
        "wiki_exact_vip_rows": int(evidence.wiki.ticker_rows[OLD_SYMBOL]),
        "wiki_exact_veon_rows": int(evidence.wiki.ticker_rows[NEW_SYMBOL]),
        "persisted_archives_verified": bool(require_persisted_archives),
        "network_accessed": False,
        **dict(evidence.metrics),
    }


def prepare_repair_frames(
    frames: Mapping[str, pd.DataFrame],
    evidence: EvidenceBundle,
    *,
    completed_session: str,
    source_version: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    missing = sorted(set(WRITE_DATASETS) - set(frames))
    if missing:
        raise ValueError("VIP/VEON repair lacks datasets: " + ", ".join(missing))
    preflight = _identity_preflight(frames)
    master, history = _rewrite_master_history(frames, evidence.sec)
    prices, actions, factors = _rewrite_prices_actions_factors(
        frames,
        evidence.sec,
        source_version=source_version,
    )
    anchors, events, index_summary = _rewrite_index_references(frames)
    archive = _append_source_archive(
        frames["source_archive"],
        evidence.archive_artifacts,
        completed_session=completed_session,
    )
    rewritten = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": archive,
    }
    summary = validate_repaired_frames(
        rewritten,
        evidence,
        completed_session=completed_session,
    )
    return rewritten, {
        **summary,
        **preflight,
        **index_summary,
        "status": "validated_offline_plan",
    }


def _looks_repaired(frames: Mapping[str, pd.DataFrame]) -> bool:
    master = frames["security_master"]
    actions = frames["corporate_actions"]
    event_id = canonical_lifecycle_event_id(
        CANONICAL_SECURITY_ID, "ticker_change", TRANSITION_DATE
    )
    return bool(
        not master["security_id"].astype(str).eq(OLD_SECURITY_ID).any()
        and actions["event_id"].astype(str).eq(event_id).any()
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

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        version = self.versions.get(dataset)
        if not version:
            return pd.DataFrame(columns=dataset_spec(dataset).required_columns)
        return self.base.read_frame(dataset, version)


def _capture_pointer_etags(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        expected = release.dataset_versions.get(dataset)
        if pointer is None or pointer.version != expected:
            raise RuntimeError(f"VIP/VEON release/pointer mismatch: {dataset}")
        output[dataset] = etag
    return output


def prepare_run(
    repository: LocalDatasetRepository,
    *,
    yahoo_cache_root: Path,
    yahoo_factory: Callable[..., YahooChartCache] = YahooChartCache,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required for VIP/VEON repair.")
    missing = sorted(set(WRITE_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in WRITE_DATASETS
    }
    pointer_etags = _capture_pointer_etags(repository, release)
    membership_provenance = _validate_index_membership_provenance(
        repository, frames
    )
    evidence = load_evidence(
        repository,
        frames["source_archive"],
        frames["daily_price_raw"],
        yahoo_cache_root=yahoo_cache_root,
        require_yahoo_pin=True,
        yahoo_factory=yahoo_factory,
    )
    if _looks_repaired(frames):
        summary = validate_repaired_frames(
            frames,
            evidence,
            completed_session=release.completed_session,
            repository=repository,
            require_persisted_archives=True,
        )
        validate_repository_snapshot(repository).raise_for_errors()
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            frames={dataset: frames[dataset].copy() for dataset in WRITE_DATASETS},
            evidence=evidence,
            warnings=release.warnings,
            summary={
                **summary,
                **membership_provenance,
                "status": "already_repaired",
                "release_version": release.version,
            },
        )
    rewritten, summary = prepare_repair_frames(
        frames,
        evidence,
        completed_session=release.completed_session,
        source_version=f"vip-veon-identity-repair/{release.version}",
    )
    candidate = _CandidateRepository(repository, release.dataset_versions, rewritten)
    validate_repository_snapshot(candidate).raise_for_errors()
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        frames=rewritten,
        evidence=evidence,
        warnings=release.warnings,
        summary={
            **summary,
            **membership_provenance,
            "release_version": release.version,
        },
    )


def acquire_yahoo_evidence_only(
    repository: LocalDatasetRepository,
    *,
    yahoo_cache_root: Path,
    yahoo_factory: Callable[..., YahooChartCache] = YahooChartCache,
) -> dict[str, Any]:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current release is required for Yahoo acquisition.")
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in ("daily_price_raw", "source_archive")
    }
    sec_old_symbol, sec = load_sec_evidence(repository.root)
    wiki = audit_rejected_wiki_source(repository, frames["source_archive"])
    cache = _yahoo_cache(yahoo_cache_root, factory=yahoo_factory)
    response = cache.fetch(
        NEW_SYMBOL,
        period1=YAHOO_PERIOD1,
        period2=YAHOO_PERIOD2,
    )
    parsed, metrics = validate_yahoo_crosscheck(
        response,
        frames["daily_price_raw"],
        require_pin=False,
        cache=cache,
    )
    after, after_etag = repository.current_release()
    if (
        after is None
        or after.to_bytes() != release.to_bytes()
        or after_etag != release_etag
    ):
        raise RuntimeError("Release changed during acquisition-only Yahoo fetch.")
    configured_match = bool(YAHOO_SHA256 and YAHOO_WRAPPER_SHA256) and (
        response.source_hash == YAHOO_SHA256
        and response.wrapper_hash == YAHOO_WRAPPER_SHA256
    )
    return {
        "status": (
            "yahoo_evidence_ready"
            if configured_match
            else "yahoo_evidence_observed_unpinned"
        ),
        "release_version": release.version,
        "source_url": response.source_url,
        "observed_sha256": response.source_hash,
        "observed_wrapper_sha256": response.wrapper_hash,
        "configured_sha256": YAHOO_SHA256,
        "configured_wrapper_sha256": YAHOO_WRAPPER_SHA256,
        "configured_hashes_match": configured_match,
        "exact_byte_count": len(response.content),
        "cache_path": str(
            cache.path(
                NEW_SYMBOL,
                period1=YAHOO_PERIOD1,
                period2=YAHOO_PERIOD2,
            )
        ),
        "yahoo_http_attempts": cache.http_attempts,
        "yahoo_max_http_attempts": MAX_YAHOO_HTTP_ATTEMPTS,
        "parsed_rows": len(parsed.bars),
        "legacy_request_period2": LEGACY_YAHOO_PERIOD2,
        "legacy_expected_rows": LEGACY_YAHOO_EXPECTED_ROWS,
        "legacy_response_sha256": LEGACY_YAHOO_SHA256,
        "legacy_wrapper_sha256": LEGACY_YAHOO_WRAPPER_SHA256,
        "old_symbol_sec_sha256": sec_old_symbol.source_hash,
        "sec_sha256": sec.source_hash,
        "wiki_full_sha256": wiki.full_response_hash,
        "wiki_exact_ticker_rows": dict(wiki.ticker_rows),
        "release_mutated": False,
        "apply_allowed": configured_match,
        **metrics,
    }


def _persist_archive_payloads(
    repository: LocalDatasetRepository,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> None:
    for artifact in artifacts:
        extension = _archive_extension(artifact)
        path = (
            repository.root
            / "archives"
            / completed_session
            / f"{artifact.source_hash}.{extension}.gz"
        )
        encoded = gzip.compress(artifact.content, mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            try:
                existing = gzip.decompress(path.read_bytes())
            except Exception as exc:
                raise ValueError(f"Existing VIP/VEON archive is unreadable: {path}") from exc
            if existing != artifact.content:
                raise RuntimeError(f"Immutable VIP/VEON archive changed: {path}")
        else:
            write_atomic(path, encoded)
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"VIP/VEON archive verification failed: {path}")


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
                "A recovery marker blocks VIP/VEON writes: "
                + ", ".join(str(item) for item in pending)
            )
        transactions = repository.root / "transactions"
        interrupted: list[Path] = []
        if transactions.exists():
            for item in transactions.rglob("*.json"):
                try:
                    status = _text(json.loads(item.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    interrupted.append(item)
        if interrupted:
            raise RuntimeError(
                "An interrupted transaction blocks VIP/VEON writes: "
                + ", ".join(str(item) for item in interrupted)
            )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, _canonical_json_bytes(dict(value)))


def _restore_transaction(
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
            observed = DataRelease.from_bytes(current.data)
            ours = observed.version == committed_release_version or all(
                observed.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not ours:
                raise RuntimeError(
                    f"Unexpected release during VIP/VEON rollback: {observed.version}"
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
            if current.data != old_pointer_bytes[dataset]:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"Unexpected VIP/VEON pointer during rollback: {pointer.version}"
                    )
                repository.objects.put(
                    key, old_pointer_bytes[dataset], if_match=current.etag
                )
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_repaired":
        return dict(prepared.summary)
    with _exclusive_repository_lock(repository):
        current, current_etag = repository.current_release()
        if (
            current is None
            or current.version != prepared.release.version
            or current_etag != prepared.release_etag
        ):
            raise RuntimeError("Current release changed after VIP/VEON preflight.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"VIP/VEON pointer changed before apply: {dataset}")
            old_pointers[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        session = prepared.release.completed_session.replace("-", "")
        planned = {
            dataset: f"vip-veon-identity-{session}-{transaction_id}-{dataset}"
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/vip-veon-identity-repair"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "vip_veon_identity_repair_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
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
                prepared.evidence.archive_artifacts,
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
                        "operation": "repair_us_vip_veon_identity",
                        "canonical_security_id": CANONICAL_SECURITY_ID,
                        "retired_security_id": OLD_SECURITY_ID,
                        "legal_name_approval_date": LEGAL_NAME_APPROVAL_DATE,
                        "transition_date": TRANSITION_DATE,
                        "old_symbol_sec_evidence_sha256": OLD_SYMBOL_SEC_SHA256,
                        "sec_evidence_sha256": SEC_SHA256,
                        "yahoo_evidence_sha256": YAHOO_SHA256,
                        "index_membership_source_sha256": (
                            NASDAQ100_HISTORY_SHA256
                        ),
                        "wiki_source_disposition": "rejected_no_target_rows",
                        "network_accessed": False,
                    },
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"VIP/VEON write conflicted: {dataset}/{result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version
            written = {
                dataset: repository.read_frame(dataset, planned[dataset])
                for dataset in WRITE_DATASETS
            }
            validate_repaired_frames(
                written,
                prepared.evidence,
                completed_session=prepared.release.completed_session,
                repository=repository,
                require_persisted_archives=True,
            )
            candidate = _CandidateRepository(repository, versions, written)
            validate_repository_snapshot(candidate).raise_for_errors()
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=(
                    DataQuality.DEGRADED
                    if prepared.warnings
                    else DataQuality.VALID
                ),
                warnings=prepared.warnings,
                expected_etag=prepared.release_etag,
            )
            latest, _ = repository.current_release()
            if latest is None or latest.to_bytes() != committed.to_bytes():
                raise RuntimeError("Committed VIP/VEON release is not current.")
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
                    / "recovery/vip-veon-identity-repair"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "VIP/VEON rollback failed; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}"
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair the VIP -> VEON continuous NASDAQ ADS identity."
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--yahoo-cache-dir", default=str(DEFAULT_YAHOO_CACHE))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fetch-yahoo-evidence",
        action="store_true",
        help=(
            "Acquisition-only: make at most one exact bounded no-retry Yahoo "
            "VEON request, validate it, print observed hashes, and never write a release."
        ),
    )
    mode.add_argument("--offline-plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[[str | Path], LocalDatasetRepository] = (
        LocalDatasetRepository
    ),
    yahoo_factory: Callable[..., YahooChartCache] = YahooChartCache,
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    yahoo_root = Path(args.yahoo_cache_dir)
    if bool(getattr(args, "fetch_yahoo_evidence", False)):
        return acquire_yahoo_evidence_only(
            repository,
            yahoo_cache_root=yahoo_root,
            yahoo_factory=yahoo_factory,
        )
    prepared = prepare_run(
        repository,
        yahoo_cache_root=yahoo_root,
        yahoo_factory=yahoo_factory,
    )
    if not bool(getattr(args, "apply", False)):
        return prepared.summary
    return apply_repair(repository, prepared)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
