#!/usr/bin/env python3
"""Repair audited US index identity collisions as one frozen transaction.

The bootstrap sources normalize historical rows to today's ticker.  That is
unsafe when a ticker was reused by a different issuer (SE, FOX/FOXA, LILA/K)
or when an index constituent changed ticker across a reorganization (AGN,
COR, BHGE/BKR, IR/TT, ARNC/HWM).  This collector repairs the identity graph,
price ownership and index references together.

``--offline-plan`` never constructs a provider client.  A normal run performs
at most :data:`MAX_EODHD_HTTP_ATTEMPTS` one-shot requests and validates a full
candidate snapshot without writing it.  ``--apply`` is the only write mode.
The immutable Yahoo chart cache may supply only the explicitly bounded old
LILA/LILAK regular-way intervals that EODHD does not expose.  Those 630-session
segments are accepted only after a pinned external CC0 history cross-checks
597 sessions and the uncorroborated 33-session tail is reported explicitly.
Old Alcoa is the one additional exception: a commit-pinned Quandl WIKI/PRICES
snapshot owns the raw AA predecessor interval, while the already-cached EODHD
AA response is retained only as an OHLC cross-check because its volume basis is
not raw across the 2016 reverse split.  Every cache and apply stays fail-closed.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from supertrend_quant.env import load_env
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
from supertrend_quant.market_store.yahoo_chart import validate_yahoo_equity_metadata


DEFAULT_CACHE_ROOT = Path("data/cache")
FETCH_START = "2015-01-01"
FOX_TRANSITION = "2019-03-19"
FOX_OLD_LAST = "2019-03-19"
COR_CHANGE = "2023-08-30"
ARNC_SEPARATION = "2020-04-01"
VALARIS_PROVIDER_CODE = "VALPQ"
VALARIS_PRICE_START = "2019-07-31"
VALARIS_PRICE_END = "2021-04-27"
VALARIS_CANCELLATION_DATE = "2021-04-30"
VALARIS_DOCUMENTED_HALT_SESSIONS = frozenset(
    {"2020-08-17", "2020-08-18", "2020-08-19"}
)
LILA_REGULAR_PRICE_START = "2015-07-02"
LILA_REGULAR_PRICE_END = "2017-12-29"
LILA_EXTERNAL_CROSSCHECK_END = "2017-11-10"
AA_CROSSCHECK_START = "2015-01-02"
AA_CROSSCHECK_END = "2016-10-31"

WIKI_ARNC_SOURCE = "quandl_wiki_arnc_commit_snapshot"
WIKI_ARNC_ROLE = "old_aa_wiki_raw_primary"
WIKI_ARNC_COMMIT = "ce85e08888de5b8c4f6fd8c2d03bba85a9034f64"
WIKI_ARNC_URL = (
    "https://media.githubusercontent.com/media/kmfranz/trading_pairs/"
    f"{WIKI_ARNC_COMMIT}/WIKI_PRICES.csv"
)
WIKI_ARNC_FULL_SHA256 = (
    "dd5127aae478d270150904fcbad6e96a42e461e13c3d48a1587edb9b89cea43e"
)
WIKI_ARNC_FULL_SIZE = 235_562_224
WIKI_ARNC_FULL_DATA_ROWS = 2_166_605
WIKI_ARNC_FULL_START = "2014-01-02"
WIKI_ARNC_FULL_END = "2016-12-19"
WIKI_ARNC_SEGMENT_ROWS = 462
WIKI_ARNC_SEGMENT_SIZE = 67_674
WIKI_ARNC_SEGMENT_SHA256 = (
    "264339847c6a6e138a48280c0e5c4a1f8cf20565a2576e6028b43cb75d3a5fa5"
)
WIKI_ARNC_HEADER = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ex-dividend",
    "split_ratio",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
)
WIKI_ARNC_DIVIDEND_DATES = (
    "2015-02-04",
    "2015-05-07",
    "2015-08-05",
    "2015-11-04",
    "2016-02-03",
    "2016-05-04",
    "2016-08-03",
)
MAX_WIKI_ARNC_HTTP_ATTEMPTS = 1
WIKI_ARNC_UPSTREAM_LICENSE = "Public Domain"
WIKI_ARNC_UPSTREAM_LICENSE_URL = (
    "https://docs.data.nasdaq.com/v1.0/docs/in-depth-usage"
)
WIKI_ARNC_UPSTREAM_TABLE_URL = (
    "https://docs.data.nasdaq.com/v1.0/docs/in-depth-usage-1"
)
WIKI_ARNC_MIRROR_PROVENANCE_URL = (
    "https://datacrushblog.wordpress.com/2016/12/20/"
    "statistical-arbitrage-trading-pairs-in-python-using-correlation-"
    "cointegration-and-the-engle-granger-approach/"
)

# Twelve bounded price probes plus two action endpoints for ten selected roles.
# The run count remains frozen so an unexpected new request fails before apply.
PRICE_PROBE_CODES = (
    "WYN",
    "SE1",
    "TFCF",
    "TFCFA",
    "FOX1",
    "FOXA1",
    "LILAV",
    "LILKV",
    "AA",
    "TNL",
    VALARIS_PROVIDER_CODE,
    "BHI",
)
FIXED_ROLE_CODES = {
    "wyn": "WYN",
    "spectra": "SE1",
    "old_lila": "LILAV",
    "old_lilak": "LILKV",
    "old_aa": "AA",
    "tnl": "TNL",
    "valaris": VALARIS_PROVIDER_CODE,
    "bhi": "BHI",
}
FOX_ROLE_CANDIDATES = {
    "old_fox": ("TFCF", "FOX1"),
    "old_foxa": ("TFCFA", "FOXA1"),
}
SELECTED_ACTION_CODE_COUNT = 10
MAX_EODHD_HTTP_ATTEMPTS = len(PRICE_PROBE_CODES) + 2 * SELECTED_ACTION_CODE_COUNT

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

OFFICIAL_EVIDENCE: dict[str, dict[str, Any]] = {
    "agn": {
        "url": "https://press.spglobal.com/2015-03-16-American-Airlines-Group-Set-to-Join-the-S-P-500",
        "facts": {
            "legacy_allergan_last_price": "2015-03-16",
            "legacy_allergan_index_remove_after_close": "2015-03-20",
            "legacy_allergan_replay_effective": "2015-03-23",
            "actavis_was_separate_sp500_member": True,
            "act_to_agn": "2015-06-15",
        },
    },
    "agn_terms": {
        "url": "https://www.sec.gov/Archives/edgar/data/850693/000119312515096184/d894643d8k.htm",
        "facts": {"effective_date": "2015-03-17", "cash": 129.22, "ratio": 0.3683},
    },
    "agn_abbvie": {
        "url": "https://www.sec.gov/Archives/edgar/data/1551152/000110465920058837/tm2018740d2_8k.htm",
        "facts": {
            "effective_date": "2020-05-08",
            "last_trading_session": "2020-05-08",
            "cash": 120.30,
            "ratio": 0.8660,
        },
    },
    "cor": {
        "url": "https://www.sec.gov/Archives/edgar/data/1140859/000110465923096698/tm2324358d1_8k.htm",
        "facts": {"abc_to_cor": COR_CHANGE, "same_issuer": True},
    },
    "fox": {
        "url": "https://www.sec.gov/Archives/edgar/data/1754301/000119312519079678/d721949dex991.htm",
        "facts": {"new_fox_regular_trading": FOX_TRANSITION, "distribution_ratio": 1 / 3},
    },
    "fox_index": {
        "url": "https://press.spglobal.com/2019-03-14-Fox-Set-to-Join-S-P-500-Adobe-to-Join-S-P-100",
        "facts": {"new_fox_add": "2019-03-19", "old_21cf_remove": "2019-03-20"},
    },
    "fox_nasdaq": {
        "url": "https://www.nasdaqtrader.com/TraderNews.aspx?id=dtn2019-05",
        "facts": {
            "old_foxa_to_tfcfa": "2019-03-19",
            "old_fox_to_tfcf": "2019-03-19",
            "new_fox_symbol_reuse": "2019-03-19",
        },
    },
    "wynd": {
        "url": "https://www.sec.gov/Archives/edgar/data/1361658/000110465918032536/a18-13272_2ex99d1.htm",
        "facts": {"wyn_to_wynd": "2018-06-01"},
    },
    "se": {
        "url": "https://www.sec.gov/Archives/edgar/data/1373835/000119312517057856/d338638dex991.htm",
        "facts": {"spectra_trading_suspended": "2017-02-27"},
    },
    "lila": {
        "url": "https://www.sec.gov/Archives/edgar/data/1570585/000157058515000134/exhibit991-libertyglobalxl.htm",
        "facts": {"tracking_share_start": "2015-07-02"},
    },
    "lila_nasdaq": {
        "url": "https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2015-79",
        "facts": {
            "lilav_lilkv_when_issued": "2015-06-22",
            "lila_lilak_regular_way": "2015-07-02",
            "class_a_cusip": "G5480U138",
            "class_c_cusip": "G5480U153",
        },
    },
    "lila_splitoff": {
        "url": "https://www.sec.gov/Archives/edgar/data/1570585/000157058517000401/ex991split-offrecordanddis.htm",
        "facts": {"old_tracking_ceases": "2017-12-29", "new_company_trades": "2018-01-02"},
    },
    "bhge": {
        "url": "https://www.sec.gov/Archives/edgar/data/808362/000119312517220863/d343454d8k12b.htm",
        "facts": {"bhi_to_bhge": "2017-07-05", "cash": 17.50, "ratio": 1.0},
    },
    "bkr": {
        "url": "https://www.sec.gov/Archives/edgar/data/1701605/000170160520000019/fiscalyear2019form10-k.htm",
        "facts": {"bhge_to_bkr": "2019-10-18"},
    },
    "ir": {
        "url": "https://press.spglobal.com/2020-02-27-Gardner-Denver-Holdings-Set-to-Join-S-P-500-Cimarex-Energy-to-Join-S-P-MidCap-400",
        "facts": {"new_ir_sp500_add": "2020-03-03", "old_ir_continues_as_tt": True},
    },
    "ir_tickers": {
        "url": "https://www.sec.gov/Archives/edgar/data/1699150/000114036120004816/form8k.htm",
        "facts": {"gdi_to_ir": "2020-03-02"},
    },
    "arnc": {
        "url": "https://www.sec.gov/Archives/edgar/data/1790982/000179098221000096/arnc-20210630.htm",
        "facts": {"new_arnc_regular_trading": ARNC_SEPARATION, "distribution_ratio": 0.25},
    },
    "arnc_index": {
        "url": "https://press.spglobal.com/2020-04-01-Arconic-Set-to-Join-S-P-SmallCap-600",
        "facts": {"howmet_remains_sp500": True, "new_arnc_not_sp500": True},
    },
    "old_alcoa_arnc_identity": {
        "url": "https://www.sec.gov/Archives/edgar/data/4281/000000428119000031/form10k_4q18.htm",
        "facts": {
            "old_aa_to_arnc_same_issuer": "2016-11-01",
            "reverse_split_approved": "2016-10-05",
            "reverse_split_adjusted_trading": "2016-10-06",
            "reverse_split_ratio": 1 / 3,
        },
    },
    "alcoa_2016_separation": {
        "url": "https://www.sec.gov/Archives/edgar/data/4281/000119312516731663/d249430dex991.htm",
        "facts": {
            "old_alcoa_becomes_arnc": "2016-11-01",
            "new_alcoa_regular_way_aa": "2016-11-01",
            "new_aa_is_separate_security": True,
            "distribution_ratio_per_old_post_split_share": 1 / 3,
            "distribution_ratio_per_old_pre_split_share": 1 / 9,
        },
    },
    "new_alcoa_2016_10k": {
        "url": "https://www.sec.gov/Archives/edgar/data/1675149/000119312517083862/d298292d10k.htm",
        "facts": {
            "new_alcoa_became_independent": "2016-11-01",
            "new_alcoa_regular_way_aa": "2016-11-01",
            "parent_close_2016_10_31": 28.72,
            "new_alcoa_when_issued_close_2016_10_31": 21.44,
        },
    },
    "tnl": {
        "url": "https://www.sec.gov/Archives/edgar/data/1361658/000136165821000025/R9.htm",
        "facts": {"wynd_to_tnl": "2021-02-17"},
    },
    "hot": {
        "url": "https://www.sec.gov/Archives/edgar/data/316206/000119312516718027/d244396d8k.htm",
        "facts": {
            "starwood_marriott_completion": "2016-09-23",
            "cash": 21.0,
            "ratio": 0.8,
        },
    },
    "valaris": {
        "url": "https://www.sec.gov/Archives/edgar/data/314808/000031480819000134/form8k_item801namechange.htm",
        "facts": {"esv_to_val": "2019-07-31"},
    },
    "valaris_suspension": {
        "url": "https://www.sec.gov/Archives/edgar/data/314808/000031480820000147/exhibit991-valxpressre.htm",
        "facts": {
            "nyse_halt_start": "2020-08-17",
            "otc_valpq_trading": "2020-08-20",
            "documented_non_trading_sessions": sorted(
                VALARIS_DOCUMENTED_HALT_SESSIONS
            ),
        },
    },
    "valaris_emergence": {
        "url": "https://www.sec.gov/Archives/edgar/data/314808/000110465921058903/tm2114630d1_8k.htm",
        "facts": {
            "legacy_shares_cancelled": VALARIS_CANCELLATION_DATE,
            "new_issuer": True,
        },
    },
}


def _stable_id(label: str) -> str:
    return "US:EODHD:" + str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"supertrendquant:us-index-identity:{label}")
    )


OLD_FOX_ID = _stable_id("TWENTY_FIRST_CENTURY_FOX_CLASS_B")
OLD_FOXA_ID = _stable_id("TWENTY_FIRST_CENTURY_FOX_CLASS_A")
SPECTRA_ID = _stable_id("SPECTRA_ENERGY_SE")
OLD_LILA_ID = _stable_id("LIBERTY_GLOBAL_LILAC_CLASS_A")
OLD_LILAK_ID = _stable_id("LIBERTY_GLOBAL_LILAC_CLASS_C")
BHI_ID = _stable_id("BAKER_HUGHES_INCORPORATED_BHI")
COV_SECURITY_ID = "US:EODHD:e03a169c-f7e7-539c-9dde-a7da5a8e861c"
PENDING_IDENTITY_WARNING = "Pending audited US index identity repairs: 11 security_ids"
VALARIS_2021_OUTCOME_STATUS = (
    "unsupported_consideration_pending_official_evidence_and_lifecycle_finalizer"
)


@dataclass(frozen=True)
class CatalogArchive:
    kind: str
    rows: tuple[dict[str, Any], ...]
    artifact: SourceArtifact


@dataclass(frozen=True)
class IdentityIds:
    agn_legacy: str
    agn_actavis: str
    abc_duplicate: str
    cor: str
    coresite_duplicate: str
    fox: str
    foxa: str
    wynd: str
    sea: str
    lila: str
    lilak: str
    bhge: str
    bkr: str
    ir: str
    tt: str
    arnc: str
    arnc_duplicate: str
    hwm: str
    hot: str
    esv: str
    mar: str
    azn: str
    azn_duplicate: str


@dataclass(frozen=True)
class LocalPreflight:
    existing: dict[str, pd.DataFrame]
    ids: IdentityIds
    catalogs: dict[str, dict[str, Any]]
    pointer_etags: dict[str, str | None]


@dataclass(frozen=True)
class FetchedHistories:
    prices: pd.DataFrame
    crosscheck_prices: pd.DataFrame
    corporate_actions: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    role_codes: dict[str, str]
    http_attempts: int


@dataclass
class PreparedCollection:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    archive_artifacts: tuple[SourceArtifact, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class OfficialEvidenceBundle:
    manifest: SourceArtifact
    raw_artifacts: tuple[SourceArtifact, ...]

    @property
    def retrieved_at(self) -> str:
        return self.manifest.retrieved_at

    @property
    def source_hash(self) -> str:
        return self.manifest.source_hash

    def source_hash_for(self, source_url: str) -> str:
        matches = [item for item in self.raw_artifacts if item.source_url == source_url]
        if len(matches) != 1:
            raise ValueError(
                f"Official boundary evidence is not uniquely archived: {source_url}"
            )
        return matches[0].source_hash


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair audited US index ticker/issuer identity collisions."
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument(
        "--fetch-official-evidence",
        action="store_true",
        help="Explicitly allow one raw HTTP request for each missing official document.",
    )
    parser.add_argument(
        "--fetch-yahoo-supplement",
        action="store_true",
        help=(
            "Explicitly allow at most two one-shot Yahoo chart JSON requests "
            "for the old LILA/LILAK regular-way segments."
        ),
    )
    parser.add_argument(
        "--fetch-boris-crosscheck",
        action="store_true",
        help=(
            "Explicitly allow at most two one-shot downloads of the pinned "
            "Kaggle CC0 LILA/LILAK overlap files."
        ),
    )
    parser.add_argument(
        "--fetch-aa-wiki-crosscheck",
        action="store_true",
        help=(
            "Explicitly allow the one-shot commit-pinned WIKI/ARNC full-snapshot "
            "download used as the Old AA raw OHLCV primary."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--offline-plan", action="store_true")
    return parser.parse_args(argv)


class CappedSingleAttemptEodhdClient(EodhdClient):
    """One attempt per request and a hard run-wide cap."""

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
                raise RuntimeError(
                    "US identity-repair EODHD call cap reached before request: "
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
                f"EODHD identity-repair single attempt failed for {safe_endpoint}: {detail}"
            ) from None


def _artifact(
    client: CappedSingleAttemptEodhdClient,
    endpoint: str,
    code: str,
    rows: Any,
    *,
    start: str,
    end: str,
    retrieved_at: str,
) -> SourceArtifact:
    path = f"{endpoint}/{code}.US"
    params = {"from": start, "to": end}
    return SourceArtifact(
        source=f"eodhd_{endpoint}",
        source_url=client.safe_url(path, params=params),
        retrieved_at=retrieved_at,
        content=json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(),
        content_type="application/json",
    )


def _price_records(
    rows: Any,
    *,
    security_id: str,
    artifact: SourceArtifact,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [
        {
            "security_id": security_id,
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


def _event_id(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def _provider_action_records(
    endpoint: str,
    rows: Any,
    *,
    security_id: str,
    artifact: SourceArtifact,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    output: list[dict[str, Any]] = []
    for row in rows:
        effective = str(row.get("date") or "")
        if not effective:
            continue
        if endpoint == "div":
            action_type = "cash_dividend"
            cash = row.get("unadjustedValue", row.get("value"))
            ratio = None
        else:
            action_type = "split"
            cash = None
            ratio = _parse_split(row.get("split"))
            if ratio is None:
                continue
        output.append(
            {
                "event_id": _event_id(
                    artifact.source, security_id, action_type, effective, cash, ratio
                ),
                "security_id": security_id,
                "action_type": action_type,
                "effective_date": effective,
                "ex_date": effective,
                "announcement_date": str(row.get("declarationDate") or ""),
                "record_date": str(row.get("recordDate") or ""),
                "payment_date": str(row.get("paymentDate") or ""),
                "cash_amount": cash,
                "ratio": ratio,
                "currency": str(row.get("currency") or "USD"),
                "new_security_id": "",
                "new_symbol": "",
                "official": False,
                "source": artifact.source,
                "source_url": artifact.source_url,
                "source_kind": "provider",
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    return output


def _expected_sessions(start: str, end: str) -> tuple[str, ...]:
    import exchange_calendars as xcals

    values = xcals.get_calendar("XNYS").sessions_in_range(start, end)
    return tuple(pd.Timestamp(value).date().isoformat() for value in values)


def _complete_price_candidate(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> bool:
    actual = set(frame.get("session", pd.Series(dtype=str)).astype(str))
    return set(_expected_sessions(start, end)).issubset(actual)


class CappedIdentityHistorySource:
    """Probe ambiguous reused codes first, then fetch actions only for winners."""

    def __init__(self, client: CappedSingleAttemptEodhdClient | None = None):
        self.client = client or CappedSingleAttemptEodhdClient()

    def fetch(self, role_ids: dict[str, str], *, completed_session: str) -> FetchedHistories:
        retrieved_at = utc_now_iso()
        artifacts: list[SourceArtifact] = []
        prices_by_code: dict[str, pd.DataFrame] = {}
        target_for_probe = {
            "WYN": role_ids["wyn"],
            "SE1": role_ids["spectra"],
            "TFCF": role_ids["old_fox"],
            "FOX1": role_ids["old_fox"],
            "TFCFA": role_ids["old_foxa"],
            "FOXA1": role_ids["old_foxa"],
            "LILAV": role_ids["old_lila"],
            "LILKV": role_ids["old_lilak"],
            "AA": role_ids["old_aa"],
            "TNL": role_ids["tnl"],
            VALARIS_PROVIDER_CODE: role_ids["valaris"],
            "BHI": role_ids["bhi"],
        }
        ranges = {
            "WYN": (FETCH_START, "2018-05-31"),
            "SE1": (FETCH_START, "2017-02-27"),
            "TFCF": (FETCH_START, "2019-03-20"),
            "FOX1": (FETCH_START, "2019-03-20"),
            "TFCFA": (FETCH_START, "2019-03-20"),
            "FOXA1": (FETCH_START, "2019-03-20"),
            "LILAV": ("2015-06-22", "2017-12-29"),
            "LILKV": ("2015-06-22", "2017-12-29"),
            "AA": (FETCH_START, "2016-10-31"),
            "TNL": ("2021-02-17", completed_session),
            VALARIS_PROVIDER_CODE: (VALARIS_PRICE_START, VALARIS_PRICE_END),
            "BHI": (FETCH_START, "2017-07-03"),
        }
        raw_by_code: dict[str, list[dict[str, Any]]] = {}
        for code in PRICE_PROBE_CODES:
            start, end = ranges[code]
            rows = self.client.get_json(
                f"eod/{code}.US", params={"from": start, "to": end}
            )
            rows = rows if isinstance(rows, list) else []
            artifact = _artifact(
                self.client,
                "eod",
                code,
                rows,
                start=start,
                end=end,
                retrieved_at=retrieved_at,
            )
            artifacts.append(artifact)
            raw_by_code[code] = rows
            prices_by_code[code] = pd.DataFrame(
                _price_records(
                    rows,
                    security_id=target_for_probe[code],
                    artifact=artifact,
                )
            )

        role_codes = dict(FIXED_ROLE_CODES)
        for role, candidates in FOX_ROLE_CANDIDATES.items():
            qualified = [
                code
                for code in candidates
                if _complete_price_candidate(
                    prices_by_code[code], start="2015-01-02", end=FOX_OLD_LAST
                )
            ]
            # Always finish and cache the bounded run.  A non-full candidate is
            # still useful raw evidence, but the later full-history gate blocks
            # apply.  Official 2019 renamed codes are ordered before legacy FOX1
            # aliases when two endpoints expose the same complete history.
            if qualified:
                selected = qualified[0]
            else:
                selected = max(
                    candidates,
                    key=lambda code: (
                        len(prices_by_code[code]),
                        str(prices_by_code[code]["session"].max())
                        if not prices_by_code[code].empty
                        else "",
                        -candidates.index(code),
                    ),
                )
            role_codes[role] = selected

        role_target_ids = {
            "wyn": role_ids["wyn"],
            "spectra": role_ids["spectra"],
            "old_fox": role_ids["old_fox"],
            "old_foxa": role_ids["old_foxa"],
            "old_lila": role_ids["old_lila"],
            "old_lilak": role_ids["old_lilak"],
            "old_aa": role_ids["old_aa"],
            "tnl": role_ids["tnl"],
            "valaris": role_ids["valaris"],
            "bhi": role_ids["bhi"],
        }
        selected_prices: list[pd.DataFrame] = []
        actions: list[dict[str, Any]] = []
        for role, code in role_codes.items():
            selected_prices.append(prices_by_code[code])
            start, end = ranges[code]
            for endpoint in ("div", "splits"):
                rows = self.client.get_json(
                    f"{endpoint}/{code}.US", params={"from": start, "to": end}
                )
                rows = rows if isinstance(rows, list) else []
                artifact = _artifact(
                    self.client,
                    endpoint,
                    code,
                    rows,
                    start=start,
                    end=end,
                    retrieved_at=retrieved_at,
                )
                artifacts.append(artifact)
                actions.extend(
                    _provider_action_records(
                        endpoint,
                        rows,
                        security_id=role_target_ids[role],
                        artifact=artifact,
                    )
                )
        if self.client.attempt_count != MAX_EODHD_HTTP_ATTEMPTS:
            raise RuntimeError(
                "Identity source did not consume the frozen successful-run request count: "
                f"expected={MAX_EODHD_HTTP_ATTEMPTS}, actual={self.client.attempt_count}."
            )
        prices = pd.concat(selected_prices, ignore_index=True)
        return FetchedHistories(
            prices=prices,
            crosscheck_prices=pd.DataFrame(
                columns=dataset_spec("daily_price_raw").required_columns
            ),
            corporate_actions=pd.DataFrame(
                actions, columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=tuple(artifacts),
            role_codes=role_codes,
            http_attempts=self.client.attempt_count,
        )


YAHOO_SUPPLEMENT_SYMBOLS = ("LILA", "LILAK")
MAX_YAHOO_HTTP_ATTEMPTS = len(YAHOO_SUPPLEMENT_SYMBOLS)
YAHOO_CHART_SOURCE = "yahoo_chart_json"
YAHOO_LILA_PRIMARY_SOURCE = "yahoo_chart_adjusted_basis_primary"
REQUIRED_YAHOO_ROLES = (
    "old_lila_regular_way_yahoo_primary",
    "old_lilak_regular_way_yahoo_primary",
)
REQUIRED_BORIS_ROLES = (
    "old_lila_boris_cc0_external_crosscheck",
    "old_lilak_boris_cc0_external_crosscheck",
)
REQUIRED_SUPPLEMENT_ROLES = (
    *REQUIRED_YAHOO_ROLES,
    *REQUIRED_BORIS_ROLES,
    WIKI_ARNC_ROLE,
)
YAHOO_CHART_REQUESTS: dict[str, dict[str, Any]] = {
    "LILA": {
        "period1": 1434931200,
        "period2": 1514764800,
        "raw_start": "2015-07-02",
        "raw_end": "2017-12-29",
        "segment_start": "2015-07-02",
        "segment_end": "2017-12-29",
        "role": REQUIRED_YAHOO_ROLES[0],
    },
    "LILAK": {
        "period1": 1434931200,
        "period2": 1514764800,
        "raw_start": "2015-06-23",
        "raw_end": "2017-12-29",
        "segment_start": "2015-07-02",
        "segment_end": "2017-12-29",
        "role": REQUIRED_YAHOO_ROLES[1],
    },
}

BORIS_KAGGLE_SOURCE = "boris_kaggle_cc0_v3"
BORIS_KAGGLE_VERSION_URL = (
    "https://www.kaggle.com/datasets/borismarjanovic/"
    "price-volume-data-for-all-us-stocks-etfs/versions/3"
)
BORIS_KAGGLE_LICENSE = "CC0: Public Domain"
BORIS_CROSSCHECK_SESSION_COUNT = 597
BORIS_KAGGLE_FILES: dict[str, dict[str, Any]] = {
    "LILA": {
        "url": (
            "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/"
            "price-volume-data-for-all-us-stocks-etfs/Data%2FStocks%2Flila.us.txt"
            "?datasetVersionNumber=3"
        ),
        "sha256": "9885111c20ca809ce8791c429cd8eb66a62470b53ab71f7c2ac6a573d576f73c",
        "raw_rows": 599,
        "segment_start": LILA_REGULAR_PRICE_START,
        "segment_end": LILA_EXTERNAL_CROSSCHECK_END,
        "segment_rows": BORIS_CROSSCHECK_SESSION_COUNT,
    },
    "LILAK": {
        "url": (
            "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/"
            "price-volume-data-for-all-us-stocks-etfs/Data%2FStocks%2Flilak.us.txt"
            "?datasetVersionNumber=3"
        ),
        "sha256": "b5a56cc0c1b5a478354d85149c2370ccde6146f7f43d94566dcc76382db610e4",
        "raw_rows": 599,
        "segment_start": LILA_REGULAR_PRICE_START,
        "segment_end": LILA_EXTERNAL_CROSSCHECK_END,
        "segment_rows": BORIS_CROSSCHECK_SESSION_COUNT,
    },
}
MAX_BORIS_HTTP_ATTEMPTS = len(BORIS_KAGGLE_FILES)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _yahoo_chart_url(symbol: str) -> str:
    normalized = str(symbol).strip().upper()
    try:
        request = YAHOO_CHART_REQUESTS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Yahoo identity supplement symbol: {symbol!r}") from exc
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{normalized}"
        f"?period1={request['period1']}&period2={request['period2']}"
        "&interval=1d&events=history"
    )


@dataclass(frozen=True)
class YahooChartCachedResponse:
    symbol: str
    source_url: str
    retrieved_at: str
    content: bytes
    content_type: str
    http_status: int

    @property
    def source_hash(self) -> str:
        return sha256_bytes(self.content)


class YahooChartSupplementCache:
    """Two-attempt, no-retry Yahoo chart cache preserving exact response bytes."""

    SCHEMA = "yahoo_chart_raw_response/v1"

    def __init__(
        self,
        root: Path,
        *,
        max_http_attempts: int = MAX_YAHOO_HTTP_ATTEMPTS,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 50 * 1024 * 1024,
    ):
        if not 0 < int(max_http_attempts) <= MAX_YAHOO_HTTP_ATTEMPTS:
            raise ValueError("Yahoo HTTP attempt cap must be one or two.")
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("Yahoo timeout/response cap must be positive.")
        self.root = Path(root)
        self.max_http_attempts = int(max_http_attempts)
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.http_attempts = 0

    def url(self, symbol: str) -> str:
        return _yahoo_chart_url(symbol)

    def path(self, symbol: str) -> Path:
        return self.root / f"{sha256_bytes(self.url(symbol).encode())}.json.gz"

    def _decode(self, symbol: str, encoded: bytes) -> YahooChartCachedResponse:
        normalized = str(symbol).strip().upper()
        try:
            envelope = json.loads(gzip.decompress(encoded))
            payload = envelope["payload"]
            payload_sha256 = str(envelope["payload_sha256"])
        except Exception as exc:
            raise RuntimeError(
                f"Invalid Yahoo chart cache envelope: {self.path(symbol)}"
            ) from exc
        if envelope.get("schema") != self.SCHEMA or not isinstance(payload, dict):
            raise RuntimeError("Wrong Yahoo chart cache schema.")
        if sha256_bytes(_canonical_json_bytes(payload)) != payload_sha256:
            raise RuntimeError("Yahoo chart cache payload hash mismatch.")
        try:
            content = base64.b64decode(str(payload["content_base64"]), validate=True)
        except Exception as exc:
            raise RuntimeError("Yahoo chart cache content encoding is invalid.") from exc
        if payload.get("content_sha256") != sha256_bytes(content):
            raise RuntimeError("Yahoo chart cache content hash mismatch.")
        if payload.get("symbol") != normalized:
            raise RuntimeError("Yahoo chart cache symbol mismatch.")
        if payload.get("source_url") != self.url(normalized):
            raise RuntimeError("Yahoo chart cache URL mismatch.")
        return YahooChartCachedResponse(
            symbol=normalized,
            source_url=self.url(normalized),
            retrieved_at=str(payload["retrieved_at"]),
            content=content,
            content_type=str(payload.get("content_type") or ""),
            http_status=int(payload["http_status"]),
        )

    def get(self, symbol: str) -> YahooChartCachedResponse | None:
        path = self.path(symbol)
        return self._decode(symbol, path.read_bytes()) if path.is_file() else None

    def fetch(self, symbol: str) -> YahooChartCachedResponse:
        normalized = str(symbol).strip().upper()
        url = self.url(normalized)
        if self.http_attempts >= self.max_http_attempts:
            raise RuntimeError("Yahoo chart HTTP attempt cap reached.")
        self.http_attempts += 1
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "SuperTrendQuant identity-repair/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                content_type = str(response.headers.get("Content-Type") or "")
                content = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            status = int(exc.code)
            content_type = str(exc.headers.get("Content-Type") or "")
            content = exc.read(self.max_response_bytes + 1)
        except URLError as exc:
            raise RuntimeError(
                f"Yahoo chart single HTTP attempt failed for {normalized}: {exc.reason}"
            ) from None
        if len(content) > self.max_response_bytes:
            raise RuntimeError("Yahoo chart response exceeds configured byte cap.")
        value = YahooChartCachedResponse(
            symbol=normalized,
            source_url=url,
            retrieved_at=utc_now_iso(),
            content=content,
            content_type=content_type,
            http_status=status,
        )
        payload = {
            "symbol": value.symbol,
            "source_url": value.source_url,
            "retrieved_at": value.retrieved_at,
            "http_status": value.http_status,
            "content_type": value.content_type,
            "content_sha256": value.source_hash,
            "content_base64": base64.b64encode(value.content).decode("ascii"),
        }
        envelope = {
            "schema": self.SCHEMA,
            "payload": payload,
            "payload_sha256": sha256_bytes(_canonical_json_bytes(payload)),
        }
        destination = self.path(normalized)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            existing = self._decode(normalized, destination.read_bytes())
            if (
                existing.content,
                existing.content_type,
                existing.http_status,
            ) != (value.content, value.content_type, value.http_status):
                raise RuntimeError("Yahoo chart cache changed for one request URL.")
            return existing
        write_atomic(
            destination,
            gzip.compress(_canonical_json_bytes(envelope), mtime=0),
        )
        return self._decode(normalized, destination.read_bytes())

    def fill_missing(
        self, symbols: Iterable[str]
    ) -> dict[str, YahooChartCachedResponse]:
        ordered = tuple(
            dict.fromkeys(str(item).strip().upper() for item in symbols)
        )
        for symbol in ordered:
            self.url(symbol)
        missing = [symbol for symbol in ordered if self.get(symbol) is None]
        remaining = self.max_http_attempts - self.http_attempts
        if len(missing) > remaining:
            raise RuntimeError(
                "Yahoo chart request set exceeds the two-attempt run cap before "
                f"any new HTTP call: {len(missing)} > {remaining}."
            )
        return {symbol: self.get(symbol) or self.fetch(symbol) for symbol in ordered}


def _parse_yahoo_chart_response(
    response: YahooChartCachedResponse,
) -> pd.DataFrame:
    if response.http_status != 200:
        raise ValueError(
            f"Yahoo chart returned HTTP {response.http_status} for {response.symbol}."
        )
    content_type = response.content_type.lower().split(";", 1)[0].strip()
    stripped = response.content.lstrip()
    if content_type != "application/json" or stripped.startswith((b"<", b"<!")):
        raise ValueError(
            f"Yahoo chart response is HTML or non-JSON for {response.symbol}."
        )
    try:
        payload = json.loads(response.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Yahoo chart response is not valid JSON for {response.symbol}."
        ) from exc
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        raise ValueError(f"Yahoo chart payload is malformed for {response.symbol}.")
    if chart.get("error") is not None:
        raise ValueError(f"Yahoo chart API error for {response.symbol}.")
    results = chart.get("result")
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
    ):
        raise ValueError(f"Yahoo chart result is malformed for {response.symbol}.")
    result = results[0]
    try:
        validate_yahoo_equity_metadata(result.get("meta"), response.symbol)
    except ValueError as exc:
        raise ValueError(f"Yahoo chart metadata is invalid for {response.symbol}: {exc}") from exc
    timestamps = result.get("timestamp")
    if (
        not isinstance(timestamps, list)
        or not timestamps
        or any(type(value) is not int or value <= 0 for value in timestamps)
        or timestamps != sorted(timestamps)
        or len(set(timestamps)) != len(timestamps)
    ):
        raise ValueError(f"Yahoo chart timestamps are invalid for {response.symbol}.")
    converted = pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce")
    if bool(pd.isna(converted).any()):
        raise ValueError(f"Yahoo chart timestamps are invalid for {response.symbol}.")
    sessions = tuple(value.date().isoformat() for value in converted)
    if len(set(sessions)) != len(sessions):
        raise ValueError(
            f"Yahoo chart timestamps do not map one-to-one to sessions for {response.symbol}."
        )
    indicators = result.get("indicators")
    quotes = indicators.get("quote") if isinstance(indicators, dict) else None
    if (
        not isinstance(quotes, list)
        or len(quotes) != 1
        or not isinstance(quotes[0], dict)
    ):
        raise ValueError(f"Yahoo chart quote block is malformed for {response.symbol}.")
    quote = quotes[0]
    values: dict[str, list[float]] = {}
    for column in ("open", "high", "low", "close", "volume"):
        raw_values = quote.get(column)
        if not isinstance(raw_values, list) or len(raw_values) != len(timestamps):
            raise ValueError(
                f"Yahoo chart {column} is not one-to-one with timestamps for "
                f"{response.symbol}."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_values
        ):
            raise ValueError(f"Yahoo chart contains invalid OHLCV for {response.symbol}.")
        values[column] = [float(value) for value in raw_values]
    frame = pd.DataFrame({"session": sessions, **values})
    positive = frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    coherent = (
        frame["volume"].ge(0)
        & frame["high"].ge(frame[["open", "low", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "high", "close"]].min(axis=1))
    )
    if not bool((positive & coherent).all()):
        raise ValueError(f"Yahoo chart contains invalid OHLCV for {response.symbol}.")
    return frame


def _yahoo_segment(
    response: YahooChartCachedResponse,
    *,
    security_id: str,
    source: str = YAHOO_CHART_SOURCE,
) -> pd.DataFrame:
    spec = YAHOO_CHART_REQUESTS[response.symbol]
    frame = _parse_yahoo_chart_response(response)
    expected_raw = _expected_sessions(spec["raw_start"], spec["raw_end"])
    actual_raw = tuple(frame["session"].astype(str))
    if actual_raw != expected_raw:
        missing = len(set(expected_raw) - set(actual_raw))
        extra = len(set(actual_raw) - set(expected_raw))
        raise ValueError(
            f"Yahoo chart exact exchange-session coverage failed for {response.symbol}: "
            f"expected={len(expected_raw)}, actual={len(actual_raw)}, "
            f"missing={missing}, extra={extra}."
        )
    frame = frame.loc[
        frame["session"].ge(spec["segment_start"])
        & frame["session"].le(spec["segment_end"])
    ].copy()
    expected_segment = _expected_sessions(spec["segment_start"], spec["segment_end"])
    if tuple(frame["session"].astype(str)) != expected_segment:
        raise ValueError(
            f"Yahoo chart regular-way slice is not one-to-one for {response.symbol}."
        )
    frame["security_id"] = security_id
    frame["currency"] = "USD"
    frame["source"] = source
    frame["source_url"] = response.source_url
    frame["retrieved_at"] = response.retrieved_at
    frame["source_hash"] = response.source_hash
    columns = list(dataset_spec("daily_price_raw").required_columns)
    if "source_url" not in columns:
        columns.append("source_url")
    return frame.loc[:, columns]


def _yahoo_artifact(
    response: YahooChartCachedResponse,
    *,
    source: str = YAHOO_CHART_SOURCE,
) -> SourceArtifact:
    return SourceArtifact(
        source=source,
        source_url=response.source_url,
        retrieved_at=response.retrieved_at,
        content=response.content,
        content_type=response.content_type,
    )


class YahooIdentitySupplementSource:
    """Explicit Yahoo supplement backed by two immutable raw JSON responses."""

    def __init__(self, root: Path, *, allow_http: bool):
        self.allow_http = bool(allow_http)
        self.cache = YahooChartSupplementCache(
            root,
            max_http_attempts=MAX_YAHOO_HTTP_ATTEMPTS,
        )

    @property
    def http_attempts(self) -> int:
        return int(self.cache.http_attempts)

    def fetch(self, ids: IdentityIds) -> FetchedHistories:
        responses = {
            symbol: self.cache.get(symbol) for symbol in YAHOO_SUPPLEMENT_SYMBOLS
        }
        missing = [symbol for symbol, value in responses.items() if value is None]
        if missing and not self.allow_http:
            raise FileNotFoundError(
                "Yahoo identity supplement cache is missing and network access was not "
                "explicitly allowed: "
                + ", ".join(missing)
                + ". Re-run with --fetch-yahoo-supplement."
            )
        if missing:
            self.cache.fill_missing(YAHOO_SUPPLEMENT_SYMBOLS)
            responses = {
                symbol: self.cache.get(symbol) for symbol in YAHOO_SUPPLEMENT_SYMBOLS
            }
        if any(value is None for value in responses.values()):
            raise RuntimeError("Yahoo identity supplement cache did not fill completely.")
        if self.http_attempts > MAX_YAHOO_HTTP_ATTEMPTS:
            raise RuntimeError("Yahoo identity supplement exceeded its two-call cap.")
        typed_responses = {
            symbol: value for symbol, value in responses.items() if value is not None
        }
        del ids
        lila = _yahoo_segment(
            typed_responses["LILA"],
            security_id=OLD_LILA_ID,
            source=YAHOO_LILA_PRIMARY_SOURCE,
        )
        lilak = _yahoo_segment(
            typed_responses["LILAK"],
            security_id=OLD_LILAK_ID,
            source=YAHOO_LILA_PRIMARY_SOURCE,
        )
        return FetchedHistories(
            prices=_concat_unique(
                (lila, lilak),
                keys=dataset_spec("daily_price_raw").primary_key,
            ),
            crosscheck_prices=pd.DataFrame(
                columns=dataset_spec("daily_price_raw").required_columns
            ),
            corporate_actions=pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=tuple(
                _yahoo_artifact(
                    typed_responses[symbol],
                    source=YAHOO_LILA_PRIMARY_SOURCE,
                )
                for symbol in YAHOO_SUPPLEMENT_SYMBOLS
            ),
            role_codes={
                spec["role"]: f"YAHOO_CHART:{symbol}"
                for symbol, spec in YAHOO_CHART_REQUESTS.items()
            },
            http_attempts=0,
        )


@dataclass(frozen=True)
class WikiArncCachedResponse:
    source_url: str
    retrieved_at: str
    content: bytes
    content_type: str
    http_status: int

    @property
    def source_hash(self) -> str:
        return sha256_bytes(self.content)


class WikiArncPinnedCache:
    """One-attempt immutable cache for the full commit-pinned WIKI snapshot."""

    SCHEMA = "wiki_arnc_commit_raw_response/v1"

    def __init__(
        self,
        root: Path,
        *,
        max_http_attempts: int = MAX_WIKI_ARNC_HTTP_ATTEMPTS,
        timeout_seconds: float = 120.0,
    ):
        if int(max_http_attempts) != MAX_WIKI_ARNC_HTTP_ATTEMPTS:
            raise ValueError("WIKI/ARNC HTTP attempt cap must be exactly one.")
        if timeout_seconds <= 0:
            raise ValueError("WIKI/ARNC timeout must be positive.")
        self.root = Path(root)
        self.max_http_attempts = int(max_http_attempts)
        self.timeout_seconds = float(timeout_seconds)
        self.http_attempts = 0

    @property
    def path(self) -> Path:
        return self.root / f"{sha256_bytes(WIKI_ARNC_URL.encode())}.csv.gz"

    @property
    def metadata_path(self) -> Path:
        return self.root / f"{sha256_bytes(WIKI_ARNC_URL.encode())}.json"

    @staticmethod
    def _validate_content(content: bytes) -> None:
        if len(content) != WIKI_ARNC_FULL_SIZE:
            raise RuntimeError(
                "WIKI/ARNC full snapshot byte size changed: "
                f"expected={WIKI_ARNC_FULL_SIZE}, actual={len(content)}."
            )
        if sha256_bytes(content) != WIKI_ARNC_FULL_SHA256:
            raise RuntimeError("WIKI/ARNC full snapshot hash changed.")

    def _decode(self) -> WikiArncCachedResponse:
        if not self.path.is_file() or not self.metadata_path.is_file():
            raise RuntimeError("WIKI/ARNC cache is partial; both blob and metadata are required.")
        try:
            metadata = json.loads(self.metadata_path.read_bytes())
            content = gzip.decompress(self.path.read_bytes())
        except Exception as exc:
            raise RuntimeError("WIKI/ARNC cache is unreadable.") from exc
        if metadata.get("schema") != self.SCHEMA:
            raise RuntimeError("WIKI/ARNC cache schema mismatch.")
        if metadata.get("source_url") != WIKI_ARNC_URL:
            raise RuntimeError("WIKI/ARNC cache URL mismatch.")
        self._validate_content(content)
        if (
            metadata.get("content_sha256") != WIKI_ARNC_FULL_SHA256
            or int(metadata.get("content_size", -1)) != WIKI_ARNC_FULL_SIZE
        ):
            raise RuntimeError("WIKI/ARNC cache metadata does not match the pinned blob.")
        return WikiArncCachedResponse(
            source_url=WIKI_ARNC_URL,
            retrieved_at=str(metadata["retrieved_at"]),
            content=content,
            content_type="text/csv",
            http_status=int(metadata["http_status"]),
        )

    def get(self) -> WikiArncCachedResponse | None:
        exists = self.path.is_file()
        metadata_exists = self.metadata_path.is_file()
        if exists != metadata_exists:
            raise RuntimeError("WIKI/ARNC cache is partial and cannot be refetched implicitly.")
        return self._decode() if exists else None

    def fetch(self) -> WikiArncCachedResponse:
        if self.http_attempts >= self.max_http_attempts:
            raise RuntimeError("WIKI/ARNC HTTP attempt cap reached.")
        self.http_attempts += 1
        request = Request(
            WIKI_ARNC_URL,
            headers={
                "Accept": "text/csv,application/octet-stream",
                "User-Agent": "SuperTrendQuant identity-repair/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                response_content_type = str(
                    response.headers.get("Content-Type") or "application/octet-stream"
                )
                content = response.read(WIKI_ARNC_FULL_SIZE + 1)
        except HTTPError as exc:
            raise RuntimeError(
                f"WIKI/ARNC single HTTP attempt failed: HTTP {exc.code}"
            ) from None
        except URLError as exc:
            raise RuntimeError(
                f"WIKI/ARNC single HTTP attempt failed: {exc.reason}"
            ) from None
        if status != 200:
            raise RuntimeError(f"WIKI/ARNC returned HTTP {status}.")
        self._validate_content(content)
        retrieved_at = utc_now_iso()
        value = WikiArncCachedResponse(
            source_url=WIKI_ARNC_URL,
            retrieved_at=retrieved_at,
            content=content,
            content_type="text/csv",
            http_status=status,
        )
        existing = self.get()
        if existing is not None:
            if existing.content != value.content:
                raise RuntimeError("WIKI/ARNC immutable cache content changed.")
            return existing
        metadata = {
            "schema": self.SCHEMA,
            "source_url": WIKI_ARNC_URL,
            "commit": WIKI_ARNC_COMMIT,
            "retrieved_at": retrieved_at,
            "http_status": status,
            "http_content_type": response_content_type,
            "content_sha256": WIKI_ARNC_FULL_SHA256,
            "content_size": WIKI_ARNC_FULL_SIZE,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(self.path, gzip.compress(content, mtime=0))
        write_atomic(self.metadata_path, _canonical_json_bytes(metadata))
        return self._decode()


def _wiki_arnc_segment_bytes(content: bytes) -> tuple[bytes, dict[str, Any]]:
    """Scan the complete immutable CSV and retain the exact raw ARNC byte slice."""

    stream = io.BytesIO(content)
    expected_header = (",".join(WIKI_ARNC_HEADER) + "\n").encode("ascii")
    header = stream.readline()
    if header != expected_header:
        raise ValueError("WIKI/ARNC full snapshot header changed.")
    data_rows = 0
    minimum = ""
    maximum = ""
    segment_lines: list[bytes] = []
    for line in stream:
        data_rows += 1
        if not line.endswith(b"\n"):
            raise ValueError("WIKI/ARNC full snapshot has a non-terminated CSV row.")
        values = line[:-1].removesuffix(b"\r").split(b",")
        if len(values) != len(WIKI_ARNC_HEADER):
            raise ValueError("WIKI/ARNC full snapshot contains a malformed CSV row.")
        try:
            ticker = values[0].decode("ascii")
            session = values[1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("WIKI/ARNC ticker/date is not ASCII.") from exc
        if (
            len(session) != 10
            or session[4] != "-"
            or session[7] != "-"
            or not session.replace("-", "").isdigit()
        ):
            raise ValueError("WIKI/ARNC full snapshot contains an invalid date.")
        minimum = session if not minimum or session < minimum else minimum
        maximum = session if not maximum or session > maximum else maximum
        if ticker == "ARNC" and AA_CROSSCHECK_START <= session <= AA_CROSSCHECK_END:
            segment_lines.append(line)
    if data_rows != WIKI_ARNC_FULL_DATA_ROWS:
        raise ValueError(
            "WIKI/ARNC full snapshot row count changed: "
            f"expected={WIKI_ARNC_FULL_DATA_ROWS}, actual={data_rows}."
        )
    if (minimum, maximum) != (WIKI_ARNC_FULL_START, WIKI_ARNC_FULL_END):
        raise ValueError(
            "WIKI/ARNC full snapshot date bounds changed: "
            f"actual={minimum}..{maximum}."
        )
    segment = header + b"".join(segment_lines)
    if len(segment_lines) != WIKI_ARNC_SEGMENT_ROWS:
        raise ValueError("WIKI/ARNC exact Old AA segment row count changed.")
    if len(segment) != WIKI_ARNC_SEGMENT_SIZE:
        raise ValueError("WIKI/ARNC exact Old AA segment byte size changed.")
    if sha256_bytes(segment) != WIKI_ARNC_SEGMENT_SHA256:
        raise ValueError("WIKI/ARNC deterministic Old AA segment hash changed.")
    return segment, {
        "full_data_rows": data_rows,
        "full_start": minimum,
        "full_end": maximum,
        "segment_rows": len(segment_lines),
        "segment_size": len(segment),
        "segment_sha256": WIKI_ARNC_SEGMENT_SHA256,
    }


def _parse_wiki_arnc_response(
    response: WikiArncCachedResponse,
    *,
    security_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if response.http_status != 200:
        raise ValueError(f"WIKI/ARNC returned HTTP {response.http_status}.")
    if response.source_url != WIKI_ARNC_URL:
        raise ValueError("WIKI/ARNC response URL is not commit-pinned.")
    if (
        len(response.content) != WIKI_ARNC_FULL_SIZE
        or response.source_hash != WIKI_ARNC_FULL_SHA256
    ):
        raise ValueError("WIKI/ARNC response does not match the pinned full blob.")
    segment, scan = _wiki_arnc_segment_bytes(response.content)
    try:
        raw = pd.read_csv(io.BytesIO(segment), dtype={"ticker": str, "date": str})
    except Exception as exc:
        raise ValueError("WIKI/ARNC deterministic segment is unreadable.") from exc
    if list(raw.columns) != list(WIKI_ARNC_HEADER):
        raise ValueError("WIKI/ARNC segment schema changed.")
    expected_sessions = _expected_sessions(AA_CROSSCHECK_START, AA_CROSSCHECK_END)
    if (
        len(expected_sessions) != WIKI_ARNC_SEGMENT_ROWS
        or tuple(raw["date"].astype(str)) != expected_sessions
        or set(raw["ticker"].astype(str)) != {"ARNC"}
    ):
        raise ValueError("WIKI/ARNC segment is not exact one-to-one XNYS history.")
    numeric_columns = list(WIKI_ARNC_HEADER[2:])
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    numeric = raw[numeric_columns]
    if not bool(numeric.apply(lambda values: values.map(math.isfinite)).all().all()):
        raise ValueError("WIKI/ARNC segment contains non-finite numeric values.")
    coherent = (
        raw[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & raw["volume"].ge(0)
        & raw["high"].ge(raw[["open", "low", "close"]].max(axis=1))
        & raw["low"].le(raw[["open", "high", "close"]].min(axis=1))
    )
    if not bool(coherent.all()):
        raise ValueError("WIKI/ARNC segment contains incoherent raw OHLCV.")
    split = raw.set_index("date")["split_ratio"]
    split_date = "2016-10-06"
    if not math.isclose(float(split.loc[split_date]), 1 / 3, abs_tol=1e-15):
        raise ValueError("WIKI/ARNC reverse-split marker changed.")
    if not bool(split.drop(index=split_date).eq(1.0).all()):
        raise ValueError("WIKI/ARNC contains an unexpected split marker.")
    dividends = raw.loc[raw["ex-dividend"].ne(0), ["date", "ex-dividend"]]
    if (
        tuple(dividends["date"].astype(str)) != WIKI_ARNC_DIVIDEND_DATES
        or not bool(dividends["ex-dividend"].eq(0.03).all())
    ):
        raise ValueError("WIKI/ARNC dividend markers changed.")
    frame = raw.rename(columns={"date": "session"}).copy()
    frame["security_id"] = security_id
    frame["currency"] = "USD"
    frame["source"] = WIKI_ARNC_SOURCE
    frame["source_url"] = response.source_url
    frame["retrieved_at"] = response.retrieved_at
    frame["source_hash"] = response.source_hash
    columns = list(dataset_spec("daily_price_raw").required_columns)
    if "source_url" not in columns:
        columns.append("source_url")
    return frame.loc[:, columns], {
        **scan,
        "source": WIKI_ARNC_SOURCE,
        "source_url": response.source_url,
        "source_sha256": response.source_hash,
        "full_size": len(response.content),
        "split_session": split_date,
        "split_ratio": float(split.loc[split_date]),
        "dividend_sessions": list(WIKI_ARNC_DIVIDEND_DATES),
    }


def _wiki_arnc_artifact(response: WikiArncCachedResponse) -> SourceArtifact:
    return SourceArtifact(
        source=WIKI_ARNC_SOURCE,
        source_url=response.source_url,
        retrieved_at=response.retrieved_at,
        content=response.content,
        content_type="text/csv",
    )


class WikiArncPinnedSource:
    """Full WIKI blob owner for Old AA raw OHLCV; HTTP is explicit opt-in."""

    def __init__(self, root: Path, *, allow_http: bool):
        self.allow_http = bool(allow_http)
        self.cache = WikiArncPinnedCache(root)

    @property
    def http_attempts(self) -> int:
        return int(self.cache.http_attempts)

    def fetch(self, ids: IdentityIds) -> FetchedHistories:
        response = self.cache.get()
        if response is None and not self.allow_http:
            raise FileNotFoundError(
                "WIKI/ARNC pinned full cache is missing and network access was not "
                "explicitly allowed. Re-run with --fetch-aa-wiki-crosscheck."
            )
        if response is None:
            response = self.cache.fetch()
        frame, _validation = _parse_wiki_arnc_response(
            response, security_id=ids.hwm
        )
        return FetchedHistories(
            prices=frame,
            crosscheck_prices=pd.DataFrame(
                columns=dataset_spec("daily_price_raw").required_columns
            ),
            corporate_actions=pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=(_wiki_arnc_artifact(response),),
            role_codes={
                WIKI_ARNC_ROLE: f"WIKI/PRICES:ARNC@{WIKI_ARNC_COMMIT}"
            },
            http_attempts=0,
        )


@dataclass(frozen=True)
class BorisKaggleCachedResponse:
    symbol: str
    source_url: str
    retrieved_at: str
    content: bytes
    content_type: str
    http_status: int

    @property
    def source_hash(self) -> str:
        return sha256_bytes(self.content)


class BorisKaggleCache:
    """Two-attempt pinned cache for the version-3 CC0 text files."""

    SCHEMA = "boris_kaggle_cc0_raw_response/v1"

    def __init__(
        self,
        root: Path,
        *,
        max_http_attempts: int = MAX_BORIS_HTTP_ATTEMPTS,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 5 * 1024 * 1024,
    ):
        if not 0 < int(max_http_attempts) <= MAX_BORIS_HTTP_ATTEMPTS:
            raise ValueError("Boris/Kaggle HTTP attempt cap must be one or two.")
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("Boris/Kaggle timeout/response cap must be positive.")
        self.root = Path(root)
        self.max_http_attempts = int(max_http_attempts)
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.http_attempts = 0

    @staticmethod
    def _normalized_symbol(symbol: str) -> str:
        normalized = str(symbol).strip().upper()
        if normalized not in BORIS_KAGGLE_FILES:
            raise ValueError(f"Unsupported Boris/Kaggle symbol: {symbol!r}")
        return normalized

    def url(self, symbol: str) -> str:
        return BORIS_KAGGLE_FILES[self._normalized_symbol(symbol)]["url"]

    def path(self, symbol: str) -> Path:
        return self.root / f"{sha256_bytes(self.url(symbol).encode())}.json.gz"

    def _decode(self, symbol: str, encoded: bytes) -> BorisKaggleCachedResponse:
        normalized = self._normalized_symbol(symbol)
        try:
            envelope = json.loads(gzip.decompress(encoded))
            payload = envelope["payload"]
            payload_sha256 = str(envelope["payload_sha256"])
        except Exception as exc:
            raise RuntimeError(
                f"Invalid Boris/Kaggle cache envelope: {self.path(symbol)}"
            ) from exc
        if envelope.get("schema") != self.SCHEMA or not isinstance(payload, dict):
            raise RuntimeError("Wrong Boris/Kaggle cache schema.")
        if sha256_bytes(_canonical_json_bytes(payload)) != payload_sha256:
            raise RuntimeError("Boris/Kaggle cache payload hash mismatch.")
        try:
            content = base64.b64decode(str(payload["content_base64"]), validate=True)
        except Exception as exc:
            raise RuntimeError("Boris/Kaggle cache content encoding is invalid.") from exc
        expected_hash = BORIS_KAGGLE_FILES[normalized]["sha256"]
        if payload.get("content_sha256") != sha256_bytes(content):
            raise RuntimeError("Boris/Kaggle cache content hash mismatch.")
        if sha256_bytes(content) != expected_hash:
            raise RuntimeError("Boris/Kaggle content no longer matches its pinned hash.")
        if payload.get("symbol") != normalized:
            raise RuntimeError("Boris/Kaggle cache symbol mismatch.")
        if payload.get("source_url") != self.url(normalized):
            raise RuntimeError("Boris/Kaggle cache URL mismatch.")
        return BorisKaggleCachedResponse(
            symbol=normalized,
            source_url=self.url(normalized),
            retrieved_at=str(payload["retrieved_at"]),
            content=content,
            content_type=str(payload.get("content_type") or ""),
            http_status=int(payload["http_status"]),
        )

    def get(self, symbol: str) -> BorisKaggleCachedResponse | None:
        path = self.path(symbol)
        return self._decode(symbol, path.read_bytes()) if path.is_file() else None

    def fetch(self, symbol: str) -> BorisKaggleCachedResponse:
        normalized = self._normalized_symbol(symbol)
        if self.http_attempts >= self.max_http_attempts:
            raise RuntimeError("Boris/Kaggle HTTP attempt cap reached.")
        self.http_attempts += 1
        request = Request(
            self.url(normalized),
            headers={
                "Accept": "text/csv,text/plain,application/octet-stream",
                "User-Agent": "SuperTrendQuant identity-repair/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                content_type = str(response.headers.get("Content-Type") or "")
                content = response.read(self.max_response_bytes + 1)
        except (HTTPError, URLError) as exc:
            detail = getattr(exc, "code", None) or getattr(exc, "reason", None)
            raise RuntimeError(
                f"Boris/Kaggle single HTTP attempt failed for {normalized}: {detail}"
            ) from None
        if status != 200:
            raise RuntimeError(
                f"Boris/Kaggle returned HTTP {status} for {normalized}."
            )
        if len(content) > self.max_response_bytes:
            raise RuntimeError("Boris/Kaggle response exceeds configured byte cap.")
        expected_hash = BORIS_KAGGLE_FILES[normalized]["sha256"]
        if sha256_bytes(content) != expected_hash:
            raise RuntimeError(
                f"Boris/Kaggle raw hash changed for {normalized}; cache not written."
            )
        value = BorisKaggleCachedResponse(
            symbol=normalized,
            source_url=self.url(normalized),
            retrieved_at=utc_now_iso(),
            content=content,
            content_type=content_type,
            http_status=status,
        )
        payload = {
            "symbol": value.symbol,
            "source_url": value.source_url,
            "retrieved_at": value.retrieved_at,
            "http_status": value.http_status,
            "content_type": value.content_type,
            "content_sha256": value.source_hash,
            "content_base64": base64.b64encode(value.content).decode("ascii"),
        }
        envelope = {
            "schema": self.SCHEMA,
            "payload": payload,
            "payload_sha256": sha256_bytes(_canonical_json_bytes(payload)),
        }
        destination = self.path(normalized)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            existing = self._decode(normalized, destination.read_bytes())
            if existing.content != value.content:
                raise RuntimeError("Boris/Kaggle cache changed for a pinned URL.")
            return existing
        write_atomic(
            destination,
            gzip.compress(_canonical_json_bytes(envelope), mtime=0),
        )
        return self._decode(normalized, destination.read_bytes())

    def fill_missing(
        self, symbols: Iterable[str]
    ) -> dict[str, BorisKaggleCachedResponse]:
        ordered = tuple(
            dict.fromkeys(self._normalized_symbol(item) for item in symbols)
        )
        missing = [symbol for symbol in ordered if self.get(symbol) is None]
        remaining = self.max_http_attempts - self.http_attempts
        if len(missing) > remaining:
            raise RuntimeError(
                "Boris/Kaggle request set exceeds its two-attempt run cap before "
                f"network access: {len(missing)} > {remaining}."
            )
        return {symbol: self.get(symbol) or self.fetch(symbol) for symbol in ordered}


def _parse_boris_kaggle_response(
    response: BorisKaggleCachedResponse,
    *,
    security_id: str,
) -> pd.DataFrame:
    if response.http_status != 200:
        raise ValueError(
            f"Boris/Kaggle returned HTTP {response.http_status} for {response.symbol}."
        )
    expected_hash = BORIS_KAGGLE_FILES[response.symbol]["sha256"]
    if response.source_hash != expected_hash:
        raise ValueError(f"Boris/Kaggle raw hash mismatch for {response.symbol}.")
    if response.content.lstrip().startswith((b"<", b"<!")):
        raise ValueError(f"Boris/Kaggle returned HTML for {response.symbol}.")
    try:
        raw = pd.read_csv(io.BytesIO(response.content))
    except Exception as exc:
        raise ValueError(
            f"Boris/Kaggle CSV is unreadable for {response.symbol}."
        ) from exc
    expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
    if list(raw.columns) != expected_columns:
        raise ValueError(f"Boris/Kaggle CSV schema changed for {response.symbol}.")
    spec = BORIS_KAGGLE_FILES[response.symbol]
    if len(raw) != int(spec["raw_rows"]):
        raise ValueError(
            f"Boris/Kaggle raw row count changed for {response.symbol}: {len(raw)}."
        )
    sessions = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="coerce")
    if sessions.isna().any():
        raise ValueError(f"Boris/Kaggle contains invalid dates for {response.symbol}.")
    raw["session"] = sessions.dt.date.astype(str)
    if raw["session"].duplicated().any() or not raw["session"].is_monotonic_increasing:
        raise ValueError(f"Boris/Kaggle sessions are not unique/sorted for {response.symbol}.")
    for column in ("Open", "High", "Low", "Close", "Volume", "OpenInt"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    numeric = raw[["Open", "High", "Low", "Close", "Volume", "OpenInt"]]
    if not bool(numeric.apply(lambda values: values.map(math.isfinite)).all().all()):
        raise ValueError(f"Boris/Kaggle contains non-finite values for {response.symbol}.")
    coherent = (
        numeric[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
        & numeric["Volume"].ge(0)
        & numeric["OpenInt"].ge(0)
        & numeric["High"].ge(numeric[["Open", "Low", "Close"]].max(axis=1))
        & numeric["Low"].le(numeric[["Open", "High", "Close"]].min(axis=1))
    )
    if not bool(coherent.all()):
        raise ValueError(f"Boris/Kaggle contains incoherent OHLCV for {response.symbol}.")
    frame = raw.loc[
        raw["session"].ge(str(spec["segment_start"]))
        & raw["session"].le(str(spec["segment_end"]))
    ].copy()
    expected_sessions = _expected_sessions(
        str(spec["segment_start"]),
        str(spec["segment_end"]),
    )
    if (
        len(expected_sessions) != int(spec["segment_rows"])
        or tuple(frame["session"].astype(str)) != expected_sessions
    ):
        raise ValueError(
            f"Boris/Kaggle exact cross-check coverage failed for {response.symbol}."
        )
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    frame["security_id"] = security_id
    frame["currency"] = "USD"
    frame["source"] = BORIS_KAGGLE_SOURCE
    frame["source_url"] = response.source_url
    frame["retrieved_at"] = response.retrieved_at
    frame["source_hash"] = response.source_hash
    columns = list(dataset_spec("daily_price_raw").required_columns)
    if "source_url" not in columns:
        columns.append("source_url")
    return frame.loc[:, columns]


def _boris_artifact(response: BorisKaggleCachedResponse) -> SourceArtifact:
    return SourceArtifact(
        source=BORIS_KAGGLE_SOURCE,
        source_url=response.source_url,
        retrieved_at=response.retrieved_at,
        content=response.content,
        content_type=response.content_type,
    )


class BorisKaggleCrosscheckSource:
    """Pinned CC0 overlap evidence; upstream provider independence is not claimed."""

    def __init__(self, root: Path, *, allow_http: bool):
        self.allow_http = bool(allow_http)
        self.cache = BorisKaggleCache(root)

    @property
    def http_attempts(self) -> int:
        return int(self.cache.http_attempts)

    def fetch(self, ids: IdentityIds) -> FetchedHistories:
        symbols = tuple(BORIS_KAGGLE_FILES)
        responses = {symbol: self.cache.get(symbol) for symbol in symbols}
        missing = [symbol for symbol, value in responses.items() if value is None]
        if missing and not self.allow_http:
            raise FileNotFoundError(
                "Boris/Kaggle cross-check cache is missing and network access was not "
                "explicitly allowed: "
                + ", ".join(missing)
                + ". Re-run with --fetch-boris-crosscheck."
            )
        if missing:
            self.cache.fill_missing(symbols)
            responses = {symbol: self.cache.get(symbol) for symbol in symbols}
        if any(value is None for value in responses.values()):
            raise RuntimeError("Boris/Kaggle cross-check cache did not fill completely.")
        typed = {symbol: value for symbol, value in responses.items() if value is not None}
        targets = {"LILA": OLD_LILA_ID, "LILAK": OLD_LILAK_ID}
        frames = tuple(
            _parse_boris_kaggle_response(
                typed[symbol], security_id=targets[symbol]
            )
            for symbol in symbols
        )
        return FetchedHistories(
            prices=pd.DataFrame(
                columns=dataset_spec("daily_price_raw").required_columns
            ),
            crosscheck_prices=_concat_unique(
                frames,
                keys=dataset_spec("daily_price_raw").primary_key,
            ),
            corporate_actions=pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=tuple(_boris_artifact(typed[symbol]) for symbol in symbols),
            role_codes={
                role: f"BORIS_KAGGLE_V3:{symbol}"
                for role, symbol in zip(REQUIRED_BORIS_ROLES, symbols, strict=True)
            },
            http_attempts=0,
        )


OFFICIAL_EVIDENCE_URLS = tuple(
    dict.fromkeys(str(value["url"]) for value in OFFICIAL_EVIDENCE.values())
)
MAX_OFFICIAL_HTTP_ATTEMPTS = 26
if len(OFFICIAL_EVIDENCE_URLS) != MAX_OFFICIAL_HTTP_ATTEMPTS:
    raise RuntimeError(
        "Official evidence inventory changed without an audited network-cap update: "
        f"urls={len(OFFICIAL_EVIDENCE_URLS)}, cap={MAX_OFFICIAL_HTTP_ATTEMPTS}."
    )


class OfficialEvidenceSource:
    """Explicit one-attempt raw official-document cache."""

    def __init__(self, root: Path, *, allow_http: bool):
        load_env()
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.http_attempts = 0

    def _path(self, url: str) -> Path:
        return self.root / f"{sha256_bytes(url.encode())}.json.gz"

    def _decode(self, url: str, payload: bytes) -> SourceArtifact:
        try:
            envelope = json.loads(gzip.decompress(payload))
            content = base64.b64decode(envelope["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError(f"Official evidence cache is unreadable: {self._path(url)}") from exc
        if envelope.get("schema") != "official_identity_evidence_raw/v1":
            raise ValueError("Official evidence cache schema mismatch.")
        if str(envelope.get("source_url")) != url:
            raise ValueError("Official evidence cache URL mismatch.")
        if str(envelope.get("source_hash")) != sha256_bytes(content):
            raise ValueError("Official evidence cache content hash mismatch.")
        return SourceArtifact(
            source="official_identity_evidence_raw",
            source_url=url,
            retrieved_at=str(envelope["retrieved_at"]),
            content=content,
            content_type=str(envelope.get("content_type") or "application/octet-stream"),
        )

    def get(self, url: str) -> SourceArtifact | None:
        path = self._path(url)
        return self._decode(url, path.read_bytes()) if path.is_file() else None

    def _fetch(self, url: str) -> SourceArtifact:
        if self.http_attempts >= MAX_OFFICIAL_HTTP_ATTEMPTS:
            raise RuntimeError("Official evidence HTTP attempt cap reached.")
        self.http_attempts += 1
        user_agent = os.getenv(
            "SEC_USER_AGENT", "SuperTrendQuant identity-repair contact-required"
        )
        request = Request(
            url,
            headers={"User-Agent": user_agent, "Accept-Encoding": "identity"},
        )
        try:
            with urlopen(request, timeout=60) as response:
                status = int(getattr(response, "status", 200))
                content_type = str(
                    response.headers.get("Content-Type") or "application/octet-stream"
                )
                content = response.read(50 * 1024 * 1024 + 1)
        except HTTPError as exc:
            raise RuntimeError(
                f"Official evidence single request failed for {url}: HTTP {exc.code}"
            ) from None
        except URLError as exc:
            raise RuntimeError(
                f"Official evidence single request failed for {url}: {exc.reason}"
            ) from None
        if status != 200 or len(content) > 50 * 1024 * 1024:
            raise RuntimeError(
                f"Official evidence response rejected for {url}: status={status}, bytes={len(content)}."
            )
        retrieved_at = utc_now_iso()
        envelope = {
            "schema": "official_identity_evidence_raw/v1",
            "source_url": url,
            "retrieved_at": retrieved_at,
            "content_type": content_type,
            "source_hash": sha256_bytes(content),
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
        destination = self._path(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            existing = self._decode(url, destination.read_bytes())
            if existing.content != content:
                raise RuntimeError(f"Official evidence changed for immutable URL cache: {url}")
            return existing
        encoded = json.dumps(
            envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        write_atomic(destination, gzip.compress(encoded, mtime=0))
        return self._decode(url, destination.read_bytes())

    def load(self) -> OfficialEvidenceBundle:
        cached = {url: self.get(url) for url in OFFICIAL_EVIDENCE_URLS}
        missing = [url for url, artifact in cached.items() if artifact is None]
        if missing and not self.allow_http:
            raise FileNotFoundError(
                "Raw official identity evidence cache is incomplete and HTTP was not "
                "explicitly allowed; re-run with --fetch-official-evidence. Missing URLs: "
                + ", ".join(missing)
            )
        for url in missing:
            cached[url] = self._fetch(url)
        if any(item is None for item in cached.values()):
            raise RuntimeError("Official identity evidence cache did not fill completely.")
        manifest = _official_evidence_artifact(retrieved_at=utc_now_iso())
        bundle = OfficialEvidenceBundle(
            manifest=manifest,
            raw_artifacts=tuple(cached[url] for url in OFFICIAL_EVIDENCE_URLS),
        )
        for url in OFFICIAL_EVIDENCE_URLS:
            bundle.source_hash_for(url)
        return bundle


def _read_release_frames(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, pd.DataFrame]:
    missing = tuple(name for name in WRITE_DATASETS if name not in release.dataset_versions)
    if missing:
        raise RuntimeError(
            "Frozen release lacks identity-repair datasets: " + ", ".join(missing)
        )
    return {
        name: repository.read_frame(name, release.dataset_versions[name])
        for name in WRITE_DATASETS
    }


def _capture_pointer_etags(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        expected = release.dataset_versions.get(dataset, "")
        actual = pointer.version if pointer is not None else ""
        if not expected or actual != expected:
            raise RuntimeError(
                f"{dataset} pointer is not the frozen release version: "
                f"expected={expected or 'missing'}, actual={actual or 'missing'}."
            )
        output[dataset] = etag
    return output


def _read_catalog_artifact(
    repository: LocalDatasetRepository,
    row: pd.Series,
) -> SourceArtifact:
    path = repository.root / str(row["object_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Archived EODHD catalog is missing: {path}")
    try:
        content = gzip.decompress(path.read_bytes())
    except Exception as exc:
        raise ValueError(f"Archived EODHD catalog is unreadable: {path}") from exc
    expected_hash = str(row["source_hash"])
    if sha256_bytes(content) != expected_hash:
        raise ValueError(f"Archived EODHD catalog hash mismatch: {path}")
    return SourceArtifact(
        source=str(row["source"]),
        source_url=str(row["source_url"]),
        retrieved_at=str(row["retrieved_at"]),
        content=content,
        content_type=str(row["content_type"]),
    )


def load_catalog_entries(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], tuple[SourceArtifact, ...]]:
    candidates = source_archive.loc[
        source_archive["dataset"].astype(str).eq("eodhd_exchange_symbols")
    ].copy()
    by_kind: dict[str, SourceArtifact] = {}
    for _index, row in candidates.iterrows():
        url = str(row["source_url"])
        kind = "delisted" if "delisted=1" in url else "active" if "delisted=0" in url else ""
        if not kind:
            continue
        artifact = _read_catalog_artifact(repository, row)
        if kind in by_kind and by_kind[kind].source_hash != artifact.source_hash:
            raise ValueError(f"Frozen release has ambiguous {kind} EODHD catalogs.")
        by_kind[kind] = artifact
    if set(by_kind) != {"active", "delisted"}:
        raise ValueError("Frozen active and delisted EODHD catalogs are both required.")

    entries: dict[str, dict[str, Any]] = {}
    entry_kind: dict[str, str] = {}
    for kind, artifact in by_kind.items():
        value = json.loads(artifact.content)
        if not isinstance(value, list):
            raise ValueError(f"{kind} EODHD catalog payload is not a JSON list.")
        for raw in value:
            code = str(raw.get("Code") or "").strip().upper()
            if not code:
                continue
            if code in entries:
                # The active row is authoritative for a currently listed code.
                if kind == "active":
                    entries[code] = dict(raw)
                    entry_kind[code] = kind
                continue
            entries[code] = dict(raw)
            entry_kind[code] = kind

    required = {
        *(item.upper() for item in PRICE_PROBE_CODES),
        "BHI",
        "GDI",
        "GDI1",
        "ABC",
        "COR",
        "AA",
        "BKR",
        "HWM",
        "IR",
        "TT",
        "FOX",
        "FOXA",
        "LILA",
        "LILAK",
    }
    missing = sorted(required - set(entries))
    if missing:
        raise ValueError("Frozen EODHD catalogs lack audited codes: " + ", ".join(missing))
    for code in (item for item in PRICE_PROBE_CODES if item not in {"AA", "TNL"}):
        if entry_kind.get(code.upper()) != "delisted":
            raise ValueError(f"Audited historical candidate {code} is not delisted.")
    if entry_kind.get("AA") != "active":
        raise ValueError("AA strict stitched-history probe must use the active catalog row.")
    if entry_kind.get("TNL") != "active":
        raise ValueError("TNL continuation fetch must use the active catalog row.")
    expected_isins = {
        "ABC": "US03073E1055",
        "COR": "US03073E1055",
        "BHI": "US0572241075",
        "WYN": "US98310W1080",
        "GDI": "US36555P1075",
        "TNL": "US8941641024",
    }
    for code, expected in expected_isins.items():
        actual = str(entries[code].get("Isin") or "").upper()
        if actual != expected:
            raise ValueError(
                f"Frozen catalog ISIN mismatch for {code}: expected={expected}, actual={actual or 'missing'}."
            )
    # EODHD's archived VALPQ row has no ISIN.  It is nevertheless the exact
    # legacy common-equity continuation: the frozen catalog identifies the
    # PINK common stock as Ensco Rowan PLC, while the bounded price history and
    # CIK-bound SEC filings below prove the ESV -> VAL -> VALPQ lineage.  Do
    # not substitute VAL_old's unrelated catalog ISIN onto this row.
    valpq = entries[VALARIS_PROVIDER_CODE]
    valpq_identity = {
        "Name": str(valpq.get("Name") or ""),
        "Exchange": str(valpq.get("Exchange") or ""),
        "Type": str(valpq.get("Type") or ""),
        "Isin": str(valpq.get("Isin") or ""),
    }
    if valpq_identity != {
        "Name": "Ensco Rowan PLC",
        "Exchange": "PINK",
        "Type": "Common Stock",
        "Isin": "",
    }:
        raise ValueError(
            "Frozen VALPQ catalog identity changed: "
            + json.dumps(valpq_identity, sort_keys=True)
        )
    entries["__kind__"] = entry_kind
    return entries, tuple(by_kind.values())


def _unique_id(
    master: pd.DataFrame,
    *,
    provider_symbol: str,
    name_token: str = "",
) -> str:
    mask = master.get("provider_symbol", pd.Series("", index=master.index)).astype(str).eq(
        provider_symbol
    )
    if name_token:
        mask &= master["name"].astype(str).str.lower().str.contains(
            name_token.lower(), regex=False
        )
    matches = master.loc[mask, "security_id"].astype(str).drop_duplicates()
    if len(matches) != 1:
        raise ValueError(
            f"Expected one identity for {provider_symbol}/{name_token or '*'}, found {len(matches)}."
        )
    return str(matches.iloc[0])


def _resolve_identity_ids(master: pd.DataFrame) -> IdentityIds:
    return IdentityIds(
        agn_legacy=_unique_id(
            master, provider_symbol="AGN_old.US", name_token="allergan inc"
        ),
        agn_actavis=_unique_id(
            master, provider_symbol="AGN.US", name_token="allergan plc"
        ),
        abc_duplicate=_unique_id(master, provider_symbol="ABC.US"),
        cor=_unique_id(master, provider_symbol="COR.US", name_token="cencora"),
        coresite_duplicate=_unique_id(
            master, provider_symbol="COR_old.US", name_token="coresite"
        ),
        fox=_unique_id(master, provider_symbol="FOX.US", name_token="fox corp"),
        foxa=_unique_id(master, provider_symbol="FOXA.US", name_token="fox corp"),
        wynd=_unique_id(master, provider_symbol="WYND.US"),
        sea=_unique_id(master, provider_symbol="SE.US", name_token="sea ltd"),
        lila=_unique_id(master, provider_symbol="LILA.US", name_token="liberty latin"),
        lilak=_unique_id(master, provider_symbol="LILAK.US", name_token="liberty latin"),
        bhge=_unique_id(master, provider_symbol="BHGE.US"),
        bkr=_unique_id(master, provider_symbol="BKR.US"),
        ir=_unique_id(master, provider_symbol="IR.US"),
        tt=_unique_id(master, provider_symbol="TT.US"),
        arnc=_unique_id(master, provider_symbol="ARNC.US"),
        arnc_duplicate=_unique_id(master, provider_symbol="ARNC_old.US"),
        hwm=_unique_id(master, provider_symbol="HWM.US"),
        hot=_unique_id(master, provider_symbol="HOT.US", name_token="starwood"),
        esv=_unique_id(master, provider_symbol="ESV.US", name_token="valaris"),
        mar=_unique_id(master, provider_symbol="MAR.US", name_token="marriott"),
        azn=_unique_id(master, provider_symbol="AZN.US", name_token="astrazeneca"),
        azn_duplicate=_unique_id(
            master, provider_symbol="AZN_old.US", name_token="astrazeneca"
        ),
    )


def _price_range(prices: pd.DataFrame, security_id: str) -> tuple[str, str, int]:
    rows = prices.loc[prices["security_id"].astype(str).eq(security_id)]
    if rows.empty:
        return "", "", 0
    sessions = pd.to_datetime(rows["session"], errors="coerce").dropna()
    if sessions.empty:
        return "", "", 0
    return (
        sessions.min().date().isoformat(),
        sessions.max().date().isoformat(),
        len(rows),
    )


def validate_azn_exact_duplicate(
    prices: pd.DataFrame,
    ids: IdentityIds,
) -> dict[str, Any]:
    """Prove AZN_old is an OHLC duplicate before discarding its identity."""

    columns = ["session", "open", "high", "low", "close", "volume"]
    azn = prices.loc[
        prices["security_id"].astype(str).eq(ids.azn), columns
    ]
    azn_old = prices.loc[
        prices["security_id"].astype(str).eq(ids.azn_duplicate), columns
    ]
    azn_overlap = azn_old.merge(azn, on="session", suffixes=("_old", "_new"))
    if len(azn_overlap) != len(azn_old) or len(azn_overlap) < 2_700:
        raise ValueError("AZN_old rows do not fully overlap canonical AZN history.")
    for column in ("open", "high", "low", "close"):
        old = pd.to_numeric(azn_overlap[f"{column}_old"], errors="coerce")
        new = pd.to_numeric(azn_overlap[f"{column}_new"], errors="coerce")
        if not old.eq(new).all():
            raise ValueError(f"AZN_old is no longer an exact AZN {column} duplicate.")
    old_volume = pd.to_numeric(azn_overlap["volume_old"], errors="coerce")
    new_volume = pd.to_numeric(azn_overlap["volume_new"], errors="coerce")
    equal_volume = int(old_volume.eq(new_volume).sum())
    if equal_volume:
        raise ValueError(
            "AZN duplicate audit assumption changed: volume series unexpectedly overlap."
        )
    return {
        "overlap_sessions": len(azn_overlap),
        "ohlc_equal_sessions": len(azn_overlap),
        "volume_equal_sessions": equal_volume,
        "canonical_security_id": ids.azn,
        "removed_security_id": ids.azn_duplicate,
    }


def _assert_local_shape(existing: dict[str, pd.DataFrame], ids: IdentityIds) -> None:
    prices = existing["daily_price_raw"]
    expected = {
        ids.agn_legacy: ("2015-01-02", "2015-03-16"),
        ids.agn_actavis: ("2015-01-02", "2020-05-08"),
        ids.cor: ("2015-01-02", None),
        ids.fox: ("2019-03-13", None),
        ids.foxa: ("2019-03-12", None),
        ids.wynd: ("2018-06-01", "2021-02-16"),
        ids.sea: ("2017-10-20", None),
        ids.lila: ("2018-01-02", None),
        ids.lilak: ("2018-01-02", None),
        ids.bhge: ("2017-07-05", None),
        ids.bkr: ("2015-01-02", None),
        ids.ir: ("2017-05-12", None),
        ids.tt: ("2015-01-02", None),
        ids.arnc: ("2016-11-01", None),
        ids.hwm: ("2016-11-01", None),
        ids.hot: ("2015-01-02", "2017-03-20"),
        ids.esv: ("2015-01-02", "2019-07-30"),
        ids.azn: ("2015-01-02", None),
        ids.azn_duplicate: ("2015-01-02", "2026-01-30"),
    }
    for security_id, (minimum, maximum) in expected.items():
        first, last, count = _price_range(prices, security_id)
        if not count or first != minimum or (maximum is not None and last != maximum):
            raise ValueError(
                "Frozen local price shape changed for identity repair: "
                f"security_id={security_id}, expected={minimum}..{maximum or '*'}, "
                f"actual={first or 'missing'}..{last or 'missing'}."
            )
    cor_old = prices.loc[
        prices["security_id"].astype(str).eq(ids.coresite_duplicate),
        ["session", "open", "high", "low", "close", "volume"],
    ]
    cor = prices.loc[
        prices["security_id"].astype(str).eq(ids.cor),
        ["session", "open", "high", "low", "close", "volume"],
    ]
    overlap = cor_old.merge(cor, on="session", suffixes=("_old", "_new"))
    if len(overlap) != len(cor_old):
        raise ValueError("CoreSite-contaminated COR rows do not fully overlap Cencora.")
    for column in ("open", "high", "low", "close", "volume"):
        old = pd.to_numeric(overlap[f"{column}_old"], errors="coerce")
        new = pd.to_numeric(overlap[f"{column}_new"], errors="coerce")
        if not old.eq(new).all():
            raise ValueError("COR_old is no longer an exact Cencora duplicate.")
    bhge = prices.loc[
        prices["security_id"].astype(str).eq(ids.bhge), ["session", "close"]
    ]
    bkr = prices.loc[
        prices["security_id"].astype(str).eq(ids.bkr), ["session", "close"]
    ]
    baker_overlap = bhge.merge(bkr, on="session", suffixes=("_bhge", "_bkr"))
    if len(baker_overlap) < 500:
        raise ValueError("BHGE/BKR local continuity overlap is unexpectedly short.")
    left = pd.to_numeric(baker_overlap["close_bhge"], errors="coerce")
    right = pd.to_numeric(baker_overlap["close_bkr"], errors="coerce")
    if not left.eq(right).all():
        raise ValueError("BHGE/BKR local close histories no longer prove continuity.")

    # AZN_old is a provider-generated identity duplicate, not an issuer
    # transition.  Prove exact OHLC equality before any rewrite can drop it.
    validate_azn_exact_duplicate(prices, ids)
    azn_events = existing["index_membership_events"].loc[
        existing["index_membership_events"]["security_id"]
        .astype(str)
        .eq(ids.azn_duplicate)
    ]
    observed_azn_events = {
        (
            str(row.index_id),
            str(row.effective_date),
            str(row.operation).upper(),
        )
        for row in azn_events.itertuples(index=False)
    }
    expected_azn_events = {
        ("nasdaq100", "2022-02-22", "ADD"),
        ("nasdaq100", "2026-01-20", "REMOVE"),
    }
    if observed_azn_events != expected_azn_events or len(azn_events) != 2:
        raise ValueError(
            "Frozen AZN_old Nasdaq membership shape changed: "
            f"expected={sorted(expected_azn_events)}, "
            f"actual={sorted(observed_azn_events)}."
        )
    if existing["index_constituent_anchors"]["security_id"].astype(str).isin(
        {ids.azn, ids.azn_duplicate}
    ).any():
        raise ValueError("Frozen AZN duplicate audit unexpectedly found an anchor row.")


def build_local_preflight(
    repository: LocalDatasetRepository,
    release: DataRelease,
    *,
    require_successor_snapshot: bool = False,
) -> LocalPreflight:
    if MAX_EODHD_HTTP_ATTEMPTS != 32:
        raise RuntimeError("The audited successful-run EODHD cap must remain exactly 32.")
    existing = _read_release_frames(repository, release)
    ids = _resolve_identity_ids(existing["security_master"])
    _assert_local_shape(existing, ids)
    if require_successor_snapshot:
        if PENDING_IDENTITY_WARNING not in set(release.warnings):
            raise ValueError(
                "Identity repair must run after the successor collector; missing release "
                f"warning: {PENDING_IDENTITY_WARNING}"
            )
        cov_gaps = _coverage_missing(
            existing["daily_price_raw"],
            COV_SECURITY_ID,
            "2015-01-02",
            "2015-01-26",
        )
        if cov_gaps:
            raise ValueError(
                "Successor snapshot has incomplete COV history: "
                f"{len(cov_gaps)} missing sessions."
            )
    catalogs, _catalog_artifacts = load_catalog_entries(
        repository, existing["source_archive"]
    )
    return LocalPreflight(
        existing=existing,
        ids=ids,
        catalogs=catalogs,
        pointer_etags=_capture_pointer_etags(repository, release),
    )


def _role_ids(ids: IdentityIds) -> dict[str, str]:
    return {
        "wyn": ids.wynd,
        "spectra": SPECTRA_ID,
        "old_fox": OLD_FOX_ID,
        "old_foxa": OLD_FOXA_ID,
        "old_lila": OLD_LILA_ID,
        "old_lilak": OLD_LILAK_ID,
        "old_aa": ids.hwm,
        "tnl": ids.wynd,
        "valaris": ids.esv,
        "bhi": BHI_ID,
    }


def _bundle_signature(release: DataRelease) -> dict[str, Any]:
    return {
        "format": "supertrendquant-us-index-identity-fetch-v2-valpq",
        "release_version": release.version,
        "completed_session": release.completed_session,
        "price_probe_codes": list(PRICE_PROBE_CODES),
        "maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
    }


def _bundle_cache_path(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> Path:
    return (
        repository.root
        / "state/us-index-identity-repairs"
        / f"{release.version}.valpq-v2.json.gz"
    )


def _bundle_value(
    fetched: FetchedHistories,
    *,
    signature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "role_codes": dict(sorted(fetched.role_codes.items())),
        "http_attempts": int(fetched.http_attempts),
        "prices": fetched.prices.to_dict("records"),
        "crosscheck_prices": fetched.crosscheck_prices.to_dict("records"),
        "corporate_actions": fetched.corporate_actions.to_dict("records"),
        "artifacts": [
            {
                "source": item.source,
                "source_url": item.source_url,
                "retrieved_at": item.retrieved_at,
                "content_type": item.content_type,
                "content_base64": base64.b64encode(item.content).decode("ascii"),
                "content_sha256": item.source_hash,
            }
            for item in fetched.artifacts
        ],
    }
    if signature is not None:
        value["signature"] = signature
    return value


def _fetched_from_value(value: dict[str, Any]) -> FetchedHistories:
    artifacts: list[SourceArtifact] = []
    for item in value.get("artifacts", []):
        content = base64.b64decode(str(item["content_base64"]), validate=True)
        if sha256_bytes(content) != str(item["content_sha256"]):
            raise ValueError("Identity bundle artifact hash mismatch.")
        artifacts.append(
            SourceArtifact(
                source=str(item["source"]),
                source_url=str(item["source_url"]),
                retrieved_at=str(item["retrieved_at"]),
                content=content,
                content_type=str(item["content_type"]),
            )
        )
    def records_frame(dataset: str, key: str) -> pd.DataFrame:
        records = value.get(key, [])
        if not isinstance(records, list) or not all(
            isinstance(item, dict) for item in records
        ):
            raise ValueError(f"Identity bundle {key} records are malformed.")
        columns = tuple(
            dict.fromkeys(
                (
                    *dataset_spec(dataset).required_columns,
                    *(column for item in records for column in item),
                )
            )
        )
        return pd.DataFrame(records, columns=columns)

    return FetchedHistories(
        prices=records_frame("daily_price_raw", "prices"),
        crosscheck_prices=records_frame(
            "daily_price_raw", "crosscheck_prices"
        ),
        corporate_actions=records_frame(
            "corporate_actions", "corporate_actions"
        ),
        artifacts=tuple(artifacts),
        role_codes={str(k): str(v) for k, v in value.get("role_codes", {}).items()},
        http_attempts=int(value.get("http_attempts", 0)),
    )


def _write_bundle_cache(
    path: Path,
    release: DataRelease,
    fetched: FetchedHistories,
) -> None:
    value = _bundle_value(fetched, signature=_bundle_signature(release))
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, gzip.compress(payload, mtime=0))


def _read_bundle_cache(
    path: Path,
    release: DataRelease,
) -> FetchedHistories | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(gzip.decompress(path.read_bytes()))
    except Exception as exc:
        raise ValueError(f"Identity fetch cache is unreadable: {path}") from exc
    if value.get("signature") != _bundle_signature(release):
        raise ValueError(f"Identity fetch cache signature mismatch: {path}")
    fetched = _fetched_from_value(value)
    if fetched.http_attempts != MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError("Identity fetch cache does not record exactly 32 HTTP attempts.")
    return fetched


def _concat_unique(
    frames: Iterable[pd.DataFrame],
    *,
    keys: tuple[str, ...],
) -> pd.DataFrame:
    values = [frame for frame in frames if frame is not None and not frame.empty]
    if not values:
        return pd.DataFrame()
    return pd.concat(values, ignore_index=True).drop_duplicates(list(keys), keep="last")


def _frame_key_set(frame: pd.DataFrame, keys: tuple[str, ...]) -> set[tuple[str, ...]]:
    if frame.empty:
        return set()
    return set(map(tuple, frame.loc[:, list(keys)].astype(str).to_numpy()))


def validate_yahoo_identity_supplement(
    supplement: FetchedHistories,
    ids: IdentityIds,
) -> dict[str, Any]:
    expected_roles = {
        spec["role"]: f"YAHOO_CHART:{symbol}"
        for symbol, spec in YAHOO_CHART_REQUESTS.items()
    }
    if supplement.role_codes != expected_roles:
        raise ValueError("Yahoo identity supplement role binding changed.")
    if supplement.http_attempts != 0 or not supplement.corporate_actions.empty:
        raise ValueError("Yahoo identity supplement has an invalid call/action payload.")
    if len(supplement.artifacts) != len(YAHOO_SUPPLEMENT_SYMBOLS):
        raise ValueError("Yahoo identity supplement must retain exactly two artifacts.")

    expected_sessions = _expected_sessions(
        LILA_REGULAR_PRICE_START,
        LILA_REGULAR_PRICE_END,
    )
    targets = {"LILA": OLD_LILA_ID, "LILAK": OLD_LILAK_ID}
    result: dict[str, Any] = {}
    for symbol, security_id in targets.items():
        rows = supplement.prices.loc[
            supplement.prices["security_id"].astype(str).eq(security_id)
        ].sort_values("session")
        if tuple(rows["session"].astype(str)) != expected_sessions:
            raise ValueError(
                f"Yahoo {symbol} primary must have exactly 630 regular-way sessions."
            )
        artifacts = [
            item
            for item in supplement.artifacts
            if item.source_url == _yahoo_chart_url(symbol)
            and item.source == YAHOO_LILA_PRIMARY_SOURCE
        ]
        if len(artifacts) != 1:
            raise ValueError(f"Yahoo {symbol} primary raw artifact is not unique.")
        artifact = artifacts[0]
        if set(rows["source"].astype(str)) != {YAHOO_LILA_PRIMARY_SOURCE}:
            raise ValueError(f"Yahoo {symbol} primary provenance changed.")
        if (
            set(rows["source_url"].astype(str)) != {artifact.source_url}
            or set(rows["source_hash"].astype(str)) != {artifact.source_hash}
        ):
            raise ValueError(f"Yahoo {symbol} rows lost exact raw URL/hash provenance.")
        result[symbol.lower()] = {
            "security_id": security_id,
            "sessions": len(expected_sessions),
            "start": LILA_REGULAR_PRICE_START,
            "end": LILA_REGULAR_PRICE_END,
            "source": YAHOO_LILA_PRIMARY_SOURCE,
            "source_url": artifact.source_url,
            "source_sha256": artifact.source_hash,
        }
    if set(supplement.prices["security_id"].astype(str)) != set(targets.values()):
        raise ValueError("Yahoo primary contains an unexpected identity.")
    if not supplement.crosscheck_prices.empty:
        raise ValueError("Yahoo LILA/LILAK supplement may not contain cross-check rows.")
    return result


def validate_boris_crosscheck_bundle(
    supplement: FetchedHistories,
    ids: IdentityIds,
) -> dict[str, Any]:
    expected_roles = {
        role: f"BORIS_KAGGLE_V3:{symbol}"
        for role, symbol in zip(
            REQUIRED_BORIS_ROLES,
            BORIS_KAGGLE_FILES,
            strict=True,
        )
    }
    if supplement.role_codes != expected_roles:
        raise ValueError("Boris/Kaggle cross-check role binding changed.")
    if (
        supplement.http_attempts != 0
        or not supplement.prices.empty
        or not supplement.corporate_actions.empty
    ):
        raise ValueError("Boris/Kaggle bundle may contain cross-check prices only.")
    if len(supplement.artifacts) != len(BORIS_KAGGLE_FILES):
        raise ValueError(
            "Boris/Kaggle bundle must retain exactly two raw artifacts."
        )
    targets = {"LILA": OLD_LILA_ID, "LILAK": OLD_LILAK_ID}
    result: dict[str, Any] = {}
    for symbol, security_id in targets.items():
        spec = BORIS_KAGGLE_FILES[symbol]
        expected_sessions = _expected_sessions(
            str(spec["segment_start"]),
            str(spec["segment_end"]),
        )
        rows = supplement.crosscheck_prices.loc[
            supplement.crosscheck_prices["security_id"].astype(str).eq(security_id)
        ].sort_values("session")
        if tuple(rows["session"].astype(str)) != expected_sessions:
            raise ValueError(
                f"Boris/Kaggle {symbol} coverage is not exactly "
                f"{spec['segment_rows']} sessions."
            )
        artifacts = [
            item
            for item in supplement.artifacts
            if item.source_url == spec["url"]
            and item.source == BORIS_KAGGLE_SOURCE
            and item.source_hash == spec["sha256"]
        ]
        if len(artifacts) != 1:
            raise ValueError(f"Boris/Kaggle {symbol} raw artifact is not unique/pinned.")
        artifact = artifacts[0]
        if (
            set(rows["source"].astype(str)) != {BORIS_KAGGLE_SOURCE}
            or set(rows["source_url"].astype(str)) != {artifact.source_url}
            or set(rows["source_hash"].astype(str)) != {artifact.source_hash}
        ):
            raise ValueError(f"Boris/Kaggle {symbol} rows lost exact provenance.")
        result[symbol.lower()] = {
            "security_id": security_id,
            "sessions": len(expected_sessions),
            "start": str(spec["segment_start"]),
            "end": str(spec["segment_end"]),
            "source_url": artifact.source_url,
            "source_sha256": artifact.source_hash,
        }
    if set(supplement.crosscheck_prices["security_id"].astype(str)) != set(
        targets.values()
    ):
        raise ValueError("Boris/Kaggle cross-check contains an unexpected identity.")
    return result


def validate_wiki_arnc_bundle(
    supplement: FetchedHistories,
    ids: IdentityIds,
) -> dict[str, Any]:
    expected_role_codes = {
        WIKI_ARNC_ROLE: f"WIKI/PRICES:ARNC@{WIKI_ARNC_COMMIT}"
    }
    if supplement.role_codes != expected_role_codes:
        raise ValueError("WIKI/ARNC primary role binding changed.")
    if (
        supplement.http_attempts != 0
        or not supplement.crosscheck_prices.empty
        or not supplement.corporate_actions.empty
    ):
        raise ValueError("WIKI/ARNC bundle may contain primary raw prices only.")
    if len(supplement.artifacts) != 1:
        raise ValueError("WIKI/ARNC bundle must retain the one full raw artifact.")
    artifact = supplement.artifacts[0]
    if not (
        artifact.source == WIKI_ARNC_SOURCE
        and artifact.source_url == WIKI_ARNC_URL
        and artifact.source_hash == WIKI_ARNC_FULL_SHA256
        and len(artifact.content) == WIKI_ARNC_FULL_SIZE
        and artifact.content_type == "text/csv"
    ):
        raise ValueError("WIKI/ARNC full raw artifact lost pinned provenance.")
    reparsed, scan = _parse_wiki_arnc_response(
        WikiArncCachedResponse(
            source_url=artifact.source_url,
            retrieved_at=artifact.retrieved_at,
            content=artifact.content,
            content_type=artifact.content_type,
            http_status=200,
        ),
        security_id=ids.hwm,
    )
    keys = list(dataset_spec("daily_price_raw").primary_key)
    left = supplement.prices.sort_values(keys).reset_index(drop=True)
    right = reparsed.sort_values(keys).reset_index(drop=True)
    columns = [
        "security_id",
        "session",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "currency",
        "source",
        "source_url",
        "source_hash",
    ]
    if len(left) != WIKI_ARNC_SEGMENT_ROWS or not left[columns].equals(right[columns]):
        raise ValueError("WIKI/ARNC primary rows do not match deterministic extraction.")
    return {
        **scan,
        "security_id": ids.hwm,
        "published_basis": "unadjusted_ohlcv",
        "upstream": "Quandl/Nasdaq Data Link WIKI/PRICES",
        "transport": "third-party immutable GitHub LFS mirror",
        "upstream_license": WIKI_ARNC_UPSTREAM_LICENSE,
        "upstream_license_url": WIKI_ARNC_UPSTREAM_LICENSE_URL,
        "upstream_table_url": WIKI_ARNC_UPSTREAM_TABLE_URL,
        "mirror_provenance_url": WIKI_ARNC_MIRROR_PROVENANCE_URL,
        "full_blob_archived_on_apply": True,
    }


def merge_fetched_histories(
    primary: FetchedHistories,
    supplement: FetchedHistories,
    *,
    ids: IdentityIds,
) -> FetchedHistories:
    validate_yahoo_identity_supplement(supplement, ids)
    overlap = set(primary.role_codes) & set(supplement.role_codes)
    if overlap:
        raise ValueError(
            "Supplement may not override primary EODHD role codes: "
            + ", ".join(sorted(overlap))
        )
    price_keys = dataset_spec("daily_price_raw").primary_key
    collisions = _frame_key_set(primary.prices, price_keys) & _frame_key_set(
        supplement.prices, price_keys
    )
    if collisions:
        raise ValueError(
            "Yahoo LILA/LILAK primary overlaps main EODHD price keys: "
            f"{len(collisions)}."
        )
    return FetchedHistories(
        prices=pd.concat((primary.prices, supplement.prices), ignore_index=True),
        crosscheck_prices=pd.concat(
            (primary.crosscheck_prices, supplement.crosscheck_prices),
            ignore_index=True,
        ),
        corporate_actions=primary.corporate_actions.copy(),
        artifacts=tuple((*primary.artifacts, *supplement.artifacts)),
        role_codes={**primary.role_codes, **supplement.role_codes},
        http_attempts=primary.http_attempts,
    )


def merge_boris_crosscheck(
    primary: FetchedHistories,
    supplement: FetchedHistories,
    *,
    ids: IdentityIds,
) -> FetchedHistories:
    validate_boris_crosscheck_bundle(supplement, ids)
    overlap = set(primary.role_codes) & set(supplement.role_codes)
    if overlap:
        raise ValueError(
            "Boris/Kaggle may not override existing role codes: "
            + ", ".join(sorted(overlap))
        )
    keys = dataset_spec("daily_price_raw").primary_key
    collisions = _frame_key_set(primary.crosscheck_prices, keys) & _frame_key_set(
        supplement.crosscheck_prices,
        keys,
    )
    if collisions:
        raise ValueError(
            "Boris/Kaggle cross-check overlaps an existing cross-check key: "
            f"{len(collisions)}."
        )
    return FetchedHistories(
        prices=primary.prices.copy(),
        crosscheck_prices=pd.concat(
            (primary.crosscheck_prices, supplement.crosscheck_prices),
            ignore_index=True,
        ),
        corporate_actions=primary.corporate_actions.copy(),
        artifacts=tuple((*primary.artifacts, *supplement.artifacts)),
        role_codes={**primary.role_codes, **supplement.role_codes},
        http_attempts=primary.http_attempts,
    )


def _exact_eodhd_aa_artifact(fetched: FetchedHistories) -> SourceArtifact:
    candidates: list[SourceArtifact] = []
    for artifact in fetched.artifacts:
        parsed = urlparse(artifact.source_url)
        query = parse_qs(parsed.query)
        if (
            artifact.source == "eodhd_eod"
            and parsed.path.endswith("/eod/AA.US")
            and query == {"from": [FETCH_START], "to": [AA_CROSSCHECK_END]}
        ):
            candidates.append(artifact)
    if len(candidates) != 1:
        raise ValueError("Old AA requires one exact bounded raw EODHD artifact.")
    return candidates[0]


def merge_wiki_arnc_primary(
    primary: FetchedHistories,
    supplement: FetchedHistories,
    *,
    ids: IdentityIds,
) -> FetchedHistories:
    """Replace EODHD Old AA rows with WIKI raw OHLCV and retain EODHD as evidence."""

    validate_wiki_arnc_bundle(supplement, ids)
    overlap = set(primary.role_codes) & set(supplement.role_codes)
    if overlap:
        raise ValueError("WIKI/ARNC primary may not override an existing role code.")
    expected = _expected_sessions(AA_CROSSCHECK_START, AA_CROSSCHECK_END)
    old_aa = primary.prices.loc[
        primary.prices["security_id"].astype(str).eq(ids.hwm)
        & primary.prices["session"].astype(str).ge(AA_CROSSCHECK_START)
        & primary.prices["session"].astype(str).le(AA_CROSSCHECK_END)
    ].sort_values("session")
    if tuple(old_aa["session"].astype(str)) != expected:
        raise ValueError("EODHD Old AA cross-check does not have exact session coverage.")
    artifact = _exact_eodhd_aa_artifact(primary)
    if not (
        set(old_aa["source"].astype(str)) == {"eodhd_eod"}
        and set(old_aa["source_url"].astype(str)) == {artifact.source_url}
        and set(old_aa["source_hash"].astype(str)) == {artifact.source_hash}
    ):
        raise ValueError("EODHD Old AA rows lost exact raw artifact provenance.")
    remove_mask = (
        primary.prices["security_id"].astype(str).eq(ids.hwm)
        & primary.prices["session"].astype(str).ge(AA_CROSSCHECK_START)
        & primary.prices["session"].astype(str).le(AA_CROSSCHECK_END)
    )
    price_keys = dataset_spec("daily_price_raw").primary_key
    retained = primary.prices.loc[~remove_mask].copy()
    collisions = _frame_key_set(retained, price_keys) & _frame_key_set(
        supplement.prices, price_keys
    )
    if collisions:
        raise ValueError("WIKI/ARNC primary collides with retained published prices.")
    cross_collisions = _frame_key_set(primary.crosscheck_prices, price_keys) & _frame_key_set(
        old_aa, price_keys
    )
    if cross_collisions:
        raise ValueError("EODHD Old AA cross-check collides with existing evidence.")
    return FetchedHistories(
        prices=pd.concat((retained, supplement.prices), ignore_index=True),
        crosscheck_prices=pd.concat(
            (primary.crosscheck_prices, old_aa), ignore_index=True
        ),
        corporate_actions=primary.corporate_actions.copy(),
        artifacts=tuple((*primary.artifacts, *supplement.artifacts)),
        role_codes={**primary.role_codes, **supplement.role_codes},
        http_attempts=primary.http_attempts,
    )


def _coverage_missing(
    prices: pd.DataFrame,
    security_id: str,
    start: str,
    end: str,
    *,
    excluded_sessions: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    required = {"security_id", "session"}
    missing_columns = sorted(required - set(prices.columns))
    if missing_columns:
        raise ValueError(
            "Price coverage frame is missing columns: " + ", ".join(missing_columns)
        )
    actual = set(
        prices.loc[
            prices["security_id"].astype(str).eq(security_id), "session"
        ].astype(str)
    )
    return tuple(
        session
        for session in _expected_sessions(start, end)
        if session not in actual and session not in excluded_sessions
    )


def _valaris_expected_sessions() -> tuple[str, ...]:
    exchange_sessions = _expected_sessions(VALARIS_PRICE_START, VALARIS_PRICE_END)
    if VALARIS_DOCUMENTED_HALT_SESSIONS - set(exchange_sessions):
        raise RuntimeError("A documented Valaris halt date is outside the price window.")
    return tuple(
        session
        for session in exchange_sessions
        if session not in VALARIS_DOCUMENTED_HALT_SESSIONS
    )


def validate_valaris_fetched_history(
    fetched: FetchedHistories,
    ids: IdentityIds,
) -> dict[str, Any]:
    """Prove VALPQ is the exact legacy VAL primary history, not a ticker alias."""

    provider_code = str(fetched.role_codes.get("valaris") or "").strip()
    if provider_code != VALARIS_PROVIDER_CODE:
        raise ValueError(
            f"Legacy Valaris history must use {VALARIS_PROVIDER_CODE}, got "
            f"{provider_code or 'missing'}."
        )
    rows = fetched.prices.loc[
        fetched.prices["security_id"].astype(str).eq(ids.esv)
    ].copy()
    expected_sessions = _valaris_expected_sessions()
    actual_sessions = tuple(sorted(rows["session"].astype(str)))
    if actual_sessions != expected_sessions:
        raise ValueError(
            "Legacy Valaris VALPQ must have exact "
            f"{VALARIS_PRICE_START}..{VALARIS_PRICE_END} XNYS sessions: "
            f"expected={len(expected_sessions)}, actual={len(actual_sessions)}."
        )
    if set(rows["source"].astype(str)) != {"eodhd_eod"}:
        raise ValueError("Legacy Valaris VALPQ primary history lost EODHD provenance.")

    artifacts = []
    for artifact in fetched.artifacts:
        parsed = urlparse(artifact.source_url)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if (
            artifact.source == "eodhd_eod"
            and parsed.path.endswith(f"/eod/{VALARIS_PROVIDER_CODE}.US")
            and query == {
                "from": [VALARIS_PRICE_START],
                "to": [VALARIS_PRICE_END],
            }
        ):
            artifacts.append(artifact)
    if len(artifacts) != 1:
        raise ValueError(
            "Legacy Valaris VALPQ requires one exact bounded raw EODHD artifact."
        )
    artifact = artifacts[0]
    if (
        set(rows["source_url"].astype(str)) != {artifact.source_url}
        or set(rows["source_hash"].astype(str)) != {artifact.source_hash}
    ):
        raise ValueError(
            "Legacy Valaris VALPQ rows do not match the exact raw URL/hash artifact."
        )
    return {
        "provider_code": VALARIS_PROVIDER_CODE,
        "security_id": ids.esv,
        "start": VALARIS_PRICE_START,
        "end": VALARIS_PRICE_END,
        "session_count": len(expected_sessions),
        "missing_sessions": 0,
        "documented_halt_sessions": sorted(VALARIS_DOCUMENTED_HALT_SESSIONS),
        "halt_evidence_url": OFFICIAL_EVIDENCE["valaris_suspension"]["url"],
        "source_url": artifact.source_url,
        "source_sha256": artifact.source_hash,
    }


def _validate_price_crosscheck(
    primary_prices: pd.DataFrame,
    crosscheck_prices: pd.DataFrame,
    *,
    security_id: str,
    start: str,
    end: str,
    label: str,
    primary_source: str,
    crosscheck_source: str,
) -> dict[str, Any]:
    primary_gaps = _coverage_missing(primary_prices, security_id, start, end)
    cross_gaps = _coverage_missing(crosscheck_prices, security_id, start, end)
    if primary_gaps or cross_gaps:
        raise ValueError(
            f"{label} cross-check requires complete overlap sessions: "
            f"primary_missing={len(primary_gaps)}, crosscheck_missing={len(cross_gaps)}."
        )
    expected_sessions = _expected_sessions(start, end)
    left = primary_prices.loc[
        primary_prices["security_id"].astype(str).eq(security_id)
        & primary_prices["session"].astype(str).ge(start)
        & primary_prices["session"].astype(str).le(end),
        ["session", "close", "source"],
    ].sort_values("session")
    right = crosscheck_prices.loc[
        crosscheck_prices["security_id"].astype(str).eq(security_id)
        & crosscheck_prices["session"].astype(str).ge(start)
        & crosscheck_prices["session"].astype(str).le(end),
        ["session", "close", "source"],
    ].sort_values("session")
    if (
        tuple(left["session"].astype(str)) != expected_sessions
        or tuple(right["session"].astype(str)) != expected_sessions
    ):
        raise ValueError(
            f"{label} cross-check does not have exact one-to-one exchange sessions."
        )
    if set(left["source"].astype(str)) != {primary_source}:
        raise ValueError(f"{label} primary provenance changed.")
    if set(right["source"].astype(str)) != {crosscheck_source}:
        raise ValueError(f"{label} cross-check provenance changed.")
    joined = left.merge(
        right,
        on="session",
        suffixes=("_primary", "_crosscheck"),
        validate="one_to_one",
    )
    if len(joined) != len(expected_sessions):
        raise ValueError(f"{label} cross-check does not have one-to-one session overlap.")
    primary = pd.to_numeric(joined["close_primary"], errors="coerce")
    crosscheck = pd.to_numeric(joined["close_crosscheck"], errors="coerce")
    ratio = primary / crosscheck
    scale = float(ratio.median())
    normalized_error = (ratio / scale - 1.0).abs()
    return_corr = float(primary.pct_change().corr(crosscheck.pct_change()))
    p99 = float(normalized_error.quantile(0.99))
    if not (scale > 0 and return_corr >= 0.995 and p99 <= 0.05):
        raise ValueError(
            f"{label} scale-normalized cross-check failed: "
            f"scale={scale:.8f}, return_correlation={return_corr:.8f}, "
            f"p99_scaled_close_error={p99:.8f}."
        )
    return {
        "sessions": len(joined),
        "scale": scale,
        "return_correlation": return_corr,
        "p99_scaled_close_error": p99,
        "primary_source": primary_source,
        "crosscheck_source": crosscheck_source,
    }


def _validate_aa_raw_ohlc_crosscheck(
    published_prices: pd.DataFrame,
    crosscheck_prices: pd.DataFrame,
    *,
    security_id: str,
) -> dict[str, Any]:
    """Cross-check WIKI raw OHLC against EODHD raw OHLC and reject its volume basis."""

    expected = _expected_sessions(AA_CROSSCHECK_START, AA_CROSSCHECK_END)
    columns = ["session", "open", "high", "low", "close", "volume", "source"]
    primary = published_prices.loc[
        published_prices["security_id"].astype(str).eq(security_id)
        & published_prices["session"].astype(str).ge(AA_CROSSCHECK_START)
        & published_prices["session"].astype(str).le(AA_CROSSCHECK_END),
        columns,
    ].sort_values("session")
    crosscheck = crosscheck_prices.loc[
        crosscheck_prices["security_id"].astype(str).eq(security_id)
        & crosscheck_prices["session"].astype(str).ge(AA_CROSSCHECK_START)
        & crosscheck_prices["session"].astype(str).le(AA_CROSSCHECK_END),
        columns,
    ].sort_values("session")
    if (
        tuple(primary["session"].astype(str)) != expected
        or tuple(crosscheck["session"].astype(str)) != expected
    ):
        raise ValueError("Old AA raw cross-check requires exact one-to-one sessions.")
    if set(primary["source"].astype(str)) != {WIKI_ARNC_SOURCE}:
        raise ValueError("Old AA published primary must be WIKI raw OHLCV.")
    if set(crosscheck["source"].astype(str)) != {"eodhd_eod"}:
        raise ValueError("Old AA raw cross-check must retain EODHD provenance.")
    joined = primary.merge(
        crosscheck,
        on="session",
        suffixes=("_wiki", "_eodhd"),
        validate="one_to_one",
    )
    thresholds = {
        "open": {"return_corr": 0.9999, "p99_rel": 0.003, "max_rel": 0.01},
        "high": {"return_corr": 0.9999, "p99_rel": 0.003, "max_rel": 0.01},
        "low": {"return_corr": 0.9999, "p99_rel": 0.003, "max_rel": 0.01},
        "close": {
            "return_corr": 0.99999,
            "p99_rel": 0.00002,
            "max_abs": 0.0002,
        },
    }
    metrics: dict[str, Any] = {}
    for column, gate in thresholds.items():
        wiki = pd.to_numeric(joined[f"{column}_wiki"], errors="coerce")
        eodhd = pd.to_numeric(joined[f"{column}_eodhd"], errors="coerce")
        if wiki.isna().any() or eodhd.isna().any() or not bool(
            wiki.map(math.isfinite).all() and eodhd.map(math.isfinite).all()
        ):
            raise ValueError(f"Old AA {column} cross-check contains non-finite values.")
        absolute = (eodhd - wiki).abs()
        relative = absolute / wiki.abs()
        report = {
            "return_correlation": float(
                wiki.pct_change(fill_method=None).iloc[1:].corr(
                    eodhd.pct_change(fill_method=None).iloc[1:]
                )
            ),
            "value_correlation": float(wiki.corr(eodhd)),
            "median_eodhd_to_wiki_ratio": float((eodhd / wiki).median()),
            "max_absolute_error": float(absolute.max()),
            "p99_relative_error": float(relative.quantile(0.99)),
            "max_relative_error": float(relative.max()),
        }
        if (
            report["return_correlation"] < gate["return_corr"]
            or report["p99_relative_error"] > gate["p99_rel"]
            or (
                "max_rel" in gate
                and report["max_relative_error"] > gate["max_rel"]
            )
            or (
                "max_abs" in gate
                and report["max_absolute_error"] > gate["max_abs"]
            )
        ):
            raise ValueError(
                f"Old AA raw {column} cross-check failed: "
                + json.dumps(report, sort_keys=True)
            )
        metrics[column] = report

    wiki_volume = pd.to_numeric(joined["volume_wiki"], errors="coerce")
    eodhd_volume = pd.to_numeric(joined["volume_eodhd"], errors="coerce")
    if wiki_volume.isna().any() or eodhd_volume.isna().any():
        raise ValueError("Old AA volume comparison contains invalid values.")
    ratio = eodhd_volume / wiki_volume
    before = joined["session"].astype(str).lt("2016-10-06")
    after = ~before
    pre_median = float(ratio.loc[before].median())
    post_median = float(ratio.loc[after].median())
    basis_jump = post_median / pre_median
    volume_relative = (eodhd_volume - wiki_volume).abs() / wiki_volume
    volume = {
        "level_correlation": float(wiki_volume.corr(eodhd_volume)),
        "median_eodhd_to_wiki_ratio": float(ratio.median()),
        "pre_split_sessions": int(before.sum()),
        "pre_split_median_ratio": pre_median,
        "post_split_sessions": int(after.sum()),
        "post_split_median_ratio": post_median,
        "post_to_pre_basis_jump": basis_jump,
        "p99_relative_error": float(volume_relative.quantile(0.99)),
        "max_absolute_error": float((eodhd_volume - wiki_volume).abs().max()),
        "basis_mismatch_confirmed": True,
        "used_for_publication": False,
    }
    if not (
        0.40 <= pre_median <= 0.44
        and 1.22 <= post_median <= 1.28
        and 2.95 <= basis_jump <= 3.05
        and volume["p99_relative_error"] >= 0.50
    ):
        raise ValueError(
            "Old AA EODHD volume no longer exhibits the audited basis mismatch: "
            + json.dumps(volume, sort_keys=True)
        )
    return {
        "sessions": len(joined),
        "published_source": WIKI_ARNC_SOURCE,
        "published_price_basis": "unadjusted_ohlcv",
        "crosscheck_source": "eodhd_eod",
        "crosscheck_fields": ["open", "high", "low", "close"],
        "adjusted_close_used": False,
        "ohlc": metrics,
        "volume_mismatch": volume,
    }


def _validate_lila_external_crosscheck(
    primary_prices: pd.DataFrame,
    crosscheck_prices: pd.DataFrame,
    *,
    security_id: str,
    label: str,
) -> dict[str, Any]:
    report = _validate_price_crosscheck(
        primary_prices,
        crosscheck_prices,
        security_id=security_id,
        start=LILA_REGULAR_PRICE_START,
        end=LILA_EXTERNAL_CROSSCHECK_END,
        label=label,
        primary_source=YAHOO_LILA_PRIMARY_SOURCE,
        crosscheck_source=BORIS_KAGGLE_SOURCE,
    )
    total_sessions = _expected_sessions(
        LILA_REGULAR_PRICE_START,
        LILA_REGULAR_PRICE_END,
    )
    tail_sessions = tuple(
        session for session in total_sessions if session > LILA_EXTERNAL_CROSSCHECK_END
    )
    if len(total_sessions) != 630 or len(tail_sessions) != 33:
        raise RuntimeError("Frozen LILA/LILAK overlap/tail counts changed.")
    return {
        **report,
        "primary_sessions": len(total_sessions),
        "crosschecked_sessions": report["sessions"],
        "crosschecked_ratio": report["sessions"] / len(total_sessions),
        "uncrosschecked_tail_sessions": len(tail_sessions),
        "uncrosschecked_tail_start": tail_sessions[0],
        "uncrosschecked_tail_end": tail_sessions[-1],
        "external_artifact": True,
        "upstream_provider_disclosed": False,
        "independent_provider_claimed": False,
        "license": BORIS_KAGGLE_LICENSE,
        "license_url": BORIS_KAGGLE_VERSION_URL,
    }


def validate_fetched_histories(
    fetched: FetchedHistories,
    ids: IdentityIds,
    *,
    completed_session: str,
    require_old_aa: bool,
) -> dict[str, Any]:
    expected_roles = {
        "wyn",
        "spectra",
        "old_fox",
        "old_foxa",
        "old_lila",
        "old_lilak",
        "old_aa",
        "tnl",
        "valaris",
        "bhi",
        *REQUIRED_SUPPLEMENT_ROLES,
    }
    missing_roles = sorted(expected_roles - set(fetched.role_codes))
    if missing_roles:
        raise ValueError("Fetched identity roles are missing: " + ", ".join(missing_roles))
    if fetched.http_attempts != MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError(
            f"Fetched identity bundle call count must be {MAX_EODHD_HTTP_ATTEMPTS}."
        )
    valaris_validation = validate_valaris_fetched_history(fetched, ids)
    yahoo_validation = validate_yahoo_identity_supplement(
        FetchedHistories(
            prices=fetched.prices.loc[
                fetched.prices["security_id"].astype(str).isin(
                    {OLD_LILA_ID, OLD_LILAK_ID}
                )
                & fetched.prices["session"].astype(str).ge(
                    LILA_REGULAR_PRICE_START
                )
            ].copy(),
            crosscheck_prices=pd.DataFrame(
                columns=dataset_spec("daily_price_raw").required_columns
            ),
            corporate_actions=pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=tuple(
                artifact
                for artifact in fetched.artifacts
                if artifact.source_url
                in {_yahoo_chart_url(symbol) for symbol in YAHOO_SUPPLEMENT_SYMBOLS}
            ),
            role_codes={
                role: fetched.role_codes[role]
                for role in REQUIRED_YAHOO_ROLES
            },
            http_attempts=0,
        ),
        ids,
    )
    boris_validation = validate_boris_crosscheck_bundle(
        FetchedHistories(
            prices=pd.DataFrame(
                columns=dataset_spec("daily_price_raw").required_columns
            ),
            crosscheck_prices=fetched.crosscheck_prices.loc[
                fetched.crosscheck_prices["security_id"].astype(str).isin(
                    {OLD_LILA_ID, OLD_LILAK_ID}
                )
            ].copy(),
            corporate_actions=pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=tuple(
                artifact
                for artifact in fetched.artifacts
                if artifact.source_url
                in {value["url"] for value in BORIS_KAGGLE_FILES.values()}
            ),
            role_codes={
                role: fetched.role_codes[role] for role in REQUIRED_BORIS_ROLES
            },
            http_attempts=0,
        ),
        ids,
    )
    wiki_validation = validate_wiki_arnc_bundle(
        FetchedHistories(
            prices=fetched.prices.loc[
                fetched.prices["security_id"].astype(str).eq(ids.hwm)
                & fetched.prices["session"].astype(str).ge(AA_CROSSCHECK_START)
                & fetched.prices["session"].astype(str).le(AA_CROSSCHECK_END)
            ].copy(),
            crosscheck_prices=pd.DataFrame(
                columns=dataset_spec("daily_price_raw").required_columns
            ),
            corporate_actions=pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=tuple(
                artifact
                for artifact in fetched.artifacts
                if artifact.source_url == WIKI_ARNC_URL
            ),
            role_codes={WIKI_ARNC_ROLE: fetched.role_codes[WIKI_ARNC_ROLE]},
            http_attempts=0,
        ),
        ids,
    )
    windows = {
        "wyn": (ids.wynd, "2015-01-02", "2018-05-30"),
        "spectra": (SPECTRA_ID, "2015-01-02", "2017-02-24"),
        "old_fox": (OLD_FOX_ID, "2015-01-02", FOX_OLD_LAST),
        "old_foxa": (OLD_FOXA_ID, "2015-01-02", FOX_OLD_LAST),
        "old_lila": (
            OLD_LILA_ID,
            LILA_REGULAR_PRICE_START,
            LILA_REGULAR_PRICE_END,
        ),
        "old_lilak": (
            OLD_LILAK_ID,
            LILA_REGULAR_PRICE_START,
            LILA_REGULAR_PRICE_END,
        ),
        "tnl": (ids.wynd, "2021-02-17", completed_session),
        "valaris": (ids.esv, VALARIS_PRICE_START, VALARIS_PRICE_END),
        "bhi": (BHI_ID, "2015-01-02", "2017-07-03"),
    }
    missing: dict[str, tuple[str, ...]] = {}
    for role, (security_id, start, end) in windows.items():
        gaps = _coverage_missing(
            fetched.prices,
            security_id,
            start,
            end,
            excluded_sessions=(
                VALARIS_DOCUMENTED_HALT_SESSIONS
                if role == "valaris"
                else frozenset()
            ),
        )
        if gaps:
            missing[role] = gaps
    if require_old_aa:
        gaps = _coverage_missing(
            fetched.prices, ids.hwm, "2015-01-02", "2016-10-31"
        )
        if gaps:
            missing["old_aa"] = gaps
    if missing:
        detail = "; ".join(
            f"{role}={len(gaps)} missing ({gaps[0]}..{gaps[-1]})"
            for role, gaps in sorted(missing.items())
        )
        raise ValueError("Full historical identity coverage failed: " + detail)
    aa_artifact = _exact_eodhd_aa_artifact(fetched)
    eodhd_aa_rows = fetched.crosscheck_prices.loc[
        fetched.crosscheck_prices["security_id"].astype(str).eq(ids.hwm)
        & fetched.crosscheck_prices["session"].astype(str).ge(AA_CROSSCHECK_START)
        & fetched.crosscheck_prices["session"].astype(str).le(AA_CROSSCHECK_END)
    ]
    if not (
        set(eodhd_aa_rows["source_url"].astype(str)) == {aa_artifact.source_url}
        and set(eodhd_aa_rows["source_hash"].astype(str)) == {aa_artifact.source_hash}
    ):
        raise ValueError("Old AA EODHD cross-check lost exact artifact URL/hash.")
    aa_crosscheck = {
        **_validate_aa_raw_ohlc_crosscheck(
            fetched.prices,
            fetched.crosscheck_prices,
            security_id=ids.hwm,
        ),
        "eodhd_source_url": aa_artifact.source_url,
        "eodhd_source_sha256": aa_artifact.source_hash,
        "wiki_primary": wiki_validation,
    }
    lila_crosscheck = _validate_lila_external_crosscheck(
        fetched.prices,
        fetched.crosscheck_prices,
        security_id=OLD_LILA_ID,
        label="Old LILA regular-way",
    )
    lilak_crosscheck = _validate_lila_external_crosscheck(
        fetched.prices,
        fetched.crosscheck_prices,
        security_id=OLD_LILAK_ID,
        label="Old LILAK regular-way",
    )
    if not fetched.artifacts:
        raise ValueError("Fetched identity bundle has no raw source artifacts.")
    result = {
        role: {
            "provider_code": fetched.role_codes[role],
            "security_id": security_id,
            "start": start,
            "end": end,
            "missing_sessions": 0,
        }
        for role, (security_id, start, end) in windows.items()
    }
    result["old_aa_crosscheck"] = aa_crosscheck
    result["valaris_valpq_validation"] = valaris_validation
    result["old_lila_regular_primary_validation"] = yahoo_validation["lila"]
    result["old_lilak_regular_primary_validation"] = yahoo_validation["lilak"]
    result["old_lila_external_artifact_validation"] = boris_validation["lila"]
    result["old_lilak_external_artifact_validation"] = boris_validation["lilak"]
    result["old_lila_regular_boundary_validation"] = {
        "primary_source": YAHOO_LILA_PRIMARY_SOURCE,
        "external_crosscheck_source": BORIS_KAGGLE_SOURCE,
        "external_price_overlap": True,
        "independent_provider_claimed": False,
        "boundary_evidence": OFFICIAL_EVIDENCE["lila_nasdaq"]["url"],
        "effective_date": "2015-07-02",
        "price_crosscheck": lila_crosscheck,
    }
    result["old_lilak_regular_boundary_validation"] = {
        "primary_source": YAHOO_LILA_PRIMARY_SOURCE,
        "external_crosscheck_source": BORIS_KAGGLE_SOURCE,
        "external_price_overlap": True,
        "independent_provider_claimed": False,
        "boundary_evidence": OFFICIAL_EVIDENCE["lila_nasdaq"]["url"],
        "effective_date": "2015-07-02",
        "price_crosscheck": lilak_crosscheck,
    }
    return result


def build_offline_plan(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    preflight = build_local_preflight(repository, release)
    kind = preflight.catalogs["__kind__"]
    return {
        "status": "offline_plan",
        "eodhd_accessed": False,
        "yahoo_accessed": False,
        "boris_kaggle_accessed": False,
        "aa_wiki_accessed": False,
        "official_evidence_accessed": False,
        "eodhd_http_attempts_this_run": 0,
        "yahoo_http_attempts_this_run": 0,
        "boris_kaggle_http_attempts_this_run": 0,
        "aa_wiki_http_attempts_this_run": 0,
        "official_http_attempts_this_run": 0,
        "release_version": release.version,
        "completed_session": release.completed_session,
        "write_datasets": list(WRITE_DATASETS),
        "eodhd_price_probe_codes": list(PRICE_PROBE_CODES),
        "eodhd_price_probe_calls": len(PRICE_PROBE_CODES),
        "eodhd_selected_action_calls": 2 * SELECTED_ACTION_CODE_COUNT,
        "maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "successful_uncached_run_exact_eodhd_http_attempts": (
            MAX_EODHD_HTTP_ATTEMPTS
        ),
        "maximum_yahoo_http_attempts": MAX_YAHOO_HTTP_ATTEMPTS,
        "maximum_boris_kaggle_http_attempts": MAX_BORIS_HTTP_ATTEMPTS,
        "maximum_aa_wiki_http_attempts": MAX_WIKI_ARNC_HTTP_ATTEMPTS,
        "maximum_official_http_attempts": MAX_OFFICIAL_HTTP_ATTEMPTS,
        "network_opt_in_flags": {
            "yahoo": "--fetch-yahoo-supplement",
            "boris_kaggle": "--fetch-boris-crosscheck",
            "aa_wiki": "--fetch-aa-wiki-crosscheck",
            "official_documents": "--fetch-official-evidence",
        },
        "successor_snapshot_required_warning": PENDING_IDENTITY_WARNING,
        "successor_snapshot_ready": PENDING_IDENTITY_WARNING in set(release.warnings),
        "cache_path": str(_bundle_cache_path(repository, release)),
        "cache_exists": _bundle_cache_path(repository, release).is_file(),
        "boris_kaggle_cache_paths": {
            symbol: str(
                BorisKaggleCache(
                    repository.root / "state/boris-kaggle-us-index-identity"
                ).path(symbol)
            )
            for symbol in BORIS_KAGGLE_FILES
        },
        "aa_wiki_cache": {
            "blob_path": str(
                WikiArncPinnedCache(
                    repository.root / "state/wiki-arnc-us-index-identity"
                ).path
            ),
            "metadata_path": str(
                WikiArncPinnedCache(
                    repository.root / "state/wiki-arnc-us-index-identity"
                ).metadata_path
            ),
            "source_url": WIKI_ARNC_URL,
            "commit": WIKI_ARNC_COMMIT,
            "full_sha256": WIKI_ARNC_FULL_SHA256,
            "full_size": WIKI_ARNC_FULL_SIZE,
            "full_data_rows": WIKI_ARNC_FULL_DATA_ROWS,
            "full_blob_archived_on_apply": True,
        },
        "catalog_candidates": {
            code: {
                "kind": kind[code.upper()],
                "name": preflight.catalogs[code.upper()].get("Name"),
                "isin": preflight.catalogs[code.upper()].get("Isin"),
            }
            for code in PRICE_PROBE_CODES
        },
        "local_only_repairs": [
            "AGN/ACT",
            "ABC/COR",
            "BHI/BHGE/BKR",
            "IR/TT",
            "ARNC/HWM-from-2016",
            "HOT/MAR boundary trim",
            "AZN_old exact duplicate -> AZN",
        ],
        "provider_required_roles": [
            "WYN/WYND",
            "Spectra Energy SE",
            "old 21CF FOX",
            "old 21CF FOXA",
            "old Liberty LILA",
            "old Liberty LILAK",
            "TNL continuation",
            "ESV/VAL via VALPQ continuation",
            "BHI predecessor",
        ],
        "required_cross_sources": [
            {
                "role": "old Alcoa Inc AA before 2016-11-01",
                "required_range": "2015-01-02..2016-10-31",
                "reason": (
                    "The commit-pinned WIKI ARNC raw segment owns publication because "
                    "it preserves Old Alcoa raw OHLCV. The bounded EODHD AA response "
                    "is retained only for raw OHLC comparison; its stitched volume is "
                    "explicitly rejected from publication."
                ),
                "eodhd_probe_code": "AA",
                "published_primary_url": WIKI_ARNC_URL,
                "published_primary_sha256": WIKI_ARNC_FULL_SHA256,
                "published_segment_sha256": WIKI_ARNC_SEGMENT_SHA256,
                "published_segment_rows": WIKI_ARNC_SEGMENT_ROWS,
                "eodhd_crosscheck_fields": ["open", "high", "low", "close"],
                "eodhd_volume_used_for_publication": False,
                "blocks_apply_if_either_source_fails": True,
            },
            {
                "role": "old Liberty Global LILA/LILAK tracking shares",
                "required_range": "2015-07-02..2017-12-29",
                "reason": (
                    "EODHD LILAV/LILKV preserve the when-issued evidence but expose "
                    "no regular-way history. Exact Yahoo LILA/LILAK artifacts provide "
                    "630 primary sessions; pinned Kaggle CC0 files externally "
                    "cross-check 597 sessions, with the 33-session tail disclosed."
                ),
                "eodhd_when_issued_probe_codes": ["LILAV", "LILKV"],
                "yahoo_primary_codes": ["LILA", "LILAK"],
                "external_crosscheck_codes": list(BORIS_KAGGLE_FILES),
                "external_crosscheck_end": LILA_EXTERNAL_CROSSCHECK_END,
                "uncrosschecked_tail_sessions": 33,
                "yahoo_urls": [
                    _yahoo_chart_url("LILA"),
                    _yahoo_chart_url("LILAK"),
                ],
                "blocks_apply_if_either_source_fails": True,
            },
            {
                "role": "legacy Valaris VAL identity",
                "required_price_range": (
                    f"{VALARIS_PRICE_START}..{VALARIS_PRICE_END}"
                ),
                "official_cancellation_date": VALARIS_CANCELLATION_DATE,
                "reason": (
                    "The delisted VALPQ endpoint must provide every primary price "
                    "session, while the later legal identity end must bind to the "
                    "exact archived SEC cancellation URL/hash."
                ),
                "eodhd_probe_code": VALARIS_PROVIDER_CODE,
                "official_url": OFFICIAL_EVIDENCE["valaris_emergence"]["url"],
                "blocks_apply_if_either_source_fails": True,
            },
        ],
        "supplement_required": (
            "immutable Yahoo chart artifacts, pinned Kaggle CC0 overlap files, and "
            "the full commit-pinned WIKI/ARNC blob; "
            "Yahoo primary ownership is restricted to the exact old LILA/LILAK "
            "2015-07-02..2017-12-29 intervals"
        ),
        "official_evidence_urls": {
            key: value["url"] for key, value in OFFICIAL_EVIDENCE.items()
        },
        "partial_apply_assessment": {
            "supported_by_current_collector": False,
            "current_behavior": (
                "all repair roles are validated and rewritten in one atomic candidate; "
                "a LILA or LILAK failure blocks every other repair"
            ),
            "safe_design_requirements": [
                "declare immutable repair cohorts and their security/index dependencies",
                "rewrite only one dependency-closed cohort into a full snapshot",
                "run the unchanged dataset, repository, replay, provenance, and full-history gates",
                "retain a precise pending warning for every unapplied cohort",
                "commit each validated cohort atomically and restrict Yahoo primary prices to the audited LILA/LILAK window",
            ],
            "unsafe_shortcut_rejected": (
                "do not skip LILA/LILAK checks while clearing the aggregate pending warning"
            ),
        },
    }


def _official_evidence_artifact(*, retrieved_at: str) -> SourceArtifact:
    content = json.dumps(
        {
            "collector": "collect_us_index_identity_repairs",
            "evidence": OFFICIAL_EVIDENCE,
            "retrieved_at": retrieved_at,
            "semantics": (
                "Reviewed official facts used to cut provider ticker histories into "
                "legal security identities; this manifest is not a cached copy of filings."
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SourceArtifact(
        source="official_us_identity_evidence_manifest",
        source_url="local://us-index-identity-repairs/official-evidence-v1",
        retrieved_at=retrieved_at,
        content=content,
        content_type="application/json",
    )


def _metadata(
    evidence: SourceArtifact | OfficialEvidenceBundle,
    *,
    source_url: str,
) -> dict[str, Any]:
    source_hash = (
        evidence.source_hash_for(source_url)
        if isinstance(evidence, OfficialEvidenceBundle)
        else evidence.source_hash
    )
    return {
        "source": "official_identity_repair",
        "source_url": source_url,
        "retrieved_at": evidence.retrieved_at,
        "source_hash": source_hash,
    }


def _master_row(
    template: pd.Series,
    *,
    security_id: str,
    symbol: str,
    provider_symbol: str,
    name: str,
    active_from: str,
    active_to: str,
    source_url: str,
    evidence: SourceArtifact,
    exchange: str | None = None,
) -> dict[str, Any]:
    row = template.to_dict()
    row.update(
        {
            "security_id": security_id,
            "primary_symbol": symbol,
            "provider_symbol": provider_symbol,
            "action_provider_symbol": provider_symbol,
            "name": name,
            "active_from": active_from,
            "active_to": active_to,
            **_metadata(evidence, source_url=source_url),
        }
    )
    if exchange:
        row["exchange"] = exchange
    return row


def _history_rows(
    *,
    security_id: str,
    exchange: str,
    intervals: Iterable[tuple[str, str, str, str]],
    evidence: SourceArtifact,
) -> list[dict[str, Any]]:
    return [
        {
            "security_id": security_id,
            "symbol": symbol,
            "exchange": exchange,
            "effective_from": start,
            "effective_to": end,
            **_metadata(evidence, source_url=source_url),
        }
        for symbol, start, end, source_url in intervals
    ]


def rewrite_security_identities(
    master: pd.DataFrame,
    history: pd.DataFrame,
    *,
    ids: IdentityIds,
    role_codes: dict[str, str],
    evidence: SourceArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    drop_ids = {
        ids.abc_duplicate,
        ids.coresite_duplicate,
        ids.arnc_duplicate,
        ids.bhge,
        ids.azn_duplicate,
    }
    affected = {
        ids.agn_legacy,
        ids.agn_actavis,
        ids.cor,
        ids.fox,
        ids.foxa,
        ids.wynd,
        ids.sea,
        ids.lila,
        ids.lilak,
        ids.bhge,
        ids.bkr,
        ids.ir,
        ids.tt,
        ids.arnc,
        ids.hwm,
        ids.hot,
        ids.esv,
        OLD_FOX_ID,
        OLD_FOXA_ID,
        SPECTRA_ID,
        OLD_LILA_ID,
        OLD_LILAK_ID,
        BHI_ID,
    }
    by_id = {
        str(row.security_id): pd.Series(row._asdict())
        for row in master.itertuples(index=False)
    }
    for security_id in (
        ids.agn_legacy,
        ids.agn_actavis,
        ids.cor,
        ids.fox,
        ids.foxa,
        ids.wynd,
        ids.sea,
        ids.lila,
        ids.lilak,
        ids.bhge,
        ids.bkr,
        ids.ir,
        ids.tt,
        ids.arnc,
        ids.hwm,
        ids.hot,
        ids.esv,
    ):
        if security_id not in by_id:
            raise ValueError(f"Identity repair master template is missing: {security_id}")

    urls = {key: str(value["url"]) for key, value in OFFICIAL_EVIDENCE.items()}
    rows: list[dict[str, Any]] = []
    rows.extend(
        [
            _master_row(
                by_id[ids.agn_legacy],
                security_id=ids.agn_legacy,
                symbol="AGN",
                provider_symbol="AGN_old.US",
                name="Allergan Inc",
                active_from="2015-01-02",
                active_to="2015-03-22",
                source_url=urls["agn"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.agn_actavis],
                security_id=ids.agn_actavis,
                symbol="AGN",
                provider_symbol="AGN.US",
                name="Allergan plc (formerly Actavis plc)",
                active_from="2015-01-02",
                active_to="2020-05-11",
                source_url=urls["agn_abbvie"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.cor],
                security_id=ids.cor,
                symbol="COR",
                provider_symbol="COR.US",
                name="Cencora Inc (formerly AmerisourceBergen)",
                active_from="2015-01-02",
                active_to="",
                source_url=urls["cor"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.fox],
                security_id=ids.fox,
                symbol="FOX",
                provider_symbol="FOX.US",
                name="Fox Corporation Class B",
                active_from="2019-03-12",
                active_to="",
                source_url=urls["fox"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.foxa],
                security_id=ids.foxa,
                symbol="FOXA",
                provider_symbol="FOXA.US",
                name="Fox Corporation Class A",
                active_from="2019-03-12",
                active_to="",
                source_url=urls["fox"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.fox],
                security_id=OLD_FOX_ID,
                symbol="TFCF",
                provider_symbol=f"{role_codes['old_fox']}.US",
                name="Twenty-First Century Fox Inc Class B",
                active_from="2015-01-02",
                active_to="2019-03-19",
                source_url=urls["fox_index"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.foxa],
                security_id=OLD_FOXA_ID,
                symbol="TFCFA",
                provider_symbol=f"{role_codes['old_foxa']}.US",
                name="Twenty-First Century Fox Inc Class A",
                active_from="2015-01-02",
                active_to="2019-03-19",
                source_url=urls["fox_index"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.wynd],
                security_id=ids.wynd,
                symbol="TNL",
                provider_symbol="TNL.US",
                name="Travel + Leisure Co (formerly Wyndham Destinations)",
                active_from="2015-01-02",
                active_to="",
                source_url=urls["tnl"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.sea],
                security_id=ids.sea,
                symbol="SE",
                provider_symbol="SE.US",
                name="Sea Ltd",
                active_from="2017-10-20",
                active_to="",
                source_url=urls["se"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.sea],
                security_id=SPECTRA_ID,
                symbol="SE",
                provider_symbol="SE1.US",
                name="Spectra Energy Corp",
                active_from="2015-01-02",
                active_to="2017-02-27",
                source_url=urls["se"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.lila],
                security_id=ids.lila,
                symbol="LILA",
                provider_symbol="LILA.US",
                name="Liberty Latin America Ltd Class A",
                active_from="2018-01-02",
                active_to="",
                source_url=urls["lila_splitoff"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.lilak],
                security_id=ids.lilak,
                symbol="LILAK",
                provider_symbol="LILAK.US",
                name="Liberty Latin America Ltd Class C",
                active_from="2018-01-02",
                active_to="",
                source_url=urls["lila_splitoff"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.lila],
                security_id=OLD_LILA_ID,
                symbol="LILA",
                provider_symbol="LILAV.US",
                name="Liberty Global LiLAC Tracking Share Class A",
                active_from=LILA_REGULAR_PRICE_START,
                active_to="2017-12-29",
                source_url=urls["lila_nasdaq"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.lilak],
                security_id=OLD_LILAK_ID,
                symbol="LILAK",
                provider_symbol="LILKV.US",
                name="Liberty Global LiLAC Tracking Share Class C",
                active_from=LILA_REGULAR_PRICE_START,
                active_to="2017-12-29",
                source_url=urls["lila_nasdaq"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.bkr],
                security_id=BHI_ID,
                symbol="BHI",
                provider_symbol="BHI.US",
                name="Baker Hughes Incorporated",
                active_from="2015-01-02",
                active_to="2017-07-03",
                source_url=urls["bhge"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.bkr],
                security_id=ids.bkr,
                symbol="BKR",
                provider_symbol="BKR.US",
                name="Baker Hughes Company (formerly BHGE)",
                active_from="2017-07-05",
                active_to="",
                source_url=urls["bkr"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.tt],
                security_id=ids.tt,
                symbol="TT",
                provider_symbol="TT.US",
                name="Trane Technologies plc (formerly Ingersoll-Rand plc)",
                active_from="2015-01-02",
                active_to="",
                source_url=urls["ir"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.ir],
                security_id=ids.ir,
                symbol="IR",
                provider_symbol="IR.US",
                name="Ingersoll Rand Inc (formerly Gardner Denver)",
                active_from="2017-05-12",
                active_to="",
                source_url=urls["ir_tickers"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.hwm],
                security_id=ids.hwm,
                symbol="HWM",
                provider_symbol="HWM.US",
                name="Howmet Aerospace Inc (old Alcoa/Arconic lineage)",
                active_from="2015-01-02",
                active_to="",
                source_url=urls["old_alcoa_arnc_identity"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.arnc],
                security_id=ids.arnc,
                symbol="ARNC",
                provider_symbol="ARNC.US",
                name="Arconic Corporation (2020 spin company)",
                active_from=ARNC_SEPARATION,
                active_to="2023-08-17",
                source_url=urls["arnc"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.hot],
                security_id=ids.hot,
                symbol="HOT",
                provider_symbol="HOT.US",
                name="Starwood Hotels & Resorts Worldwide Inc",
                active_from="2015-01-02",
                active_to="2016-09-22",
                source_url=urls["hot"],
                evidence=evidence,
            ),
            _master_row(
                by_id[ids.esv],
                security_id=ids.esv,
                symbol="VAL",
                provider_symbol=f"{VALARIS_PROVIDER_CODE}.US",
                name="Valaris plc (formerly Ensco plc)",
                active_from="2015-01-02",
                active_to=VALARIS_CANCELLATION_DATE,
                source_url=urls["valaris_emergence"],
                evidence=evidence,
            ),
        ]
    )

    output_master = master.loc[
        ~master["security_id"].astype(str).isin(affected | drop_ids)
    ].copy()
    output_master = _concat_unique(
        (output_master, pd.DataFrame(rows)),
        keys=dataset_spec("security_master").primary_key,
    )
    exchange = {
        str(row.security_id): str(row.exchange)
        for row in output_master.itertuples(index=False)
    }
    intervals: dict[str, tuple[tuple[str, str, str, str], ...]] = {
        ids.agn_legacy: (("AGN", "2015-01-01", "2015-03-22", urls["agn"]),),
        ids.agn_actavis: (
            ("ACT", "2015-01-01", "2015-06-14", urls["agn_terms"]),
            ("AGN", "2015-06-15", "2020-05-11", urls["agn_abbvie"]),
        ),
        ids.cor: (
            ("ABC", "2015-01-01", "2023-08-29", urls["cor"]),
            ("COR", COR_CHANGE, "", urls["cor"]),
        ),
        OLD_FOX_ID: (
            ("FOX", "2015-01-01", "2019-03-18", urls["fox_index"]),
            ("TFCF", "2019-03-19", "2019-03-19", urls["fox_index"]),
        ),
        OLD_FOXA_ID: (
            ("FOXA", "2015-01-01", "2019-03-18", urls["fox_index"]),
            ("TFCFA", "2019-03-19", "2019-03-19", urls["fox_index"]),
        ),
        ids.fox: (("FOX", "2019-03-12", "", urls["fox"]),),
        ids.foxa: (("FOXA", "2019-03-12", "", urls["fox"]),),
        ids.wynd: (
            ("WYN", "2015-01-01", "2018-05-31", urls["wynd"]),
            ("WYND", "2018-06-01", "2021-02-16", urls["wynd"]),
            ("TNL", "2021-02-17", "", urls["tnl"]),
        ),
        SPECTRA_ID: (("SE", "2015-01-01", "2017-02-26", urls["se"]),),
        ids.sea: (("SE", "2017-10-20", "", urls["se"]),),
        OLD_LILA_ID: (
            (
                "LILA",
                LILA_REGULAR_PRICE_START,
                LILA_REGULAR_PRICE_END,
                urls["lila_nasdaq"],
            ),
        ),
        OLD_LILAK_ID: (
            (
                "LILAK",
                LILA_REGULAR_PRICE_START,
                LILA_REGULAR_PRICE_END,
                urls["lila_nasdaq"],
            ),
        ),
        ids.lila: (("LILA", "2018-01-02", "", urls["lila_splitoff"]),),
        ids.lilak: (("LILAK", "2018-01-02", "", urls["lila_splitoff"]),),
        BHI_ID: (("BHI", "2015-01-01", "2017-07-03", urls["bhge"]),),
        ids.bkr: (
            ("BHGE", "2017-07-05", "2019-10-17", urls["bhge"]),
            ("BKR", "2019-10-18", "", urls["bkr"]),
        ),
        ids.tt: (
            ("IR", "2015-01-01", "2020-02-28", urls["ir"]),
            ("TT", "2020-03-02", "", urls["ir"]),
        ),
        ids.ir: (
            ("GDI", "2017-05-12", "2020-02-28", urls["ir_tickers"]),
            ("IR", "2020-03-02", "", urls["ir_tickers"]),
        ),
        ids.hwm: (
            (
                "AA",
                "2015-01-01",
                "2016-10-31",
                urls["old_alcoa_arnc_identity"],
            ),
            (
                "ARNC",
                "2016-11-01",
                "2020-03-31",
                urls["old_alcoa_arnc_identity"],
            ),
            ("HWM", ARNC_SEPARATION, "", urls["arnc_index"]),
        ),
        ids.arnc: (("ARNC", ARNC_SEPARATION, "2023-08-17", urls["arnc"]),),
        ids.hot: (("HOT", "2015-01-01", "2016-09-22", urls["hot"]),),
        ids.esv: (
            ("ESV", "2015-01-01", "2019-07-30", urls["valaris"]),
            (
                "VAL",
                VALARIS_PRICE_START,
                VALARIS_CANCELLATION_DATE,
                urls["valaris_emergence"],
            ),
        ),
    }
    output_history = history.loc[
        ~history["security_id"].astype(str).isin(affected | drop_ids)
    ].copy()
    new_history = [
        row
        for security_id, values in intervals.items()
        for row in _history_rows(
            security_id=security_id,
            exchange=exchange[security_id],
            intervals=values,
            evidence=evidence,
        )
    ]
    output_history = _concat_unique(
        (output_history, pd.DataFrame(new_history)),
        keys=dataset_spec("symbol_history").primary_key,
    )
    return (
        output_master.reset_index(drop=True),
        output_history.reset_index(drop=True),
        {
            "new_security_ids": {
                "old_fox": OLD_FOX_ID,
                "old_foxa": OLD_FOXA_ID,
                "spectra": SPECTRA_ID,
                "old_lila": OLD_LILA_ID,
                "old_lilak": OLD_LILAK_ID,
                "bhi": BHI_ID,
            },
            "removed_duplicate_security_ids": sorted(drop_ids),
        },
    )


def _date_slice(
    frame: pd.DataFrame,
    security_id: str,
    date_column: str,
    *,
    start: str = "",
    end: str = "",
) -> pd.DataFrame:
    mask = frame["security_id"].astype(str).eq(security_id)
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if start:
        mask &= dates.ge(pd.Timestamp(start))
    if end:
        mask &= dates.le(pd.Timestamp(end))
    return frame.loc[mask].copy()


def _official_action(
    *,
    security_id: str,
    action_type: str,
    effective_date: str,
    source_url: str,
    evidence: SourceArtifact,
    new_security_id: str = "",
    new_symbol: str = "",
    cash_amount: float | None = None,
    ratio: float | None = None,
) -> dict[str, Any]:
    return {
        "event_id": _event_id(
            "official_identity_repair",
            security_id,
            action_type,
            effective_date,
            new_security_id,
            new_symbol,
            cash_amount,
            ratio,
        ),
        "security_id": security_id,
        "action_type": action_type,
        "effective_date": effective_date,
        "ex_date": effective_date,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": cash_amount,
        "ratio": ratio,
        "currency": "USD",
        "new_security_id": new_security_id,
        "new_symbol": new_symbol,
        "official": True,
        "source": "official_identity_repair",
        "source_url": source_url,
        "source_kind": "official_filing",
        "retrieved_at": evidence.retrieved_at,
        "source_hash": (
            evidence.source_hash_for(source_url)
            if isinstance(evidence, OfficialEvidenceBundle)
            else evidence.source_hash
        ),
    }


def rewrite_prices_and_actions(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    fetched: FetchedHistories,
    *,
    ids: IdentityIds,
    completed_session: str,
    evidence: SourceArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], dict[str, Any]]:
    affected = {
        ids.agn_legacy,
        ids.agn_actavis,
        ids.abc_duplicate,
        ids.cor,
        ids.coresite_duplicate,
        ids.fox,
        ids.foxa,
        ids.wynd,
        ids.sea,
        ids.lila,
        ids.lilak,
        ids.bhge,
        ids.bkr,
        ids.ir,
        ids.tt,
        ids.arnc,
        ids.arnc_duplicate,
        ids.hwm,
        ids.hot,
        ids.esv,
        ids.azn,
        ids.azn_duplicate,
        OLD_FOX_ID,
        OLD_FOXA_ID,
        SPECTRA_ID,
        OLD_LILA_ID,
        OLD_LILAK_ID,
        BHI_ID,
    }
    keep_prices = prices.loc[
        ~prices["security_id"].astype(str).isin(affected)
    ].copy()
    fetched_prices = fetched.prices.copy()
    fetched_prices = fetched_prices.loc[
        pd.to_datetime(fetched_prices["session"], errors="coerce").le(
            pd.Timestamp(completed_session)
        )
    ]
    when_issued_mask = (
        fetched_prices["security_id"].astype(str).isin(
            {OLD_LILA_ID, OLD_LILAK_ID}
        )
        & fetched_prices["session"].astype(str).lt(LILA_REGULAR_PRICE_START)
    )
    when_issued_rows_archived_only = int(when_issued_mask.sum())
    fetched_prices = fetched_prices.loc[~when_issued_mask].copy()
    # EODHD retains one synthetic 2019-03-20 bar for each retired 21CF share
    # class, repeating the official final 2019-03-19 close after the old
    # securities had ceased trading.  Keep the exact raw response in
    # source_archive, but never publish a price beyond the reviewed identity
    # boundary.
    fox_terminal_overrun = (
        fetched_prices["security_id"].astype(str).isin(
            {OLD_FOX_ID, OLD_FOXA_ID}
        )
        & fetched_prices["session"].astype(str).gt(FOX_OLD_LAST)
    )
    fox_terminal_overrun_rows_archived_only = int(fox_terminal_overrun.sum())
    fox_roles_present = {"old_fox", "old_foxa"}.issubset(fetched.role_codes)
    if fox_roles_present and fox_terminal_overrun_rows_archived_only != 2:
        raise ValueError(
            "Old 21CF provider terminal overrun inventory changed: "
            f"expected=2, actual={fox_terminal_overrun_rows_archived_only}."
        )
    if not fox_roles_present and fox_terminal_overrun_rows_archived_only:
        raise ValueError("Old 21CF terminal rows appeared without bound identity roles.")
    fetched_prices = fetched_prices.loc[~fox_terminal_overrun].copy()
    old_aa_published = fetched_prices.loc[
        fetched_prices["security_id"].astype(str).eq(ids.hwm)
        & fetched_prices["session"].astype(str).le(AA_CROSSCHECK_END)
    ]
    if not old_aa_published.empty and set(
        old_aa_published["source"].astype(str)
    ) != {WIKI_ARNC_SOURCE}:
        raise ValueError("Old AA publication must use WIKI raw OHLCV exclusively.")

    price_pieces = [
        keep_prices,
        _date_slice(prices, ids.agn_legacy, "session"),
        _date_slice(prices, ids.agn_actavis, "session"),
        _date_slice(prices, ids.cor, "session"),
        _date_slice(prices, ids.fox, "session", start="2019-03-12"),
        _date_slice(prices, ids.foxa, "session", start="2019-03-12"),
        _date_slice(prices, ids.wynd, "session", start="2018-06-01", end="2021-02-16"),
        _date_slice(prices, ids.sea, "session", start="2017-10-20"),
        _date_slice(prices, ids.lila, "session", start="2018-01-02"),
        _date_slice(prices, ids.lilak, "session", start="2018-01-02"),
        _date_slice(prices, ids.bkr, "session", start="2017-07-05"),
        _date_slice(prices, ids.ir, "session", start="2017-05-12"),
        _date_slice(prices, ids.tt, "session", start="2015-01-02"),
        _date_slice(prices, ids.hwm, "session", start="2016-11-01"),
        _date_slice(prices, ids.arnc, "session", start=ARNC_SEPARATION),
        _date_slice(prices, ids.hot, "session", end="2016-09-22"),
        _date_slice(prices, ids.esv, "session", end="2019-07-30"),
        _date_slice(prices, ids.azn, "session"),
        fetched_prices,
    ]
    output_prices = _concat_unique(
        price_pieces, keys=dataset_spec("daily_price_raw").primary_key
    ).reset_index(drop=True)

    keep_actions = actions.loc[
        ~actions["security_id"].astype(str).isin(affected)
    ].copy()
    fetched_actions = fetched.corporate_actions.copy()
    replaced_old_aa_provider_splits = fetched_actions[
        fetched_actions["security_id"].astype(str).eq(ids.hwm)
        & fetched_actions["action_type"].astype(str).eq("split")
        & fetched_actions["effective_date"].astype(str).eq("2016-10-06")
    ]
    fetched_actions = fetched_actions.drop(index=replaced_old_aa_provider_splits.index)
    action_pieces = [
        keep_actions,
        _date_slice(actions, ids.agn_legacy, "effective_date"),
        _date_slice(actions, ids.agn_actavis, "effective_date"),
        _date_slice(actions, ids.cor, "effective_date"),
        _date_slice(actions, ids.fox, "effective_date", start="2019-03-12"),
        _date_slice(actions, ids.foxa, "effective_date", start="2019-03-12"),
        _date_slice(actions, ids.wynd, "effective_date", start="2018-06-01", end="2021-02-16"),
        _date_slice(actions, ids.sea, "effective_date", start="2017-10-20"),
        _date_slice(actions, ids.lila, "effective_date", start="2018-01-02"),
        _date_slice(actions, ids.lilak, "effective_date", start="2018-01-02"),
        _date_slice(actions, ids.bkr, "effective_date", start="2017-07-05"),
        _date_slice(actions, ids.ir, "effective_date", start="2017-05-12"),
        _date_slice(actions, ids.tt, "effective_date", start="2015-01-02"),
        _date_slice(actions, ids.hwm, "effective_date", start="2016-11-01"),
        _date_slice(actions, ids.arnc, "effective_date", start=ARNC_SEPARATION),
        _date_slice(actions, ids.hot, "effective_date", end="2016-09-22"),
        _date_slice(actions, ids.esv, "effective_date", end="2019-07-30"),
        _date_slice(actions, ids.azn, "effective_date"),
        fetched_actions,
    ]
    urls = {key: str(value["url"]) for key, value in OFFICIAL_EVIDENCE.items()}
    official = pd.DataFrame(
        [
            _official_action(
                security_id=ids.agn_legacy,
                action_type="stock_merger",
                effective_date="2015-03-17",
                source_url=urls["agn_terms"],
                evidence=evidence,
                new_security_id=ids.agn_actavis,
                new_symbol="ACT",
                cash_amount=129.22,
                ratio=0.3683,
            ),
            _official_action(
                security_id=ids.agn_actavis,
                action_type="ticker_change",
                effective_date="2015-06-15",
                source_url=urls["agn_terms"],
                evidence=evidence,
                new_security_id=ids.agn_actavis,
                new_symbol="AGN",
            ),
            _official_action(
                security_id=ids.cor,
                action_type="ticker_change",
                effective_date=COR_CHANGE,
                source_url=urls["cor"],
                evidence=evidence,
                new_security_id=ids.cor,
                new_symbol="COR",
            ),
            _official_action(
                security_id=OLD_FOX_ID,
                action_type="spinoff",
                effective_date=FOX_TRANSITION,
                source_url=urls["fox"],
                evidence=evidence,
                new_security_id=ids.fox,
                new_symbol="FOX",
                ratio=1 / 3,
            ),
            _official_action(
                security_id=OLD_FOXA_ID,
                action_type="spinoff",
                effective_date=FOX_TRANSITION,
                source_url=urls["fox"],
                evidence=evidence,
                new_security_id=ids.foxa,
                new_symbol="FOXA",
                ratio=1 / 3,
            ),
            _official_action(
                security_id=ids.wynd,
                action_type="ticker_change",
                effective_date="2018-06-01",
                source_url=urls["wynd"],
                evidence=evidence,
                new_security_id=ids.wynd,
                new_symbol="WYND",
            ),
            _official_action(
                security_id=ids.wynd,
                action_type="ticker_change",
                effective_date="2021-02-17",
                source_url=urls["tnl"],
                evidence=evidence,
                new_security_id=ids.wynd,
                new_symbol="TNL",
            ),
            _official_action(
                security_id=BHI_ID,
                action_type="stock_merger",
                effective_date="2017-07-05",
                source_url=urls["bhge"],
                evidence=evidence,
                new_security_id=ids.bkr,
                new_symbol="BHGE",
                cash_amount=17.50,
                ratio=1.0,
            ),
            _official_action(
                security_id=ids.bkr,
                action_type="ticker_change",
                effective_date="2019-10-18",
                source_url=urls["bkr"],
                evidence=evidence,
                new_security_id=ids.bkr,
                new_symbol="BKR",
            ),
            _official_action(
                security_id=ids.tt,
                action_type="ticker_change",
                effective_date="2020-03-02",
                source_url=urls["ir"],
                evidence=evidence,
                new_security_id=ids.tt,
                new_symbol="TT",
            ),
            _official_action(
                security_id=ids.ir,
                action_type="ticker_change",
                effective_date="2020-03-02",
                source_url=urls["ir_tickers"],
                evidence=evidence,
                new_security_id=ids.ir,
                new_symbol="IR",
            ),
            _official_action(
                security_id=ids.hwm,
                action_type="split",
                effective_date="2016-10-06",
                source_url=urls["old_alcoa_arnc_identity"],
                evidence=evidence,
                ratio=1 / 3,
            ),
            _official_action(
                security_id=ids.hwm,
                action_type="ticker_change",
                effective_date="2016-11-01",
                source_url=urls["old_alcoa_arnc_identity"],
                evidence=evidence,
                new_security_id=ids.hwm,
                new_symbol="ARNC",
            ),
            _official_action(
                security_id=ids.hwm,
                action_type="spinoff",
                effective_date="2016-11-01",
                source_url=urls["alcoa_2016_separation"],
                evidence=evidence,
                new_symbol="AA",
                ratio=1 / 3,
            ),
            _official_action(
                security_id=ids.hwm,
                action_type="spinoff",
                effective_date=ARNC_SEPARATION,
                source_url=urls["arnc"],
                evidence=evidence,
                new_security_id=ids.arnc,
                new_symbol="ARNC",
                ratio=0.25,
            ),
            _official_action(
                security_id=ids.esv,
                action_type="ticker_change",
                effective_date="2019-07-31",
                source_url=urls["valaris"],
                evidence=evidence,
                new_security_id=ids.esv,
                new_symbol="VAL",
            ),
            _official_action(
                security_id=ids.hot,
                action_type="stock_merger",
                effective_date="2016-09-23",
                source_url=urls["hot"],
                evidence=evidence,
                new_security_id=ids.mar,
                new_symbol="MAR",
                cash_amount=21.0,
                ratio=0.8,
            ),
            # Legacy VAL holders received warrant consideration in the 2021
            # reorganization.  A fabricated zero-cash delisting would corrupt
            # economic returns, so the lifecycle finalizer must remain fail-closed
            # until exact official recovery terms are reviewed.
        ],
        columns=dataset_spec("corporate_actions").required_columns,
    )
    output_actions = _concat_unique(
        (*action_pieces, official),
        keys=dataset_spec("corporate_actions").primary_key,
    ).reset_index(drop=True)
    return (
        output_prices,
        output_actions,
        affected,
        {
            "price_rows_before": len(prices),
            "price_rows_after": len(output_prices),
            "action_rows_before": len(actions),
            "action_rows_after": len(output_actions),
            "hot_rows_trimmed": int(
                len(_date_slice(prices, ids.hot, "session", start="2016-09-23"))
            ),
            "lila_lilak_when_issued_price_rows_archived_only": (
                when_issued_rows_archived_only
            ),
            "old_fox_terminal_overrun_rows_archived_only": (
                fox_terminal_overrun_rows_archived_only
            ),
            "azn_duplicate_price_rows_removed": int(
                len(_date_slice(prices, ids.azn_duplicate, "session"))
            ),
            "azn_duplicate_action_rows_removed": int(
                len(_date_slice(actions, ids.azn_duplicate, "effective_date"))
            ),
            "legacy_valaris_2021_outcome": VALARIS_2021_OUTCOME_STATUS,
            "old_aa_provider_split_rows_replaced_by_official": int(
                len(replaced_old_aa_provider_splits)
            ),
        },
    )


def rebuild_affected_factors(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    affected_ids: set[str],
    evidence: SourceArtifact,
) -> pd.DataFrame:
    retained = factors.loc[
        ~factors["security_id"].astype(str).isin(affected_ids)
    ].copy()
    repaired_prices = prices.loc[
        prices["security_id"].astype(str).isin(affected_ids)
    ].copy()
    repaired_actions = actions.loc[
        actions["security_id"].astype(str).isin(affected_ids)
    ].copy()
    rebuilt = build_adjustment_factors(
        repaired_prices,
        repaired_actions,
        source_version=f"identity-repair:{evidence.source_hash}",
    )
    return _concat_unique(
        (retained, rebuilt),
        keys=dataset_spec("adjustment_factors").primary_key,
    ).reset_index(drop=True)


def _identity_event_row(
    template: dict[str, Any],
    *,
    index_id: str,
    effective_date: str,
    operation: str,
    security_id: str,
    source_url: str,
    evidence: SourceArtifact,
    official: bool,
    source_kind: str,
    announcement_date: str = "",
) -> dict[str, Any]:
    row = dict(template)
    row.update(
        {
            "event_id": _event_id(
                "official_identity_repair",
                index_id,
                effective_date,
                operation,
                security_id,
            ),
            "index_id": index_id,
            "announcement_date": announcement_date,
            "effective_date": effective_date,
            "operation": operation,
            "security_id": security_id,
            "official": official,
            "source": "official_identity_repair",
            "source_url": source_url,
            "source_kind": source_kind,
            "retrieved_at": evidence.retrieved_at,
            "source_hash": (
                evidence.source_hash_for(source_url)
                if isinstance(evidence, OfficialEvidenceBundle)
                else evidence.source_hash
            ),
        }
    )
    return row


def rewrite_index_references(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    *,
    ids: IdentityIds,
    evidence: SourceArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    urls = {key: str(value["url"]) for key, value in OFFICIAL_EVIDENCE.items()}
    anchor_map = {
        ids.abc_duplicate: ids.cor,
        ids.fox: OLD_FOX_ID,
        ids.foxa: OLD_FOXA_ID,
        ids.sea: SPECTRA_ID,
        ids.bhge: BHI_ID,
        ids.ir: ids.tt,
        ids.arnc: ids.hwm,
    }
    output_anchors = anchors.copy()
    original_anchor_ids = output_anchors["security_id"].astype(str)
    changed_anchor_rows = output_anchors["security_id"].astype(str).isin(anchor_map)
    output_anchors.loc[changed_anchor_rows, "security_id"] = output_anchors.loc[
        changed_anchor_rows, "security_id"
    ].astype(str).map(anchor_map)
    output_anchors.loc[changed_anchor_rows, "source"] = "official_identity_repair"
    anchor_urls = {
        ids.abc_duplicate: urls["cor"],
        ids.fox: urls["fox_nasdaq"],
        ids.foxa: urls["fox_nasdaq"],
        ids.sea: urls["se"],
        ids.bhge: urls["bhge"],
        ids.ir: urls["ir"],
        ids.arnc: urls["old_alcoa_arnc_identity"],
    }
    output_anchors.loc[changed_anchor_rows, "source_url"] = original_anchor_ids.loc[
        changed_anchor_rows
    ].map(anchor_urls)
    output_anchors.loc[changed_anchor_rows, "source_kind"] = (
        "community_membership_official_identity"
    )
    output_anchors.loc[changed_anchor_rows, "retrieved_at"] = evidence.retrieved_at
    changed_urls = output_anchors.loc[changed_anchor_rows, "source_url"].astype(str)
    output_anchors.loc[changed_anchor_rows, "source_hash"] = changed_urls.map(
        lambda url: evidence.source_hash_for(url)
        if isinstance(evidence, OfficialEvidenceBundle)
        else evidence.source_hash
    )

    # AZN_old is an exact provider duplicate.  If a future bootstrap adds an
    # anchor for it, preserve the original membership evidence and only repair
    # the identity key; this is not an official corporate boundary.
    azn_anchor_rows = output_anchors["security_id"].astype(str).eq(
        ids.azn_duplicate
    )
    output_anchors.loc[azn_anchor_rows, "security_id"] = ids.azn

    legacy_anchor = output_anchors.loc[
        output_anchors["security_id"].astype(str).eq(ids.agn_legacy)
        & output_anchors["index_id"].astype(str).eq("sp500")
    ]
    if len(legacy_anchor) != 1:
        raise ValueError("AGN repair requires exactly one legacy S&P 500 anchor row.")
    actavis_anchor = legacy_anchor.iloc[0].to_dict()
    actavis_anchor.update(
        {
            "security_id": ids.agn_actavis,
            "official": False,
            "source": "official_identity_repair",
            "source_url": urls["agn"],
            "source_kind": "community_membership_official_identity",
            "retrieved_at": evidence.retrieved_at,
            "source_hash": (
                evidence.source_hash_for(urls["agn"])
                if isinstance(evidence, OfficialEvidenceBundle)
                else evidence.source_hash
            ),
        }
    )
    output_anchors = _concat_unique(
        (output_anchors, pd.DataFrame([actavis_anchor])),
        keys=dataset_spec("index_constituent_anchors").primary_key,
    ).reset_index(drop=True)

    template = events.iloc[0].to_dict()
    working = events.copy()
    event_dates = pd.to_datetime(working["effective_date"], errors="coerce")
    security = working["security_id"].astype(str)
    index_id = working["index_id"].astype(str)

    # Canonical ticker changes are composition-neutral.  Both sides of each
    # community REMOVE/ADD pair must disappear or replay will break continuity.
    drop = (
        security.isin({ids.abc_duplicate, ids.coresite_duplicate})
        & event_dates.eq(pd.Timestamp(COR_CHANGE))
    )
    drop |= security.isin({ids.bhge, ids.bkr}) & event_dates.eq(
        pd.Timestamp("2019-10-18")
    )
    drop |= security.isin({ids.arnc, ids.hwm}) & event_dates.eq(
        pd.Timestamp("2020-04-06")
    )
    output_events = working.loc[~drop].copy()

    # Preserve the community Nasdaq membership evidence while moving the two
    # AZN ADD/REMOVE rows from the duplicate provider ID to canonical AZN.US.
    # Regenerate event_id because security_id participates in event identity.
    azn_event_rows = output_events["security_id"].astype(str).eq(
        ids.azn_duplicate
    )
    if azn_event_rows.any():
        output_events.loc[azn_event_rows, "security_id"] = ids.azn
        output_events.loc[azn_event_rows, "event_id"] = [
            _event_id(
                "provider_identity_dedup",
                str(row.index_id),
                str(row.effective_date),
                str(row.operation).upper(),
                ids.azn,
                str(row.source_hash),
            )
            for row in output_events.loc[azn_event_rows].itertuples(index=False)
        ]

    # Existing rows whose operation/date are valid but identity was normalized
    # to a later issuer are remapped and assigned a new deterministic event_id.
    remaps: list[tuple[pd.Series, str, str]] = []
    for idx, row in output_events.iterrows():
        sid = str(row["security_id"])
        date = str(row["effective_date"])
        target = ""
        source_url = ""
        if sid == ids.agn_legacy and date == "2020-05-12":
            target, source_url = ids.agn_actavis, urls["agn_abbvie"]
        elif sid == ids.fox and date == "2015-09-21":
            target, source_url = OLD_FOX_ID, urls["fox_index"]
        elif sid == ids.sea and date <= "2017-02-27":
            target, source_url = SPECTRA_ID, urls["se"]
        elif sid == ids.lila and date < "2018-01-02":
            target, source_url = OLD_LILA_ID, urls["lila_nasdaq"]
        elif sid == ids.lilak and date < "2018-01-02":
            target, source_url = OLD_LILAK_ID, urls["lila_nasdaq"]
        elif sid == ids.tt and date == "2020-03-03":
            target, source_url = ids.ir, urls["ir"]
        if target:
            remaps.append((row, target, source_url))
            output_events = output_events.drop(index=idx)
    remapped_rows = [
        _identity_event_row(
            row.to_dict(),
            index_id=str(row["index_id"]),
            effective_date=str(row["effective_date"]),
            operation=str(row["operation"]).upper(),
            security_id=target,
            source_url=source_url,
            evidence=evidence,
            official=bool(row.get("official", False)),
            source_kind="community_membership_official_identity",
            announcement_date=str(row.get("announcement_date") or ""),
        )
        for row, target, source_url in remaps
    ]

    additions = [
        _identity_event_row(
            template,
            index_id="sp500",
            effective_date="2015-03-23",
            operation="REMOVE",
            security_id=ids.agn_legacy,
            source_url=urls["agn"],
            evidence=evidence,
            official=True,
            source_kind="official_index_notice",
            announcement_date="2015-03-16",
        ),
        *[
            _identity_event_row(
                template,
                index_id="sp500",
                effective_date="2019-03-19",
                operation="ADD",
                security_id=sid,
                source_url=urls["fox_index"],
                evidence=evidence,
                official=True,
                source_kind="official_index_notice",
                announcement_date="2019-03-14",
            )
            for sid in (ids.foxa, ids.fox)
        ],
        *[
            _identity_event_row(
                template,
                index_id="sp500",
                effective_date="2019-03-20",
                operation="REMOVE",
                security_id=sid,
                source_url=urls["fox_index"],
                evidence=evidence,
                official=True,
                source_kind="official_index_notice",
                announcement_date="2019-03-14",
            )
            for sid in (OLD_FOXA_ID, OLD_FOX_ID)
        ],
        *[
            _identity_event_row(
                template,
                index_id="nasdaq100",
                effective_date="2019-03-19",
                operation=operation,
                security_id=sid,
                source_url=urls["fox_nasdaq"],
                evidence=evidence,
                official=False,
                source_kind="derived_membership_official_identity",
            )
            for operation, sid in (
                ("REMOVE", OLD_FOXA_ID),
                ("REMOVE", OLD_FOX_ID),
                ("ADD", ids.foxa),
                ("ADD", ids.fox),
            )
        ],
        _identity_event_row(
            template,
            index_id="sp500",
            effective_date="2017-07-05",
            operation="REMOVE",
            security_id=BHI_ID,
            source_url=urls["bhge"],
            evidence=evidence,
            official=False,
            source_kind="derived_issuer_transition",
        ),
        _identity_event_row(
            template,
            index_id="sp500",
            effective_date="2017-07-05",
            operation="ADD",
            security_id=ids.bkr,
            source_url=urls["bhge"],
            evidence=evidence,
            official=False,
            source_kind="derived_issuer_transition",
        ),
    ]
    output_events = _concat_unique(
        (output_events, pd.DataFrame(remapped_rows), pd.DataFrame(additions)),
        keys=dataset_spec("index_membership_events").primary_key,
    ).reset_index(drop=True)

    # Semantic duplicates are forbidden even if their source event_ids differ.
    semantic = ["index_id", "effective_date", "operation", "security_id"]
    output_events = output_events.drop_duplicates(semantic, keep="last").reset_index(
        drop=True
    )
    sp_anchor = output_anchors.loc[
        output_anchors["index_id"].astype(str).eq("sp500")
    ]
    first_anchor = pd.to_datetime(sp_anchor["anchor_date"]).min()
    first_count = int(
        pd.to_datetime(sp_anchor["anchor_date"]).eq(first_anchor).sum()
    )
    if first_count != 500:
        raise ValueError(
            f"AGN two-seat repair must restore 500 S&P anchor rows, found {first_count}."
        )
    return (
        output_anchors,
        output_events,
        {
            "anchor_rows_remapped": int(changed_anchor_rows.sum()),
            "actavis_anchor_added": 1,
            "events_removed_as_ticker_continuity": int(drop.sum()),
            "events_identity_remapped": len(remapped_rows),
            "azn_anchor_rows_identity_remapped": int(azn_anchor_rows.sum()),
            "azn_events_identity_remapped": int(azn_event_rows.sum()),
            "events_added": len(additions),
            "sp500_initial_anchor_count": first_count,
        },
    )


def _artifact_extension(artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower()
    if "json" in content_type:
        return "json"
    if "csv" in content_type:
        return "csv"
    if "pdf" in content_type:
        return "pdf"
    if "html" in content_type:
        return "html"
    return "bin"


def append_source_archive(
    source_archive: pd.DataFrame,
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
        (source_archive, pd.DataFrame(rows)),
        keys=dataset_spec("source_archive").primary_key,
    ).reset_index(drop=True)


def rewrite_candidate_frames(
    preflight: LocalPreflight,
    fetched: FetchedHistories,
    *,
    release: DataRelease,
    evidence: OfficialEvidenceBundle,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], tuple[SourceArtifact, ...]]:
    existing = preflight.existing
    master, history, identity_stats = rewrite_security_identities(
        existing["security_master"],
        existing["symbol_history"],
        ids=preflight.ids,
        role_codes=fetched.role_codes,
        evidence=evidence,
    )
    prices, actions, affected_ids, market_stats = rewrite_prices_and_actions(
        existing["daily_price_raw"],
        existing["corporate_actions"],
        fetched,
        ids=preflight.ids,
        completed_session=release.completed_session,
        evidence=evidence,
    )
    factors = rebuild_affected_factors(
        existing["adjustment_factors"],
        prices,
        actions,
        affected_ids=affected_ids,
        evidence=evidence,
    )
    anchors, events, index_stats = rewrite_index_references(
        existing["index_constituent_anchors"],
        existing["index_membership_events"],
        ids=preflight.ids,
        evidence=evidence,
    )
    artifacts = tuple(
        dict.fromkeys(
            (*fetched.artifacts, *evidence.raw_artifacts, evidence.manifest)
        )
    )
    archive = append_source_archive(
        existing["source_archive"],
        artifacts,
        completed_session=release.completed_session,
    )
    frames = {
        **existing,
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": archive,
    }
    return frames, {
        "identity": identity_stats,
        "market": market_stats,
        "index": index_stats,
        "affected_security_count": len(affected_ids),
        "raw_artifact_count": len(artifacts),
    }, artifacts


class _FrameRepository:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def current_manifest(self, dataset: str):
        return object() if dataset in self.frames else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        return self.frames[dataset].copy()


def _active_history(
    history: pd.DataFrame,
    security_id: str,
    date: pd.Timestamp,
) -> bool:
    rows = history.loc[history["security_id"].astype(str).eq(security_id)]
    starts = pd.to_datetime(rows["effective_from"], errors="coerce")
    ends = pd.to_datetime(rows["effective_to"], errors="coerce")
    return bool((starts.le(date) & (ends.isna() | ends.ge(date))).any())


def validate_replay_gate(
    frames: dict[str, pd.DataFrame],
    *,
    completed_session: str,
    ids: IdentityIds,
) -> dict[str, Any]:
    anchors = frames["index_constituent_anchors"]
    events = frames["index_membership_events"]
    history = frames["symbol_history"]
    replayer = IndexEventReplayer(anchors, events)
    checked = 0
    for index_id in ("sp500", "nasdaq100"):
        index_anchor = anchors.loc[anchors["index_id"].astype(str).eq(index_id)]
        start = pd.to_datetime(index_anchor["anchor_date"], errors="coerce").min()
        dates = {pd.Timestamp(start), pd.Timestamp(completed_session)}
        dates.update(
            pd.to_datetime(
                events.loc[
                    events["index_id"].astype(str).eq(index_id), "effective_date"
                ],
                errors="coerce",
            ).dropna()
        )
        for date in sorted(dates):
            membership = replayer.members_on(index_id, date)
            if membership.warnings:
                raise ValueError(
                    f"Index replay warnings remain for {index_id} on {date.date()}: "
                    + "; ".join(membership.warnings)
                )
            missing = [
                security_id
                for security_id in membership.security_ids
                if not _active_history(history, security_id, pd.Timestamp(date))
            ]
            if missing:
                raise ValueError(
                    f"Index replay has no active symbol for {index_id} on {date.date()}: "
                    + ", ".join(missing)
                )
            checked += len(membership.security_ids)

    count_mar20 = len(replayer.members_on("sp500", "2015-03-20").security_ids)
    count_mar23 = len(replayer.members_on("sp500", "2015-03-23").security_ids)
    if count_mar20 != count_mar23:
        raise ValueError(
            "AGN/AAL close-to-open replacement changed S&P membership cardinality: "
            f"2015-03-20={count_mar20}, 2015-03-23={count_mar23}."
        )
    continuity = {
        "cor": (ids.cor, "2023-08-29", "2023-08-30"),
        "bkr": (ids.bkr, "2019-10-17", "2019-10-18"),
        "hwm": (ids.hwm, "2020-03-31", "2020-04-06"),
    }
    for label, (security_id, before, after) in continuity.items():
        before_members = set(replayer.members_on("sp500", before).security_ids)
        after_members = set(replayer.members_on("sp500", after).security_ids)
        if security_id not in before_members or security_id not in after_members:
            raise ValueError(f"Canonical {label} identity lost index position continuity.")
    return {
        "member_snapshots_checked": checked,
        "agn_count_before": count_mar20,
        "agn_count_after": count_mar23,
        "canonical_position_continuity": sorted(continuity),
    }


def validate_full_history_gate(
    frames: dict[str, pd.DataFrame],
    *,
    completed_session: str,
    ids: IdentityIds,
) -> dict[str, Any]:
    prices = frames["daily_price_raw"]
    windows = {
        "cov": (COV_SECURITY_ID, "2015-01-02", "2015-01-26"),
        "agn_legacy": (ids.agn_legacy, "2015-01-02", "2015-03-16"),
        "agn_actavis": (ids.agn_actavis, "2015-01-02", "2020-05-08"),
        "cor": (ids.cor, "2015-01-02", completed_session),
        "old_fox": (OLD_FOX_ID, "2015-01-02", FOX_OLD_LAST),
        "old_foxa": (OLD_FOXA_ID, "2015-01-02", FOX_OLD_LAST),
        "new_fox": (ids.fox, FOX_TRANSITION, completed_session),
        "new_foxa": (ids.foxa, FOX_TRANSITION, completed_session),
        "wyn_wynd_tnl": (ids.wynd, "2015-01-02", completed_session),
        "spectra": (SPECTRA_ID, "2015-01-02", "2017-02-24"),
        "sea": (ids.sea, "2017-10-20", completed_session),
        "old_lila": (
            OLD_LILA_ID,
            LILA_REGULAR_PRICE_START,
            LILA_REGULAR_PRICE_END,
        ),
        "old_lilak": (
            OLD_LILAK_ID,
            LILA_REGULAR_PRICE_START,
            LILA_REGULAR_PRICE_END,
        ),
        "new_lila": (ids.lila, "2018-01-02", completed_session),
        "new_lilak": (ids.lilak, "2018-01-02", completed_session),
        "bhi": (BHI_ID, "2015-01-02", "2017-07-03"),
        "bkr": (ids.bkr, "2017-07-05", completed_session),
        "old_ir_tt": (ids.tt, "2015-01-02", completed_session),
        "gdi_ir": (ids.ir, "2017-05-12", completed_session),
        "aa_arnc_hwm": (ids.hwm, "2015-01-02", completed_session),
        "new_arnc": (ids.arnc, ARNC_SEPARATION, "2023-08-17"),
        "hot": (ids.hot, "2015-01-02", "2016-09-22"),
        "esv_val": (ids.esv, "2015-01-02", VALARIS_PRICE_END),
        "azn": (ids.azn, "2015-01-02", completed_session),
    }
    summary: dict[str, Any] = {}
    for label, (security_id, start, end) in windows.items():
        gaps = _coverage_missing(
            prices,
            security_id,
            start,
            end,
            excluded_sessions=(
                VALARIS_DOCUMENTED_HALT_SESSIONS
                if label == "esv_val"
                else frozenset()
            ),
        )
        if gaps:
            raise ValueError(
                f"Full-history gate failed for {label}/{security_id}: "
                f"{len(gaps)} sessions missing ({gaps[0]}..{gaps[-1]})."
            )
        summary[label] = {
            "security_id": security_id,
            "start": start,
            "end": end,
            "missing_sessions": 0,
        }

    duplicate_survivors: dict[str, int] = {}
    for dataset, frame in frames.items():
        if "security_id" not in frame.columns:
            continue
        count = int(
            frame["security_id"].astype(str).eq(ids.azn_duplicate).sum()
        )
        if count:
            duplicate_survivors[dataset] = count
    if duplicate_survivors:
        raise ValueError(
            "AZN_old duplicate survived canonicalization: "
            + ", ".join(
                f"{dataset}={count}"
                for dataset, count in sorted(duplicate_survivors.items())
            )
        )
    return summary


def validate_boundary_provenance(frames: dict[str, pd.DataFrame]) -> None:
    archive = frames["source_archive"]
    archived_pairs = set(
        zip(archive["source_url"].astype(str), archive["source_hash"].astype(str))
    )
    for dataset in (
        "security_master",
        "symbol_history",
        "corporate_actions",
        "index_constituent_anchors",
        "index_membership_events",
    ):
        frame = frames[dataset]
        repaired = frame.loc[frame["source"].astype(str).eq("official_identity_repair")]
        for row in repaired.itertuples(index=False):
            pair = (str(row.source_url), str(row.source_hash))
            if pair not in archived_pairs:
                raise ValueError(
                    f"{dataset} boundary row lacks exact raw URL/hash archive: {pair[0]}"
                )


def validate_valaris_cancellation_boundary(
    frames: dict[str, pd.DataFrame],
    *,
    ids: IdentityIds,
) -> dict[str, Any]:
    """Bind the last VALPQ price to the later official legacy-share cancellation."""

    official = OFFICIAL_EVIDENCE["valaris_emergence"]
    official_url = str(official["url"])
    official_date = str(official["facts"].get("legacy_shares_cancelled") or "")
    if official_date != VALARIS_CANCELLATION_DATE:
        raise ValueError("Reviewed Valaris cancellation fact changed unexpectedly.")

    master = frames["security_master"]
    master_rows = master.loc[master["security_id"].astype(str).eq(ids.esv)]
    if len(master_rows) != 1:
        raise ValueError("Legacy Valaris requires exactly one security_master identity.")
    master_row = master_rows.iloc[0]
    evidence_hash = str(master_row.get("source_hash") or "")
    if not (
        str(master_row.get("provider_symbol") or "")
        == f"{VALARIS_PROVIDER_CODE}.US"
        and str(master_row.get("primary_symbol") or "") == "VAL"
        and str(master_row.get("active_to") or "") == VALARIS_CANCELLATION_DATE
        and str(master_row.get("source_url") or "") == official_url
        and len(evidence_hash) == 64
    ):
        raise ValueError(
            "Legacy Valaris master does not bind VALPQ to the official cancellation boundary."
        )

    history = frames["symbol_history"]
    history_rows = history.loc[
        history["security_id"].astype(str).eq(ids.esv)
        & history["symbol"].astype(str).eq("VAL")
    ]
    if len(history_rows) != 1:
        raise ValueError("Legacy Valaris requires exactly one VAL symbol interval.")
    history_row = history_rows.iloc[0]
    if not (
        str(history_row.get("effective_from") or "") == VALARIS_PRICE_START
        and str(history_row.get("effective_to") or "")
        == VALARIS_CANCELLATION_DATE
        and str(history_row.get("source_url") or "") == official_url
        and str(history_row.get("source_hash") or "") == evidence_hash
    ):
        raise ValueError(
            "Legacy VAL symbol interval lacks the exact official cancellation boundary."
        )

    prices = frames["daily_price_raw"]
    identity_prices = prices.loc[
        prices["security_id"].astype(str).eq(ids.esv)
    ].copy()
    sessions = identity_prices["session"].astype(str)
    valpq_sessions = tuple(
        sorted(
            sessions.loc[
                sessions.ge(VALARIS_PRICE_START)
            ]
        )
    )
    expected_sessions = _valaris_expected_sessions()
    if valpq_sessions != expected_sessions:
        raise ValueError(
            "Legacy Valaris candidate does not preserve the exact VALPQ price window "
            f"{VALARIS_PRICE_START}..{VALARIS_PRICE_END}."
        )

    archive = frames["source_archive"]
    archived = archive.loc[
        archive["source_url"].astype(str).eq(official_url)
        & archive["source_hash"].astype(str).eq(evidence_hash)
    ]
    if not (
        len(archived) == 1
        and str(archived.iloc[0].get("archive_id") or "") == evidence_hash
    ):
        raise ValueError(
            "Legacy Valaris lacks the exact official cancellation URL/hash archive."
        )
    return {
        "security_id": ids.esv,
        "provider_symbol": f"{VALARIS_PROVIDER_CODE}.US",
        "price_start": VALARIS_PRICE_START,
        "last_price_session": VALARIS_PRICE_END,
        "price_session_count": len(expected_sessions),
        "official_cancellation_date": VALARIS_CANCELLATION_DATE,
        "official_evidence_url": official_url,
        "official_evidence_sha256": evidence_hash,
        "economic_outcome": VALARIS_2021_OUTCOME_STATUS,
    }


def validate_old_alcoa_identity_boundary(
    frames: dict[str, pd.DataFrame],
    *,
    ids: IdentityIds,
) -> dict[str, Any]:
    identity_url = OFFICIAL_EVIDENCE["old_alcoa_arnc_identity"]["url"]
    separation_url = OFFICIAL_EVIDENCE["alcoa_2016_separation"]["url"]
    master = frames["security_master"]
    master_rows = master.loc[master["security_id"].astype(str).eq(ids.hwm)]
    if len(master_rows) != 1 or str(master_rows.iloc[0]["source_url"]) != identity_url:
        raise ValueError("Old Alcoa/ARNC/HWM master lacks exact 2016 issuer evidence.")
    history = frames["symbol_history"].loc[
        frames["symbol_history"]["security_id"].astype(str).eq(ids.hwm)
    ]
    observed = {
        (
            str(row.symbol),
            str(row.effective_from),
            str(row.effective_to),
            str(row.source_url),
        )
        for row in history.itertuples(index=False)
    }
    expected = {
        ("AA", "2015-01-01", "2016-10-31", identity_url),
        ("ARNC", "2016-11-01", "2020-03-31", identity_url),
        ("HWM", ARNC_SEPARATION, "", OFFICIAL_EVIDENCE["arnc_index"]["url"]),
    }
    if observed != expected:
        raise ValueError("Old Alcoa AA -> ARNC -> HWM symbol history is not exact.")
    actions = frames["corporate_actions"].loc[
        frames["corporate_actions"]["security_id"].astype(str).eq(ids.hwm)
        & frames["corporate_actions"]["source"].astype(str).eq(
            "official_identity_repair"
        )
    ]
    split = actions.loc[
        actions["action_type"].astype(str).eq("split")
        & actions["effective_date"].astype(str).eq("2016-10-06")
    ]
    ticker = actions.loc[
        actions["action_type"].astype(str).eq("ticker_change")
        & actions["effective_date"].astype(str).eq("2016-11-01")
    ]
    new_aa = actions.loc[
        actions["action_type"].astype(str).eq("spinoff")
        & actions["effective_date"].astype(str).eq("2016-11-01")
    ]
    if not (
        len(split) == 1
        and math.isclose(float(split.iloc[0]["ratio"]), 1 / 3, abs_tol=1e-15)
        and str(split.iloc[0]["source_url"]) == identity_url
    ):
        raise ValueError("Old Alcoa official 2016 reverse split is not exact.")
    if not (
        len(ticker) == 1
        and str(ticker.iloc[0]["new_security_id"]) == ids.hwm
        and str(ticker.iloc[0]["new_symbol"]) == "ARNC"
        and str(ticker.iloc[0]["source_url"]) == identity_url
    ):
        raise ValueError("Old Alcoa AA -> ARNC same-issuer action is not exact.")
    if not (
        len(new_aa) == 1
        and not str(new_aa.iloc[0]["new_security_id"] or "").strip()
        and str(new_aa.iloc[0]["new_symbol"]) == "AA"
        and math.isclose(float(new_aa.iloc[0]["ratio"]), 1 / 3, abs_tol=1e-15)
        and str(new_aa.iloc[0]["source_url"]) == separation_url
    ):
        raise ValueError("New Alcoa AA is not encoded as a separate distributed security.")
    prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"].astype(str).eq(ids.hwm)
        & frames["daily_price_raw"]["session"].astype(str).ge(AA_CROSSCHECK_START)
        & frames["daily_price_raw"]["session"].astype(str).le(AA_CROSSCHECK_END)
    ].sort_values("session")
    if not (
        len(prices) == WIKI_ARNC_SEGMENT_ROWS
        and set(prices["source"].astype(str)) == {WIKI_ARNC_SOURCE}
        and set(prices["source_url"].astype(str)) == {WIKI_ARNC_URL}
        and set(prices["source_hash"].astype(str)) == {WIKI_ARNC_FULL_SHA256}
        and math.isclose(float(prices.iloc[-1]["close"]), 28.72, abs_tol=1e-12)
    ):
        raise ValueError("Published Old Alcoa prices are not exact WIKI raw OHLCV.")
    archive = frames["source_archive"]
    full_blob = archive.loc[
        archive["source_url"].astype(str).eq(WIKI_ARNC_URL)
        & archive["source_hash"].astype(str).eq(WIKI_ARNC_FULL_SHA256)
        & archive["archive_id"].astype(str).eq(WIKI_ARNC_FULL_SHA256)
    ]
    if len(full_blob) != 1:
        raise ValueError("WIKI/ARNC full blob is not scheduled in source_archive.")
    return {
        "security_id": ids.hwm,
        "same_issuer_ticker_change": "AA->ARNC on 2016-11-01",
        "new_aa_separate_security": True,
        "reverse_split_session": "2016-10-06",
        "reverse_split_ratio": 1 / 3,
        "published_sessions": len(prices),
        "published_source": WIKI_ARNC_SOURCE,
        "published_source_sha256": WIKI_ARNC_FULL_SHA256,
        "official_identity_url": identity_url,
        "official_separation_url": separation_url,
        "full_blob_archived_on_apply": True,
    }


def validate_candidate_frames(
    frames: dict[str, pd.DataFrame],
    *,
    completed_session: str,
    ids: IdentityIds,
) -> tuple[tuple[str, ...], dict[str, Any]]:
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
    repository = _FrameRepository(frames)
    strict = validate_repository_snapshot(repository)
    strict.raise_for_errors()
    warnings.extend(
        issue.message for issue in strict.issues if issue.severity != "error"
    )
    validate_boundary_provenance(frames)
    valaris_boundary = validate_valaris_cancellation_boundary(frames, ids=ids)
    old_alcoa_boundary = validate_old_alcoa_identity_boundary(frames, ids=ids)
    replay = validate_replay_gate(
        frames, completed_session=completed_session, ids=ids
    )
    full_history = validate_full_history_gate(
        frames, completed_session=completed_session, ids=ids
    )
    return tuple(dict.fromkeys(warnings)), {
        "strict_index_identity_validation": (
            "valid_with_wiki_raw_aa_and_disclosed_597_of_630_lila_overlap"
        ),
        "replay": replay,
        "full_history": full_history,
        "valaris_cancellation_boundary": valaris_boundary,
        "old_alcoa_identity_boundary": old_alcoa_boundary,
        "r2_identity_gate_ready": True,
        "deferred_lifecycle_outcomes": {
            "legacy_valaris_2021_04_30": VALARIS_2021_OUTCOME_STATUS,
        },
        "next_required_step": "run lifecycle finalizer and strict publisher validation",
    }


def _assert_release_unchanged(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
) -> None:
    current, etag = repository.current_release()
    if current is None or current.version != release.version or etag != release_etag:
        raise RuntimeError("Current release changed after identity repair preflight.")


def prepare_collection(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
    preflight: LocalPreflight,
    *,
    source: Any | None,
    yahoo_source: YahooIdentitySupplementSource,
    wiki_source: WikiArncPinnedSource,
    boris_source: BorisKaggleCrosscheckSource,
    evidence_source: OfficialEvidenceSource,
) -> PreparedCollection:
    cache_path = _bundle_cache_path(repository, release)
    primary = _read_bundle_cache(cache_path, release)
    eodhd_attempts_this_run = 0
    if primary is None:
        if source is None:
            raise RuntimeError("EODHD source is required because the identity cache is missing.")
        primary = source.fetch(
            _role_ids(preflight.ids), completed_session=release.completed_session
        )
        # Persist the complete bounded raw run before semantic/full-history
        # validation so an inadequate candidate never causes repeat API spend.
        _write_bundle_cache(cache_path, release, primary)
        eodhd_attempts_this_run = int(
            getattr(getattr(source, "client", None), "attempt_count", primary.http_attempts)
        )

    # Unpinned operator-provided bundles are deliberately not accepted.  Yahoo
    # ownership is restricted to two exact old-identity intervals, while the
    # external overlap is accepted only from pinned version-3 CC0 files.
    supplement = yahoo_source.fetch(preflight.ids)
    fetched = merge_fetched_histories(primary, supplement, ids=preflight.ids)
    wiki = wiki_source.fetch(preflight.ids)
    fetched = merge_wiki_arnc_primary(fetched, wiki, ids=preflight.ids)
    boris = boris_source.fetch(preflight.ids)
    fetched = merge_boris_crosscheck(fetched, boris, ids=preflight.ids)
    fetch_coverage = validate_fetched_histories(
        fetched,
        preflight.ids,
        completed_session=release.completed_session,
        require_old_aa=True,
    )
    evidence = evidence_source.load()
    frames, rewrite_stats, archive_artifacts = rewrite_candidate_frames(
        preflight,
        fetched,
        release=release,
        evidence=evidence,
    )
    warnings, gates = validate_candidate_frames(
        frames,
        completed_session=release.completed_session,
        ids=preflight.ids,
    )
    _assert_release_unchanged(repository, release, release_etag)
    summary = {
        "status": "validated_dry_run",
        "release_version": release.version,
        "completed_session": release.completed_session,
        "maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "eodhd_http_attempts_this_run": eodhd_attempts_this_run,
        "eodhd_bundle_http_attempts": primary.http_attempts,
        "maximum_yahoo_http_attempts": MAX_YAHOO_HTTP_ATTEMPTS,
        "yahoo_http_attempts_this_run": yahoo_source.http_attempts,
        "maximum_boris_kaggle_http_attempts": MAX_BORIS_HTTP_ATTEMPTS,
        "boris_kaggle_http_attempts_this_run": boris_source.http_attempts,
        "maximum_aa_wiki_http_attempts": MAX_WIKI_ARNC_HTTP_ATTEMPTS,
        "aa_wiki_http_attempts_this_run": wiki_source.http_attempts,
        "aa_wiki_full_blob_url": WIKI_ARNC_URL,
        "aa_wiki_full_blob_sha256": WIKI_ARNC_FULL_SHA256,
        "aa_wiki_full_blob_archived_on_apply": True,
        "maximum_official_http_attempts": MAX_OFFICIAL_HTTP_ATTEMPTS,
        "official_http_attempts_this_run": evidence_source.http_attempts,
        "primary_cache": str(cache_path),
        "fetch_coverage": fetch_coverage,
        "rewrite": rewrite_stats,
        "gates": gates,
        "write_datasets": list(WRITE_DATASETS),
        "warning_to_clear_on_apply": PENDING_IDENTITY_WARNING,
        "warnings": list(warnings),
    }
    return PreparedCollection(
        release=release,
        release_etag=release_etag,
        pointer_etags=preflight.pointer_etags,
        frames=frames,
        archive_artifacts=archive_artifacts,
        warnings=warnings,
        summary=summary,
    )


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
                raise RuntimeError(f"Archive payload is unreadable: {path}") from exc
            if existing != artifact.content:
                raise RuntimeError(f"Archive payload conflicts with content hash: {path}")
            continue
        write_atomic(path, gzip.compress(artifact.content, mtime=0))
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"Archive payload verification failed: {path}")


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    lock_path = repository.root / ".locks/market-store-write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery_root = repository.root / "recovery/us-index-identity-repairs"
        pending = tuple(recovery_root.glob("*.json")) if recovery_root.exists() else ()
        if pending:
            raise RuntimeError(
                "US index identity recovery marker blocks writes: "
                + ", ".join(str(path) for path in pending)
            )
        transaction_root = repository.root / "transactions/us-index-identity-repairs"
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
                "Interrupted US index identity transaction requires recovery: "
                + ", ".join(str(path) for path in interrupted)
            )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_transaction_record(path: Path, value: dict[str, Any]) -> None:
    write_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
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
            ours = observed.version == committed_release_version or all(
                observed.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not ours:
                raise RuntimeError(f"unexpected release during rollback: {observed.version}")
            repository.objects.put(release_key, old_release_bytes, if_match=current.etag)
        if repository.objects.get(release_key).data != old_release_bytes:
            raise RuntimeError("release rollback verification failed")
    except Exception as exc:
        errors.append(f"{release_key}: {type(exc).__name__}: {exc}")

    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            if current.data != old_pointer_bytes[dataset]:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected pointer version during rollback: {pointer.version}"
                    )
                repository.objects.put(
                    key, old_pointer_bytes[dataset], if_match=current.etag
                )
            if repository.objects.get(key).data != old_pointer_bytes[dataset]:
                raise RuntimeError("pointer rollback verification failed")
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> None:
    current, _etag = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes():
        raise RuntimeError("Committed US index identity release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, _pointer_etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(
                f"Applied release pointer mismatch for {dataset}: expected={version}."
            )
    validate_repository_snapshot(repository).raise_for_errors()


def apply_collection(
    repository: LocalDatasetRepository,
    prepared: PreparedCollection,
) -> dict[str, Any]:
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(
            repository, prepared.release, prepared.release_etag
        )
        if PENDING_IDENTITY_WARNING not in set(prepared.release.warnings):
            raise RuntimeError("Pending identity warning disappeared before apply.")
        old_release = repository.objects.get("releases/current.json")
        old_pointer_bytes: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before apply.")
            old_pointer_bytes[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        session = prepared.release.completed_session.replace("-", "")
        planned_versions = {
            dataset: f"us-index-identity-{session}-{transaction_id}-{dataset}"
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/us-index-identity-repairs"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value).decode("ascii")
                for dataset, value in old_pointer_bytes.items()
            },
            "planned_versions": planned_versions,
            "write_datasets": list(WRITE_DATASETS),
            "created_at": utc_now_iso(),
        }
        _write_transaction_record(journal_path, journal)
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
                        "operation": "collect_us_index_identity_repairs",
                        "strict_index_identity_gate": (
                            "valid_with_wiki_raw_aa_and_disclosed_597_of_630_lila_overlap"
                        ),
                    },
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned_versions[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version

            post = validate_repository_snapshot(repository)
            post.raise_for_errors()
            inherited = tuple(
                warning
                for warning in prepared.release.warnings
                if warning != PENDING_IDENTITY_WARNING
            )
            warnings = tuple(
                dict.fromkeys(
                    (
                        *inherited,
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
                quality=DataQuality.DEGRADED if warnings else DataQuality.VALID,
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
            _write_transaction_record(journal_path, journal)
            return {
                **prepared.summary,
                "status": "applied",
                "transaction_id": transaction_id,
                "new_release_version": committed.version,
                "quality": committed.quality,
                "warnings": list(committed.warnings),
                "cleared_warning": PENDING_IDENTITY_WARNING,
            }
        except BaseException as original:
            rollback_errors = _restore_transaction_state(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointer_bytes,
                planned_versions=planned_versions,
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
            _write_transaction_record(journal_path, journal)
            if rollback_errors:
                recovery = (
                    repository.root
                    / "recovery/us-index-identity-repairs"
                    / f"{transaction_id}.json"
                )
                _write_transaction_record(recovery, journal)
                raise RuntimeError(
                    "US index identity rollback failed; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}"
                ) from original
            raise


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[[str | Path], LocalDatasetRepository] = LocalDatasetRepository,
    source_factory: Callable[..., Any] = CappedIdentityHistorySource,
    yahoo_source_factory: Callable[..., YahooIdentitySupplementSource] = YahooIdentitySupplementSource,
    wiki_source_factory: Callable[..., WikiArncPinnedSource] = WikiArncPinnedSource,
    boris_source_factory: Callable[..., BorisKaggleCrosscheckSource] = BorisKaggleCrosscheckSource,
    evidence_source_factory: Callable[..., OfficialEvidenceSource] = OfficialEvidenceSource,
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local data release is required.")
    if args.offline_plan:
        return build_offline_plan(repository, release)
    preflight = build_local_preflight(
        repository, release, require_successor_snapshot=True
    )
    primary_cache = _read_bundle_cache(_bundle_cache_path(repository, release), release)
    source = None if primary_cache is not None else source_factory()
    yahoo_source = yahoo_source_factory(
        repository.root / "state/yahoo-us-index-identity",
        allow_http=bool(args.fetch_yahoo_supplement),
    )
    wiki_source = wiki_source_factory(
        repository.root / "state/wiki-arnc-us-index-identity",
        allow_http=bool(getattr(args, "fetch_aa_wiki_crosscheck", False)),
    )
    boris_source = boris_source_factory(
        repository.root / "state/boris-kaggle-us-index-identity",
        allow_http=bool(getattr(args, "fetch_boris_crosscheck", False)),
    )
    evidence_source = evidence_source_factory(
        repository.root / "state/official-us-index-identity",
        allow_http=bool(args.fetch_official_evidence),
    )
    prepared = prepare_collection(
        repository,
        release,
        release_etag,
        preflight,
        source=source,
        yahoo_source=yahoo_source,
        wiki_source=wiki_source,
        boris_source=boris_source,
        evidence_source=evidence_source,
    )
    return apply_collection(repository, prepared) if args.apply else prepared.summary


def main(argv: list[str] | None = None) -> int:
    result = run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
