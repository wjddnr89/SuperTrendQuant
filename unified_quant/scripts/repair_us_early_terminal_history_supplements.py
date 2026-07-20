#!/usr/bin/env python3
"""Fetch, plan, and atomically apply five early terminal-history supplements.

The bootstrap starts on 2015-01-01, leaving five securities with fewer than
the 60 XNYS sessions required by the terminal-provider exception gate.  This
module closes only those exact gaps:

* COV is reconstructed offline from two already archived, hash-pinned raw
  histories (DirectIndex and the frozen Quandl WIKI mirror);
* PETM, CFN, and the legacy ``AGN_old.US`` identity use one EOD-only EODHD
  response each.  Every accepted response contains one existing 2015-01-02
  overlap row, so a provider alias cannot be accepted without matching the
  stored identity exactly;
* SWY's one allowed EODHD request stopped on an undisclosed adjustment.  It is
  not retried: an explicit offline import extracts exact raw rows and the
  2014-12-23 dividend from the already frozen WIKI archive.

The default command is a read-only plan.  ``--fetch-missing`` is the only
network path and is capped at four non-retry EODHD attempts through both the
global provider budget and a batch-specific immutable attempt ledger.  The SWY
fallback is written only by ``--import-reviewed-swy-wiki ZIP`` after validating
the full frozen archive and its private-use license boundary.  ``--apply``
never accesses the network; it commits prices, two reviewed dividends, rebuilt
factors, extended identity intervals, and content-addressed provenance in one
CAS transaction.
"""

from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import gzip
import hashlib
import io
import json
import math
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import exchange_calendars as xcals
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import EodhdClient
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
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
OPERATION = "repair_us_early_terminal_history_supplements"
WRITE_DATASETS = (
    "corporate_actions",
    "daily_price_raw",
    "adjustment_factors",
    "security_master",
    "symbol_history",
    "source_archive",
)
REQUIRED_DATASETS = (
    *WRITE_DATASETS,
    "lifecycle_resolutions",
    "index_constituent_anchors",
    "index_membership_events",
)
TRANSACTION_DIR = "transactions/us-early-terminal-history-supplements"
RECOVERY_DIR = "recovery/us-early-terminal-history-supplements"
STATE_DIR = "state/us-early-terminal-history-supplements"
REVIEWED_AT = "2026-07-19T00:00:00Z"
TERMINAL_WINDOW_SESSIONS = 60
MAX_EODHD_HTTP_ATTEMPTS = 4
EODHD_OVERLAP_SESSION = "2015-01-02"
IDENTITY_REPAIR_SOURCE = "reviewed_early_terminal_history_supplement"
WIKI_SOURCE_URL = (
    "https://raw.githubusercontent.com/teddykoker/survivorship-free-spy/"
    "0bcc715e2dd37b7ecec65c549be843574120bd58/survivorship-free/data/COV.csv"
)
KAGGLE_WIKI_SOURCE_URL = (
    "https://www.kaggle.com/datasets/marketneutral/"
    "quandl-wiki-prices-us-equites"
)


@dataclass(frozen=True)
class HistoryCase:
    symbol: str
    security_id: str
    provider_symbol: str
    exchange: str
    name: str
    old_active_from: str
    old_history_from: str
    active_to: str
    history_to: str
    terminal_session: str
    supplement_start: str
    missing_session_count: int
    old_price_rows: int
    old_price_source: str
    old_price_hash: str
    old_price_retrieved_at: str
    old_identity_source: str
    old_identity_url: str
    old_identity_hash: str
    old_identity_retrieved_at: str
    terminal_event_id: str
    terminal_action_type: str
    terminal_action_date: str
    lifecycle_candidate_id: str
    provider_uncertainty: str = ""

    @property
    def request_url(self) -> str:
        return (
            f"https://eodhd.com/api/eod/{self.provider_symbol}?"
            f"from={self.supplement_start}&to={EODHD_OVERLAP_SESSION}"
        )


CASES: tuple[HistoryCase, ...] = (
    HistoryCase(
        symbol="PETM",
        security_id="US:EODHD:1bdba556-a266-51b0-aa4f-f7b4540d601a",
        provider_symbol="PETM.US",
        exchange="NASDAQ",
        name="PetSmart Inc",
        old_active_from="2015-01-02",
        old_history_from="2015-01-01",
        active_to="2015-03-11",
        history_to="",
        terminal_session="2015-03-11",
        supplement_start="2014-12-12",
        missing_session_count=13,
        old_price_rows=47,
        old_price_source="eodhd_eod",
        old_price_hash="6abe7378fb4832207ce83eb95064119e4af6fc0e097c05f415be055f33deb68c",
        old_price_retrieved_at="2026-07-16T15:56:24.262529Z",
        old_identity_source="eodhd_exchange_symbols",
        old_identity_url="https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
        old_identity_hash="2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        old_identity_retrieved_at="2026-07-16T15:56:01.033938Z",
        terminal_event_id="f956911448cfd2c4851ea945da4b6d3cfdc32bf36851f9f90d36e1691cd7274f",
        terminal_action_type="cash_merger",
        terminal_action_date="2015-03-11",
        lifecycle_candidate_id="817e880be922b53b0bc3f9792a416504eaaa069859ae9e038af9d21860637f43",
    ),
    HistoryCase(
        symbol="SWY",
        security_id="US:EODHD:c822e178-af6a-5fb0-8961-5bb4fad3126f",
        provider_symbol="SWY.US",
        exchange="NYSE",
        name="Safeway Inc",
        old_active_from="2015-01-02",
        old_history_from="2015-01-01",
        active_to="2015-01-29",
        history_to="",
        terminal_session="2015-01-29",
        supplement_start="2014-11-03",
        missing_session_count=41,
        old_price_rows=19,
        old_price_source="eodhd_eod",
        old_price_hash="c92322481ddac09a83ecae4e1a38e47ddcee6fb96a374f4f26f0b103264d1659",
        old_price_retrieved_at="2026-07-16T15:58:07.921225Z",
        old_identity_source="eodhd_exchange_symbols",
        old_identity_url="https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
        old_identity_hash="2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        old_identity_retrieved_at="2026-07-16T15:56:01.033938Z",
        terminal_event_id="9cc65aed5a2272b739c793904b6953ff32619c981c1b1a9368cf6bc8066b49e0",
        terminal_action_type="cash_merger",
        terminal_action_date="2015-01-30",
        lifecycle_candidate_id="ea26fc8bd698fb1726ae05b8320f39dab53ec7aa41209e38bf801bf68d3c4e5a",
    ),
    HistoryCase(
        symbol="CFN",
        security_id="US:EODHD:ecb335df-157d-5e46-be2e-12ef9f49835e",
        provider_symbol="CFN.US",
        exchange="NYSE",
        name="CareFusion Corp",
        old_active_from="2015-01-02",
        old_history_from="2015-01-01",
        active_to="2015-03-16",
        history_to="",
        terminal_session="2015-03-16",
        supplement_start="2014-12-17",
        missing_session_count=10,
        old_price_rows=50,
        old_price_source="eodhd_eod",
        old_price_hash="1c8b5a588e464fe62783f5cb07327afa5b744a9e25ba515fe0856deade90ec21",
        old_price_retrieved_at="2026-07-16T15:58:25.726911Z",
        old_identity_source="eodhd_exchange_symbols",
        old_identity_url="https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
        old_identity_hash="2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        old_identity_retrieved_at="2026-07-16T15:56:01.033938Z",
        terminal_event_id="f0f24b8509e766ef3bcfb3b870b8645fdd5729bc460c7947eca0e12a7b37f6fc",
        terminal_action_type="stock_merger",
        terminal_action_date="2015-03-17",
        lifecycle_candidate_id="f8f241d51f921fdb75090244d9d3d96846110a97b9717fd122874323e134d145",
    ),
    HistoryCase(
        symbol="AGN",
        security_id="US:EODHD:9f13974d-7f81-5aac-a3a7-ed1d184bd76b",
        provider_symbol="AGN_old.US",
        exchange="NYSE",
        name="Allergan Inc",
        old_active_from="2015-01-02",
        old_history_from="2015-01-01",
        active_to="2015-03-22",
        history_to="2015-03-22",
        terminal_session="2015-03-16",
        supplement_start="2014-12-17",
        missing_session_count=10,
        old_price_rows=50,
        old_price_source="eodhd_eod",
        old_price_hash="5b4429428427d90a403ba27f5deb62f770a89c31214b37c2bd8a50f1c74339e6",
        old_price_retrieved_at="2026-07-16T15:57:42.122050Z",
        old_identity_source="official_identity_repair",
        old_identity_url=(
            "https://press.spglobal.com/2015-03-16-American-Airlines-Group-"
            "Set-to-Join-the-S-P-500"
        ),
        old_identity_hash="433bc0f8f521931344213bd6bd180efd4636a870cc3151b5fbba815433bc3c49",
        old_identity_retrieved_at="2026-07-18T02:25:18.591534Z",
        terminal_event_id="66f9915d5d513afda9c2b595e79e9e3208dba3b3e211b97d4b658835c56dc263",
        terminal_action_type="stock_merger",
        terminal_action_date="2015-03-17",
        lifecycle_candidate_id="7f3aa6ef234837000a9c5cffc5a2477077f2342068a904b775c18a1df61d52ff",
        provider_uncertainty=(
            "AGN_old.US is proven for the stored 2015 overlap, but pre-2015 "
            "legacy-alias coverage remains unconfirmed until the one capped response."
        ),
    ),
    HistoryCase(
        symbol="COV",
        security_id="US:EODHD:e03a169c-f7e7-539c-9dde-a7da5a8e861c",
        provider_symbol="COV.US",
        exchange="NYSE",
        name="Covidien plc",
        old_active_from="2015-01-01",
        old_history_from="2015-01-01",
        active_to="2015-01-26",
        history_to="2015-01-26",
        terminal_session="2015-01-26",
        supplement_start="2014-10-29",
        missing_session_count=44,
        old_price_rows=16,
        old_price_source="directindex_pinned_csv",
        old_price_hash="458e4272937f87f40330679d76bd6e9d6e3fd833fc22093c42f947ab3427cd03",
        old_price_retrieved_at="2026-07-17T23:28:20.293601Z",
        old_identity_source="sec_edgar+eodhd_terminal_price",
        old_identity_url=(
            "https://www.sec.gov/Archives/edgar/data/1385187/"
            "000119312515020714/d860155d8k.htm"
        ),
        old_identity_hash="afea2cbe511a1a7e7b7f9eca5df277be82895191261966fbb925f950eaea6807",
        old_identity_retrieved_at="2026-07-17T23:38:02.068464Z",
        terminal_event_id="213538ac37fdde085bd57c19ed76f83e12ab41de48c80c70ec071572399ecebe",
        terminal_action_type="stock_merger",
        terminal_action_date="2015-01-26",
        lifecycle_candidate_id="7cd588bf6a419ecb262fa32e3c3e48937a2b0901580e79ef24868b2ce76105ca",
    ),
)
EODHD_CASES = CASES[:-1]
COV_CASE = CASES[-1]
SWY_CASE = EODHD_CASES[1]


DIRECTINDEX_URL = (
    "https://raw.githubusercontent.com/bdi2357/DirectIndexing/"
    "8359b09ac8a00f1688ec9a1323a5533f0fc151d1/data/PriceVolume/COV.csv"
)
DIRECTINDEX_ENVELOPE_HASH = (
    "ae3a1d865cd2497fa901d0584ff18f2f5da9d737347fb66922cc48eb2e25db38"
)
DIRECTINDEX_RAW_HASH = COV_CASE.old_price_hash
DIRECTINDEX_RAW_BYTES = 101_615
DIRECTINDEX_README_ENVELOPE_HASH = (
    "05dbbe5b378120cac1048e56f56735dd561bac5da34afcd92d6e712fc8ce9326"
)
DIRECTINDEX_LICENSE_ENVELOPE_HASH = (
    "f24bfb27c63ec53e313da0439f8799b2fba39a69bcd95bf6111ae75d63d8d042"
)
WIKI_ENVELOPE_HASH = (
    "9f8b4305ae058d63c3d2af8888b4264f08d7407847258def0f8dc04c5f6d0c24"
)
WIKI_RAW_HASH = "41bba9dde2282b8d69e5776c05af9b6b0b5026cd2664699974d98ed9e65b7731"
WIKI_RAW_BYTES = 40_069
WIKI_COV_SELECTED_HASH = (
    "3955b8039728b65c58fa47772bbd175e3f5ab7dd124837fe2023d52c75d34ba8"
)
WIKI_COV_SELECTED_BYTES = 4_863
COV_MAX_OHLC_ABS_DELTA = 0.001
COV_DIVIDEND_DATE = "2014-12-30"
COV_DIVIDEND_AMOUNT = 0.36
COV_DIVIDEND_SOURCE = "quandl_wiki_adjusted_cov_extract"
COV_DIVIDEND_EVENT_ID = hashlib.sha256(
    f"{COV_DIVIDEND_SOURCE}|{COV_CASE.security_id}|cash_dividend|{COV_DIVIDEND_DATE}".encode()
).hexdigest()


# SWY's one permitted EODHD request was deliberately rejected because its
# adjusted close disclosed a historical adjustment that the EOD-only response
# could not explain.  The exact raw prices and the missing 2014-12-23 dividend
# are instead recovered from a frozen WIKI archive already reviewed for the COL
# repair.  The archive's formal license is Unknown, so this evidence is allowed
# only in the user's private/internal store and must never be published.
SWY_WIKI_ZIP_SHA256 = (
    "36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae"
)
SWY_WIKI_ZIP_BYTES = 463_184_323
SWY_WIKI_MEMBER_NAME = "WIKI_PRICES.csv"
SWY_WIKI_MEMBER_SHA256 = (
    "ca7fb174c7948db85638917d25ff65d438e27d5cb23675da784c54db01e3d003"
)
SWY_WIKI_MEMBER_BYTES = 1_797_003_576
SWY_WIKI_MEMBER_COMPRESSED_BYTES = 463_184_155
SWY_WIKI_MEMBER_CRC32 = 0x946874CE
SWY_WIKI_ALL_ROWS = 6_240
SWY_WIKI_ALL_ROWS_SHA256 = (
    "2e9b28e31f2bdbc9edc5aaa601e4969347a1026e8c6e4f69626494f5c81d91da"
)
SWY_WIKI_EXTRACT_SHA256 = (
    "1596f04ff1f9db4ef99ea5bab5d6644d62df0068e0dd1c891e81395896b86402"
)
SWY_WIKI_EXTRACT_BYTES = 5_596
SWY_WIKI_METADATA_SHA256 = (
    "e83992cf9a4051e35f91e717616b5005c04deb4f290d366679e67b235cd9401b"
)
SWY_WIKI_PARENT_PROVENANCE_SHA256 = (
    "5d99f922bb7c45afe31a473f89b441f42df0cb0769b01fdfa842353304b3d636"
)
SWY_WIKI_SOURCE = "kaggle_quandl_wiki_swy_extract"
SWY_WIKI_CACHE_SCHEMA = "us_early_terminal_history_swy_wiki_cache/v1"
SWY_WIKI_COLUMNS = (
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
SWY_DIVIDEND_DATE = "2014-12-23"
SWY_DIVIDEND_AMOUNT = 0.23
SWY_DIVIDEND_SOURCE = "quandl_wiki_swy_reviewed_dividend"
SWY_DIVIDEND_EVENT_ID = hashlib.sha256(
    f"{SWY_DIVIDEND_SOURCE}|{SWY_CASE.security_id}|cash_dividend|"
    f"{SWY_DIVIDEND_DATE}".encode()
).hexdigest()
PRIVATE_WIKI_WARNING = (
    "Frozen Quandl WIKI SWY evidence has formal license Unknown; private/internal "
    "use only; publication and redistribution are blocked."
)


@dataclass(frozen=True)
class EvidenceArtifact:
    dataset: str
    source: str
    source_url: str
    retrieved_at: str
    content: bytes
    content_type: str
    effective_date: str
    suffix: str

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @property
    def archive_id(self) -> str:
        return self.source_hash

    @property
    def object_path(self) -> str:
        return (
            f"archives/{self.effective_date}/{self.source_hash}."
            f"{self.suffix}.gz"
        )


@dataclass(frozen=True)
class SupplementalEvidence:
    case: HistoryCase
    prices: pd.DataFrame
    primary_artifact: EvidenceArtifact
    identity_artifact: EvidenceArtifact
    all_artifacts: tuple[EvidenceArtifact, ...]
    overlap_rows: int
    cross_validation: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    planned_versions: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    archive_objects: Mapping[str, bytes]
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _date(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).date().isoformat()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _xnys_sessions(start: str, end: str) -> tuple[str, ...]:
    values = xcals.get_calendar("XNYS").sessions_in_range(start, end)
    return tuple(pd.Timestamp(value).date().isoformat() for value in values)


def _terminal_window(case: HistoryCase) -> tuple[str, ...]:
    sessions = _xnys_sessions("2014-01-01", case.terminal_session)
    return sessions[-TERMINAL_WINDOW_SESSIONS:]


def _missing_sessions(case: HistoryCase) -> tuple[str, ...]:
    return tuple(
        session
        for session in _terminal_window(case)
        if session < case.old_active_from
    )


def _request_sessions(case: HistoryCase) -> tuple[str, ...]:
    if case is COV_CASE:
        return ()
    return _xnys_sessions(case.supplement_start, EODHD_OVERLAP_SESSION)


def request_inventory() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "symbol": case.symbol,
            "security_id": case.security_id,
            "provider_symbol": case.provider_symbol,
            "endpoint": "eod",
            "from": case.supplement_start,
            "to": EODHD_OVERLAP_SESSION,
            "safe_url": case.request_url,
            "expected_sessions": list(_request_sessions(case)),
            "expected_response_rows": len(_request_sessions(case)),
            "expected_insert_rows": case.missing_session_count,
            "overlap_session": EODHD_OVERLAP_SESSION,
            "provider_uncertainty": case.provider_uncertainty,
        }
        for case in EODHD_CASES
    )


def request_inventory_sha256() -> str:
    return _sha256(_canonical_json(request_inventory()))


# Filled after the inventory is deliberately changed and reviewed.  The static
# contract below prevents a seemingly harmless request-window edit from
# silently spending a different number of EODHD calls or accepting more rows.
TRUSTED_REQUEST_INVENTORY_SHA256 = (
    "8f8c46ef969a06085270efc50c3e50dae0d3aebed3b1edb07929fff718762ac7"
)


def _static_contract() -> None:
    if len(CASES) != 5 or len(EODHD_CASES) != MAX_EODHD_HTTP_ATTEMPTS:
        raise RuntimeError("Early terminal-history cohort/call cap changed.")
    if len({case.security_id for case in CASES}) != len(CASES):
        raise RuntimeError("Early terminal-history security IDs are duplicated.")
    for case in CASES:
        terminal = _terminal_window(case)
        missing = _missing_sessions(case)
        if (
            len(terminal) != TERMINAL_WINDOW_SESSIONS
            or terminal[0] != case.supplement_start
            or terminal[-1] != case.terminal_session
            or len(missing) != case.missing_session_count
        ):
            raise RuntimeError(f"{case.symbol} exact XNYS window changed.")
        if case is not COV_CASE:
            requested = _request_sessions(case)
            if (
                requested[:-1] != missing
                or requested[-1] != EODHD_OVERLAP_SESSION
                or len(requested) != case.missing_session_count + 1
            ):
                raise RuntimeError(f"{case.symbol} minimum EODHD request changed.")
    if sum(case.missing_session_count for case in CASES) != 118:
        raise RuntimeError("Early terminal-history total insert count changed.")
    if sum(len(_request_sessions(case)) for case in EODHD_CASES) != 78:
        raise RuntimeError("Early terminal-history EODHD response count changed.")
    if TRUSTED_REQUEST_INVENTORY_SHA256 != "TO_BE_FILLED" and (
        request_inventory_sha256() != TRUSTED_REQUEST_INVENTORY_SHA256
    ):
        raise RuntimeError("Early terminal-history request inventory is not code-pinned.")


def _archive_row(
    archive: pd.DataFrame,
    *,
    archive_id: str,
    dataset: str,
    source_url: str,
    source_hash: str,
) -> Mapping[str, Any]:
    rows = archive.loc[
        archive["archive_id"].astype(str).eq(archive_id)
        & archive["dataset"].astype(str).eq(dataset)
        & archive["source_url"].astype(str).eq(source_url)
        & archive["source_hash"].astype(str).eq(source_hash)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Source archive exact binding is absent/duplicated: {dataset}/{archive_id}."
        )
    return rows.iloc[0].to_dict()


def _read_gzip_archive_object(
    repository: LocalDatasetRepository,
    row: Mapping[str, Any],
    *,
    expected_hash: str,
) -> bytes:
    path = repository.root / _text(row.get("object_path"))
    if not path.is_file() or path.suffix != ".gz":
        raise FileNotFoundError(f"Archived object is missing: {path}")
    try:
        content = gzip.decompress(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"Archived object is not deterministic gzip: {path}") from exc
    if _sha256(content) != expected_hash:
        raise ValueError(f"Archived object content hash changed: {path}")
    return content


def _decode_exact_envelope(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    envelope_hash: str,
    dataset: str,
    source_url: str,
    raw_hash: str,
) -> tuple[bytes, Mapping[str, Any]]:
    row = _archive_row(
        archive,
        archive_id=envelope_hash,
        dataset=dataset,
        source_url=source_url,
        source_hash=envelope_hash,
    )
    envelope_bytes = _read_gzip_archive_object(
        repository, row, expected_hash=envelope_hash
    )
    try:
        envelope = json.loads(envelope_bytes)
        content = base64.b64decode(envelope["content_base64"], validate=True)
    except Exception as exc:
        raise ValueError(f"{dataset} source envelope is invalid.") from exc
    if (
        _text(envelope.get("source_url")) != source_url
        or _text(envelope.get("content_sha256")) != raw_hash
        or _sha256(content) != raw_hash
    ):
        raise ValueError(f"{dataset} nested raw provenance changed.")
    return content, row


def _parse_directindex_cov(content: bytes) -> pd.DataFrame:
    if len(content) != DIRECTINDEX_RAW_BYTES or _sha256(content) != DIRECTINDEX_RAW_HASH:
        raise ValueError("Pinned COV DirectIndex raw bytes/hash changed.")
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise ValueError("Pinned COV DirectIndex CSV is unreadable.") from exc
    expected_columns = (
        "Unnamed: 0",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adjusted_close",
        "Volume",
    )
    if tuple(raw.columns) != expected_columns or len(raw) != 1_918:
        raise ValueError("Pinned COV DirectIndex raw inventory changed.")
    raw["session"] = pd.to_datetime(raw["Date"], errors="coerce").dt.date.astype(str)
    window = raw.loc[raw["session"].isin(_terminal_window(COV_CASE))].copy()
    window = window.sort_values("session", kind="stable").reset_index(drop=True)
    if tuple(window["session"]) != _terminal_window(COV_CASE):
        raise ValueError("COV DirectIndex does not contain the exact 60-session window.")
    frame = pd.DataFrame(
        {
            "security_id": COV_CASE.security_id,
            "session": window["session"],
            "open": pd.to_numeric(window["Open"], errors="coerce"),
            "high": pd.to_numeric(window["High"], errors="coerce"),
            "low": pd.to_numeric(window["Low"], errors="coerce"),
            "close": pd.to_numeric(window["Close"], errors="coerce"),
            "volume": pd.to_numeric(window["Volume"], errors="coerce"),
            "currency": "USD",
            "source": COV_CASE.old_price_source,
            "retrieved_at": COV_CASE.old_price_retrieved_at,
            "source_hash": DIRECTINDEX_RAW_HASH,
            "source_url": DIRECTINDEX_URL,
        }
    )
    _validate_ohlcv(frame, label="COV DirectIndex")
    return frame.loc[
        :,
        list(dataset_spec("daily_price_raw").required_columns) + ["source_url"],
    ]


WIKI_COLUMNS = (
    "session",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _extract_wiki_cov_selected(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> bytes:
    raw, _ = _decode_exact_envelope(
        repository,
        archive,
        envelope_hash=WIKI_ENVELOPE_HASH,
        dataset="cov_quandl_wiki_cov_csv",
        source_url=WIKI_SOURCE_URL,
        raw_hash=WIKI_RAW_HASH,
    )
    if len(raw) != WIKI_RAW_BYTES:
        raise ValueError("Pinned Quandl WIKI COV raw byte count changed.")
    lines = raw.decode("utf-8").splitlines(keepends=True)
    if not lines or lines[0] != "date,open,high,low,close,volume\n":
        raise ValueError("Pinned Quandl WIKI COV header changed.")
    expected = set(_terminal_window(COV_CASE))
    selected = [
        line
        for line in lines[1:]
        if line.split(",", 1)[0] in expected
    ]
    content = (lines[0] + "".join(selected)).encode()
    if len(content) != WIKI_COV_SELECTED_BYTES or _sha256(content) != WIKI_COV_SELECTED_HASH:
        raise ValueError("Frozen Quandl WIKI COV selected evidence changed.")
    return content


def _parse_wiki_cov_selected(content: bytes) -> pd.DataFrame:
    if len(content) != WIKI_COV_SELECTED_BYTES or _sha256(content) != WIKI_COV_SELECTED_HASH:
        raise ValueError("Frozen Quandl WIKI COV selection is not exact.")
    raw = pd.read_csv(io.BytesIO(content)).rename(columns={"date": "session"})
    if tuple(raw.columns) != WIKI_COLUMNS or len(raw) != TERMINAL_WINDOW_SESSIONS:
        raise ValueError("Frozen Quandl WIKI COV schema/inventory changed.")
    if tuple(raw["session"].astype(str)) != _terminal_window(COV_CASE):
        raise ValueError("Frozen Quandl WIKI COV sessions changed.")
    for column in WIKI_COLUMNS[1:]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw[list(WIKI_COLUMNS[1:])].isna().any().any():
        raise ValueError("Frozen Quandl WIKI COV contains nonnumeric evidence.")
    return raw


def _validate_ohlcv(frame: pd.DataFrame, *, label: str) -> None:
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = (
        numeric[["open", "high", "low", "close"]].notna().all(axis=1)
        & numeric[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & numeric["volume"].notna()
        & numeric["volume"].ge(0)
        & numeric["high"].ge(numeric[["open", "close", "low"]].max(axis=1))
        & numeric["low"].le(numeric[["open", "close", "high"]].min(axis=1))
    )
    if not bool(valid.all()):
        raise ValueError(f"{label} contains invalid raw OHLCV rows.")


def _load_cov_primary_and_crosscheck(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[EvidenceArtifact, ...], Mapping[str, Any]]:
    direct_content, _ = _decode_exact_envelope(
        repository,
        archive,
        envelope_hash=DIRECTINDEX_ENVELOPE_HASH,
        dataset="cov_directindex_cov_csv",
        source_url=DIRECTINDEX_URL,
        raw_hash=DIRECTINDEX_RAW_HASH,
    )
    # These exact rows prove the pinned source's documentation and MIT license
    # remain in the release; the CSV alone is not accepted as provenance.
    for digest, dataset in (
        (DIRECTINDEX_README_ENVELOPE_HASH, "cov_directindex_readme"),
        (DIRECTINDEX_LICENSE_ENVELOPE_HASH, "cov_directindex_license"),
    ):
        rows = archive.loc[
            archive["archive_id"].astype(str).eq(digest)
            & archive["dataset"].astype(str).eq(dataset)
            & archive["source_hash"].astype(str).eq(digest)
        ]
        if len(rows) != 1:
            raise ValueError(f"COV DirectIndex provenance is missing: {dataset}.")
        _read_gzip_archive_object(
            repository, rows.iloc[0].to_dict(), expected_hash=digest
        )
    wiki_content = _extract_wiki_cov_selected(repository, archive)
    direct = _parse_directindex_cov(direct_content)
    wiki = _parse_wiki_cov_selected(wiki_content)
    merged = direct.merge(
        wiki,
        on="session",
        suffixes=("_direct", "_wiki"),
        validate="one_to_one",
    )
    # The pinned WIKI file is dividend-adjusted while DirectIndex is raw.  A
    # single factor applies through 2014-12-29, then becomes exactly one on the
    # 2014-12-30 ex-date.  This lets the second archive independently confirm
    # all 60 sessions, volumes, and the complete return path without pretending
    # adjusted prices are raw prices.
    pre = merged["session"].astype(str).lt(COV_DIVIDEND_DATE)
    post = ~pre
    pre_close_ratio = (
        pd.to_numeric(merged.loc[pre, "close_wiki"], errors="coerce")
        / pd.to_numeric(merged.loc[pre, "close_direct"], errors="coerce")
    )
    adjustment_factor = float(pre_close_ratio.median())
    if not (
        math.isfinite(adjustment_factor)
        and 0 < adjustment_factor < 1
        and bool((pre_close_ratio - adjustment_factor).abs().le(5e-12).all())
    ):
        raise ValueError("COV archived pre-dividend adjustment factor changed.")
    maximum: dict[str, float] = {}
    for field in ("open", "high", "low", "close"):
        direct_values = pd.to_numeric(
            merged[f"{field}_direct"], errors="coerce"
        )
        expected_adjusted = direct_values.where(post, direct_values * adjustment_factor)
        delta = (
            expected_adjusted
            - pd.to_numeric(merged[f"{field}_wiki"], errors="coerce")
        ).abs()
        if delta.isna().any() or bool(delta.gt(COV_MAX_OHLC_ABS_DELTA).any()):
            raise ValueError(f"COV archived {field} cross-validation changed.")
        maximum[field] = float(delta.max())
    volume_delta = (
        pd.to_numeric(merged["volume_direct"], errors="coerce")
        - pd.to_numeric(merged["volume_wiki"], errors="coerce")
    ).abs()
    if volume_delta.isna().any() or bool(volume_delta.ne(0).any()):
        raise ValueError("COV archived volume cross-validation changed.")
    maximum["volume"] = float(volume_delta.max())
    previous_close = float(
        merged.loc[
            merged["session"].astype(str).eq("2014-12-29"), "close_direct"
        ].iloc[0]
    )
    inferred_dividend = previous_close * (1.0 - adjustment_factor)
    if (
        round(inferred_dividend, 2) != COV_DIVIDEND_AMOUNT
        or abs(inferred_dividend - COV_DIVIDEND_AMOUNT) > 0.001
    ):
        raise ValueError("COV archived adjusted/raw dividend inference changed.")
    direct_artifact = EvidenceArtifact(
        dataset="cov_directindex_raw_csv",
        source="cov_directindex_cov_csv",
        source_url=DIRECTINDEX_URL,
        retrieved_at=COV_CASE.old_price_retrieved_at,
        content=direct_content,
        content_type="text/csv",
        effective_date="2026-07-15",
        suffix="csv",
    )
    wiki_artifact = EvidenceArtifact(
        dataset="quandl_wiki_cov_adjusted_extract",
        source=COV_DIVIDEND_SOURCE,
        source_url=WIKI_SOURCE_URL,
        retrieved_at="2026-07-17T23:28:20.996366Z",
        content=wiki_content,
        content_type="text/csv",
        effective_date="2026-07-15",
        suffix="csv",
    )
    return direct, (direct_artifact, wiki_artifact), {
        "status": "passed",
        "sessions_compared": TERMINAL_WINDOW_SESSIONS,
        "primary_source_hash": DIRECTINDEX_RAW_HASH,
        "independent_parent_raw_sha256": WIKI_RAW_HASH,
        "independent_extract_sha256": WIKI_COV_SELECTED_HASH,
        "independent_price_basis": "dividend_adjusted",
        "inferred_pre_dividend_factor": adjustment_factor,
        "maximum_absolute_delta": maximum,
        "ohlc_absolute_tolerance": COV_MAX_OHLC_ABS_DELTA,
        "volume_tolerance": 0,
        "dividend": {
            "effective_date": COV_DIVIDEND_DATE,
            "cash_amount": COV_DIVIDEND_AMOUNT,
            "inferred_unrounded_cash_amount": inferred_dividend,
            "inference": "raw_previous_close_x_one_minus_adjusted_scale",
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _swy_wiki_cache_path(root: Path) -> Path:
    return (
        Path(root)
        / STATE_DIR
        / f"swy-wiki-{SWY_WIKI_EXTRACT_SHA256}.json.gz"
    )


def _validate_swy_wiki_parent_provenance(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> Mapping[str, Any]:
    expected = (
        (
            SWY_WIKI_METADATA_SHA256,
            "kaggle_dataset_metadata",
            "kaggle_dataset_metadata",
        ),
        (
            SWY_WIKI_PARENT_PROVENANCE_SHA256,
            "reviewed_col_scaling_provenance",
            "reviewed_col_scaling_repair",
        ),
    )
    contents: dict[str, bytes] = {}
    for digest, dataset, source in expected:
        rows = archive.loc[
            archive["archive_id"].astype(str).eq(digest)
            & archive["dataset"].astype(str).eq(dataset)
            & archive["source"].astype(str).eq(source)
            & archive["source_hash"].astype(str).eq(digest)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Frozen WIKI parent provenance is missing: {dataset}/{digest}."
            )
        contents[digest] = _read_gzip_archive_object(
            repository, rows.iloc[0].to_dict(), expected_hash=digest
        )
    try:
        metadata = json.loads(contents[SWY_WIKI_METADATA_SHA256])
        provenance = json.loads(contents[SWY_WIKI_PARENT_PROVENANCE_SHA256])
    except Exception as exc:
        raise ValueError("Frozen WIKI parent provenance is invalid JSON.") from exc
    frozen = provenance.get("frozen_evidence", {})
    policy = provenance.get("license_policy", {})
    if not (
        metadata.get("ref") == "marketneutral/quandl-wiki-prices-us-equites"
        and metadata.get("currentVersionNumber") == 1
        and metadata.get("licenseName") == "Unknown"
        and frozen.get("zip_sha256") == SWY_WIKI_ZIP_SHA256
        and int(frozen.get("zip_size", -1)) == SWY_WIKI_ZIP_BYTES
        and frozen.get("member_name") == SWY_WIKI_MEMBER_NAME
        and frozen.get("member_sha256") == SWY_WIKI_MEMBER_SHA256
        and int(frozen.get("member_size", -1)) == SWY_WIKI_MEMBER_BYTES
        and frozen.get("member_crc32") == f"{SWY_WIKI_MEMBER_CRC32:08x}"
        and frozen.get("metadata_sha256") == SWY_WIKI_METADATA_SHA256
        and frozen.get("metadata_license_name") == "Unknown"
        and policy.get("allowed_scope") == "private_internal_only"
        and policy.get("formal_license_name") == "Unknown"
        and policy.get("publication_allowed") is False
        and policy.get("redistribution_allowed") is False
        and policy.get("fail_closed") is True
    ):
        raise ValueError("Frozen WIKI license/archive parent contract changed.")
    return {
        "metadata_sha256": SWY_WIKI_METADATA_SHA256,
        "parent_provenance_sha256": SWY_WIKI_PARENT_PROVENANCE_SHA256,
        "formal_license_name": "Unknown",
        "allowed_scope": "private_internal_only",
        "publication_allowed": False,
        "redistribution_allowed": False,
        "fail_closed": True,
    }


def _failed_swy_attempt(root: Path) -> Mapping[str, Any]:
    ledger = _read_fetch_ledger(root)
    matches = [
        item
        for item in ledger["attempts"]
        if _text(item.get("symbol")) == SWY_CASE.symbol
    ]
    if len(matches) != 1:
        raise RuntimeError("The one reviewed SWY EODHD attempt is absent/duplicated.")
    attempt = matches[0]
    if not (
        _text(attempt.get("provider_symbol")) == SWY_CASE.provider_symbol
        and _text(attempt.get("source_url")) == SWY_CASE.request_url
        and _text(attempt.get("status")) == "http_or_validation_failed_no_retry"
        and int(attempt.get("budget_usage_after_claim", 0)) > 0
        and "undisclosed historical adjustment" in _text(attempt.get("error"))
    ):
        raise RuntimeError("The failed SWY attempt is not the reviewed adjustment stop.")
    return attempt


def _build_swy_wiki_provenance(
    *,
    extract: bytes,
    attempt: Mapping[str, Any],
    license_binding: Mapping[str, Any],
) -> bytes:
    return _canonical_json(
        {
            "schema": "us_early_terminal_history_swy_wiki_evidence/v1",
            "reviewed_at": REVIEWED_AT,
            "security_id": SWY_CASE.security_id,
            "symbol": SWY_CASE.symbol,
            "source_url": KAGGLE_WIKI_SOURCE_URL,
            "frozen_evidence": {
                "zip_sha256": SWY_WIKI_ZIP_SHA256,
                "zip_size": SWY_WIKI_ZIP_BYTES,
                "member_name": SWY_WIKI_MEMBER_NAME,
                "member_sha256": SWY_WIKI_MEMBER_SHA256,
                "member_size": SWY_WIKI_MEMBER_BYTES,
                "member_compressed_size": SWY_WIKI_MEMBER_COMPRESSED_BYTES,
                "member_crc32": f"{SWY_WIKI_MEMBER_CRC32:08x}",
                "all_swy_row_count": SWY_WIKI_ALL_ROWS,
                "all_swy_rows_sha256": SWY_WIKI_ALL_ROWS_SHA256,
                "extract_sha256": _sha256(extract),
                "extract_size": len(extract),
                "extract_sessions": list(_request_sessions(SWY_CASE)),
            },
            "eodhd_fail_closed_binding": {
                "attempt_sha256": _sha256(_canonical_json(dict(attempt))),
                "status": _text(attempt.get("status")),
                "budget_usage_after_claim": int(
                    attempt.get("budget_usage_after_claim", 0)
                ),
                "automatic_retry_allowed": False,
                "reason": "undisclosed_historical_adjustment",
            },
            "action_evidence": {
                "action_type": "cash_dividend",
                "ex_date": SWY_DIVIDEND_DATE,
                "cash_amount": SWY_DIVIDEND_AMOUNT,
                "currency": "USD",
                "split_ratios": [1.0],
            },
            "license_policy": dict(license_binding),
        }
    )


def _extract_reviewed_swy_wiki(
    zip_path: Path,
) -> bytes:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Frozen WIKI ZIP is absent: {zip_path}")
    if zip_path.stat().st_size != SWY_WIKI_ZIP_BYTES:
        raise ValueError("Frozen WIKI ZIP byte count changed.")
    if _sha256_file(zip_path) != SWY_WIKI_ZIP_SHA256:
        raise ValueError("Frozen WIKI ZIP hash changed.")
    expected_header = (",".join(SWY_WIKI_COLUMNS) + "\n").encode()
    member_digest = hashlib.sha256()
    swy_digest = hashlib.sha256()
    selected: list[bytes] = []
    swy_rows = 0
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != [SWY_WIKI_MEMBER_NAME]:
            raise ValueError("Frozen WIKI ZIP member inventory changed.")
        info = archive.getinfo(SWY_WIKI_MEMBER_NAME)
        if not (
            info.file_size == SWY_WIKI_MEMBER_BYTES
            and info.compress_size == SWY_WIKI_MEMBER_COMPRESSED_BYTES
            and info.CRC == SWY_WIKI_MEMBER_CRC32
            and info.compress_type == zipfile.ZIP_DEFLATED
        ):
            raise ValueError("Frozen WIKI ZIP member metadata changed.")
        with archive.open(info, "r") as member:
            for line_number, line in enumerate(member):
                member_digest.update(line)
                if line_number == 0:
                    if line != expected_header:
                        raise ValueError("Frozen WIKI CSV header changed.")
                    continue
                fields = line.split(b",", 2)
                if fields[0] != b"SWY":
                    continue
                swy_rows += 1
                swy_digest.update(line)
                session = fields[1].decode("ascii")
                if SWY_CASE.supplement_start <= session <= EODHD_OVERLAP_SESSION:
                    selected.append(line)
    content = expected_header + b"".join(selected)
    if not (
        member_digest.hexdigest() == SWY_WIKI_MEMBER_SHA256
        and swy_rows == SWY_WIKI_ALL_ROWS
        and swy_digest.hexdigest() == SWY_WIKI_ALL_ROWS_SHA256
        and len(content) == SWY_WIKI_EXTRACT_BYTES
        and _sha256(content) == SWY_WIKI_EXTRACT_SHA256
    ):
        raise ValueError("Frozen WIKI SWY extraction contract changed.")
    return content


def _write_swy_wiki_cache(
    root: Path,
    *,
    extract: bytes,
    provenance: bytes,
) -> None:
    payload_bytes = _canonical_json(
        {
            "schema": SWY_WIKI_CACHE_SCHEMA,
            "request_inventory_sha256": request_inventory_sha256(),
            "symbol": SWY_CASE.symbol,
            "security_id": SWY_CASE.security_id,
            "source_url": KAGGLE_WIKI_SOURCE_URL,
            "retrieved_at": REVIEWED_AT,
            "extract_sha256": _sha256(extract),
            "extract_base64": base64.b64encode(extract).decode("ascii"),
            "provenance_sha256": _sha256(provenance),
            "provenance_base64": base64.b64encode(provenance).decode("ascii"),
        }
    )
    wrapper = _canonical_json(
        {
            "payload_sha256": _sha256(payload_bytes),
            "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
        }
    )
    compressed = gzip.compress(wrapper, mtime=0)
    path = _swy_wiki_cache_path(root)
    if path.is_file():
        if path.read_bytes() != compressed:
            raise RuntimeError("Immutable reviewed SWY WIKI cache already differs.")
        return
    write_atomic(path, compressed)


def _parse_swy_wiki_extract(content: bytes) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    if len(content) != SWY_WIKI_EXTRACT_BYTES or _sha256(content) != SWY_WIKI_EXTRACT_SHA256:
        raise ValueError("Reviewed SWY WIKI extract bytes/hash changed.")
    raw = pd.read_csv(io.BytesIO(content))
    if tuple(raw.columns) != SWY_WIKI_COLUMNS or len(raw) != len(
        _request_sessions(SWY_CASE)
    ):
        raise ValueError("Reviewed SWY WIKI schema/inventory changed.")
    if set(raw["ticker"].astype(str)) != {SWY_CASE.symbol} or tuple(
        raw["date"].astype(str)
    ) != _request_sessions(SWY_CASE):
        raise ValueError("Reviewed SWY WIKI session identity changed.")
    numeric_columns = SWY_WIKI_COLUMNS[2:]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw[list(numeric_columns)].isna().any().any():
        raise ValueError("Reviewed SWY WIKI contains nonnumeric evidence.")
    if set(raw["split_ratio"].astype(float)) != {1.0}:
        raise ValueError("Reviewed SWY WIKI split inventory changed.")
    dividends = raw.loc[raw["ex-dividend"].astype(float).ne(0)]
    if not (
        len(dividends) == 1
        and _text(dividends.iloc[0]["date"]) == SWY_DIVIDEND_DATE
        and math.isclose(
            float(dividends.iloc[0]["ex-dividend"]),
            SWY_DIVIDEND_AMOUNT,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Reviewed SWY WIKI dividend inventory changed.")
    pre = raw["date"].astype(str).lt(SWY_DIVIDEND_DATE)
    factor = float((raw.loc[pre, "adj_close"] / raw.loc[pre, "close"]).median())
    close_ratios = raw.loc[pre, "adj_close"] / raw.loc[pre, "close"]
    if not (
        0 < factor < 1
        and bool((close_ratios - factor).abs().le(5e-12).all())
        and round(
            float(raw.loc[raw["date"].eq("2014-12-22"), "close"].iloc[0])
            * (1.0 - factor),
            2,
        )
        == SWY_DIVIDEND_AMOUNT
    ):
        raise ValueError("Reviewed SWY WIKI dividend adjustment relation changed.")
    for raw_field, adjusted_field in (
        ("open", "adj_open"),
        ("high", "adj_high"),
        ("low", "adj_low"),
        ("close", "adj_close"),
    ):
        expected = raw[raw_field].where(~pre, raw[raw_field] * factor)
        if bool((expected - raw[adjusted_field]).abs().gt(1e-9).any()):
            raise ValueError(f"Reviewed SWY WIKI {adjusted_field} relation changed.")
    if bool((raw["volume"] - raw["adj_volume"]).abs().gt(0).any()):
        raise ValueError("Reviewed SWY WIKI adjusted volume changed.")
    prices = pd.DataFrame(
        {
            "security_id": SWY_CASE.security_id,
            "session": raw["date"].astype(str),
            "open": raw["open"],
            "high": raw["high"],
            "low": raw["low"],
            "close": raw["close"],
            "volume": raw["volume"],
            "currency": "USD",
            "source": SWY_WIKI_SOURCE,
            "retrieved_at": REVIEWED_AT,
            "source_hash": SWY_WIKI_EXTRACT_SHA256,
            "source_url": KAGGLE_WIKI_SOURCE_URL,
        }
    )
    _validate_ohlcv(prices, label="SWY frozen WIKI")
    return prices.loc[
        :,
        list(dataset_spec("daily_price_raw").required_columns) + ["source_url"],
    ], {
        "status": "passed",
        "provider": "frozen_quandl_wiki",
        "expected_response_rows": len(_request_sessions(SWY_CASE)),
        "insert_rows": SWY_CASE.missing_session_count,
        "overlap_session": EODHD_OVERLAP_SESSION,
        "overlap_rows": 1,
        "raw_price_basis": True,
        "dividend_ex_date": SWY_DIVIDEND_DATE,
        "dividend_cash_amount": SWY_DIVIDEND_AMOUNT,
        "inferred_pre_dividend_factor": factor,
        "formal_license_name": "Unknown",
        "allowed_scope": "private_internal_only",
        "publication_allowed": False,
        "redistribution_allowed": False,
    }


def _read_swy_wiki_cache(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> tuple[pd.DataFrame, EvidenceArtifact, EvidenceArtifact, Mapping[str, Any]] | None:
    path = _swy_wiki_cache_path(repository.root)
    if not path.is_file():
        return None
    try:
        wrapper = json.loads(gzip.decompress(path.read_bytes()))
        payload_bytes = base64.b64decode(wrapper["payload_base64"], validate=True)
        if _sha256(payload_bytes) != _text(wrapper.get("payload_sha256")):
            raise ValueError("payload wrapper hash mismatch")
        payload = json.loads(payload_bytes)
        extract = base64.b64decode(payload["extract_base64"], validate=True)
        provenance = base64.b64decode(
            payload["provenance_base64"], validate=True
        )
    except Exception as exc:
        raise ValueError("Immutable reviewed SWY WIKI cache is invalid.") from exc
    if not (
        payload.get("schema") == SWY_WIKI_CACHE_SCHEMA
        and payload.get("request_inventory_sha256") == request_inventory_sha256()
        and _text(payload.get("symbol")) == SWY_CASE.symbol
        and _text(payload.get("security_id")) == SWY_CASE.security_id
        and _text(payload.get("source_url")) == KAGGLE_WIKI_SOURCE_URL
        and _text(payload.get("retrieved_at")) == REVIEWED_AT
        and _text(payload.get("extract_sha256")) == SWY_WIKI_EXTRACT_SHA256
        and _sha256(extract) == SWY_WIKI_EXTRACT_SHA256
        and _text(payload.get("provenance_sha256")) == _sha256(provenance)
    ):
        raise ValueError("Immutable reviewed SWY WIKI cache metadata changed.")
    license_binding = _validate_swy_wiki_parent_provenance(repository, archive)
    attempt = _failed_swy_attempt(repository.root)
    expected_provenance = _build_swy_wiki_provenance(
        extract=extract,
        attempt=attempt,
        license_binding=license_binding,
    )
    if provenance != expected_provenance:
        raise ValueError("Reviewed SWY WIKI provenance binding changed.")
    prices, cross = _parse_swy_wiki_extract(extract)
    cross = {
        **dict(cross),
        "failed_eodhd_attempt_sha256": _sha256(_canonical_json(dict(attempt))),
        "automatic_eodhd_retry_allowed": False,
    }
    extract_artifact = EvidenceArtifact(
        dataset="kaggle_quandl_wiki_swy_extract",
        source=SWY_WIKI_SOURCE,
        source_url=KAGGLE_WIKI_SOURCE_URL,
        retrieved_at=REVIEWED_AT,
        content=extract,
        content_type="text/csv",
        effective_date="2026-07-15",
        suffix="csv",
    )
    provenance_artifact = EvidenceArtifact(
        dataset="reviewed_swy_wiki_history_provenance",
        source="reviewed_early_terminal_history_supplement",
        source_url=KAGGLE_WIKI_SOURCE_URL,
        retrieved_at=REVIEWED_AT,
        content=provenance,
        content_type="application/json",
        effective_date="2026-07-15",
        suffix="json",
    )
    return prices, extract_artifact, provenance_artifact, cross


def _response_cache_path(root: Path, case: HistoryCase) -> Path:
    request_hash = _sha256(
        _canonical_json(
            {
                "inventory_sha256": request_inventory_sha256(),
                "security_id": case.security_id,
                "provider_symbol": case.provider_symbol,
                "from": case.supplement_start,
                "to": EODHD_OVERLAP_SESSION,
                "endpoint": "eod",
            }
        )
    )
    return Path(root) / STATE_DIR / "responses" / f"{request_hash}.json.gz"


def _fetch_ledger_path(root: Path) -> Path:
    return (
        Path(root)
        / STATE_DIR
        / f"fetch-ledger-{request_inventory_sha256()}.json"
    )


def _read_fetch_ledger(root: Path) -> dict[str, Any]:
    path = _fetch_ledger_path(root)
    if not path.is_file():
        return {
            "schema": "us_early_terminal_history_fetch_ledger/v1",
            "request_inventory_sha256": request_inventory_sha256(),
            "maximum_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
            "attempts": [],
        }
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise RuntimeError("Early terminal-history fetch ledger is unreadable.") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != "us_early_terminal_history_fetch_ledger/v1"
        or value.get("request_inventory_sha256") != request_inventory_sha256()
        or int(value.get("maximum_http_attempts", -1))
        != MAX_EODHD_HTTP_ATTEMPTS
        or not isinstance(value.get("attempts"), list)
        or len(value["attempts"]) > MAX_EODHD_HTTP_ATTEMPTS
    ):
        raise RuntimeError("Early terminal-history fetch ledger contract changed.")
    symbols = [_text(item.get("symbol")) for item in value["attempts"]]
    if len(set(symbols)) != len(symbols) or any(
        symbol not in {case.symbol for case in EODHD_CASES} for symbol in symbols
    ):
        raise RuntimeError("Early terminal-history fetch ledger is duplicated/unknown.")
    return value


def _write_fetch_ledger(root: Path, value: Mapping[str, Any]) -> None:
    write_atomic(
        _fetch_ledger_path(root),
        json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, indent=2
        ).encode()
        + b"\n",
    )


def _parse_eodhd_response(
    case: HistoryCase,
    content: bytes,
    *,
    source_hash: str,
    retrieved_at: str,
) -> pd.DataFrame:
    if _sha256(content) != source_hash:
        raise ValueError(f"{case.symbol} EODHD response hash changed.")
    try:
        payload = json.loads(content)
    except Exception as exc:
        raise ValueError(f"{case.symbol} EODHD response is invalid JSON.") from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(f"{case.symbol} EODHD EOD response is not a row list.")
    expected_sessions = _request_sessions(case)
    if len(payload) != len(expected_sessions):
        raise ValueError(
            f"{case.symbol} EODHD response row count is not exact: "
            f"expected={len(expected_sessions)}, observed={len(payload)}."
        )
    observed_sessions = tuple(_text(item.get("date")) for item in payload)
    if observed_sessions != expected_sessions:
        raise ValueError(
            f"{case.symbol} EODHD response does not contain the exact bounded XNYS inventory."
        )
    records: list[dict[str, Any]] = []
    for item in payload:
        required = ("open", "high", "low", "close", "volume", "adjusted_close")
        if any(key not in item for key in required):
            raise ValueError(f"{case.symbol} EODHD response lacks raw OHLCV fields.")
        try:
            values = {key: float(item[key]) for key in required}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{case.symbol} EODHD response contains nonnumeric OHLCV."
            ) from exc
        if not math.isclose(
            values["adjusted_close"], values["close"], abs_tol=1e-12
        ):
            raise ValueError(
                f"{case.symbol} EOD-only response signals an undisclosed historical "
                "adjustment; dividends/splits must be audited before apply."
            )
        records.append(
            {
                "security_id": case.security_id,
                "session": _text(item["date"]),
                "open": values["open"],
                "high": values["high"],
                "low": values["low"],
                "close": values["close"],
                "volume": values["volume"],
                "currency": "USD",
                "source": "eodhd_eod",
                "retrieved_at": retrieved_at,
                "source_hash": source_hash,
                "source_url": case.request_url,
            }
        )
    frame = pd.DataFrame(records)
    _validate_ohlcv(frame, label=f"{case.symbol} EODHD")
    return frame.loc[
        :,
        list(dataset_spec("daily_price_raw").required_columns) + ["source_url"],
    ]


def _write_response_cache(
    root: Path,
    case: HistoryCase,
    *,
    content: bytes,
    retrieved_at: str,
    http_status: int,
    content_type: str,
    budget_usage_after_claim: int,
) -> None:
    path = _response_cache_path(root, case)
    payload = {
        "schema": "us_early_terminal_history_eod_response/v1",
        "request_inventory_sha256": request_inventory_sha256(),
        "symbol": case.symbol,
        "security_id": case.security_id,
        "provider_symbol": case.provider_symbol,
        "source_url": case.request_url,
        "retrieved_at": retrieved_at,
        "http_status": int(http_status),
        "content_type": content_type,
        "content_sha256": _sha256(content),
        "content_base64": base64.b64encode(content).decode("ascii"),
        "budget_usage_after_claim": int(budget_usage_after_claim),
    }
    payload_bytes = _canonical_json(payload)
    wrapper = _canonical_json(
        {
            "payload_sha256": _sha256(payload_bytes),
            "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
        }
    )
    compressed = gzip.compress(wrapper, mtime=0)
    if path.is_file():
        if path.read_bytes() != compressed:
            raise RuntimeError(
                f"Immutable {case.symbol} EODHD response cache already differs."
            )
        return
    write_atomic(path, compressed)


def _read_response_cache(
    root: Path,
    case: HistoryCase,
) -> tuple[pd.DataFrame, EvidenceArtifact, Mapping[str, Any]] | None:
    path = _response_cache_path(root, case)
    if not path.is_file():
        return None
    try:
        wrapper = json.loads(gzip.decompress(path.read_bytes()))
        payload_bytes = base64.b64decode(
            wrapper["payload_base64"], validate=True
        )
        if _sha256(payload_bytes) != _text(wrapper.get("payload_sha256")):
            raise ValueError("payload wrapper hash mismatch")
        payload = json.loads(payload_bytes)
        content = base64.b64decode(payload["content_base64"], validate=True)
    except Exception as exc:
        raise ValueError(
            f"Immutable {case.symbol} EODHD response cache is invalid."
        ) from exc
    if (
        payload.get("schema") != "us_early_terminal_history_eod_response/v1"
        or payload.get("request_inventory_sha256") != request_inventory_sha256()
        or _text(payload.get("symbol")) != case.symbol
        or _text(payload.get("security_id")) != case.security_id
        or _text(payload.get("provider_symbol")) != case.provider_symbol
        or _text(payload.get("source_url")) != case.request_url
        or int(payload.get("http_status", 0)) != 200
        or "json" not in _text(payload.get("content_type")).lower()
        or _sha256(content) != _text(payload.get("content_sha256"))
        or int(payload.get("budget_usage_after_claim", 0)) <= 0
    ):
        raise ValueError(f"{case.symbol} EODHD response cache metadata changed.")
    source_hash = _sha256(content)
    retrieved_at = _text(payload.get("retrieved_at"))
    prices = _parse_eodhd_response(
        case,
        content,
        source_hash=source_hash,
        retrieved_at=retrieved_at,
    )
    artifact = EvidenceArtifact(
        dataset="eodhd_eod_history_supplement",
        source="eodhd_eod",
        source_url=case.request_url,
        retrieved_at=retrieved_at,
        content=content,
        content_type="application/json",
        effective_date="2026-07-15",
        suffix="json",
    )
    return prices, artifact, payload


class EodOnlySource(Protocol):
    def claim(self) -> int: ...

    def request(self, case: HistoryCase) -> tuple[bytes, str, int, str]: ...


class SingleAttemptEodhdEodSource:
    """One budget claim and one HTTP request; deliberately no retry loop."""

    def __init__(self, client: EodhdClient | None = None):
        self.client = client or EodhdClient()

    def claim(self) -> int:
        return self.client.budget.claim()

    def request(self, case: HistoryCase) -> tuple[bytes, str, int, str]:
        endpoint = f"/eod/{case.provider_symbol}"
        try:
            response = self.client.session.get(
                self.client.base_url + endpoint,
                params={
                    "from": case.supplement_start,
                    "to": EODHD_OVERLAP_SESSION,
                    "api_token": self.client.token,
                    "fmt": "json",
                },
                timeout=120,
            )
            status = int(getattr(response, "status_code", 0))
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            status = int(getattr(getattr(exc, "response", None), "status_code", 0) or 0)
            detail = f"HTTP {status}" if status else type(exc).__name__
            raise RuntimeError(
                f"EODHD single EOD attempt failed for {case.provider_symbol}: {detail}."
            ) from None
        if not isinstance(payload, list):
            raise RuntimeError(
                f"EODHD single EOD response is not a list for {case.provider_symbol}."
            )
        content = _canonical_json(payload)
        return (
            content,
            utc_now_iso(),
            status,
            _text(getattr(response, "headers", {}).get("Content-Type"))
            or "application/json",
        )


def _identity_manifest_artifact(
    case: HistoryCase,
    *,
    primary: EvidenceArtifact,
    cross_validation: Mapping[str, Any],
) -> EvidenceArtifact:
    content = _canonical_json(
        {
            "schema": "reviewed_early_terminal_history_identity/v1",
            "reviewed_at": REVIEWED_AT,
            "security_id": case.security_id,
            "symbol": case.symbol,
            "provider_symbol": case.provider_symbol,
            "name": case.name,
            "exchange": case.exchange,
            "old_identity": {
                "active_from": case.old_active_from,
                "history_from": case.old_history_from,
                "active_to": case.active_to,
                "history_to": case.history_to,
                "source": case.old_identity_source,
                "source_url": case.old_identity_url,
                "source_hash": case.old_identity_hash,
                "retrieved_at": case.old_identity_retrieved_at,
            },
            "new_identity": {
                "active_from": case.supplement_start,
                "history_from": case.supplement_start,
                "active_to": case.active_to,
                "history_to": case.history_to,
            },
            "terminal_binding": {
                "terminal_session": case.terminal_session,
                "terminal_event_id": case.terminal_event_id,
                "terminal_action_type": case.terminal_action_type,
                "terminal_action_date": case.terminal_action_date,
                "lifecycle_candidate_id": case.lifecycle_candidate_id,
            },
            "price_evidence": {
                "source_url": primary.source_url,
                "source_hash": primary.source_hash,
                "missing_sessions": list(_missing_sessions(case)),
                "terminal_window_sessions": list(_terminal_window(case)),
                "cross_validation": dict(cross_validation),
            },
            "provider_uncertainty": case.provider_uncertainty,
        }
    )
    return EvidenceArtifact(
        dataset="reviewed_early_terminal_history_identity",
        source=IDENTITY_REPAIR_SOURCE,
        source_url=primary.source_url,
        retrieved_at=primary.retrieved_at,
        content=content,
        content_type="application/json",
        effective_date="2026-07-15",
        suffix="json",
    )


def _load_supplemental_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> tuple[dict[str, SupplementalEvidence], tuple[str, ...]]:
    output: dict[str, SupplementalEvidence] = {}
    missing: list[str] = []
    for case in EODHD_CASES:
        cached = _read_response_cache(repository.root, case)
        if cached is None and case is SWY_CASE:
            reviewed = _read_swy_wiki_cache(repository, archive)
            if reviewed is not None:
                prices, primary, provenance, cross = reviewed
                identity = _identity_manifest_artifact(
                    case, primary=primary, cross_validation=cross
                )
                output[case.symbol] = SupplementalEvidence(
                    case=case,
                    prices=prices,
                    primary_artifact=primary,
                    identity_artifact=identity,
                    all_artifacts=(primary, provenance, identity),
                    overlap_rows=1,
                    cross_validation=cross,
                )
                continue
        if cached is None:
            missing.append(case.symbol)
            continue
        prices, primary, payload = cached
        cross = {
            "status": "passed",
            "provider": "eodhd",
            "endpoint": "eod",
            "expected_response_rows": len(_request_sessions(case)),
            "insert_rows": case.missing_session_count,
            "overlap_session": EODHD_OVERLAP_SESSION,
            "overlap_rows": 1,
            "adjusted_close_equals_close": True,
            "budget_usage_after_claim": int(payload["budget_usage_after_claim"]),
        }
        identity = _identity_manifest_artifact(
            case, primary=primary, cross_validation=cross
        )
        output[case.symbol] = SupplementalEvidence(
            case=case,
            prices=prices,
            primary_artifact=primary,
            identity_artifact=identity,
            all_artifacts=(primary, identity),
            overlap_rows=1,
            cross_validation=cross,
        )
    cov_prices, cov_artifacts, cov_cross = _load_cov_primary_and_crosscheck(
        repository, archive
    )
    cov_identity = _identity_manifest_artifact(
        COV_CASE, primary=cov_artifacts[0], cross_validation=cov_cross
    )
    output[COV_CASE.symbol] = SupplementalEvidence(
        case=COV_CASE,
        prices=cov_prices,
        primary_artifact=cov_artifacts[0],
        identity_artifact=cov_identity,
        all_artifacts=(*cov_artifacts, cov_identity),
        overlap_rows=COV_CASE.old_price_rows,
        cross_validation=cov_cross,
    )
    return output, tuple(missing)


def _one_target_row(
    frame: pd.DataFrame,
    case: HistoryCase,
    *,
    symbol_history: bool = False,
) -> pd.Series:
    rows = frame.loc[frame["security_id"].astype(str).eq(case.security_id)].copy()
    if symbol_history:
        rows = rows.loc[rows["symbol"].astype(str).eq(case.symbol)]
    if len(rows) != 1:
        raise ValueError(f"{case.symbol} exact identity row is absent/duplicated.")
    return rows.iloc[0]


def _validate_terminal_binding(
    frames: Mapping[str, pd.DataFrame], case: HistoryCase
) -> None:
    actions = frames["corporate_actions"]
    action = actions.loc[actions["event_id"].astype(str).eq(case.terminal_event_id)]
    if len(action) != 1:
        raise ValueError(f"{case.symbol} terminal action binding changed.")
    action_row = action.iloc[0]
    if (
        _text(action_row.get("security_id")) != case.security_id
        or _text(action_row.get("action_type")) != case.terminal_action_type
        or _date(action_row.get("effective_date")) != case.terminal_action_date
        or _date(action_row.get("ex_date")) != case.terminal_action_date
        or bool(action_row.get("official")) is not True
    ):
        raise ValueError(f"{case.symbol} official terminal action drifted.")
    resolutions = frames["lifecycle_resolutions"]
    resolution = resolutions.loc[
        resolutions["candidate_id"].astype(str).eq(case.lifecycle_candidate_id)
    ]
    if len(resolution) != 1:
        raise ValueError(f"{case.symbol} lifecycle resolution binding changed.")
    resolution_row = resolution.iloc[0]
    if (
        _text(resolution_row.get("security_id")) != case.security_id
        or _text(resolution_row.get("symbol")) != case.symbol
        or _date(resolution_row.get("last_price_date")) != case.terminal_session
        or _text(resolution_row.get("resolution")) != "applied"
        or _text(resolution_row.get("event_id")) != case.terminal_event_id
    ):
        raise ValueError(f"{case.symbol} applied lifecycle resolution drifted.")


def _identity_state(
    master: pd.DataFrame,
    history: pd.DataFrame,
    case: HistoryCase,
    evidence: SupplementalEvidence,
) -> str:
    master_row = _one_target_row(master, case)
    history_row = _one_target_row(history, case, symbol_history=True)
    for row, expected in (
        (master_row, {"primary_symbol": case.symbol, "exchange": case.exchange}),
        (history_row, {"symbol": case.symbol, "exchange": case.exchange}),
    ):
        if any(_text(row.get(key)) != value for key, value in expected.items()):
            raise ValueError(f"{case.symbol} identity symbol/exchange changed.")
    if (
        _text(master_row.get("provider_symbol")) != case.provider_symbol
        or _text(master_row.get("action_provider_symbol")) != case.provider_symbol
        or _text(master_row.get("name")) != case.name
        or _date(master_row.get("active_to")) != case.active_to
        or _date(history_row.get("effective_to")) != case.history_to
    ):
        raise ValueError(f"{case.symbol} provider/name/end identity changed.")
    old = bool(
        _date(master_row.get("active_from")) == case.old_active_from
        and _date(history_row.get("effective_from")) == case.old_history_from
        and all(
            _text(row.get("source")) == case.old_identity_source
            and _text(row.get("source_url")) == case.old_identity_url
            and _text(row.get("source_hash")) == case.old_identity_hash
            and _text(row.get("retrieved_at")) == case.old_identity_retrieved_at
            for row in (master_row, history_row)
        )
    )
    repaired = bool(
        _date(master_row.get("active_from")) == case.supplement_start
        and _date(history_row.get("effective_from")) == case.supplement_start
        and all(
            _text(row.get("source")) == IDENTITY_REPAIR_SOURCE
            and _text(row.get("source_url"))
            == evidence.identity_artifact.source_url
            and _text(row.get("source_hash"))
            == evidence.identity_artifact.source_hash
            and _text(row.get("retrieved_at"))
            == evidence.identity_artifact.retrieved_at
            for row in (master_row, history_row)
        )
    )
    if old == repaired:
        raise ValueError(f"{case.symbol} identity is neither exact old nor repaired.")
    return "old" if old else "repaired"


def _expected_old_sessions(case: HistoryCase) -> tuple[str, ...]:
    return _xnys_sessions(case.old_active_from, case.terminal_session)


def _validate_rows_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label: str,
) -> None:
    fields = ("session", "open", "high", "low", "close", "volume")
    one = left.loc[:, fields].sort_values("session", kind="stable").reset_index(drop=True)
    two = right.loc[:, fields].sort_values("session", kind="stable").reset_index(drop=True)
    if tuple(one["session"].astype(str)) != tuple(two["session"].astype(str)):
        raise ValueError(f"{label} session inventory changed.")
    for field in fields[1:]:
        a = pd.to_numeric(one[field], errors="coerce")
        b = pd.to_numeric(two[field], errors="coerce")
        if a.isna().any() or b.isna().any() or not bool(
            (a - b).abs().le(1e-12).all()
        ):
            raise ValueError(f"{label} {field} overlap differs.")


def _validate_swy_overlap(
    stored: pd.DataFrame,
    wiki: pd.DataFrame,
) -> None:
    if len(stored) != 1 or len(wiki) != 1:
        raise ValueError("SWY stored/WIKI overlap inventory changed.")
    if _text(stored.iloc[0].get("session")) != EODHD_OVERLAP_SESSION or _text(
        wiki.iloc[0].get("session")
    ) != EODHD_OVERLAP_SESSION:
        raise ValueError("SWY stored/WIKI overlap session changed.")
    for field in ("open", "high", "low", "close"):
        if not math.isclose(
            float(stored.iloc[0][field]),
            float(wiki.iloc[0][field]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"SWY stored/WIKI overlap {field} differs.")
    stored_volume = float(stored.iloc[0]["volume"])
    wiki_volume = float(wiki.iloc[0]["volume"])
    volume_delta = abs(stored_volume - wiki_volume)
    if volume_delta > 10_000 or volume_delta / stored_volume > 0.01:
        raise ValueError("SWY stored/WIKI overlap volume differs materially.")


def _price_state(
    prices: pd.DataFrame,
    case: HistoryCase,
    evidence: SupplementalEvidence,
) -> str:
    rows = prices.loc[prices["security_id"].astype(str).eq(case.security_id)].copy()
    rows["_session"] = pd.to_datetime(rows["session"], errors="coerce").dt.date.astype(str)
    if rows["_session"].duplicated().any():
        raise ValueError(f"{case.symbol} prices duplicate a session.")
    sessions = tuple(rows.sort_values("_session", kind="stable")["_session"])
    old_sessions = _expected_old_sessions(case)
    terminal_sessions = _terminal_window(case)
    if sessions == old_sessions and len(rows) == case.old_price_rows:
        if (
            set(rows["source"].astype(str)) != {case.old_price_source}
            or set(rows["source_hash"].astype(str)) != {case.old_price_hash}
            or set(rows["retrieved_at"].astype(str))
            != {case.old_price_retrieved_at}
        ):
            raise ValueError(f"{case.symbol} old stored price provenance changed.")
        if case is COV_CASE:
            overlap = evidence.prices.loc[
                evidence.prices["session"].astype(str).isin(old_sessions)
            ]
            _validate_rows_equal(rows, overlap, label="COV old/DirectIndex")
        else:
            overlap = evidence.prices.loc[
                evidence.prices["session"].astype(str).eq(EODHD_OVERLAP_SESSION)
            ]
            existing = rows.loc[rows["_session"].eq(EODHD_OVERLAP_SESSION)]
            if case is SWY_CASE and evidence.primary_artifact.source == SWY_WIKI_SOURCE:
                _validate_swy_overlap(existing, overlap)
            else:
                _validate_rows_equal(
                    existing, overlap, label=f"{case.symbol} EODHD identity"
                )
        return "old"
    if sessions != terminal_sessions or len(rows) != TERMINAL_WINDOW_SESSIONS:
        raise ValueError(f"{case.symbol} prices are neither exact old nor repaired.")
    missing = rows.loc[rows["_session"].isin(_missing_sessions(case))]
    expected_missing = evidence.prices.loc[
        evidence.prices["session"].astype(str).isin(_missing_sessions(case))
    ]
    _validate_rows_equal(
        missing,
        expected_missing,
        label=f"{case.symbol} repaired supplemental evidence",
    )
    if (
        len(missing) != case.missing_session_count
        or set(missing["source_hash"].astype(str))
        != {evidence.primary_artifact.source_hash}
        or set(missing["source_url"].astype(str))
        != {evidence.primary_artifact.source_url}
    ):
        raise ValueError(f"{case.symbol} repaired rows lost exact provenance.")
    if case is COV_CASE:
        _validate_rows_equal(
            rows,
            evidence.prices,
            label="COV repaired DirectIndex evidence",
        )
    else:
        retained = rows.loc[rows["_session"].isin(_expected_old_sessions(case))]
        if (
            len(retained) != case.old_price_rows
            or set(retained["source"].astype(str)) != {case.old_price_source}
            or set(retained["source_hash"].astype(str)) != {case.old_price_hash}
            or set(retained["retrieved_at"].astype(str))
            != {case.old_price_retrieved_at}
        ):
            raise ValueError(f"{case.symbol} retained price provenance changed.")
        overlap = evidence.prices.loc[
            evidence.prices["session"].astype(str).eq(EODHD_OVERLAP_SESSION)
        ]
        existing = retained.loc[
            retained["_session"].eq(EODHD_OVERLAP_SESSION)
        ]
        if case is SWY_CASE and evidence.primary_artifact.source == SWY_WIKI_SOURCE:
            _validate_swy_overlap(existing, overlap)
        else:
            _validate_rows_equal(
                existing, overlap, label=f"{case.symbol} repaired EODHD identity"
            )
    return "repaired"


def _cov_dividend_state(actions: pd.DataFrame, evidence: SupplementalEvidence) -> str:
    rows = actions.loc[actions["event_id"].astype(str).eq(COV_DIVIDEND_EVENT_ID)]
    if rows.empty:
        return "old"
    if len(rows) != 1:
        raise ValueError("COV 2014 dividend is duplicated.")
    row = rows.iloc[0]
    wiki = evidence.all_artifacts[1]
    if not (
        _text(row.get("security_id")) == COV_CASE.security_id
        and _text(row.get("action_type")) == "cash_dividend"
        and _date(row.get("effective_date")) == COV_DIVIDEND_DATE
        and _date(row.get("ex_date")) == COV_DIVIDEND_DATE
        and math.isclose(float(row.get("cash_amount")), COV_DIVIDEND_AMOUNT)
        and _text(row.get("currency")) == "USD"
        and bool(row.get("official")) is False
        and _text(row.get("source")) == COV_DIVIDEND_SOURCE
        and _text(row.get("source_url")) == wiki.source_url
        and _text(row.get("source_hash")) == wiki.source_hash
    ):
        raise ValueError("COV 2014 dividend evidence changed.")
    return "repaired"


def _swy_dividend_state(actions: pd.DataFrame, evidence: SupplementalEvidence) -> str:
    rows = actions.loc[actions["event_id"].astype(str).eq(SWY_DIVIDEND_EVENT_ID)]
    if rows.empty:
        return "old"
    if len(rows) != 1:
        raise ValueError("SWY 2014 dividend is duplicated.")
    row = rows.iloc[0]
    primary = evidence.primary_artifact
    if not (
        _text(row.get("security_id")) == SWY_CASE.security_id
        and _text(row.get("action_type")) == "cash_dividend"
        and _date(row.get("effective_date")) == SWY_DIVIDEND_DATE
        and _date(row.get("ex_date")) == SWY_DIVIDEND_DATE
        and math.isclose(float(row.get("cash_amount")), SWY_DIVIDEND_AMOUNT)
        and _text(row.get("currency")) == "USD"
        and bool(row.get("official")) is False
        and _text(row.get("source")) == SWY_DIVIDEND_SOURCE
        and _text(row.get("source_url")) == primary.source_url
        and _text(row.get("source_hash")) == primary.source_hash
    ):
        raise ValueError("SWY 2014 dividend evidence changed.")
    return "repaired"


def _case_states(
    frames: Mapping[str, pd.DataFrame],
    evidence: Mapping[str, SupplementalEvidence],
) -> dict[str, str]:
    states: dict[str, str] = {}
    for case in CASES:
        _validate_terminal_binding(frames, case)
        identity = _identity_state(
            frames["security_master"], frames["symbol_history"], case, evidence[case.symbol]
        )
        price = _price_state(frames["daily_price_raw"], case, evidence[case.symbol])
        if case is COV_CASE:
            action = _cov_dividend_state(
                frames["corporate_actions"], evidence[case.symbol]
            )
        elif case is SWY_CASE:
            action = _swy_dividend_state(
                frames["corporate_actions"], evidence[case.symbol]
            )
        else:
            action = identity
        if len({identity, price, action}) != 1:
            raise RuntimeError(
                f"{case.symbol} history supplement is partially applied: "
                f"identity={identity}, price={price}, action={action}."
            )
        states[case.symbol] = identity
    if len(set(states.values())) != 1:
        raise RuntimeError(f"Early terminal-history batch is partially applied: {states}.")
    return states


def _rewrite_identity(
    frame: pd.DataFrame,
    evidence: Mapping[str, SupplementalEvidence],
    *,
    history: bool,
) -> pd.DataFrame:
    output = frame.copy()
    for case in CASES:
        mask = output["security_id"].astype(str).eq(case.security_id)
        if history:
            mask &= output["symbol"].astype(str).eq(case.symbol)
        if int(mask.sum()) != 1:
            raise ValueError(f"{case.symbol} identity rewrite target changed.")
        artifact = evidence[case.symbol].identity_artifact
        output.loc[mask, "effective_from" if history else "active_from"] = (
            case.supplement_start
        )
        output.loc[mask, "source"] = IDENTITY_REPAIR_SOURCE
        output.loc[mask, "source_url"] = artifact.source_url
        output.loc[mask, "source_hash"] = artifact.source_hash
        output.loc[mask, "retrieved_at"] = artifact.retrieved_at
    spec = dataset_spec("symbol_history" if history else "security_master")
    return output.loc[:, list(spec.required_columns) + [
        column for column in output.columns if column not in spec.required_columns
    ]]


def _rewrite_prices(
    prices: pd.DataFrame, evidence: Mapping[str, SupplementalEvidence]
) -> pd.DataFrame:
    columns = list(dataset_spec("daily_price_raw").required_columns)
    if "source_url" in prices.columns:
        columns.append("source_url")
    additions: list[pd.DataFrame] = []
    for case in CASES:
        source = evidence[case.symbol].prices.copy()
        additions.append(
            source.loc[
                source["session"].astype(str).isin(_missing_sessions(case)),
                columns,
            ]
        )
    output = pd.concat([prices, *additions], ignore_index=True)
    output["_session"] = pd.to_datetime(output["session"], errors="coerce")
    output = output.sort_values(["security_id", "_session"], kind="stable")
    output = output.drop(columns="_session").reset_index(drop=True)
    return output


def _rewrite_actions(
    actions: pd.DataFrame,
    evidence: Mapping[str, SupplementalEvidence],
) -> pd.DataFrame:
    cov_evidence = evidence[COV_CASE.symbol]
    wiki = cov_evidence.all_artifacts[1]
    cov_row = {
        "event_id": COV_DIVIDEND_EVENT_ID,
        "security_id": COV_CASE.security_id,
        "action_type": "cash_dividend",
        "effective_date": COV_DIVIDEND_DATE,
        "ex_date": COV_DIVIDEND_DATE,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": COV_DIVIDEND_AMOUNT,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_url": wiki.source_url,
        "source_kind": "provider_crosscheck",
        "source": COV_DIVIDEND_SOURCE,
        "retrieved_at": wiki.retrieved_at,
        "source_hash": wiki.source_hash,
    }
    swy_primary = evidence[SWY_CASE.symbol].primary_artifact
    swy_row = {
        "event_id": SWY_DIVIDEND_EVENT_ID,
        "security_id": SWY_CASE.security_id,
        "action_type": "cash_dividend",
        "effective_date": SWY_DIVIDEND_DATE,
        "ex_date": SWY_DIVIDEND_DATE,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": SWY_DIVIDEND_AMOUNT,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_url": swy_primary.source_url,
        "source_kind": "provider_crosscheck",
        "source": SWY_DIVIDEND_SOURCE,
        "retrieved_at": swy_primary.retrieved_at,
        "source_hash": swy_primary.source_hash,
    }
    output = pd.concat(
        [actions, pd.DataFrame([cov_row, swy_row])], ignore_index=True
    )
    output["_date"] = pd.to_datetime(output["effective_date"], errors="coerce")
    output = output.sort_values(["_date", "event_id"], kind="stable")
    return output.drop(columns="_date").reset_index(drop=True)


def _artifact_archive_row(artifact: EvidenceArtifact) -> dict[str, Any]:
    return {
        "archive_id": artifact.archive_id,
        "dataset": artifact.dataset,
        "object_path": artifact.object_path,
        "content_type": artifact.content_type,
        "effective_date": artifact.effective_date,
        "source": artifact.source,
        "source_url": artifact.source_url,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
    }


def _rewrite_archive(
    archive: pd.DataFrame,
    artifacts: tuple[EvidenceArtifact, ...],
) -> tuple[pd.DataFrame, dict[str, bytes], int]:
    output = archive.copy()
    objects: dict[str, bytes] = {}
    added = 0
    for artifact in artifacts:
        row = _artifact_archive_row(artifact)
        existing = output.loc[
            output["archive_id"].astype(str).eq(artifact.archive_id)
        ]
        if existing.empty:
            output = pd.concat([output, pd.DataFrame([row])], ignore_index=True)
            added += 1
        elif len(existing) != 1 or any(
            _text(existing.iloc[0].get(key)) != _text(value)
            for key, value in row.items()
        ):
            raise ValueError(
                f"Content-addressed archive ID conflicts: {artifact.archive_id}."
            )
        objects[artifact.object_path] = artifact.content
    output = output.sort_values("archive_id", kind="stable").reset_index(drop=True)
    return output, objects, added


def _adjustment_source_version(daily_version: str, action_version: str) -> str:
    digest = _sha256(f"{daily_version}|{action_version}".encode())
    return f"early-terminal-history:{digest}"


def _new_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    return {
        dataset: (
            f"early-terminal-history-{release.completed_session}-{token}-{dataset}"
        )
        for dataset in WRITE_DATASETS
    }


def _pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}.")
        output[dataset] = etag
    return output


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


def _snapshot_error_fingerprint(report) -> tuple[str, ...]:
    return tuple(
        sorted(
            json.dumps(
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity,
                    "row_count": issue.row_count,
                    "fingerprints": list(issue.fingerprints),
                },
                sort_keys=True,
            )
            for issue in report.issues
            if issue.severity == "error"
        )
    )


def _factor_economics_equal(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    fields = ["security_id", "session", "split_factor", "total_return_factor"]
    left = old.loc[:, fields].copy()
    right = new.loc[:, fields].copy()
    left["session"] = pd.to_datetime(left["session"]).dt.normalize()
    right["session"] = pd.to_datetime(right["session"]).dt.normalize()
    joined = left.merge(
        right,
        on=["security_id", "session"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    if len(joined) != len(left):
        return False
    for field in ("split_factor", "total_return_factor"):
        if not bool(
            (
                pd.to_numeric(joined[f"{field}_old"])
                - pd.to_numeric(joined[f"{field}_new"])
            )
            .abs()
            .le(1e-12)
            .all()
        ):
            return False
    return True


def _budget_snapshot(root: Path) -> Mapping[str, Any]:
    path = Path(root) / "state/eodhd_call_budget.json"
    if not path.is_file():
        return {"state_present": False}
    try:
        value = json.loads(path.read_bytes())
    except Exception:
        return {"state_present": True, "state_valid": False}
    return {
        "state_present": True,
        "state_valid": isinstance(value, dict),
        "period": _text(value.get("period")) if isinstance(value, dict) else "",
        "used": int(value.get("used", 0)) if isinstance(value, dict) else 0,
        "daily_limit": int(value.get("daily_limit", 0))
        if isinstance(value, dict)
        else 0,
        "reserve": int(value.get("reserve", 0)) if isinstance(value, dict) else 0,
    }


def _awaiting_summary(
    *,
    release: DataRelease,
    evidence: Mapping[str, SupplementalEvidence],
    missing: tuple[str, ...],
    root: Path,
) -> dict[str, Any]:
    ledger = _read_fetch_ledger(root)
    attempted = {_text(item.get("symbol")) for item in ledger["attempts"]}
    resolved = sorted(set(case.symbol for case in EODHD_CASES) - set(missing))
    eodhd_cached = sorted(
        symbol
        for symbol, item in evidence.items()
        if symbol != COV_CASE.symbol
        and item.primary_artifact.source == "eodhd_eod"
    )
    offline_reviewed = sorted(
        symbol
        for symbol, item in evidence.items()
        if symbol != COV_CASE.symbol
        and item.primary_artifact.source != "eodhd_eod"
    )
    missing_cases = [case for case in EODHD_CASES if case.symbol in set(missing)]
    remaining_attempt_capacity = MAX_EODHD_HTTP_ATTEMPTS - len(ledger["attempts"])
    return {
        "status": "awaiting_eodhd_fetch",
        "apply_ready": False,
        "base_release_version": release.version,
        "request_inventory_sha256": request_inventory_sha256(),
        "requests": list(request_inventory()),
        "eodhd_call_cap": MAX_EODHD_HTTP_ATTEMPTS,
        "eodhd_expected_calls": len(missing),
        "eodhd_http_attempts_this_plan": 0,
        "eodhd_ledger_attempts": len(ledger["attempts"]),
        "eodhd_remaining_attempt_capacity": remaining_attempt_capacity,
        "eodhd_attempted_symbols": sorted(attempted),
        "resolved_symbols": resolved,
        "cached_symbols": eodhd_cached,
        "offline_reviewed_symbols": offline_reviewed,
        "missing_symbols": list(missing),
        "expected_eodhd_response_rows": sum(
            len(_request_sessions(case)) for case in missing_cases
        ),
        "expected_eodhd_insert_rows": sum(
            case.missing_session_count for case in missing_cases
        ),
        "resolved_eodhd_response_rows": sum(
            len(_request_sessions(case))
            for case in EODHD_CASES
            if case.symbol in eodhd_cached
        ),
        "resolved_eodhd_insert_rows": sum(
            case.missing_session_count
            for case in EODHD_CASES
            if case.symbol in eodhd_cached
        ),
        "swy_wiki_insert_rows": (
            SWY_CASE.missing_session_count if SWY_CASE.symbol in offline_reviewed else 0
        ),
        "cov_insert_rows": COV_CASE.missing_session_count,
        "total_insert_rows": 118,
        "terminal_window_sessions_per_security": TERMINAL_WINDOW_SESSIONS,
        "cov_cross_validation": dict(evidence["COV"].cross_validation),
        "legacy_agn_provider_uncertainty": EODHD_CASES[-1].provider_uncertainty,
        "budget_snapshot": dict(_budget_snapshot(root)),
        "expected_downstream_yahoo_refetches": 5,
        "network_accessed": False,
        "r2_accessed": False,
        "writes_performed": False,
    }


def prepare_repair(repository: LocalDatasetRepository) -> PreparedRepair:
    _static_contract()
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing_datasets = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing_datasets:
        raise RuntimeError(
            "Current release lacks early-history datasets: "
            + ", ".join(missing_datasets)
        )
    pointer_etags = _pointer_etags(repository, release)
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    evidence, missing = _load_supplemental_evidence(
        repository, frames["source_archive"]
    )
    for case in CASES:
        _validate_terminal_binding(frames, case)
        # Every resolved source is checked immediately; unresolved EODHD
        # targets receive only the exact pre-fetch inventory checks.
        if case.symbol in evidence:
            identity = _identity_state(
                frames["security_master"],
                frames["symbol_history"],
                case,
                evidence[case.symbol],
            )
            price = _price_state(
                frames["daily_price_raw"], case, evidence[case.symbol]
            )
            if case is COV_CASE:
                action = _cov_dividend_state(
                    frames["corporate_actions"], evidence[case.symbol]
                )
            elif case is SWY_CASE:
                action = _swy_dividend_state(
                    frames["corporate_actions"], evidence[case.symbol]
                )
            else:
                action = identity
            if len({identity, price, action}) != 1:
                raise RuntimeError(
                    f"{case.symbol} resolved history evidence is partially applied."
                )
        else:
            master = _one_target_row(frames["security_master"], case)
            history = _one_target_row(
                frames["symbol_history"], case, symbol_history=True
            )
            if not (
                _text(master.get("provider_symbol")) == case.provider_symbol
                and _text(master.get("name")) == case.name
                and _text(history.get("symbol")) == case.symbol
            ):
                raise ValueError(f"{case.symbol} pre-fetch identity changed.")
            prices = frames["daily_price_raw"].loc[
                frames["daily_price_raw"]["security_id"].astype(str).eq(
                    case.security_id
                )
            ]
            sessions = tuple(
                pd.to_datetime(prices["session"]).dt.date.astype(str).sort_values()
            )
            if sessions not in {
                _expected_old_sessions(case),
                _terminal_window(case),
            }:
                raise ValueError(f"{case.symbol} pre-fetch price inventory changed.")
    if missing:
        summary = _awaiting_summary(
            release=release,
            evidence=evidence,
            missing=missing,
            root=repository.root,
        )
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            planned_versions={},
            frames={},
            archive_objects={},
            summary=summary,
        )

    states = _case_states(frames, evidence)
    state = next(iter(states.values()))
    artifacts = tuple(
        artifact
        for case in CASES
        for artifact in evidence[case.symbol].all_artifacts
    )
    # Artifact IDs are globally content-addressed.  A duplicate is allowed only
    # if it is byte-identical, which the mapping construction verifies.
    artifact_by_path: dict[str, bytes] = {}
    for artifact in artifacts:
        previous = artifact_by_path.get(artifact.object_path)
        if previous is not None and previous != artifact.content:
            raise ValueError("Early-history artifact path has divergent bytes.")
        artifact_by_path[artifact.object_path] = artifact.content

    if state == "repaired":
        if PRIVATE_WIKI_WARNING not in release.warnings:
            raise RuntimeError("Repaired SWY WIKI history lacks private-use warning.")
        archive, objects, added_archive_rows = _rewrite_archive(
            frames["source_archive"], artifacts
        )
        if added_archive_rows:
            raise RuntimeError(
                "Repaired history supplements lack content-addressed archive rows."
            )
        for path, content in objects.items():
            final = repository.root / path
            if not final.is_file():
                raise FileNotFoundError(f"Repaired archive object is missing: {path}")
            if _sha256(gzip.decompress(final.read_bytes())) != _sha256(content):
                raise ValueError(f"Repaired archive object changed: {path}")
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            planned_versions={},
            frames={},
            archive_objects={},
            summary={
                "status": "already_repaired",
                "apply_ready": True,
                "base_release_version": release.version,
                "target_count": len(CASES),
                "total_insert_rows": 0,
                "eodhd_http_attempts_this_plan": 0,
                "network_accessed": False,
                "r2_accessed": False,
            },
        )
    if state != "old":  # pragma: no cover - guarded by _case_states
        raise AssertionError(state)

    planned_versions = _new_versions(release)
    prices = _rewrite_prices(frames["daily_price_raw"], evidence)
    actions = _rewrite_actions(frames["corporate_actions"], evidence)
    master = _rewrite_identity(frames["security_master"], evidence, history=False)
    history = _rewrite_identity(frames["symbol_history"], evidence, history=True)
    archive, archive_objects, archive_rows_added = _rewrite_archive(
        frames["source_archive"], artifacts
    )
    lineage = _adjustment_source_version(
        planned_versions["daily_price_raw"], planned_versions["corporate_actions"]
    )
    factors = build_adjustment_factors(prices, actions, source_version=lineage)
    overrides = {
        "corporate_actions": actions,
        "daily_price_raw": prices,
        "adjustment_factors": factors,
        "security_master": master,
        "symbol_history": history,
        "source_archive": archive,
    }
    for dataset, frame in overrides.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=release.completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    candidate = _CandidateRepository(repository, release.dataset_versions, overrides)
    before_snapshot = validate_repository_snapshot(repository)
    after_snapshot = validate_repository_snapshot(candidate)
    if _snapshot_error_fingerprint(before_snapshot) != _snapshot_error_fingerprint(
        after_snapshot
    ):
        raise RuntimeError(
            "Early terminal-history supplement changed repository snapshot errors."
        )
    # Existing sessions must retain exact split/total-return economics.  The
    # new COV dividend only adjusts newly inserted pre-2015 rows.
    if not _factor_economics_equal(frames["adjustment_factors"], factors):
        raise RuntimeError(
            "Early terminal-history supplement changed existing factor economics."
        )
    candidate_frames = dict(frames)
    candidate_frames.update(overrides)
    after_states = _case_states(candidate_frames, evidence)
    if set(after_states.values()) != {"repaired"}:
        raise RuntimeError("Early terminal-history candidate is not idempotent.")
    if len(prices) - len(frames["daily_price_raw"]) != 118:
        raise RuntimeError("Early terminal-history insert count changed.")
    if len(factors) - len(frames["adjustment_factors"]) != 118:
        raise RuntimeError("Early terminal-history factor count changed.")
    if len(actions) - len(frames["corporate_actions"]) != 2:
        raise RuntimeError("Early terminal-history dividend action count changed.")
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frames=overrides,
        archive_objects=archive_objects,
        summary={
            "status": "validated_offline_plan",
            "apply_ready": True,
            "base_release_version": release.version,
            "target_count": len(CASES),
            "targets": [
                {
                    "symbol": case.symbol,
                    "security_id": case.security_id,
                    "provider_symbol": case.provider_symbol,
                    "supplement_start": case.supplement_start,
                    "terminal_session": case.terminal_session,
                    "insert_rows": case.missing_session_count,
                    "evidence_sha256": evidence[
                        case.symbol
                    ].primary_artifact.source_hash,
                    "identity_manifest_sha256": evidence[
                        case.symbol
                    ].identity_artifact.source_hash,
                }
                for case in CASES
            ],
            "request_inventory_sha256": request_inventory_sha256(),
            "eodhd_call_cap": MAX_EODHD_HTTP_ATTEMPTS,
            "eodhd_http_attempts_this_plan": 0,
            "eodhd_response_rows": 36,
            "eodhd_insert_rows": 33,
            "swy_wiki_response_rows": 42,
            "swy_wiki_insert_rows": 41,
            "cov_insert_rows": 44,
            "total_insert_rows": 118,
            "cov_dividend_rows_added": 1,
            "swy_dividend_rows_added": 1,
            "adjustment_factor_rows_added": 118,
            "existing_factor_economics_changed": 0,
            "source_archive_rows_added": archive_rows_added,
            "source_archive_object_count": len(archive_objects),
            "factor_source_version": lineage,
            "planned_versions": dict(planned_versions),
            "snapshot_errors_unchanged": True,
            "expected_downstream_yahoo_refetches": 5,
            "network_accessed": False,
            "r2_accessed": False,
            "formal_wiki_license_name": "Unknown",
            "allowed_scope": "private_internal_only",
            "publication_allowed": False,
            "redistribution_allowed": False,
        },
    )


def _assert_inputs_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, etag = repository.current_release()
    if (
        release is None
        or release.to_bytes() != prepared.release.to_bytes()
        or etag != prepared.release_etag
    ):
        raise RuntimeError(
            "Current release changed after early-history planning."
        )
    for dataset in REQUIRED_DATASETS:
        pointer, pointer_etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or pointer.version != prepared.release.dataset_versions[dataset]
            or pointer_etag != prepared.pointer_etags[dataset]
        ):
            raise RuntimeError(
                f"{dataset} pointer changed after early-history planning."
            )


@contextmanager
def _exclusive_fetch_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/us-early-terminal-history-fetch.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Early terminal-history fetch is already running.") from exc
        yield


def import_reviewed_swy_wiki(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    zip_path: Path,
) -> dict[str, Any]:
    """Materialize the exact offline SWY fallback after the one EODHD stop."""

    with _exclusive_fetch_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        release = prepared.release
        archive = repository.read_frame(
            "source_archive", release.dataset_versions["source_archive"]
        )
        existing = _read_swy_wiki_cache(repository, archive)
        if existing is not None:
            ready = prepare_repair(repository)
            return {
                **dict(ready.summary),
                "mode": "import_reviewed_swy_wiki",
                "writes_performed": False,
                "network_accessed": False,
                "eodhd_http_attempts_this_run": 0,
            }
        license_binding = _validate_swy_wiki_parent_provenance(
            repository, archive
        )
        attempt = _failed_swy_attempt(repository.root)
        extract = _extract_reviewed_swy_wiki(Path(zip_path))
        provenance = _build_swy_wiki_provenance(
            extract=extract,
            attempt=attempt,
            license_binding=license_binding,
        )
        prices, _ = _parse_swy_wiki_extract(extract)
        stored_prices = repository.read_frame(
            "daily_price_raw", release.dataset_versions["daily_price_raw"]
        )
        stored_overlap = stored_prices.loc[
            stored_prices["security_id"].astype(str).eq(SWY_CASE.security_id)
            & pd.to_datetime(stored_prices["session"], errors="coerce")
            .dt.date.astype(str)
            .eq(EODHD_OVERLAP_SESSION)
        ]
        wiki_overlap = prices.loc[
            prices["session"].astype(str).eq(EODHD_OVERLAP_SESSION)
        ]
        _validate_swy_overlap(stored_overlap, wiki_overlap)
        _write_swy_wiki_cache(
            repository.root,
            extract=extract,
            provenance=provenance,
        )
        ready = prepare_repair(repository)
        return {
            **dict(ready.summary),
            "mode": "import_reviewed_swy_wiki",
            "writes_performed": True,
            "state_cache_written": str(
                _swy_wiki_cache_path(repository.root).relative_to(repository.root)
            ),
            "swy_extract_sha256": SWY_WIKI_EXTRACT_SHA256,
            "swy_provenance_sha256": _sha256(provenance),
            "network_accessed": False,
            "eodhd_http_attempts_this_run": 0,
        }


def fetch_missing(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    source: EodOnlySource | None = None,
) -> dict[str, Any]:
    with _exclusive_fetch_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        current = prepare_repair(repository)
        _assert_inputs_unchanged(repository, current)
        if current.summary["status"] != "awaiting_eodhd_fetch":
            return {
                **dict(current.summary),
                "mode": "fetch_missing",
                "eodhd_http_attempts_this_run": 0,
            }
        missing = set(current.summary["missing_symbols"])
        ledger = _read_fetch_ledger(repository.root)
        attempted = {_text(item.get("symbol")) for item in ledger["attempts"]}
        blocked = sorted(missing & attempted)
        if blocked:
            raise RuntimeError(
                "A prior single attempt has no usable immutable cache; automatic "
                "retry is forbidden: " + ", ".join(blocked)
            )
        if len(ledger["attempts"]) + len(missing) > MAX_EODHD_HTTP_ATTEMPTS:
            raise RuntimeError("Early terminal-history batch call cap would be exceeded.")
        network = source or SingleAttemptEodhdEodSource()
        calls = 0
        results: list[dict[str, Any]] = []
        for case in EODHD_CASES:
            if case.symbol not in missing:
                continue
            _assert_inputs_unchanged(repository, current)
            attempt = {
                "symbol": case.symbol,
                "provider_symbol": case.provider_symbol,
                "source_url": case.request_url,
                "reserved_at": utc_now_iso(),
                "status": "reserved_before_budget_claim",
            }
            ledger["attempts"].append(attempt)
            _write_fetch_ledger(repository.root, ledger)
            try:
                usage = int(network.claim())
            except BaseException as exc:
                attempt.update(
                    {
                        "status": "budget_claim_failed_no_http",
                        "completed_at": utc_now_iso(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                _write_fetch_ledger(repository.root, ledger)
                raise
            attempt.update(
                {
                    "status": "budget_claimed_before_http",
                    "budget_usage_after_claim": usage,
                }
            )
            _write_fetch_ledger(repository.root, ledger)
            calls += 1
            try:
                content, retrieved_at, status, content_type = network.request(case)
                # Validate before cache publication.  A semantically invalid
                # provider result still consumes the one-attempt ledger entry.
                _parse_eodhd_response(
                    case,
                    content,
                    source_hash=_sha256(content),
                    retrieved_at=retrieved_at,
                )
                _write_response_cache(
                    repository.root,
                    case,
                    content=content,
                    retrieved_at=retrieved_at,
                    http_status=status,
                    content_type=content_type,
                    budget_usage_after_claim=usage,
                )
                attempt.update(
                    {
                        "status": "cached_valid",
                        "response_sha256": _sha256(content),
                        "response_rows": len(_request_sessions(case)),
                        "completed_at": utc_now_iso(),
                    }
                )
                results.append(
                    {
                        "symbol": case.symbol,
                        "status": "cached_valid",
                        "response_sha256": _sha256(content),
                        "response_rows": len(_request_sessions(case)),
                    }
                )
            except BaseException as exc:
                attempt.update(
                    {
                        "status": "http_or_validation_failed_no_retry",
                        "completed_at": utc_now_iso(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                _write_fetch_ledger(repository.root, ledger)
                raise
            _write_fetch_ledger(repository.root, ledger)
        ready = prepare_repair(repository)
        return {
            **dict(ready.summary),
            "mode": "fetch_missing",
            "eodhd_http_attempts_this_run": calls,
            "fetch_results": results,
        }


@contextmanager
def _exclusive_write_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved early-history recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted early-history transaction blocks writes: {journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, indent=2
        ).encode()
        + b"\n",
    )


def _write_archive_objects(
    repository: LocalDatasetRepository,
    objects: Mapping[str, bytes],
) -> None:
    for relative, content in sorted(objects.items()):
        if _sha256(content) not in Path(relative).name:
            raise RuntimeError(f"Archive object path is not content-addressed: {relative}.")
        path = repository.root / relative
        compressed = gzip.compress(content, mtime=0)
        if path.is_file():
            if path.read_bytes() != compressed:
                try:
                    observed = gzip.decompress(path.read_bytes())
                except OSError as exc:
                    raise RuntimeError(
                        f"Existing archive object is invalid gzip: {relative}."
                    ) from exc
                if observed != content:
                    raise RuntimeError(
                        f"Existing content-addressed archive object differs: {relative}."
                    )
            continue
        write_atomic(path, compressed)


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
) -> dict[str, Any]:
    manifest = repository.manifest_for_version(
        dataset, prepared.release.dataset_versions[dataset]
    )
    metadata = dict(manifest.metadata)
    metadata.update(
        {
            "operation": OPERATION,
            "input_release_version": prepared.release.version,
            "request_inventory_sha256": request_inventory_sha256(),
            "target_security_ids": [case.security_id for case in CASES],
            "inserted_price_rows": 118,
            "fetch_http_attempt_cap": MAX_EODHD_HTTP_ATTEMPTS,
            "apply_network_accessed": False,
            "r2_accessed": False,
        }
    )
    if dataset == "adjustment_factors":
        metadata.update(
            {
                "source_version": prepared.summary["factor_source_version"],
                "source_daily_price_version": prepared.planned_versions[
                    "daily_price_raw"
                ],
                "source_corporate_actions_version": prepared.planned_versions[
                    "corporate_actions"
                ],
                "inserted_rows": 118,
                "existing_economic_rows_changed": 0,
            }
        )
    elif dataset == "source_archive":
        metadata.update(
            {
                "added_rows": prepared.summary["source_archive_rows_added"],
                "content_addressed": True,
            }
        )
    return metadata


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
    old_versions: Mapping[str, str],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            expected_versions = {**dict(old_versions), **dict(planned_versions)}
            belongs = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or observed.dataset_versions == expected_versions
            if not belongs:
                raise RuntimeError(
                    f"unexpected release during early-history rollback: {observed.version}"
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
            old = old_pointer_bytes[dataset]
            if current.data != old:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected {dataset} pointer during early-history rollback: "
                        f"{pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository,
    committed: DataRelease,
    *,
    expected_out_of_scope_pointer_etags: Mapping[str, str | None],
) -> None:
    current, _ = repository.current_release()
    if current is None or current.to_bytes() != committed.to_bytes():
        raise RuntimeError("Committed early-history release is not current.")
    for dataset, version in committed.dataset_versions.items():
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"Applied early-history pointer mismatch: {dataset}.")
        if (
            dataset not in WRITE_DATASETS
            and etag != expected_out_of_scope_pointer_etags.get(dataset)
        ):
            raise RuntimeError(f"Out-of-scope pointer changed: {dataset}.")
    replay = prepare_repair(repository)
    if replay.summary["status"] != "already_repaired":
        raise RuntimeError("Applied early-history repair is not idempotent.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_write_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        current_plan = prepare_repair(repository)
        if current_plan.summary["status"] == "awaiting_eodhd_fetch":
            raise RuntimeError(
                "Early terminal-history apply is blocked until every exact EODHD "
                "or reviewed offline source passes validation."
            )
        if current_plan.summary["status"] == "already_repaired":
            return {
                **dict(current_plan.summary),
                "mode": "apply",
                "writes_performed": False,
            }
        if current_plan.summary["status"] != "validated_offline_plan":
            raise RuntimeError("Early terminal-history locked plan is not apply-ready.")
        base_snapshot_errors = _snapshot_error_fingerprint(
            validate_repository_snapshot(repository)
        )
        planned = dict(current_plan.planned_versions)
        if set(planned) != set(WRITE_DATASETS) or len(set(planned.values())) != len(
            WRITE_DATASETS
        ):
            raise RuntimeError("Prepared early-history versions are invalid.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version
                != current_plan.release.dataset_versions[dataset]
                or value.etag != current_plan.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} changed before early-history apply.")
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_early_terminal_history_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": current_plan.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": planned,
            "request_inventory_sha256": request_inventory_sha256(),
            "archive_object_hashes": sorted(
                _sha256(value) for value in current_plan.archive_objects.values()
            ),
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            _write_archive_objects(repository, current_plan.archive_objects)
            inject("after_archive_objects")
            versions = dict(current_plan.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    current_plan.frames[dataset],
                    completed_session=current_plan.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(
                        repository, current_plan, dataset
                    ),
                    expected_pointer_etag=current_plan.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict or result.manifest.version != planned[dataset]:
                    raise RuntimeError(f"Unexpected {dataset} write result.")
                versions[dataset] = result.manifest.version
                inject(f"after_write:{dataset}")
            for dataset in REQUIRED_DATASETS:
                if dataset in WRITE_DATASETS:
                    continue
                pointer, etag = repository.current_pointer(dataset)
                if (
                    pointer is None
                    or pointer.version
                    != current_plan.release.dataset_versions[dataset]
                    or etag != current_plan.pointer_etags[dataset]
                ):
                    raise RuntimeError(
                        f"Out-of-scope pointer changed during apply: {dataset}."
                    )
            committed = repository.commit_release(
                current_plan.release.completed_session,
                versions,
                quality=current_plan.release.quality,
                warnings=tuple(
                    dict.fromkeys(
                        (*current_plan.release.warnings, PRIVATE_WIKI_WARNING)
                    )
                ),
                expected_etag=current_plan.release_etag,
            )
            inject("after_release_commit")
            applied_snapshot_errors = _snapshot_error_fingerprint(
                validate_repository_snapshot(repository)
            )
            if applied_snapshot_errors != base_snapshot_errors:
                raise RuntimeError(
                    "Early terminal-history apply changed repository snapshot errors."
                )
            _assert_applied_release(
                repository,
                committed,
                expected_out_of_scope_pointer_etags=current_plan.pointer_etags,
            )
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": committed.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            return {
                **dict(current_plan.summary),
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "transaction_id": transaction_id,
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=committed.version if committed else "",
                old_versions=current_plan.release.dataset_versions,
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
                    "Early-history rollback was incomplete; recovery marker blocks "
                    f"writes: {recovery}; errors={rollback_errors}."
                ) from original
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, fetch, or atomically apply five exact early terminal-history "
            "supplements."
        )
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch-missing", action="store_true")
    mode.add_argument(
        "--import-reviewed-swy-wiki",
        type=Path,
        metavar="ZIP",
    )
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository)
    if args.import_reviewed_swy_wiki is not None:
        result = import_reviewed_swy_wiki(
            repository,
            prepared,
            zip_path=args.import_reviewed_swy_wiki,
        )
    elif args.fetch_missing:
        result = fetch_missing(repository, prepared)
    elif args.apply:
        result = apply_repair(repository, prepared)
    else:
        result = {
            **dict(prepared.summary),
            "mode": "plan",
            "writes_performed": False,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
