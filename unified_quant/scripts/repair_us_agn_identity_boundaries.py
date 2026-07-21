#!/usr/bin/env python3
"""Audit two locally evidenced Allergan boundaries and fail closed on apply.

This repair is intentionally narrower than the nine-item market-exit audit.
Two stored identity rows have enough immutable evidence for a candidate change:

* legacy Allergan Inc. (AGN) last traded on 2015-03-16; its SEC filing says
  trading was suspended before the 2015-03-17 open; and
* Allergan plc (AGN, formerly ACT) has a stored last price and official merger
  completion on 2020-05-08.

Neither candidate is safe in the current one-boundary identity model: the
securities remained S&P 500 members after their last tradable session, so an
``active_to`` trim makes the operational index replay lose its active symbol.
The current release is therefore reported as
``blocked_operational_model_conflict`` and ``--apply`` performs no writes.

The ACT -> AGN ticker-change row is *not* repaired here.  Its current source
archive object is the legacy Allergan acquisition filing and does not establish
the 2015-06-15 ticker change.  WIN, CHK, FTR, DISH, SATS, and ENDP likewise
remain fail-closed because the local archive cannot prove their complete market
tails or exact boundaries.

Plan is the default and performs no writes.  The dormant transaction path is
kept fully tested for a future model in which the candidate passes the
operational gate; it uses two inherited deltas, a global writer lock,
pointer/release compare-and-swap, a durable journal, verified rollback, and an
idempotent replay.  There is no network, EODHD, or R2 code path.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import html
import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import duckdb
import exchange_calendars as xcals
import pandas as pd

from supertrend_quant.market_store.cross_validation import (
    dataframe_sha256,
)
from supertrend_quant.market_store.lifecycle_coverage import lifecycle_candidate_id
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.validation import validate_dataset


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_YAHOO_CACHE = DEFAULT_CACHE_ROOT / "state/us_cross_validation/yahoo_chart"
OPERATION = "repair_us_agn_identity_boundaries"
TRANSACTION_DIR = "transactions/us-agn-identity-boundaries"
RECOVERY_DIR = "recovery/us-agn-identity-boundaries"
WRITE_DATASETS = ("security_master", "symbol_history")
REQUIRED_DATASETS = (
    *WRITE_DATASETS,
    "daily_price_raw",
    "adjustment_factors",
    "corporate_actions",
    "lifecycle_resolutions",
    "source_archive",
    "index_constituent_anchors",
    "index_membership_events",
)

LEGACY_AGN_ID = "US:EODHD:9f13974d-7f81-5aac-a3a7-ed1d184bd76b"
ACTAVIS_AGN_ID = "US:EODHD:79ce1c42-8ff6-5c13-b9cf-3df82c913734"
ABBV_ID = "US:EODHD:3f3cd70b-d1b0-5b4e-a702-d3ab94fc57fe"

LEGACY_EVENT_ID = (
    "66f9915d5d513afda9c2b595e79e9e3208dba3b3e211b97d4b658835c56dc263"
)
ACT_TICKER_EVENT_ID = (
    "e6768339a53c5aad1b2e882ac192bbaa8d0b1981a744b1585ff71c4d494aa2ad"
)
LATER_EVENT_ID = (
    "6bf6b19283dd5b00c4d4e0d777ec9a79ac069e98b599559b22da96162d985f9e"
)

LEGACY_OLD_ACTIVE_TO = "2015-03-22"
LEGACY_ACTIVE_TO = "2015-03-16"
LEGACY_EVENT_DATE = "2015-03-17"
LATER_OLD_ACTIVE_TO = "2020-05-11"
LATER_ACTIVE_TO = "2020-05-08"
ACT_ACTIVE_TO = "2015-06-14"
ACT_EFFECTIVE_FROM = "2015-01-01"
AGN_EFFECTIVE_FROM = "2015-06-15"


@dataclass(frozen=True)
class EvidenceSpec:
    name: str
    source_url: str
    source_hash: str
    required_text_groups: tuple[tuple[str, ...], ...]


LEGACY_EVIDENCE = EvidenceSpec(
    name="legacy_agn_merger",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/850693/"
        "000119312515096184/0001193125-15-096184.txt"
    ),
    source_hash=(
        "247cdf622803b6b29c900ed16bc6fec15ac8c5a3e1d04be98dd88a2f44976457"
    ),
    required_text_groups=(
        ("on march 17, 2015",),
        ("the merger had been completed",),
        (
            "trading of company common stock on the nyse be suspended before "
            "the opening of trading on march 17, 2015",
        ),
    ),
)
ACT_TICKER_EVIDENCE = EvidenceSpec(
    name="act_ticker_current_binding",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/850693/"
        "000119312515096184/d894643d8k.htm"
    ),
    source_hash=(
        "aa68d1a454bce099ded3bd54f982fdba31f9b70e97199591a08577bc2b6dfd0e"
    ),
    required_text_groups=(
        ("on march 17, 2015",),
        ("the merger had been completed",),
    ),
)
LATER_EVIDENCE = EvidenceSpec(
    name="later_agn_merger",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/1551152/"
        "000110465920058837/tm2018740d2_8k.htm"
    ),
    source_hash=(
        "59c028cefd03b8dba0f0ebd7f58a396ad0d412073d314433a10d707338193f90"
    ),
    required_text_groups=(
        ("on may 8, 2020",),
        ("completed the previously announced acquisition of allergan",),
        ("the scheme became effective",),
    ),
)
EVIDENCE_SPECS = (LEGACY_EVIDENCE, ACT_TICKER_EVIDENCE, LATER_EVIDENCE)

EXPECTED_PRICE_PROFILES: Mapping[str, tuple[int, str, str]] = {
    LEGACY_AGN_ID: (60, "2014-12-17", LEGACY_ACTIVE_TO),
    ACTAVIS_AGN_ID: (1_347, "2015-01-02", LATER_ACTIVE_TO),
}

BLOCKED_TARGETS: Mapping[str, str] = {
    "AGN_legacy": (
        "the SEC trading boundary is 2015-03-16, but the official S&P removal "
        "is effective 2015-03-23; shortening active_to creates four active-index "
        "sessions without a symbol identity"
    ),
    "AGN_later": (
        "the merger/last-price boundary is 2020-05-08, but stored S&P removal "
        "is effective 2020-05-12; shortening active_to creates an active-index "
        "session without a symbol identity"
    ),
    "ACT": (
        "current 2015-06-15 ticker-change event is bound to a 2015-03-17 "
        "legacy Allergan acquisition filing, not an ACT-to-AGN ticker notice"
    ),
    "WIN": "no independent exact first-OTC date/ticker and price tail",
    "CHK": "official CHKAQ dates exist but the independent CHKAQ price tail is absent",
    "FTR": "stored post-exit rows are placeholders and no independent FTRCQ tail exists",
    "DISH": "merger boundary conflicts with the stored terminal and current-price evidence",
    "SATS": "open identity has an incomplete terminal calendar and no terminal event",
    "ENDP": "ENDPQ transition is known but the complete OTC tail/cancellation boundary is absent",
}


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    logical_frames: Mapping[str, pd.DataFrame]
    deltas: Mapping[str, pd.DataFrame]
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


def _noop(_stage: str) -> None:
    return None


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
    value = _text(value)
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _plain_text(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    decoded = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", decoded)
    decoded = re.sub(r"(?s)<[^>]+>", " ", decoded)
    return " ".join(html.unescape(decoded).lower().split())


def _safe_object_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    path = (base / object_path).resolve()
    _require(path != base and base in path.parents, "Archive object path escapes cache root.")
    return path


def _verify_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    spec: EvidenceSpec,
) -> dict[str, Any]:
    rows = archive.loc[
        archive["source_url"].map(_text).eq(spec.source_url)
        & archive["source_hash"].map(_text).str.lower().eq(spec.source_hash)
    ]
    _require(len(rows) == 1, f"{spec.name} exact archive URL/hash row is missing.")
    row = rows.iloc[0]
    _require(
        _text(row.get("archive_id")).lower() == spec.source_hash
        and bool(_text(row.get("retrieved_at")))
        and spec.source_hash in _text(row.get("object_path")),
        f"{spec.name} archive metadata changed.",
    )
    path = _safe_object_path(repository.root, _text(row.get("object_path")))
    _require(path.is_file(), f"{spec.name} archive payload is missing.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise RuntimeError(f"{spec.name} archive payload is invalid gzip.") from exc
    _require(
        hashlib.sha256(payload).hexdigest() == spec.source_hash,
        f"{spec.name} archive payload hash changed.",
    )
    normalized = _plain_text(payload)
    claims = [
        any(option in normalized for option in group)
        for group in spec.required_text_groups
    ]
    _require(all(claims), f"{spec.name} required official claims are absent.")
    return {
        "source_url": spec.source_url,
        "source_hash": spec.source_hash,
        "retrieved_at": _text(row.get("retrieved_at")),
        "claims_passed": len(claims),
        "payload_bytes": len(payload),
    }


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    _require(len(rows) == 1, f"{label} row inventory is not exact: {len(rows)}.")
    return rows.iloc[0]


def _identity_rows(
    master: pd.DataFrame, history: pd.DataFrame
) -> dict[str, tuple[Any, ...]]:
    legacy_master = _one_row(
        master,
        master["security_id"].map(_text).eq(LEGACY_AGN_ID),
        "legacy AGN security_master",
    )
    later_master = _one_row(
        master,
        master["security_id"].map(_text).eq(ACTAVIS_AGN_ID),
        "Actavis/AGN security_master",
    )
    legacy_history = _one_row(
        history,
        history["security_id"].map(_text).eq(LEGACY_AGN_ID)
        & history["symbol"].map(_text).str.upper().eq("AGN"),
        "legacy AGN symbol_history",
    )
    act_history = _one_row(
        history,
        history["security_id"].map(_text).eq(ACTAVIS_AGN_ID)
        & history["symbol"].map(_text).str.upper().eq("ACT"),
        "ACT symbol_history",
    )
    later_history = _one_row(
        history,
        history["security_id"].map(_text).eq(ACTAVIS_AGN_ID)
        & history["symbol"].map(_text).str.upper().eq("AGN"),
        "later AGN symbol_history",
    )
    _require(
        _text(legacy_master.get("primary_symbol")).upper() == "AGN"
        and _text(legacy_master.get("provider_symbol")) == "AGN_old.US"
        and _date(legacy_master.get("active_from")) == "2014-12-17"
        and _text(later_master.get("primary_symbol")).upper() == "AGN"
        and _text(later_master.get("provider_symbol")) == "AGN.US"
        and _date(later_master.get("active_from")) == "2015-01-02"
        and _date(legacy_history.get("effective_from")) == "2014-12-17"
        and _date(act_history.get("effective_from")) == ACT_EFFECTIVE_FROM
        and _date(act_history.get("effective_to")) == ACT_ACTIVE_TO
        and _date(later_history.get("effective_from")) == AGN_EFFECTIVE_FROM,
        "Allergan identity topology changed.",
    )
    return {
        "legacy_master": (legacy_master.name, legacy_master),
        "later_master": (later_master.name, later_master),
        "legacy_history": (legacy_history.name, legacy_history),
        "act_history": (act_history.name, act_history),
        "later_history": (later_history.name, later_history),
    }


def _price_profiles(
    repository: LocalDatasetRepository,
    version: str,
) -> dict[str, dict[str, Any]]:
    paths = [str(path) for path in repository.parquet_paths("daily_price_raw", version)]
    _require(bool(paths), "daily_price_raw parquet inventory is empty.")
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            "SELECT CAST(security_id AS VARCHAR) security_id, COUNT(*) row_count, "
            "CAST(MIN(session) AS VARCHAR) first_session, "
            "CAST(MAX(session) AS VARCHAR) last_session "
            "FROM read_parquet(?, union_by_name=true) "
            "WHERE security_id = ANY(?) GROUP BY security_id",
            [paths, [LEGACY_AGN_ID, ACTAVIS_AGN_ID]],
        ).fetchdf()
    finally:
        connection.close()
    output = {
        _text(row.security_id): {
            "row_count": int(row.row_count),
            "first_session": _date(row.first_session),
            "last_session": _date(row.last_session),
        }
        for row in frame.itertuples(index=False)
    }
    for security_id, expected in EXPECTED_PRICE_PROFILES.items():
        _require(
            output.get(security_id)
            == {
                "row_count": expected[0],
                "first_session": expected[1],
                "last_session": expected[2],
            },
            f"Allergan price profile changed for {security_id}.",
        )
    return output


def _verify_actions_and_resolutions(
    actions: pd.DataFrame,
    resolutions: pd.DataFrame,
) -> dict[str, Any]:
    expected = {
        LEGACY_EVENT_ID: (
            LEGACY_AGN_ID,
            "stock_merger",
            LEGACY_EVENT_DATE,
            ACTAVIS_AGN_ID,
            "ACT",
            LEGACY_EVIDENCE.source_hash,
        ),
        ACT_TICKER_EVENT_ID: (
            ACTAVIS_AGN_ID,
            "ticker_change",
            AGN_EFFECTIVE_FROM,
            ACTAVIS_AGN_ID,
            "AGN",
            ACT_TICKER_EVIDENCE.source_hash,
        ),
        LATER_EVENT_ID: (
            ACTAVIS_AGN_ID,
            "stock_merger",
            LATER_ACTIVE_TO,
            ABBV_ID,
            "ABBV",
            LATER_EVIDENCE.source_hash,
        ),
    }
    for event_id, values in expected.items():
        row = _one_row(
            actions,
            actions["event_id"].map(_text).eq(event_id),
            f"Allergan action {event_id}",
        )
        actual = (
            _text(row.get("security_id")),
            _text(row.get("action_type")).lower(),
            _date(row.get("effective_date")),
            _text(row.get("new_security_id")),
            _text(row.get("new_symbol")).upper(),
            _text(row.get("source_hash")).lower(),
        )
        _require(
            actual == values and bool(row.get("official")),
            f"Action changed: {event_id}.",
        )
    expected_resolutions = (
        (LEGACY_AGN_ID, LEGACY_ACTIVE_TO, LEGACY_EVENT_ID, ACTAVIS_AGN_ID, "ACT"),
        (ACTAVIS_AGN_ID, LATER_ACTIVE_TO, LATER_EVENT_ID, ABBV_ID, "ABBV"),
    )
    for security_id, last_price, event_id, successor_id, symbol in expected_resolutions:
        candidate_id = lifecycle_candidate_id(security_id, last_price)
        row = _one_row(
            resolutions,
            resolutions["candidate_id"].map(_text).eq(candidate_id),
            f"Allergan lifecycle resolution {candidate_id}",
        )
        _require(
            _text(row.get("security_id")) == security_id
            and _date(row.get("last_price_date")) == last_price
            and _text(row.get("resolution")) == "applied"
            and _text(row.get("event_id")) == event_id
            and _text(row.get("successor_security_id")) == successor_id
            and _text(row.get("successor_symbol")).upper() == symbol,
            f"Lifecycle resolution changed: {candidate_id}.",
        )
    return {
        "official_actions_verified": 3,
        "terminal_resolutions_verified": 2,
        "act_ticker_source_misbound": True,
    }


def _cache_key(symbol: str, period1: int, period2: int) -> str:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}"
        "&events=history&includeAdjustedClose=true&interval=1d"
    )
    return hashlib.sha256(url.encode()).hexdigest()


def _corrected_cache_state(cache_root: Path) -> dict[str, Any]:
    requests = (
        ("legacy_agn", "AGN", 1_418_774_400, 1_426_550_400),
        ("later_agn", "AGN", 1_434_326_400, 1_588_982_400),
    )
    rows = []
    for name, symbol, period1, period2 in requests:
        key = _cache_key(symbol, period1, period2)
        path = cache_root / f"{key}.json.gz"
        rows.append(
            {
                "name": name,
                "symbol": symbol,
                "request_period1": period1,
                "request_period2": period2,
                "cache_key": key,
                "present": path.is_file(),
            }
        )
    return {
        "requests": rows,
        "present": sum(bool(row["present"]) for row in rows),
        "missing": sum(not bool(row["present"]) for row in rows),
        "network_permitted": False,
    }


def _index_membership_conflicts(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
) -> list[dict[str, Any]]:
    specs = (
        (LEGACY_AGN_ID, LEGACY_ACTIVE_TO, "2015-03-23"),
        (ACTAVIS_AGN_ID, LATER_ACTIVE_TO, "2020-05-12"),
    )
    calendar = xcals.get_calendar("XNYS")
    output: list[dict[str, Any]] = []
    for security_id, proposed_active_to, removal_date in specs:
        anchor_rows = anchors.loc[
            anchors["security_id"].map(_text).eq(security_id)
            & anchors["index_id"].map(_text).str.lower().eq("sp500")
            & anchors["anchor_date"].map(_date).le(proposed_active_to)
        ]
        removal_rows = events.loc[
            events["security_id"].map(_text).eq(security_id)
            & events["index_id"].map(_text).str.lower().eq("sp500")
            & events["operation"].map(_text).str.upper().eq("REMOVE")
            & events["effective_date"].map(_date).eq(removal_date)
        ]
        _require(
            len(anchor_rows) == 1 and len(removal_rows) == 1,
            f"Allergan S&P membership evidence changed for {security_id}.",
        )
        first_day = pd.Timestamp(proposed_active_to) + pd.Timedelta(days=1)
        last_day = pd.Timestamp(removal_date) - pd.Timedelta(days=1)
        sessions = calendar.sessions_in_range(first_day, last_day)
        session_strings = [
            pd.Timestamp(value).tz_localize(None).date().isoformat()
            for value in sessions
        ]
        _require(bool(session_strings), f"Expected an index/identity gap for {security_id}.")
        output.append(
            {
                "security_id": security_id,
                "index_id": "sp500",
                "proposed_identity_active_to": proposed_active_to,
                "remove_effective_date": removal_date,
                "active_index_sessions_without_identity": session_strings,
                "gap_session_count": len(session_strings),
                "operational_validation_code": "index_member_missing_active_symbol",
            }
        )
    return output


def _repaired_state(rows: Mapping[str, tuple[Any, ...]]) -> tuple[bool, bool]:
    old = (
        _date(rows["legacy_master"][1].get("active_to")) == LEGACY_OLD_ACTIVE_TO
        and _date(rows["legacy_history"][1].get("effective_to")) == LEGACY_OLD_ACTIVE_TO
        and _date(rows["later_master"][1].get("active_to")) == LATER_OLD_ACTIVE_TO
        and _date(rows["later_history"][1].get("effective_to")) == LATER_OLD_ACTIVE_TO
    )
    repaired = (
        _date(rows["legacy_master"][1].get("active_to")) == LEGACY_ACTIVE_TO
        and _date(rows["legacy_history"][1].get("effective_to")) == LEGACY_ACTIVE_TO
        and _date(rows["later_master"][1].get("active_to")) == LATER_ACTIVE_TO
        and _date(rows["later_history"][1].get("effective_to")) == LATER_ACTIVE_TO
        and _text(rows["legacy_master"][1].get("source_hash")).lower()
        == LEGACY_EVIDENCE.source_hash
        and _text(rows["legacy_history"][1].get("source_hash")).lower()
        == LEGACY_EVIDENCE.source_hash
    )
    return old, repaired


def _build_frames(
    master: pd.DataFrame,
    history: pd.DataFrame,
    evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], bool]:
    rows = _identity_rows(master, history)
    old, repaired = _repaired_state(rows)
    _require(old or repaired, "Allergan identity boundaries are partial or unexpected.")
    if repaired:
        return (
            {"security_master": master.copy(), "symbol_history": history.copy()},
            {
                "security_master": master.iloc[:0].copy(),
                "symbol_history": history.iloc[:0].copy(),
            },
            False,
        )
    new_master = master.copy(deep=True)
    new_history = history.copy(deep=True)
    legacy_retrieved = evidence[LEGACY_EVIDENCE.name]["retrieved_at"]
    for key, frame, boundary_column, boundary in (
        ("legacy_master", new_master, "active_to", LEGACY_ACTIVE_TO),
        ("later_master", new_master, "active_to", LATER_ACTIVE_TO),
        ("legacy_history", new_history, "effective_to", LEGACY_ACTIVE_TO),
        ("later_history", new_history, "effective_to", LATER_ACTIVE_TO),
    ):
        index = rows[key][0]
        frame.at[index, boundary_column] = boundary
        if key.startswith("legacy"):
            frame.at[index, "source"] = "official_identity_boundary_repair"
            frame.at[index, "source_url"] = LEGACY_EVIDENCE.source_url
            frame.at[index, "source_hash"] = LEGACY_EVIDENCE.source_hash
            frame.at[index, "retrieved_at"] = legacy_retrieved
    delta_master = new_master.loc[
        new_master["security_id"].map(_text).isin({LEGACY_AGN_ID, ACTAVIS_AGN_ID})
    ].copy()
    delta_history = new_history.loc[
        (new_history["security_id"].map(_text).eq(LEGACY_AGN_ID))
        | (
            new_history["security_id"].map(_text).eq(ACTAVIS_AGN_ID)
            & new_history["symbol"].map(_text).str.upper().eq("AGN")
        )
    ].copy()
    _require(len(delta_master) == 2 and len(delta_history) == 2, "Repair delta is not 2+2 rows.")
    return (
        {"security_master": new_master, "symbol_history": new_history},
        {"security_master": delta_master, "symbol_history": delta_history},
        True,
    )


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    yahoo_cache: Path = DEFAULT_YAHOO_CACHE,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    _require(release is not None, "A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    _require(not missing, "Current release lacks: " + ", ".join(missing))
    pointer_etags: dict[str, str | None] = {}
    for dataset, version in release.dataset_versions.items():
        pointer, etag = repository.current_pointer(dataset)
        _require(
            pointer is not None and pointer.version == version,
            f"Release/current pointer mismatch: {dataset}.",
        )
        pointer_etags[dataset] = etag
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in (
            "security_master",
            "symbol_history",
            "corporate_actions",
            "lifecycle_resolutions",
            "source_archive",
            "index_constituent_anchors",
            "index_membership_events",
        )
    }
    evidence = {
        spec.name: _verify_evidence(repository, frames["source_archive"], spec)
        for spec in EVIDENCE_SPECS
    }
    # The current ACT evidence is exact bytes but proves the wrong corporate
    # event.  Absence of the actual June claim is a required fail-closed fact.
    act_row = frames["source_archive"].loc[
        frames["source_archive"]["source_hash"].map(_text).str.lower().eq(
            ACT_TICKER_EVIDENCE.source_hash
        )
    ].iloc[0]
    act_payload = gzip.decompress(
        _safe_object_path(repository.root, _text(act_row.get("object_path"))).read_bytes()
    )
    act_text = _plain_text(act_payload)
    _require(
        "june 15, 2015" not in act_text
        and "ticker" not in act_text
        and "trading symbol" not in act_text,
        "ACT evidence content changed; re-review ticker-change scope.",
    )
    action_proof = _verify_actions_and_resolutions(
        frames["corporate_actions"], frames["lifecycle_resolutions"]
    )
    price_profiles = _price_profiles(
        repository, release.dataset_versions["daily_price_raw"]
    )
    logical, deltas, changed = _build_frames(
        frames["security_master"], frames["symbol_history"], evidence
    )
    membership_conflicts = (
        _index_membership_conflicts(
            frames["index_constituent_anchors"],
            frames["index_membership_events"],
        )
        if changed
        else []
    )
    for dataset in WRITE_DATASETS:
        validate_dataset(
            dataset,
            logical[dataset],
            completed_session=release.completed_session,
        ).raise_for_errors()
    _identity_rows(logical["security_master"], logical["symbol_history"])
    old_hashes = {
        dataset: dataframe_sha256(frames[dataset], tuple())
        for dataset in WRITE_DATASETS
    }
    new_hashes = {
        dataset: dataframe_sha256(logical[dataset], tuple())
        for dataset in WRITE_DATASETS
    }
    cache_state = _corrected_cache_state(yahoo_cache)
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        logical_frames=logical,
        deltas=deltas,
        summary={
            "status": (
                "blocked_operational_model_conflict"
                if changed and membership_conflicts
                else "validated_offline_plan"
                if changed
                else "already_repaired"
            ),
            "base_release_version": release.version,
            "identity_rows_changed": 0,
            "candidate_identity_rows_changed": 4 if changed else 0,
            "security_master_rows_changed": 0,
            "symbol_history_rows_changed": 0,
            "price_rows_changed": 0,
            "corporate_action_rows_changed": 0,
            "adjustment_factor_rows_changed": 0,
            "lifecycle_resolution_rows_changed": 0,
            "proposed_boundaries": {
                "legacy_agn": LEGACY_ACTIVE_TO,
                "later_agn": LATER_ACTIVE_TO,
            },
            "operational_model_conflicts": membership_conflicts,
            "evidence": evidence,
            "action_and_resolution_proof": action_proof,
            "price_profiles": price_profiles,
            "dataset_hashes_before": old_hashes,
            "dataset_hashes_after": new_hashes,
            "corrected_range_yahoo_cache": cache_state,
            "blocked_targets": dict(BLOCKED_TARGETS),
            "standalone_boundary_evidence_complete": 2,
            "safe_dataset_repairs": 0,
            "dataset_repair_inventory_total": 9,
            "network_accessed": False,
            "http_attempts": 0,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


@contextmanager
def _exclusive_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved AGN recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(f"Interrupted AGN transaction blocks writes: {journal}.")
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_base_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, release_etag = repository.current_release()
    _require(
        release is not None
        and release.to_bytes() == prepared.release.to_bytes()
        and release_etag == prepared.release_etag,
        "Current release changed after AGN planning.",
    )
    for dataset, version in prepared.release.dataset_versions.items():
        pointer, etag = repository.current_pointer(dataset)
        _require(
            pointer is not None
            and pointer.version == version
            and etag == prepared.pointer_etags[dataset],
            f"{dataset} pointer changed after AGN planning.",
        )


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
            expected = {**dict(old_versions), **dict(planned_versions)}
            belongs = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or observed.dataset_versions == expected
            _require(belongs, f"unexpected release during AGN rollback: {observed.version}")
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
                observed = CurrentPointer.from_bytes(current.data)
                _require(
                    observed.version == planned_versions[dataset],
                    f"unexpected {dataset} pointer during AGN rollback: {observed.version}",
                )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _verify_committed(
    repository: LocalDatasetRepository,
    release: DataRelease,
    *,
    yahoo_cache: Path,
) -> None:
    current, _ = repository.current_release()
    _require(current is not None and current.to_bytes() == release.to_bytes(), "AGN release is not current.")
    replay = prepare_repair(repository, yahoo_cache=yahoo_cache)
    _require(replay.summary["status"] == "already_repaired", "AGN repair is not idempotent.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    yahoo_cache: Path = DEFAULT_YAHOO_CACHE,
    inject_failure: FailureInjector = _noop,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_repaired":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    if prepared.summary["status"] != "validated_offline_plan":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    with _exclusive_lock(repository):
        _assert_base_unchanged(repository, prepared)
        current_plan = prepare_repair(repository, yahoo_cache=yahoo_cache)
        if current_plan.summary["status"] == "already_repaired":
            return {**current_plan.summary, "mode": "apply", "writes_performed": False}
        old_release = repository.objects.get("releases/current.json")
        old_pointers = {
            dataset: repository.objects.get(repository.current_key(dataset)).data
            for dataset in WRITE_DATASETS
        }
        transaction_id = uuid.uuid4().hex
        planned_versions = {
            dataset: (
                "agn-boundaries-"
                f"{current_plan.release.completed_session.replace('-', '')}-"
                f"{transaction_id}-{dataset}"
            )
            for dataset in WRITE_DATASETS
        }
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_agn_identity_boundaries_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": current_plan.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": planned_versions,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            inject_failure("after_journal")
            versions = dict(current_plan.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    current_plan.deltas[dataset],
                    completed_session=current_plan.release.completed_session,
                    metadata={
                        "operation": OPERATION,
                        "input_release_version": current_plan.release.version,
                        "identity_rows_changed": 2,
                        "network_accessed": False,
                        "http_attempts": 0,
                        "eodhd_calls": 0,
                        "r2_accessed": False,
                    },
                    expected_pointer_etag=current_plan.pointer_etags[dataset],
                    version=planned_versions[dataset],
                    inherit_parent=True,
                )
                _require(not result.conflict, f"{dataset} write conflicted: {result.conflict_path}.")
                _require(
                    result.manifest.version == planned_versions[dataset]
                    and result.manifest.parent_version
                    == current_plan.release.dataset_versions[dataset]
                    and result.manifest.metadata.get("inherits_parent") is True,
                    f"{dataset} inherited delta manifest changed.",
                )
                versions[dataset] = result.manifest.version
                inject_failure(f"after_write:{dataset}")
            for dataset, version in current_plan.release.dataset_versions.items():
                if dataset in WRITE_DATASETS:
                    continue
                pointer, etag = repository.current_pointer(dataset)
                _require(
                    pointer is not None
                    and pointer.version == version
                    and etag == current_plan.pointer_etags[dataset],
                    f"Out-of-scope pointer changed during AGN apply: {dataset}.",
                )
            committed = repository.commit_release(
                current_plan.release.completed_session,
                versions,
                quality=current_plan.release.quality,
                warnings=current_plan.release.warnings,
                expected_etag=current_plan.release_etag,
            )
            inject_failure("after_release_commit")
            _verify_committed(repository, committed, yahoo_cache=yahoo_cache)
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": committed.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            return {
                **current_plan.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "new_dataset_versions": planned_versions,
                "transaction_id": transaction_id,
                "quality": committed.quality,
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned_versions,
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
                marker = repository.root / RECOVERY_DIR / f"{transaction_id}.json"
                _write_journal(marker, journal)
                raise RuntimeError(
                    f"AGN rollback incomplete; recovery marker blocks writes: {marker}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--yahoo-cache", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    yahoo_cache = args.yahoo_cache or (
        args.cache_root / "state/us_cross_validation/yahoo_chart"
    )
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository, yahoo_cache=yahoo_cache)
    result = (
        apply_repair(repository, prepared, yahoo_cache=yahoo_cache)
        if args.apply
        else {**prepared.summary, "mode": "plan", "writes_performed": False}
    )
    result["summary_sha256"] = sha256_bytes(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
