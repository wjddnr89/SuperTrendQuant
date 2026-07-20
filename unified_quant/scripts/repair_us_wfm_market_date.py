#!/usr/bin/env python3
"""Repair the WFM terminal market date against immutable SEC evidence.

The archived completion 8-K contains both the 2017-08-23 shareholder vote and
the actual 2017-08-28 merger close.  WFM continued trading through 2017-08-25
and NASDAQ suspended it before the 2017-08-28 open.  This offline repair moves
the cash-merger action to that first post-terminal XNYS session, rekeys the
linked resolution, closes symbol history on the last real session, and rebuilds
the complete factor inventory against the planned corporate-actions version.

Plan mode is the default.  There is no network, EODHD, or R2 code path.  Apply
uses one writer lock, release and pointer CAS, a durable transaction journal,
verified rollback, and post-commit idempotency replay.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.lifecycle_coverage import lifecycle_candidate_id
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
WRITE_DATASETS = (
    "corporate_actions",
    "lifecycle_resolutions",
    "symbol_history",
    "adjustment_factors",
)
REQUIRED_DATASETS = (
    *WRITE_DATASETS,
    "security_master",
    "daily_price_raw",
    "source_archive",
    "index_membership_events",
)
TRANSACTION_DIR = "transactions/us-wfm-market-date"
RECOVERY_DIR = "recovery/us-wfm-market-date"
OPERATION = "repair_us_wfm_market_date"
REPAIR_REVIEWED_AT = "2026-07-18T13:00:00Z"

WFM_SECURITY_ID = "US:EODHD:c24f6a80-3a51-56f7-9c55-68916e553fad"
WFM_SYMBOL = "WFM"
WFM_LAST_SESSION = "2017-08-25"
CORRECTED_MARKET_DATE = "2017-08-28"
WRONG_PARSED_DATE = "2017-08-23"
OLD_EVENT_ID = "8bfbf4dfaeb0c692221b77aae1d0b5437db87b9766aff0d89444917c87b6a2cf"
NEW_EVENT_ID = "25bce725b19ce21cebac0fa09351a30e5b89479256f7d1ed25f9218b557754c4"
WFM_CANDIDATE_ID = "98d3b9997c6c9cab63831a97574cb9126c7db0cfeff661b447755caa60d3d97c"
CASH_PER_SHARE = 42.0

OFFICIAL_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/865436/"
    "000114420417045261/0001144204-17-045261.txt"
)
OFFICIAL_SOURCE_HASH = (
    "6e0bf4ec75189e50acd6594d2fd2e676a6ea4b4fe5a876dc38794133074bd152"
)
OFFICIAL_SOURCE_BYTES = 239_310
OFFICIAL_RETRIEVED_AT = "2026-07-18T10:30:27.716652Z"
ACTION_SOURCE = "sec_edgar+stored_price_crosscheck"
ACTION_SOURCE_KIND = "official_crosscheck"
ARCHIVE_DATASET = "sec_edgar_filing"
ARCHIVE_CONTENT_TYPE = "text/plain"

OLD_HISTORY_SOURCE = "eodhd_exchange_symbols"
OLD_HISTORY_URL = "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
OLD_HISTORY_RETRIEVED_AT = "2026-07-16T15:56:01.033938Z"
OLD_HISTORY_HASH = (
    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
)
REPAIRED_HISTORY_SOURCE = "official_wfm_market_date_repair"

RESOLUTION_REVIEWED_BY = "us_lifecycle_finalizer_v1"
RESOLUTION_REVIEWED_AT = "2026-07-18T00:00:00Z"
RESOLUTION_SOURCE = "lifecycle_finalizer"


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    planned_versions: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


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
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _number(value: Any) -> float | None:
    if not _text(value):
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _bool_true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _static_contract() -> None:
    if canonical_lifecycle_event_id(
        WFM_SECURITY_ID, "cash_merger", WRONG_PARSED_DATE
    ) != OLD_EVENT_ID:
        raise RuntimeError("Pinned WFM old event_id is not canonical.")
    if canonical_lifecycle_event_id(
        WFM_SECURITY_ID, "cash_merger", CORRECTED_MARKET_DATE
    ) != NEW_EVENT_ID:
        raise RuntimeError("Pinned WFM corrected event_id is not canonical.")
    if lifecycle_candidate_id(WFM_SECURITY_ID, WFM_LAST_SESSION) != WFM_CANDIDATE_ID:
        raise RuntimeError("Pinned WFM lifecycle candidate_id is not canonical.")


def _archive_path(completed_session: str) -> str:
    return f"archives/{completed_session}/{OFFICIAL_SOURCE_HASH}.txt.gz"


def _safe_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    target = (base / object_path).resolve()
    if target == base or base not in target.parents:
        raise ValueError(f"WFM evidence path escapes repository: {object_path}.")
    return target


def _normalized_official_text(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    without_tags = re.sub(r"<[^>]+>", " ", decoded)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _verify_official_text(payload: bytes) -> None:
    text = _normalized_official_text(payload)
    required = (
        r"On August 28, 2017.{0,180}?completed its previously announced acquisition",
        r"merged with and into Whole Foods Market on August 28, 2017",
        r"converted into the right to receive \$42\.00 in cash",
        r"suspend trading of Whole Foods Market Shares prior to market open on August 28, 2017",
        r"On August 23, 2017.{0,180}?special meeting of shareholders",
    )
    missing = [pattern for pattern in required if not re.search(pattern, text, re.I)]
    if missing:
        raise ValueError(
            "Pinned WFM filing no longer proves completion/trading/cash semantics: "
            + ", ".join(missing)
        )


def _verify_official_archive(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
) -> bytes:
    expected_path = _archive_path(completed_session)
    related = (
        archive["archive_id"].astype(str).eq(OFFICIAL_SOURCE_HASH)
        | archive["source_hash"].astype(str).eq(OFFICIAL_SOURCE_HASH)
        | archive["object_path"].astype(str).eq(expected_path)
        | archive.get(
            "source_url", pd.Series(index=archive.index, dtype="object")
        ).astype(str).eq(OFFICIAL_SOURCE_URL)
    )
    rows = archive.loc[related]
    if len(rows) != 1:
        raise ValueError(
            "Current release must contain exactly one WFM official evidence row; "
            f"found {len(rows)}."
        )
    row = rows.iloc[0]
    expected = {
        "archive_id": OFFICIAL_SOURCE_HASH,
        "dataset": ARCHIVE_DATASET,
        "object_path": expected_path,
        "content_type": ARCHIVE_CONTENT_TYPE,
        "effective_date": completed_session,
        "source": ARCHIVE_DATASET,
        "retrieved_at": OFFICIAL_RETRIEVED_AT,
        "source_hash": OFFICIAL_SOURCE_HASH,
        "source_url": OFFICIAL_SOURCE_URL,
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if (
            _date(row.get(key)) != expected_value
            if key == "effective_date"
            else _text(row.get(key)) != expected_value
        )
    ]
    if mismatches:
        raise ValueError(
            "WFM official source_archive row changed: " + ", ".join(mismatches) + "."
        )
    path = _safe_path(repository.root, expected_path)
    if not path.is_file():
        raise FileNotFoundError(f"WFM official evidence payload is missing: {path}.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError("WFM official evidence payload is invalid gzip.") from exc
    observed = hashlib.sha256(payload).hexdigest()
    if observed != OFFICIAL_SOURCE_HASH or len(payload) != OFFICIAL_SOURCE_BYTES:
        raise ValueError(
            "WFM official evidence payload hash/size changed: "
            f"hash={observed}; bytes={len(payload)}."
        )
    _verify_official_text(payload)
    return payload


def _base_action_terms(row: Mapping[str, Any]) -> bool:
    return bool(
        _text(row.get("security_id")) == WFM_SECURITY_ID
        and _text(row.get("action_type")) == "cash_merger"
        and _date(row.get("record_date")) == ""
        and _number(row.get("cash_amount")) == CASH_PER_SHARE
        and _number(row.get("ratio")) is None
        and _text(row.get("currency")) == "USD"
        and _text(row.get("new_security_id")) == ""
        and _text(row.get("new_symbol")) == ""
        and _bool_true(row.get("official"))
        and _text(row.get("source_url")) == OFFICIAL_SOURCE_URL
        and _text(row.get("source_kind")) == ACTION_SOURCE_KIND
        and _text(row.get("source")) == ACTION_SOURCE
        and _text(row.get("retrieved_at")) == OFFICIAL_RETRIEVED_AT
        and _text(row.get("source_hash")) == OFFICIAL_SOURCE_HASH
        and _text(row.get("metadata")) == ""
    )


def _old_action(row: Mapping[str, Any]) -> bool:
    return bool(
        _base_action_terms(row)
        and _text(row.get("event_id")) == OLD_EVENT_ID
        and _date(row.get("effective_date")) == WRONG_PARSED_DATE
        and _date(row.get("ex_date")) == WRONG_PARSED_DATE
        and _date(row.get("announcement_date")) == CORRECTED_MARKET_DATE
        and _date(row.get("payment_date")) == ""
    )


def _repaired_action(row: Mapping[str, Any]) -> bool:
    return bool(
        _base_action_terms(row)
        and _text(row.get("event_id")) == NEW_EVENT_ID
        and _date(row.get("effective_date")) == CORRECTED_MARKET_DATE
        and _date(row.get("ex_date")) == CORRECTED_MARKET_DATE
        # The completion filing proves the market date, but it does not prove
        # a separate announcement or payment date.  Preserve those fields.
        and _date(row.get("announcement_date")) == CORRECTED_MARKET_DATE
        and _date(row.get("payment_date")) == ""
    )


def _rewrite_action(actions: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    related = actions["event_id"].astype(str).isin({OLD_EVENT_ID, NEW_EVENT_ID})
    if int(related.sum()) != 1:
        raise ValueError(
            "Current release must contain exactly one old/repaired WFM merger action; "
            f"found {int(related.sum())}."
        )
    index = actions.index[related][0]
    row = actions.loc[index]
    old = _old_action(row)
    repaired = _repaired_action(row)
    if old == repaired:
        raise ValueError("WFM merger action is neither exact old nor exact repaired state.")
    output = actions.copy(deep=True)
    if old:
        updates = {
            "event_id": NEW_EVENT_ID,
            "effective_date": CORRECTED_MARKET_DATE,
            "ex_date": CORRECTED_MARKET_DATE,
        }
        for column, value in updates.items():
            output.at[index, column] = value
    changed_rows = output.ne(actions) & ~(output.isna() & actions.isna())
    permitted = {
        (index, column)
        for column in (
            "event_id",
            "effective_date",
            "ex_date",
        )
    }
    observed = {
        (row_index, column)
        for row_index, row_values in changed_rows.iterrows()
        for column, changed in row_values.items()
        if bool(changed)
    }
    if old and observed != permitted:
        raise AssertionError(f"WFM action repair changed unexpected cells: {observed}.")
    if repaired and observed:
        raise AssertionError("Repaired WFM action changed during replay.")
    return output, old


def _resolution_terms(row: Mapping[str, Any], event_id: str) -> bool:
    return bool(
        _text(row.get("candidate_id")) == WFM_CANDIDATE_ID
        and _text(row.get("security_id")) == WFM_SECURITY_ID
        and _text(row.get("symbol")) == WFM_SYMBOL
        and _date(row.get("last_price_date")) == WFM_LAST_SESSION
        and _text(row.get("resolution")) == "applied"
        and _text(row.get("event_id")) == event_id
        and _text(row.get("exception_code")) == ""
        and _text(row.get("exception_reason")) == ""
        and _text(row.get("reviewed_by")) == RESOLUTION_REVIEWED_BY
        and _text(row.get("reviewed_at")) == RESOLUTION_REVIEWED_AT
        and _text(row.get("recheck_after")) == ""
        and _text(row.get("successor_security_id")) == ""
        and _text(row.get("successor_symbol")) == ""
        and _text(row.get("source_url")) == OFFICIAL_SOURCE_URL
        and _text(row.get("source")) == RESOLUTION_SOURCE
        and _text(row.get("retrieved_at")) == OFFICIAL_RETRIEVED_AT
        and _text(row.get("source_hash")) == OFFICIAL_SOURCE_HASH
    )


def _rewrite_resolution(resolutions: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    related = resolutions["candidate_id"].astype(str).eq(WFM_CANDIDATE_ID)
    if int(related.sum()) != 1:
        raise ValueError(
            "Current release must contain exactly one WFM lifecycle resolution; "
            f"found {int(related.sum())}."
        )
    index = resolutions.index[related][0]
    row = resolutions.loc[index]
    old = _resolution_terms(row, OLD_EVENT_ID)
    repaired = _resolution_terms(row, NEW_EVENT_ID)
    if old == repaired:
        raise ValueError("WFM resolution is neither exact old nor exact repaired state.")
    output = resolutions.copy(deep=True)
    if old:
        output.at[index, "event_id"] = NEW_EVENT_ID
    for column in resolutions.columns:
        if column == "event_id":
            continue
        if not resolutions[column].equals(output[column]):
            raise AssertionError(f"WFM resolution repair changed {column}.")
    changed = resolutions["event_id"].astype(str).ne(output["event_id"].astype(str))
    if int(changed.sum()) != int(old):
        raise AssertionError("WFM resolution rekey changed an unexpected row.")
    return output, old


def _history_base(row: Mapping[str, Any]) -> bool:
    return bool(
        _text(row.get("security_id")) == WFM_SECURITY_ID
        and _text(row.get("symbol")) == WFM_SYMBOL
        and _text(row.get("exchange")) == "NASDAQ"
        and _date(row.get("effective_from")) == "2015-01-01"
    )


def _old_history(row: Mapping[str, Any]) -> bool:
    return bool(
        _history_base(row)
        and _date(row.get("effective_to")) == ""
        and _text(row.get("source")) == OLD_HISTORY_SOURCE
        and _text(row.get("source_url")) == OLD_HISTORY_URL
        and _text(row.get("retrieved_at")) == OLD_HISTORY_RETRIEVED_AT
        and _text(row.get("source_hash")) == OLD_HISTORY_HASH
    )


def _repaired_history(row: Mapping[str, Any]) -> bool:
    return bool(
        _history_base(row)
        and _date(row.get("effective_to")) == WFM_LAST_SESSION
        and _text(row.get("source")) == REPAIRED_HISTORY_SOURCE
        and _text(row.get("source_url")) == OFFICIAL_SOURCE_URL
        and _text(row.get("retrieved_at")) == OFFICIAL_RETRIEVED_AT
        and _text(row.get("source_hash")) == OFFICIAL_SOURCE_HASH
    )


def _rewrite_history(history: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    related = history["security_id"].astype(str).eq(WFM_SECURITY_ID)
    if int(related.sum()) != 1:
        raise ValueError(
            "Current release must contain exactly one WFM symbol-history row; "
            f"found {int(related.sum())}."
        )
    index = history.index[related][0]
    row = history.loc[index]
    old = _old_history(row)
    repaired = _repaired_history(row)
    if old == repaired:
        raise ValueError("WFM symbol history is neither exact old nor repaired state.")
    output = history.copy(deep=True)
    if old:
        updates = {
            "effective_to": WFM_LAST_SESSION,
            "source": REPAIRED_HISTORY_SOURCE,
            "source_url": OFFICIAL_SOURCE_URL,
            "retrieved_at": OFFICIAL_RETRIEVED_AT,
            "source_hash": OFFICIAL_SOURCE_HASH,
        }
        for column, value in updates.items():
            output.at[index, column] = value
    for column in history.columns:
        if column in {"effective_to", "source", "source_url", "retrieved_at", "source_hash"}:
            continue
        if not history[column].equals(output[column]):
            raise AssertionError(f"WFM history repair changed {column}.")
    return output, old


def _verify_boundaries(
    master: pd.DataFrame,
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> None:
    master_rows = master["security_id"].astype(str).eq(WFM_SECURITY_ID)
    if int(master_rows.sum()) != 1:
        raise ValueError("WFM security_master identity inventory is not exact.")
    row = master.loc[master_rows].iloc[0]
    if not (
        _text(row.get("primary_symbol")) == WFM_SYMBOL
        and _text(row.get("exchange")) == "NASDAQ"
        and _date(row.get("active_to")) == WFM_LAST_SESSION
    ):
        raise ValueError("WFM security_master terminal boundary changed.")
    price_rows = prices["security_id"].astype(str).eq(WFM_SECURITY_ID)
    sessions = pd.to_datetime(prices.loc[price_rows, "session"], errors="coerce")
    if sessions.isna().any() or len(sessions) == 0:
        raise ValueError("WFM price-session inventory is invalid.")
    if sessions.max().date().isoformat() != WFM_LAST_SESSION:
        raise ValueError("WFM last price session is not 2017-08-25.")
    removal = membership.loc[
        membership["security_id"].astype(str).eq(WFM_SECURITY_ID)
        & membership["index_id"].astype(str).str.lower().eq("sp500")
        & membership["operation"].astype(str).str.upper().eq("REMOVE")
        & pd.to_datetime(
            membership["effective_date"], errors="coerce"
        ).dt.date.astype(str).eq(CORRECTED_MARKET_DATE)
    ]
    if len(removal) != 1:
        raise ValueError("WFM S&P 500 removal on 2017-08-28 is not exact.")


def _adjustment_source_version(price_version: str, action_version: str) -> str:
    if not price_version or not action_version:
        raise RuntimeError("WFM factor lineage requires exact input versions.")
    return f"{price_version}+{action_version}"


def _new_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"wfm-market-date-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


def _expected_factors(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
    columns: list[str],
) -> pd.DataFrame:
    output = build_adjustment_factors(
        prices, actions, source_version=source_version
    ).reindex(columns=columns)
    output["source_version"] = source_version
    output["calculated_at"] = REPAIR_REVIEWED_AT
    output["source"] = "derived"
    output["retrieved_at"] = REPAIR_REVIEWED_AT
    output["source_hash"] = source_version
    return output


def _assert_factor_economics(
    current: pd.DataFrame, expected: pd.DataFrame
) -> int:
    keys = ["security_id", "session"]
    values = ["split_factor", "total_return_factor"]
    left = current[keys + values].copy()
    right = expected[keys + values].copy()
    for frame in (left, right):
        frame["security_id"] = frame["security_id"].astype(str)
        frame["session"] = pd.to_datetime(frame["session"], errors="raise").dt.normalize()
        frame.sort_values(keys, inplace=True, ignore_index=True)
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise ValueError("WFM repair would change adjustment-factor key inventory.")
    changed = np.zeros(len(left), dtype=bool)
    for column in values:
        old = pd.to_numeric(left[column], errors="raise").to_numpy(dtype=float)
        new = pd.to_numeric(right[column], errors="raise").to_numpy(dtype=float)
        changed |= ~((old == new) | (np.isnan(old) & np.isnan(new)))
    count = int(changed.sum())
    if count:
        sample = left.loc[changed, keys].head(10).to_dict("records")
        raise ValueError(
            "WFM cash-merger date correction unexpectedly changes factor economics: "
            + json.dumps(sample, default=str, sort_keys=True)
        )
    return count


def _rebind_factors(
    current: pd.DataFrame,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
) -> tuple[pd.DataFrame, bool, int]:
    expected = _expected_factors(
        prices,
        actions,
        source_version=source_version,
        columns=list(current.columns),
    )
    changed = _assert_factor_economics(current, expected)
    exact_lineage = (
        set(current["source_version"].astype(str)) == {source_version}
        and set(current["source_hash"].astype(str)) == {source_version}
        and set(current["source"].astype(str)) == {"derived"}
    )
    return (
        current.reset_index(drop=True) if exact_lineage else expected.reset_index(drop=True),
        not exact_lineage,
        changed,
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
        return self.base.read_frame(dataset, version) if version else pd.DataFrame()


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


def prepare_repair(repository: LocalDatasetRepository) -> PreparedRepair:
    _static_contract()
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks WFM repair datasets: " + ", ".join(missing))
    pointer_etags = _pointer_etags(repository, release)
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    _verify_official_archive(
        repository,
        frames["source_archive"],
        completed_session=release.completed_session,
    )
    _verify_boundaries(
        frames["security_master"],
        frames["daily_price_raw"],
        frames["index_membership_events"],
    )
    actions, action_changed = _rewrite_action(frames["corporate_actions"])
    resolutions, resolution_changed = _rewrite_resolution(
        frames["lifecycle_resolutions"]
    )
    history, history_changed = _rewrite_history(frames["symbol_history"])
    state = (action_changed, resolution_changed, history_changed)
    if any(state) and not all(state):
        raise RuntimeError("WFM market-date repair is partially applied.")

    planned_versions: dict[str, str] = {}
    if action_changed:
        planned_versions = _new_versions(release)
        factor_lineage = _adjustment_source_version(
            release.dataset_versions["daily_price_raw"],
            planned_versions["corporate_actions"],
        )
    else:
        factor_lineage = _adjustment_source_version(
            release.dataset_versions["daily_price_raw"],
            release.dataset_versions["corporate_actions"],
        )
    factors, factor_rebound, economic_changed = _rebind_factors(
        frames["adjustment_factors"],
        frames["daily_price_raw"],
        actions,
        source_version=factor_lineage,
    )
    if action_changed and not factor_rebound:
        raise RuntimeError("Changed WFM action did not rebind every factor row.")
    if not action_changed and factor_rebound:
        raise RuntimeError("Repaired WFM action has stale factor provenance.")
    if economic_changed != 0:
        raise AssertionError("WFM expected factor economic change count is not zero.")

    overrides = {
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "symbol_history": history,
        "adjustment_factors": factors,
    }
    for dataset, frame in overrides.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=release.completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    validate_repository_snapshot(
        _CandidateRepository(repository, release.dataset_versions, overrides)
    ).raise_for_errors()
    changed = action_changed
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frames=overrides if changed else {},
        summary={
            "status": "validated_offline_plan" if changed else "already_repaired",
            "base_release_version": release.version,
            "security_id": WFM_SECURITY_ID,
            "symbol": WFM_SYMBOL,
            "old_event_id": OLD_EVENT_ID,
            "new_event_id": NEW_EVENT_ID,
            "old_parsed_date": WRONG_PARSED_DATE,
            "last_price_date": WFM_LAST_SESSION,
            "corrected_market_date": CORRECTED_MARKET_DATE,
            "announcement_date": CORRECTED_MARKET_DATE,
            "announcement_date_policy": "preserved_existing_value",
            "payment_date": "",
            "payment_date_policy": "blank_not_inferred_from_completion_filing",
            "cash_amount": CASH_PER_SHARE,
            "corporate_action_rows_rekeyed": int(action_changed),
            "lifecycle_resolution_rows_relinked": int(resolution_changed),
            "symbol_history_rows_closed": int(history_changed),
            "security_master_rows_verified": 1,
            "adjustment_factor_rows": len(factors),
            "adjustment_factor_economic_rows_changed": 0,
            "adjustment_factor_provenance_rows_rebound": (
                len(factors) if changed else 0
            ),
            "factor_source_version": factor_lineage,
            "official_source_url": OFFICIAL_SOURCE_URL,
            "official_source_hash": OFFICIAL_SOURCE_HASH,
            "planned_versions": dict(planned_versions),
            "write_datasets": list(WRITE_DATASETS),
            "other_dataset_versions_unchanged": True,
            "network_accessed": False,
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
            raise RuntimeError("Unresolved WFM recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted WFM transaction blocks writes: {journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_inputs_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, release_etag = repository.current_release()
    if (
        release is None
        or release.version != prepared.release.version
        or release_etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after WFM planning.")
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or pointer.version != prepared.release.dataset_versions[dataset]
            or etag != prepared.pointer_etags[dataset]
        ):
            raise RuntimeError(f"{dataset} pointer changed after WFM planning.")


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
            "operation": OPERATION,
            "wfm_old_event_id": OLD_EVENT_ID,
            "wfm_new_event_id": NEW_EVENT_ID,
            "official_source_url": OFFICIAL_SOURCE_URL,
            "official_source_hash": OFFICIAL_SOURCE_HASH,
            "input_release_version": prepared.release.version,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    input_versions = dict(prepared.release.dataset_versions)
    input_versions.update(prepared.planned_versions)
    metadata["input_versions"] = input_versions
    if dataset == "adjustment_factors":
        price_version = prepared.release.dataset_versions["daily_price_raw"]
        action_version = prepared.planned_versions["corporate_actions"]
        lineage = _adjustment_source_version(price_version, action_version)
        factors = prepared.frames["adjustment_factors"]
        if (
            set(factors["source_version"].astype(str)) != {lineage}
            or set(factors["source_hash"].astype(str)) != {lineage}
            or set(factors["source"].astype(str)) != {"derived"}
        ):
            raise RuntimeError("Prepared WFM factors have stale planned lineage.")
        metadata.update(
            {
                "source_version": lineage,
                "source_daily_price_version": price_version,
                "source_corporate_actions_version": action_version,
                "expected_economic_rows_changed": 0,
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
                    f"unexpected release during WFM rollback: {observed.version}"
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
                        f"unexpected {dataset} pointer during WFM rollback: "
                        f"{pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository, release: DataRelease
) -> None:
    current, _ = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes():
        raise RuntimeError("Committed WFM release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, _ = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"Applied WFM release pointer mismatch: {dataset}.")
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    resolutions = repository.read_frame(
        "lifecycle_resolutions", release.dataset_versions["lifecycle_resolutions"]
    )
    history = repository.read_frame(
        "symbol_history", release.dataset_versions["symbol_history"]
    )
    action = actions.loc[actions["event_id"].astype(str).eq(NEW_EVENT_ID)]
    resolution = resolutions.loc[
        resolutions["candidate_id"].astype(str).eq(WFM_CANDIDATE_ID)
    ]
    history_row = history.loc[
        history["security_id"].astype(str).eq(WFM_SECURITY_ID)
    ]
    if (
        len(action) != 1
        or not _repaired_action(action.iloc[0])
        or len(resolution) != 1
        or not _resolution_terms(resolution.iloc[0], NEW_EVENT_ID)
        or len(history_row) != 1
        or not _repaired_history(history_row.iloc[0])
    ):
        raise RuntimeError("Applied WFM repaired rows are not exact.")
    price_version = release.dataset_versions["daily_price_raw"]
    action_version = release.dataset_versions["corporate_actions"]
    factor_version = release.dataset_versions["adjustment_factors"]
    lineage = _adjustment_source_version(price_version, action_version)
    manifest = repository.manifest_for_version("adjustment_factors", factor_version)
    if any(
        _text(manifest.metadata.get(key)) != value
        for key, value in {
            "source_version": lineage,
            "source_daily_price_version": price_version,
            "source_corporate_actions_version": action_version,
        }.items()
    ):
        raise RuntimeError("WFM factor manifest lineage is not release-exact.")
    factors = repository.read_frame("adjustment_factors", factor_version)
    if (
        set(factors["source_version"].astype(str)) != {lineage}
        or set(factors["source_hash"].astype(str)) != {lineage}
        or set(factors["source"].astype(str)) != {"derived"}
    ):
        raise RuntimeError("WFM factor rows are not release-exact.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_repaired":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        archive = repository.read_frame(
            "source_archive",
            prepared.release.dataset_versions["source_archive"],
        )
        _verify_official_archive(
            repository,
            archive,
            completed_session=prepared.release.completed_session,
        )
        planned = dict(prepared.planned_versions)
        if set(planned) != set(WRITE_DATASETS) or len(set(planned.values())) != len(
            WRITE_DATASETS
        ):
            raise RuntimeError("Prepared WFM repair has invalid planned versions.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before WFM apply.")
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_wfm_market_date_transaction/v1",
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
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(repository, prepared, dataset),
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                if result.manifest.version != planned[dataset]:
                    raise RuntimeError(f"Unexpected {dataset} version was written.")
                versions[dataset] = result.manifest.version
                inject(f"after_write:{dataset}")
            if any(
                versions.get(dataset) != version
                for dataset, version in prepared.release.dataset_versions.items()
                if dataset not in WRITE_DATASETS
            ):
                raise RuntimeError("WFM repair changed an out-of-scope dataset version.")
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=prepared.release.quality,
                warnings=prepared.release.warnings,
                expected_etag=prepared.release_etag,
            )
            inject("after_release_commit")
            validate_repository_snapshot(repository).raise_for_errors()
            _assert_applied_release(repository, committed)
            replay = prepare_repair(repository)
            if replay.summary["status"] != "already_repaired":
                raise RuntimeError("WFM market-date repair is not idempotent.")
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
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "transaction_id": transaction_id,
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=committed.version if committed else "",
                old_versions=prepared.release.dataset_versions,
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
                    "WFM rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair WFM's 2017 terminal market date offline."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository)
    result = (
        apply_repair(repository, prepared)
        if args.apply
        else {**prepared.summary, "mode": "plan", "writes_performed": False}
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
