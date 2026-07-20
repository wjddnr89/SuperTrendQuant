"""Repair the local HCP -> PEAK -> DOC identity without provider calls.

EODHD keeps Healthpeak's continuous prices under the current DOC code, while
the historical PEAK endpoint contains the correct dividend stream but an
unrelated short price series.  This migration creates a distinct PEAK security
using the local DOC prices and local PEAK actions, then repoints only the PEAK
index events.  It never accesses the network.
"""

from __future__ import annotations

import hashlib
import uuid

import duckdb
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.manifest import sha256_bytes, utc_now_iso
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.validation import validate_repository_snapshot


DOC_SECURITY_ID = "US:EODHD:8d68d53d-1e0e-5bea-b83e-4c20e4c84e46"
BAD_PEAK_SECURITY_ID = "US:EODHD:e4bccc5f-0b61-541a-8c35-5957620e131f"
PEAK_SECURITY_ID = "US:EODHD:" + str(
    uuid.uuid5(uuid.NAMESPACE_URL, "eodhd:US:DOC:symbol:PEAK")
)
PEAK_FROM = "2019-11-05"
PEAK_TO = "2024-03-03"
DOC_FROM = "2024-03-04"


def main() -> None:
    repository = LocalDatasetRepository("data/cache")
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A local market-data release is required.")

    master = repository.read_frame(
        "security_master", release.dataset_versions["security_master"]
    )
    if PEAK_SECURITY_ID in set(master["security_id"].astype(str)):
        print({"status": "already_repaired", "release": release.version})
        return

    versions = dict(release.dataset_versions)
    stamp = utc_now_iso()
    provenance = b"Healthpeak HCP->PEAK->DOC local identity repair"
    source_hash = sha256_bytes(provenance)

    if "action_provider_symbol" not in master:
        master["action_provider_symbol"] = master.get("provider_symbol", "")
    else:
        master["action_provider_symbol"] = master[
            "action_provider_symbol"
        ].fillna(master.get("provider_symbol", ""))
    base_master = master.loc[
        master["security_id"].astype(str) == DOC_SECURITY_ID
    ].iloc[0].to_dict()
    base_master.update(
        {
            "security_id": PEAK_SECURITY_ID,
            "primary_symbol": "PEAK",
            "provider_symbol": "DOC.US",
            "action_provider_symbol": "PEAK.US",
            "name": "Healthpeak Properties Inc",
            "active_from": PEAK_FROM,
            "active_to": PEAK_TO,
            "source": "derived_ticker_identity",
            "source_url": "local://us-bootstrap/ticker-identity/HCP-PEAK-DOC",
            "retrieved_at": stamp,
            "source_hash": source_hash,
        }
    )
    master = pd.concat([master, pd.DataFrame([base_master])], ignore_index=True)
    master_result = repository.write_frame(
        "security_master",
        master,
        completed_session=release.completed_session,
        metadata={"operation": "repair_healthpeak_identity"},
    )
    versions["security_master"] = master_result.manifest.version

    history = repository.read_frame(
        "symbol_history", release.dataset_versions["symbol_history"]
    )
    doc_mask = (
        (history["security_id"].astype(str) == DOC_SECURITY_ID)
        & (history["symbol"].astype(str) == "DOC")
    )
    bad_peak_mask = history["security_id"].astype(str) == BAD_PEAK_SECURITY_ID
    history = history.loc[
        ~(
            (history["symbol"].astype(str) == "PEAK")
            & ~bad_peak_mask
        )
    ].copy()
    doc_mask = (
        (history["security_id"].astype(str) == DOC_SECURITY_ID)
        & (history["symbol"].astype(str) == "DOC")
    )
    history.loc[doc_mask, "effective_from"] = "2015-01-01"
    history.loc[bad_peak_mask, "symbol"] = "PEAK"
    history.loc[bad_peak_mask, "effective_from"] = "2015-01-01"
    history.loc[bad_peak_mask, "effective_to"] = "2019-09-16"
    base_history = history.loc[doc_mask].iloc[0].to_dict()
    base_history.update(
        {
            "security_id": PEAK_SECURITY_ID,
            "symbol": "PEAK",
            "effective_from": PEAK_FROM,
            "effective_to": PEAK_TO,
            "source": "derived_ticker_identity",
            "source_url": "local://us-bootstrap/ticker-identity/HCP-PEAK-DOC",
            "retrieved_at": stamp,
            "source_hash": source_hash,
        }
    )
    history = pd.concat([history, pd.DataFrame([base_history])], ignore_index=True)
    history_result = repository.write_frame(
        "symbol_history",
        history,
        completed_session=release.completed_session,
        metadata={"operation": "repair_healthpeak_identity"},
    )
    versions["symbol_history"] = history_result.manifest.version

    price_paths = [
        str(path)
        for path in repository.parquet_paths(
            "daily_price_raw", release.dataset_versions["daily_price_raw"]
        )
    ]
    connection = duckdb.connect()
    prices = connection.execute(
        """
        SELECT * FROM read_parquet(?)
        WHERE security_id = ? AND session BETWEEN ? AND ?
        ORDER BY session
        """,
        [price_paths, DOC_SECURITY_ID, PEAK_FROM, PEAK_TO],
    ).fetchdf()
    if prices.empty:
        raise RuntimeError("Local DOC prices are unavailable for the PEAK interval.")
    prices["security_id"] = PEAK_SECURITY_ID
    prices["source"] = "derived_ticker_identity"
    prices["retrieved_at"] = stamp
    prices["source_hash"] = source_hash
    completed_row = connection.execute(
        "SELECT * FROM read_parquet(?) WHERE session = ? LIMIT 1",
        [price_paths, release.completed_session],
    ).fetchdf()
    price_delta = pd.concat([prices, completed_row], ignore_index=True)
    price_result = repository.write_frame(
        "daily_price_raw",
        price_delta,
        completed_session=release.completed_session,
        metadata={"operation": "repair_healthpeak_identity"},
        inherit_parent=True,
    )
    versions["daily_price_raw"] = price_result.manifest.version

    all_actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    action_dates = pd.to_datetime(all_actions["effective_date"], errors="coerce")
    actions = all_actions.loc[
        (all_actions["security_id"].astype(str) == BAD_PEAK_SECURITY_ID)
        & (action_dates >= pd.Timestamp(PEAK_FROM))
        & (action_dates <= pd.Timestamp(PEAK_TO))
    ].copy()
    actions["security_id"] = PEAK_SECURITY_ID
    actions["event_id"] = [
        hashlib.sha256(
            f"{row.source}|{PEAK_SECURITY_ID}|{row.action_type}|{row.effective_date}".encode()
        ).hexdigest()
        for row in actions.itertuples(index=False)
    ]
    actions["source"] = "derived_ticker_identity"
    actions["retrieved_at"] = stamp
    actions["source_hash"] = source_hash
    action_result = repository.write_frame(
        "corporate_actions",
        actions,
        completed_session=release.completed_session,
        incomplete_action_policy="warn",
        metadata={"operation": "repair_healthpeak_identity"},
        inherit_parent=True,
    )
    versions["corporate_actions"] = action_result.manifest.version

    factors = build_adjustment_factors(
        prices,
        actions,
        source_version="derived:HCP-PEAK-DOC",
    )
    factor_result = repository.write_frame(
        "adjustment_factors",
        factors,
        completed_session=release.completed_session,
        metadata={"operation": "repair_healthpeak_identity"},
        inherit_parent=True,
    )
    versions["adjustment_factors"] = factor_result.manifest.version

    events = repository.read_frame(
        "index_membership_events",
        release.dataset_versions["index_membership_events"],
    )
    repaired_events = int(
        (events["security_id"].astype(str) == BAD_PEAK_SECURITY_ID).sum()
    )
    events.loc[
        events["security_id"].astype(str) == BAD_PEAK_SECURITY_ID,
        "security_id",
    ] = PEAK_SECURITY_ID
    event_result = repository.write_frame(
        "index_membership_events",
        events,
        completed_session=release.completed_session,
        metadata={"operation": "repair_healthpeak_identity"},
    )
    versions["index_membership_events"] = event_result.manifest.version

    report = validate_repository_snapshot(repository)
    report.raise_for_errors()
    warnings = tuple(
        dict.fromkeys(
            (*release.warnings, *(issue.message for issue in report.issues))
        )
    )
    repaired_release = repository.commit_release(
        release.completed_session,
        versions,
        quality=DataQuality.DEGRADED if warnings else DataQuality.VALID,
        warnings=warnings,
        expected_etag=release_etag,
    )
    print(
        {
            "status": "repaired",
            "release": repaired_release.version,
            "peak_security_id": PEAK_SECURITY_ID,
            "price_rows": len(prices),
            "action_rows": len(actions),
            "factor_rows": len(factors),
            "repointed_events": repaired_events,
            "quality": repaired_release.quality,
            "warnings": repaired_release.warnings,
        }
    )


if __name__ == "__main__":
    main()
