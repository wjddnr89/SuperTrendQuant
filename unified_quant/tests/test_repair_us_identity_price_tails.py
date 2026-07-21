from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pandas as pd
import pytest
import yaml

from supertrend_quant.market_store.adjustments import RATIO_ACTIONS
from supertrend_quant.market_store.manifest import CurrentPointer, DataRelease, sha256_bytes
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.storage import LocalObjectStore


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts/repair_us_identity_price_tails.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_identity_price_tails_for_test", SCRIPT_PATH
)
assert SPEC and SPEC.loader
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

FINALIZER_PATH = (
    Path(__file__).parents[1] / "scripts/finalize_us_lifecycle_coverage.py"
)
FINALIZER_SPEC = importlib.util.spec_from_file_location(
    "finalize_us_lifecycle_coverage_for_simple7_test", FINALIZER_PATH
)
assert FINALIZER_SPEC and FINALIZER_SPEC.loader
finalizer = importlib.util.module_from_spec(FINALIZER_SPEC)
sys.modules[FINALIZER_SPEC.name] = finalizer
FINALIZER_SPEC.loader.exec_module(finalizer)


@pytest.fixture(scope="module")
def pinned_frames() -> dict[str, pd.DataFrame]:
    repository = LocalDatasetRepository(Path("data/cache"))
    release, _ = repository.current_release()
    assert release is not None
    assert release.version == script.PINNED_RELEASE_VERSION
    security_ids = {
        security_id
        for case in script.CASES
        for security_id in (case.security_id, case.successor_security_id)
    }
    frames = {
        dataset: script._read_security_subset(
            repository,
            dataset,
            release.dataset_versions[dataset],
            security_ids,
        )
        for dataset in (
            "daily_price_raw",
            "adjustment_factors",
            "security_master",
            "symbol_history",
        )
    }
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    event_ids = {case.event_id for case in script.CASES}
    frames["corporate_actions"] = actions.loc[
        actions["event_id"].astype(str).isin(event_ids)
    ].reset_index(drop=True)
    return frames


@pytest.fixture(scope="module")
def prepared_plan() -> tuple[LocalDatasetRepository, script.PreparedRepair]:
    repository = LocalDatasetRepository(Path("data/cache"))
    prepared = script.prepare_repair(repository)
    assert prepared.summary["status"] == "validated_offline_plan"
    return repository, prepared


def _replace_security_subset(
    full: pd.DataFrame, subset: pd.DataFrame
) -> pd.DataFrame:
    security_ids = set(subset["security_id"].astype(str))
    return pd.concat(
        [
            full.loc[~full["security_id"].astype(str).isin(security_ids)],
            subset,
        ],
        ignore_index=True,
        sort=False,
    )


def _read_windowed_market_frame(
    repository: LocalDatasetRepository,
    *,
    dataset: str,
    version: str,
    windows: pd.DataFrame,
    full_security_ids: set[str],
) -> pd.DataFrame:
    full_windows = pd.DataFrame(
        [
            {
                "security_id": security_id,
                "start_session": "1900-01-01",
                "end_session": "2100-01-01",
            }
            for security_id in sorted(full_security_ids)
        ]
    )
    wanted = pd.concat([windows, full_windows], ignore_index=True, sort=False)
    wanted["_start"] = pd.to_datetime(wanted["start_session"], errors="raise")
    wanted["_end"] = pd.to_datetime(wanted["end_session"], errors="raise")
    merged_windows: list[dict[str, object]] = []
    for security_id, values in wanted.sort_values(
        ["security_id", "_start", "_end"]
    ).groupby("security_id", sort=False):
        current_start: pd.Timestamp | None = None
        current_end: pd.Timestamp | None = None
        for raw_start, raw_end in values[["_start", "_end"]].itertuples(
            index=False, name=None
        ):
            start = pd.Timestamp(raw_start)
            end = pd.Timestamp(raw_end)
            if current_start is None or start > current_end + pd.Timedelta(days=1):
                if current_start is not None:
                    merged_windows.append(
                        {
                            "security_id": security_id,
                            "_start": current_start,
                            "_end": current_end,
                        }
                    )
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        if current_start is not None:
            merged_windows.append(
                {
                    "security_id": security_id,
                    "_start": current_start,
                    "_end": current_end,
                }
            )
    wanted = pd.DataFrame(merged_windows)
    parts: list[pd.DataFrame] = []
    for year in range(max(2015, int(wanted["_start"].dt.year.min())), 2027):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31)
        year_windows = wanted.loc[
            wanted["_start"].le(year_end) & wanted["_end"].ge(year_start),
            ["security_id", "_start", "_end"],
        ].copy()
        if year_windows.empty:
            continue
        paths = [
            str(path)
            for path in repository.parquet_paths(
                dataset,
                version,
                min_session=year_start.date().isoformat(),
                max_session=year_end.date().isoformat(),
            )
        ]
        if not paths:
            continue
        connection = duckdb.connect()
        try:
            connection.execute("SET memory_limit='384MB'")
            connection.execute("SET threads=2")
            connection.execute("SET preserve_insertion_order=false")
            fetched = connection.execute(
                "SELECT * FROM read_parquet(?, union_by_name=true) "
                "WHERE security_id = ANY(?) "
                "AND CAST(session AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)",
                [
                    paths,
                    sorted(set(year_windows["security_id"].astype(str))),
                    year_start.date().isoformat(),
                    year_end.date().isoformat(),
                ],
            ).fetchdf()
        finally:
            connection.close()
        if fetched.empty:
            continue
        fetched["_session"] = pd.to_datetime(fetched["session"], errors="raise")
        joined = fetched.merge(year_windows, on="security_id", how="inner")
        joined = joined.loc[
            joined["_session"].between(joined["_start"], joined["_end"])
        ]
        parts.append(
            joined.loc[:, fetched.columns].drop(columns=["_session"])
        )
    frame = pd.concat(parts, ignore_index=True, sort=False)
    spec = dataset_spec(dataset)
    derived = [
        column
        for column in spec.partition_columns
        if column in frame.columns and column not in spec.required_columns
    ]
    if derived:
        frame = frame.drop(columns=derived)
    return frame.drop_duplicates(list(spec.primary_key), keep="last").reset_index(
        drop=True
    )


def _sparse_finalizer_repository(
    repository: LocalDatasetRepository,
    prepared: script.PreparedRepair,
    candidates,
):
    assert prepared.planned_release is not None
    base = prepared.release
    planned = prepared.planned_release
    full_master = repository.read_frame(
        "security_master", base.dataset_versions["security_master"]
    )
    full_history = repository.read_frame(
        "symbol_history", base.dataset_versions["symbol_history"]
    )
    master = _replace_security_subset(
        full_master, prepared.frames["security_master"]
    )
    history = _replace_security_subset(
        full_history, prepared.frames["symbol_history"]
    )
    actions = repository.read_frame(
        "corporate_actions", base.dataset_versions["corporate_actions"]
    )
    anchors = repository.read_frame(
        "index_constituent_anchors",
        base.dataset_versions["index_constituent_anchors"],
    )
    membership_events = repository.read_frame(
        "index_membership_events",
        base.dataset_versions["index_membership_events"],
    )

    window_rows: list[dict[str, str]] = []

    def add_window(security_id: str, start: str, end: str) -> None:
        if security_id and start and end:
            window_rows.append(
                {
                    "security_id": security_id,
                    "start_session": start,
                    "end_session": end,
                }
            )

    candidate_last = {
        candidate.security_id: pd.Timestamp(candidate.last_price_date).normalize()
        for candidate in candidates
    }
    for security_id, last in candidate_last.items():
        add_window(
            security_id,
            (last - pd.Timedelta(days=21)).date().isoformat(),
            last.date().isoformat(),
        )
    lifecycle_types = {
        "cash_merger",
        "stock_merger",
        "spinoff",
        "ticker_change",
        "delisting",
    }
    boundary_actions = actions.loc[
        actions["action_type"]
        .astype(str)
        .str.lower()
        .isin({*lifecycle_types, *RATIO_ACTIONS})
    ]
    for _, action in boundary_actions.iterrows():
        action_type = str(action.get("action_type") or "").strip().lower()
        boundary_value = (
            action.get("ex_date")
            if action_type in RATIO_ACTIONS and script._text(action.get("ex_date"))
            else action.get("effective_date")
        )
        effective = pd.to_datetime(boundary_value, errors="coerce")
        if pd.isna(effective):
            continue
        effective = effective.normalize()
        old_id = str(action.get("security_id") or "").strip()
        add_window(
            old_id,
            (effective - pd.Timedelta(days=21)).date().isoformat(),
            (effective + pd.Timedelta(days=21)).date().isoformat(),
        )
        if action_type in lifecycle_types:
            add_window(
                str(action.get("new_security_id") or "").strip(),
                effective.date().isoformat(),
                (effective + pd.Timedelta(days=21)).date().isoformat(),
            )

    for _, anchor in anchors.iterrows():
        anchor_date = pd.to_datetime(anchor.get("anchor_date"), errors="coerce")
        if pd.isna(anchor_date):
            continue
        anchor_date = anchor_date.normalize()
        add_window(
            str(anchor.get("security_id") or "").strip(),
            anchor_date.date().isoformat(),
            (anchor_date + pd.Timedelta(days=21)).date().isoformat(),
        )
    index_security_ids = set(anchors["security_id"].astype(str)) | set(
        membership_events["security_id"].astype(str)
    )
    for _, event in membership_events.iterrows():
        effective = pd.to_datetime(event.get("effective_date"), errors="coerce")
        if pd.isna(effective):
            continue
        effective = effective.normalize()
        add_window(
            str(event.get("security_id") or "").strip(),
            (effective - pd.Timedelta(days=21)).date().isoformat(),
            (effective + pd.Timedelta(days=21)).date().isoformat(),
        )
    completed = pd.Timestamp(planned.completed_session).normalize()
    for security_id in index_security_ids:
        add_window(
            security_id,
            (completed - pd.Timedelta(days=21)).date().isoformat(),
            completed.date().isoformat(),
        )

    report = json.loads(prepared.evidence_report_bytes)
    for candidate in candidates:
        record = report["records"][candidate.security_id]
        event = record.get("verified_event") or record.get("parsed")
        if not isinstance(event, dict):
            continue
        effective = pd.to_datetime(event.get("effective_date"), errors="coerce")
        if pd.isna(effective):
            continue
        effective = effective.normalize()
        add_window(
            candidate.security_id,
            (effective - pd.Timedelta(days=21)).date().isoformat(),
            effective.date().isoformat(),
        )
        try:
            successor_id = finalizer._successor_for_event(event, master, history)
        except (RuntimeError, ValueError):
            successor_id = ""
        add_window(
            successor_id,
            effective.date().isoformat(),
            (effective + pd.Timedelta(days=21)).date().isoformat(),
        )

    windows = pd.DataFrame(
        window_rows,
        columns=("security_id", "start_session", "end_session"),
    ).drop_duplicates()
    simple_security_ids = set(
        prepared.frames["daily_price_raw"]["security_id"].astype(str)
    )
    full_security_ids = {
        finalizer.FRC_EXACT_SECURITY_ID,
        finalizer.AVP_EXACT_SECURITY_ID,
        finalizer.AVP_EXACT_SUCCESSOR_ID,
        finalizer.SIVB_EXACT_SECURITY_ID,
        finalizer.TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE["security_id"],
        *(
            key.rsplit("|", 1)[0]
            for key in finalizer.REVIEWED_CROSS_BASIS_TERMINAL_PRICE_PROVENANCE
        ),
        *(
            value["successor_security_id"]
            for value in finalizer.REVIEWED_CROSS_BASIS_TERMINAL_PRICE_PROVENANCE.values()
        ),
    } - simple_security_ids
    prices = _read_windowed_market_frame(
        repository,
        dataset="daily_price_raw",
        version=base.dataset_versions["daily_price_raw"],
        windows=windows.loc[
            ~windows["security_id"].isin(simple_security_ids)
        ].reset_index(drop=True),
        full_security_ids=full_security_ids,
    )
    factors = _read_windowed_market_frame(
        repository,
        dataset="adjustment_factors",
        version=base.dataset_versions["adjustment_factors"],
        windows=windows.loc[
            ~windows["security_id"].isin(simple_security_ids)
        ].reset_index(drop=True),
        full_security_ids=full_security_ids,
    )
    prices = pd.concat(
        [prices, prepared.frames["daily_price_raw"]],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(["security_id", "session"], keep="last")
    factors = pd.concat(
        [factors, prepared.frames["adjustment_factors"]],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(["security_id", "session"], keep="last")
    factor_lineage = script._adjustment_source_version(
        planned.dataset_versions["daily_price_raw"],
        planned.dataset_versions["corporate_actions"],
    )
    factors["source_version"] = factor_lineage
    factors["calculated_at"] = script.REPAIR_REVIEWED_AT
    factors["source"] = "derived"
    factors["retrieved_at"] = script.REPAIR_REVIEWED_AT
    factors["source_hash"] = factor_lineage

    frames = {
        "security_master": master.reset_index(drop=True),
        "symbol_history": history.reset_index(drop=True),
        "daily_price_raw": prices.reset_index(drop=True),
        "corporate_actions": actions,
        "adjustment_factors": factors.reset_index(drop=True),
        "source_archive": prepared.frames["source_archive"],
        "lifecycle_resolutions": prepared.frames["lifecycle_resolutions"],
        "index_constituent_anchors": anchors,
        "index_membership_events": membership_events,
    }
    factor_manifest = SimpleNamespace(
        metadata=script._metadata_for_write(
            repository, prepared, "adjustment_factors"
        )
    )

    class SparseRepository:
        root = repository.root

        def read_frame(self, dataset: str, version: str) -> pd.DataFrame:
            if version == planned.dataset_versions.get(dataset):
                return frames[dataset].copy(deep=True)
            return repository.read_frame(dataset, version)

        def current_release(self):
            return planned, "projected-etag"

        def current_pointer(self, dataset: str):
            return SimpleNamespace(version=planned.dataset_versions[dataset]), (
                f"etag-{dataset}"
            )

        def manifest_for_version(self, dataset: str, version: str):
            if (
                dataset == "adjustment_factors"
                and version == planned.dataset_versions[dataset]
            ):
                return factor_manifest
            return repository.manifest_for_version(dataset, version)

    return SparseRepository(), frames


def _repaired_subset(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = script._rewrite_prices(frames["daily_price_raw"])
    master = script._bind_identity_source_urls(
        script._rewrite_identity(frames["security_master"], history=False),
        frames["corporate_actions"],
        history=False,
    )
    history = script._bind_identity_source_urls(
        script._rewrite_identity(frames["symbol_history"], history=True),
        frames["corporate_actions"],
        history=True,
    )
    factors, changes = script._prepare_factors(
        frames["adjustment_factors"],
        frames["daily_price_raw"],
        prices,
        source_version="planned-price+current-actions",
    )
    assert changes == 0
    return prices, factors, master, history


def test_registry_is_finite_code_pinned_and_excludes_symc() -> None:
    script._static_contract()
    assert {case.symbol for case in script.CASES} == {
        "FLT",
        "CDAY",
        "XEC",
        "HCP",
        "UTX",
        "COG",
        "CTRP",
    }
    assert sum(case.tail_rows for case in script.CASES) == 616
    assert script.registry_inventory_sha256() == (
        "5e5274d6ddec6eec037bdd127ea1c38a93c0e218ef96e3ed5b9e9af5fd3259ee"
    )
    assert all(not row["reassign_tail_to_successor"] for row in script.repair_registry())


def test_registry_tamper_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    changed = (replace(script.CASES[0], tail_rows=45), *script.CASES[1:])
    monkeypatch.setattr(script, "CASES", changed)
    with pytest.raises(RuntimeError, match="row inventory|fingerprint"):
        script._static_contract()


def test_lifecycle_hints_delta_is_exact_and_any_byte_tamper_fails() -> None:
    current = script._workspace_path(script.LIFECYCLE_HINTS_PATH).read_bytes()
    assert sha256_bytes(current) == script.EXPECTED_CURRENT_LIFECYCLE_HINTS_SHA256
    assert len(script.UTX_HINT_ADDITIVE_BLOCK) == 1164
    assert current.count(script.UTX_HINT_ADDITIVE_BLOCK) == 1
    assert sha256_bytes(script._legacy_lifecycle_hints_bytes(current)) == (
        script.EXPECTED_BASE_LIFECYCLE_HINTS_SHA256
    )

    for offset in (
        current.index(script.UTX_HINT_ADDITIVE_BLOCK) + 20,
        current.index(b"identity_bound_hints:"),
    ):
        tampered = bytearray(current)
        tampered[offset] ^= 1
        with pytest.raises(RuntimeError, match="Current lifecycle hints bytes changed"):
            script._legacy_lifecycle_hints_bytes(bytes(tampered))


def test_fresh_lifecycle_report_and_permanent_utx_resolution_are_bound(
    prepared_plan: tuple[LocalDatasetRepository, script.PreparedRepair],
) -> None:
    repository, prepared = prepared_plan
    assert prepared.planned_release is not None
    report = json.loads(prepared.evidence_report_bytes)
    assert report["hints_sha256"] == script.EXPECTED_CURRENT_LIFECYCLE_HINTS_SHA256
    assert report["candidate_set_sha256"] == (
        script.EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
    )
    assert report["release_version"] == prepared.planned_release.version
    script._validate_utx_report_evidence(report)

    context = script._project_candidate_context(
        script._read_candidate_context(repository, prepared.release)
    )
    candidates = script._build_candidate_values(context, prepared.planned_release)
    binding = script._report_binding(report, prepared.planned_release, candidates)
    script.validate_lifecycle_report_binding(
        report, binding, purpose="test projected simple7 report"
    )

    utx_security_id = next(
        case.security_id for case in script.CASES if case.symbol == "UTX"
    )
    rows = prepared.frames["lifecycle_resolutions"].loc[
        prepared.frames["lifecycle_resolutions"]["security_id"]
        .astype(str)
        .eq(utx_security_id)
    ]
    assert len(rows) == 1
    resolution = rows.iloc[0]
    assert resolution["resolution"] == "exception"
    assert resolution["exception_code"] == "unsupported_consideration"
    assert resolution["exception_reason"] == script.UTX_EXCEPTION_REASON
    assert script._text(resolution["recheck_after"]) == ""
    assert resolution["source_hash"] == script.UTX_DISTRIBUTION_SOURCE_HASH
    assert prepared.summary["lifecycle_coverage"] == {
        "coverage_gate_version": 1,
        "selection_rule": "us_terminal_v1",
        "candidate_set_sha256": (
            script.EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
        ),
        "resolution_set_sha256": (
            script.EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256
        ),
        **script.EXPECTED_LIFECYCLE_COVERAGE,
    }

    metadata = script._metadata_for_write(
        repository, prepared, "lifecycle_resolutions"
    )
    for key, value in prepared.summary["lifecycle_coverage"].items():
        assert metadata[key] == value
    assert metadata["evidence_report_sha256"] == sha256_bytes(
        prepared.evidence_report_bytes
    )
    assert metadata["utx_resolution"] == "fail_closed_unsupported_consideration"


def test_lifecycle_report_binding_and_utx_artifact_tamper_fail_closed(
    prepared_plan: tuple[LocalDatasetRepository, script.PreparedRepair],
) -> None:
    repository, prepared = prepared_plan
    assert prepared.planned_release is not None
    report = json.loads(prepared.evidence_report_bytes)
    context = script._project_candidate_context(
        script._read_candidate_context(repository, prepared.release)
    )
    candidates = script._build_candidate_values(context, prepared.planned_release)
    binding = script._report_binding(report, prepared.planned_release, candidates)

    report["hints_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="hints_sha256"):
        script.validate_lifecycle_report_binding(
            report, binding, purpose="test tampered report"
        )

    report = json.loads(prepared.evidence_report_bytes)
    report["official_exception_evidence"][
        "utx_2020_carr_otis_distributions"
    ]["observed_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="official exception evidence"):
        script._validate_utx_report_evidence(report)


def test_lifecycle_report_payload_persist_conflict_and_rollback(tmp_path: Path) -> None:
    payload = b'{"report":"fresh"}\n'
    object_path = f"archives/{sha256_bytes(payload)}.json.gz"
    prepared = script.PreparedRepair(
        release=DataRelease(
            version="base",
            created_at="2026-07-19T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions={},
        ),
        release_etag="etag",
        pointer_etags={},
        planned_versions={},
        frames={},
        summary={"lifecycle_evidence_report_sha256": sha256_bytes(payload)},
        planned_release=DataRelease(
            version="planned",
            created_at="2026-07-19T00:01:00Z",
            completed_session="2026-07-15",
            dataset_versions={},
        ),
        evidence_report_bytes=payload,
        evidence_report_object_path=object_path,
    )
    repository = SimpleNamespace(root=tmp_path)
    assert script._persist_lifecycle_report_payload(repository, prepared)
    assert not script._persist_lifecycle_report_payload(repository, prepared)

    destination = tmp_path / object_path
    destination.write_bytes(gzip.compress(b"tampered", mtime=0))
    with pytest.raises(RuntimeError, match="conflicts"):
        script._persist_lifecycle_report_payload(repository, prepared)
    assert script._remove_created_lifecycle_report_payload(
        repository, prepared, created=True
    )

    destination.write_bytes(gzip.compress(payload, mtime=0))
    assert script._remove_created_lifecycle_report_payload(
        repository, prepared, created=True
    ) == ()
    assert not destination.exists()


def test_old_exact_tail_and_identity_pins_hold(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    script._old_state(pinned_frames)


def test_old_price_tail_tamper_fails_closed(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    changed = {key: value.copy(deep=True) for key, value in pinned_frames.items()}
    case = next(case for case in script.CASES if case.symbol == "FLT")
    sessions = script._session_series(changed["daily_price_raw"])
    row = changed["daily_price_raw"].index[
        changed["daily_price_raw"]["security_id"].astype(str).eq(case.security_id)
        & sessions.eq(case.transition_date)
    ][0]
    changed["daily_price_raw"].loc[row, "close"] += 0.01
    with pytest.raises(RuntimeError, match="FLT old price-tail bytes changed"):
        script._old_state(changed)


def test_exact_price_repair_deletes_616_and_never_reassigns_cog(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    old = pinned_frames["daily_price_raw"]
    repaired = script._rewrite_prices(old)
    assert len(old) - len(repaired) == 616
    for case in script.CASES:
        assert script._old_tail(repaired, case).empty
        remaining = repaired.loc[repaired["security_id"].astype(str).eq(case.security_id)]
        assert script._date(remaining["session"].max()) == case.old_last_good_session

    hcp = next(case for case in script.CASES if case.symbol == "HCP")
    peak = repaired.loc[
        repaired["security_id"].astype(str).eq(hcp.successor_security_id)
        & script._session_series(repaired).eq("2019-11-05")
    ].iloc[0]
    assert [float(peak[field]) for field in ("open", "high", "low", "close", "volume")] == [
        34.75,
        34.82,
        33.85,
        34.41,
        8_054_269.0,
    ]
    assert str(peak["source_hash"]) == hcp.old_tail_source_hash

    xec = next(case for case in script.CASES if case.symbol == "XEC")
    cog = next(case for case in script.CASES if case.symbol == "COG")
    # XEC's flat synthetic rows are not copied into independently traded COG.
    old_cog = old.loc[
        old["security_id"].astype(str).eq(cog.security_id)
        & script._session_series(old).eq("2021-10-01")
    ].reset_index(drop=True)
    new_cog = repaired.loc[
        repaired["security_id"].astype(str).eq(cog.security_id)
        & script._session_series(repaired).eq("2021-10-01")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_cog, new_cog, check_dtype=True)
    assert xec.successor_security_id == cog.security_id


def test_hcp_replacement_source_tamper_fails_closed(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    prices = pinned_frames["daily_price_raw"].copy(deep=True)
    case = next(case for case in script.CASES if case.symbol == "HCP")
    row = prices.index[
        prices["security_id"].astype(str).eq(case.security_id)
        & script._session_series(prices).eq(case.transition_date)
    ][0]
    prices.loc[row, "source_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="HCP exact replacement OHLCV/source hash"):
        script._rewrite_prices(prices)


def test_identity_closes_at_last_good_with_exact_official_provenance(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    _, _, master, history = _repaired_subset(pinned_frames)
    for case in script.CASES:
        action = pinned_frames["corporate_actions"].loc[
            pinned_frames["corporate_actions"]["event_id"].astype(str).eq(case.event_id)
        ].iloc[0]
        for frame, is_history, end_field in (
            (master, False, "active_to"),
            (history, True, "effective_to"),
        ):
            row = script._identity_rows(frame, case, history=is_history).iloc[0]
            assert script._date(row[end_field]) == case.old_last_good_session
            assert row["source"] == script.REPAIRED_IDENTITY_SOURCE
            assert row["source_hash"] == case.official_source_hash
            assert row["source_url"] == action["source_url"]


def test_minimal_factor_delete_preserves_all_retained_economics_and_lineage(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    prices, factors, _, _ = _repaired_subset(pinned_frames)
    assert len(pinned_frames["adjustment_factors"]) - len(factors) == 616
    assert script._affected_factor_economics_sha256(factors) == (
        script.EXPECTED_REPAIRED_AFFECTED_FACTOR_ECONOMICS_SHA256
    )
    assert set(factors["source_version"].astype(str)) == {
        "planned-price+current-actions"
    }
    factor_keys = set(
        zip(
            factors["security_id"].astype(str),
            pd.to_datetime(factors["session"]).dt.normalize(),
        )
    )
    price_keys = set(
        zip(
            prices["security_id"].astype(str),
            pd.to_datetime(prices["session"]).dt.normalize(),
        )
    )
    assert factor_keys == price_keys


def test_in_place_factor_materialization_keeps_one_frame(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    factors = pinned_frames["adjustment_factors"].copy(deep=True)
    output, changed = script._rewrite_factors_minimal(
        factors,
        source_version="planned-price+current-actions",
        copy_frame=False,
    )
    assert output is factors
    assert changed == 0
    assert len(output) == len(pinned_frames["adjustment_factors"]) - 616
    assert script._affected_factor_economics_sha256(output) == (
        script.EXPECTED_REPAIRED_AFFECTED_FACTOR_ECONOMICS_SHA256
    )


def test_affected_factor_tamper_fails_closed(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    factors = pinned_frames["adjustment_factors"].copy(deep=True)
    factors.loc[factors.index[0], "total_return_factor"] += 1e-6
    prices = script._rewrite_prices(pinned_frames["daily_price_raw"])
    with pytest.raises(RuntimeError, match="baseline factor economics changed"):
        script._prepare_factors(
            factors,
            pinned_frames["daily_price_raw"],
            prices,
            source_version="planned-price+current-actions",
        )


def test_hcp_replacement_changes_no_triple_supertrend_state(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    prices, factors, _, _ = _repaired_subset(pinned_frames)
    impact = script._hcp_signal_impact(
        pinned_frames["daily_price_raw"],
        prices,
        pinned_frames["adjustment_factors"],
        factors,
    )
    assert all(
        value["count"] == 0
        for mode in impact.values()
        for value in mode.values()
    )


def test_cross_validation_work_is_exact_plan_not_generic_tolerance() -> None:
    plan = script.cross_validation_change_plan()
    assert plan["status"] == "post_apply_policy_work_required"
    assert plan["exception_count"] == 7
    assert not plan["generic_date_tolerance"]
    assert not plan["generic_successor_inheritance"]
    assert len({row["target_id"] for row in plan["entries"]}) == 7
    assert all(len(row["official_source_hash"]) == 64 for row in plan["entries"])
    assert all(len(row["successor_target_id"]) == 64 for row in plan["entries"])
    xec = next(row for row in plan["entries"] if row["symbol"] == "XEC")
    assert xec["successor_symbol"] == "COG"
    assert xec["successor_path"] == "finite_chain_required"


def test_cross_validation_draft_matches_code_pinned_plan() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs/drafts/us_identity_price_tail_cross_validation_plan.yaml"
    )
    draft = yaml.safe_load(path.read_text())
    plan = script.cross_validation_change_plan()
    assert draft["status"] == "post_apply_only_not_authorized"
    assert draft["repair_registry_sha256"] == script.registry_inventory_sha256()
    assert draft["candidate_content_sha256"] == (
        script.EXPECTED_CANDIDATE_CONTENT_SHA256
    )
    assert draft["exception_inventory_sha256"] == plan["exception_inventory_sha256"]
    assert draft["generic_date_tolerance"] is False
    assert draft["generic_successor_inheritance"] is False
    assert {
        (row["symbol"], row["target_id"], row["successor_target_id"])
        for row in draft["entries"]
    } == {
        (row["symbol"], row["target_id"], row["successor_target_id"])
        for row in plan["entries"]
    }


def test_repaired_state_is_content_and_lineage_exact(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    prices, factors, master, history = _repaired_subset(pinned_frames)
    release = DataRelease(
        version="repaired-release",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions={
            "daily_price_raw": "planned-price",
            "corporate_actions": "current-actions",
            "adjustment_factors": "planned-factors",
        },
    )
    script._verify_repaired_state(
        {
            "daily_price_raw": prices,
            "adjustment_factors": factors,
            "security_master": master,
            "symbol_history": history,
            "corporate_actions": pinned_frames["corporate_actions"],
        },
        release,
    )


def test_candidate_content_hash_is_deterministic_and_version_free(
    pinned_frames: dict[str, pd.DataFrame],
) -> None:
    prices, factors, master, history = _repaired_subset(pinned_frames)
    first = script._candidate_content_projection(
        prices=prices, factors=factors, master=master, history=history
    )
    factors = factors.copy(deep=True)
    factors["source_version"] = "different-random-planned-version"
    factors["source_hash"] = "different-random-planned-version"
    second = script._candidate_content_projection(
        prices=prices, factors=factors, master=master, history=history
    )
    assert first == second
    assert first["aggregate"] == script.EXPECTED_CANDIDATE_CONTENT_SHA256


def test_plan_frame_reader_projects_heavy_tables_by_sid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subset_calls: list[str] = []
    full_reads: list[str] = []

    def subset(_repository, dataset, _version, _security_ids):
        subset_calls.append(dataset)
        return pd.DataFrame()

    repository = SimpleNamespace(
        read_frame=lambda dataset, _version: (
            full_reads.append(dataset) or pd.DataFrame()
        )
    )
    release = SimpleNamespace(
        dataset_versions={
            dataset: f"v-{dataset}"
            for dataset in (
                "daily_price_raw",
                "adjustment_factors",
                "security_master",
                "symbol_history",
                    "corporate_actions",
                    "source_archive",
                    "lifecycle_resolutions",
                )
            }
    )
    monkeypatch.setattr(script, "_read_security_subset", subset)
    script._read_affected_frames(repository, release)
    assert set(subset_calls) == {
        "daily_price_raw",
        "adjustment_factors",
        "security_master",
        "symbol_history",
        "corporate_actions",
    }
    assert subset_calls.count("daily_price_raw") == 2
    assert full_reads == ["source_archive", "lifecycle_resolutions"]


def test_manifest_idempotence_requires_every_exact_pin() -> None:
    output_versions = {
        **{dataset: f"old-{dataset}" for dataset in script.REQUIRED_DATASETS},
        **{dataset: f"new-{dataset}" for dataset in script.WRITE_DATASETS},
    }
    release = DataRelease(
        version="repaired-release",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions=output_versions,
        warnings=(script.UTX_RELEASE_WARNING,),
    )
    common = {
        "schema": script.REPAIR_SCHEMA,
        "operation": script.OPERATION,
        "input_release_version": script.PINNED_RELEASE_VERSION,
        "output_versions": output_versions,
        "registry": script.repair_registry(),
        "registry_inventory_sha256": script.registry_inventory_sha256(),
        "candidate_content_sha256": script.EXPECTED_CANDIDATE_CONTENT_SHA256,
        "lifecycle_candidate_set_sha256": (
            script.EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
        ),
        "lifecycle_resolution_set_sha256": (
            script.EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256
        ),
        "lifecycle_evidence_report_sha256": "b" * 64,
        "lifecycle_evidence_report_object_path": "archives/report.json.gz",
        "evidence_report_sha256": "b" * 64,
        "evidence_report_object_path": "archives/report.json.gz",
        "lifecycle_report_release_version": release.version,
        "lifecycle_coverage": {
            **script.EXPECTED_LIFECYCLE_COVERAGE,
            "candidate_set_sha256": (
                script.EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
            ),
            "resolution_set_sha256": (
                script.EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256
            ),
        },
        **script.EXPECTED_LIFECYCLE_COVERAGE,
        "candidate_set_sha256": (
            script.EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
        ),
        "resolution_set_sha256": (
            script.EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256
        ),
        "utx_release_warning": script.UTX_RELEASE_WARNING,
        "network_accessed": False,
        "eodhd_calls": 0,
        "r2_accessed": False,
    }
    factor_lineage = script._adjustment_source_version(
        output_versions["daily_price_raw"], output_versions["corporate_actions"]
    )
    manifests = {
        dataset: SimpleNamespace(metadata=dict(common))
        for dataset in script.WRITE_DATASETS
    }
    manifests["adjustment_factors"].metadata.update(
        {
            "source_version": factor_lineage,
            "source_daily_price_version": output_versions["daily_price_raw"],
            "source_corporate_actions_version": output_versions["corporate_actions"],
            "economic_rows_changed": 0,
        }
    )
    repository = SimpleNamespace(
        manifest_for_version=lambda dataset, _version: manifests[dataset]
    )
    assert script._exact_repair_manifests(repository, release)
    manifests["daily_price_raw"].metadata["registry_inventory_sha256"] = "0" * 64
    assert not script._exact_repair_manifests(repository, release)


def _pointer(dataset: str, version: str) -> bytes:
    return CurrentPointer(
        dataset=dataset,
        version=version,
        manifest_path=f"datasets/{dataset}/versions/{version}/manifest.json",
        manifest_sha256="a" * 64,
        updated_at="2026-07-19T00:00:00Z",
    ).to_bytes()


def test_rollback_restores_release_and_all_write_pointers(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    old_versions = {dataset: f"old-{dataset}" for dataset in script.REQUIRED_DATASETS}
    planned = {dataset: f"new-{dataset}" for dataset in script.WRITE_DATASETS}
    old_release = DataRelease(
        version="old-release",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions=old_versions,
    )
    new_release = DataRelease(
        version="new-release",
        created_at="2026-07-19T00:01:00Z",
        completed_session="2026-07-15",
        dataset_versions={**old_versions, **planned},
    )
    old_pointers = {
        dataset: _pointer(dataset, old_versions[dataset])
        for dataset in script.WRITE_DATASETS
    }
    store.put("releases/current.json", new_release.to_bytes())
    for dataset in script.WRITE_DATASETS:
        store.put(f"datasets/{dataset}/current.json", _pointer(dataset, planned[dataset]))
    repository = SimpleNamespace(
        objects=store,
        current_key=lambda dataset: f"datasets/{dataset}/current.json",
    )
    errors = script._restore_transaction(
        repository,
        old_release_bytes=old_release.to_bytes(),
        old_pointer_bytes=old_pointers,
        planned_versions=planned,
        planned_release_bytes=new_release.to_bytes(),
        committed_release_version=new_release.version,
    )
    assert errors == ()
    assert store.get("releases/current.json").data == old_release.to_bytes()
    for dataset in script.WRITE_DATASETS:
        assert store.get(f"datasets/{dataset}/current.json").data == old_pointers[dataset]


def test_rollback_failure_injection_refuses_foreign_metadata_release(
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path)
    old_versions = {dataset: f"old-{dataset}" for dataset in script.REQUIRED_DATASETS}
    planned = {dataset: f"new-{dataset}" for dataset in script.WRITE_DATASETS}
    expected_versions = {**old_versions, **planned}
    old_release = DataRelease(
        version="old-release",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions=old_versions,
    )
    planned_release = DataRelease(
        version="our-planned-release",
        created_at="2026-07-19T00:01:00Z",
        completed_session="2026-07-15",
        dataset_versions=expected_versions,
    )
    foreign_release = DataRelease(
        version="foreign-metadata-release",
        created_at="2026-07-19T00:02:00Z",
        completed_session="2026-07-15",
        dataset_versions=expected_versions,
        warnings=("foreign metadata-only publication",),
    )
    old_pointers = {
        dataset: _pointer(dataset, old_versions[dataset])
        for dataset in script.WRITE_DATASETS
    }

    # Simulate a foreign publisher replacing current.json after our release
    # commit but before an injected failure starts rollback.
    store.put("releases/current.json", foreign_release.to_bytes())
    for dataset in script.WRITE_DATASETS:
        store.put(f"datasets/{dataset}/current.json", _pointer(dataset, planned[dataset]))
    repository = SimpleNamespace(
        objects=store,
        current_key=lambda dataset: f"datasets/{dataset}/current.json",
    )
    release_before = store.get("releases/current.json")
    pointers_before = {
        dataset: store.get(f"datasets/{dataset}/current.json")
        for dataset in script.WRITE_DATASETS
    }

    errors = script._restore_transaction(
        repository,
        old_release_bytes=old_release.to_bytes(),
        old_pointer_bytes=old_pointers,
        planned_versions=planned,
        planned_release_bytes=planned_release.to_bytes(),
        committed_release_version=planned_release.version,
    )

    assert errors
    assert "unexpected release during identity-tail rollback" in errors[0]
    release_after = store.get("releases/current.json")
    assert (release_after.data, release_after.etag) == (
        release_before.data,
        release_before.etag,
    )
    for dataset in script.WRITE_DATASETS:
        pointer_after = store.get(f"datasets/{dataset}/current.json")
        assert (pointer_after.data, pointer_after.etag) == (
            pointers_before[dataset].data,
            pointers_before[dataset].etag,
        )


def test_rollback_preflight_refuses_foreign_pointer_without_any_mutation(
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path)
    old_versions = {dataset: f"old-{dataset}" for dataset in script.REQUIRED_DATASETS}
    planned = {dataset: f"new-{dataset}" for dataset in script.WRITE_DATASETS}
    old_release = DataRelease(
        version="old-release",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions=old_versions,
    )
    planned_release = DataRelease(
        version="our-planned-release",
        created_at="2026-07-19T00:01:00Z",
        completed_session="2026-07-15",
        dataset_versions={**old_versions, **planned},
    )
    old_pointers = {
        dataset: _pointer(dataset, old_versions[dataset])
        for dataset in script.WRITE_DATASETS
    }
    store.put("releases/current.json", planned_release.to_bytes())
    foreign_dataset = script.WRITE_DATASETS[-1]
    for dataset in script.WRITE_DATASETS:
        version = "foreign-version" if dataset == foreign_dataset else planned[dataset]
        store.put(f"datasets/{dataset}/current.json", _pointer(dataset, version))
    repository = SimpleNamespace(
        objects=store,
        current_key=lambda dataset: f"datasets/{dataset}/current.json",
    )
    release_before = store.get("releases/current.json")
    pointers_before = {
        dataset: store.get(f"datasets/{dataset}/current.json")
        for dataset in script.WRITE_DATASETS
    }

    errors = script._restore_transaction(
        repository,
        old_release_bytes=old_release.to_bytes(),
        old_pointer_bytes=old_pointers,
        planned_versions=planned,
        planned_release_bytes=planned_release.to_bytes(),
        committed_release_version=planned_release.version,
    )

    assert errors
    assert f"unexpected {foreign_dataset} pointer" in errors[0]
    release_after = store.get("releases/current.json")
    assert (release_after.data, release_after.etag) == (
        release_before.data,
        release_before.etag,
    )
    for dataset in script.WRITE_DATASETS:
        pointer_after = store.get(f"datasets/{dataset}/current.json")
        assert (pointer_after.data, pointer_after.etag) == (
            pointers_before[dataset].data,
            pointers_before[dataset].etag,
        )


def test_apply_failure_injection_preserves_foreign_publication_and_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = {*script.REQUIRED_DATASETS, "lifecycle_resolutions"}
    old_versions = {dataset: f"old-{dataset}" for dataset in datasets}
    planned_versions = {
        dataset: f"new-{dataset}" for dataset in script.WRITE_DATASETS
    }
    base_release = DataRelease(
        version="base-release",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions=old_versions,
    )
    planned_release = DataRelease(
        version="our-planned-release",
        created_at="2026-07-19T00:01:00Z",
        completed_session=base_release.completed_session,
        dataset_versions={**old_versions, **planned_versions},
        quality=base_release.quality,
        warnings=(script.UTX_RELEASE_WARNING,),
    )
    report_payload = b'{"report":"rollback evidence"}\n'
    report_path = f"archives/{sha256_bytes(report_payload)}.json.gz"

    class FakeRepository:
        def __init__(self) -> None:
            self.root = tmp_path
            self.objects = LocalObjectStore(tmp_path)
            self.objects.put("releases/current.json", base_release.to_bytes())
            for dataset, version in old_versions.items():
                self.objects.put(self.current_key(dataset), _pointer(dataset, version))

        @staticmethod
        def current_key(dataset: str) -> str:
            return f"datasets/{dataset}/current.json"

        def current_release(self):
            value = self.objects.get("releases/current.json")
            return DataRelease.from_bytes(value.data), value.etag

        def current_pointer(self, dataset: str):
            value = self.objects.get(self.current_key(dataset))
            return CurrentPointer.from_bytes(value.data), value.etag

        def write_frame(self, dataset: str, _frame: pd.DataFrame, **kwargs):
            version = kwargs["version"]
            self.objects.put(
                self.current_key(dataset),
                _pointer(dataset, version),
                if_match=kwargs["expected_pointer_etag"],
            )
            return SimpleNamespace(
                conflict=False,
                manifest=SimpleNamespace(version=version),
            )

    repository = FakeRepository()
    _, release_etag = repository.current_release()
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1] for dataset in old_versions
    }
    prepared = script.PreparedRepair(
        release=base_release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frames={},
        summary={
            "status": "validated_offline_plan",
            "lifecycle_evidence_report_sha256": sha256_bytes(report_payload),
            "lifecycle_evidence_report_object_path": report_path,
        },
        planned_release=planned_release,
        evidence_report_bytes=report_payload,
        evidence_report_object_path=report_path,
    )
    monkeypatch.setattr(
        script,
        "_materialize_full_write_frame",
        lambda _repository, _prepared, _dataset: pd.DataFrame(),
    )
    monkeypatch.setattr(script, "_metadata_for_write", lambda *_args: {})
    captured: dict[str, Any] = {}

    def inject(stage: str) -> None:
        if stage != "after_release_commit":
            return
        current = repository.objects.get("releases/current.json")
        foreign = DataRelease(
            version="foreign-metadata-release",
            created_at="2026-07-19T00:02:00Z",
            completed_session=planned_release.completed_session,
            dataset_versions=planned_release.dataset_versions,
            quality=planned_release.quality,
            warnings=(*planned_release.warnings, "foreign publication"),
        )
        repository.objects.put(
            "releases/current.json",
            foreign.to_bytes(),
            if_match=current.etag,
        )
        captured["release"] = repository.objects.get("releases/current.json")
        captured["pointers"] = {
            dataset: repository.objects.get(repository.current_key(dataset))
            for dataset in script.WRITE_DATASETS
        }
        raise RuntimeError("injected failure after foreign publication")

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        script.apply_repair(repository, prepared, inject_failure=inject)

    release_after = repository.objects.get("releases/current.json")
    assert (release_after.data, release_after.etag) == (
        captured["release"].data,
        captured["release"].etag,
    )
    for dataset in script.WRITE_DATASETS:
        pointer_after = repository.objects.get(repository.current_key(dataset))
        pointer_before = captured["pointers"][dataset]
        assert (pointer_after.data, pointer_after.etag) == (
            pointer_before.data,
            pointer_before.etag,
        )
    archive = tmp_path / report_path
    assert archive.is_file()
    assert gzip.decompress(archive.read_bytes()) == report_payload
    recovery = tuple((tmp_path / script.RECOVERY_DIR).glob("*.json"))
    assert len(recovery) == 1
    assert json.loads(recovery[0].read_bytes())["status"] == "rollback_failed"


def test_compare_and_swap_guard_rejects_changed_release() -> None:
    release = DataRelease(
        version="base",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions={dataset: f"v-{dataset}" for dataset in script.REQUIRED_DATASETS},
    )
    prepared = script.PreparedRepair(
        release=release,
        release_etag="expected-etag",
        pointer_etags={dataset: "pointer-etag" for dataset in script.REQUIRED_DATASETS},
        planned_versions={},
        frames={},
        summary={},
    )
    repository = SimpleNamespace(
        current_release=lambda: (release, "changed-etag"),
        current_pointer=lambda dataset: (
            SimpleNamespace(version=release.dataset_versions[dataset]),
            "pointer-etag",
        ),
    )
    with pytest.raises(RuntimeError, match="release changed"):
        script._assert_inputs_unchanged(repository, prepared)


def test_stale_plan_rejects_lifecycle_resolution_pointer_drift() -> None:
    datasets = {*script.REQUIRED_DATASETS, "lifecycle_resolutions"}
    release = DataRelease(
        version="base",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions={dataset: f"v-{dataset}" for dataset in datasets},
    )
    prepared = script.PreparedRepair(
        release=release,
        release_etag="release-etag",
        pointer_etags={dataset: f"etag-{dataset}" for dataset in datasets},
        planned_versions={},
        frames={},
        summary={},
    )

    def current_pointer(dataset: str):
        etag = f"etag-{dataset}"
        if dataset == "lifecycle_resolutions":
            etag = "drifted-lifecycle-etag"
        return SimpleNamespace(version=f"v-{dataset}"), etag

    repository = SimpleNamespace(
        current_release=lambda: (release, "release-etag"),
        current_pointer=current_pointer,
    )
    with pytest.raises(RuntimeError, match="lifecycle_resolutions pointer changed"):
        script._assert_inputs_unchanged(repository, prepared)


def test_post_commit_preserves_synthetic_extra_dataset_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = {*script.REQUIRED_DATASETS, "lifecycle_resolutions", "synthetic_extra"}
    versions = {dataset: f"v-{dataset}" for dataset in datasets}
    committed = DataRelease(
        version="committed",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions=versions,
    )
    expected_etags = {dataset: f"etag-{dataset}" for dataset in datasets}

    def current_pointer(dataset: str):
        return SimpleNamespace(version=versions[dataset]), expected_etags[dataset]

    repository = SimpleNamespace(
        current_release=lambda: (committed, "release-etag"),
        current_pointer=current_pointer,
    )
    monkeypatch.setattr(
        script,
        "prepare_repair",
        lambda _repository: SimpleNamespace(summary={"status": "already_repaired"}),
    )
    script._assert_applied_release(
        repository,
        committed,
        expected_out_of_scope_pointer_etags=expected_etags,
    )
    expected_etags["synthetic_extra"] = "drifted-extra-etag"
    with pytest.raises(RuntimeError, match="Out-of-scope pointer changed: synthetic_extra"):
        script._assert_applied_release(
            repository,
            committed,
            expected_out_of_scope_pointer_etags={
                **expected_etags,
                "synthetic_extra": "etag-synthetic_extra",
            },
        )


def test_successful_apply_replays_as_idempotent_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = {
        *script.REQUIRED_DATASETS,
        "lifecycle_resolutions",
        "synthetic_extra",
    }
    base_versions = {dataset: f"old-{dataset}" for dataset in datasets}
    planned_versions = {
        dataset: f"new-{dataset}" for dataset in script.WRITE_DATASETS
    }
    base_release = DataRelease(
        version="base-release",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions=base_versions,
    )

    class FakeRepository:
        def __init__(self) -> None:
            self.root = tmp_path
            self.objects = LocalObjectStore(tmp_path)
            self.objects.put("releases/current.json", base_release.to_bytes())
            for dataset, version in base_versions.items():
                self.objects.put(self.current_key(dataset), _pointer(dataset, version))

        @staticmethod
        def current_key(dataset: str) -> str:
            return f"datasets/{dataset}/current.json"

        def current_release(self):
            value = self.objects.get("releases/current.json")
            return DataRelease.from_bytes(value.data), value.etag

        def current_pointer(self, dataset: str):
            value = self.objects.get(self.current_key(dataset))
            return CurrentPointer.from_bytes(value.data), value.etag

        def write_frame(self, dataset: str, _frame: pd.DataFrame, **kwargs):
            version = kwargs["version"]
            pointer = _pointer(dataset, version)
            self.objects.put(
                self.current_key(dataset),
                pointer,
                if_match=kwargs["expected_pointer_etag"],
            )
            return SimpleNamespace(
                conflict=False, manifest=SimpleNamespace(version=version)
            )

        def commit_release(
            self,
            completed_session: str,
            dataset_versions: dict[str, str],
            *,
            quality: str,
            warnings: tuple[str, ...],
            expected_etag: str,
        ) -> DataRelease:
            committed = DataRelease(
                version="committed-release",
                created_at="2026-07-19T00:01:00Z",
                completed_session=completed_session,
                dataset_versions=dataset_versions,
                quality=quality,
                warnings=warnings,
            )
            self.objects.put(
                "releases/current.json",
                committed.to_bytes(),
                if_match=expected_etag,
            )
            return committed

    repository = FakeRepository()

    def make_prepared() -> script.PreparedRepair:
        release, release_etag = repository.current_release()
        etags = {
            dataset: repository.current_pointer(dataset)[1]
            for dataset in release.dataset_versions
        }
        repaired = release.version == "committed-release"
        return script.PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=etags,
            planned_versions={} if repaired else planned_versions,
            frames={},
            summary={
                "status": "already_repaired" if repaired else "validated_offline_plan"
            },
        )

    monkeypatch.setattr(script, "prepare_repair", lambda _repository: make_prepared())
    monkeypatch.setattr(
        script,
        "_materialize_full_write_frame",
        lambda _repository, _prepared, _dataset: pd.DataFrame(),
    )
    monkeypatch.setattr(script, "_metadata_for_write", lambda *_args: {})

    first = script.apply_repair(repository, make_prepared())
    assert first["status"] == "applied"
    assert first["writes_performed"] is True
    lifecycle_pointer, _ = repository.current_pointer("lifecycle_resolutions")
    assert lifecycle_pointer.version == planned_versions["lifecycle_resolutions"]
    extra_pointer, _ = repository.current_pointer("synthetic_extra")
    assert extra_pointer.version == base_versions["synthetic_extra"]

    second = script.apply_repair(repository, make_prepared())
    assert second["status"] == "already_repaired"
    assert second["writes_performed"] is False


def test_apply_materializes_only_one_full_dataset_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[tuple[str, str]] = []
    frame = pd.DataFrame({"sentinel": [1]})
    repository = SimpleNamespace(
        read_frame=lambda dataset, version: (
            reads.append((dataset, version)) or frame
        )
    )
    prepared = SimpleNamespace(
        release=SimpleNamespace(
            dataset_versions={
                "adjustment_factors": "old-factors",
                "corporate_actions": "old-actions",
            }
        ),
        planned_versions={"daily_price_raw": "new-prices"},
    )
    monkeypatch.setattr(
        script,
        "_rewrite_factors_minimal",
        lambda value, **_kwargs: (value, 0),
    )
    output = script._materialize_full_write_frame(
        repository, prepared, "adjustment_factors"
    )
    assert output is frame
    assert reads == [("adjustment_factors", "old-factors")]


def test_module_has_no_remote_client_path() -> None:
    source = SCRIPT_PATH.read_text()
    assert "import requests" not in source
    assert "urllib.request" not in source
    assert "boto3" not in source
    assert "eodhd.com/api" not in source
    assert "current_plan.frames[dataset]" not in source
    assert "sequential_one_dataset_at_a_time" in source


def test_sparse_projected_release_passes_full_lifecycle_finalizer(
    prepared_plan: tuple[LocalDatasetRepository, script.PreparedRepair],
    record_property,
) -> None:
    repository, prepared = prepared_plan
    assert prepared.planned_release is not None
    context = script._project_candidate_context(
        script._read_candidate_context(repository, prepared.release)
    )
    candidates = script._build_candidate_values(context, prepared.planned_release)
    sparse_repository, projected_frames = _sparse_finalizer_repository(
        repository, prepared, candidates
    )
    assert len(projected_frames["daily_price_raw"]) < 100_000
    assert set(
        zip(
            projected_frames["daily_price_raw"]["security_id"].astype(str),
            pd.to_datetime(projected_frames["daily_price_raw"]["session"]),
        )
    ) == set(
        zip(
            projected_frames["adjustment_factors"]["security_id"].astype(str),
            pd.to_datetime(projected_frames["adjustment_factors"]["session"]),
        )
    )
    document = finalizer.ReportDocument(
        path=Path("projected-simple7-lifecycle-report.json"),
        content=prepared.evidence_report_bytes,
        value=json.loads(prepared.evidence_report_bytes),
    )
    specs = finalizer.load_official_lifecycle_exception_evidence(
        finalizer.DEFAULT_HINTS
    )
    result = finalizer.prepare_finalization(
        sparse_repository,
        prepared.planned_release,
        "projected-etag",
        document,
        sec_cache=finalizer.DEFAULT_SEC_CACHE,
        official_evidence_specs=specs,
        candidates=candidates,
        hints_path=finalizer.DEFAULT_HINTS,
    )
    coverage = result.coverage_report.manifest_metadata()
    assert all(
        coverage[key] == value
        for key, value in script.EXPECTED_LIFECYCLE_COVERAGE.items()
    )
    assert coverage["candidate_set_sha256"] == (
        script.EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
    )
    assert result.evidence_report_sha256 == sha256_bytes(
        prepared.evidence_report_bytes
    )
    utx = result.frames["lifecycle_resolutions"].loc[
        result.frames["lifecycle_resolutions"]["security_id"]
        .astype(str)
        .eq("US:EODHD:aefd1dd7-529d-5b6a-80e9-65d0e14102a6")
    ]
    assert len(utx) == 1
    assert utx.iloc[0]["resolution"] == "exception"
    assert utx.iloc[0]["exception_code"] == "unsupported_consideration"
    assert script._text(utx.iloc[0]["recheck_after"]) == ""
    hash_columns = (
        "candidate_id",
        "security_id",
        "resolution",
        "event_id",
        "exception_code",
        "exception_reason",
        "reviewed_by",
        "reviewed_at",
        "recheck_after",
        "successor_security_id",
        "successor_symbol",
        "source_hash",
    )
    before = prepared.frames["lifecycle_resolutions"].set_index(
        "security_id", drop=False
    )
    after = result.frames["lifecycle_resolutions"].set_index(
        "security_id", drop=False
    )
    resolution_differences = {
        security_id: {
            column: (script._text(before.loc[security_id, column]), script._text(after.loc[security_id, column]))
            for column in hash_columns
            if script._text(before.loc[security_id, column])
            != script._text(after.loc[security_id, column])
        }
        for security_id in sorted(set(before.index) & set(after.index))
        if any(
            script._text(before.loc[security_id, column])
            != script._text(after.loc[security_id, column])
            for column in hash_columns
        )
    }
    record_property("projected_price_rows", len(projected_frames["daily_price_raw"]))
    record_property("candidate_set_sha256", coverage["candidate_set_sha256"])
    record_property("resolution_set_sha256", coverage["resolution_set_sha256"])
    record_property("evidence_report_sha256", result.evidence_report_sha256)
    assert coverage["resolution_set_sha256"] == (
        script.EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256
    ), json.dumps(resolution_differences, sort_keys=True)
