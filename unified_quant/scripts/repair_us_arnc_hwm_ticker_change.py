#!/usr/bin/env python3
"""Plan or atomically add the exact 2020 ARNC -> HWM ticker transition.

The parent Arconic Inc. security continued as Howmet Aerospace under HWM on
2020-04-01 while a separate child security began trading as ARNC.  The
spin-off, both identities, prices, factors and index membership already exist;
this offline repair adds only the missing same-security ticker_change action.

Plan mode is the default.  There is no network, EODHD or R2 path.  Apply uses
an inherited one-row corporate_actions delta, one writer lock, release/pointer
CAS, a durable rollback journal and bounded post-commit verification.  Every
other dataset version and every pre-existing corporate-action file stays byte
identical.
"""

from __future__ import annotations

import argparse
import base64
import duckdb
import fcntl
import json
import math
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_us_arnc_2020_transition as audit  # noqa: E402
from supertrend_quant.market_store.adjustments import (  # noqa: E402
    CASH_DISTRIBUTION_ACTIONS,
    RATIO_ACTIONS,
)
from supertrend_quant.market_store.cross_validation import (  # noqa: E402
    TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
    dataframe_sha256,
    reviewed_nonterminal_extraction_mismatches,
    reviewed_nonterminal_extraction_sha256,
    reviewed_nonterminal_extractions,
    reviewed_nonterminal_inventory_sha256,
)
from supertrend_quant.market_store.lifecycle import (  # noqa: E402
    build_lifecycle_candidates,
)
from supertrend_quant.market_store.lifecycle_coverage import (  # noqa: E402
    lifecycle_candidate_id,
    validate_lifecycle_coverage,
)
from supertrend_quant.market_store.manifest import (  # noqa: E402
    CurrentPointer,
    DataRelease,
    sha256_bytes,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import (  # noqa: E402
    LocalDatasetRepository,
)
from supertrend_quant.market_store.schemas import dataset_spec  # noqa: E402
from supertrend_quant.market_store.official_lifecycle_evidence import (  # noqa: E402
    include_bound_official_applied_event_candidates,
    load_official_lifecycle_exception_evidence,
)
from supertrend_quant.market_store.storage import (  # noqa: E402
    ConditionalWriteFailed,
    ObjectNotFound,
)
from supertrend_quant.market_store.validation import (  # noqa: E402
    validate_dataset,
    validate_manifest_files,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_POLICY = SCRIPT_DIR.parent / "configs/us_cross_validation.yaml"
DEFAULT_HINTS = SCRIPT_DIR.parent / "configs/us_lifecycle_hints.yaml"
DEFAULT_BACKTEST_SUMMARY = (
    SCRIPT_DIR.parent.parent
    / "results/research/us/backtests/"
    "sp500_triple_supertrend_alone_max_20260715-20260718T230255094849Z/"
    "summary.json"
)
DATASET = "corporate_actions"
OPERATION = "repair_us_arnc_hwm_ticker_change"
TRANSACTION_DIR = "transactions/us-arnc-hwm-ticker"
RECOVERY_DIR = "recovery/us-arnc-hwm-ticker"
TRANSACTION_SCHEMA = "us_arnc_hwm_ticker_transaction/v1"
REQUIRED_DATASETS = (
    "corporate_actions",
    "adjustment_factors",
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "lifecycle_resolutions",
    "source_archive",
    "index_constituent_anchors",
    "index_membership_events",
)
ACTION_SOURCE = "official_identity_repair"
POST_SYMC_OPERATION = "repair_us_symc_nlok_identity"
EXPECTED_ACTION_ROWS_BEFORE = 24_052
EXPECTED_ACTION_ROWS_AFTER = 24_053
EXPECTED_REVIEWED_EXTRACTION_SHA256 = (
    "0121bd4918ff07fbab92be65b4ca12bd5546e83e3804aeb39266b573d2cb0ec5"
)
EXPECTED_BACKTEST_SUMMARY_SHA256 = (
    "df52ba71f8a842f9aa43e67b124bb65692c86f70baeb55c0aa477d124c06cc06"
)
EXPECTED_LIFECYCLE_COVERAGE: Mapping[str, Any] = {
    "coverage_gate_version": 1,
    "selection_rule": "us_terminal_v1",
    "candidate_set_sha256": (
        "32cf8a701a37041584b4a8117064c858d122d3fa50b6f76f19f3e05bd4060c64"
    ),
    "resolution_set_sha256": (
        "150ea3f58b1ccc638955b8411a4b0fd7f0c7efec68876f6991cfbcb2264253c5"
    ),
    "candidate_count": 181,
    "resolution_count": 181,
    "applied_count": 169,
    "exception_count": 12,
    "open_count": 0,
}


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    version_state: Mapping[str, Mapping[str, Any]]
    policy_path: Path
    policy_sha256: str
    action_delta: pd.DataFrame
    planned_action_version: str
    planned_release: DataRelease | None
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


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_policy(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = yaml.safe_load(payload)
    _require(isinstance(value, dict), "ARNC cross-validation policy is invalid.")
    events = value.get("events")
    _require(isinstance(events, dict), "ARNC events policy is missing.")
    reviewed = reviewed_nonterminal_extractions(events)
    expected = audit.proposed_nonterminal_extraction()
    _require(
        reviewed.get(audit.TICKER_CHANGE_EVENT_ID) == expected,
        "ARNC reviewed nonterminal policy row is missing or differs.",
    )
    _require(
        reviewed_nonterminal_inventory_sha256(events)
        == TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
        "ARNC reviewed nonterminal inventory is not code-pinned.",
    )
    return value, sha256_bytes(payload)


class _CandidateRepository:
    """Minimal in-memory view for bounded lifecycle candidate rebuilding."""

    def __init__(self, frames: Mapping[str, pd.DataFrame]):
        self.frames = dict(frames)

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        return self.frames[dataset].copy(deep=False)


def _read_security_subset(
    repository: LocalDatasetRepository,
    dataset: str,
    version: str,
    security_ids: set[str],
) -> pd.DataFrame:
    """Read a finite security scope directly from Parquet with DuckDB pushdown."""

    paths = [str(path) for path in repository.parquet_paths(dataset, version)]
    _require(bool(paths), f"{dataset} Parquet inventory is empty.")
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            "SELECT * FROM read_parquet(?, union_by_name=true) "
            "WHERE security_id = ANY(?)",
            [paths, sorted(security_ids)],
        ).fetchdf()
    finally:
        connection.close()
    spec = dataset_spec(dataset)
    derived_partitions = [
        column
        for column in spec.partition_columns
        if column in frame.columns and column not in spec.required_columns
    ]
    if derived_partitions:
        frame = frame.drop(columns=derived_partitions)
    return frame.drop_duplicates(
        list(spec.primary_key), keep="last"
    ).reset_index(drop=True)


def _read_terminal_price_summary(
    repository: LocalDatasetRepository,
    version: str,
) -> pd.DataFrame:
    """Read one last session per security instead of all daily price rows."""

    paths = [str(path) for path in repository.parquet_paths("daily_price_raw", version)]
    _require(bool(paths), "daily_price_raw Parquet inventory is empty.")
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            "SELECT CAST(security_id AS VARCHAR) AS security_id, "
            "MAX(session) AS session FROM read_parquet(?, union_by_name=true) "
            "GROUP BY security_id",
            [paths],
        ).fetchdf()
    finally:
        connection.close()
    _require(
        not frame.empty and not frame.duplicated("security_id").any(),
        "Lifecycle terminal-session summary changed.",
    )
    return frame.reset_index(drop=True)


def _candidate_values(
    frames: Mapping[str, pd.DataFrame],
    release: DataRelease,
) -> tuple[Any, ...]:
    candidate_repository = _CandidateRepository(frames)
    specs = load_official_lifecycle_exception_evidence(DEFAULT_HINTS)
    return include_bound_official_applied_event_candidates(
        build_lifecycle_candidates(candidate_repository, release=release),
        candidate_repository,
        release,
        specs,
    )


def _lifecycle_projection(
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    actions: pd.DataFrame,
) -> dict[str, Any]:
    context = {
        dataset: frames[dataset]
        for dataset in (
            "security_master",
            "symbol_history",
            "index_constituent_anchors",
            "index_membership_events",
        )
    }
    context["daily_price_raw"] = frames["_terminal_price_summary"]
    context["corporate_actions"] = actions
    candidates = _candidate_values(context, release)
    candidate_frame = pd.DataFrame([asdict(item) for item in candidates])
    if candidate_frame.empty:
        candidate_frame = pd.DataFrame(columns=("security_id", "last_price_date"))
    coverage = validate_lifecycle_coverage(
        candidate_frame,
        frames["lifecycle_resolutions"],
        actions,
        completed_session=release.completed_session,
    )
    coverage.raise_for_errors()
    metadata = coverage.manifest_metadata()
    _require(
        coverage.valid and metadata == dict(EXPECTED_LIFECYCLE_COVERAGE),
        "ARNC/HWM lifecycle coverage is not the exact 181/181 post-SYMC state.",
    )
    return metadata


def _verify_triple_supertrend_preflight() -> dict[str, Any]:
    _require(DEFAULT_BACKTEST_SUMMARY.is_file(), "Pinned ARNC backtest summary is missing.")
    payload = DEFAULT_BACKTEST_SUMMARY.read_bytes()
    _require(
        sha256_bytes(payload) == EXPECTED_BACKTEST_SUMMARY_SHA256,
        "Pinned ARNC backtest summary changed.",
    )
    projection = audit._backtest_projection(DEFAULT_BACKTEST_SUMMARY)
    _require(
        projection.get("summary_present") is True
        and projection.get("event_window_position_count") == 0
        and projection.get("expected_existing_run_equity_effect") == "none",
        "Pinned ARNC Triple Supertrend impact changed.",
    )
    return projection


def _version_state(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    etags: dict[str, str | None] = {}
    state: dict[str, dict[str, Any]] = {}
    for dataset in REQUIRED_DATASETS:
        version = release.dataset_versions.get(dataset, "")
        _require(bool(version), f"Current release lacks {dataset}.")
        pointer, etag = repository.current_pointer(dataset)
        _require(
            pointer is not None and pointer.version == version,
            f"Release/current pointer mismatch: {dataset}.",
        )
        chain_state: list[dict[str, Any]] = []
        for manifest in repository.manifest_chain(dataset, version):
            version_root = (
                repository.root
                / repository.version_prefix(dataset, manifest.version)
            )
            validate_manifest_files(version_root, manifest).raise_for_errors()
            chain_state.append(
                {
                    "version": manifest.version,
                    "manifest_sha256": sha256_bytes(
                        (version_root / "manifest.json").read_bytes()
                    ),
                    "files": [
                        {
                            "path": item.path,
                            "sha256": item.sha256,
                            "size_bytes": item.size_bytes,
                            "row_count": item.row_count,
                        }
                        for item in manifest.files
                    ],
                }
            )
        etags[dataset] = etag
        state[dataset] = {
            "version": version,
            "pointer_etag": etag,
            "pointer_sha256": sha256_bytes(
                repository.objects.get(repository.current_key(dataset)).data
            ),
            "chain": chain_state,
        }
    return etags, state


def _validate_release_manifest_files(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> None:
    """Hash every Parquet file in every current inherited manifest chain."""

    for dataset in REQUIRED_DATASETS:
        version = release.dataset_versions.get(dataset, "")
        _require(bool(version), f"Release lacks protected dataset {dataset}.")
        for manifest in repository.manifest_chain(dataset, version):
            validate_manifest_files(
                repository.root
                / repository.version_prefix(dataset, manifest.version),
                manifest,
            ).raise_for_errors()


def _official_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> tuple[str, dict[str, bool]]:
    sec_payload, _ = audit._archive_payload(
        repository,
        archive,
        source_url=audit.SEC_SOURCE_URL,
        source_hash=audit.SEC_SOURCE_HASH,
    )
    sp_payload, sp_row = audit._archive_payload(
        repository,
        archive,
        source_url=audit.SP_SOURCE_URL,
        source_hash=audit.SP_SOURCE_HASH,
    )
    checks = audit._official_evidence_checks(
        audit._plain_html(sec_payload), audit._plain_html(sp_payload)
    )
    retrieved_at = ""
    matches = archive.loc[
        archive["source_url"].map(_text).eq(audit.SP_SOURCE_URL)
        & archive["source_hash"].map(_text).str.lower().eq(audit.SP_SOURCE_HASH)
    ]
    if len(matches) == 1:
        retrieved_at = _text(matches.iloc[0].get("retrieved_at"))
    _require(bool(retrieved_at), "ARNC S&P evidence retrieved_at is missing.")
    _require(
        sp_row["payload_sha256_verified"] is True and all(checks.values()),
        "ARNC official evidence is not exact.",
    )
    return retrieved_at, checks


def expected_action(retrieved_at: str) -> dict[str, Any]:
    row = dict(audit.proposed_action())
    row["retrieved_at"] = retrieved_at
    row["source"] = ACTION_SOURCE
    return row


def _action_state(
    actions: pd.DataFrame,
    expected: Mapping[str, Any],
) -> str:
    event_rows = actions.loc[
        actions["event_id"].map(_text).eq(audit.TICKER_CHANGE_EVENT_ID)
    ]
    key_rows = actions.loc[
        actions["security_id"].map(_text).eq(audit.PARENT_SECURITY_ID)
        & actions["action_type"].map(_text).str.lower().eq("ticker_change")
        & actions["effective_date"].map(_date).eq(audit.EFFECTIVE_DATE)
    ]
    if event_rows.empty and key_rows.empty:
        return "missing"
    _require(
        len(event_rows) == 1 and len(key_rows) == 1,
        "ARNC ticker-change action is duplicated or key-conflicting.",
    )
    row = event_rows.iloc[0].to_dict()
    _require(
        reviewed_nonterminal_extraction_mismatches(
            row, audit.proposed_nonterminal_extraction()
        )
        == (),
        "Installed ARNC ticker-change reviewed fields differ.",
    )
    string_fields = (
        "event_id",
        "security_id",
        "action_type",
        "effective_date",
        "ex_date",
        "announcement_date",
        "record_date",
        "payment_date",
        "currency",
        "new_security_id",
        "new_symbol",
        "source_url",
        "source_kind",
        "source",
        "retrieved_at",
        "source_hash",
    )
    mismatches = [
        field
        for field in string_fields
        if _text(row.get(field)) != _text(expected.get(field))
    ]
    if not _is_null(row.get("cash_amount")):
        mismatches.append("cash_amount")
    if not _is_null(row.get("ratio")):
        mismatches.append("ratio")
    if not _is_null(row.get("metadata")):
        mismatches.append("metadata")
    if row.get("official") is not True:
        mismatches.append("official")
    _require(not mismatches, "Installed ARNC ticker action differs: " + ", ".join(mismatches))
    return "exact"


def _verify_scope(
    repository: LocalDatasetRepository,
    frames: Mapping[str, pd.DataFrame],
) -> tuple[str, dict[str, Any]]:
    master = frames["security_master"]
    history = frames["symbol_history"]
    prices = frames["daily_price_raw"]
    actions = frames["corporate_actions"]
    resolutions = frames["lifecycle_resolutions"]

    parent = master.loc[
        master["security_id"].map(_text).eq(audit.PARENT_SECURITY_ID)
    ]
    child = master.loc[
        master["security_id"].map(_text).eq(audit.CHILD_SECURITY_ID)
    ]
    _require(
        len(parent) == 1
        and len(child) == 1
        and _text(parent.iloc[0].get("primary_symbol")).upper()
        == audit.NEW_PARENT_SYMBOL
        and _text(child.iloc[0].get("primary_symbol")).upper()
        == audit.OLD_SYMBOL,
        "ARNC/HWM master identity changed.",
    )
    parent_intervals = {
        (
            _text(row.get("symbol")).upper(),
            _date(row.get("effective_from")),
            _date(row.get("effective_to")),
        )
        for row in history.loc[
            history["security_id"].map(_text).eq(audit.PARENT_SECURITY_ID)
        ].to_dict(orient="records")
    }
    _require(
        {
            (audit.OLD_SYMBOL, "2016-11-01", audit.LAST_OLD_SYMBOL_SESSION),
            (audit.NEW_PARENT_SYMBOL, audit.EFFECTIVE_DATE, ""),
        }
        <= parent_intervals,
        "ARNC/HWM symbol boundary changed.",
    )
    parent_sessions = {
        _date(value)
        for value in prices.loc[
            prices["security_id"].map(_text).eq(audit.PARENT_SECURITY_ID),
            "session",
        ]
    }
    child_sessions = sorted(
        _date(value)
        for value in prices.loc[
            prices["security_id"].map(_text).eq(audit.CHILD_SECURITY_ID),
            "session",
        ]
    )
    _require(
        audit.LAST_OLD_SYMBOL_SESSION in parent_sessions
        and audit.EFFECTIVE_DATE in parent_sessions
        and child_sessions
        and child_sessions[0] == audit.EFFECTIVE_DATE,
        "ARNC/HWM price boundary changed.",
    )
    spinoff = actions.loc[
        actions["event_id"].map(_text).eq(audit.SPINOFF_EVENT_ID)
    ]
    _require(len(spinoff) == 1, "ARNC 2020 spin-off action is missing.")
    spin = spinoff.iloc[0]
    _require(
        _text(spin.get("security_id")) == audit.PARENT_SECURITY_ID
        and _text(spin.get("action_type")).lower() == "spinoff"
        and _date(spin.get("effective_date")) == audit.EFFECTIVE_DATE
        and _text(spin.get("new_security_id")) == audit.CHILD_SECURITY_ID
        and _text(spin.get("new_symbol")).upper() == audit.OLD_SYMBOL
        and math.isclose(float(spin.get("ratio")), 0.25, abs_tol=1e-12)
        and _text(spin.get("source_hash")).lower() == audit.SEC_SOURCE_HASH,
        "ARNC 2020 spin-off economics changed.",
    )
    retrieved_at, claims = _official_evidence(
        repository, frames["source_archive"]
    )
    expected = expected_action(retrieved_at)
    state = _action_state(actions, expected)
    candidate_id = lifecycle_candidate_id(
        audit.PARENT_SECURITY_ID, audit.LAST_OLD_SYMBOL_SESSION
    )
    forbidden = resolutions.loc[
        resolutions["event_id"].map(_text).eq(audit.TICKER_CHANGE_EVENT_ID)
        | resolutions["candidate_id"].map(_text).eq(candidate_id)
    ]
    _require(
        forbidden.empty,
        "ARNC same-SID continuation must not have a terminal lifecycle resolution.",
    )
    anchors = frames["index_constituent_anchors"]
    membership = frames["index_membership_events"]
    anchored = anchors.loc[
        anchors["security_id"].map(_text).eq(audit.PARENT_SECURITY_ID)
        & anchors["index_id"].map(_text).str.lower().eq("sp500")
        & anchors["anchor_date"].map(_date).le(audit.LAST_OLD_SYMBOL_SESSION)
    ]
    events = membership.loc[
        membership["security_id"].map(_text).eq(audit.PARENT_SECURITY_ID)
        & membership["index_id"].map(_text).str.lower().eq("sp500")
        & membership["effective_date"].map(_date).le(audit.EFFECTIVE_DATE)
    ].sort_values("effective_date")
    member = not anchored.empty
    if not events.empty:
        member = _text(events.iloc[-1].get("operation")).upper() == "ADD"
    _require(member, "ARNC/HWM parent was not an S&P 500 member at transition.")
    return state, {
        "expected_action": expected,
        "official_claims": claims,
        "forbidden_lifecycle_candidate_id": candidate_id,
        "sp500_member_at_transition": True,
    }


def _load_scope_frames(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, pd.DataFrame]:
    """Load small inputs, a two-SID price slice and terminal price summaries.

    Neither multi-million-row daily prices nor adjustment factors are ever
    materialized.  DuckDB scans the Parquet files and returns only the exact
    repair scope plus one terminal session per security for lifecycle replay.
    """

    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
        if dataset not in {"daily_price_raw", "adjustment_factors"}
    }
    frames["daily_price_raw"] = _read_security_subset(
        repository,
        "daily_price_raw",
        release.dataset_versions["daily_price_raw"],
        {audit.PARENT_SECURITY_ID, audit.CHILD_SECURITY_ID},
    )
    frames["_terminal_price_summary"] = _read_terminal_price_summary(
        repository,
        release.dataset_versions["daily_price_raw"],
    )
    return frames


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> PreparedRepair:
    policy, policy_sha256 = _load_policy(policy_path)
    release, release_etag = repository.current_release()
    _require(release is not None, "A current local release is required.")
    pointer_etags, version_state = _version_state(repository, release)
    lifecycle_manifest = repository.manifest_for_version(
        "lifecycle_resolutions",
        release.dataset_versions["lifecycle_resolutions"],
    )
    lifecycle_manifest_projection = {
        key: lifecycle_manifest.metadata.get(key)
        for key in EXPECTED_LIFECYCLE_COVERAGE
    }
    _require(
        lifecycle_manifest.metadata.get("operation") == POST_SYMC_OPERATION
        and lifecycle_manifest_projection == dict(EXPECTED_LIFECYCLE_COVERAGE),
        "ARNC/HWM repair requires the exact reviewed post-SYMC lifecycle parent.",
    )
    frames = _load_scope_frames(repository, release)
    state, proof = _verify_scope(repository, frames)
    actions = frames[DATASET]
    existing_hash = dataframe_sha256(actions, ("event_id",))
    factor_hash = sha256_bytes(
        json.dumps(
            version_state["adjustment_factors"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    expected = proof["expected_action"]
    if state == "missing":
        delta = pd.DataFrame([expected]).reindex(columns=actions.columns)
        candidate = pd.concat([actions, delta], ignore_index=True)
    else:
        delta = actions.iloc[:0].copy()
        candidate = actions.copy()
    base_actions = candidate.loc[
        ~candidate["event_id"].map(_text).eq(audit.TICKER_CHANGE_EVENT_ID)
    ].copy()
    base_hash = dataframe_sha256(base_actions, ("event_id",))
    _require(
        len(actions)
        == (
            EXPECTED_ACTION_ROWS_BEFORE
            if state == "missing"
            else EXPECTED_ACTION_ROWS_AFTER
        )
        and len(base_actions) == EXPECTED_ACTION_ROWS_BEFORE
        and len(candidate) == EXPECTED_ACTION_ROWS_AFTER
        and len(candidate) == len(actions) + int(state == "missing"),
        "ARNC repair action inventory is not the exact 24052 -> 24053 delta.",
    )
    if state == "missing":
        _require(
            base_hash == existing_hash,
            "ARNC repair changed pre-existing corporate-action bytes.",
        )
    else:
        _require(
            dataframe_sha256(candidate, ("event_id",)) == existing_hash,
            "Already-repaired ARNC action inventory changed during replay.",
        )
    _require(
        _text(expected["action_type"]) not in RATIO_ACTIONS
        and _text(expected["action_type"]) not in CASH_DISTRIBUTION_ACTIONS
        and _is_null(expected["ratio"])
        and _is_null(expected["cash_amount"]),
        "ARNC action unexpectedly has adjustment-factor economics.",
    )
    reviewed_extraction_sha256 = reviewed_nonterminal_extraction_sha256(
        audit.proposed_nonterminal_extraction()
    )
    _require(
        audit.TICKER_CHANGE_EVENT_ID
        == "fb3d264732079815004e26780f47e9c816133970ad35ab903054fa5c97406a48"
        and reviewed_extraction_sha256 == EXPECTED_REVIEWED_EXTRACTION_SHA256
        and _text(expected.get("source_hash")).lower() == audit.SP_SOURCE_HASH,
        "ARNC event/review/source preflight hashes changed.",
    )
    validate_dataset(
        DATASET,
        candidate,
        completed_session=release.completed_session,
        incomplete_action_policy="block",
    ).raise_for_errors()
    base_coverage = _lifecycle_projection(release, frames, base_actions)
    candidate_coverage = _lifecycle_projection(release, frames, candidate)
    _require(
        base_coverage == candidate_coverage,
        "ARNC nonterminal action changed lifecycle coverage.",
    )
    triple_supertrend = _verify_triple_supertrend_preflight()
    candidate_hash = dataframe_sha256(candidate, ("event_id",))
    if state == "missing":
        planned_action_version = (
            "arnc-hwm-ticker-"
            f"{release.completed_session.replace('-', '')}-"
            f"{uuid.uuid4().hex}-{DATASET}"
        )
        output_versions = dict(release.dataset_versions)
        output_versions[DATASET] = planned_action_version
        planned_release = DataRelease.create(
            release.completed_session,
            output_versions,
            quality=release.quality,
            warnings=release.warnings,
        )
    else:
        planned_action_version = ""
        planned_release = None
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        version_state=version_state,
        policy_path=policy_path,
        policy_sha256=policy_sha256,
        action_delta=delta,
        planned_action_version=planned_action_version,
        planned_release=planned_release,
        summary={
            "status": (
                "validated_offline_plan" if state == "missing" else "already_repaired"
            ),
            "base_release_version": release.version,
            "event_id": audit.TICKER_CHANGE_EVENT_ID,
            "security_id": audit.PARENT_SECURITY_ID,
            "old_symbol": audit.OLD_SYMBOL,
            "new_symbol": audit.NEW_PARENT_SYMBOL,
            "effective_date": audit.EFFECTIVE_DATE,
            "corporate_action_rows_added": int(state == "missing"),
            "corporate_action_delta_only": True,
            "corporate_action_rows_before": EXPECTED_ACTION_ROWS_BEFORE,
            "corporate_action_rows_after": EXPECTED_ACTION_ROWS_AFTER,
            "preexisting_action_inventory_sha256": base_hash,
            "candidate_action_inventory_sha256": candidate_hash,
            "adjustment_factor_storage_inventory_sha256": factor_hash,
            "adjustment_factor_economic_rows_changed": 0,
            "factor_dataset_version_preserved": release.dataset_versions[
                "adjustment_factors"
            ],
            "protected_dataset_versions": {
                dataset: release.dataset_versions[dataset]
                for dataset in REQUIRED_DATASETS
                if dataset != DATASET
            },
            "lifecycle_resolution_change": "none",
            "lifecycle_coverage_before": base_coverage,
            "lifecycle_coverage_after": candidate_coverage,
            "terminal_resolution_forbidden": True,
            "forbidden_lifecycle_candidate_id": proof[
                "forbidden_lifecycle_candidate_id"
            ],
            "security_master_change": "none",
            "symbol_history_change": "none",
            "daily_price_raw_change": "none",
            "full_daily_price_raw_materialized": False,
            "adjustment_factors_materialized": False,
            "plan_materialization": (
                "two_sid_duckdb_price_scope_plus_terminal_session_summary"
            ),
            "existing_spinoff_change": "none",
            "index_membership_change": "none",
            "cross_validation_policy_sha256": policy_sha256,
            "reviewed_nonterminal_inventory_sha256": (
                reviewed_nonterminal_inventory_sha256(policy["events"])
            ),
            "reviewed_nonterminal_extraction_sha256": reviewed_extraction_sha256,
            "official_source_url": audit.SP_SOURCE_URL,
            "official_source_hash": audit.SP_SOURCE_HASH,
            "official_claims": proof["official_claims"],
            "triple_supertrend_preflight": {
                "summary_sha256": triple_supertrend["summary_sha256"],
                "event_window_position_count": triple_supertrend[
                    "event_window_position_count"
                ],
                "expected_existing_run_equity_effect": triple_supertrend[
                    "expected_existing_run_equity_effect"
                ],
                "price_or_factor_rows_changed": 0,
            },
            "planned_action_version": planned_action_version,
            "planned_release_version": (
                planned_release.version if planned_release is not None else ""
            ),
            "network_accessed": False,
            "http_attempts": 0,
            "eodhd_calls": 0,
            "r2_accessed": False,
            "writes_performed": False,
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
            raise RuntimeError("Unresolved ARNC/HWM recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted ARNC/HWM transaction blocks writes: {journal}."
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
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    release, release_etag = repository.current_release()
    _require(
        release is not None
        and release.to_bytes() == prepared.release.to_bytes()
        and release_etag == prepared.release_etag,
        "Current release changed after ARNC/HWM planning.",
    )
    _require(
        prepared.policy_path.is_file()
        and sha256_bytes(prepared.policy_path.read_bytes())
        == prepared.policy_sha256,
        "Cross-validation policy changed after ARNC/HWM planning.",
    )
    _require(
        set(prepared.pointer_etags) == set(REQUIRED_DATASETS)
        and set(prepared.version_state) == set(REQUIRED_DATASETS),
        "Prepared ARNC/HWM input inventory is incomplete.",
    )
    pointer_etags, version_state = _version_state(repository, release)
    _require(
        pointer_etags == dict(prepared.pointer_etags),
        "ARNC/HWM pointer inventory changed after planning.",
    )
    _require(
        version_state == dict(prepared.version_state),
        "ARNC/HWM immutable manifest inventory changed after planning.",
    )
    archive = repository.read_frame(
        "source_archive", prepared.release.dataset_versions["source_archive"]
    )
    _official_evidence(repository, archive)
    _validate_release_manifest_files(repository, prepared.release)
    _assert_immutable_version_state(repository, prepared)


def _assert_immutable_version_state(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    for dataset, expected in prepared.version_state.items():
        version = str(expected["version"])
        actual_chain: list[dict[str, Any]] = []
        for manifest in repository.manifest_chain(dataset, version):
            version_root = (
                repository.root
                / repository.version_prefix(dataset, manifest.version)
            )
            validate_manifest_files(version_root, manifest).raise_for_errors()
            actual_chain.append(
                {
                    "version": manifest.version,
                    "manifest_sha256": sha256_bytes(
                        (version_root / "manifest.json").read_bytes()
                    ),
                    "files": [
                        {
                            "path": item.path,
                            "sha256": item.sha256,
                            "size_bytes": item.size_bytes,
                            "row_count": item.row_count,
                        }
                        for item in manifest.files
                    ],
                }
            )
        _require(
            actual_chain == expected["chain"],
            f"Immutable {dataset} inherited manifest chain changed.",
        )
        if dataset != DATASET:
            pointer = repository.objects.get(repository.current_key(dataset))
            _require(
                sha256_bytes(pointer.data) == expected["pointer_sha256"],
                f"Protected {dataset} pointer bytes changed.",
            )


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_release_bytes: bytes,
    owned_pointer_bytes: Mapping[str, bytes],
) -> tuple[str, ...]:
    """Restore an exactly owned publication after an all-or-none preflight."""

    # A foreign byte in either the release or any protected pointer makes the
    # entire rollback non-owning.  Inspect everything before the first put.
    try:
        current_release = repository.objects.get("releases/current.json")
        if current_release.data not in {old_release_bytes, planned_release_bytes}:
            observed = DataRelease.from_bytes(current_release.data)
            raise RuntimeError(
                f"unexpected release during ARNC/HWM rollback: {observed.version}"
            )
        current_pointers: dict[str, Any] = {}
        for dataset in REQUIRED_DATASETS:
            current = repository.objects.get(repository.current_key(dataset))
            old = old_pointer_bytes[dataset]
            owned = owned_pointer_bytes.get(dataset)
            if current.data != old and (owned is None or current.data != owned):
                observed = CurrentPointer.from_bytes(current.data)
                raise RuntimeError(
                    "unexpected pointer during ARNC/HWM rollback: "
                    f"{dataset}/{observed.version}"
                )
            current_pointers[dataset] = current
    except Exception as exc:
        return (f"rollback preflight: {type(exc).__name__}: {exc}",)

    errors: list[str] = []
    try:
        if current_release.data != old_release_bytes:
            repository.objects.put(
                "releases/current.json",
                old_release_bytes,
                if_match=current_release.etag,
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in REQUIRED_DATASETS:
        key = repository.current_key(dataset)
        try:
            current = current_pointers[dataset]
            if current.data != old_pointer_bytes[dataset]:
                repository.objects.put(
                    key,
                    old_pointer_bytes[dataset],
                    if_match=current.etag,
                )
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _capture_owned_pointer(
    repository: LocalDatasetRepository,
    *,
    version: str,
    manifest_bytes: bytes,
) -> bytes:
    value = repository.objects.get(repository.current_key(DATASET))
    pointer = CurrentPointer.from_bytes(value.data)
    expected_path = f"{repository.version_prefix(DATASET, version)}/manifest.json"
    _require(
        pointer.dataset == DATASET
        and pointer.version == version
        and pointer.manifest_path == expected_path
        and pointer.manifest_sha256 == sha256_bytes(manifest_bytes),
        "Written ARNC/HWM corporate_actions pointer is not exact.",
    )
    return value.data


def _commit_exact_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
    *,
    expected_etag: str | None,
) -> DataRelease:
    payload = release.to_bytes()
    immutable_key = f"releases/{release.version}.json"
    try:
        repository.objects.put(immutable_key, payload, if_none_match=True)
    except ConditionalWriteFailed:
        try:
            existing = repository.objects.get(immutable_key)
        except ObjectNotFound as exc:  # pragma: no cover - race guard
            raise RuntimeError("Prepared ARNC/HWM immutable release conflicted.") from exc
        _require(
            existing.data == payload,
            "Prepared ARNC/HWM immutable release version conflicts.",
        )
    repository.objects.put(
        "releases/current.json",
        payload,
        if_match=expected_etag,
        if_none_match=expected_etag is None,
    )
    return release


def _verify_written_state(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    release: DataRelease,
    *,
    manifest_bytes: bytes,
) -> None:
    """Replay exact ARNC gates without loading either full heavy table."""

    manifest = repository.current_manifest(DATASET)
    _require(
        manifest is not None
        and manifest.version == prepared.planned_action_version
        and manifest.to_bytes() == manifest_bytes
        and manifest.parent_version == prepared.release.dataset_versions[DATASET]
        and manifest.metadata.get("inherits_parent") is True,
        "Written ARNC/HWM corporate_actions manifest changed.",
    )
    _validate_release_manifest_files(repository, release)
    _assert_immutable_version_state(repository, prepared)
    frames = _load_scope_frames(repository, release)
    state, _proof = _verify_scope(repository, frames)
    _require(state == "exact", "Written ARNC/HWM action is not exact.")
    actions = frames[DATASET]
    base = actions.loc[
        ~actions["event_id"].map(_text).eq(audit.TICKER_CHANGE_EVENT_ID)
    ].copy()
    _require(
        len(base) == EXPECTED_ACTION_ROWS_BEFORE
        and len(actions) == EXPECTED_ACTION_ROWS_AFTER
        and dataframe_sha256(base, ("event_id",))
        == prepared.summary["preexisting_action_inventory_sha256"]
        and dataframe_sha256(actions, ("event_id",))
        == prepared.summary["candidate_action_inventory_sha256"],
        "Written ARNC/HWM action inventory changed.",
    )
    _require(
        _lifecycle_projection(release, frames, actions)
        == prepared.summary["lifecycle_coverage_after"],
        "Written ARNC/HWM lifecycle projection changed.",
    )


def _assert_committed_publication(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    committed: DataRelease,
    *,
    old_pointer_bytes: Mapping[str, bytes],
    owned_action_pointer_bytes: bytes,
) -> None:
    current, _etag = repository.current_release()
    _require(
        current is not None and current.to_bytes() == committed.to_bytes(),
        "Committed ARNC/HWM release is not current.",
    )
    # Stream immutable file hashes once more after publication.  This is not a
    # plan replay and never materializes either heavy Parquet dataset.
    _validate_release_manifest_files(repository, committed)
    _assert_immutable_version_state(repository, prepared)
    for dataset in REQUIRED_DATASETS:
        observed = repository.objects.get(repository.current_key(dataset))
        expected = (
            owned_action_pointer_bytes
            if dataset == DATASET
            else old_pointer_bytes[dataset]
        )
        _require(
            observed.data == expected,
            f"Committed ARNC/HWM pointer bytes changed: {dataset}.",
        )


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    inject_failure = inject_failure or (lambda _stage: None)
    with _exclusive_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        if prepared.summary["status"] == "already_repaired":
            current_frames = _load_scope_frames(repository, prepared.release)
            current_state, _ = _verify_scope(repository, current_frames)
            actions = current_frames[DATASET]
            _require(
                current_state == "exact"
                and prepared.action_delta.empty
                and len(actions) == EXPECTED_ACTION_ROWS_AFTER
                and dataframe_sha256(actions, ("event_id",))
                == prepared.summary["candidate_action_inventory_sha256"]
                and _lifecycle_projection(
                    prepared.release, current_frames, actions
                )
                == prepared.summary["lifecycle_coverage_after"],
                "Already-repaired ARNC/HWM no-op state is no longer exact.",
            )
            return {
                **prepared.summary,
                "mode": "apply",
                "writes_performed": False,
            }
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in REQUIRED_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            _require(
                pointer.version == prepared.release.dataset_versions[dataset]
                and value.etag == prepared.pointer_etags[dataset]
                and sha256_bytes(value.data)
                == prepared.version_state[dataset]["pointer_sha256"],
                f"ARNC/HWM pointer changed before apply: {dataset}.",
            )
            old_pointers[dataset] = value.data
        planned_release = prepared.planned_release
        _require(
            planned_release is not None
            and bool(prepared.planned_action_version)
            and planned_release.dataset_versions
            == {
                **dict(prepared.release.dataset_versions),
                DATASET: prepared.planned_action_version,
            }
            and prepared.summary.get("planned_action_version")
            == prepared.planned_action_version
            and prepared.summary.get("planned_release_version")
            == planned_release.version
            and prepared.summary.get("corporate_action_rows_before")
            == EXPECTED_ACTION_ROWS_BEFORE
            and prepared.summary.get("corporate_action_rows_after")
            == EXPECTED_ACTION_ROWS_AFTER
            and prepared.summary.get("plan_materialization")
            == "two_sid_duckdb_price_scope_plus_terminal_session_summary"
            and prepared.summary.get("full_daily_price_raw_materialized") is False
            and prepared.summary.get("adjustment_factors_materialized") is False,
            "Prepared ARNC/HWM transaction contract is incomplete.",
        )
        transaction_id = uuid.uuid4().hex
        journal_path = (
            repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": TRANSACTION_SCHEMA,
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value).decode("ascii")
                for dataset, value in old_pointers.items()
            },
            "planned_corporate_actions_version": prepared.planned_action_version,
            "planned_release_version": planned_release.version,
            "planned_release_sha256": sha256_bytes(planned_release.to_bytes()),
            "event_id": audit.TICKER_CHANGE_EVENT_ID,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        owned_action_pointer_bytes = b""
        try:
            inject_failure("after_journal")
            current_manifest = repository.manifest_for_version(
                DATASET, prepared.release.dataset_versions[DATASET]
            )
            metadata = dict(current_manifest.metadata)
            metadata.update(
                {
                    "operation": OPERATION,
                    "arnc_hwm_event_id": audit.TICKER_CHANGE_EVENT_ID,
                    "official_source_url": audit.SP_SOURCE_URL,
                    "official_source_hash": audit.SP_SOURCE_HASH,
                    "corporate_action_rows_added": 1,
                    "preexisting_action_parent_version": (
                        prepared.release.dataset_versions[DATASET]
                    ),
                    "factor_dataset_version_preserved": (
                        prepared.release.dataset_versions["adjustment_factors"]
                    ),
                    "output_release_version": planned_release.version,
                    "output_action_inventory_sha256": prepared.summary[
                        "candidate_action_inventory_sha256"
                    ],
                    "lifecycle_coverage": prepared.summary[
                        "lifecycle_coverage_after"
                    ],
                    "full_daily_price_raw_materialized": False,
                    "adjustment_factors_materialized": False,
                    "network_accessed": False,
                    "eodhd_calls": 0,
                    "r2_accessed": False,
                }
            )
            result = repository.append_frame(
                DATASET,
                prepared.action_delta,
                completed_session=prepared.release.completed_session,
                incomplete_action_policy="block",
                metadata=metadata,
                expected_pointer_etag=prepared.pointer_etags[DATASET],
                version=prepared.planned_action_version,
            )
            _require(
                not result.conflict,
                f"corporate_actions write conflicted: {result.conflict_path}",
            )
            manifest_bytes = result.manifest.to_bytes()
            _require(
                result.manifest.version == prepared.planned_action_version,
                "Written ARNC/HWM action version changed.",
            )
            owned_action_pointer_bytes = _capture_owned_pointer(
                repository,
                version=prepared.planned_action_version,
                manifest_bytes=manifest_bytes,
            )
            inject_failure("after_action_write")
            _verify_written_state(
                repository,
                prepared,
                planned_release,
                manifest_bytes=manifest_bytes,
            )
            committed = _commit_exact_release(
                repository,
                planned_release,
                expected_etag=prepared.release_etag,
            )
            inject_failure("after_release_commit")
            _assert_committed_publication(
                repository,
                prepared,
                committed,
                old_pointer_bytes=old_pointers,
                owned_action_pointer_bytes=owned_action_pointer_bytes,
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
                **prepared.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "new_corporate_actions_version": result.manifest.version,
                "transaction_id": transaction_id,
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_release_bytes=planned_release.to_bytes(),
                owned_pointer_bytes=(
                    {DATASET: owned_action_pointer_bytes}
                    if owned_action_pointer_bytes
                    else {}
                ),
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
                    "ARNC/HWM rollback was incomplete; recovery marker blocks "
                    f"writes: {recovery}; errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository, policy_path=args.policy)
    if args.apply:
        result = apply_repair(repository, prepared)
    else:
        result = {**prepared.summary, "mode": "plan", "writes_performed": False}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
