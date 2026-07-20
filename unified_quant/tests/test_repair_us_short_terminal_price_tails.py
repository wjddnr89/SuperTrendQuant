from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
    ManifestFile,
)
from supertrend_quant.market_store.repository import (
    DatasetWriteResult,
    LocalDatasetRepository,
)
from supertrend_quant.market_store.storage import LocalObjectStore
from supertrend_quant.market_store.validation import (
    ValidationIssue,
    ValidationReport,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_short_terminal_price_tails.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_short_terminal_price_tails", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"
TARGET_IDS = {case.security_id for case in script.CASES}


def _control_hashes(repository: LocalDatasetRepository) -> dict[str, str]:
    keys = ["releases/current.json"] + [
        repository.current_key(dataset) for dataset in script.REQUIRED_DATASETS
    ]
    return {
        key: hashlib.sha256(repository.objects.get(key).data).hexdigest()
        for key in keys
    }


@pytest.fixture(scope="module")
def actual_plan() -> dict:
    if not (REPOSITORY_ROOT / "releases/current.json").is_file():
        pytest.skip("Actual current-release cache is unavailable.")
    repository = LocalDatasetRepository(REPOSITORY_ROOT)
    release, _ = repository.current_release()
    if release is None or not set(script.REQUIRED_DATASETS).issubset(
        release.dataset_versions
    ):
        pytest.skip("Actual release lacks short-tail datasets.")
    before = _control_hashes(repository)
    prepared = script.prepare_repair(repository)
    after = _control_hashes(repository)
    if prepared.summary["status"] == "already_repaired":
        pytest.skip("Actual release is already short-tail repaired.")
    current = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in script.REQUIRED_DATASETS
    }
    evidence = {
        case.symbol: script._verify_case_evidence(
            repository, current["source_archive"], case
        )
        for case in script.CASES
    }
    return {
        "repository": repository,
        "release": release,
        "before": before,
        "after": after,
        "prepared": prepared,
        "current": current,
        "evidence": evidence,
    }


def test_actual_plan_is_read_only_offline_and_exact(actual_plan: dict) -> None:
    prepared = actual_plan["prepared"]
    summary = prepared.summary
    assert actual_plan["before"] == actual_plan["after"]
    assert summary["status"] == "validated_offline_plan"
    assert summary["target_count"] == 6
    assert summary["removed_daily_price_rows"] == 13
    assert summary["removed_adjustment_factor_rows"] == 13
    assert summary["adjustment_factor_economic_rows_changed"] == 0
    assert summary["network_accessed"] is False
    assert summary["eodhd_calls"] == 0
    assert summary["r2_accessed"] is False
    assert summary["source_archive_immutable"] is True
    assert summary["index_membership_events_unchanged"] is True
    assert summary["index_constituent_anchors_unchanged"] is True
    assert summary["target_terminal_issues_after"] == 0
    assert summary["non_target_terminal_issues_unchanged"] is True
    assert summary["snapshot_identity_gap"] == script.EXPECTED_SNAPSHOT_IDENTITY_GAP


def test_exact_terminal_sessions_and_market_dates_are_planned(actual_plan: dict) -> None:
    prepared = actual_plan["prepared"]
    prices = prepared.frames["daily_price_raw"].copy()
    actions = prepared.frames["corporate_actions"].copy()
    resolutions = prepared.frames["lifecycle_resolutions"].copy()
    master = prepared.frames["security_master"].copy()
    history = prepared.frames["symbol_history"].copy()
    price_sessions = pd.to_datetime(prices["session"]).dt.date.astype(str)

    for case in script.CASES:
        target = prices["security_id"].astype(str).eq(case.security_id)
        assert price_sessions.loc[target].max() == case.last_real_session
        assert not price_sessions.loc[target].isin(case.removed_sessions).any()

        action = actions.loc[
            actions["security_id"].astype(str).eq(case.security_id)
            & actions["action_type"].astype(str).eq(case.action_type)
        ].iloc[0]
        assert script._date(action["effective_date"]) == case.market_transition_session
        assert script._date(action["ex_date"]) == case.market_transition_session
        assert str(action["event_id"]) == case.new_event_id
        # The legal date remains independently represented by announcement_date.
        assert script._date(action["announcement_date"]) == case.announcement_date

        resolution = resolutions.loc[
            resolutions["security_id"].astype(str).eq(case.security_id)
        ].iloc[0]
        assert str(resolution["candidate_id"]) == case.new_candidate_id
        assert script._date(resolution["last_price_date"]) == case.last_real_session
        assert str(resolution["event_id"]) == case.new_event_id

        master_row = master.loc[
            master["security_id"].astype(str).eq(case.security_id)
        ].iloc[0]
        history_row = history.loc[
            history["security_id"].astype(str).eq(case.security_id)
            & history["symbol"].astype(str).eq(case.symbol)
        ].iloc[0]
        assert script._date(master_row["active_to"]) == case.last_real_session
        assert script._date(history_row["effective_to"]) == case.last_real_session


def test_only_exact_target_rows_and_factor_provenance_change(actual_plan: dict) -> None:
    current = actual_plan["current"]
    prepared = actual_plan["prepared"]
    for dataset in (
        "corporate_actions",
        "daily_price_raw",
        "lifecycle_resolutions",
        "security_master",
        "symbol_history",
    ):
        old = current[dataset]
        new = prepared.frames[dataset]
        old_non_target = old.loc[
            ~old["security_id"].astype(str).isin(TARGET_IDS)
        ].reset_index(drop=True)
        new_non_target = new.loc[
            ~new["security_id"].astype(str).isin(TARGET_IDS)
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(old_non_target, new_non_target)

    old_factors = current["adjustment_factors"]
    new_factors = prepared.frames["adjustment_factors"]
    new_keys = set(
        zip(
            new_factors["security_id"].astype(str),
            pd.to_datetime(new_factors["session"]).dt.normalize(),
        )
    )
    retained = [
        (str(row.security_id), pd.Timestamp(row.session).normalize()) in new_keys
        for row in old_factors.itertuples(index=False)
    ]
    assert script._factor_economics_equal(old_factors.loc[retained], new_factors) == 0
    lineage = prepared.summary["factor_source_version"]
    assert set(new_factors["source_version"].astype(str)) == {lineage}
    assert set(new_factors["source_hash"].astype(str)) == {lineage}


def test_in_memory_replay_is_idempotent_and_cross_dataset_complete(
    actual_plan: dict,
) -> None:
    prepared = actual_plan["prepared"]
    evidence = actual_plan["evidence"]
    frames = prepared.frames
    for case in script.CASES:
        state = script._case_state(
            symbol=case.symbol,
            price_state=script._price_state(
                frames["daily_price_raw"], case, evidence[case.symbol]["source"]
            ),
            master_state=script._identity_state(
                frames["security_master"], case, history=False
            ),
            history_state=script._identity_state(
                frames["symbol_history"], case, history=True
            ),
            action_state=script._action_state(frames["corporate_actions"], case),
            resolution_state=script._resolution_state(
                frames["lifecycle_resolutions"], case
            ),
        )
        assert state == "repaired"

    factors, removed = script._prepare_factors(
        frames["adjustment_factors"],
        frames["daily_price_raw"],
        frames["daily_price_raw"],
        frames["corporate_actions"],
        source_version=prepared.summary["factor_source_version"],
        repaired_state=True,
    )
    assert removed == 0
    pd.testing.assert_frame_equal(factors, frames["adjustment_factors"])


def test_partial_or_mutated_price_tail_fails_closed(actual_plan: dict) -> None:
    current = actual_plan["current"]["daily_price_raw"]
    case = next(value for value in script.CASES if value.symbol == "FLIR")
    records = actual_plan["evidence"][case.symbol]["source"]
    target = current.loc[
        current["security_id"].astype(str).eq(case.security_id)
    ].copy()
    sessions = pd.to_datetime(target["session"]).dt.date.astype(str)
    partial = target.loc[~sessions.eq(case.removed_sessions[0])]
    with pytest.raises(ValueError, match="neither exact raw nor exact repaired"):
        script._price_state(partial, case, records)

    mutated = target.copy()
    row = pd.to_datetime(mutated["session"]).dt.date.astype(str).eq(
        case.removed_sessions[-1]
    )
    mutated.loc[row, "close"] = float(mutated.loc[row, "close"].iloc[0]) + 0.01
    with pytest.raises(ValueError, match="Parquet OHLCV differs"):
        script._price_state(mutated, case, records)


def test_successor_duplicate_evidence_drift_fails_closed(actual_plan: dict) -> None:
    case = next(value for value in script.CASES if value.symbol == "NLOK")
    prices = actual_plan["current"]["daily_price_raw"].copy()
    target = prices["security_id"].astype(str).eq(case.successor_security_id)
    sessions = pd.to_datetime(prices["session"]).dt.date.astype(str)
    row = target & sessions.eq(case.successor_selected_sessions[0])
    prices.loc[row, "close"] = float(prices.loc[row, "close"].iloc[0]) + 0.01
    with pytest.raises(ValueError, match="successor evidence changed"):
        script._verify_successor_duplicate_evidence(
            prices,
            case,
            actual_plan["evidence"][case.symbol]["successor"],
        )


def test_archive_payload_tamper_and_partial_cross_state_fail_closed(
    tmp_path: Path,
) -> None:
    payload = b"exact-short-tail-evidence"
    digest = hashlib.sha256(payload).hexdigest()
    object_path = f"archives/2026-07-15/{digest}.json.gz"
    path = tmp_path / object_path
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(payload, mtime=0))
    repository = type("Repository", (), {"root": tmp_path})()
    row = {"object_path": object_path}
    assert script._archive_payload(
        repository, row, digest=digest, expected_bytes=len(payload)
    ) == payload
    path.write_bytes(gzip.compress(payload + b"-tampered", mtime=0))
    with pytest.raises(ValueError, match="hash/size changed"):
        script._archive_payload(
            repository, row, digest=digest, expected_bytes=len(payload)
        )

    with pytest.raises(RuntimeError, match="partially applied"):
        script._case_state(
            symbol="FLIR",
            price_state="repaired",
            master_state="old",
            history_state="old",
            action_state="neutral",
            resolution_state="old",
        )


def test_snapshot_exception_is_exact_fingerprint_only(monkeypatch) -> None:
    expected = script.EXPECTED_SNAPSHOT_IDENTITY_GAP
    exact = ValidationReport(
        "snapshot",
        (
            ValidationIssue(
                expected["code"],
                "reviewed NBL replay gap",
                row_count=1,
                fingerprints=(expected["fingerprint"],),
            ),
        ),
    )
    monkeypatch.setattr(script, "validate_repository_snapshot", lambda _repo: exact)
    assert script._validate_candidate_snapshot(object()) is True

    drift = ValidationReport(
        "snapshot",
        (
            ValidationIssue(
                expected["code"],
                "drifted replay gap",
                row_count=1,
                fingerprints=("0" * 64,),
            ),
        ),
    )
    monkeypatch.setattr(script, "validate_repository_snapshot", lambda _repo: drift)
    with pytest.raises(ValueError, match="drifted replay gap"):
        script._validate_candidate_snapshot(object())


def _exact_early_history_factor_fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
    script.FactorManifestBinding,
]:
    stem = "early-terminal-history-2026-07-15-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    daily_version = f"{stem}-daily_price_raw"
    action_version = f"{stem}-corporate_actions"
    factor_version = f"{stem}-adjustment_factors"
    lineage = script._early_history_source_version(daily_version, action_version)
    prices = pd.DataFrame(
        {
            "security_id": ["US:TEST:ONE", "US:TEST:ONE"],
            "session": ["2026-07-14", "2026-07-15"],
            "close": [10.0, 11.0],
        }
    )
    actions = pd.DataFrame(
        columns=[
            "event_id",
            "security_id",
            "action_type",
            "effective_date",
            "ex_date",
            "ratio",
            "cash_amount",
        ]
    )
    factors = pd.DataFrame(
        {
            "security_id": ["US:TEST:ONE", "US:TEST:ONE"],
            "session": ["2026-07-14", "2026-07-15"],
            "split_factor": [1.0, 1.0],
            "total_return_factor": [1.0, 1.0],
            "source_version": [lineage, lineage],
            "calculated_at": [
                "2026-07-18T22:55:51Z",
                "2026-07-18T22:55:51Z",
            ],
            "source": ["derived", "derived"],
            "retrieved_at": [
                "2026-07-18T22:55:51Z",
                "2026-07-18T22:55:51Z",
            ],
            "source_hash": [lineage, lineage],
        }
    )
    manifest = DatasetManifest(
        dataset="adjustment_factors",
        version=factor_version,
        created_at="2026-07-18T22:57:26Z",
        completed_session="2026-07-15",
        files=(
            ManifestFile(
                path="year=2026/part-00000.parquet",
                sha256="1" * 64,
                size_bytes=1,
                row_count=len(factors),
                min_session="2026-07-14",
                max_session="2026-07-15",
            ),
        ),
        metadata={
            "operation": script.EARLY_HISTORY_OPERATION,
            "source_version": lineage,
            "source_daily_price_version": daily_version,
            "source_corporate_actions_version": action_version,
            "request_inventory_sha256": (
                script.EARLY_HISTORY_REQUEST_INVENTORY_SHA256
            ),
            "inserted_price_rows": script.EARLY_HISTORY_INSERTED_ROWS,
            "inserted_rows": script.EARLY_HISTORY_INSERTED_ROWS,
            "existing_economic_rows_changed": 0,
            "inherits_parent": False,
            "apply_network_accessed": False,
            "r2_accessed": False,
            "short_terminal_tail_registry_sha256": (
                script.registry_inventory_sha256()
            ),
            "short_terminal_tail_registry": script.registry_draft(),
        },
    )
    binding = script.FactorManifestBinding(
        manifest=manifest,
        factor_version=factor_version,
        daily_price_version=daily_version,
        corporate_actions_version=action_version,
    )
    short_tail_lineage = script._adjustment_source_version(
        daily_version, action_version
    )
    return prices, actions, factors, short_tail_lineage, binding


def _replay_exact_early_history_factors(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    factors: pd.DataFrame,
    short_tail_lineage: str,
    binding: script.FactorManifestBinding | None,
) -> tuple[pd.DataFrame, int]:
    return script._prepare_factors(
        factors,
        prices,
        prices,
        actions,
        source_version=short_tail_lineage,
        repaired_state=True,
        factor_manifest_binding=binding,
    )


def test_exact_current_early_history_full_rebuild_is_idempotent_only_when_bound(
) -> None:
    prices, actions, factors, short_tail_lineage, binding = (
        _exact_early_history_factor_fixture()
    )
    with pytest.raises(RuntimeError, match="provenance is stale"):
        _replay_exact_early_history_factors(
            prices, actions, factors, short_tail_lineage, None
        )

    replay, removed = _replay_exact_early_history_factors(
        prices, actions, factors, short_tail_lineage, binding
    )
    assert removed == 0
    pd.testing.assert_frame_equal(replay, factors)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("operation", "generic_factor_rebuild"),
        ("source_version", f"{script.EARLY_HISTORY_LINEAGE_PREFIX}{'0' * 64}"),
        ("source_daily_price_version", "stale-daily-price-version"),
        ("request_inventory_sha256", "0" * 64),
        ("inherits_parent", True),
        ("short_terminal_tail_registry_sha256", "0" * 64),
    ),
)
def test_early_history_manifest_tamper_fails_closed(
    field: str, value: object
) -> None:
    prices, actions, factors, short_tail_lineage, binding = (
        _exact_early_history_factor_fixture()
    )
    metadata = dict(binding.manifest.metadata)
    metadata[field] = value
    tampered = replace(
        binding,
        manifest=replace(binding.manifest, metadata=metadata),
    )
    with pytest.raises(RuntimeError, match="provenance is stale"):
        _replay_exact_early_history_factors(
            prices, actions, factors, short_tail_lineage, tampered
        )


def test_early_history_row_provenance_economics_and_keys_tamper_fail_closed(
) -> None:
    prices, actions, factors, short_tail_lineage, binding = (
        _exact_early_history_factor_fixture()
    )

    provenance_tamper = factors.copy()
    provenance_tamper.loc[0, "source_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="provenance is stale"):
        _replay_exact_early_history_factors(
            prices, actions, provenance_tamper, short_tail_lineage, binding
        )

    economics_tamper = factors.copy()
    economics_tamper.loc[0, "split_factor"] = 0.5
    with pytest.raises(ValueError, match="changed retained factor economics"):
        _replay_exact_early_history_factors(
            prices, actions, economics_tamper, short_tail_lineage, binding
        )

    with pytest.raises(ValueError, match="keys do not exactly match"):
        _replay_exact_early_history_factors(
            prices,
            actions,
            factors.iloc[:-1].reset_index(drop=True),
            short_tail_lineage,
            binding,
        )


class _TransactionRepository:
    """Small CAS repository used only for transaction-control tests."""

    def __init__(self, root: Path):
        self.root = root
        self.objects = LocalObjectStore(root)
        self.manifests: dict[tuple[str, str], DatasetManifest] = {}
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.write_records: list[tuple[str, str]] = []
        self.release_count = 0
        versions = {
            dataset: f"base-{dataset}" for dataset in script.REQUIRED_DATASETS
        }
        for dataset, version in versions.items():
            manifest = DatasetManifest.create(
                dataset,
                version,
                "2026-07-15",
                (),
                metadata={"preserved": dataset},
            )
            self.manifests[(dataset, version)] = manifest
            manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
            self.objects.put(manifest_path, manifest.to_bytes(), if_none_match=True)
            pointer = CurrentPointer.create(manifest, manifest_path)
            self.objects.put(
                self.current_key(dataset), pointer.to_bytes(), if_none_match=True
            )
            self.frames[(dataset, version)] = pd.DataFrame({"base": [dataset]})
        release = DataRelease(
            version="base-release",
            created_at="2026-07-19T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions=versions,
            quality="valid",
            warnings=(),
        )
        self.objects.put("releases/current.json", release.to_bytes(), if_none_match=True)

    @staticmethod
    def current_key(dataset: str) -> str:
        return LocalDatasetRepository.current_key(dataset)

    def current_release(self):
        value = self.objects.get("releases/current.json")
        return DataRelease.from_bytes(value.data), value.etag

    def current_pointer(self, dataset: str):
        value = self.objects.get(self.current_key(dataset))
        return CurrentPointer.from_bytes(value.data), value.etag

    def manifest_for_version(self, dataset: str, version: str):
        return self.manifests[(dataset, version)]

    def read_frame(self, dataset: str, version: str | None = None):
        if version is None:
            pointer, _ = self.current_pointer(dataset)
            assert pointer is not None
            version = pointer.version
        return self.frames[(dataset, version)].copy()

    def write_frame(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        completed_session: str,
        incomplete_action_policy: str,
        metadata: dict,
        expected_pointer_etag: str | None,
        version: str,
    ):
        manifest = DatasetManifest.create(
            dataset, version, completed_session, (), metadata=metadata
        )
        manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
        self.objects.put(manifest_path, manifest.to_bytes(), if_none_match=True)
        pointer = CurrentPointer.create(manifest, manifest_path)
        self.objects.put(
            self.current_key(dataset),
            pointer.to_bytes(),
            if_match=expected_pointer_etag,
        )
        self.manifests[(dataset, version)] = manifest
        self.frames[(dataset, version)] = frame.copy()
        self.write_records.append((dataset, version))
        return DatasetWriteResult(manifest, ValidationReport(dataset))

    def commit_release(
        self,
        completed_session: str,
        dataset_versions: dict[str, str],
        *,
        quality: str,
        warnings: tuple[str, ...],
        expected_etag: str | None,
    ):
        self.release_count += 1
        release = DataRelease(
            version=f"committed-release-{self.release_count}",
            created_at=f"2026-07-19T00:00:0{self.release_count}Z",
            completed_session=completed_session,
            dataset_versions=dict(dataset_versions),
            quality=quality,
            warnings=warnings,
        )
        self.objects.put(
            "releases/current.json", release.to_bytes(), if_match=expected_etag
        )
        return release


def _transaction_plan(
    repository: _TransactionRepository,
    token: str,
    *,
    status: str = "validated_offline_plan",
) -> script.PreparedRepair:
    release, release_etag = repository.current_release()
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in script.REQUIRED_DATASETS
    }
    planned = (
        {
            dataset: f"short-terminal-{token}-{dataset}"
            for dataset in script.WRITE_DATASETS
        }
        if status != "already_repaired"
        else {}
    )
    frames = (
        {dataset: pd.DataFrame({"plan": [token]}) for dataset in script.WRITE_DATASETS}
        if planned
        else {}
    )
    return script.PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned,
        frames=frames,
        summary={
            "status": status,
            "plan_origin": token,
            "candidate_set_sha256": f"candidate:{token}",
            "resolution_set_sha256": f"resolution:{token}",
            "planned_versions": dict(planned),
        },
    )


def _control_bytes(repository: _TransactionRepository) -> dict[str, bytes]:
    keys = ["releases/current.json"] + [
        repository.current_key(dataset) for dataset in script.REQUIRED_DATASETS
    ]
    return {key: repository.objects.get(key).data for key in keys}


def _install_transaction_guards(
    monkeypatch: pytest.MonkeyPatch,
    plan_provider: Callable[[_TransactionRepository], script.PreparedRepair],
) -> None:
    monkeypatch.setattr(script, "prepare_repair", plan_provider)
    monkeypatch.setattr(script, "_validate_candidate_snapshot", lambda _repo: False)

    def assert_applied(
        repository: _TransactionRepository,
        committed: DataRelease,
        *,
        expected_out_of_scope_pointer_etags: dict[str, str | None],
    ) -> None:
        current, _ = repository.current_release()
        assert current.to_bytes() == committed.to_bytes()
        for dataset in script.REQUIRED_DATASETS:
            pointer, etag = repository.current_pointer(dataset)
            assert pointer.version == committed.dataset_versions[dataset]
            if dataset not in script.WRITE_DATASETS:
                assert etag == expected_out_of_scope_pointer_etags[dataset]

    monkeypatch.setattr(script, "_assert_applied_release", assert_applied)


@pytest.mark.parametrize(
    "failure_stage", ("after_write:adjustment_factors", "after_release_commit")
)
def test_transaction_failure_rolls_back_all_control_pointers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked = _transaction_plan(repository, "locked")
    _install_transaction_guards(monkeypatch, lambda _repo: locked)
    before = _control_bytes(repository)

    def fail(stage: str) -> None:
        if stage == failure_stage:
            raise RuntimeError(f"injected failure at {stage}")

    with pytest.raises(RuntimeError, match="injected failure"):
        script.apply_repair(repository, caller, inject_failure=fail)

    assert _control_bytes(repository) == before
    journals = tuple((tmp_path / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["status"] == "rolled_back"
    assert not (tmp_path / script.RECOVERY_DIR).exists()


def test_transaction_replay_is_locked_idempotent_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked = _transaction_plan(repository, "locked")
    _install_transaction_guards(monkeypatch, lambda _repo: locked)
    first = script.apply_repair(repository, caller)
    assert first["status"] == "applied"

    caller_noop = _transaction_plan(
        repository, "caller-noop", status="already_repaired"
    )
    locked_noop = _transaction_plan(
        repository, "locked-noop", status="already_repaired"
    )
    before = _control_bytes(repository)
    writes_before = len(repository.write_records)
    monkeypatch.setattr(script, "prepare_repair", lambda _repo: locked_noop)
    replay = script.apply_repair(repository, caller_noop)
    assert replay["status"] == "already_repaired"
    assert replay["writes_performed"] is False
    assert len(repository.write_records) == writes_before
    assert _control_bytes(repository) == before


def test_transaction_rejects_stale_release_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    current, _ = repository.current_release()
    changed = replace(
        current,
        version="changed-release",
        created_at="2026-07-19T00:01:00Z",
    )
    repository.objects.put("releases/current.json", changed.to_bytes())
    monkeypatch.setattr(
        script,
        "prepare_repair",
        lambda _repo: pytest.fail("stale plan reached locked replan"),
    )
    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(repository, caller)
    assert repository.write_records == []
    assert not (tmp_path / script.TRANSACTION_DIR).exists()
