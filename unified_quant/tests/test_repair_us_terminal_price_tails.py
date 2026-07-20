from __future__ import annotations

import gzip
import hashlib
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Callable
from unittest.mock import patch

import pandas as pd
import pytest

from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
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
    / "repair_us_terminal_price_tails.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_terminal_price_tails",
    SCRIPT_PATH,
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


def _target_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["security_id"].astype(str).isin(TARGET_IDS)].copy()


@pytest.fixture(scope="module")
def actual_plan() -> dict:
    if not (REPOSITORY_ROOT / "releases/current.json").is_file():
        pytest.skip("Actual current-release cache is unavailable.")
    repository = LocalDatasetRepository(REPOSITORY_ROOT)
    release, _ = repository.current_release()
    if release is None or not set(script.REQUIRED_DATASETS).issubset(
        release.dataset_versions
    ):
        pytest.skip("Actual release lacks terminal-tail datasets.")
    before = _control_hashes(repository)
    prepared = script.prepare_repair(repository)
    after = _control_hashes(repository)
    if prepared.summary["status"] == "already_repaired":
        pytest.skip("Actual release is already terminal-tail repaired.")

    old_targets: dict[str, pd.DataFrame] = {}
    for dataset in script.WRITE_DATASETS:
        current = repository.read_frame(dataset, release.dataset_versions[dataset])
        old_targets[dataset] = _target_rows(current)
    new_targets = {
        dataset: _target_rows(prepared.frames[dataset])
        for dataset in script.WRITE_DATASETS
    }
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    evidence = {
        case.symbol: script._verify_case_evidence(
            repository,
            archive,
            case,
            completed_session=release.completed_session,
        )
        for case in script.CASES
    }
    factor_lineage = {
        column: set(prepared.frames["adjustment_factors"][column].astype(str))
        for column in (
            "source_version",
            "calculated_at",
            "source",
            "retrieved_at",
            "source_hash",
        )
    }
    return {
        "release": release,
        "before": before,
        "after": after,
        "summary": dict(prepared.summary),
        "old": old_targets,
        "new": new_targets,
        "evidence": evidence,
        "factor_lineage": factor_lineage,
    }


def test_actual_plan_is_read_only_offline_and_exact(actual_plan: dict) -> None:
    summary = actual_plan["summary"]
    assert actual_plan["before"] == actual_plan["after"]
    assert summary["status"] == "validated_offline_plan"
    assert summary["removed_daily_price_rows"] == 14
    assert summary["removed_adjustment_factor_rows"] == 14
    assert summary["daily_price_rows_before"] == 2_095_761
    assert summary["daily_price_rows_after"] == 2_095_747
    assert summary["adjustment_factor_rows_before"] == 2_095_761
    assert summary["adjustment_factor_rows_after"] == 2_095_747
    assert summary["network_accessed"] is False
    assert summary["eodhd_calls"] == 0
    assert summary["r2_accessed"] is False
    assert summary["source_archive_immutable"] is True
    assert summary["index_membership_events_unchanged"] is True
    assert summary["index_constituent_anchors_unchanged"] is True
    assert summary["snapshot_identity_gap"] == script.EXPECTED_SNAPSHOT_IDENTITY_GAP
    if actual_plan["release"].version == "20260715-20260718T122234681534Z":
        assert (
            summary["candidate_set_sha256"]
            == script.EXPECTED_CURRENT_CANDIDATE_SET_SHA256
        )


def test_exact_reviewed_price_and_factor_rows_are_removed(actual_plan: dict) -> None:
    old_prices = actual_plan["old"]["daily_price_raw"]
    new_prices = actual_plan["new"]["daily_price_raw"]
    old_factors = actual_plan["old"]["adjustment_factors"]
    new_factors = actual_plan["new"]["adjustment_factors"]
    for case in script.CASES:
        old = old_prices.loc[old_prices["security_id"].astype(str).eq(case.security_id)]
        new = new_prices.loc[new_prices["security_id"].astype(str).eq(case.security_id)]
        old_sessions = set(pd.to_datetime(old["session"]).dt.date.astype(str))
        new_sessions = set(pd.to_datetime(new["session"]).dt.date.astype(str))
        removed = sorted(old_sessions - new_sessions)
        assert len(removed) == case.removed_tail_count
        assert removed[0] == case.removed_tail_start
        assert removed[-1] == case.removed_tail_end
        assert max(new_sessions) == case.last_real_session

        old_factor = old_factors.loc[
            old_factors["security_id"].astype(str).eq(case.security_id)
        ]
        new_factor = new_factors.loc[
            new_factors["security_id"].astype(str).eq(case.security_id)
        ]
        old_factor_sessions = set(
            pd.to_datetime(old_factor["session"]).dt.date.astype(str)
        )
        new_factor_sessions = set(
            pd.to_datetime(new_factor["session"]).dt.date.astype(str)
        )
        assert sorted(old_factor_sessions - new_factor_sessions) == removed


def test_retained_factor_economics_and_planned_lineage_are_exact(
    actual_plan: dict,
) -> None:
    old = actual_plan["old"]["adjustment_factors"].copy()
    new = actual_plan["new"]["adjustment_factors"].copy()
    new_keys = set(
        zip(
            new["security_id"].astype(str),
            pd.to_datetime(new["session"]).dt.normalize(),
        )
    )
    keep = [
        (str(row.security_id), pd.Timestamp(row.session).normalize()) in new_keys
        for row in old.itertuples(index=False)
    ]
    assert script._factor_economics_equal(old.loc[keep], new) == 0
    lineage = actual_plan["summary"]["factor_source_version"]
    assert actual_plan["factor_lineage"] == {
        "source_version": {lineage},
        "calculated_at": {script.REPAIR_REVIEWED_AT},
        "source": {"derived"},
        "retrieved_at": {script.REPAIR_REVIEWED_AT},
        "source_hash": {lineage},
    }
    assert actual_plan["summary"]["adjustment_factor_economic_rows_changed"] == 0
    assert (
        actual_plan["summary"]["adjustment_factor_provenance_rows_rebound"]
        == 2_095_747
    )


def test_terminal_readiness_targets_move_from_three_to_zero(actual_plan: dict) -> None:
    summary = actual_plan["summary"]
    assert summary["target_terminal_issues_before"] == 3
    assert summary["target_terminal_issue_codes_before"] == [
        "source_reentry_after_terminal_action",
        "source_reentry_after_terminal_action",
        "source_reentry_after_terminal_action",
    ]
    assert summary["target_terminal_issues_after"] == 0
    assert summary["target_terminal_issue_codes_after"] == []
    assert summary["terminal_issues_before_total"] == 12
    assert summary["terminal_issues_after_total"] == 9
    assert summary["non_target_terminal_issues_unchanged"] is True


def test_identity_action_resolution_rewrites_are_narrow(actual_plan: dict) -> None:
    expected_changed = {
        "security_master": {
            "active_to",
            "exchange",
            "source",
            "source_url",
            "retrieved_at",
            "source_hash",
        },
        "symbol_history": {
            "effective_to",
            "exchange",
            "source",
            "source_url",
            "retrieved_at",
            "source_hash",
        },
        "lifecycle_resolutions": {
            "candidate_id",
            "last_price_date",
            "event_id",
            "reviewed_by",
            "reviewed_at",
            "source",
            "retrieved_at",
        },
    }
    for dataset, expected in expected_changed.items():
        old = actual_plan["old"][dataset].sort_values("security_id").reset_index(drop=True)
        new = actual_plan["new"][dataset].sort_values("security_id").reset_index(drop=True)
        changed = {
            column for column in old.columns if not old[column].equals(new[column])
        }
        # Exchange changes only for NBL, but the column is still one reviewed field.
        assert changed == expected

    old_actions = actual_plan["old"]["corporate_actions"]
    new_actions = actual_plan["new"]["corporate_actions"]
    old_mergers = old_actions.loc[old_actions["action_type"].astype(str).eq("stock_merger")]
    new_mergers = new_actions.loc[new_actions["action_type"].astype(str).eq("stock_merger")]
    old_mergers = old_mergers.sort_values("security_id").reset_index(drop=True)
    new_mergers = new_mergers.sort_values("security_id").reset_index(drop=True)
    changed = {
        column
        for column in old_mergers.columns
        if not old_mergers[column].equals(new_mergers[column])
    }
    assert changed == {"event_id", "effective_date", "ex_date"}
    cxo = new_mergers.loc[
        new_mergers["security_id"].astype(str).eq(script.CASES[2].security_id)
    ].iloc[0]
    assert cxo["event_id"] == script.CASES[2].new_event_id
    assert script._date(cxo["effective_date"]) == "2021-01-19"


def test_partial_price_tail_or_terminal_mutation_fails_closed(actual_plan: dict) -> None:
    case = next(value for value in script.CASES if value.symbol == "XLNX")
    old = actual_plan["old"]["daily_price_raw"]
    old = old.loc[old["security_id"].astype(str).eq(case.security_id)].copy()
    sessions = pd.to_datetime(old["session"]).dt.date.astype(str)
    partial = old.loc[~sessions.eq(case.removed_tail_start)].copy()
    with pytest.raises(ValueError, match="neither exact raw nor exact repaired"):
        script._price_state(
            partial,
            case,
            actual_plan["evidence"][case.symbol]["source"],
        )

    mutated = old.copy()
    terminal = pd.to_datetime(mutated["session"]).dt.date.astype(str).eq(
        case.last_real_session
    )
    mutated.loc[terminal, "close"] = float(mutated.loc[terminal, "close"].iloc[0]) + 0.01
    with pytest.raises(ValueError, match="Parquet OHLCV differs"):
        script._price_state(
            mutated,
            case,
            actual_plan["evidence"][case.symbol]["source"],
        )


def test_original_archive_hashes_and_registry_ids_are_code_pinned(
    actual_plan: dict,
) -> None:
    expected = {
        "NBL": (
            "f468258431303cd0278595d4457bac8af8e741cb2737cea142f28f1e25d5c5da",
            "43174e67dc9c4faedd3eb34b645f5f394c8dc83f264dfc0697525fc1661dc5ff",
            "ac20b7b59563fe0b7f96a244092df17a3c5de3188d75144ff1c46cfa4ccb7955",
        ),
        "XLNX": (
            "5e23869bd317191de08bc0fb9c021b9cac03d1a1c0bc2d9e0d93f090888b1fb3",
            "a77056ceded0f8213430f8943774378ec591c47f7efd5236b410acdade52b0ac",
            "93d51e30eff9d6edf166c175d59a20421be00dc3790335712bd1e3ef758a08db",
        ),
        "CXO": (
            "59e7e2843948065b50ced2daf2012cab37061643a03e0929426b7507997a64bb",
            "d50b081e7df7a511ac7a3ec685191b6e3c9af09927502da9a045cbfd999bfe17",
            "f326b63d0229f68816a663f468e1723b8925db58b8f21607cb3b16e72cfb531c",
        ),
    }
    draft = {row["symbol"]: row for row in script.registry_draft()}
    assert set(draft) == set(expected)
    for symbol, (raw_hash, tail_hash, candidate_id) in expected.items():
        assert draft[symbol]["raw_source_hash"] == raw_hash
        assert draft[symbol]["removed_tail_sha256"] == tail_hash
        assert draft[symbol]["candidate_id"] == candidate_id
        assert len(draft[symbol]["registry_item_sha256"]) == 64
    assert (
        actual_plan["summary"]["registry_inventory_sha256"]
        == script.registry_inventory_sha256()
    )


def test_archive_payload_tamper_fails_closed(tmp_path: Path) -> None:
    payload = b"exact-offline-evidence"
    digest = hashlib.sha256(payload).hexdigest()
    object_path = f"archives/2026-07-15/{digest}.json.gz"
    path = tmp_path / object_path
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(payload, mtime=0))
    repository = SimpleNamespace(root=tmp_path)
    row = {"object_path": object_path}
    assert script._archive_payload(
        repository,
        row,
        digest=digest,
        expected_bytes=len(payload),
    ) == payload
    path.write_bytes(gzip.compress(payload + b"-tampered", mtime=0))
    with pytest.raises(ValueError, match="hash/size changed"):
        script._archive_payload(
            repository,
            row,
            digest=digest,
            expected_bytes=len(payload),
        )


def test_partial_cross_dataset_state_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="partially applied"):
        script._case_state(
            price_state="repaired",
            master_state="old",
            history_state="old",
            action_state="neutral",
            resolution_state="old",
            symbol="NBL",
        )


def test_snapshot_identity_gap_fingerprint_is_exact_and_fail_closed() -> None:
    expected = script.EXPECTED_SNAPSHOT_IDENTITY_GAP
    fingerprint = script.index_member_identity_gap_fingerprint(
        index_id=expected["index_id"],
        replay_date=expected["replay_date"],
        security_id=expected["security_id"],
        next_remove_event_id=expected["next_remove_event_id"],
        next_remove_effective_date=expected["next_remove_effective_date"],
        next_remove_source=expected["next_remove_source"],
        next_remove_source_hash=expected["next_remove_source_hash"],
    )
    assert fingerprint == expected["fingerprint"]

    exact = ValidationReport(
        "snapshot",
        (
            ValidationIssue(
                expected["code"],
                "expected exact NBL replay gap",
                row_count=1,
                fingerprints=(fingerprint,),
            ),
        ),
    )
    with patch.object(script, "validate_repository_snapshot", return_value=exact):
        assert script._validate_candidate_snapshot(SimpleNamespace()) is True

    mutated = ValidationReport(
        "snapshot",
        (
            ValidationIssue(
                expected["code"],
                "same code and count but different replay lineage",
                row_count=1,
                fingerprints=("0" * 64,),
            ),
        ),
    )
    with patch.object(
        script, "validate_repository_snapshot", return_value=mutated
    ), pytest.raises(ValueError, match="different replay lineage"):
        script._validate_candidate_snapshot(SimpleNamespace())


def _transaction_frames(factor_lineage: str) -> dict[str, pd.DataFrame]:
    source = {
        "source": "transaction_test",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "a" * 64,
    }
    security_id = "US:TEST:OLD"
    action_id = "test-action"
    return {
        "corporate_actions": pd.DataFrame(
            [
                {
                    "event_id": action_id,
                    "security_id": security_id,
                    "action_type": "cash_dividend",
                    "effective_date": "2026-07-15",
                    "ex_date": "2026-07-15",
                    "announcement_date": "2026-07-14",
                    "record_date": "",
                    "payment_date": "",
                    "cash_amount": 1.0,
                    "ratio": None,
                    "currency": "USD",
                    "new_security_id": "",
                    "new_symbol": "",
                    "official": True,
                    "source_url": "https://example.test/action",
                    "source_kind": "official_filing",
                    **source,
                }
            ]
        ),
        "daily_price_raw": pd.DataFrame(
            [
                {
                    "security_id": security_id,
                    "session": "2026-07-15",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0,
                    "volume": 100.0,
                    "currency": "USD",
                    **source,
                }
            ]
        ),
        "adjustment_factors": pd.DataFrame(
            [
                {
                    "security_id": security_id,
                    "session": "2026-07-15",
                    "split_factor": 1.0,
                    "total_return_factor": 1.0,
                    "source_version": factor_lineage,
                    "calculated_at": script.REPAIR_REVIEWED_AT,
                    "source": "derived",
                    "retrieved_at": script.REPAIR_REVIEWED_AT,
                    "source_hash": factor_lineage,
                }
            ]
        ),
        "lifecycle_resolutions": pd.DataFrame(
            [
                {
                    "candidate_id": "test-candidate",
                    "security_id": security_id,
                    "symbol": "OLD",
                    "last_price_date": "2026-07-15",
                    "resolution": "applied",
                    "event_id": action_id,
                    "exception_code": "",
                    "exception_reason": "",
                    "reviewed_by": "transaction-test",
                    "reviewed_at": "2026-07-18T00:00:00Z",
                    "recheck_after": "",
                    "successor_security_id": "",
                    "successor_symbol": "",
                    "source_url": "https://example.test/action",
                    **source,
                }
            ]
        ),
        "security_master": pd.DataFrame(
            [
                {
                    "security_id": security_id,
                    "primary_symbol": "OLD",
                    "name": "Old Security",
                    "exchange": "XNYS",
                    "asset_type": "Common Stock",
                    "currency": "USD",
                    "country": "US",
                    "active_from": "2015-01-02",
                    "active_to": "",
                    **source,
                }
            ]
        ),
        "symbol_history": pd.DataFrame(
            [
                {
                    "security_id": security_id,
                    "symbol": "OLD",
                    "exchange": "XNYS",
                    "effective_from": "2015-01-02",
                    "effective_to": "",
                    **source,
                }
            ]
        ),
    }


class _TransactionRepository:
    """Small CAS repository used only to exercise the apply transaction."""

    def __init__(self, root: Path):
        self.root = root
        self.objects = LocalObjectStore(root)
        self.manifests: dict[tuple[str, str], DatasetManifest] = {}
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.write_records: list[tuple[str, str, pd.DataFrame]] = []
        self.release_count = 0
        versions = {
            dataset: f"base-{dataset}" for dataset in script.REQUIRED_DATASETS
        }
        for dataset, version in versions.items():
            metadata = {"preserved_metadata": f"preserved:{dataset}"}
            if dataset == "lifecycle_resolutions":
                metadata["evidence_report_sha256"] = script.EVIDENCE_REPORT_HASH
            manifest = DatasetManifest.create(
                dataset,
                version,
                "2026-07-15",
                (),
                metadata=metadata,
            )
            self.manifests[(dataset, version)] = manifest
            manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
            self.objects.put(manifest_path, manifest.to_bytes(), if_none_match=True)
            pointer = CurrentPointer.create(manifest, manifest_path)
            self.objects.put(
                self.current_key(dataset), pointer.to_bytes(), if_none_match=True
            )
            self.frames[(dataset, version)] = pd.DataFrame(
                {"base_marker": [dataset]}
            )
        release = DataRelease(
            version="base-release",
            created_at="2026-07-18T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions=versions,
            quality="valid",
            warnings=(),
        )
        self.objects.put("releases/current.json", release.to_bytes(), if_none_match=True)

    @staticmethod
    def current_key(dataset: str) -> str:
        return LocalDatasetRepository.current_key(dataset)

    def current_release(self) -> tuple[DataRelease | None, str | None]:
        value = self.objects.get("releases/current.json")
        return DataRelease.from_bytes(value.data), value.etag

    def current_pointer(self, dataset: str) -> tuple[CurrentPointer | None, str | None]:
        value = self.objects.get(self.current_key(dataset))
        return CurrentPointer.from_bytes(value.data), value.etag

    def manifest_for_version(self, dataset: str, version: str) -> DatasetManifest:
        return self.manifests[(dataset, version)]

    def read_frame(self, dataset: str, version: str | None = None) -> pd.DataFrame:
        if version is None:
            pointer, _ = self.current_pointer(dataset)
            assert pointer is not None
            version = pointer.version
        return self.frames[(dataset, version)].copy(deep=True)

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
    ) -> DatasetWriteResult:
        if (dataset, version) in self.manifests:
            raise FileExistsError(f"Dataset version already exists: {dataset}/{version}")
        validation = script.validate_dataset(
            dataset,
            frame,
            completed_session=completed_session,
            incomplete_action_policy=incomplete_action_policy,
        )
        validation.raise_for_errors()
        manifest = DatasetManifest.create(
            dataset,
            version,
            completed_session,
            (),
            metadata=metadata,
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
        self.frames[(dataset, version)] = frame.copy(deep=True)
        self.write_records.append((dataset, version, frame.copy(deep=True)))
        return DatasetWriteResult(manifest, validation)

    def commit_release(
        self,
        completed_session: str,
        dataset_versions: dict[str, str],
        *,
        quality: str,
        warnings: tuple[str, ...],
        expected_etag: str | None,
    ) -> DataRelease:
        self.release_count += 1
        release = DataRelease(
            version=f"committed-release-{self.release_count}",
            created_at=f"2026-07-18T00:00:0{self.release_count}Z",
            completed_session=completed_session,
            dataset_versions=dict(dataset_versions),
            quality=quality,
            warnings=warnings,
        )
        self.objects.put(
            "releases/current.json", release.to_bytes(), if_match=expected_etag
        )
        return release


def _transaction_control_bytes(repository: _TransactionRepository) -> dict[str, bytes]:
    keys = ["releases/current.json"] + [
        repository.current_key(dataset) for dataset in script.REQUIRED_DATASETS
    ]
    return {key: repository.objects.get(key).data for key in keys}


def _transaction_plan(
    repository: _TransactionRepository,
    token: str,
    *,
    status: str = "validated_offline_plan",
) -> script.PreparedRepair:
    release, release_etag = repository.current_release()
    assert release is not None
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in script.REQUIRED_DATASETS
    }
    planned_versions = (
        {
            dataset: f"terminal-price-tails-{token}-{dataset}"
            for dataset in script.WRITE_DATASETS
        }
        if status != "already_repaired"
        else {}
    )
    if planned_versions:
        lineage = script._adjustment_source_version(
            planned_versions["daily_price_raw"],
            planned_versions["corporate_actions"],
        )
        frames = _transaction_frames(lineage)
    else:
        frames = {}
    return script.PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frames=frames,
        summary={
            "status": status,
            "plan_origin": token,
            "candidate_set_sha256": f"candidate:{token}",
            "resolution_set_sha256": f"resolution:{token}",
            "target_terminal_issues_after": 0,
            "planned_versions": dict(planned_versions),
        },
    )


def _install_transaction_apply_guards(
    monkeypatch: pytest.MonkeyPatch,
    plan_provider: Callable[[_TransactionRepository], script.PreparedRepair],
) -> None:
    monkeypatch.setattr(script, "prepare_repair", plan_provider)
    monkeypatch.setattr(script, "_verify_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        script, "_verify_case_evidence", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        script, "_validate_candidate_snapshot", lambda repository: False
    )

    def assert_applied(
        repository: _TransactionRepository,
        release: DataRelease,
        *,
        expected_candidate_set_sha256: str,
        expected_out_of_scope_pointer_etags: dict[str, str | None],
    ) -> None:
        current, _ = repository.current_release()
        assert current is not None and current.to_bytes() == release.to_bytes()
        assert expected_candidate_set_sha256.startswith("candidate:locked")
        for dataset, version in release.dataset_versions.items():
            pointer, etag = repository.current_pointer(dataset)
            assert pointer is not None and pointer.version == version
            if dataset not in script.WRITE_DATASETS:
                assert etag == expected_out_of_scope_pointer_etags[dataset]

    monkeypatch.setattr(script, "_assert_applied_release", assert_applied)


@pytest.mark.parametrize(
    ("dataset", "column", "mutated_value"),
    (
        ("corporate_actions", "cash_amount", 2.0),
        ("daily_price_raw", "close", 10.5),
        ("adjustment_factors", "total_return_factor", 1.1),
        ("security_master", "name", "Caller-mutated security"),
        ("symbol_history", "exchange", "OTC"),
    ),
)
def test_apply_writes_only_locked_replan_after_schema_valid_frame_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    column: str,
    mutated_value: object,
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked = _transaction_plan(repository, "locked-frame")
    caller.frames[dataset].at[0, column] = mutated_value
    script.validate_dataset(
        dataset,
        caller.frames[dataset],
        completed_session=caller.release.completed_session,
        incomplete_action_policy="block",
    ).raise_for_errors()
    _install_transaction_apply_guards(monkeypatch, lambda _repository: locked)

    result = script.apply_repair(repository, caller)

    assert result["status"] == "applied"
    assert result["plan_origin"] == "locked-frame"
    assert [item[0] for item in repository.write_records] == list(
        script.WRITE_DATASETS
    )
    for written_dataset, version, frame in repository.write_records:
        assert version == locked.planned_versions[written_dataset]
        pd.testing.assert_frame_equal(frame, locked.frames[written_dataset])
    release, _ = repository.current_release()
    assert release is not None
    for written_dataset in script.WRITE_DATASETS:
        assert (
            release.dataset_versions[written_dataset]
            == locked.planned_versions[written_dataset]
        )
        manifest = repository.manifest_for_version(
            written_dataset, release.dataset_versions[written_dataset]
        )
        assert manifest.metadata["preserved_metadata"] == (
            f"preserved:{written_dataset}"
        )
    factor_manifest = repository.manifest_for_version(
        "adjustment_factors", release.dataset_versions["adjustment_factors"]
    )
    expected_lineage = script._adjustment_source_version(
        release.dataset_versions["daily_price_raw"],
        release.dataset_versions["corporate_actions"],
    )
    assert factor_manifest.metadata["source_version"] == expected_lineage
    assert factor_manifest.metadata["source_daily_price_version"] == (
        release.dataset_versions["daily_price_raw"]
    )
    assert factor_manifest.metadata["source_corporate_actions_version"] == (
        release.dataset_versions["corporate_actions"]
    )


def test_apply_ignores_caller_summary_and_planned_version_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked = _transaction_plan(repository, "locked-contract")
    caller.summary["status"] = "already_repaired"
    caller.summary["candidate_set_sha256"] = "caller-tampered"
    for dataset in script.WRITE_DATASETS:
        caller.planned_versions[dataset] = "caller-duplicate-version"
    _install_transaction_apply_guards(monkeypatch, lambda _repository: locked)

    result = script.apply_repair(repository, caller)

    assert result["status"] == "applied"
    assert result["plan_origin"] == "locked-contract"
    assert {item[1] for item in repository.write_records} == set(
        locked.planned_versions.values()
    )


@pytest.mark.parametrize("changed_control", ("release", "pointer"))
def test_apply_rejects_post_plan_release_or_pointer_change_before_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_control: str,
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    if changed_control == "release":
        release, _ = repository.current_release()
        assert release is not None
        changed = replace(
            release,
            version="changed-release",
            created_at="2026-07-18T00:01:00Z",
        )
        repository.objects.put("releases/current.json", changed.to_bytes())
        pattern = "Current release changed"
    else:
        pointer, _ = repository.current_pointer("daily_price_raw")
        assert pointer is not None
        repository.objects.put(
            repository.current_key("daily_price_raw"),
            replace(pointer, updated_at="2026-07-18T00:01:00Z").to_bytes(),
        )
        pattern = "daily_price_raw pointer changed"
    monkeypatch.setattr(
        script,
        "prepare_repair",
        lambda _repository: pytest.fail("stale input reached locked replanning"),
    )

    with pytest.raises(RuntimeError, match=pattern):
        script.apply_repair(repository, caller)

    assert not repository.write_records
    assert not (tmp_path / script.TRANSACTION_DIR).exists()


@pytest.mark.parametrize(
    "failure_stage",
    (
        "after_write:corporate_actions",
        "after_write:adjustment_factors",
        "after_release_commit",
    ),
)
def test_apply_failure_rolls_back_and_retry_uses_unique_locked_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked_first = _transaction_plan(repository, "locked-attempt-1")
    locked_retry = _transaction_plan(repository, "locked-attempt-2")
    plans = iter((locked_first, locked_retry))
    _install_transaction_apply_guards(monkeypatch, lambda _repository: next(plans))
    before = _transaction_control_bytes(repository)

    def fail(stage: str) -> None:
        if stage == failure_stage:
            raise RuntimeError(f"injected failure at {stage}")

    with pytest.raises(RuntimeError, match="injected failure"):
        script.apply_repair(repository, caller, inject_failure=fail)

    assert _transaction_control_bytes(repository) == before
    journals = tuple((tmp_path / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    assert '"status": "rolled_back"' in journals[0].read_text()
    assert not (tmp_path / script.RECOVERY_DIR).exists()

    result = script.apply_repair(repository, caller)

    assert result["status"] == "applied"
    assert result["plan_origin"] == "locked-attempt-2"
    assert set(locked_first.planned_versions.values()).isdisjoint(
        set(locked_retry.planned_versions.values())
    )


def test_apply_replay_is_locked_idempotent_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked = _transaction_plan(repository, "locked-first")
    _install_transaction_apply_guards(monkeypatch, lambda _repository: locked)
    first = script.apply_repair(repository, caller)
    assert first["status"] == "applied"

    caller_noop = _transaction_plan(
        repository, "caller-noop", status="already_repaired"
    )
    locked_noop = _transaction_plan(
        repository, "locked-noop", status="already_repaired"
    )
    caller_noop.summary["status"] = "validated_offline_plan"
    before = _transaction_control_bytes(repository)
    writes_before = len(repository.write_records)
    monkeypatch.setattr(script, "prepare_repair", lambda _repository: locked_noop)

    replay = script.apply_repair(repository, caller_noop)

    assert replay["status"] == "already_repaired"
    assert replay["plan_origin"] == "locked-noop"
    assert replay["writes_performed"] is False
    assert len(repository.write_records) == writes_before
    assert _transaction_control_bytes(repository) == before


def test_apply_rejects_same_version_out_of_scope_pointer_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked = _transaction_plan(repository, "locked-out-of-scope")
    _install_transaction_apply_guards(monkeypatch, lambda _repository: locked)

    def mutate_out_of_scope(stage: str) -> None:
        if stage != "after_write:symbol_history":
            return
        dataset = "source_archive"
        pointer, _ = repository.current_pointer(dataset)
        assert pointer is not None
        repository.objects.put(
            repository.current_key(dataset),
            replace(pointer, updated_at="2026-07-18T00:02:00Z").to_bytes(),
        )

    with pytest.raises(RuntimeError, match="Out-of-scope pointer changed"):
        script.apply_repair(
            repository, caller, inject_failure=mutate_out_of_scope
        )

    release, _ = repository.current_release()
    assert release is not None and release.version == "base-release"
    for dataset in script.WRITE_DATASETS:
        pointer, _ = repository.current_pointer(dataset)
        assert pointer is not None
        assert pointer.version == release.dataset_versions[dataset]
    journals = tuple((tmp_path / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    assert '"status": "rolled_back"' in journals[0].read_text()
