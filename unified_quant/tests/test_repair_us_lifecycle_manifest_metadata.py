from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from supertrend_quant.market_store.lifecycle_coverage import (
    lifecycle_candidate_id,
    lifecycle_candidate_set_sha256,
    lifecycle_resolution_set_sha256,
)
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
    ManifestFile,
    sha256_bytes,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "unified_quant/scripts/repair_us_lifecycle_manifest_metadata.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_lifecycle_manifest_metadata", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


COMPLETED_SESSION = "2026-07-15"
REPORT_RELEASE_VERSION = "fixture-report-release"
BASE_RELEASE_VERSION = "fixture-base-release"


@dataclass(frozen=True)
class FixtureState:
    repository: LocalDatasetRepository
    base_release: DataRelease
    report_sha256: str
    truth_bindings: dict[str, object]


def _put_manifest(
    repository: LocalDatasetRepository,
    dataset: str,
    version: str,
    *,
    metadata: dict[str, object],
    payload: bytes | None = None,
    parent_version: str = "",
) -> DatasetManifest:
    files: tuple[ManifestFile, ...] = ()
    if payload is not None:
        path = repository.root / repository.version_prefix(dataset, version) / "part-00000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files = (
            ManifestFile(
                path="part-00000.parquet",
                sha256=sha256_bytes(payload),
                size_bytes=len(payload),
                row_count=1,
            ),
        )
    manifest = DatasetManifest.create(
        dataset=dataset,
        version=version,
        completed_session=COMPLETED_SESSION,
        files=files,
        parent_version=parent_version,
        metadata=metadata,
    )
    manifest_key = f"{repository.version_prefix(dataset, version)}/manifest.json"
    repository.objects.put(manifest_key, manifest.to_bytes(), if_none_match=True)
    pointer = CurrentPointer.create(manifest, manifest_key)
    repository.objects.put(repository.current_key(dataset), pointer.to_bytes())
    return manifest


def _write_release(repository: LocalDatasetRepository, release: DataRelease) -> None:
    repository.objects.put(
        f"releases/{release.version}.json", release.to_bytes(), if_none_match=True
    )
    current, etag = repository.current_release()
    repository.objects.put(
        "releases/current.json",
        release.to_bytes(),
        if_match=etag,
        if_none_match=current is None,
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> FixtureState:
    repository = LocalDatasetRepository(tmp_path / "cache")
    repository.root.mkdir(parents=True)
    versions = {
        "adjustment_factors": "report-adjustment_factors",
        "corporate_actions": "report-corporate_actions",
        "daily_price_raw": "report-daily_price_raw",
        "index_constituent_anchors": "report-index_constituent_anchors",
        "index_membership_events": "report-index_membership_events",
        "lifecycle_resolutions": "report-lifecycle_resolutions",
        "security_master": "report-security_master",
        "source_archive": "report-source_archive",
        "symbol_history": "report-symbol_history",
    }
    security_id = "US:FIXTURE:ONE"
    last_price_date = "2024-01-02"
    candidate_id = lifecycle_candidate_id(security_id, last_price_date)
    candidates = pd.DataFrame(
        [
            {
                "security_id": security_id,
                "last_price_date": last_price_date,
                "candidate_id": candidate_id,
            }
        ]
    )
    candidate_hash = lifecycle_candidate_set_sha256(candidates)
    report = {
        "report_schema": "fixture/v1",
        "release_version": REPORT_RELEASE_VERSION,
        "completed_session": COMPLETED_SESSION,
        "input_dataset_versions": versions,
        "candidate_count": 1,
        "candidate_set_sha256": candidate_hash,
        "records": {
            security_id: {
                "candidate": {
                    "security_id": security_id,
                    "last_price_date": last_price_date,
                    "symbol": "ONE",
                }
            }
        },
    }
    report_payload = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    report_sha = sha256_bytes(report_payload)
    report_path = f"archives/{COMPLETED_SESSION}/{report_sha}.json.gz"
    repository.objects.put(
        report_path, gzip.compress(report_payload, mtime=0), if_none_match=True
    )

    resolutions = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "security_id": security_id,
                "symbol": "ONE",
                "last_price_date": last_price_date,
                "resolution": "applied",
                "event_id": "event-one",
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": "fixture-reviewer",
                "reviewed_at": "2026-07-15T00:00:00Z",
                "recheck_after": "",
                "successor_security_id": "",
                "successor_symbol": "",
                "source_url": "https://example.test/report",
                "source": "fixture",
                "retrieved_at": "2026-07-15T00:00:00Z",
                "source_hash": report_sha,
            }
        ]
    )
    resolution_hash = lifecycle_resolution_set_sha256(resolutions)
    coverage = {
        "coverage_gate_version": 1,
        "selection_rule": "us_terminal_v1",
        "candidate_set_sha256": candidate_hash,
        "resolution_set_sha256": resolution_hash,
        "candidate_count": 1,
        "resolution_count": 1,
        "applied_count": 1,
        "exception_count": 0,
        "open_count": 0,
    }
    truth_bindings: dict[str, object] = {
        **coverage,
        "lifecycle_coverage": coverage,
        "lifecycle_candidate_set_sha256": candidate_hash,
        "lifecycle_resolution_set_sha256": resolution_hash,
        "evidence_report_sha256": report_sha,
        "evidence_report_object_path": report_path,
        "lifecycle_evidence_report_sha256": report_sha,
        "lifecycle_evidence_report_object_path": report_path,
        "lifecycle_report_release_version": REPORT_RELEASE_VERSION,
        "lifecycle_report_input_versions": versions,
        "superseded_evidence_report_sha256": "a" * 64,
        "current_source_report_sha256": "a" * 64,
        "pinned_base_ancestor_report_sha256": "b" * 64,
        "lifecycle_hints_sha256": "c" * 64,
        "full_lifecycle_finalizer_gate_passed": True,
    }
    truth_result = repository.write_frame(
        "lifecycle_resolutions",
        resolutions,
        completed_session=COMPLETED_SESSION,
        metadata={"operation": "truth", **truth_bindings},
        version=versions["lifecycle_resolutions"],
    )
    assert not truth_result.conflict

    archive = pd.DataFrame(
        [
            {
                "archive_id": report_sha,
                "dataset": "lifecycle_evidence_report",
                "object_path": report_path,
                "content_type": "application/json",
                "effective_date": COMPLETED_SESSION,
                "source": "lifecycle_evidence_report",
                "retrieved_at": "2026-07-15T00:00:00Z",
                "source_hash": report_sha,
                "source_url": "generated://fixture/lifecycle-report",
            }
        ]
    )
    stale_coverage = {
        **coverage,
        "candidate_count": 2,
        "resolution_count": 2,
        "applied_count": 2,
        "candidate_set_sha256": "d" * 64,
        "resolution_set_sha256": "e" * 64,
    }
    stale_metadata: dict[str, object] = {
        **stale_coverage,
        "operation": "fixture-parent",
        "lifecycle_coverage": stale_coverage,
        "lifecycle_candidate_set_sha256": "d" * 64,
        "lifecycle_resolution_set_sha256": "e" * 64,
        "evidence_report_sha256": report_sha,
        "evidence_report_object_path": report_path,
        "lifecycle_evidence_report_sha256": "f" * 64,
        "lifecycle_evidence_report_object_path": (
            f"archives/{COMPLETED_SESSION}/{'f' * 64}.json.gz"
        ),
        "lifecycle_report_release_version": "superseded-release",
        "lifecycle_report_input_versions": {"superseded": "true"},
        "superseded_evidence_report_sha256": "0" * 64,
        "current_source_report_sha256": "0" * 64,
        "pinned_base_ancestor_report_sha256": "0" * 64,
        "lifecycle_hints_sha256": "0" * 64,
        "full_lifecycle_finalizer_gate_passed": True,
        "needs_lifecycle_refinalization": True,
    }
    archive_result = repository.write_frame(
        "source_archive",
        archive,
        completed_session=COMPLETED_SESSION,
        metadata=stale_metadata,
        version=versions["source_archive"],
    )
    assert not archive_result.conflict

    for dataset in versions:
        if dataset in {"lifecycle_resolutions", "source_archive"}:
            continue
        metadata = stale_metadata if dataset in script.TARGET_DATASETS else {"fixture": True}
        _put_manifest(
            repository,
            dataset,
            versions[dataset],
            metadata=dict(metadata),
            payload=(f"exact-{dataset}\n".encode() if dataset in script.TARGET_DATASETS else None),
        )

    report_release = DataRelease(
        version=REPORT_RELEASE_VERSION,
        created_at="2026-07-15T00:00:00Z",
        completed_session=COMPLETED_SESSION,
        dataset_versions=versions,
        quality="degraded",
        warnings=("fixture warning",),
    )
    _write_release(repository, report_release)

    current_archive_result = repository.write_frame(
        "source_archive",
        archive,
        completed_session=COMPLETED_SESSION,
        metadata=stale_metadata,
        version="current-source_archive",
    )
    assert not current_archive_result.conflict
    current_versions = {**versions, "source_archive": current_archive_result.manifest.version}
    base_release = DataRelease(
        version=BASE_RELEASE_VERSION,
        created_at="2026-07-19T00:00:00Z",
        completed_session=COMPLETED_SESSION,
        dataset_versions=current_versions,
        quality="degraded",
        warnings=("fixture warning",),
    )
    _write_release(repository, base_release)

    target_versions = {
        dataset: current_versions[dataset] for dataset in script.TARGET_DATASETS
    }
    target_hashes = {
        dataset: sha256_bytes(
            repository.objects.get(
                f"{repository.version_prefix(dataset, version)}/manifest.json"
            ).data
        )
        for dataset, version in target_versions.items()
    }
    truth_hash = sha256_bytes(
        repository.objects.get(
            f"{repository.version_prefix('lifecycle_resolutions', versions['lifecycle_resolutions'])}/manifest.json"
        ).data
    )
    monkeypatch.setattr(script, "EXPECTED_BASE_RELEASE_VERSION", BASE_RELEASE_VERSION)
    monkeypatch.setattr(
        script, "EXPECTED_BASE_RELEASE_SHA256", sha256_bytes(base_release.to_bytes())
    )
    monkeypatch.setattr(script, "EXPECTED_TARGET_PARENT_VERSIONS", target_versions)
    monkeypatch.setattr(
        script, "EXPECTED_TARGET_PARENT_MANIFEST_SHA256", target_hashes
    )
    monkeypatch.setattr(
        script, "EXPECTED_TRUTH_VERSION", versions["lifecycle_resolutions"]
    )
    monkeypatch.setattr(script, "EXPECTED_TRUTH_MANIFEST_SHA256", truth_hash)
    monkeypatch.setattr(script, "EXPECTED_REPORT_SHA256", report_sha)
    return FixtureState(repository, base_release, report_sha, truth_bindings)


def _pointer_bytes(repository: LocalDatasetRepository) -> dict[str, bytes]:
    return {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.TARGET_DATASETS
    }


def test_real_release_plan_is_exact_and_no_write() -> None:
    repository = LocalDatasetRepository(PROJECT_ROOT / "data/cache")
    release, _ = repository.current_release()
    if release is None or release.version != script.EXPECTED_BASE_RELEASE_VERSION:
        pytest.skip("Exact reviewed local release is unavailable.")
    before_release = repository.objects.get("releases/current.json").data
    before_pointers = _pointer_bytes(repository)
    before_versions = {
        dataset: set(
            repository.objects.list(f"datasets/{dataset}/versions")
        )
        for dataset in script.TARGET_DATASETS
    }

    prepared = script.prepare_repair(repository)

    assert prepared.summary["status"] == "validated_plan"
    assert prepared.summary["writes_required"] is True
    assert prepared.summary["writes_performed"] is False
    assert prepared.summary["parquet_files_added"] == 0
    assert prepared.summary["economic_rows_changed"] == 0
    assert prepared.summary["lifecycle_truth"]["coverage"] == {
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
    assert repository.objects.get("releases/current.json").data == before_release
    assert _pointer_bytes(repository) == before_pointers
    assert {
        dataset: set(repository.objects.list(f"datasets/{dataset}/versions"))
        for dataset in script.TARGET_DATASETS
    } == before_versions


def test_apply_writes_only_empty_child_manifests_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _fixture(tmp_path, monkeypatch)
    repository = state.repository
    prepared = script.prepare_repair(repository)
    before_inventories = prepared.inspection.data_inventories
    before_source = repository.read_frame(
        "source_archive", state.base_release.dataset_versions["source_archive"]
    )
    parent_file_bytes = {
        (dataset, item["version"], item["path"]): (
            repository.root
            / repository.version_prefix(dataset, item["version"])
            / item["path"]
        ).read_bytes()
        for dataset, inventory in before_inventories.items()
        for item in inventory["files"]
    }

    result = script.apply_repair(repository, prepared)

    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    release, _ = repository.current_release()
    assert release is not None
    for dataset in script.TARGET_DATASETS:
        manifest = repository.current_manifest(dataset)
        assert manifest is not None
        assert manifest.files == ()
        assert manifest.parent_version == state.base_release.dataset_versions[dataset]
        assert manifest.metadata["inherits_parent"] is True
        assert manifest.metadata["lifecycle_coverage"] == state.truth_bindings[
            "lifecycle_coverage"
        ]
        assert "needs_lifecycle_refinalization" not in manifest.metadata
        marker = manifest.metadata["lifecycle_metadata_repair"]
        assert marker["parquet_files_added"] == 0
        assert marker["economic_rows_changed"] == 0
        assert marker["data_inventory_sha256"] == before_inventories[dataset][
            "sha256"
        ]
    pd.testing.assert_frame_equal(
        repository.read_frame("source_archive", release.dataset_versions["source_archive"]),
        before_source,
    )
    for key, payload in parent_file_bytes.items():
        dataset, version, relative = key
        assert (
            repository.root
            / repository.version_prefix(dataset, version)
            / relative
        ).read_bytes() == payload

    replay = script.prepare_repair(repository)
    assert replay.summary["status"] == "already_repaired"
    assert replay.summary["writes_required"] is False
    assert replay.summary["writes_performed"] is False
    assert script.apply_repair(repository, replay)["writes_performed"] is False


@pytest.mark.parametrize(
    "failure_stage",
    ["after_security_master_pointer", "after_release_commit"],
)
def test_apply_failure_rolls_back_exact_pointer_and_release_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    state = _fixture(tmp_path, monkeypatch)
    repository = state.repository
    prepared = script.prepare_repair(repository)
    old_release = repository.objects.get("releases/current.json").data
    old_pointers = _pointer_bytes(repository)

    def fail(stage: str) -> None:
        if stage == failure_stage:
            raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        script.apply_repair(repository, prepared, failure_injector=fail)

    assert repository.objects.get("releases/current.json").data == old_release
    assert _pointer_bytes(repository) == old_pointers
    assert not (
        repository.root / f"releases/{prepared.planned_release.version}.json"
    ).exists()
    for dataset, manifest in prepared.planned_manifests.items():
        assert not (
            repository.root / repository.version_prefix(dataset, manifest.version)
        ).exists()
    recovery = repository.root / "recovery/lifecycle-manifest-metadata"
    assert not recovery.exists() or not tuple(recovery.glob("*.json"))
    journals = tuple(
        (repository.root / "transactions/lifecycle-manifest-metadata").glob("*.json")
    )
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["status"] == "rolled_back"


def test_apply_rejects_toctou_report_object_change_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _fixture(tmp_path, monkeypatch)
    repository = state.repository
    prepared = script.prepare_repair(repository)
    old_release = repository.objects.get("releases/current.json").data
    old_pointers = _pointer_bytes(repository)
    report_path = repository.root / prepared.inspection.report_object.key
    payload = gzip.decompress(report_path.read_bytes())
    # Same payload/provenance hash but different exact stored bytes.
    report_path.write_bytes(gzip.compress(payload, mtime=1))

    with pytest.raises(RuntimeError, match="changed after lifecycle metadata planning"):
        script.apply_repair(repository, prepared)

    assert repository.objects.get("releases/current.json").data == old_release
    assert _pointer_bytes(repository) == old_pointers
    for dataset, manifest in prepared.planned_manifests.items():
        assert not (
            repository.root / repository.version_prefix(dataset, manifest.version)
        ).exists()


def test_partial_correct_metadata_state_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _fixture(tmp_path, monkeypatch)
    repository = state.repository
    dataset = "security_master"
    version = state.base_release.dataset_versions[dataset]
    key = f"{repository.version_prefix(dataset, version)}/manifest.json"
    manifest = DatasetManifest.from_bytes(repository.objects.get(key).data)
    metadata = dict(manifest.metadata)
    metadata.update(state.truth_bindings)
    metadata.pop("needs_lifecycle_refinalization", None)
    changed = DatasetManifest.from_dict({**manifest.to_dict(), "metadata": metadata})
    (repository.root / key).write_bytes(changed.to_bytes())
    pointer = CurrentPointer.create(changed, key)
    (repository.root / repository.current_key(dataset)).write_bytes(pointer.to_bytes())
    target_hashes = dict(script.EXPECTED_TARGET_PARENT_MANIFEST_SHA256)
    target_hashes[dataset] = sha256_bytes(changed.to_bytes())
    monkeypatch.setattr(
        script, "EXPECTED_TARGET_PARENT_MANIFEST_SHA256", target_hashes
    )

    with pytest.raises(RuntimeError, match="Partial lifecycle metadata repair state"):
        script.prepare_repair(repository)


def test_report_object_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _fixture(tmp_path, monkeypatch)
    repository = state.repository
    report_path = repository.root / (
        f"archives/{COMPLETED_SESSION}/{state.report_sha256}.json.gz"
    )
    report_path.write_bytes(gzip.compress(b"{}\n", mtime=0))
    with pytest.raises(RuntimeError, match="payload hash changed"):
        script.prepare_repair(repository)


def test_truth_resolution_file_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _fixture(tmp_path, monkeypatch)
    repository = state.repository
    manifest = repository.manifest_for_version(
        "lifecycle_resolutions", script.EXPECTED_TRUTH_VERSION
    )
    assert len(manifest.files) == 1
    path = (
        repository.root
        / repository.version_prefix(
            "lifecycle_resolutions", script.EXPECTED_TRUTH_VERSION
        )
        / manifest.files[0].path
    )
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        script.prepare_repair(repository)


def test_script_has_no_provider_or_r2_write_path() -> None:
    source = SCRIPT_PATH.read_text()
    assert "requests." not in source
    assert "boto3" not in source
    assert "EODHD_API_KEY" not in source
    assert "write_frame(" not in source
    assert "append_frame(" not in source
    assert "parquet_files_added\": 0" in source
