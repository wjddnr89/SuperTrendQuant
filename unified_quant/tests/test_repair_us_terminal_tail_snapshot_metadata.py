from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
    ManifestFile,
    sha256_bytes,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.validation import validate_manifest_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "unified_quant/scripts/repair_us_terminal_tail_snapshot_metadata.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_terminal_tail_snapshot_metadata", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

PUBLISH_PATH = PROJECT_ROOT / "unified_quant/scripts/publish_and_verify_r2.py"
PUBLISH_SPEC = importlib.util.spec_from_file_location(
    "publish_and_verify_r2_terminal_tail_metadata_test", PUBLISH_PATH
)
assert PUBLISH_SPEC is not None and PUBLISH_SPEC.loader is not None
publish_script = importlib.util.module_from_spec(PUBLISH_SPEC)
PUBLISH_SPEC.loader.exec_module(publish_script)


COMPLETED_SESSION = "2026-07-15"
BASE_RELEASE_VERSION = "fixture-terminal-tail-metadata-base"
PARTIAL_DATASETS = frozenset(
    {"daily_price_raw", "security_master", "symbol_history"}
)


@dataclass(frozen=True)
class FixtureState:
    repository: LocalDatasetRepository
    release: DataRelease
    parent_versions: dict[str, str]
    parent_hashes: dict[str, str]


def _put_parent_manifest(
    repository: LocalDatasetRepository,
    dataset: str,
    version: str,
    *,
    metadata: dict[str, object],
) -> DatasetManifest:
    payload = f"immutable-economic-bytes:{dataset}\n".encode()
    version_root = repository.root / repository.version_prefix(dataset, version)
    version_root.mkdir(parents=True, exist_ok=True)
    data_path = version_root / "part-00000.parquet"
    data_path.write_bytes(payload)
    manifest = DatasetManifest.create(
        dataset=dataset,
        version=version,
        completed_session=COMPLETED_SESSION,
        files=(
            ManifestFile(
                path="part-00000.parquet",
                sha256=sha256_bytes(payload),
                size_bytes=len(payload),
                row_count=1,
            ),
        ),
        metadata=metadata,
    )
    manifest_key = f"{repository.version_prefix(dataset, version)}/manifest.json"
    repository.objects.put(manifest_key, manifest.to_bytes(), if_none_match=True)
    repository.objects.put(
        repository.current_key(dataset),
        CurrentPointer.create(manifest, manifest_key).to_bytes(),
    )
    return manifest


def _write_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> None:
    repository.objects.put(
        f"releases/{release.version}.json", release.to_bytes(), if_none_match=True
    )
    repository.objects.put("releases/current.json", release.to_bytes())


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> FixtureState:
    repository = LocalDatasetRepository(tmp_path / "cache")
    repository.root.mkdir(parents=True)
    registry = script._code_pinned_target_registry()
    parent_versions = {
        dataset: f"fixture-parent-{dataset}" for dataset in script.TARGET_DATASETS
    }
    parent_hashes: dict[str, str] = {}
    for dataset in script.TARGET_DATASETS:
        metadata: dict[str, object] = {"fixture": True}
        if dataset in PARTIAL_DATASETS:
            metadata.update(
                {
                    script.REGISTRY_FIELD: registry,
                    script.REGISTRY_SHA_FIELD: (
                        script.TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
                    ),
                }
            )
        manifest = _put_parent_manifest(
            repository,
            dataset,
            parent_versions[dataset],
            metadata=metadata,
        )
        parent_hashes[dataset] = sha256_bytes(manifest.to_bytes())

    release = DataRelease(
        version=BASE_RELEASE_VERSION,
        created_at="2026-07-19T00:00:00Z",
        completed_session=COMPLETED_SESSION,
        dataset_versions=parent_versions,
        quality="degraded",
        warnings=("fixture warning",),
    )
    _write_release(repository, release)
    monkeypatch.setattr(script, "EXPECTED_BASE_RELEASE_VERSION", release.version)
    monkeypatch.setattr(
        script, "EXPECTED_BASE_RELEASE_SHA256", sha256_bytes(release.to_bytes())
    )
    monkeypatch.setattr(script, "EXPECTED_PARENT_VERSIONS", parent_versions)
    monkeypatch.setattr(
        script, "EXPECTED_PARENT_MANIFEST_SHA256", parent_hashes
    )
    return FixtureState(repository, release, parent_versions, parent_hashes)


def _parquet_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_bytes(path.read_bytes())
        for path in sorted(root.rglob("*.parquet"))
    }


def _pointer_bytes(
    repository: LocalDatasetRepository,
) -> dict[str, bytes]:
    return {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.TARGET_DATASETS
    }


def test_plan_is_read_only_and_removes_partial_snapshot_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    before_files = {
        str(path.relative_to(fixture.repository.root)): sha256_bytes(path.read_bytes())
        for path in sorted(fixture.repository.root.rglob("*"))
        if path.is_file()
    }
    prepared = script.prepare_repair(fixture.repository, plan_id="fixture-plan")
    after_files = {
        str(path.relative_to(fixture.repository.root)): sha256_bytes(path.read_bytes())
        for path in sorted(fixture.repository.root.rglob("*"))
        if path.is_file()
    }

    assert before_files == after_files
    assert prepared.summary["status"] == "validated_plan"
    assert prepared.summary["writes_performed"] is False
    assert prepared.summary["registry_present_before"] == sorted(PARTIAL_DATASETS)
    assert prepared.summary["registry_missing_before"] == sorted(
        set(script.TARGET_DATASETS) - PARTIAL_DATASETS
    )
    assert prepared.summary["manifest_only_child_count"] == 9
    assert prepared.summary["parquet_files_added"] == 0
    assert prepared.summary["parquet_bytes_added"] == 0
    assert prepared.summary["economic_rows_changed"] == 0
    assert len(prepared.summary["repair_plan_sha256"]) == 64
    assert all(not manifest.files for manifest in prepared.planned_manifests.values())
    assert all(
        manifest.parent_version == fixture.parent_versions[dataset]
        for dataset, manifest in prepared.planned_manifests.items()
    )

    class PlannedRepositoryView:
        def manifest_for_version(self, dataset: str, version: str):
            manifest = prepared.planned_manifests[dataset]
            assert version == manifest.version
            return manifest

    fingerprints = publish_script._terminal_tail_identity_gap_fingerprints(
        PlannedRepositoryView(), prepared.planned_release
    )
    assert fingerprints == (
        publish_script.TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS[
            next(
                iter(
                    publish_script.TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS
                )
            )
        ]["fingerprint"],
    )


def test_temp_apply_adds_no_parquet_preserves_inventory_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    before_parquet = _parquet_inventory(fixture.repository.root)
    prepared = script.prepare_repair(fixture.repository, plan_id="fixture-apply")
    result = script.apply_repair(fixture.repository, prepared)

    assert result["status"] == "applied"
    assert _parquet_inventory(fixture.repository.root) == before_parquet
    current, _ = fixture.repository.current_release()
    assert current is not None
    for dataset in script.TARGET_DATASETS:
        manifest = fixture.repository.manifest_for_version(
            dataset, current.dataset_versions[dataset]
        )
        assert manifest.files == ()
        assert manifest.parent_version == fixture.parent_versions[dataset]
        assert manifest.metadata[script.REGISTRY_SHA_FIELD] == (
            script.TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
        )
        validate_manifest_files(
            fixture.repository.root
            / fixture.repository.version_prefix(dataset, manifest.version),
            manifest,
        ).raise_for_errors()

    again = script.prepare_repair(fixture.repository, plan_id="unused")
    assert again.summary["status"] == "already_repaired"
    assert again.summary["writes_required"] is False
    assert again.summary["economic_rows_changed"] == 0


def test_temp_apply_rolls_back_release_and_all_pointers_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    before_release = fixture.repository.objects.get("releases/current.json").data
    before_pointers = _pointer_bytes(fixture.repository)
    before_parquet = _parquet_inventory(fixture.repository.root)
    prepared = script.prepare_repair(fixture.repository, plan_id="fixture-rollback")

    def fail(stage: str) -> None:
        if stage == "after_release_commit":
            raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        script.apply_repair(
            fixture.repository, prepared, failure_injector=fail
        )

    assert fixture.repository.objects.get("releases/current.json").data == before_release
    assert _pointer_bytes(fixture.repository) == before_pointers
    assert _parquet_inventory(fixture.repository.root) == before_parquet
    assert not tuple(
        (
            fixture.repository.root
            / "recovery/terminal-tail-snapshot-metadata"
        ).glob("*.json")
    )
    assert script.prepare_repair(
        fixture.repository, plan_id="fixture-after-rollback"
    ).summary["status"] == "validated_plan"


def test_temp_apply_fails_closed_when_pointer_bytes_drift_after_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    prepared = script.prepare_repair(fixture.repository, plan_id="fixture-cas")
    dataset = script.TARGET_DATASETS[0]
    key = fixture.repository.current_key(dataset)
    current = fixture.repository.objects.get(key)
    # Semantically identical JSON with different bytes proves the transaction
    # freezes pointer bytes/ETags rather than only comparing the version text.
    fixture.repository.objects.put(key, current.data.rstrip() + b" \n")
    before_parquet = _parquet_inventory(fixture.repository.root)

    with pytest.raises(RuntimeError, match="changed after.*planning"):
        script.apply_repair(fixture.repository, prepared)

    assert _parquet_inventory(fixture.repository.root) == before_parquet
    assert not any(
        fixture.repository.root.glob(
            "datasets/*/versions/terminal-tail-snapshot-metadata-*"
        )
    )
