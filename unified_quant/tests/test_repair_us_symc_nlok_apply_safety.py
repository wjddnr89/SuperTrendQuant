from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
)


PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT_PATH = PROJECT_ROOT / "unified_quant/scripts/repair_us_symc_nlok_identity.py"
SPEC = importlib.util.spec_from_file_location(
    "repair_us_symc_nlok_identity_apply_safety_test", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


class _MemoryObjects:
    def __init__(self, values: dict[str, bytes]):
        self.values = dict(values)
        self.puts: list[str] = []

    def get(self, key: str):
        if key not in self.values:
            raise script.ObjectNotFound(key)
        data = self.values[key]
        return SimpleNamespace(data=data, etag=sha256_bytes(data))

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
    ):
        if if_none_match and key in self.values:
            raise script.ConditionalWriteFailed(key)
        if if_match is not None:
            if key not in self.values or self.get(key).etag != if_match:
                raise script.ConditionalWriteFailed(key)
        self.values[key] = data
        self.puts.append(key)
        return self.get(key)


def _pointer(
    dataset: str,
    version: str,
    *,
    manifest_sha256: str = "a" * 64,
    updated_at: str = "2026-07-19T00:00:00Z",
) -> bytes:
    return CurrentPointer(
        dataset=dataset,
        version=version,
        manifest_path=f"datasets/{dataset}/versions/{version}/manifest.json",
        manifest_sha256=manifest_sha256,
        updated_at=updated_at,
    ).to_bytes()


def _rollback_state():
    old_versions = {dataset: f"old-{dataset}" for dataset in script.WRITE_DATASETS}
    new_versions = {dataset: f"new-{dataset}" for dataset in script.WRITE_DATASETS}
    old_release = DataRelease(
        version="old-release",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions=old_versions,
    )
    planned_release = DataRelease(
        version="planned-release",
        created_at="2026-07-19T00:01:00Z",
        completed_session="2026-07-15",
        dataset_versions=new_versions,
    )
    old_pointers = {
        dataset: _pointer(dataset, old_versions[dataset])
        for dataset in script.WRITE_DATASETS
    }
    owned_pointers = {
        dataset: _pointer(dataset, new_versions[dataset])
        for dataset in script.WRITE_DATASETS
    }
    values = {"releases/current.json": planned_release.to_bytes()}
    values.update(
        {
            f"datasets/{dataset}/current.json": owned_pointers[dataset]
            for dataset in script.WRITE_DATASETS
        }
    )
    objects = _MemoryObjects(values)
    repository = SimpleNamespace(
        objects=objects,
        current_key=lambda dataset: f"datasets/{dataset}/current.json",
    )
    return (
        repository,
        old_release,
        planned_release,
        old_pointers,
        owned_pointers,
    )


def test_rollback_preflights_every_value_before_any_restore() -> None:
    repository, old, planned, old_pointers, owned_pointers = _rollback_state()
    errors = script._restore_transaction(
        repository,
        old_release_bytes=old.to_bytes(),
        old_pointer_bytes=old_pointers,
        planned_release_bytes=planned.to_bytes(),
        owned_pointer_bytes=owned_pointers,
    )
    assert errors == ()
    assert repository.objects.values["releases/current.json"] == old.to_bytes()
    assert len(repository.objects.puts) == 1 + len(script.WRITE_DATASETS)
    for dataset in script.WRITE_DATASETS:
        assert repository.objects.values[
            f"datasets/{dataset}/current.json"
        ] == old_pointers[dataset]


@pytest.mark.parametrize("foreign_kind", ["release", "pointer"])
def test_rollback_foreign_state_is_left_byte_for_byte_untouched(
    foreign_kind: str,
) -> None:
    repository, old, planned, old_pointers, owned_pointers = _rollback_state()
    if foreign_kind == "release":
        # Same release version and dataset map, different metadata: version-only
        # ownership would incorrectly roll this foreign publication back.
        foreign = DataRelease(
            version=planned.version,
            created_at="2026-07-19T00:02:00Z",
            completed_session=planned.completed_session,
            dataset_versions=planned.dataset_versions,
            warnings=("foreign publication",),
        )
        repository.objects.values["releases/current.json"] = foreign.to_bytes()
    else:
        dataset = script.WRITE_DATASETS[-1]
        repository.objects.values[f"datasets/{dataset}/current.json"] = _pointer(
            dataset,
            planned.dataset_versions[dataset],
            manifest_sha256="f" * 64,
            updated_at="2026-07-19T00:02:00Z",
        )
    before = dict(repository.objects.values)
    repository.objects.puts.clear()
    errors = script._restore_transaction(
        repository,
        old_release_bytes=old.to_bytes(),
        old_pointer_bytes=old_pointers,
        planned_release_bytes=planned.to_bytes(),
        owned_pointer_bytes=owned_pointers,
    )
    assert errors and errors[0].startswith("rollback preflight:")
    assert repository.objects.puts == []
    assert repository.objects.values == before


def test_lifecycle_archive_tracks_creation_and_removes_only_owned_payload(
    tmp_path: Path,
) -> None:
    payload = b'{"report":"symc-nlok"}\n'
    object_path = f"archives/2026-07-15/{sha256_bytes(payload)}.json.gz"
    repository = SimpleNamespace(root=tmp_path)
    assert script._persist_lifecycle_report(
        repository, object_path=object_path, content=payload
    )
    assert not script._persist_lifecycle_report(
        repository, object_path=object_path, content=payload
    )
    assert script._remove_created_lifecycle_report(
        repository,
        object_path=object_path,
        content=payload,
        created=True,
    ) == ()
    assert not (tmp_path / object_path).exists()


class _FakeManifest:
    def __init__(self, dataset: str, version: str):
        self.dataset = dataset
        self.version = version
        self.metadata: dict[str, object] = {}
        self.files = ()

    def to_bytes(self) -> bytes:
        return (
            json.dumps(
                {"dataset": self.dataset, "version": self.version},
                sort_keys=True,
            )
            + "\n"
        ).encode()


class _FakeRepository:
    def __init__(self, root: Path, release: DataRelease):
        self.root = root
        self.write_calls: list[str] = []
        values = {"releases/current.json": release.to_bytes()}
        values.update(
            {
                self.current_key(dataset): _pointer(dataset, version)
                for dataset, version in release.dataset_versions.items()
            }
        )
        self.objects = _MemoryObjects(values)

    @staticmethod
    def current_key(dataset: str) -> str:
        return f"datasets/{dataset}/current.json"

    @staticmethod
    def version_prefix(dataset: str, version: str) -> str:
        return f"datasets/{dataset}/versions/{version}"

    def current_release(self):
        value = self.objects.get("releases/current.json")
        return DataRelease.from_bytes(value.data), value.etag

    def current_pointer(self, dataset: str):
        value = self.objects.get(self.current_key(dataset))
        return CurrentPointer.from_bytes(value.data), value.etag

    def manifest_for_version(self, dataset: str, version: str):
        return _FakeManifest(dataset, version)

    def write_frame(self, dataset: str, _frame: pd.DataFrame, **kwargs):
        self.write_calls.append(dataset)
        version = kwargs["version"]
        manifest = _FakeManifest(dataset, version)
        pointer = _pointer(
            dataset,
            version,
            manifest_sha256=sha256_bytes(manifest.to_bytes()),
        )
        self.objects.put(
            self.current_key(dataset),
            pointer,
            if_match=kwargs["expected_pointer_etag"],
        )
        return SimpleNamespace(conflict=False, manifest=manifest)


def _synthetic_apply(tmp_path: Path):
    old_versions = {dataset: f"old-{dataset}" for dataset in script.WRITE_DATASETS}
    planned_versions = {
        dataset: f"planned-{dataset}" for dataset in script.WRITE_DATASETS
    }
    base = DataRelease(
        version="base-release",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions=old_versions,
    )
    planned = DataRelease(
        version="planned-release",
        created_at="2026-07-19T00:01:00Z",
        completed_session=base.completed_session,
        dataset_versions=planned_versions,
    )
    repository = _FakeRepository(tmp_path, base)
    _release, release_etag = repository.current_release()
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in script.WRITE_DATASETS
    }
    payload = b'{"report":"transaction evidence"}\n'
    object_path = f"archives/2026-07-15/{sha256_bytes(payload)}.json.gz"
    prepared = script.PreparedRepair(
        release=base,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        version_state={},
        frames={
            dataset: pd.DataFrame({"fixture": [dataset]})
            for dataset in script.WRITE_DATASETS
        },
        planned_versions=planned_versions,
        planned_release=planned,
        lifecycle_report_content=payload,
        lifecycle_report_object_path=object_path,
        lifecycle_metadata={
            "parent_release_kind": "identity_price_tails_descendant",
            "lifecycle_hints_sha256": script.CURRENT_LIFECYCLE_HINTS_SHA256,
            "full_lifecycle_finalizer_gate_passed": True,
        },
        warnings=(),
        summary={
            "status": "validated_offline_plan",
            "expected_output_row_counts": {
                dataset: (
                    2_095_793
                    if dataset in script.HEAVY_WRITE_DATASETS
                    else 1
                )
                for dataset in script.WRITE_DATASETS
            },
            "plan_materialization": (
                "affected_heavy_tables_plus_full_small_tables"
            ),
            "full_market_price_frames_materialized": False,
        },
        candidate_frames={
            dataset: pd.DataFrame()
            for dataset in script.LIFECYCLE_CANDIDATE_DATASETS
        },
    )
    return repository, prepared, object_path, payload


def _install_fake_heavy_writer(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def write(repository, prepared, dataset, *, metadata):
        calls.append(dataset)
        version = prepared.planned_versions[dataset]
        manifest = _FakeManifest(dataset, version)
        pointer = _pointer(
            dataset,
            version,
            manifest_sha256=sha256_bytes(manifest.to_bytes()),
        )
        repository.objects.put(
            repository.current_key(dataset),
            pointer,
            if_match=prepared.pointer_etags[dataset],
        )
        return SimpleNamespace(conflict=False, manifest=manifest)

    monkeypatch.setattr(script, "_write_prepared_heavy_dataset", write)
    return calls


def test_normal_prewrite_failure_removes_new_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, prepared, object_path, _payload = _synthetic_apply(tmp_path)
    monkeypatch.setattr(script, "_assert_inputs_unchanged", lambda *_: None)

    def inject(stage: str) -> None:
        if stage == "after_archive_payload":
            raise RuntimeError("injected prewrite failure")

    with pytest.raises(RuntimeError, match="injected prewrite failure"):
        script.apply_repair(repository, prepared, failure_injector=inject)
    assert not (tmp_path / object_path).exists()
    assert not tuple((tmp_path / script.RECOVERY_DIR).glob("*.json"))
    journals = tuple((tmp_path / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_bytes())["status"] == "rolled_back"


@pytest.mark.parametrize("corrupt_after_write", [False, True])
def test_archive_write_then_error_is_cleaned_or_blocks_with_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_after_write: bool,
) -> None:
    repository, prepared, object_path, payload = _synthetic_apply(tmp_path)
    monkeypatch.setattr(script, "_assert_inputs_unchanged", lambda *_: None)

    def persist_then_fail(_repository, *, object_path: str, content: bytes):
        destination = script._lifecycle_report_destination(
            repository,
            object_path=object_path,
            content=content,
        )
        encoded = gzip.compress(
            b"corrupt" if corrupt_after_write else content,
            mtime=0,
        )
        script.write_atomic(destination, encoded)
        raise RuntimeError("injected archive readback failure")

    monkeypatch.setattr(script, "_persist_lifecycle_report", persist_then_fail)
    if corrupt_after_write:
        with pytest.raises(RuntimeError, match="rollback failed"):
            script.apply_repair(repository, prepared)
        archive = tmp_path / object_path
        assert archive.is_file()
        assert gzip.decompress(archive.read_bytes()) == b"corrupt"
        recovery = tuple((tmp_path / script.RECOVERY_DIR).glob("*.json"))
        assert len(recovery) == 1
        assert json.loads(recovery[0].read_bytes())["status"] == "rollback_failed"
    else:
        with pytest.raises(RuntimeError, match="injected archive readback failure"):
            script.apply_repair(repository, prepared)
        assert not (tmp_path / object_path).exists()
        assert not tuple((tmp_path / script.RECOVERY_DIR).glob("*.json"))
    assert payload == prepared.lifecycle_report_content


def test_foreign_postcommit_release_is_preserved_with_archive_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, prepared, object_path, payload = _synthetic_apply(tmp_path)
    monkeypatch.setattr(script, "_assert_inputs_unchanged", lambda *_: None)
    monkeypatch.setattr(script, "_verify_written_publication", lambda *_: None)
    heavy_calls = _install_fake_heavy_writer(monkeypatch)
    captured: dict[str, object] = {}

    def inject(stage: str) -> None:
        if stage != "after_commit":
            return
        current = repository.objects.get("releases/current.json")
        assert prepared.planned_release is not None
        foreign = DataRelease(
            version=prepared.planned_release.version,
            created_at="2026-07-19T00:02:00Z",
            completed_session=prepared.planned_release.completed_session,
            dataset_versions=prepared.planned_release.dataset_versions,
            warnings=("foreign publication",),
        )
        repository.objects.put(
            "releases/current.json",
            foreign.to_bytes(),
            if_match=current.etag,
        )
        captured["release"] = foreign.to_bytes()
        captured["pointers"] = {
            dataset: repository.objects.get(repository.current_key(dataset)).data
            for dataset in script.WRITE_DATASETS
        }
        raise RuntimeError("injected foreign publication")

    with pytest.raises(RuntimeError, match="rollback failed"):
        script.apply_repair(repository, prepared, failure_injector=inject)
    assert repository.objects.get("releases/current.json").data == captured["release"]
    for dataset, pointer in captured["pointers"].items():
        assert repository.objects.get(repository.current_key(dataset)).data == pointer
    archive = tmp_path / object_path
    assert archive.is_file() and gzip.decompress(archive.read_bytes()) == payload
    recovery = tuple((tmp_path / script.RECOVERY_DIR).glob("*.json"))
    assert len(recovery) == 1
    assert json.loads(recovery[0].read_bytes())["status"] == "rollback_failed"
    assert set(heavy_calls) == set(script.HEAVY_WRITE_DATASETS)


def test_successful_apply_does_not_reenter_full_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, prepared, object_path, payload = _synthetic_apply(tmp_path)
    monkeypatch.setattr(script, "_assert_inputs_unchanged", lambda *_: None)
    monkeypatch.setattr(script, "_verify_written_publication", lambda *_: None)
    heavy_calls = _install_fake_heavy_writer(monkeypatch)
    monkeypatch.setattr(
        script,
        "prepare_repair",
        lambda *_: pytest.fail("post-commit apply must not re-enter prepare_repair"),
    )
    result = script.apply_repair(repository, prepared)
    assert result["status"] == "applied"
    assert set(heavy_calls) == set(script.HEAVY_WRITE_DATASETS)
    assert set(repository.write_calls) == (
        set(script.WRITE_DATASETS) - set(script.HEAVY_WRITE_DATASETS)
    )
    archive = tmp_path / object_path
    assert archive.is_file() and gzip.decompress(archive.read_bytes()) == payload


def test_written_scope_reader_never_full_reads_price_or_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subset_calls: list[str] = []
    full_calls: list[str] = []

    def subset(_repository, dataset, _version, _security_ids):
        subset_calls.append(dataset)
        return pd.DataFrame()

    repository = SimpleNamespace(
        read_frame=lambda dataset, _version: (
            full_calls.append(dataset) or pd.DataFrame()
        )
    )
    monkeypatch.setattr(script.identity_tails, "_read_security_subset", subset)
    versions = {dataset: f"v-{dataset}" for dataset in script.WRITE_DATASETS}
    frames = script._written_identity_scopes(repository, versions)
    assert set(subset_calls) == set(script.IDENTITY_WRITE_DATASETS)
    assert full_calls == ["source_archive"]
    assert set(frames) == set(script.WRITE_DATASETS)
