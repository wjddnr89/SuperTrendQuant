from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.manifest import CurrentPointer, DataRelease
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_wiki14_price_only.py"
)
SPEC = importlib.util.spec_from_file_location("repair_us_wiki14_price_only", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


ARCHIVE_COLUMNS = [
    "archive_id",
    "dataset",
    "object_path",
    "content_type",
    "effective_date",
    "source",
    "retrieved_at",
    "source_hash",
    "source_url",
]


def _artifact(name: str, payload: bytes) -> script.ArchiveArtifact:
    return script.ArchiveArtifact(
        dataset=name,
        source=name,
        source_url="https://example.test/frozen",
        content_type="application/json",
        extension="json",
        payload=payload,
        retrieved_at="2026-07-19T00:00:00Z",
    )


def test_exact_14_identity_inventory_and_aggregate_pins() -> None:
    assert [target.symbol for target in script.TARGETS] == [
        "ADT", "CAM", "COL", "EMC", "EVHC", "FB", "FOX", "FOXA",
        "INFO", "NFX", "SCG", "SNDK", "STI", "TE",
    ]
    assert len({target.security_id for target in script.TARGETS}) == 14
    assert len({target.target_id for target in script.TARGETS}) == 14
    assert sum(target.full_wiki_rows for target in script.TARGETS) == 67_867
    assert sum(target.overlap_rows for target in script.TARGETS) == 8_499
    assert all(target.provider_symbol for target in script.TARGETS)
    assert all(len(target.relation_sha256) == 64 for target in script.TARGETS)
    assert all(len(target.signal_sha256) == 64 for target in script.TARGETS)
    assert (
        script._canonical_sha(script._extract_inventory(script.TARGETS))
        == script.EXTRACT_INVENTORY_SHA256
    )
    assert len(script.PROVENANCE_SHA256) == 64
    assert len(script.ARCHIVE_ARTIFACT_INVENTORY_SHA256) == 64
    assert set(script.IDENTITY_SCHEMA_PINS) == {
        target.symbol for target in script.TARGETS
    }
    assert (
        script._canonical_sha(script._identity_schema_inventory())
        == script.IDENTITY_SCHEMA_INVENTORY_SHA256
    )


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("master", "primary_symbol", "WRONG"),
        ("master", "exchange", "NASDAQ"),
        ("master", "asset_type", "ETF"),
        ("master", "currency", "KRW"),
        ("master", "country", "KR"),
        ("history", "symbol", "WRONG"),
        ("history", "exchange", "NASDAQ"),
    ],
)
def test_identity_schema_mutations_fail_closed(
    owner: str,
    field: str,
    value: str,
) -> None:
    target = script.TARGETS[0]
    master = {
        "primary_symbol": "ADT",
        "exchange": "NYSE",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
    }
    history = {"symbol": "ADT", "exchange": "NYSE"}
    (master if owner == "master" else history)[field] = value
    with pytest.raises(ValueError, match="exact identity schema changed"):
        script._assert_identity_schema(target, master, history)


def test_raw_price_currency_is_exact_and_fail_closed() -> None:
    target = script.TARGETS[0]
    assert script._assert_raw_price_currency(
        target, pd.DataFrame({"currency": ["USD", "USD"]})
    ) == "USD"
    for values in (["KRW"], ["USD", "KRW"], [None]):
        with pytest.raises(ValueError, match="raw price currency changed"):
            script._assert_raw_price_currency(
                target, pd.DataFrame({"currency": values})
            )


def test_policy_is_exact_price_only_and_unknown_license_fail_closed() -> None:
    assert "licenseName=Unknown" in script.WIKI_LICENSE_WARNING
    assert "private/internal-only" in script.WIKI_LICENSE_WARNING
    assert "redistribution/public publication blocked" in script.WIKI_LICENSE_WARNING
    assert script.DATASET == "source_archive"
    assert script.REQUIRED_DATASETS == (
        "daily_price_raw", "corporate_actions", "adjustment_factors",
        "security_master", "symbol_history", "source_archive",
    )
    source = SCRIPT_PATH.read_text()
    assert "validate_repository_snapshot" not in source
    assert '"duckdb_exact_14_security_subset"' in source


def test_partial_and_tampered_archive_fail_closed_and_preserve_old_rows(
    tmp_path: Path,
) -> None:
    repository = LocalDatasetRepository(tmp_path / "cache")
    old = _artifact("old", b"old\n")
    first = _artifact("first", b"first\n")
    second = _artifact("second", b"second\n")
    old_path = repository.root / old.object_path("2026-07-15")
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_bytes(gzip.compress(old.payload, mtime=0))
    archive = pd.DataFrame(
        [
            script._artifact_row(
                old, columns=ARCHIVE_COLUMNS
            )
        ],
        columns=ARCHIVE_COLUMNS,
    )
    candidate, changed = script._append_or_verify_artifacts(
        repository,
        archive,
        (first, second),
    )
    assert changed is True
    pd.testing.assert_frame_equal(candidate.iloc[:1].reset_index(drop=True), archive)

    first_path = repository.root / first.object_path("2026-07-15")
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_bytes(gzip.compress(first.payload, mtime=0))
    partial = pd.concat(
        [
            archive,
            pd.DataFrame(
                [
                    script._artifact_row(
                        first,
                        columns=ARCHIVE_COLUMNS,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="partially archived"):
        script._append_or_verify_artifacts(
            repository,
            partial,
            (first, second),
        )
    first_path.write_bytes(gzip.compress(b"tampered\n", mtime=0))
    with pytest.raises(ValueError, match="payload hash changed"):
        script._append_or_verify_artifacts(
            repository,
            partial,
            (first,),
        )


def test_artifact_binding_is_stable_across_descendant_release_dates(
    tmp_path: Path,
) -> None:
    repository = LocalDatasetRepository(tmp_path / "cache")
    artifact = _artifact("stable", b"stable\n")
    row = script._artifact_row(artifact, columns=ARCHIVE_COLUMNS)
    path = repository.root / artifact.object_path(script.ARCHIVE_EFFECTIVE_DATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(artifact.payload, mtime=0))

    assert row["effective_date"] == "2026-07-15"
    assert "/2026-07-15/" in row["object_path"]
    script._verify_artifact_row(repository, row, artifact)
    candidate, changed = script._append_or_verify_artifacts(
        repository,
        pd.DataFrame([row], columns=ARCHIVE_COLUMNS),
        (artifact,),
    )
    assert changed is False
    assert len(candidate) == 1


def test_orphan_payload_is_reused_only_when_exact(tmp_path: Path) -> None:
    repository = LocalDatasetRepository(tmp_path / "cache")
    artifact = _artifact("orphan", b"orphan\n")
    path = repository.root / artifact.object_path(script.ARCHIVE_EFFECTIVE_DATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    exact_bytes = gzip.compress(artifact.payload, mtime=0)
    path.write_bytes(exact_bytes)

    script._write_artifact(repository, artifact)
    assert path.read_bytes() == exact_bytes

    path.write_bytes(gzip.compress(b"conflict\n", mtime=0))
    with pytest.raises(ValueError, match="bytes conflict"):
        script._write_artifact(repository, artifact)


def test_existing_bbby_bbt_metadata_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = LocalDatasetRepository(tmp_path / "cache")
    rows = []
    for digest, values in script.BBBY_BBT_ARTIFACT_PINS.items():
        rows.append({"archive_id": digest, **values})
    archive = pd.DataFrame(rows)
    monkeypatch.setattr(script, "_read_archived_payload", lambda *args: b"pinned")
    script._verify_bbby_bbt_evidence(repository, archive)

    archive.loc[0, "source_url"] = "https://wrong.test"
    with pytest.raises(ValueError, match="metadata changed: source_url"):
        script._verify_bbby_bbt_evidence(repository, archive)


@pytest.mark.parametrize(
    ("dataset", "value_column", "updated_value"),
    [
        ("daily_price_raw", "close", 11.5),
        ("adjustment_factors", "split_factor", 2.0),
    ],
)
def test_heavy_subset_reader_filters_ids_and_prefers_descendant_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    value_column: str,
    updated_value: float,
) -> None:
    repository = LocalDatasetRepository(tmp_path / "cache")
    common = {
        "session": "2020-01-02",
        "source": "fixture",
        "retrieved_at": "2026-07-19T00:00:00Z",
        "source_hash": "a" * 64,
    }
    if dataset == "daily_price_raw":
        values = {
            "open": 10.0,
            "high": 12.0,
            "low": 9.0,
            "close": 11.0,
            "volume": 100.0,
            "currency": "USD",
        }
    else:
        values = {
            "split_factor": 1.0,
            "total_return_factor": 1.0,
            "source_version": "fixture",
            "calculated_at": "2026-07-19T00:00:00Z",
        }
    base = pd.DataFrame(
        [
            {"security_id": "target", **common, **values},
            {"security_id": "other", **common, **values},
        ]
    )
    first = repository.write_frame(
        dataset,
        base,
        completed_session="2020-01-02",
        version=f"base-{dataset}",
    )
    _, etag = repository.current_pointer(dataset)
    delta_values = dict(values)
    delta_values[value_column] = updated_value
    delta = pd.DataFrame(
        [{"security_id": "target", **common, **delta_values}]
    )
    second = repository.write_frame(
        dataset,
        delta,
        completed_session="2020-01-02",
        expected_pointer_etag=etag,
        version=f"descendant-{dataset}",
        inherit_parent=True,
    )
    monkeypatch.setattr(
        repository,
        "read_frame",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("full-frame read is forbidden")
        ),
    )

    observed = script._read_security_subset(
        repository, dataset, second.manifest.version, ("target",)
    )
    assert len(observed) == 1
    assert observed.iloc[0]["security_id"] == "target"
    assert float(observed.iloc[0][value_column]) == updated_value
    assert first.manifest.version != second.manifest.version


def _transaction_fixture(tmp_path: Path):
    repository = LocalDatasetRepository(tmp_path / "cache")
    base_artifact = _artifact("base", b"base\n")
    base_frame = pd.DataFrame(
        [
            script._artifact_row(
                base_artifact,
                columns=ARCHIVE_COLUMNS,
            )
        ],
        columns=ARCHIVE_COLUMNS,
    )
    initial = repository.write_frame(
        "source_archive",
        base_frame,
        completed_session="2026-07-15",
        version="base-source-archive",
    )
    repository.commit_release(
        "2026-07-15",
        {"source_archive": initial.manifest.version},
        quality=DataQuality.VALID,
        warnings=(),
    )
    release, release_etag = repository.current_release()
    assert release is not None
    pointer_etags, version_state = script._version_state(repository, release)
    new_artifact = _artifact("new", b"new\n")
    candidate = pd.concat(
        [
            base_frame,
            pd.DataFrame(
                [
                    script._artifact_row(
                        new_artifact,
                        columns=ARCHIVE_COLUMNS,
                    )
                ],
                columns=ARCHIVE_COLUMNS,
            ),
        ],
        ignore_index=True,
    )
    planned_version = "wiki14-fixture-source-archive"
    planned_versions = dict(release.dataset_versions)
    planned_versions["source_archive"] = planned_version
    planned_release = DataRelease.create(
        release.completed_session,
        planned_versions,
        quality=release.quality,
        warnings=(script.WIKI_LICENSE_WARNING,),
    )
    prepared = script.PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        version_state=version_state,
        frame=candidate,
        artifacts=(new_artifact,),
        wiki_zip_path=Path("fixture.zip"),
        targets=script.TARGETS,
        allowed_index_identity_gap_fingerprints=(),
        planned_source_archive_version=planned_version,
        planned_release=planned_release,
        source_archive_inventory_sha256=(
            script._source_archive_inventory_sha256(candidate)
        ),
        summary={
            "status": "validated_offline_plan",
            "plan_read_mode": "duckdb_exact_14_security_subset",
            "full_market_price_frames_materialized": False,
        },
    )
    return repository, prepared


def test_apply_requires_ack_and_stale_release_cas_blocks_before_writes(
    tmp_path: Path,
) -> None:
    repository, prepared = _transaction_fixture(tmp_path)
    with pytest.raises(PermissionError, match="private_internal_only"):
        script.apply_repair(repository, prepared)
    release, etag = repository.current_release()
    assert release is not None
    repository.commit_release(
        release.completed_session,
        release.dataset_versions,
        quality=release.quality,
        warnings=release.warnings,
        expected_etag=etag,
    )
    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(
            repository,
            prepared,
            ack_private_internal_only_local_repair=True,
        )


def test_transaction_rollback_restores_release_and_pointer(
    tmp_path: Path,
) -> None:
    repository, prepared = _transaction_fixture(tmp_path)
    before_release = repository.objects.get("releases/current.json").data
    before_pointer = repository.objects.get(
        repository.current_key("source_archive")
    ).data
    def fail(stage: str) -> None:
        if stage == "after_release_commit":
            raise RuntimeError("injected WIKI14 rollback")

    with pytest.raises(RuntimeError, match="injected WIKI14 rollback"):
        script.apply_repair(
            repository,
            prepared,
            ack_private_internal_only_local_repair=True,
            inject_failure=fail,
        )
    assert repository.objects.get("releases/current.json").data == before_release
    assert (
        repository.objects.get(repository.current_key("source_archive")).data
        == before_pointer
    )
    journals = tuple((repository.root / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text())
    assert journal["status"] == "rolled_back"
    assert journal["rollback_errors"] == []
    artifact_path = repository.root / prepared.artifacts[0].object_path(
        script.ARCHIVE_EFFECTIVE_DATE
    )
    assert not artifact_path.exists()


@pytest.mark.parametrize("foreign_object", ["release", "pointer"])
def test_foreign_publication_is_not_mutated_and_creates_recovery_marker(
    tmp_path: Path,
    foreign_object: str,
) -> None:
    repository, prepared = _transaction_fixture(tmp_path)
    foreign_bytes: dict[str, bytes] = {}

    def fail(stage: str) -> None:
        if stage != "after_release_commit":
            return
        if foreign_object == "release":
            current = repository.objects.get("releases/current.json")
            value = DataRelease.from_bytes(current.data)
            foreign = replace(value, version=value.version + "-foreign").to_bytes()
            repository.objects.put(
                "releases/current.json", foreign, if_match=current.etag
            )
            foreign_bytes["key"] = foreign
        else:
            key = repository.current_key("source_archive")
            current = repository.objects.get(key)
            value = CurrentPointer.from_bytes(current.data)
            foreign = replace(value, version=value.version + "-foreign").to_bytes()
            repository.objects.put(key, foreign, if_match=current.etag)
            foreign_bytes["key"] = foreign
        raise RuntimeError("foreign publication injected")

    with pytest.raises(RuntimeError, match="rollback incomplete"):
        script.apply_repair(
            repository,
            prepared,
            ack_private_internal_only_local_repair=True,
            inject_failure=fail,
        )

    key = (
        "releases/current.json"
        if foreign_object == "release"
        else repository.current_key("source_archive")
    )
    assert repository.objects.get(key).data == foreign_bytes["key"]
    recovery = tuple((repository.root / script.RECOVERY_DIR).glob("*.json"))
    assert len(recovery) == 1
    assert json.loads(recovery[0].read_text())["status"] == "rollback_failed"
    artifact_path = repository.root / prepared.artifacts[0].object_path(
        script.ARCHIVE_EFFECTIVE_DATE
    )
    assert artifact_path.is_file()


def test_created_archive_change_fails_cleanup_and_blocks_with_recovery(
    tmp_path: Path,
) -> None:
    repository, prepared = _transaction_fixture(tmp_path)
    artifact_path = repository.root / prepared.artifacts[0].object_path(
        script.ARCHIVE_EFFECTIVE_DATE
    )

    def fail(stage: str) -> None:
        if stage == "after_release_commit":
            artifact_path.write_bytes(gzip.compress(b"foreign\n", mtime=0))
            raise RuntimeError("archive mutation injected")

    with pytest.raises(RuntimeError, match="rollback incomplete"):
        script.apply_repair(
            repository,
            prepared,
            ack_private_internal_only_local_repair=True,
            inject_failure=fail,
        )
    assert gzip.decompress(artifact_path.read_bytes()) == b"foreign\n"
    assert len(tuple((repository.root / script.RECOVERY_DIR).glob("*.json"))) == 1


def test_created_archive_same_payload_different_gzip_is_foreign(
    tmp_path: Path,
) -> None:
    repository, prepared = _transaction_fixture(tmp_path)
    artifact = prepared.artifacts[0]
    artifact_path = repository.root / artifact.object_path(
        script.ARCHIVE_EFFECTIVE_DATE
    )
    foreign_storage_bytes = gzip.compress(artifact.payload, mtime=1)
    assert foreign_storage_bytes != script._stored_artifact_bytes(artifact)

    def fail(stage: str) -> None:
        if stage == "after_release_commit":
            artifact_path.write_bytes(foreign_storage_bytes)
            raise RuntimeError("same-payload archive replacement injected")

    with pytest.raises(RuntimeError, match="rollback incomplete"):
        script.apply_repair(
            repository,
            prepared,
            ack_private_internal_only_local_repair=True,
            inject_failure=fail,
        )
    assert artifact_path.read_bytes() == foreign_storage_bytes
    assert gzip.decompress(artifact_path.read_bytes()) == artifact.payload
    assert len(tuple((repository.root / script.RECOVERY_DIR).glob("*.json"))) == 1


def test_preexisting_exact_archive_survives_safe_rollback(tmp_path: Path) -> None:
    repository, prepared = _transaction_fixture(tmp_path)
    artifact = prepared.artifacts[0]
    artifact_path = repository.root / artifact.object_path(
        script.ARCHIVE_EFFECTIVE_DATE
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    exact = gzip.compress(artifact.payload, mtime=0)
    artifact_path.write_bytes(exact)

    def fail(stage: str) -> None:
        if stage == "after_release_commit":
            raise RuntimeError("safe rollback injected")

    with pytest.raises(RuntimeError, match="safe rollback injected"):
        script.apply_repair(
            repository,
            prepared,
            ack_private_internal_only_local_repair=True,
            inject_failure=fail,
        )
    assert artifact_path.read_bytes() == exact
    assert not tuple((repository.root / script.RECOVERY_DIR).glob("*.json"))


def test_manifest_file_change_is_rejected_before_journal_or_archive(
    tmp_path: Path,
) -> None:
    repository, prepared = _transaction_fixture(tmp_path)
    manifest = repository.manifest_for_version(
        "source_archive", prepared.release.dataset_versions["source_archive"]
    )
    parquet = (
        repository.root
        / repository.version_prefix("source_archive", manifest.version)
        / manifest.files[0].path
    )
    parquet.write_bytes(parquet.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="Size mismatch"):
        script.apply_repair(
            repository,
            prepared,
            ack_private_internal_only_local_repair=True,
        )
    assert not tuple((repository.root / script.TRANSACTION_DIR).glob("*.json"))
    artifact_path = repository.root / prepared.artifacts[0].object_path(
        script.ARCHIVE_EFFECTIVE_DATE
    )
    assert not artifact_path.exists()


def test_transaction_success_changes_only_source_archive_and_replays_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, prepared = _transaction_fixture(tmp_path)
    before_release, _ = repository.current_release()
    assert before_release is not None
    monkeypatch.setattr(
        script,
        "prepare_repair",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("postcommit plan replay is forbidden")
        ),
    )

    result = script.apply_repair(
        repository,
        prepared,
        ack_private_internal_only_local_repair=True,
    )

    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    after_release, after_release_etag = repository.current_release()
    assert after_release is not None
    assert after_release.quality == before_release.quality
    assert set(after_release.dataset_versions) == {"source_archive"}
    assert (
        after_release.dataset_versions["source_archive"]
        != before_release.dataset_versions["source_archive"]
    )
    assert script.WIKI_LICENSE_WARNING in after_release.warnings
    current_pointer_etags, current_version_state = script._version_state(
        repository, after_release
    )
    current_replay = replace(
        prepared,
        release=after_release,
        release_etag=after_release_etag,
        pointer_etags=current_pointer_etags,
        version_state=current_version_state,
        planned_source_archive_version="",
        planned_release=None,
        summary={"status": "already_applied"},
    )
    noop_result = script.apply_repair(repository, current_replay)
    assert noop_result["writes_performed"] is False


def test_already_applied_is_idempotent_noop_without_ack(tmp_path: Path) -> None:
    repository, prepared = _transaction_fixture(tmp_path)
    noop = replace(prepared, summary={"status": "already_applied"})
    before = repository.objects.get("releases/current.json").data
    result = script.apply_repair(repository, noop)
    assert result["writes_performed"] is False
    assert repository.objects.get("releases/current.json").data == before


def test_already_applied_stale_plan_is_rejected(tmp_path: Path) -> None:
    repository, prepared = _transaction_fixture(tmp_path)
    noop = replace(prepared, summary={"status": "already_applied"})
    release, etag = repository.current_release()
    assert release is not None
    repository.commit_release(
        release.completed_session,
        release.dataset_versions,
        quality=release.quality,
        warnings=release.warnings,
        expected_etag=etag,
    )
    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(repository, noop)


def test_provenance_explicitly_separates_action_and_factor_gaps() -> None:
    source = SCRIPT_PATH.read_text()
    assert "price_only_pass_must_not_imply_action_pass" in source
    assert "price_only_pass_must_not_imply_factor_pass" in source
    assert '"generic_symbol_or_ticker_exception_allowed": False' in source
    assert '"redistribution_allowed": False' in source
    assert '"public_publication_allowed": False' in source
