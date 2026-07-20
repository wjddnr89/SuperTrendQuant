from __future__ import annotations

import dataclasses
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_formula_one_archive_url.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_formula_one_archive_url", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


COMPLETED_SESSION = "2026-07-15"
RETRIEVED_AT = "2026-07-18T00:00:00Z"


def _target(payload: bytes = b"reviewed Formula One SEC filing") -> script.ArchiveTarget:
    return script.ArchiveTarget(
        source_hash=hashlib.sha256(payload).hexdigest(),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1560385/"
            "000104746917000332/a2230745z424b3.htm"
        ),
        dataset="official_identity_evidence_raw",
        source="official_identity_evidence_raw",
        content_type="text/html",
        retrieved_at=RETRIEVED_AT,
        extension="html",
    )


def _archive_row(
    target: script.ArchiveTarget,
    *,
    source_url: object = None,
) -> dict[str, object]:
    return {
        "archive_id": target.source_hash,
        "dataset": target.dataset,
        "object_path": target.object_path(COMPLETED_SESSION),
        "content_type": target.content_type,
        "effective_date": COMPLETED_SESSION,
        "source": target.source,
        "retrieved_at": target.retrieved_at,
        "source_hash": target.source_hash,
        "source_url": source_url,
    }


def _unrelated_archive_row() -> dict[str, object]:
    digest = hashlib.sha256(b"unrelated").hexdigest()
    return {
        "archive_id": digest,
        "dataset": "unrelated_official_evidence",
        "object_path": f"archives/{COMPLETED_SESSION}/{digest}.txt.gz",
        "content_type": "text/plain",
        "effective_date": COMPLETED_SESSION,
        "source": "unrelated_official_evidence",
        "retrieved_at": RETRIEVED_AT,
        "source_hash": digest,
        "source_url": "https://www.sec.gov/Archives/edgar/data/1/2/example.txt",
    }


def _security_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": "US:TEST:FORMULA-ONE",
                "primary_symbol": "FWONA",
                "name": "Formula One Group",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "2017-01-25",
                "active_to": "",
                "source": "fixture",
                "retrieved_at": RETRIEVED_AT,
                "source_hash": hashlib.sha256(b"master").hexdigest(),
            }
        ]
    )


def _build_repository(
    root: Path,
    *,
    payload: bytes = b"reviewed Formula One SEC filing",
    archived_payload: bytes | None = None,
    source_url: object = None,
    mutate_row: Callable[[dict[str, object]], None] | None = None,
    duplicate_target: bool = False,
) -> tuple[LocalDatasetRepository, script.ArchiveTarget]:
    repository = LocalDatasetRepository(root)
    target = _target(payload)
    archive_path = root / target.object_path(COMPLETED_SESSION)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(
        gzip.compress(
            payload if archived_payload is None else archived_payload,
            mtime=0,
        )
    )
    row = _archive_row(target, source_url=source_url)
    if mutate_row is not None:
        mutate_row(row)
    rows = [row, _unrelated_archive_row()]
    if duplicate_target:
        duplicate = dict(_archive_row(target, source_url=None))
        duplicate["archive_id"] = hashlib.sha256(b"duplicate-id").hexdigest()
        rows.append(duplicate)
    archive_result = repository.write_frame(
        "source_archive",
        pd.DataFrame(rows),
        completed_session=COMPLETED_SESSION,
        version="base-source-archive",
    )
    master_result = repository.write_frame(
        "security_master",
        _security_master(),
        completed_session=COMPLETED_SESSION,
        version="base-security-master",
    )
    repository.commit_release(
        COMPLETED_SESSION,
        {
            "source_archive": archive_result.manifest.version,
            "security_master": master_result.manifest.version,
        },
        quality=DataQuality.DEGRADED,
        warnings=("fixture warning preserved",),
    )
    return repository, target


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_plan_is_read_only_and_changes_only_the_exact_null_url(tmp_path: Path):
    repository, target = _build_repository(tmp_path)
    before_files = _tree_hashes(tmp_path)
    release, _ = repository.current_release()
    assert release is not None
    before = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )

    prepared = script.prepare_repair(repository, target=target)

    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["source_archive_rows_changed"] == 1
    assert prepared.summary["network_accessed"] is False
    assert prepared.summary["eodhd_calls"] == 0
    assert prepared.summary["r2_accessed"] is False
    assert _tree_hashes(tmp_path) == before_files
    target_mask = prepared.frame["archive_id"].astype(str).eq(target.source_hash)
    assert prepared.frame.loc[target_mask, "source_url"].tolist() == [
        target.source_url
    ]
    unrelated = before["archive_id"].astype(str).ne(target.source_hash)
    pd.testing.assert_frame_equal(
        before.loc[unrelated].reset_index(drop=True),
        prepared.frame.loc[unrelated].reset_index(drop=True),
    )
    for column in before.columns:
        if column != "source_url":
            pd.testing.assert_series_equal(
                before[column], prepared.frame[column], check_names=False
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("archive_id", "1" * 64),
        ("dataset", "wrong_dataset"),
        ("object_path", "archives/wrong/payload.html.gz"),
        ("content_type", "application/json"),
        ("effective_date", "2026-07-14"),
        ("source", "wrong_source"),
        ("retrieved_at", "2026-07-18T00:00:01Z"),
        ("source_hash", "2" * 64),
    ],
)
def test_plan_rejects_any_non_url_row_change(
    tmp_path: Path, field: str, value: object
):
    def mutate(row: dict[str, object]) -> None:
        row[field] = value

    repository, target = _build_repository(tmp_path, mutate_row=mutate)

    with pytest.raises(ValueError, match="differs outside source_url"):
        script.prepare_repair(repository, target=target)


def test_plan_rejects_unexpected_existing_url(tmp_path: Path):
    repository, target = _build_repository(
        tmp_path, source_url="https://example.test/not-the-sec-filing"
    )

    with pytest.raises(ValueError, match="neither null nor the exact SEC URL"):
        script.prepare_repair(repository, target=target)


def test_one_row_repair_rejects_missing_source_url_column(tmp_path: Path):
    repository, target = _build_repository(tmp_path)
    release, _ = repository.current_release()
    assert release is not None
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    ).drop(columns=["source_url"])

    with pytest.raises(ValueError, match="lacks the optional source_url column"):
        script._rewrite_archive_url(
            repository,
            archive,
            target=target,
            completed_session=COMPLETED_SESSION,
        )


def test_plan_rejects_duplicate_related_rows(tmp_path: Path):
    repository, target = _build_repository(tmp_path, duplicate_target=True)

    with pytest.raises(ValueError, match="exactly one Formula One archive row; found 2"):
        script.prepare_repair(repository, target=target)


def test_plan_rehashes_the_decompressed_immutable_payload(tmp_path: Path):
    repository, target = _build_repository(
        tmp_path, archived_payload=b"tampered SEC payload"
    )

    with pytest.raises(ValueError, match="archive content hash changed"):
        script.prepare_repair(repository, target=target)


def test_already_repaired_is_idempotent_and_performs_no_writes(tmp_path: Path):
    repository, target = _build_repository(tmp_path, source_url=_target().source_url)
    before_files = _tree_hashes(tmp_path)

    prepared = script.prepare_repair(repository, target=target)
    result = script.apply_repair(repository, prepared)

    assert prepared.summary["status"] == "already_repaired"
    assert result["writes_performed"] is False
    assert _tree_hashes(tmp_path) == before_files


def test_apply_writes_only_source_archive_and_replays_idempotently(tmp_path: Path):
    repository, target = _build_repository(tmp_path)
    before_release, _ = repository.current_release()
    assert before_release is not None
    before_master_pointer, _ = repository.current_pointer("security_master")
    assert before_master_pointer is not None
    prepared = script.prepare_repair(repository, target=target)

    result = script.apply_repair(repository, prepared)

    after_release, _ = repository.current_release()
    after_master_pointer, _ = repository.current_pointer("security_master")
    assert after_release is not None and after_master_pointer is not None
    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    assert (
        after_release.dataset_versions["source_archive"]
        != before_release.dataset_versions["source_archive"]
    )
    assert (
        after_release.dataset_versions["security_master"]
        == before_release.dataset_versions["security_master"]
    )
    assert after_master_pointer.version == before_master_pointer.version
    assert after_release.quality == before_release.quality
    assert after_release.warnings == before_release.warnings
    assert script.prepare_repair(repository, target=target).summary["status"] == (
        "already_repaired"
    )


@pytest.mark.parametrize(
    "failure_stage", ["after_source_archive_write", "after_release_commit"]
)
def test_apply_rolls_back_release_and_pointer_on_failure(
    tmp_path: Path, failure_stage: str
):
    repository, target = _build_repository(tmp_path)
    old_release = repository.objects.get("releases/current.json").data
    old_pointer = repository.objects.get(
        repository.current_key("source_archive")
    ).data
    prepared = script.prepare_repair(repository, target=target)

    def inject(stage: str) -> None:
        if stage == failure_stage:
            raise RuntimeError(f"injected:{stage}")

    with pytest.raises(RuntimeError, match=f"injected:{failure_stage}"):
        script.apply_repair(repository, prepared, inject_failure=inject)

    assert repository.objects.get("releases/current.json").data == old_release
    assert (
        repository.objects.get(repository.current_key("source_archive")).data
        == old_pointer
    )
    assert script.prepare_repair(repository, target=target).summary["status"] == (
        "validated_offline_plan"
    )
    journals = list((tmp_path / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["status"] == "rolled_back"
    recovery = tmp_path / script.RECOVERY_DIR
    assert not recovery.exists() or not list(recovery.glob("*.json"))


def test_apply_fails_closed_on_pointer_cas_change(tmp_path: Path):
    repository, target = _build_repository(tmp_path)
    old_release = repository.objects.get("releases/current.json").data
    old_pointer = repository.objects.get(
        repository.current_key("source_archive")
    ).data
    prepared = script.prepare_repair(repository, target=target)
    stale = dataclasses.replace(prepared, pointer_etag="stale-etag")

    with pytest.raises(RuntimeError, match="pointer changed after URL planning"):
        script.apply_repair(repository, stale)

    assert repository.objects.get("releases/current.json").data == old_release
    assert (
        repository.objects.get(repository.current_key("source_archive")).data
        == old_pointer
    )


def test_cli_defaults_to_read_only_plan():
    args = script._parse_args([])
    assert args.apply is False
    assert args.cache_root == script.DEFAULT_CACHE_ROOT
