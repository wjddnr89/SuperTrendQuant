#!/usr/bin/env python3
"""Repair the missing SEC URL on the Formula One evidence archive row.

The historical Formula One identity repair archived the reviewed SEC payload
under its exact content SHA-256, but an optional ``source_url`` column was
accidentally dropped while the ``source_archive`` frame was constructed.  This
one-row repair is deliberately narrow and offline:

* the current release must contain exactly one row related to the pinned hash;
* every field except ``source_url`` must match the reviewed row exactly;
* the immutable gzip payload is decompressed and rehashed before planning;
* plan mode is read-only and is the default;
* apply writes only a new immutable ``source_archive`` version, then commits a
  release with every other dataset version unchanged;
* pointer/release CAS, a durable journal and verified rollback protect apply;
* there is no network, EODHD or R2 code path.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import json
import math
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DATASET = "source_archive"
OPERATION = "repair_us_formula_one_archive_url"
TRANSACTION_DIR = "transactions/us-formula-one-archive-url"
RECOVERY_DIR = "recovery/us-formula-one-archive-url"


@dataclass(frozen=True)
class ArchiveTarget:
    source_hash: str
    source_url: str
    dataset: str
    source: str
    content_type: str
    retrieved_at: str
    extension: str

    def object_path(self, completed_session: str) -> str:
        return (
            f"archives/{completed_session}/{self.source_hash}."
            f"{self.extension}.gz"
        )


FORMULA_ONE_TARGET = ArchiveTarget(
    source_hash=(
        "6a4fe3ee6fea801819f375c2c4426cfb3b619e659dbe93ae5cfdcfe6d4cc45ce"
    ),
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/1560385/"
        "000104746917000332/a2230745z424b3.htm"
    ),
    dataset="sec_edgar_filing",
    source="sec_edgar_filing",
    content_type="text/html",
    retrieved_at="2026-07-18T03:52:54.718209Z",
    extension="html",
)


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etag: str | None
    frame: pd.DataFrame
    target: ArchiveTarget
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


def _noop_injector(_stage: str) -> None:
    return None


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _safe_archive_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    path = (base / object_path).resolve()
    if path == base or base not in path.parents:
        raise ValueError(f"Formula One archive path escapes repository: {object_path}.")
    return path


def _target_indexes(
    archive: pd.DataFrame,
    *,
    target: ArchiveTarget,
    completed_session: str,
) -> list[Any]:
    expected_path = target.object_path(completed_session)
    related = (
        archive["archive_id"].astype(str).eq(target.source_hash)
        | archive["source_hash"].astype(str).eq(target.source_hash)
        | archive["object_path"].astype(str).eq(expected_path)
    )
    return list(archive.index[related])


def _assert_exact_row_except_url(
    row: Mapping[str, Any],
    *,
    target: ArchiveTarget,
    completed_session: str,
) -> None:
    expected = {
        "archive_id": target.source_hash,
        "dataset": target.dataset,
        "object_path": target.object_path(completed_session),
        "content_type": target.content_type,
        "effective_date": completed_session,
        "source": target.source,
        "retrieved_at": target.retrieved_at,
        "source_hash": target.source_hash,
    }
    mismatches = [
        field
        for field, value in expected.items()
        if (
            _date(row.get(field)) != value
            if field == "effective_date"
            else _text(row.get(field)) != value
        )
    ]
    if mismatches:
        raise ValueError(
            "Formula One source_archive row differs outside source_url: "
            + ", ".join(mismatches)
            + "."
        )


def _verify_payload(
    repository: LocalDatasetRepository,
    row: Mapping[str, Any],
    *,
    target: ArchiveTarget,
) -> None:
    path = _safe_archive_path(repository.root, _text(row.get("object_path")))
    if not path.is_file():
        raise FileNotFoundError(f"Formula One archive payload is missing: {path}.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError(f"Formula One archive payload is not valid gzip: {path}.") from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != target.source_hash:
        raise ValueError(
            "Formula One archive content hash changed: "
            f"expected={target.source_hash}; observed={digest}."
        )


def _rewrite_archive_url(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    target: ArchiveTarget,
    completed_session: str,
) -> tuple[pd.DataFrame, bool]:
    if "source_url" not in archive.columns:
        raise ValueError(
            "Current source_archive lacks the optional source_url column; "
            "one-row repair is unsafe."
        )
    indexes = _target_indexes(
        archive, target=target, completed_session=completed_session
    )
    if len(indexes) != 1:
        raise ValueError(
            "Current release must contain exactly one Formula One archive row; "
            f"found {len(indexes)}."
        )
    index = indexes[0]
    row = archive.loc[index]
    _assert_exact_row_except_url(
        row, target=target, completed_session=completed_session
    )
    _verify_payload(repository, row, target=target)
    existing_url = _text(row.get("source_url"))
    if existing_url not in {"", target.source_url}:
        raise ValueError(
            "Formula One archive source_url is neither null nor the exact SEC URL: "
            f"{existing_url}."
        )
    changed = not bool(existing_url)
    output = archive.copy(deep=True)
    output.at[index, "source_url"] = target.source_url
    if len(output) != len(archive) or list(output.columns) != list(archive.columns):
        raise AssertionError("Formula One archive repair changed frame topology.")
    for column in archive.columns:
        if column == "source_url":
            continue
        if not archive[column].equals(output[column]):
            raise AssertionError(
                f"Formula One archive repair changed non-URL column: {column}."
            )
    changed_urls = archive["source_url"].fillna("").astype(str).ne(
        output["source_url"].fillna("").astype(str)
    )
    if int(changed_urls.sum()) != int(changed):
        raise AssertionError("Formula One archive repair changed an unexpected URL row.")
    return output, changed


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        source_archive: pd.DataFrame,
    ):
        self.base = base
        self.versions = dict(versions)
        self.source_archive = source_archive.copy()

    def current_manifest(self, dataset: str):
        version = self.versions.get(dataset)
        return (
            self.base.manifest_for_version(dataset, version)
            if version
            else None
        )

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset == DATASET:
            return self.source_archive.copy()
        version = self.versions.get(dataset)
        if not version:
            return pd.DataFrame()
        return self.base.read_frame(dataset, version)


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    target: ArchiveTarget = FORMULA_ONE_TARGET,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    version = release.dataset_versions.get(DATASET)
    if not version:
        raise RuntimeError("Current release has no source_archive dataset.")
    pointer, pointer_etag = repository.current_pointer(DATASET)
    if pointer is None or pointer.version != version:
        raise RuntimeError("source_archive release/current pointer mismatch.")
    archive = repository.read_frame(DATASET, version)
    repaired, changed = _rewrite_archive_url(
        repository,
        archive,
        target=target,
        completed_session=release.completed_session,
    )
    validate_dataset(
        DATASET, repaired, completed_session=release.completed_session
    ).raise_for_errors()
    candidate = _CandidateRepository(repository, release.dataset_versions, repaired)
    validate_repository_snapshot(candidate).raise_for_errors()
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etag=pointer_etag,
        frame=repaired,
        target=target,
        summary={
            "status": "validated_offline_plan" if changed else "already_repaired",
            "base_release_version": release.version,
            "source_archive_base_version": version,
            "target_archive_id": target.source_hash,
            "target_source_url": target.source_url,
            "source_archive_rows_changed": int(changed),
            "source_archive_only": True,
            "other_dataset_versions_unchanged": True,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    lock_path = repository.root / ".locks/market-store-write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError(
                "Unresolved Formula One archive-URL recovery marker blocks writes."
            )
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        "Interrupted Formula One archive-URL transaction blocks writes: "
                        f"{journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_base_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, release_etag = repository.current_release()
    if (
        release is None
        or release.version != prepared.release.version
        or release_etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after Formula One URL planning.")
    pointer, pointer_etag = repository.current_pointer(DATASET)
    if (
        pointer is None
        or pointer.version != prepared.release.dataset_versions[DATASET]
        or pointer_etag != prepared.pointer_etag
    ):
        raise RuntimeError("source_archive pointer changed after URL planning.")


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: bytes,
    planned_version: str,
    committed_release_version: str,
    old_versions: Mapping[str, str],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            expected_versions = {**dict(old_versions), DATASET: planned_version}
            belongs = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or observed.dataset_versions == expected_versions
            if not belongs:
                raise RuntimeError(
                    f"unexpected release during rollback: {observed.version}"
                )
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    try:
        key = repository.current_key(DATASET)
        current = repository.objects.get(key)
        if current.data != old_pointer_bytes:
            observed = CurrentPointer.from_bytes(current.data)
            if observed.version != planned_version:
                raise RuntimeError(
                    f"unexpected source_archive pointer during rollback: {observed.version}"
                )
            repository.objects.put(key, old_pointer_bytes, if_match=current.etag)
    except Exception as exc:
        errors.append(f"{repository.current_key(DATASET)}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: FailureInjector = _noop_injector,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_repaired":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    with _exclusive_repository_lock(repository):
        _assert_base_unchanged(repository, prepared)
        old_release = repository.objects.get("releases/current.json")
        old_pointer = repository.objects.get(repository.current_key(DATASET))
        transaction_id = uuid.uuid4().hex
        planned_version = (
            "formula-one-archive-url-"
            f"{prepared.release.completed_session.replace('-', '')}-"
            f"{transaction_id}-{DATASET}"
        )
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_formula_one_archive_url_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": base64.b64encode(old_pointer.data).decode("ascii"),
            "planned_source_archive_version": planned_version,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            inject_failure("after_journal")
            result = repository.write_frame(
                DATASET,
                prepared.frame,
                completed_session=prepared.release.completed_session,
                metadata={
                    "operation": OPERATION,
                    "target_archive_id": prepared.target.source_hash,
                    "source_archive_rows_changed": 1,
                    "network_accessed": False,
                    "eodhd_calls": 0,
                    "r2_accessed": False,
                },
                expected_pointer_etag=prepared.pointer_etag,
                version=planned_version,
            )
            if result.conflict:
                raise RuntimeError(
                    f"source_archive write conflicted: {result.conflict_path}."
                )
            if result.manifest.version != planned_version:
                raise RuntimeError("Unexpected source_archive version was written.")
            inject_failure("after_source_archive_write")
            versions = dict(prepared.release.dataset_versions)
            versions[DATASET] = result.manifest.version
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=prepared.release.quality,
                warnings=prepared.release.warnings,
                expected_etag=prepared.release_etag,
            )
            if any(
                committed.dataset_versions.get(name) != version
                for name, version in prepared.release.dataset_versions.items()
                if name != DATASET
            ):
                raise RuntimeError("A non-source_archive dataset version changed.")
            inject_failure("after_release_commit")
            validate_repository_snapshot(repository).raise_for_errors()
            replay = prepare_repair(repository, target=prepared.target)
            if replay.summary["status"] != "already_repaired":
                raise RuntimeError("Formula One archive-URL repair is not idempotent.")
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
                "new_source_archive_version": result.manifest.version,
                "transaction_id": transaction_id,
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointer.data,
                planned_version=planned_version,
                committed_release_version=committed.version if committed else "",
                old_versions=prepared.release.dataset_versions,
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
                    "Formula One archive-URL rollback was incomplete; recovery marker "
                    f"blocks writes: {recovery}; errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair the exact missing SEC URL on Formula One archive evidence."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository)
    result = (
        apply_repair(repository, prepared)
        if args.apply
        else {**prepared.summary, "mode": "plan", "writes_performed": False}
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
