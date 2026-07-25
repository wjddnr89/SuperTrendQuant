from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
    ManifestFile,
    sha256_file,
    utc_now_iso,
    write_atomic,
)
from .schemas import DATASET_SPECS, dataset_spec
from .storage import ConditionalWriteFailed, LocalObjectStore, ObjectNotFound
from .validation import (
    ValidationReport,
    validate_dataset,
    validate_manifest_files,
    validate_revisions,
)


@dataclass(frozen=True)
class DatasetWriteResult:
    manifest: DatasetManifest
    validation: ValidationReport
    conflict: bool = False
    conflict_path: str = ""


class LocalDatasetRepository:
    """Immutable local Parquet versions with a small CAS current pointer."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.objects = LocalObjectStore(self.root)

    @staticmethod
    def current_key(dataset: str) -> str:
        return f"datasets/{dataset}/current.json"

    @staticmethod
    def version_prefix(dataset: str, version: str) -> str:
        return f"datasets/{dataset}/versions/{version}"

    def current_pointer(self, dataset: str) -> tuple[CurrentPointer | None, str | None]:
        try:
            value = self.objects.get(self.current_key(dataset))
        except ObjectNotFound:
            return None, None
        return CurrentPointer.from_bytes(value.data), value.etag

    def current_release(self) -> tuple[DataRelease | None, str | None]:
        try:
            value = self.objects.get("releases/current.json")
        except ObjectNotFound:
            return None, None
        return DataRelease.from_bytes(value.data), value.etag

    def commit_release(
        self,
        completed_session: str,
        dataset_versions: dict[str, str],
        *,
        quality: str,
        warnings: tuple[str, ...] = (),
        expected_etag: str | None = None,
    ) -> DataRelease:
        _, actual_etag = self.current_release()
        if expected_etag is None:
            expected_etag = actual_etag
        release = DataRelease.create(
            completed_session,
            dataset_versions,
            quality=quality,
            warnings=warnings,
        )
        immutable_key = f"releases/{release.version}.json"
        self.objects.put(immutable_key, release.to_bytes(), if_none_match=True)
        try:
            self.objects.put(
                "releases/current.json",
                release.to_bytes(),
                if_match=expected_etag,
                if_none_match=expected_etag is None,
            )
        except ConditionalWriteFailed:
            self.objects.put(
                f"conflicts/releases/{release.version}.json",
                release.to_bytes(),
                if_none_match=True,
            )
            raise
        return release

    def current_manifest(self, dataset: str) -> DatasetManifest | None:
        pointer, _ = self.current_pointer(dataset)
        if pointer is None:
            return None
        manifest_bytes = self.objects.get(pointer.manifest_path).data
        manifest = DatasetManifest.from_bytes(manifest_bytes)
        report = validate_manifest_files(self.root / self.version_prefix(dataset, manifest.version), manifest)
        report.raise_for_errors()
        return manifest

    def manifest_for_version(self, dataset: str, version: str) -> DatasetManifest:
        return self._manifest_for_version(dataset, version)

    def manifest_chain(
        self,
        dataset: str,
        version: str | None = None,
    ) -> tuple[DatasetManifest, ...]:
        latest = self._manifest_for_version(dataset, version)
        chain = [latest]
        seen = {latest.version}
        while bool(chain[-1].metadata.get("inherits_parent")):
            parent = chain[-1].parent_version
            if not parent:
                raise ValueError(f"Inherited dataset version has no parent: {dataset}/{chain[-1].version}")
            if parent in seen:
                raise ValueError(f"Dataset manifest cycle detected: {dataset}/{parent}")
            previous = self._manifest_for_version(dataset, parent)
            chain.append(previous)
            seen.add(parent)
        chain.reverse()
        return tuple(chain)

    def parquet_paths(
        self,
        dataset: str,
        version: str | None = None,
        *,
        min_session: str = "",
        max_session: str = "",
    ) -> tuple[Path, ...]:
        paths: list[Path] = []
        for manifest in self.manifest_chain(dataset, version):
            version_root = self.root / self.version_prefix(dataset, manifest.version)
            paths.extend(
                version_root / item.path
                for item in manifest.files
                if _file_overlaps_session_range(
                    item,
                    min_session=min_session,
                    max_session=max_session,
                )
            )
        return tuple(paths)

    def write_frame(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        completed_session: str,
        incomplete_action_policy: str = "warn",
        metadata: dict[str, Any] | None = None,
        expected_pointer_etag: str | None = None,
        version: str | None = None,
        inherit_parent: bool = False,
    ) -> DatasetWriteResult:
        spec = dataset_spec(dataset)
        frame = _with_dataset_defaults(dataset, frame)
        report = validate_dataset(
            dataset,
            frame,
            incomplete_action_policy=incomplete_action_policy,
            completed_session=completed_session,
        )
        report.raise_for_errors()
        current, actual_etag = self.current_pointer(dataset)
        if expected_pointer_etag is None:
            expected_pointer_etag = actual_etag
        version = version or _new_version(completed_session)
        final_prefix = self.version_prefix(dataset, version)
        final_root = self.root / final_prefix
        if final_root.exists():
            raise FileExistsError(f"Dataset version already exists: {dataset}/{version}")
        staging = self.root / ".staging" / dataset / f"{version}-{uuid.uuid4().hex}"
        try:
            files = _write_partitioned_parquet(
                staging,
                frame,
                spec.date_columns,
                spec.partition_columns,
                spec.primary_key,
            )
            manifest_metadata = dict(metadata or {})
            manifest_metadata["inherits_parent"] = bool(inherit_parent and current is not None)
            logical_warnings = tuple(manifest_metadata.pop("_logical_warnings", ()))
            logical_quality = str(
                manifest_metadata.pop("_logical_quality", report.quality)
            )
            unresolved_action_count = int(
                manifest_metadata.pop(
                    "_unresolved_action_count",
                    sum(
                        issue.row_count
                        for issue in report.issues
                        if issue.code == "incomplete_corporate_action"
                    ),
                )
            )
            manifest = DatasetManifest.create(
                dataset=dataset,
                version=version,
                completed_session=completed_session,
                files=files,
                quality=logical_quality,
                parent_version=current.version if current else "",
                source_mode=str(manifest_metadata.get("source_mode", "")),
                official_coverage_start=str(
                    manifest_metadata.get("official_coverage_start", "")
                ),
                official_coverage_end=str(
                    manifest_metadata.get("official_coverage_end", "")
                ),
                unresolved_action_count=unresolved_action_count,
                conflict_count=int(manifest_metadata.get("conflict_count", 0)),
                warnings=(
                    logical_warnings
                    if logical_warnings
                    else tuple(
                        issue.message
                        for issue in report.issues
                        if issue.severity != "error"
                    )
                ),
                metadata=manifest_metadata,
            )
            write_atomic(staging / "manifest.json", manifest.to_bytes())
            final_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        manifest_path = f"{final_prefix}/manifest.json"
        pointer = CurrentPointer.create(manifest, manifest_path)
        try:
            self.objects.put(
                self.current_key(dataset),
                pointer.to_bytes(),
                if_match=expected_pointer_etag,
                if_none_match=expected_pointer_etag is None,
            )
        except ConditionalWriteFailed:
            conflict_path = f"conflicts/{dataset}/{version}/manifest.json"
            self.objects.put(conflict_path, manifest.to_bytes(), if_none_match=True)
            return DatasetWriteResult(manifest, report, conflict=True, conflict_path=conflict_path)
        return DatasetWriteResult(manifest, report)

    def append_frame(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        completed_session: str,
        incomplete_action_policy: str = "warn",
        metadata: dict[str, Any] | None = None,
        expected_pointer_etag: str | None = None,
        version: str | None = None,
    ) -> DatasetWriteResult:
        values = {"operation": "append", **(metadata or {})}
        current = self.current_manifest(dataset)
        logical_report: ValidationReport | None = None
        if current is not None and dataset == "corporate_actions":
            spec = dataset_spec(dataset)
            normalized_delta = _with_dataset_defaults(dataset, frame)
            logical = pd.concat(
                [self.read_frame(dataset, current.version), normalized_delta],
                ignore_index=True,
            ).drop_duplicates(list(spec.primary_key), keep="last")
            logical_report = validate_dataset(
                dataset,
                logical,
                incomplete_action_policy=incomplete_action_policy,
                completed_session=completed_session,
            )
            logical_report.raise_for_errors()
            values["_logical_quality"] = str(logical_report.quality)
            values["_logical_warnings"] = tuple(
                issue.message
                for issue in logical_report.issues
                if issue.severity != "error"
            )
            values["_unresolved_action_count"] = sum(
                issue.row_count
                for issue in logical_report.issues
                if issue.code == "incomplete_corporate_action"
            )
        if current is not None and dataset == "daily_price_raw":
            revision_report = validate_revisions(
                self.read_frame(dataset, current.version),
                frame,
                primary_key=dataset_spec(dataset).primary_key,
                value_columns=("open", "high", "low", "close", "volume", "currency"),
            )
            if not revision_report.valid:
                values["operation"] = "quarantined_source_revision"
                result = self.write_frame(
                    dataset,
                    frame,
                    completed_session=completed_session,
                    incomplete_action_policy=incomplete_action_policy,
                    metadata=values,
                    expected_pointer_etag=f"revision-conflict-{uuid.uuid4().hex}",
                    version=version,
                    inherit_parent=True,
                )
                return DatasetWriteResult(
                    result.manifest,
                    revision_report,
                    conflict=True,
                    conflict_path=result.conflict_path,
                )
        result = self.write_frame(
            dataset,
            frame,
            completed_session=completed_session,
            incomplete_action_policy=incomplete_action_policy,
            metadata=values,
            expected_pointer_etag=expected_pointer_etag,
            version=version,
            inherit_parent=True,
        )
        if logical_report is None:
            return result
        return DatasetWriteResult(
            result.manifest,
            logical_report,
            conflict=result.conflict,
            conflict_path=result.conflict_path,
        )

    def read_frame(self, dataset: str, version: str | None = None) -> pd.DataFrame:
        paths = self.parquet_paths(dataset, version)
        if not paths:
            return pd.DataFrame(columns=dataset_spec(dataset).required_columns)
        frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
        spec = dataset_spec(dataset)
        # PyArrow reconstructs Hive directory keys such as ``year`` and ``month``
        # when a Parquet file lives below ``year=.../month=...``.  Those keys are
        # derived storage metadata, not logical dataset columns.  Letting them
        # escape from the repository makes a read -> rewrite depend on inferred
        # categorical dtypes from every partition and can fail during Parquet
        # conversion when old versions contain a different inferred dtype.
        derived_partitions = [
            column
            for column in spec.partition_columns
            if column in frame.columns and column not in spec.required_columns
        ]
        if derived_partitions:
            frame = frame.drop(columns=derived_partitions)
        return frame.drop_duplicates(list(spec.primary_key), keep="last").reset_index(drop=True)

    def compact(self, dataset: str, *, completed_session: str | None = None) -> DatasetWriteResult:
        current, etag = self.current_pointer(dataset)
        if current is None:
            raise ObjectNotFound(f"No current version for {dataset}")
        manifest = self.current_manifest(dataset)
        frame = self.read_frame(dataset, current.version)
        spec = dataset_spec(dataset)
        frame = frame.drop_duplicates(list(spec.primary_key), keep="last")
        return self.write_frame(
            dataset,
            frame,
            completed_session=completed_session or manifest.completed_session,
            metadata={"operation": "compact", "compacted_from": manifest.version},
            expected_pointer_etag=etag,
        )

    def status(self) -> tuple[dict[str, Any], ...]:
        release, _ = self.current_release()
        manifests: dict[str, DatasetManifest | None] = {}
        invalid: dict[str, str] = {}
        for dataset in DATASET_SPECS:
            try:
                manifests[dataset] = self.current_manifest(dataset)
            except (ObjectNotFound, ValueError) as exc:
                manifests[dataset] = None
                invalid[dataset] = str(exc)
        total_size_bytes = sum(
            item.size_bytes
            for dataset, manifest in manifests.items()
            if manifest is not None
            for chained in self.manifest_chain(dataset, manifest.version)
            for item in chained.files
        )
        unresolved_action_count = (
            manifests["corporate_actions"].unresolved_action_count
            if manifests.get("corporate_actions") is not None
            else 0
        )
        conflict_count = len(self.conflicts())
        rows: list[dict[str, Any]] = [
            (
                {
                    "dataset": "__release__",
                    "status": release.quality,
                    "version": release.version,
                    "completed_session": release.completed_session,
                    "datasets": release.dataset_versions,
                    "warnings": list(release.warnings),
                    "total_size_bytes": total_size_bytes,
                    "unresolved_action_count": unresolved_action_count,
                    "conflict_count": conflict_count,
                }
                if release is not None
                else {"dataset": "__release__", "status": "missing"}
            )
        ]
        for dataset in DATASET_SPECS:
            if dataset in invalid:
                rows.append(
                    {"dataset": dataset, "status": "invalid", "detail": invalid[dataset]}
                )
                continue
            manifest = manifests[dataset]
            if manifest is None:
                rows.append({"dataset": dataset, "status": "missing"})
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "status": manifest.quality,
                    "version": manifest.version,
                    "completed_session": manifest.completed_session,
                    "published_by": manifest.published_by,
                    "files": len(manifest.files),
                    "rows": len(self.read_frame(dataset, manifest.version)),
                    "size_bytes": sum(
                        item.size_bytes
                        for chained in self.manifest_chain(dataset, manifest.version)
                        for item in chained.files
                    ),
                    "chain_depth": len(self.manifest_chain(dataset, manifest.version)),
                    "source_mode": manifest.source_mode,
                    "official_coverage": {
                        "start": manifest.official_coverage_start,
                        "end": manifest.official_coverage_end,
                    },
                    "unresolved_action_count": manifest.unresolved_action_count,
                    "conflict_count": manifest.conflict_count,
                    "warnings": list(manifest.warnings),
                }
            )
        return tuple(rows)

    def conflicts(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for key in self.objects.list("conflicts"):
            if not key.endswith("/manifest.json"):
                continue
            manifest = DatasetManifest.from_bytes(self.objects.get(key).data)
            rows.append(
                {
                    "dataset": manifest.dataset,
                    "version": manifest.version,
                    "created_at": manifest.created_at,
                    "path": key,
                }
            )
        return tuple(rows)

    def _manifest_for_version(self, dataset: str, version: str | None) -> DatasetManifest:
        if version is None:
            manifest = self.current_manifest(dataset)
            if manifest is None:
                raise ObjectNotFound(f"No current version for {dataset}")
            return manifest
        key = f"{self.version_prefix(dataset, version)}/manifest.json"
        return DatasetManifest.from_bytes(self.objects.get(key).data)


def _new_version(completed_session: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    session = completed_session.replace("-", "") or "unknown"
    return f"{session}-{timestamp}"


def _file_overlaps_session_range(
    item: ManifestFile,
    *,
    min_session: str,
    max_session: str,
) -> bool:
    """Prune manifest files whose recorded date range cannot match a query."""

    if min_session and item.max_session and item.max_session < min_session:
        return False
    if max_session and item.min_session and item.min_session > max_session:
        return False
    return True


def _write_partitioned_parquet(
    root: Path,
    frame: pd.DataFrame,
    date_columns: tuple[str, ...],
    partition_columns: tuple[str, ...],
    sort_columns: tuple[str, ...] = (),
) -> tuple[ManifestFile, ...]:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to write Parquet market data.") from exc
    root.mkdir(parents=True, exist_ok=True)
    working = _parquet_safe_frame(frame)
    available_sort = [column for column in sort_columns if column in working]
    if available_sort:
        working = working.sort_values(available_sort, kind="stable")
    date_column = date_columns[0] if date_columns else ""
    if date_column and partition_columns:
        parsed = pd.to_datetime(working[date_column], errors="raise")
        if "year" in partition_columns:
            working["_partition_year"] = parsed.dt.year
        if "month" in partition_columns:
            working["_partition_month"] = parsed.dt.month
        group_columns = [f"_partition_{name}" for name in partition_columns]
        groups = working.groupby(group_columns, dropna=False, sort=True)
    else:
        groups = [((), working)]

    manifest_files: list[ManifestFile] = []
    for number, (keys, group) in enumerate(groups):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        parts = [
            f"{name}={int(value):02d}" if name == "month" else f"{name}={int(value)}"
            for name, value in zip(partition_columns, key_values)
        ]
        relative = Path(*parts, f"part-{number:05d}.parquet")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = group.drop(columns=[column for column in group if column.startswith("_partition_")])
        payload.to_parquet(destination, index=False, engine="pyarrow", compression="zstd")
        sessions = pd.to_datetime(payload[date_column], errors="coerce") if date_column else None
        manifest_files.append(
            ManifestFile(
                path=str(relative).replace(os.sep, "/"),
                sha256=sha256_file(destination),
                size_bytes=destination.stat().st_size,
                row_count=len(payload),
                min_session=(sessions.min().date().isoformat() if sessions is not None and sessions.notna().any() else ""),
                max_session=(sessions.max().date().isoformat() if sessions is not None and sessions.notna().any() else ""),
            )
        )
    return tuple(manifest_files)


def _parquet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in (name for name in out.columns if str(out[name].dtype) == "object"):
        if out[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            out[column] = out[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )
    return out


def _with_dataset_defaults(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    defaults: dict[str, object] = {}
    if dataset == "corporate_actions":
        defaults = {
            "announcement_date": "",
            "record_date": "",
            "payment_date": "",
            "source_url": "",
            "source_kind": "provider",
        }
    elif dataset == "index_constituent_anchors":
        defaults = {"source_url": "", "source_kind": "community"}
    elif dataset == "index_membership_events":
        defaults = {
            "announcement_date": "",
            "source_url": "",
            "source_kind": "community",
        }
    elif dataset == "custom_universe_overlays":
        defaults = {"source_url": "", "source_kind": "user"}
    missing = {column: value for column, value in defaults.items() if column not in frame}
    if not missing:
        return frame
    output = frame.copy()
    for column, value in missing.items():
        output[column] = value
    return output
