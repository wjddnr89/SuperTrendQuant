from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import R2Config
from .manifest import CurrentPointer, DataRelease, DatasetManifest, sha256_bytes, write_atomic
from .schemas import dataset_spec


class ConditionalWriteFailed(RuntimeError):
    pass


class ObjectNotFound(FileNotFoundError):
    pass


@dataclass(frozen=True)
class ObjectValue:
    data: bytes
    etag: str


class ObjectStore(Protocol):
    def get(self, key: str) -> ObjectValue:
        ...

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
    ) -> str:
        ...

    def list(self, prefix: str) -> tuple[str, ...]:
        ...


class LocalObjectStore:
    """Filesystem implementation used for local operation and CAS tests."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key.lstrip("/")).resolve()
        root = self.root.resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"Object key escapes store root: {key}")
        return path

    def get(self, key: str) -> ObjectValue:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFound(key)
        data = path.read_bytes()
        return ObjectValue(data=data, etag=sha256_bytes(data))

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
    ) -> str:
        path = self._path(key)
        existing = path.read_bytes() if path.is_file() else None
        if if_none_match and existing is not None:
            raise ConditionalWriteFailed(f"Object already exists: {key}")
        if if_match is not None:
            actual = sha256_bytes(existing) if existing is not None else ""
            if actual != if_match.strip('"'):
                raise ConditionalWriteFailed(f"ETag changed for {key}")
        write_atomic(path, data)
        return sha256_bytes(data)

    def list(self, prefix: str) -> tuple[str, ...]:
        base = self._path(prefix)
        if base.is_file():
            return (prefix,)
        if not base.exists():
            return ()
        return tuple(
            str(path.relative_to(self.root)).replace(os.sep, "/")
            for path in sorted(base.rglob("*"))
            if path.is_file()
        )


class R2ObjectStore:
    def __init__(self, config: R2Config):
        if not config.enabled:
            raise ValueError("R2 is disabled in configuration.")
        try:
            import boto3
        except ModuleNotFoundError as exc:
            raise RuntimeError("boto3 is required for Cloudflare R2 support.") from exc
        access_key = os.getenv(config.access_key_env)
        secret_key = os.getenv(config.secret_key_env)
        endpoint_url = os.getenv(config.endpoint_env)
        if not endpoint_url or not access_key or not secret_key:
            raise RuntimeError(
                "R2 environment values are missing: "
                f"{config.endpoint_env}, {config.access_key_env}, {config.secret_key_env}"
            )
        self.bucket = config.bucket
        self.prefix = config.prefix.strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=config.region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def _key(self, key: str) -> str:
        return "/".join(part for part in (self.prefix, key.lstrip("/")) if part)

    def get(self, key: str) -> ObjectValue:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as exc:
            if _client_error_code(exc) in {"NoSuchKey", "404", "NotFound"}:
                raise ObjectNotFound(key) from exc
            raise
        return ObjectValue(
            data=response["Body"].read(),
            etag=str(response.get("ETag", "")).strip('"'),
        )

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
    ) -> str:
        kwargs: dict[str, object] = {
            "Bucket": self.bucket,
            "Key": self._key(key),
            "Body": data,
        }
        if if_match is not None:
            kwargs["IfMatch"] = if_match
        if if_none_match:
            kwargs["IfNoneMatch"] = "*"
        try:
            response = self.client.put_object(**kwargs)
        except Exception as exc:
            if _client_error_code(exc) in {"PreconditionFailed", "412", "ConditionalRequestConflict"}:
                raise ConditionalWriteFailed(f"Conditional R2 write failed: {key}") from exc
            raise
        return str(response.get("ETag", "")).strip('"')

    def list(self, prefix: str) -> tuple[str, ...]:
        remote_prefix = self._key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        values: list[str] = []
        root_prefix = f"{self.prefix}/" if self.prefix else ""
        for page in paginator.paginate(Bucket=self.bucket, Prefix=remote_prefix):
            for item in page.get("Contents", ()):
                key = str(item["Key"])
                values.append(key.removeprefix(root_prefix))
        return tuple(values)


@dataclass(frozen=True)
class PublishResult:
    pointer: CurrentPointer
    pointer_etag: str
    conflict: bool = False
    conflict_prefix: str = ""


class DatasetPublisher:
    def __init__(self, store: ObjectStore):
        self.store = store

    @staticmethod
    def version_prefix(dataset: str, version: str) -> str:
        return f"datasets/{dataset}/versions/{version}"

    @staticmethod
    def current_key(dataset: str) -> str:
        return f"datasets/{dataset}/current.json"

    def current(self, dataset: str) -> tuple[CurrentPointer | None, str | None]:
        try:
            value = self.store.get(self.current_key(dataset))
        except ObjectNotFound:
            return None, None
        return CurrentPointer.from_bytes(value.data), value.etag

    def publish(
        self,
        local_version_root: str | Path,
        manifest: DatasetManifest,
        *,
        expected_pointer_etag: str | None,
    ) -> PublishResult:
        root = Path(local_version_root)
        prefix = self.version_prefix(manifest.dataset, manifest.version)
        for item in manifest.files:
            self._put_immutable(f"{prefix}/{item.path}", (root / item.path).read_bytes())
        manifest_path = f"{prefix}/manifest.json"
        manifest_bytes = manifest.to_bytes()
        self._put_immutable(manifest_path, manifest_bytes)
        pointer = CurrentPointer.create(manifest, manifest_path)
        try:
            pointer_etag = self.store.put(
                self.current_key(manifest.dataset),
                pointer.to_bytes(),
                if_match=expected_pointer_etag,
                if_none_match=expected_pointer_etag is None,
            )
        except ConditionalWriteFailed:
            conflict_prefix = f"conflicts/{manifest.dataset}/{manifest.version}"
            self.store.put(f"{conflict_prefix}/manifest.json", manifest_bytes, if_none_match=True)
            return PublishResult(pointer, "", conflict=True, conflict_prefix=conflict_prefix)
        return PublishResult(pointer, pointer_etag)

    def upload_version(
        self,
        local_version_root: str | Path,
        manifest: DatasetManifest,
    ) -> None:
        root = Path(local_version_root)
        prefix = self.version_prefix(manifest.dataset, manifest.version)
        for item in manifest.files:
            self._put_immutable(f"{prefix}/{item.path}", (root / item.path).read_bytes())
        self._put_immutable(f"{prefix}/manifest.json", manifest.to_bytes())

    def advance_current(
        self,
        manifest: DatasetManifest,
        *,
        expected_pointer_etag: str | None,
    ) -> PublishResult:
        manifest_path = f"{self.version_prefix(manifest.dataset, manifest.version)}/manifest.json"
        pointer = CurrentPointer.create(manifest, manifest_path)
        try:
            etag = self.store.put(
                self.current_key(manifest.dataset),
                pointer.to_bytes(),
                if_match=expected_pointer_etag,
                if_none_match=expected_pointer_etag is None,
            )
        except ConditionalWriteFailed:
            conflict_prefix = f"conflicts/{manifest.dataset}/{manifest.version}"
            self._put_immutable(f"{conflict_prefix}/manifest.json", manifest.to_bytes())
            return PublishResult(pointer, "", conflict=True, conflict_prefix=conflict_prefix)
        return PublishResult(pointer, etag)

    def _put_immutable(self, key: str, data: bytes) -> None:
        try:
            self.store.put(key, data, if_none_match=True)
        except ConditionalWriteFailed:
            existing = self.store.get(key)
            if existing.data != data:
                raise ConditionalWriteFailed(f"Immutable object differs: {key}")


class DatasetCache:
    def __init__(self, root: str | Path, store: ObjectStore):
        self.root = Path(root)
        self.store = store

    def sync(self, dataset: str) -> DatasetManifest:
        pointer_value = self.store.get(DatasetPublisher.current_key(dataset))
        pointer = CurrentPointer.from_bytes(pointer_value.data)
        manifest_value = self.store.get(pointer.manifest_path)
        if sha256_bytes(manifest_value.data) != pointer.manifest_sha256:
            raise ValueError(f"Remote manifest hash mismatch for {dataset}")
        manifest = DatasetManifest.from_bytes(manifest_value.data)
        self._sync_manifest_chain(dataset, manifest, manifest_value.data, set())
        write_atomic(self.root / "datasets" / dataset / "current.json", pointer_value.data)
        if dataset == "source_archive":
            self._sync_archive_payloads(manifest)
        return manifest

    def sync_release(self, datasets: tuple[str, ...] | None = None) -> DataRelease:
        release_value = self.store.get("releases/current.json")
        release = DataRelease.from_bytes(release_value.data)
        selected = set(datasets or release.dataset_versions)
        for dataset, version in release.dataset_versions.items():
            if dataset not in selected:
                continue
            manifest_path = f"{DatasetPublisher.version_prefix(dataset, version)}/manifest.json"
            manifest_value = self.store.get(manifest_path)
            manifest = DatasetManifest.from_bytes(manifest_value.data)
            self._sync_manifest_chain(dataset, manifest, manifest_value.data, set())
            pointer = CurrentPointer.create(manifest, manifest_path)
            write_atomic(self.root / "datasets" / dataset / "current.json", pointer.to_bytes())
            if dataset == "source_archive":
                self._sync_archive_payloads(manifest)
        immutable_value = self.store.get(f"releases/{release.version}.json")
        if immutable_value.data != release_value.data:
            raise ValueError(f"Release current/immutable mismatch: {release.version}")
        if datasets is None or set(release.dataset_versions).issubset(selected):
            write_atomic(self.root / f"releases/{release.version}.json", immutable_value.data)
            write_atomic(self.root / "releases/current.json", release_value.data)
        return release

    def _sync_manifest_chain(
        self,
        dataset: str,
        manifest: DatasetManifest,
        manifest_bytes: bytes,
        seen: set[str],
    ) -> None:
        if manifest.version in seen:
            raise ValueError(f"Remote manifest cycle detected: {dataset}/{manifest.version}")
        seen.add(manifest.version)
        if bool(manifest.metadata.get("inherits_parent")):
            if not manifest.parent_version:
                raise ValueError(f"Remote inherited version has no parent: {dataset}/{manifest.version}")
            parent_path = (
                f"{DatasetPublisher.version_prefix(dataset, manifest.parent_version)}/manifest.json"
            )
            parent_value = self.store.get(parent_path)
            parent = DatasetManifest.from_bytes(parent_value.data)
            self._sync_manifest_chain(dataset, parent, parent_value.data, seen)
        version_root = self.root / "datasets" / dataset / "versions" / manifest.version
        for item in manifest.files:
            local = version_root / item.path
            if not local.is_file() or sha256_bytes(local.read_bytes()) != item.sha256:
                remote_key = f"{DatasetPublisher.version_prefix(dataset, manifest.version)}/{item.path}"
                value = self.store.get(remote_key)
                if sha256_bytes(value.data) != item.sha256:
                    raise ValueError(f"Remote file hash mismatch: {remote_key}")
                write_atomic(local, value.data)
        write_atomic(version_root / "manifest.json", manifest_bytes)

    def _sync_archive_payloads(self, manifest: DatasetManifest) -> None:
        try:
            import pandas as pd
        except ModuleNotFoundError:
            return
        paths = []
        current = manifest
        while True:
            root = self.root / "datasets" / current.dataset / "versions" / current.version
            paths.extend(root / item.path for item in current.files)
            if not bool(current.metadata.get("inherits_parent")):
                break
            parent_path = root.parent / current.parent_version / "manifest.json"
            current = DatasetManifest.from_bytes(parent_path.read_bytes())
        if not paths:
            return
        frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
        if "object_path" not in frame:
            return
        for object_path in frame["object_path"].dropna().astype(str).drop_duplicates():
            local = self.root / object_path
            if local.is_file():
                continue
            value = self.store.get(object_path)
            write_atomic(local, value.data)


@dataclass(frozen=True)
class RepositoryPublishResult:
    dataset: str
    version: str
    published: bool
    conflict: bool = False
    detail: str = ""


def publish_repository(
    repository,
    store: ObjectStore,
    datasets: tuple[str, ...],
) -> tuple[RepositoryPublishResult, ...]:
    publisher = DatasetPublisher(store)
    output: list[RepositoryPublishResult] = []
    original_release, _ = repository.current_release()
    for dataset in datasets:
        publish_detail = ""
        local = repository.current_manifest(dataset)
        if local is None:
            output.append(RepositoryPublishResult(dataset, "", False, detail="local dataset missing"))
            continue
        remote, remote_etag = publisher.current(dataset)
        chain = repository.manifest_chain(dataset, local.version)
        chain_versions = {item.version for item in chain}
        if remote is not None and remote.version == local.version:
            output.append(RepositoryPublishResult(dataset, local.version, False, detail="already current"))
            continue
        if remote is not None and remote.version not in chain_versions:
            remote_manifest = DatasetManifest.from_bytes(
                store.get(remote.manifest_path).data
            )
            merged, merge_detail = _merge_divergent_dataset(
                repository,
                store,
                dataset,
                local,
                remote_manifest,
            )
            if merged is None:
                conflict_key = f"conflicts/{dataset}/{local.version}/manifest.json"
                publisher._put_immutable(conflict_key, local.to_bytes())
                try:
                    repository.objects.put(
                        conflict_key,
                        local.to_bytes(),
                        if_none_match=True,
                    )
                except ConditionalWriteFailed:
                    pass
                output.append(
                    RepositoryPublishResult(
                        dataset,
                        local.version,
                        False,
                        conflict=True,
                        detail=merge_detail,
                    )
                )
                continue
            local = merged
            publish_detail = merge_detail
            chain = repository.manifest_chain(dataset, local.version)
            chain_versions = {item.version for item in chain}
            if local.version == remote.version:
                output.append(
                    RepositoryPublishResult(
                        dataset,
                        local.version,
                        False,
                        detail=merge_detail or "remote already contains local changes",
                    )
                )
                continue
        upload = remote is None
        for manifest in chain:
            if not upload and manifest.version == remote.version:
                upload = True
                continue
            if upload:
                root = repository.root / repository.version_prefix(dataset, manifest.version)
                publisher.upload_version(root, manifest)
        if dataset == "source_archive":
            archive_frame = repository.read_frame(dataset)
            for object_path in archive_frame.get("object_path", ()):
                path = repository.root / str(object_path)
                if path.is_file():
                    publisher._put_immutable(str(object_path), path.read_bytes())
        advanced = publisher.advance_current(local, expected_pointer_etag=remote_etag)
        output.append(
            RepositoryPublishResult(
                dataset,
                local.version,
                not advanced.conflict,
                conflict=advanced.conflict,
                detail=advanced.conflict_prefix or publish_detail,
            )
        )
    local_release = original_release
    if local_release is not None and not any(item.conflict for item in output):
        merged_versions = dict(local_release.dataset_versions)
        for item in output:
            if item.dataset != "__release__" and item.version and not item.conflict:
                merged_versions[item.dataset] = item.version
        if merged_versions != local_release.dataset_versions:
            completed_session = max(
                repository.manifest_for_version(dataset, version).completed_session
                for dataset, version in merged_versions.items()
            )
            local_release = repository.commit_release(
                completed_session,
                merged_versions,
                quality=local_release.quality,
                warnings=local_release.warnings,
            )
    if local_release is not None:
        dataset_conflict = any(item.conflict for item in output)
        remote_mismatches: list[str] = []
        if not dataset_conflict:
            for dataset, version in local_release.dataset_versions.items():
                remote_manifest, _ = publisher.current(dataset)
                if remote_manifest is None:
                    remote_mismatches.append(f"{dataset}=missing (expected {version})")
                elif remote_manifest.version != version:
                    remote_mismatches.append(
                        f"{dataset}={remote_manifest.version} (expected {version})"
                    )
        if dataset_conflict or remote_mismatches:
            detail = "dataset conflict prevented release publication"
            if remote_mismatches:
                detail = "release datasets are not current remotely: " + ", ".join(remote_mismatches)
            output.append(
                RepositoryPublishResult(
                    "__release__",
                    local_release.version,
                    False,
                    conflict=True,
                    detail=detail,
                )
            )
        else:
            try:
                remote_release_value = store.get("releases/current.json")
                remote_release = DataRelease.from_bytes(remote_release_value.data)
                remote_release_etag = remote_release_value.etag
            except ObjectNotFound:
                remote_release = None
                remote_release_etag = None
            if remote_release is not None and remote_release.version == local_release.version:
                output.append(
                    RepositoryPublishResult(
                        "__release__", local_release.version, False, detail="already current"
                    )
                )
            else:
                immutable_key = f"releases/{local_release.version}.json"
                release_bytes = local_release.to_bytes()
                publisher._put_immutable(immutable_key, release_bytes)
                try:
                    store.put(
                        "releases/current.json",
                        release_bytes,
                        if_match=remote_release_etag,
                        if_none_match=remote_release_etag is None,
                    )
                except ConditionalWriteFailed:
                    publisher._put_immutable(
                        f"conflicts/releases/{local_release.version}.json",
                        release_bytes,
                    )
                    output.append(
                        RepositoryPublishResult(
                            "__release__",
                            local_release.version,
                            False,
                            conflict=True,
                            detail="release CAS failed",
                        )
                    )
                else:
                    output.append(
                        RepositoryPublishResult("__release__", local_release.version, True)
                    )
    return tuple(output)


def _merge_divergent_dataset(
    repository,
    store: ObjectStore,
    dataset: str,
    local: DatasetManifest,
    remote: DatasetManifest,
) -> tuple[DatasetManifest | None, str]:
    """Rebase append-only disjoint changes; conflicting values remain quarantined."""
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        return None, f"cannot merge divergent versions without pandas: {exc}"

    local_lineage = _local_lineage(repository, dataset, local)
    remote_lineage = _remote_lineage(store, dataset, remote)
    remote_versions = {manifest.version for manifest in remote_lineage}
    common = next(
        (manifest for manifest in local_lineage if manifest.version in remote_versions),
        None,
    )
    if common is None:
        return None, f"remote current {remote.version} has no common ancestor"

    local_frame = repository.read_frame(dataset, local.version)
    base_frame = repository.read_frame(dataset, common.version)
    with tempfile.TemporaryDirectory(prefix="stq-remote-merge-", dir=repository.root) as directory:
        remote_repository = repository.__class__(directory)
        DatasetCache(directory, store).sync(dataset)
        remote_frame = remote_repository.read_frame(dataset, remote.version)

    # The repository exposes the schema through its module-level dataset_spec;
    # infer the logical key from the already validated frames to avoid a storage
    # -> repository import cycle.
    primary_key = dataset_spec(dataset).primary_key
    local_changes, local_deletes = _frame_changes(
        dataset, base_frame, local_frame, primary_key
    )
    remote_changes, remote_deletes = _frame_changes(
        dataset, base_frame, remote_frame, primary_key
    )
    if local_deletes or remote_deletes:
        return None, "divergent snapshots contain deletions and require manual conflict review"

    overlap = set(local_changes) & set(remote_changes)
    conflicts = [
        key
        for key in overlap
        if _business_record(dataset, local_changes[key])
        != _business_record(dataset, remote_changes[key])
    ]
    if conflicts:
        return None, f"same-key value conflict on {len(conflicts)} row(s)"

    delta_records = [
        record
        for key, record in local_changes.items()
        if key not in remote_changes
    ]
    DatasetCache(repository.root, store).sync(dataset)
    if not delta_records:
        return repository.current_manifest(dataset), "remote already contains equivalent changes"
    delta = pd.DataFrame(delta_records)
    result = repository.append_frame(
        dataset,
        delta,
        completed_session=max(local.completed_session, remote.completed_session),
        metadata={
            "operation": "merge_disjoint_publishers",
            "merged_local_version": local.version,
            "merged_remote_version": remote.version,
        },
    )
    if result.conflict:
        return None, f"merge current-pointer CAS failed: {result.conflict_path}"
    return result.manifest, "disjoint changes automatically rebased"


def _local_lineage(repository, dataset: str, latest: DatasetManifest) -> list[DatasetManifest]:
    output = [latest]
    seen = {latest.version}
    current = latest
    while current.parent_version:
        if current.parent_version in seen:
            raise ValueError(f"Dataset manifest cycle detected: {dataset}/{current.parent_version}")
        current = repository.manifest_for_version(dataset, current.parent_version)
        output.append(current)
        seen.add(current.version)
    return output


def _remote_lineage(
    store: ObjectStore,
    dataset: str,
    latest: DatasetManifest,
) -> list[DatasetManifest]:
    output = [latest]
    seen = {latest.version}
    current = latest
    while current.parent_version:
        if current.parent_version in seen:
            raise ValueError(f"Remote manifest cycle detected: {dataset}/{current.parent_version}")
        key = f"{DatasetPublisher.version_prefix(dataset, current.parent_version)}/manifest.json"
        current = DatasetManifest.from_bytes(store.get(key).data)
        output.append(current)
        seen.add(current.version)
    return output


def _frame_changes(dataset: str, base, candidate, primary_key: tuple[str, ...]):
    base_records = _record_map(base, primary_key)
    candidate_records = _record_map(candidate, primary_key)
    changed = {
        key: record
        for key, record in candidate_records.items()
        if key not in base_records
        or _business_record(dataset, record)
        != _business_record(dataset, base_records[key])
    }
    deleted = set(base_records) - set(candidate_records)
    return changed, deleted


def _record_map(frame, primary_key: tuple[str, ...]) -> dict[tuple[str, ...], dict]:
    output = {}
    for record in frame.to_dict("records"):
        key = tuple(str(record[column]) for column in primary_key)
        output[key] = record
    return output


def _business_record(dataset: str, record: dict) -> dict:
    ignored = {"source", "source_url", "source_kind", "retrieved_at", "source_hash"}
    if dataset == "adjustment_factors":
        ignored.update({"source_version", "calculated_at"})
    return {
        key: _canonical_value(value)
        for key, value in record.items()
        if key not in ignored
    }


def _canonical_value(value):
    try:
        import pandas as pd

        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except (ModuleNotFoundError, TypeError, ValueError):
        pass
    if isinstance(value, (dict, list, tuple)):
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value




def _client_error_code(exc: Exception) -> str:
    response = getattr(exc, "response", {})
    if not isinstance(response, dict):
        return ""
    error = response.get("Error", {})
    return str(error.get("Code", "")) if isinstance(error, dict) else ""
