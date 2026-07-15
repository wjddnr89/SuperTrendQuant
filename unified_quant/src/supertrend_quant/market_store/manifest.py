from __future__ import annotations

import hashlib
import getpass
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import DataQuality


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestFile:
    path: str
    sha256: str
    size_bytes: int
    row_count: int
    min_session: str = ""
    max_session: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ManifestFile":
        return cls(**raw)


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    version: str
    created_at: str
    completed_session: str
    quality: str = DataQuality.VALID
    published_by: str = ""
    parent_version: str = ""
    source_mode: str = ""
    official_coverage_start: str = ""
    official_coverage_end: str = ""
    unresolved_action_count: int = 0
    conflict_count: int = 0
    files: tuple[ManifestFile, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["files"] = [asdict(item) for item in self.files]
        payload["warnings"] = list(self.warnings)
        return payload

    def to_bytes(self) -> bytes:
        return (json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetManifest":
        return cls(
            dataset=str(raw["dataset"]),
            version=str(raw["version"]),
            created_at=str(raw["created_at"]),
            completed_session=str(raw.get("completed_session", "")),
            quality=str(raw.get("quality", DataQuality.VALID)),
            published_by=str(raw.get("published_by", "")),
            parent_version=str(raw.get("parent_version", "")),
            source_mode=str(raw.get("source_mode", "")),
            official_coverage_start=str(raw.get("official_coverage_start", "")),
            official_coverage_end=str(raw.get("official_coverage_end", "")),
            unresolved_action_count=int(raw.get("unresolved_action_count", 0)),
            conflict_count=int(raw.get("conflict_count", 0)),
            files=tuple(ManifestFile.from_dict(item) for item in raw.get("files", ())),
            warnings=tuple(str(item) for item in raw.get("warnings", ())),
            metadata=dict(raw.get("metadata", {})),
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> "DatasetManifest":
        return cls.from_dict(json.loads(value))

    @classmethod
    def create(
        cls,
        dataset: str,
        version: str,
        completed_session: str,
        files: tuple[ManifestFile, ...],
        *,
        quality: str = DataQuality.VALID,
        published_by: str = "",
        parent_version: str = "",
        source_mode: str = "",
        official_coverage_start: str = "",
        official_coverage_end: str = "",
        unresolved_action_count: int = 0,
        conflict_count: int = 0,
        warnings: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> "DatasetManifest":
        return cls(
            dataset=dataset,
            version=version,
            created_at=utc_now_iso(),
            completed_session=completed_session,
            quality=str(quality),
            published_by=published_by or os.getenv("STQ_PUBLISHED_BY") or getpass.getuser(),
            parent_version=parent_version,
            source_mode=source_mode,
            official_coverage_start=official_coverage_start,
            official_coverage_end=official_coverage_end,
            unresolved_action_count=int(unresolved_action_count),
            conflict_count=int(conflict_count),
            files=files,
            warnings=warnings,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class CurrentPointer:
    dataset: str
    version: str
    manifest_path: str
    manifest_sha256: str
    updated_at: str

    def to_bytes(self) -> bytes:
        return (json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()

    @classmethod
    def create(cls, manifest: DatasetManifest, manifest_path: str) -> "CurrentPointer":
        manifest_bytes = manifest.to_bytes()
        return cls(
            dataset=manifest.dataset,
            version=manifest.version,
            manifest_path=manifest_path,
            manifest_sha256=sha256_bytes(manifest_bytes),
            updated_at=utc_now_iso(),
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> "CurrentPointer":
        return cls(**json.loads(value))


@dataclass(frozen=True)
class DataRelease:
    version: str
    created_at: str
    completed_session: str
    dataset_versions: dict[str, str]
    quality: str = DataQuality.VALID
    warnings: tuple[str, ...] = ()

    def to_bytes(self) -> bytes:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()

    @classmethod
    def from_bytes(cls, value: bytes) -> "DataRelease":
        raw = json.loads(value)
        return cls(
            version=str(raw["version"]),
            created_at=str(raw["created_at"]),
            completed_session=str(raw["completed_session"]),
            dataset_versions={str(key): str(item) for key, item in raw["dataset_versions"].items()},
            quality=str(raw.get("quality", DataQuality.VALID)),
            warnings=tuple(str(item) for item in raw.get("warnings", ())),
        )

    @classmethod
    def create(
        cls,
        completed_session: str,
        dataset_versions: dict[str, str],
        *,
        quality: str = DataQuality.VALID,
        warnings: tuple[str, ...] = (),
    ) -> "DataRelease":
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return cls(
            version=f"{completed_session.replace('-', '')}-{timestamp}",
            created_at=utc_now_iso(),
            completed_session=completed_session,
            dataset_versions=dict(sorted(dataset_versions.items())),
            quality=str(quality),
            warnings=warnings,
        )


def write_atomic(path: str | Path, value: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, destination)
