#!/usr/bin/env python3
"""Repair stale lifecycle bindings with zero Parquet/economic-data changes.

The 2026-07-19 post-SYMC release contains five otherwise-correct current
datasets whose manifests retained the superseded 182-candidate lifecycle
projection.  The authoritative ``lifecycle_resolutions`` manifest and its
hash-bound archived evidence report describe the closed 181-candidate state.

This one-shot repair is deliberately metadata-only:

* plan mode is the default and performs no writes;
* apply creates an empty inherited child manifest for each affected dataset;
* no Parquet file is copied, rewritten, or added;
* every parent data-file inventory is frozen and checked before/after apply;
* current pointers and the release pointer use compare-and-swap;
* an exact rollback journal restores all pointer bytes on failure.

No provider, network, EODHD, R2, or browser access is used.
"""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import gzip
import json
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "unified_quant/src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from supertrend_quant.market_store.lifecycle_coverage import (  # noqa: E402
    lifecycle_candidate_id,
    lifecycle_candidate_set_sha256,
    lifecycle_resolution_set_sha256,
)
from supertrend_quant.market_store.manifest import (  # noqa: E402
    CurrentPointer,
    DataRelease,
    DatasetManifest,
    sha256_bytes,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import (  # noqa: E402
    LocalDatasetRepository,
)
from supertrend_quant.market_store.storage import ObjectNotFound  # noqa: E402
from supertrend_quant.market_store.validation import (  # noqa: E402
    validate_manifest_files,
)


OPERATION = "repair_us_lifecycle_manifest_metadata"
REPAIR_SCHEMA = "us_lifecycle_manifest_metadata_repair/v1"
TARGET_DATASETS = (
    "adjustment_factors",
    "daily_price_raw",
    "security_master",
    "source_archive",
    "symbol_history",
)
TRUTH_DATASET = "lifecycle_resolutions"

# One-shot pins for the exact reviewed local release.  Unit tests replace these
# pins with an equally strict tiny fixture; they are not permissive defaults.
EXPECTED_BASE_RELEASE_VERSION = "20260715-20260719T043913592428Z"
EXPECTED_BASE_RELEASE_SHA256 = (
    "9c00eabe6b34c1c05c9f8381b84655d788a3c0469709ea5fa61d3d52f637477c"
)
EXPECTED_TARGET_PARENT_VERSIONS: Mapping[str, str] = {
    "adjustment_factors": (
        "symc-nlok-identity-20260715-e39d1a1fdf0b427f9c5285ff14bb95e0-"
        "adjustment_factors"
    ),
    "daily_price_raw": (
        "symc-nlok-identity-20260715-e39d1a1fdf0b427f9c5285ff14bb95e0-"
        "daily_price_raw"
    ),
    "security_master": (
        "symc-nlok-identity-20260715-e39d1a1fdf0b427f9c5285ff14bb95e0-"
        "security_master"
    ),
    "source_archive": (
        "wiki14-price-only-20260715-da9fae6029b54a27bc8d032cecbbf40c-"
        "source_archive"
    ),
    "symbol_history": (
        "symc-nlok-identity-20260715-e39d1a1fdf0b427f9c5285ff14bb95e0-"
        "symbol_history"
    ),
}
EXPECTED_TARGET_PARENT_MANIFEST_SHA256: Mapping[str, str] = {
    "adjustment_factors": (
        "cbf53ca288c004569c862f6a62e1c905abf1c66e94f540e8ff5f697e9074f350"
    ),
    "daily_price_raw": (
        "11b5715e0182ebbe9df4577124911e683ef306cd5e0726b702549c35d965f4eb"
    ),
    "security_master": (
        "09e032d97c5cf0d0362862e8019b61f317fca3ecb4e2af19f3cc85c95a052cd0"
    ),
    "source_archive": (
        "ebf8d6006cff925c9f372305b7a9938143939659b6b4210a7b7e2c2f2870e5b3"
    ),
    "symbol_history": (
        "ae4641dba04ef9a85dae604d5f6f3b42749710b54e4d2a6f70a267223f5b3a52"
    ),
}
EXPECTED_TRUTH_VERSION = (
    "symc-nlok-identity-20260715-e39d1a1fdf0b427f9c5285ff14bb95e0-"
    "lifecycle_resolutions"
)
EXPECTED_TRUTH_MANIFEST_SHA256 = (
    "069323fafba1e7588d3d096f4f502f980ef307502074d995c5fc1ce4b8488e24"
)
EXPECTED_REPORT_SHA256 = (
    "7760962880440d35d900305061c611041d449494cf17273a5502817892860554"
)

COVERAGE_FIELDS = (
    "coverage_gate_version",
    "selection_rule",
    "candidate_set_sha256",
    "resolution_set_sha256",
    "candidate_count",
    "resolution_count",
    "applied_count",
    "exception_count",
    "open_count",
)
AUTHORITATIVE_METADATA_FIELDS = (
    *COVERAGE_FIELDS,
    "lifecycle_coverage",
    "lifecycle_candidate_set_sha256",
    "lifecycle_resolution_set_sha256",
    "evidence_report_sha256",
    "evidence_report_object_path",
    "lifecycle_evidence_report_sha256",
    "lifecycle_evidence_report_object_path",
    "lifecycle_report_release_version",
    "lifecycle_report_input_versions",
    "superseded_evidence_report_sha256",
    "current_source_report_sha256",
    "pinned_base_ancestor_report_sha256",
    "lifecycle_hints_sha256",
    "full_lifecycle_finalizer_gate_passed",
)
MAX_REPORT_BYTES = 16 * 1024 * 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_sha(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class FrozenObject:
    key: str
    data: bytes
    etag: str

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.data)


@dataclass(frozen=True)
class Inspection:
    current_release: DataRelease
    current_release_object: FrozenObject
    base_release: DataRelease
    base_release_object: FrozenObject
    pointer_objects: Mapping[str, FrozenObject]
    current_manifests: Mapping[str, DatasetManifest]
    current_manifest_sha256: Mapping[str, str]
    truth_manifest: DatasetManifest
    truth_manifest_sha256: str
    truth_bindings: Mapping[str, Any]
    report_object: FrozenObject
    report_payload_sha256: str
    report_value: Mapping[str, Any]
    data_inventories: Mapping[str, Mapping[str, Any]]
    state_sha256: str
    repaired: bool


@dataclass(frozen=True)
class PreparedRepair:
    inspection: Inspection
    planned_manifests: Mapping[str, DatasetManifest]
    planned_pointer_bytes: Mapping[str, bytes]
    planned_release: DataRelease | None
    summary: Mapping[str, Any]


def _get_object(repository: LocalDatasetRepository, key: str) -> FrozenObject:
    try:
        value = repository.objects.get(key)
    except ObjectNotFound as exc:
        raise RuntimeError(f"Required immutable object is missing: {key}") from exc
    return FrozenObject(key=key, data=value.data, etag=value.etag)


def _manifest_object(
    repository: LocalDatasetRepository,
    dataset: str,
    version: str,
) -> tuple[DatasetManifest, FrozenObject]:
    key = f"{repository.version_prefix(dataset, version)}/manifest.json"
    value = _get_object(repository, key)
    manifest = DatasetManifest.from_bytes(value.data)
    _require(
        manifest.dataset == dataset and manifest.version == version,
        f"Manifest identity changed: {dataset}/{version}",
    )
    report = validate_manifest_files(
        repository.root / repository.version_prefix(dataset, version),
        manifest,
    )
    report.raise_for_errors()
    return manifest, value


def _current_manifest_object(
    repository: LocalDatasetRepository,
    dataset: str,
) -> tuple[CurrentPointer, FrozenObject, DatasetManifest, FrozenObject]:
    pointer_value = _get_object(repository, repository.current_key(dataset))
    pointer = CurrentPointer.from_bytes(pointer_value.data)
    _require(pointer.dataset == dataset, f"Current pointer dataset changed: {dataset}")
    manifest_value = _get_object(repository, pointer.manifest_path)
    _require(
        sha256_bytes(manifest_value.data) == pointer.manifest_sha256,
        f"Current pointer manifest hash changed: {dataset}",
    )
    manifest = DatasetManifest.from_bytes(manifest_value.data)
    _require(
        manifest.dataset == dataset and manifest.version == pointer.version,
        f"Current pointer manifest identity changed: {dataset}",
    )
    report = validate_manifest_files(
        repository.root / repository.version_prefix(dataset, manifest.version),
        manifest,
    )
    report.raise_for_errors()
    return pointer, pointer_value, manifest, manifest_value


def _manifest_provenance_lineage(
    repository: LocalDatasetRepository,
    dataset: str,
    version: str,
) -> tuple[str, ...]:
    lineage: list[str] = []
    seen: set[str] = set()
    current = version
    while current:
        _require(current not in seen, f"Manifest provenance cycle: {dataset}/{current}")
        seen.add(current)
        lineage.append(current)
        manifest, _ = _manifest_object(repository, dataset, current)
        current = _text(manifest.parent_version)
    return tuple(lineage)


def _data_inventory(
    repository: LocalDatasetRepository,
    dataset: str,
    version: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = version
    while current:
        _require(current not in seen, f"Inherited manifest cycle: {dataset}/{current}")
        seen.add(current)
        manifest, manifest_value = _manifest_object(repository, dataset, current)
        for item in manifest.files:
            entries.append(
                {
                    "version": current,
                    "manifest_sha256": manifest_value.sha256,
                    "path": item.path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "row_count": item.row_count,
                    "min_session": item.min_session,
                    "max_session": item.max_session,
                }
            )
        if not bool(manifest.metadata.get("inherits_parent")):
            break
        _require(
            bool(manifest.parent_version),
            f"Inherited manifest lacks parent: {dataset}/{current}",
        )
        current = manifest.parent_version
    entries.reverse()
    projection = {
        "dataset": dataset,
        "files": entries,
        "file_count": len(entries),
        "row_count": sum(int(item["row_count"]) for item in entries),
        "size_bytes": sum(int(item["size_bytes"]) for item in entries),
    }
    projection["sha256"] = _canonical_sha(projection)
    return projection


def _read_gzip_payload(value: bytes) -> bytes:
    try:
        payload = gzip.decompress(value)
    except (OSError, EOFError) as exc:
        raise RuntimeError("Lifecycle evidence report gzip is invalid.") from exc
    _require(
        len(payload) <= MAX_REPORT_BYTES,
        "Lifecycle evidence report exceeds the fail-closed size limit.",
    )
    return payload


def _coverage_projection(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {field: copy.deepcopy(metadata.get(field)) for field in COVERAGE_FIELDS}


def _truth_bindings(metadata: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in AUTHORITATIVE_METADATA_FIELDS if field not in metadata]
    _require(not missing, "Lifecycle truth metadata is incomplete: " + ", ".join(missing))
    values = {
        field: copy.deepcopy(metadata[field])
        for field in AUTHORITATIVE_METADATA_FIELDS
    }
    coverage = _coverage_projection(metadata)
    _require(
        values["lifecycle_coverage"] == coverage,
        "Lifecycle truth nested/top-level coverage diverged.",
    )
    _require(
        values["lifecycle_candidate_set_sha256"]
        == coverage["candidate_set_sha256"],
        "Lifecycle truth candidate hashes diverged.",
    )
    _require(
        values["lifecycle_resolution_set_sha256"]
        == coverage["resolution_set_sha256"],
        "Lifecycle truth resolution hashes diverged.",
    )
    report_sha = _text(values["evidence_report_sha256"]).lower()
    report_path = _text(values["evidence_report_object_path"])
    _require(
        len(report_sha) == 64
        and all(character in "0123456789abcdef" for character in report_sha),
        "Lifecycle truth evidence report hash is invalid.",
    )
    _require(
        values["lifecycle_evidence_report_sha256"] == report_sha
        and values["lifecycle_evidence_report_object_path"] == report_path
        and report_sha in Path(report_path).name,
        "Lifecycle truth evidence report aliases diverged.",
    )
    _require(
        coverage["open_count"] == 0
        and coverage["candidate_count"] == coverage["resolution_count"]
        and coverage["candidate_count"]
        == coverage["applied_count"] + coverage["exception_count"],
        "Lifecycle truth is not exactly closed.",
    )
    return values


def _candidate_frame_from_report(report: Mapping[str, Any]) -> pd.DataFrame:
    records = report.get("records")
    _require(isinstance(records, dict), "Lifecycle evidence report has no record map.")
    rows: list[dict[str, str]] = []
    for record_key, raw in records.items():
        _require(isinstance(raw, dict), "Lifecycle evidence report record is invalid.")
        candidate = raw.get("candidate")
        _require(isinstance(candidate, dict), "Lifecycle report candidate is missing.")
        security_id = _text(candidate.get("security_id"))
        last_price_date = _text(candidate.get("last_price_date"))
        _require(
            security_id and security_id == _text(record_key) and last_price_date,
            "Lifecycle report candidate identity changed.",
        )
        rows.append(
            {
                "security_id": security_id,
                "last_price_date": last_price_date,
                "candidate_id": lifecycle_candidate_id(security_id, last_price_date),
            }
        )
    frame = pd.DataFrame(rows, columns=("security_id", "last_price_date", "candidate_id"))
    _require(
        not frame.duplicated("candidate_id").any(),
        "Lifecycle report contains duplicate candidates.",
    )
    return frame


def _report_archive_row(
    repository: LocalDatasetRepository,
    version: str,
    report_sha256: str,
) -> dict[str, Any]:
    archive = repository.read_frame("source_archive", version)
    required = {
        "archive_id",
        "dataset",
        "object_path",
        "content_type",
        "effective_date",
        "source",
        "retrieved_at",
        "source_hash",
        "source_url",
    }
    _require(
        required.issubset(archive.columns),
        "Source archive lacks exact lifecycle report provenance columns.",
    )
    matches = archive.loc[
        archive["archive_id"].astype(str).str.lower().eq(report_sha256)
        & archive["source_hash"].astype(str).str.lower().eq(report_sha256)
    ]
    _require(len(matches) == 1, "Lifecycle evidence archive row is absent or duplicated.")
    return {
        field: ("" if pd.isna(matches.iloc[0][field]) else str(matches.iloc[0][field]))
        for field in sorted(required)
    }


def _validate_truth_and_report(
    repository: LocalDatasetRepository,
    current_release: DataRelease,
    truth_manifest: DatasetManifest,
    truth_bindings: Mapping[str, Any],
) -> tuple[FrozenObject, bytes, Mapping[str, Any]]:
    report_sha = _text(truth_bindings["evidence_report_sha256"]).lower()
    _require(
        report_sha == EXPECTED_REPORT_SHA256,
        "Lifecycle evidence report is not the one-shot reviewed report.",
    )
    report_key = _text(truth_bindings["evidence_report_object_path"])
    report_object = _get_object(repository, report_key)
    payload = _read_gzip_payload(report_object.data)
    _require(
        sha256_bytes(payload) == report_sha,
        "Lifecycle evidence report payload hash changed.",
    )
    try:
        report = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Lifecycle evidence report JSON is invalid.") from exc
    _require(isinstance(report, dict), "Lifecycle evidence report must be an object.")
    report_release_version = _text(report.get("release_version"))
    report_versions = report.get("input_dataset_versions")
    _require(
        report_release_version == truth_bindings["lifecycle_report_release_version"]
        and isinstance(report_versions, dict)
        and report_versions == truth_bindings["lifecycle_report_input_versions"],
        "Lifecycle report release/input lineage changed.",
    )
    report_release_object = _get_object(
        repository, f"releases/{report_release_version}.json"
    )
    report_release = DataRelease.from_bytes(report_release_object.data)
    _require(
        report_release.version == report_release_version
        and report_release.completed_session == current_release.completed_session
        and report_release.dataset_versions == report_versions,
        "Archived lifecycle report release binding changed.",
    )
    _require(
        report_versions.get(TRUTH_DATASET) == truth_manifest.version,
        "Lifecycle report no longer binds the truth resolution manifest.",
    )
    for dataset, expected_version in report_versions.items():
        current_version = current_release.dataset_versions.get(dataset)
        _require(current_version, f"Current release lost report dataset: {dataset}")
        lineage = _manifest_provenance_lineage(
            repository, dataset, current_version
        )
        _require(
            expected_version in lineage,
            f"Current release is not a descendant of lifecycle report input: {dataset}",
        )

    candidates = _candidate_frame_from_report(report)
    coverage = truth_bindings["lifecycle_coverage"]
    _require(
        len(candidates) == int(report.get("candidate_count", -1))
        == int(coverage["candidate_count"]),
        "Lifecycle report candidate count changed.",
    )
    candidate_hash = lifecycle_candidate_set_sha256(candidates)
    _require(
        candidate_hash == _text(report.get("candidate_set_sha256"))
        == coverage["candidate_set_sha256"],
        "Lifecycle report candidate inventory hash changed.",
    )

    resolutions = repository.read_frame(TRUTH_DATASET, truth_manifest.version)
    resolution_hash = lifecycle_resolution_set_sha256(resolutions)
    _require(
        len(resolutions) == int(coverage["resolution_count"])
        and resolution_hash == coverage["resolution_set_sha256"],
        "Lifecycle resolution bytes disagree with truth metadata.",
    )
    resolution_values = resolutions["resolution"].fillna("").astype(str).str.strip()
    _require(
        int(resolution_values.eq("applied").sum()) == int(coverage["applied_count"])
        and int(resolution_values.eq("exception").sum())
        == int(coverage["exception_count"])
        and set(resolution_values) <= {"applied", "exception"},
        "Lifecycle resolution status counts changed.",
    )
    _require(
        set(resolutions["candidate_id"].astype(str))
        == set(candidates["candidate_id"].astype(str)),
        "Lifecycle report and resolution candidate sets diverged.",
    )

    input_archive_version = _text(report_versions.get("source_archive"))
    current_archive_version = current_release.dataset_versions["source_archive"]
    input_row = _report_archive_row(repository, input_archive_version, report_sha)
    current_row = _report_archive_row(repository, current_archive_version, report_sha)
    _require(input_row == current_row, "Lifecycle report source-archive row changed in lineage.")
    _require(
        input_row["dataset"] == "lifecycle_evidence_report"
        and input_row["object_path"] == report_key
        and input_row["content_type"] == "application/json"
        and input_row["effective_date"] == current_release.completed_session
        and input_row["source"] == "lifecycle_evidence_report",
        "Lifecycle report source-archive provenance is not exact.",
    )
    return report_object, payload, report


def _repair_marker(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = metadata.get("lifecycle_metadata_repair")
    return value if isinstance(value, dict) else None


def _metadata_matches_truth(
    metadata: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> bool:
    return (
        all(metadata.get(field) == truth[field] for field in AUTHORITATIVE_METADATA_FIELDS)
        and metadata.get("needs_lifecycle_refinalization") is not True
    )


def _inspect(repository: LocalDatasetRepository) -> Inspection:
    current_release_object = _get_object(repository, "releases/current.json")
    current_release = DataRelease.from_bytes(current_release_object.data)
    base_release_object = _get_object(
        repository, f"releases/{EXPECTED_BASE_RELEASE_VERSION}.json"
    )
    base_release = DataRelease.from_bytes(base_release_object.data)
    _require(
        base_release.version == EXPECTED_BASE_RELEASE_VERSION
        and base_release_object.sha256 == EXPECTED_BASE_RELEASE_SHA256,
        "One-shot base release bytes changed.",
    )
    _require(
        base_release.dataset_versions.get(TRUTH_DATASET) == EXPECTED_TRUTH_VERSION,
        "One-shot base lifecycle truth version changed.",
    )
    _require(
        set(EXPECTED_TARGET_PARENT_VERSIONS) == set(TARGET_DATASETS)
        == set(EXPECTED_TARGET_PARENT_MANIFEST_SHA256),
        "One-shot target pin inventory is incomplete.",
    )
    for dataset in TARGET_DATASETS:
        _require(
            base_release.dataset_versions.get(dataset)
            == EXPECTED_TARGET_PARENT_VERSIONS[dataset],
            f"One-shot base target version changed: {dataset}",
        )
        _, parent_value = _manifest_object(
            repository, dataset, EXPECTED_TARGET_PARENT_VERSIONS[dataset]
        )
        _require(
            parent_value.sha256 == EXPECTED_TARGET_PARENT_MANIFEST_SHA256[dataset],
            f"One-shot base target manifest changed: {dataset}",
        )

    truth_manifest, truth_value = _manifest_object(
        repository, TRUTH_DATASET, EXPECTED_TRUTH_VERSION
    )
    _require(
        truth_value.sha256 == EXPECTED_TRUTH_MANIFEST_SHA256,
        "One-shot lifecycle truth manifest bytes changed.",
    )
    truth_bindings = _truth_bindings(truth_manifest.metadata)

    pointer_objects: dict[str, FrozenObject] = {}
    current_manifests: dict[str, DatasetManifest] = {}
    current_manifest_hashes: dict[str, str] = {}
    for dataset, release_version in current_release.dataset_versions.items():
        pointer, pointer_value, manifest, manifest_value = _current_manifest_object(
            repository, dataset
        )
        _require(
            pointer.version == release_version,
            f"Current release/pointer mismatch: {dataset}",
        )
        pointer_objects[dataset] = pointer_value
        current_manifests[dataset] = manifest
        current_manifest_hashes[dataset] = manifest_value.sha256

    _require(
        current_release.dataset_versions.get(TRUTH_DATASET) == EXPECTED_TRUTH_VERSION
        and current_manifest_hashes[TRUTH_DATASET]
        == EXPECTED_TRUTH_MANIFEST_SHA256,
        "Current lifecycle truth manifest changed.",
    )
    report_object, report_payload, report_value = _validate_truth_and_report(
        repository, current_release, truth_manifest, truth_bindings
    )

    matches = {
        dataset: _metadata_matches_truth(
            current_manifests[dataset].metadata, truth_bindings
        )
        for dataset in TARGET_DATASETS
    }
    if any(matches.values()) and not all(matches.values()):
        raise RuntimeError("Partial lifecycle metadata repair state is blocked.")
    repaired = all(matches.values())
    if repaired:
        for dataset in TARGET_DATASETS:
            marker = _repair_marker(current_manifests[dataset].metadata)
            _require(
                marker is not None
                and marker.get("schema") == REPAIR_SCHEMA
                and marker.get("operation") == OPERATION
                and marker.get("base_release_version")
                == EXPECTED_BASE_RELEASE_VERSION
                and marker.get("parent_version")
                == EXPECTED_TARGET_PARENT_VERSIONS[dataset]
                and current_manifests[dataset].metadata.get("inherits_parent") is True
                and not current_manifests[dataset].files,
                f"Correct-looking metadata lacks exact repair ownership: {dataset}",
            )
    else:
        _require(
            current_release.version == EXPECTED_BASE_RELEASE_VERSION
            and current_release_object.data == base_release_object.data,
            "Unrepaired state is not the exact one-shot base release.",
        )
        for dataset in TARGET_DATASETS:
            _require(
                current_manifests[dataset].version
                == EXPECTED_TARGET_PARENT_VERSIONS[dataset]
                and current_manifest_hashes[dataset]
                == EXPECTED_TARGET_PARENT_MANIFEST_SHA256[dataset],
                f"Unrepaired target manifest is not the reviewed parent: {dataset}",
            )

    data_inventories = {
        dataset: _data_inventory(
            repository, dataset, current_release.dataset_versions[dataset]
        )
        for dataset in TARGET_DATASETS
    }
    if repaired:
        for dataset in TARGET_DATASETS:
            marker = _repair_marker(current_manifests[dataset].metadata) or {}
            _require(
                marker.get("data_inventory_sha256")
                == data_inventories[dataset]["sha256"],
                f"Metadata repair data inventory changed: {dataset}",
            )

    state_projection = {
        "current_release_sha256": current_release_object.sha256,
        "base_release_sha256": base_release_object.sha256,
        "pointers": {
            dataset: value.sha256 for dataset, value in sorted(pointer_objects.items())
        },
        "manifests": dict(sorted(current_manifest_hashes.items())),
        "truth_bindings_sha256": _canonical_sha(truth_bindings),
        "report_object_sha256": report_object.sha256,
        "report_payload_sha256": sha256_bytes(report_payload),
        "data_inventories": {
            dataset: value["sha256"]
            for dataset, value in sorted(data_inventories.items())
        },
        "repaired": repaired,
    }
    return Inspection(
        current_release=current_release,
        current_release_object=current_release_object,
        base_release=base_release,
        base_release_object=base_release_object,
        pointer_objects=pointer_objects,
        current_manifests=current_manifests,
        current_manifest_sha256=current_manifest_hashes,
        truth_manifest=truth_manifest,
        truth_manifest_sha256=truth_value.sha256,
        truth_bindings=truth_bindings,
        report_object=report_object,
        report_payload_sha256=sha256_bytes(report_payload),
        report_value=report_value,
        data_inventories=data_inventories,
        state_sha256=_canonical_sha(state_projection),
        repaired=repaired,
    )


def _metadata_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    fields = (*AUTHORITATIVE_METADATA_FIELDS, "needs_lifecycle_refinalization")
    return {
        field: {
            "before": copy.deepcopy(before.get(field)),
            "after": copy.deepcopy(after.get(field)),
        }
        for field in fields
        if before.get(field) != after.get(field)
    }


def prepare_repair(repository: LocalDatasetRepository) -> PreparedRepair:
    inspection = _inspect(repository)
    if inspection.repaired:
        summary = {
            "status": "already_repaired",
            "mode": "plan",
            "writes_required": False,
            "writes_performed": False,
            "release_version": inspection.current_release.version,
            "target_datasets": list(TARGET_DATASETS),
            "lifecycle_coverage": copy.deepcopy(
                inspection.truth_bindings["lifecycle_coverage"]
            ),
            "evidence_report_sha256": inspection.report_payload_sha256,
            "state_sha256": inspection.state_sha256,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
        return PreparedRepair(inspection, {}, {}, None, summary)

    transaction_id = uuid.uuid4().hex
    planned_versions = {
        dataset: (
            f"lifecycle-metadata-{inspection.current_release.completed_session.replace('-', '')}-"
            f"{transaction_id}-{dataset}"
        )
        for dataset in TARGET_DATASETS
    }
    planned_manifests: dict[str, DatasetManifest] = {}
    planned_pointer_bytes: dict[str, bytes] = {}
    diffs: dict[str, Any] = {}
    for dataset in TARGET_DATASETS:
        parent = inspection.current_manifests[dataset]
        metadata = copy.deepcopy(parent.metadata)
        for field, value in inspection.truth_bindings.items():
            metadata[field] = copy.deepcopy(value)
        metadata.pop("needs_lifecycle_refinalization", None)
        metadata["inherits_parent"] = True
        metadata["lifecycle_metadata_repair"] = {
            "schema": REPAIR_SCHEMA,
            "operation": OPERATION,
            "base_release_version": inspection.current_release.version,
            "base_release_sha256": inspection.current_release_object.sha256,
            "parent_version": parent.version,
            "parent_manifest_sha256": inspection.current_manifest_sha256[dataset],
            "truth_dataset": TRUTH_DATASET,
            "truth_version": inspection.truth_manifest.version,
            "truth_manifest_sha256": inspection.truth_manifest_sha256,
            "truth_bindings_sha256": _canonical_sha(inspection.truth_bindings),
            "source_archive_version": inspection.current_release.dataset_versions[
                "source_archive"
            ],
            "source_archive_manifest_sha256": inspection.current_manifest_sha256[
                "source_archive"
            ],
            "evidence_report_object_sha256": inspection.report_object.sha256,
            "evidence_report_payload_sha256": inspection.report_payload_sha256,
            "data_inventory_sha256": inspection.data_inventories[dataset]["sha256"],
            "parquet_files_added": 0,
            "economic_rows_changed": 0,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
        manifest = replace(
            parent,
            version=planned_versions[dataset],
            created_at=utc_now_iso(),
            parent_version=parent.version,
            files=(),
            metadata=metadata,
        )
        planned_manifests[dataset] = manifest
        pointer = CurrentPointer.create(
            manifest,
            f"{repository.version_prefix(dataset, manifest.version)}/manifest.json",
        )
        planned_pointer_bytes[dataset] = pointer.to_bytes()
        diffs[dataset] = _metadata_diff(parent.metadata, metadata)

    versions = dict(inspection.current_release.dataset_versions)
    versions.update(planned_versions)
    planned_release = DataRelease.create(
        inspection.current_release.completed_session,
        versions,
        quality=inspection.current_release.quality,
        warnings=inspection.current_release.warnings,
    )
    summary = {
        "status": "validated_plan",
        "mode": "plan",
        "writes_required": True,
        "writes_performed": False,
        "base_release_version": inspection.current_release.version,
        "base_release_sha256": inspection.current_release_object.sha256,
        "planned_release_version": planned_release.version,
        "planned_versions": planned_versions,
        "target_datasets": list(TARGET_DATASETS),
        "metadata_diff": diffs,
        "lifecycle_truth": {
            "dataset": TRUTH_DATASET,
            "version": inspection.truth_manifest.version,
            "manifest_sha256": inspection.truth_manifest_sha256,
            "coverage": copy.deepcopy(inspection.truth_bindings["lifecycle_coverage"]),
        },
        "source_archive_report": {
            "object_path": inspection.report_object.key,
            "stored_object_sha256": inspection.report_object.sha256,
            "payload_sha256": inspection.report_payload_sha256,
            "report_release_version": inspection.truth_bindings[
                "lifecycle_report_release_version"
            ],
        },
        "data_inventories": copy.deepcopy(inspection.data_inventories),
        "parquet_files_added": 0,
        "economic_rows_changed": 0,
        "state_sha256": inspection.state_sha256,
        "network_accessed": False,
        "eodhd_calls": 0,
        "r2_accessed": False,
    }
    return PreparedRepair(
        inspection=inspection,
        planned_manifests=planned_manifests,
        planned_pointer_bytes=planned_pointer_bytes,
        planned_release=planned_release,
        summary=summary,
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
        recovery_root = repository.root / "recovery/lifecycle-manifest-metadata"
        pending = tuple(recovery_root.glob("*.json")) if recovery_root.exists() else ()
        _require(
            not pending,
            "Lifecycle manifest metadata recovery marker blocks writes.",
        )
        yield


def _record(path: Path, value: Mapping[str, Any]) -> None:
    write_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )


def _restore_exact_object(
    repository: LocalDatasetRepository,
    key: str,
    *,
    old_data: bytes,
    new_data: bytes,
) -> None:
    observed = _get_object(repository, key)
    if observed.data == old_data:
        return
    _require(observed.data == new_data, f"Rollback ownership changed: {key}")
    repository.objects.put(key, old_data, if_match=observed.etag)


def _delete_owned_file(path: Path, expected: bytes) -> None:
    if not path.exists():
        return
    _require(path.is_file() and path.read_bytes() == expected, f"Owned file changed: {path}")
    path.unlink()


def _delete_owned_manifest_version(
    repository: LocalDatasetRepository,
    dataset: str,
    manifest: DatasetManifest,
) -> None:
    root = repository.root / repository.version_prefix(dataset, manifest.version)
    if not root.exists():
        return
    entries = tuple(root.iterdir())
    _require(
        len(entries) == 1 and entries[0].name == "manifest.json",
        f"Metadata-only version acquired foreign files: {dataset}/{manifest.version}",
    )
    _delete_owned_file(entries[0], manifest.to_bytes())
    root.rmdir()


def _rollback(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    recovery_path: Path,
) -> tuple[str, ...]:
    errors: list[str] = []
    planned_release = prepared.planned_release
    try:
        if planned_release is not None:
            _restore_exact_object(
                repository,
                "releases/current.json",
                old_data=prepared.inspection.current_release_object.data,
                new_data=planned_release.to_bytes(),
            )
    except BaseException as exc:  # pragma: no cover - exercised via injected owner loss
        errors.append(f"release_current: {type(exc).__name__}: {exc}")
    for dataset in reversed(TARGET_DATASETS):
        try:
            _restore_exact_object(
                repository,
                repository.current_key(dataset),
                old_data=prepared.inspection.pointer_objects[dataset].data,
                new_data=prepared.planned_pointer_bytes[dataset],
            )
        except BaseException as exc:  # pragma: no cover - owner-loss safeguard
            errors.append(f"{dataset}_pointer: {type(exc).__name__}: {exc}")
    if not errors and planned_release is not None:
        try:
            release_path = repository.root / f"releases/{planned_release.version}.json"
            _delete_owned_file(release_path, planned_release.to_bytes())
            for dataset in TARGET_DATASETS:
                _delete_owned_manifest_version(
                    repository, dataset, prepared.planned_manifests[dataset]
                )
        except BaseException as exc:  # pragma: no cover - owner-loss safeguard
            errors.append(f"owned_cleanup: {type(exc).__name__}: {exc}")
    if not errors and recovery_path.exists():
        recovery_path.unlink()
    return tuple(errors)


def _assert_apply_result(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> Inspection:
    inspected = _inspect(repository)
    _require(inspected.repaired, "Metadata-only repair did not become idempotent.")
    _require(
        prepared.planned_release is not None
        and inspected.current_release.to_bytes() == prepared.planned_release.to_bytes(),
        "Applied metadata-only release bytes changed.",
    )
    for dataset in TARGET_DATASETS:
        _require(
            inspected.data_inventories[dataset]
            == prepared.inspection.data_inventories[dataset],
            f"Metadata-only repair changed data inventory: {dataset}",
        )
        manifest = inspected.current_manifests[dataset]
        _require(
            not manifest.files
            and manifest.parent_version
            == prepared.inspection.current_manifests[dataset].version
            and manifest.metadata.get("inherits_parent") is True,
            f"Metadata-only child manifest shape changed: {dataset}",
        )
    _require(
        inspected.truth_manifest_sha256
        == prepared.inspection.truth_manifest_sha256
        and inspected.report_object.data == prepared.inspection.report_object.data,
        "Lifecycle truth/report changed during metadata repair.",
    )
    return inspected


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    failure_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not prepared.summary.get("writes_required"):
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    _require(prepared.planned_release is not None, "Prepared release is missing.")
    inject = failure_injector or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        current = _inspect(repository)
        _require(
            current.state_sha256 == prepared.inspection.state_sha256
            and not current.repaired,
            "Repository changed after lifecycle metadata planning.",
        )
        transaction_id = uuid.uuid4().hex
        journal_path = (
            repository.root
            / "transactions/lifecycle-manifest-metadata"
            / f"{transaction_id}.json"
        )
        recovery_path = (
            repository.root
            / "recovery/lifecycle-manifest-metadata"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "transaction_id": transaction_id,
            "status": "prepared",
            "created_at": utc_now_iso(),
            "base_release_version": prepared.inspection.current_release.version,
            "base_release_base64": base64.b64encode(
                prepared.inspection.current_release_object.data
            ).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value.data).decode("ascii")
                for dataset, value in prepared.inspection.pointer_objects.items()
                if dataset in TARGET_DATASETS
            },
            "planned_release_version": prepared.planned_release.version,
            "planned_versions": {
                dataset: manifest.version
                for dataset, manifest in prepared.planned_manifests.items()
            },
            "state_sha256": prepared.inspection.state_sha256,
        }
        _record(journal_path, journal)
        _record(recovery_path, journal)
        try:
            for dataset in TARGET_DATASETS:
                manifest = prepared.planned_manifests[dataset]
                repository.objects.put(
                    f"{repository.version_prefix(dataset, manifest.version)}/manifest.json",
                    manifest.to_bytes(),
                    if_none_match=True,
                )
                inject(f"after_{dataset}_manifest")
            for dataset in TARGET_DATASETS:
                repository.objects.put(
                    repository.current_key(dataset),
                    prepared.planned_pointer_bytes[dataset],
                    if_match=prepared.inspection.pointer_objects[dataset].etag,
                )
                inject(f"after_{dataset}_pointer")
            repository.objects.put(
                f"releases/{prepared.planned_release.version}.json",
                prepared.planned_release.to_bytes(),
                if_none_match=True,
            )
            inject("after_release_immutable")
            repository.objects.put(
                "releases/current.json",
                prepared.planned_release.to_bytes(),
                if_match=prepared.inspection.current_release_object.etag,
            )
            inject("after_release_commit")
            inspected = _assert_apply_result(repository, prepared)
            journal["status"] = "committed"
            journal["completed_at"] = utc_now_iso()
            journal["committed_release_version"] = inspected.current_release.version
            _record(journal_path, journal)
            if recovery_path.exists():
                recovery_path.unlink()
            return {
                **prepared.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": inspected.current_release.version,
                "new_dataset_versions": {
                    dataset: inspected.current_release.dataset_versions[dataset]
                    for dataset in TARGET_DATASETS
                },
                "transaction_id": transaction_id,
                "post_state_sha256": inspected.state_sha256,
            }
        except BaseException as original:
            rollback_errors = _rollback(
                repository, prepared, recovery_path=recovery_path
            )
            journal["status"] = "rollback_failed" if rollback_errors else "rolled_back"
            journal["original_error"] = f"{type(original).__name__}: {original}"
            journal["rollback_errors"] = list(rollback_errors)
            journal["completed_at"] = utc_now_iso()
            _record(journal_path, journal)
            if rollback_errors:
                raise RuntimeError(
                    f"Lifecycle metadata repair failed ({original}); rollback failed: "
                    + "; ".join(rollback_errors)
                ) from original
            raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository)
    result = apply_repair(repository, prepared) if args.apply else dict(prepared.summary)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
