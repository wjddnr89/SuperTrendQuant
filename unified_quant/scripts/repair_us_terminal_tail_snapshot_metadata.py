#!/usr/bin/env python3
"""Install terminal-tail registry metadata without changing economic data.

The reviewed 2026-07-19 release contains the same terminal-tail registry on
only three of its nine dataset manifests.  Publication correctly rejects that
partial snapshot.  This one-shot transaction creates an empty inherited child
manifest for every release dataset and installs the already code-pinned
registry consistently.

Plan mode is the default and performs no writes.  Apply mode is implemented
for a later explicitly approved run, but is protected by exact release and
manifest pins, a repository writer lock, compare-and-swap pointer updates, an
exact rollback journal, and post-commit idempotence/data-inventory checks.  It
never accesses a provider, EODHD, R2, or the network and never writes Parquet.
"""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import json
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "unified_quant/src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from supertrend_quant.market_store.cross_validation import (  # noqa: E402
    TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS,
    TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256,
    canonical_json_sha256,
    reviewed_terminal_price_tail_corrections,
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


OPERATION = "repair_us_terminal_tail_snapshot_metadata"
REPAIR_SCHEMA = "us_terminal_tail_snapshot_metadata_repair/v1"
REGISTRY_FIELD = "terminal_tail_registry_draft"
REGISTRY_SHA_FIELD = "terminal_tail_registry_inventory_sha256"
REPAIR_MARKER_FIELD = "terminal_tail_snapshot_metadata_repair"
DEFAULT_CROSS_VALIDATION_POLICY = (
    PROJECT_ROOT / "unified_quant/configs/us_cross_validation.yaml"
)

LEGACY_REGISTRY_SHA256 = (
    "119a19f588748671ca3ac2a344fd2863bc79b1c7a1680a6d88513bbe35ccb734"
)
LEGACY_REGISTRY_EVENT_IDS = frozenset(
    {
        "dd616ed974225aa84c3b537b9d21ce2a27e13a1f5af8c7e2488b765a007cac31",
        "5607a776f99741c085a54e45eddc90282d7d7fe5fe86a3b8cab2350bb7188ca7",
        "162dee2832998354021e79efcafa8a915d3c2f0744b939c5f570c93deb2f1a55",
    }
)

TARGET_DATASETS = (
    "adjustment_factors",
    "corporate_actions",
    "daily_price_raw",
    "index_constituent_anchors",
    "index_membership_events",
    "lifecycle_resolutions",
    "security_master",
    "source_archive",
    "symbol_history",
)

# Exact one-shot pins for the current validated release.  Tests replace these
# with equally strict fixture pins; they are not permissive defaults.
EXPECTED_BASE_RELEASE_VERSION = "20260715-20260719T051324634358Z"
EXPECTED_BASE_RELEASE_SHA256 = (
    "127a97a567e10eb89086fb2b7f732119c0171bb343a67d82f1b0d4b3af651736"
)
EXPECTED_PARENT_VERSIONS: Mapping[str, str] = {
    "adjustment_factors": (
        "lifecycle-metadata-20260715-cc5c8c1a37f34563a77188240741bcef-"
        "adjustment_factors"
    ),
    "corporate_actions": (
        "arnc-hwm-ticker-20260715-77b58f7a838f410e8c297dc1427fd76e-"
        "corporate_actions"
    ),
    "daily_price_raw": (
        "lifecycle-metadata-20260715-cc5c8c1a37f34563a77188240741bcef-"
        "daily_price_raw"
    ),
    "index_constituent_anchors": (
        "symc-nlok-identity-20260715-e39d1a1fdf0b427f9c5285ff14bb95e0-"
        "index_constituent_anchors"
    ),
    "index_membership_events": (
        "symc-nlok-identity-20260715-e39d1a1fdf0b427f9c5285ff14bb95e0-"
        "index_membership_events"
    ),
    "lifecycle_resolutions": (
        "symc-nlok-identity-20260715-e39d1a1fdf0b427f9c5285ff14bb95e0-"
        "lifecycle_resolutions"
    ),
    "security_master": (
        "lifecycle-metadata-20260715-cc5c8c1a37f34563a77188240741bcef-"
        "security_master"
    ),
    "source_archive": (
        "lifecycle-metadata-20260715-cc5c8c1a37f34563a77188240741bcef-"
        "source_archive"
    ),
    "symbol_history": (
        "lifecycle-metadata-20260715-cc5c8c1a37f34563a77188240741bcef-"
        "symbol_history"
    ),
}
EXPECTED_PARENT_MANIFEST_SHA256: Mapping[str, str] = {
    "adjustment_factors": (
        "1e2d68e93dcd7fbc4296cdbfa408b9d4b6eddbbfba197a1053850aee2d0243ea"
    ),
    "corporate_actions": (
        "fc6cb7be2eabf653ac23410f67bdce496839864cb0f02ee3eb7da3f1caf10c54"
    ),
    "daily_price_raw": (
        "187e20fe5718699362d6571efd5b21e717214088d8060a3d8dbfa4802c431b04"
    ),
    "index_constituent_anchors": (
        "7703760243d60ce15c9f9ef24a03d8ea7e1dbce2ab10f9f59abbe30c7706093f"
    ),
    "index_membership_events": (
        "af084ce4398c838e141e4611fc2263d1bd0ba1e8caad7797bcfba2fddda6b139"
    ),
    "lifecycle_resolutions": (
        "069323fafba1e7588d3d096f4f502f980ef307502074d995c5fc1ce4b8488e24"
    ),
    "security_master": (
        "1aac36c77074da11bd971e0edcfe532528d3a3202e10dca599b48aad7782fec9"
    ),
    "source_archive": (
        "627a7b0f7d7fed1eec253a6120f0ca724a6a5cfe5a9c17c1012509c534df1841"
    ),
    "symbol_history": (
        "c05fce0d1a359df2f39f6a14f67a911bd07cc768589a0f3ba11abc4fff5b7f74"
    ),
}


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
    parent_manifests: Mapping[str, DatasetManifest]
    registry: list[dict[str, Any]]
    registry_present_datasets: tuple[str, ...]
    base_data_inventories: Mapping[str, Mapping[str, Any]]
    current_data_inventories: Mapping[str, Mapping[str, Any]]
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
        repository.root / repository.version_prefix(dataset, version), manifest
    )
    report.raise_for_errors()
    return manifest, value


def _current_manifest_object(
    repository: LocalDatasetRepository,
    dataset: str,
) -> tuple[CurrentPointer, FrozenObject, DatasetManifest, FrozenObject]:
    pointer_value = _get_object(repository, repository.current_key(dataset))
    pointer = CurrentPointer.from_bytes(pointer_value.data)
    _require(pointer.dataset == dataset, f"Current pointer changed: {dataset}")
    manifest_value = _get_object(repository, pointer.manifest_path)
    _require(
        manifest_value.sha256 == pointer.manifest_sha256,
        f"Current manifest pointer hash changed: {dataset}",
    )
    manifest = DatasetManifest.from_bytes(manifest_value.data)
    _require(
        manifest.dataset == dataset and manifest.version == pointer.version,
        f"Current manifest identity changed: {dataset}",
    )
    report = validate_manifest_files(
        repository.root / repository.version_prefix(dataset, manifest.version),
        manifest,
    )
    report.raise_for_errors()
    return pointer, pointer_value, manifest, manifest_value


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


def _registry_presence(metadata: Mapping[str, Any]) -> bool:
    return REGISTRY_FIELD in metadata or REGISTRY_SHA_FIELD in metadata


def _code_pinned_target_registry(
    path: Path = DEFAULT_CROSS_VALIDATION_POLICY,
) -> list[dict[str, Any]]:
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(
        isinstance(policy, dict) and isinstance(policy.get("events"), dict),
        "Cross-validation policy envelope is invalid.",
    )
    corrections = reviewed_terminal_price_tail_corrections(policy["events"])
    registry = [copy.deepcopy(value) for value in corrections.values()]
    _require(
        set(corrections)
        == set(TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS)
        and canonical_json_sha256(registry)
        == TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256,
        "Target terminal-tail policy registry is not code-pinned.",
    )
    return registry


def _validated_registry_presence(
    manifests: Mapping[str, DatasetManifest],
) -> tuple[str, ...]:
    values: list[tuple[str, list[dict[str, Any]]]] = []
    for dataset, manifest in sorted(manifests.items()):
        metadata = manifest.metadata
        if not _registry_presence(metadata):
            continue
        registry = metadata.get(REGISTRY_FIELD)
        inventory_sha = str(metadata.get(REGISTRY_SHA_FIELD, "")).strip()
        _require(
            isinstance(registry, list)
            and all(isinstance(item, dict) for item in registry),
            f"Terminal-tail registry is incomplete: {dataset}",
        )
        normalized = [dict(item) for item in registry]
        event_ids = {str(item.get("event_id", "")).strip() for item in normalized}
        actual_sha = canonical_json_sha256(normalized)
        combined = bool(
            inventory_sha
            == TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
            and actual_sha
            == TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
            and event_ids
            == set(TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS)
        )
        legacy = bool(
            inventory_sha == LEGACY_REGISTRY_SHA256
            and actual_sha == LEGACY_REGISTRY_SHA256
            and event_ids == set(LEGACY_REGISTRY_EVENT_IDS)
        )
        _require(
            (combined or legacy) and len(event_ids) == len(normalized),
            f"Terminal-tail registry is not code-pinned: {dataset}",
        )
        values.append((dataset, normalized))
    _require(values, "No code-pinned terminal-tail registry is available to install.")
    _require(
        all(value == values[0][1] for _, value in values[1:]),
        "Installed terminal-tail registries disagree.",
    )
    return tuple(dataset for dataset, _ in values)


def _repair_marker(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = metadata.get(REPAIR_MARKER_FIELD)
    return value if isinstance(value, dict) else None


def _inspect(repository: LocalDatasetRepository) -> Inspection:
    _require(
        set(TARGET_DATASETS)
        == set(EXPECTED_PARENT_VERSIONS)
        == set(EXPECTED_PARENT_MANIFEST_SHA256),
        "One-shot target pin inventory is incomplete.",
    )
    base_release_object = _get_object(
        repository, f"releases/{EXPECTED_BASE_RELEASE_VERSION}.json"
    )
    base_release = DataRelease.from_bytes(base_release_object.data)
    _require(
        base_release.version == EXPECTED_BASE_RELEASE_VERSION
        and base_release_object.sha256 == EXPECTED_BASE_RELEASE_SHA256
        and base_release.dataset_versions == dict(EXPECTED_PARENT_VERSIONS)
        and set(base_release.dataset_versions) == set(TARGET_DATASETS),
        "One-shot base release bytes or dataset inventory changed.",
    )

    parent_manifests: dict[str, DatasetManifest] = {}
    for dataset in TARGET_DATASETS:
        manifest, value = _manifest_object(
            repository, dataset, EXPECTED_PARENT_VERSIONS[dataset]
        )
        _require(
            value.sha256 == EXPECTED_PARENT_MANIFEST_SHA256[dataset],
            f"One-shot parent manifest changed: {dataset}",
        )
        parent_manifests[dataset] = manifest

    current_release_object = _get_object(repository, "releases/current.json")
    current_release = DataRelease.from_bytes(current_release_object.data)
    _require(
        set(current_release.dataset_versions) == set(TARGET_DATASETS)
        and current_release.completed_session == base_release.completed_session
        and current_release.quality == base_release.quality
        and current_release.warnings == base_release.warnings,
        "Current release envelope diverged from the one-shot base.",
    )

    pointer_objects: dict[str, FrozenObject] = {}
    current_manifests: dict[str, DatasetManifest] = {}
    current_manifest_sha256: dict[str, str] = {}
    for dataset in TARGET_DATASETS:
        pointer, pointer_value, manifest, manifest_value = _current_manifest_object(
            repository, dataset
        )
        _require(
            pointer.version == current_release.dataset_versions[dataset],
            f"Current release/pointer mismatch: {dataset}",
        )
        pointer_objects[dataset] = pointer_value
        current_manifests[dataset] = manifest
        current_manifest_sha256[dataset] = manifest_value.sha256

    registry = _code_pinned_target_registry()
    present_datasets = _validated_registry_presence(current_manifests)
    repaired = set(present_datasets) == set(TARGET_DATASETS)
    if repaired:
        for dataset in TARGET_DATASETS:
            manifest = current_manifests[dataset]
            marker = _repair_marker(manifest.metadata)
            _require(
                marker is not None
                and marker.get("schema") == REPAIR_SCHEMA
                and marker.get("operation") == OPERATION
                and marker.get("base_release_version")
                == EXPECTED_BASE_RELEASE_VERSION
                and marker.get("base_release_sha256")
                == EXPECTED_BASE_RELEASE_SHA256
                and marker.get("parent_version")
                == EXPECTED_PARENT_VERSIONS[dataset]
                and marker.get("parent_manifest_sha256")
                == EXPECTED_PARENT_MANIFEST_SHA256[dataset]
                and manifest.parent_version == EXPECTED_PARENT_VERSIONS[dataset]
                and manifest.metadata.get("inherits_parent") is True
                and not manifest.files,
                f"Installed metadata lacks exact repair ownership: {dataset}",
            )
    else:
        _require(
            current_release_object.data == base_release_object.data,
            "Partial metadata state is not the exact one-shot base release.",
        )
        for dataset in TARGET_DATASETS:
            _require(
                current_manifests[dataset].version
                == EXPECTED_PARENT_VERSIONS[dataset]
                and current_manifest_sha256[dataset]
                == EXPECTED_PARENT_MANIFEST_SHA256[dataset],
                f"Partial metadata parent is not exact: {dataset}",
            )

    base_data_inventories = {
        dataset: _data_inventory(
            repository, dataset, EXPECTED_PARENT_VERSIONS[dataset]
        )
        for dataset in TARGET_DATASETS
    }
    current_data_inventories = {
        dataset: _data_inventory(
            repository, dataset, current_release.dataset_versions[dataset]
        )
        for dataset in TARGET_DATASETS
    }
    _require(
        current_data_inventories == base_data_inventories,
        "Terminal-tail metadata state changed a Parquet data inventory.",
    )
    if repaired:
        for dataset in TARGET_DATASETS:
            marker = _repair_marker(current_manifests[dataset].metadata) or {}
            _require(
                marker.get("data_inventory_sha256")
                == base_data_inventories[dataset]["sha256"],
                f"Repair data inventory pin changed: {dataset}",
            )

    state_projection = {
        "current_release_sha256": current_release_object.sha256,
        "base_release_sha256": base_release_object.sha256,
        "pointers": {
            dataset: value.sha256 for dataset, value in sorted(pointer_objects.items())
        },
        "manifests": dict(sorted(current_manifest_sha256.items())),
        "data_inventories": {
            dataset: value["sha256"]
            for dataset, value in sorted(current_data_inventories.items())
        },
        "registry_sha256": canonical_json_sha256(registry),
        "registry_present_datasets": list(present_datasets),
        "repaired": repaired,
    }
    return Inspection(
        current_release=current_release,
        current_release_object=current_release_object,
        base_release=base_release,
        base_release_object=base_release_object,
        pointer_objects=pointer_objects,
        current_manifests=current_manifests,
        current_manifest_sha256=current_manifest_sha256,
        parent_manifests=parent_manifests,
        registry=registry,
        registry_present_datasets=present_datasets,
        base_data_inventories=base_data_inventories,
        current_data_inventories=current_data_inventories,
        state_sha256=_canonical_sha(state_projection),
        repaired=repaired,
    )


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    plan_id: str | None = None,
) -> PreparedRepair:
    inspection = _inspect(repository)
    if inspection.repaired:
        summary = {
            "status": "already_repaired",
            "mode": "plan",
            "writes_required": False,
            "writes_performed": False,
            "release_version": inspection.current_release.version,
            "target_datasets": list(TARGET_DATASETS),
            "registry_inventory_sha256": (
                TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
            ),
            "parquet_files_added": 0,
            "parquet_bytes_added": 0,
            "economic_rows_changed": 0,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
            "state_sha256": inspection.state_sha256,
        }
        return PreparedRepair(inspection, {}, {}, None, summary)

    transaction_id = plan_id or uuid.uuid4().hex
    _require(
        transaction_id
        and all(
            character.isalnum() or character == "-" for character in transaction_id
        ),
        "Plan identifier is invalid.",
    )
    planned_versions = {
        dataset: (
            "terminal-tail-snapshot-metadata-"
            f"{inspection.current_release.completed_session.replace('-', '')}-"
            f"{transaction_id}-{dataset}"
        )
        for dataset in TARGET_DATASETS
    }
    planned_manifests: dict[str, DatasetManifest] = {}
    planned_pointer_bytes: dict[str, bytes] = {}
    metadata_diff: dict[str, Any] = {}
    for dataset in TARGET_DATASETS:
        parent = inspection.current_manifests[dataset]
        metadata = copy.deepcopy(parent.metadata)
        marker = {
            "schema": REPAIR_SCHEMA,
            "operation": OPERATION,
            "base_release_version": inspection.current_release.version,
            "base_release_sha256": inspection.current_release_object.sha256,
            "parent_version": parent.version,
            "parent_manifest_sha256": inspection.current_manifest_sha256[dataset],
            "registry_inventory_sha256": (
                TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
            ),
            "data_inventory_sha256": inspection.base_data_inventories[dataset][
                "sha256"
            ],
            "parquet_files_added": 0,
            "parquet_bytes_added": 0,
            "economic_rows_changed": 0,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
        metadata.update(
            {
                "operation": OPERATION,
                "input_release_version": inspection.current_release.version,
                "inherits_parent": True,
                REGISTRY_FIELD: copy.deepcopy(inspection.registry),
                REGISTRY_SHA_FIELD: (
                    TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
                ),
                REPAIR_MARKER_FIELD: marker,
                "network_accessed": False,
                "eodhd_calls": 0,
                "r2_accessed": False,
            }
        )
        manifest = replace(
            parent,
            version=planned_versions[dataset],
            created_at=utc_now_iso(),
            parent_version=parent.version,
            files=(),
            metadata=metadata,
        )
        planned_manifests[dataset] = manifest
        manifest_key = (
            f"{repository.version_prefix(dataset, manifest.version)}/manifest.json"
        )
        planned_pointer_bytes[dataset] = CurrentPointer.create(
            manifest, manifest_key
        ).to_bytes()
        metadata_diff[dataset] = {
            "registry_before": _registry_presence(parent.metadata),
            "registry_after": True,
            "parent_version": parent.version,
            "parent_manifest_sha256": inspection.current_manifest_sha256[dataset],
            "data_inventory_sha256": inspection.base_data_inventories[dataset][
                "sha256"
            ],
        }

    versions = dict(inspection.current_release.dataset_versions)
    versions.update(planned_versions)
    planned_release = DataRelease.create(
        inspection.current_release.completed_session,
        versions,
        quality=inspection.current_release.quality,
        warnings=inspection.current_release.warnings,
    )
    plan_contract = {
        "schema": REPAIR_SCHEMA,
        "operation": OPERATION,
        "base_release_version": inspection.current_release.version,
        "base_release_sha256": inspection.current_release_object.sha256,
        "target_datasets": list(TARGET_DATASETS),
        "parent_versions": dict(EXPECTED_PARENT_VERSIONS),
        "parent_manifest_sha256": dict(EXPECTED_PARENT_MANIFEST_SHA256),
        "data_inventory_sha256": {
            dataset: value["sha256"]
            for dataset, value in sorted(inspection.base_data_inventories.items())
        },
        "registry_inventory_sha256": (
            TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
        ),
        "parquet_files_added": 0,
        "parquet_bytes_added": 0,
        "economic_rows_changed": 0,
    }
    summary = {
        "status": "validated_plan",
        "mode": "plan",
        "writes_required": True,
        "writes_performed": False,
        "base_release_version": inspection.current_release.version,
        "base_release_sha256": inspection.current_release_object.sha256,
        "planned_release_version": planned_release.version,
        "planned_versions": planned_versions,
        "planned_manifest_sha256": {
            dataset: sha256_bytes(manifest.to_bytes())
            for dataset, manifest in sorted(planned_manifests.items())
        },
        "target_datasets": list(TARGET_DATASETS),
        "registry_present_before": list(inspection.registry_present_datasets),
        "registry_missing_before": sorted(
            set(TARGET_DATASETS) - set(inspection.registry_present_datasets)
        ),
        "registry_present_after": list(TARGET_DATASETS),
        "registry_inventory_sha256": (
            TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
        ),
        "metadata_diff": metadata_diff,
        "data_inventories": {
            dataset: {
                field: value[field]
                for field in ("file_count", "row_count", "size_bytes", "sha256")
            }
            for dataset, value in sorted(inspection.base_data_inventories.items())
        },
        "manifest_only_child_count": len(TARGET_DATASETS),
        "parquet_files_added": 0,
        "parquet_bytes_added": 0,
        "economic_rows_changed": 0,
        "expected_preflight_blocker_removed": (
            "Terminal-tail snapshot exception metadata is only partially installed."
        ),
        "repair_plan_sha256": _canonical_sha(plan_contract),
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
        recovery_root = repository.root / "recovery/terminal-tail-snapshot-metadata"
        pending = tuple(recovery_root.glob("*.json")) if recovery_root.exists() else ()
        _require(not pending, "Terminal-tail metadata recovery marker blocks writes.")
        yield


def _record(path: Path, value: Mapping[str, Any]) -> None:
    write_atomic(
        path,
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
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
    except BaseException as exc:  # pragma: no cover - owner-loss safeguard
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
            _delete_owned_file(
                repository.root / f"releases/{planned_release.version}.json",
                planned_release.to_bytes(),
            )
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
    _require(
        inspected.current_data_inventories
        == prepared.inspection.base_data_inventories,
        "Metadata-only repair changed a data inventory.",
    )
    for dataset in TARGET_DATASETS:
        manifest = inspected.current_manifests[dataset]
        _require(
            not manifest.files
            and manifest.parent_version
            == prepared.inspection.current_manifests[dataset].version
            and manifest.metadata.get("inherits_parent") is True,
            f"Metadata-only child manifest shape changed: {dataset}",
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
            "Repository changed after terminal-tail metadata planning.",
        )
        transaction_id = uuid.uuid4().hex
        journal_path = (
            repository.root
            / "transactions/terminal-tail-snapshot-metadata"
            / f"{transaction_id}.json"
        )
        recovery_path = (
            repository.root
            / "recovery/terminal-tail-snapshot-metadata"
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
                dataset: base64.b64encode(
                    prepared.inspection.pointer_objects[dataset].data
                ).decode("ascii")
                for dataset in TARGET_DATASETS
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
                    f"Terminal-tail metadata repair failed ({original}); rollback failed: "
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
