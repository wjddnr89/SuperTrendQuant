"""Publish one validated local release to R2 and verify it from a cold cache.

This operator script deliberately has no market-data provider imports. It
never calls EODHD. ``--preflight-only`` runs every local gate without creating
an R2 client or using the network. Before any write, the normal mode reads
Cloudflare's R2 visibility state; remaining network operations are R2 object
reads and, unless ``--verify-only`` is selected, conflict-aware writes
performed by ``publish_repository``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from supertrend_quant.config import DEFAULT_DATA_CONFIG_PATH, load_data_store_config
from supertrend_quant.env import load_env
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
    sha256_bytes,
    sha256_file,
)
from supertrend_quant.market_store.cross_validation import (
    DEFAULT_OFFICIAL_LIFECYCLE_HINTS,
    TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS,
    TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256,
    TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS,
    canonical_json_sha256,
    validate_cross_validation_gate,
)
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.source_archive import validate_source_archive_id
from supertrend_quant.market_store.storage import (
    DatasetCache,
    DatasetPublisher,
    ObjectNotFound,
    ObjectStore,
    R2ObjectStore,
    publish_repository,
)
from supertrend_quant.market_store.terminal_readiness import (
    audit_release_terminal_transitions,
)
from supertrend_quant.market_store.terminal_readiness_exceptions import (
    validate_publication_terminal_readiness_exceptions,
)
from supertrend_quant.market_store.validation import (
    index_member_identity_gap_fingerprint,
    validate_dataset,
    validate_manifest_files,
    validate_repository_snapshot,
)


_TERMINAL_TAIL_WRITE_DATASETS = (
    "corporate_actions",
    "daily_price_raw",
    "adjustment_factors",
    "lifecycle_resolutions",
    "security_master",
    "symbol_history",
)


_WIKI_PRIVATE_INTERNAL_ONLY_WARNING = (
    "Kaggle Quandl WIKI licenseName=Unknown; private/internal-only; "
    "redistribution/public publication blocked."
)
_SWY_PRIVATE_INTERNAL_ONLY_WARNING = (
    "Frozen Quandl WIKI SWY evidence has formal license Unknown; private/internal "
    "use only; publication and redistribution are blocked."
)
_PRIVATE_INTERNAL_ONLY_RELEASE_WARNINGS = frozenset(
    {
        _WIKI_PRIVATE_INTERNAL_ONLY_WARNING,
        _SWY_PRIVATE_INTERNAL_ONLY_WARNING,
    }
)
_PRIVATE_INTERNAL_ONLY_LICENSE_POLICY = {
    "allowed_scope": "private_internal_only",
    "fail_closed": True,
    "formal_license_name": "Unknown",
    "local_apply_ack_required": True,
    "private_r2_publisher_ack_required_separately": True,
    "public_publication_allowed": False,
    "redistribution_allowed": False,
}
_SWY_PRIVATE_INTERNAL_ONLY_LICENSE_POLICY = {
    "allowed_scope": "private_internal_only",
    "fail_closed": True,
    "formal_license_name": "Unknown",
    "metadata_sha256": (
        "e83992cf9a4051e35f91e717616b5005c04deb4f290d366679e67b235cd9401b"
    ),
    "parent_provenance_sha256": (
        "5d99f922bb7c45afe31a473f89b441f42df0cb0769b01fdfa842353304b3d636"
    ),
    "publication_allowed": False,
    "redistribution_allowed": False,
}
_PRIVATE_INTERNAL_ONLY_PROVENANCE_SPECS = (
    {
        "dataset": "reviewed_us_wiki_price_arbitration",
        "source": "reviewed_us_wiki_price_arbitration",
        "source_url": (
            "https://www.kaggle.com/api/v1/datasets/download/"
            "marketneutral/quandl-wiki-prices-us-equites/WIKI_PRICES.csv"
        ),
        "source_hash": (
            "d73bf90641034b56b4ce42d9cef2fd4dff23a6db8c101cc7ed9b49af4c7140c8"
        ),
        "object_path": (
            "archives/2026-07-15/"
            "d73bf90641034b56b4ce42d9cef2fd4dff23a6db8c101cc7ed9b49af4c7140c8"
            ".json.gz"
        ),
        "content_type": "application/json",
        "schema": "us_wiki_price_arbitration/v1",
        "license_policy": _PRIVATE_INTERNAL_ONLY_LICENSE_POLICY,
        "warning": _WIKI_PRIVATE_INTERNAL_ONLY_WARNING,
    },
    {
        "dataset": "reviewed_us_wiki14_price_only_arbitration",
        "source": "reviewed_us_wiki14_price_only_arbitration",
        "source_url": (
            "https://www.kaggle.com/api/v1/datasets/download/"
            "marketneutral/quandl-wiki-prices-us-equites/WIKI_PRICES.csv"
        ),
        "source_hash": (
            "16691eab9edc01f626d00551ba17e922d3f869d928c13478aa0443fbc329209e"
        ),
        "object_path": (
            "archives/2026-07-15/"
            "16691eab9edc01f626d00551ba17e922d3f869d928c13478aa0443fbc329209e"
            ".json.gz"
        ),
        "content_type": "application/json",
        "schema": "us_wiki14_price_only_arbitration/v1",
        "license_policy": _PRIVATE_INTERNAL_ONLY_LICENSE_POLICY,
        "warning": _WIKI_PRIVATE_INTERNAL_ONLY_WARNING,
    },
    {
        "dataset": "reviewed_swy_wiki_history_provenance",
        "source": "reviewed_early_terminal_history_supplement",
        "source_url": (
            "https://www.kaggle.com/datasets/"
            "marketneutral/quandl-wiki-prices-us-equites"
        ),
        "source_hash": (
            "54ecaf9d279da1caa3f09af1103da4e477e215a87f2e73359d2c5cb5a99e6cbd"
        ),
        "object_path": (
            "archives/2026-07-15/"
            "54ecaf9d279da1caa3f09af1103da4e477e215a87f2e73359d2c5cb5a99e6cbd"
            ".json.gz"
        ),
        "content_type": "application/json",
        "schema": "us_early_terminal_history_swy_wiki_evidence/v1",
        "license_policy": _SWY_PRIVATE_INTERNAL_ONLY_LICENSE_POLICY,
        "warning": _SWY_PRIVATE_INTERNAL_ONLY_WARNING,
    },
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _verify_remote_private_access(store: ObjectStore) -> dict[str, Any]:
    """Run the R2-specific visibility gate before any publication write."""

    verifier = getattr(store, "verify_private_access", None)
    if not callable(verifier):
        return {"status": "not_applicable", "verification_method": "non_r2_store"}
    result = verifier(force=True) if isinstance(store, R2ObjectStore) else verifier()
    _require(
        isinstance(result, dict)
        and result.get("status") == "verified_private"
        and result.get("managed_r2_dev_enabled") is False
        and result.get("enabled_custom_domain_count") == 0,
        "R2 private-state verifier returned an invalid or non-private result.",
    )
    return dict(result)


def _validated_remote_release_supersede_versions(
    store: ObjectStore,
    local_release: DataRelease,
) -> dict[str, str]:
    """Return only the immutable older-release versions safe to supersede."""

    try:
        current_value = store.get("releases/current.json")
    except ObjectNotFound:
        return {}
    remote_release = DataRelease.from_bytes(current_value.data)
    immutable_value = store.get(f"releases/{remote_release.version}.json")
    _require(
        immutable_value.data == current_value.data,
        "Remote release current/immutable mismatch prevents supersede.",
    )
    if remote_release.version == local_release.version:
        return {}
    _require(
        remote_release.completed_session <= local_release.completed_session,
        "Remote release is newer than the validated local release; supersede blocked.",
    )
    return dict(remote_release.dataset_versions)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256_bytes(encoded)


def _terminal_tail_identity_gap_fingerprints(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> tuple[str, ...]:
    """Allow only replay gaps derived from the exact repaired-tail registry.

    The exception is not inferred from an issue code or row count.  Every
    dataset rewritten by the terminal-tail repair must embed the same complete
    registry and code-pinned inventory hash.  The NBL gap is then recomputed
    from its independently code-pinned pending-removal lineage.
    """

    manifests: list[tuple[str, Any]] = []
    metadata_presence: list[tuple[str, bool]] = []
    for dataset in _TERMINAL_TAIL_WRITE_DATASETS:
        version = release.dataset_versions.get(dataset, "")
        if not version:
            metadata_presence.append((dataset, False))
            continue
        manifest = repository.manifest_for_version(dataset, version)
        metadata = manifest.metadata
        present = bool(
            "terminal_tail_registry_draft" in metadata
            or "terminal_tail_registry_inventory_sha256" in metadata
        )
        metadata_presence.append((dataset, present))
        manifests.append((dataset, manifest))

    present_datasets = [dataset for dataset, present in metadata_presence if present]
    if not present_datasets:
        return ()
    _require(
        len(manifests) == len(_TERMINAL_TAIL_WRITE_DATASETS)
        and all(present for _, present in metadata_presence),
        "Terminal-tail snapshot exception metadata is only partially installed.",
    )

    registries: list[list[dict[str, Any]]] = []
    for dataset, manifest in manifests:
        registry = manifest.metadata.get("terminal_tail_registry_draft")
        inventory_hash = str(
            manifest.metadata.get("terminal_tail_registry_inventory_sha256", "")
        ).strip()
        _require(
            isinstance(registry, list)
            and all(isinstance(item, dict) for item in registry)
            and inventory_hash
            == TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
            and canonical_json_sha256(registry)
            == TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256,
            "Terminal-tail snapshot exception registry is not code-pinned: "
            + dataset,
        )
        registries.append([dict(item) for item in registry])

    registry = registries[0]
    _require(
        all(item == registry for item in registries[1:]),
        "Terminal-tail snapshot exception registries differ across datasets.",
    )
    by_event = {
        str(item.get("event_id", "")).strip(): item for item in registry
    }
    _require(
        len(by_event) == len(registry)
        and set(by_event)
        == set(TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS),
        "Terminal-tail snapshot exception event inventory changed.",
    )

    fingerprints: list[str] = []
    for event_id, expected in sorted(
        TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS.items()
    ):
        registry_item = by_event.get(event_id)
        _require(
            registry_item is not None
            and str(registry_item.get("symbol", "")).strip().upper() == "NBL"
            and str(registry_item.get("security_id", "")).strip()
            == expected["security_id"]
            and registry_item.get("index_removals_observed")
            == [
                {
                    "index_id": expected["index_id"],
                    "effective_date": expected["next_remove_effective_date"],
                }
            ],
            "Terminal-tail NBL snapshot exception is not bound to the exact registry.",
        )
        fingerprint = index_member_identity_gap_fingerprint(
            index_id=expected["index_id"],
            replay_date=expected["replay_date"],
            security_id=expected["security_id"],
            next_remove_event_id=expected["next_remove_event_id"],
            next_remove_effective_date=expected["next_remove_effective_date"],
            next_remove_source=expected["next_remove_source"],
            next_remove_source_hash=expected["next_remove_source_hash"],
        )
        _require(
            fingerprint == expected["fingerprint"],
            "Terminal-tail NBL snapshot exception fingerprint changed.",
        )
        fingerprints.append(fingerprint)
    _require(
        len(fingerprints) == len(set(fingerprints)),
        "Terminal-tail snapshot exception fingerprints are duplicated.",
    )
    return tuple(sorted(fingerprints))


def _release_from_current(repository: LocalDatasetRepository) -> DataRelease:
    release, _ = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    return release


def _build_release_lifecycle_candidates(
    repository: LocalDatasetRepository,
    release: DataRelease,
):
    """Use the collector/finalizer candidate inventory, including reviewed binds."""

    # These imports stay local because lifecycle's collection path imports the
    # ingest/validation stack. Publication must not introduce an import cycle.
    from supertrend_quant.market_store.lifecycle import build_lifecycle_candidates
    from supertrend_quant.market_store.official_lifecycle_evidence import (
        include_bound_official_applied_event_candidates,
        load_official_lifecycle_exception_evidence,
    )

    specs = load_official_lifecycle_exception_evidence(
        DEFAULT_OFFICIAL_LIFECYCLE_HINTS
    )
    return include_bound_official_applied_event_candidates(
        build_lifecycle_candidates(repository, release=release),
        repository,
        release,
        specs,
    )


def _validate_release_lifecycle_coverage(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    """Rebuild and close the exact lifecycle candidate set for one release."""

    # These imports stay local because lifecycle's collection path imports the
    # ingest/validation stack.  Publication must not introduce an import cycle.
    from dataclasses import asdict

    import pandas as pd

    from supertrend_quant.market_store.lifecycle_coverage import (
        validate_lifecycle_coverage,
    )

    versions = release.dataset_versions
    resolution_version = versions.get("lifecycle_resolutions")
    _require(
        bool(resolution_version),
        "Release must include lifecycle_resolutions before R2 access.",
    )
    action_version = versions.get("corporate_actions")
    _require(
        bool(action_version),
        "Release must include corporate_actions for lifecycle coverage.",
    )
    archive_version = versions.get("source_archive")
    _require(
        bool(archive_version),
        "Release must include source_archive for lifecycle evidence.",
    )
    candidate_source_datasets = (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "index_constituent_anchors",
        "index_membership_events",
    )
    missing_candidate_sources = tuple(
        dataset for dataset in candidate_source_datasets if not versions.get(dataset)
    )
    _require(
        not missing_candidate_sources,
        "Release must include lifecycle candidate source datasets: "
        + ", ".join(missing_candidate_sources),
    )

    candidate_rows = [
        asdict(item)
        for item in _build_release_lifecycle_candidates(repository, release)
    ]
    candidates = (
        pd.DataFrame(candidate_rows)
        if candidate_rows
        else pd.DataFrame(columns=("security_id", "last_price_date"))
    )
    resolutions = repository.read_frame(
        "lifecycle_resolutions",
        resolution_version,
    )
    temporary_exceptions = resolutions.loc[
        resolutions["resolution"].astype(str).eq("exception")
        & resolutions["recheck_after"].fillna("").astype(str).str.strip().ne("")
    ]
    _require(
        temporary_exceptions.empty,
        "Lifecycle temporary exceptions must be zero before R2 publication: "
        f"found {len(temporary_exceptions)}.",
    )
    actions = repository.read_frame("corporate_actions", action_version)
    report = validate_lifecycle_coverage(
        candidates,
        resolutions,
        actions,
        completed_session=release.completed_session,
    )
    issue_summary = "; ".join(
        f"{issue.code}[{issue.row_count}]" for issue in report.issues
    )
    _require(
        report.valid and report.open_count == 0,
        "Lifecycle coverage is not closed"
        + (f": {issue_summary}" if issue_summary else "."),
    )

    manifest = repository.manifest_for_version(
        "lifecycle_resolutions",
        resolution_version,
    )
    expected_metadata = report.manifest_metadata()
    coverage_metadata_keys = (
        "candidate_set_sha256",
        "candidate_count",
        "resolution_count",
        "applied_count",
        "exception_count",
        "open_count",
    )
    for key in coverage_metadata_keys:
        actual_value = manifest.metadata.get(key)
        expected_value = expected_metadata[key]
        _require(
            type(actual_value) is type(expected_value)
            and actual_value == expected_value,
            "Lifecycle resolution manifest metadata mismatch for "
            f"{key}: expected {expected_value!r}, found {actual_value!r}",
        )

    evidence_report_sha256 = str(
        manifest.metadata.get("evidence_report_sha256", "")
    ).strip()
    _require(
        bool(evidence_report_sha256),
        "Lifecycle resolution manifest requires evidence_report_sha256.",
    )
    archives = repository.read_frame("source_archive", archive_version)
    archive_ids = {
        str(value).strip()
        for value in archives.get("archive_id", pd.Series(dtype="object"))
        if pd.notna(value) and str(value).strip()
    }
    _require(
        evidence_report_sha256 in archive_ids,
        "Lifecycle evidence_report_sha256 is absent from source_archive.archive_id: "
        f"{evidence_report_sha256}",
    )
    return {
        **expected_metadata,
        "temporary_exception_count": 0,
        "evidence_report_sha256": evidence_report_sha256,
    }


def _validate_terminal_transition_readiness(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    """Block publication unless every issue is clear or exactly reviewed.

    The underlying generic audit stays strict.  Publication alone may accept
    code-pinned delayed zero-cancellation actions that are economically
    unreachable after an exact, long-prior target-index exit.  Those accepted
    rows remain visible as degraded-quality reviewed exceptions.
    """

    report = audit_release_terminal_transitions(repository, release)
    result = validate_publication_terminal_readiness_exceptions(
        repository,
        release,
        report,
    )
    if not result["ready"]:
        issues = list(result["issues"])
        preview = ", ".join(
            f"{item.get('symbol') or item.get('security_id')}:"
            f"{item.get('code')}"
            for item in issues[:8]
        )
        if len(issues) > 8:
            preview += f", +{len(issues) - 8} more"
        raise RuntimeError(
            "Terminal-transition readiness is blocked"
            + (f" for release {release.version}" if release.version else "")
            + (f": {preview}" if preview else "")
        )
    return result


def _safe_object_path(root: Path, object_path: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / object_path).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise RuntimeError(f"Archive object path escapes the cache root: {object_path}")
    return resolved


def _private_internal_only_source_archive_restrictions(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    """Inspect exact hash-pinned private/internal-only archive provenance.

    A release warning alone is sufficient to require the publisher
    acknowledgement.  For each known reviewed WIKI provenance row, the raw
    JSON bytes are independently hash-checked and parsed before its exact
    fail-closed policy is accepted.  A near-match warning or a changed known
    provenance row is an error rather than a reason to silently skip the gate.
    """

    warnings = tuple(str(item) for item in release.warnings)
    warning_set = set(warnings)
    suspicious_warnings = sorted(
        warning
        for warning in warning_set
        if warning not in _PRIVATE_INTERNAL_ONLY_RELEASE_WARNINGS
        and "wiki" in warning.casefold()
        and (
            "private/internal" in warning.casefold()
            or (
                "license" in warning.casefold()
                and "unknown" in warning.casefold()
            )
            or (
                "publication" in warning.casefold()
                and "blocked" in warning.casefold()
            )
        )
    )
    _require(
        not suspicious_warnings,
        "Unrecognized WIKI private-publication restriction warning: "
        + "; ".join(suspicious_warnings),
    )
    warning_hits = sorted(
        warning_set.intersection(_PRIVATE_INTERNAL_ONLY_RELEASE_WARNINGS)
    )

    archive_version = release.dataset_versions.get("source_archive")
    if not archive_version:
        evidence = {
            "restricted": bool(warning_hits),
            "source_archive_version": "",
            "reviewed_provenance": [],
            "release_warning_restrictions": warning_hits,
        }
        return {**evidence, "evidence_sha256": _json_sha256(evidence)}

    frame = repository.read_frame("source_archive", archive_version)
    required_columns = {
        "archive_id",
        "content_type",
        "dataset",
        "object_path",
        "source",
        "source_hash",
        "source_url",
    }
    missing = sorted(required_columns - set(frame.columns))
    _require(
        not missing,
        "Source archive lacks private-publication gate columns: "
        + ", ".join(missing),
    )

    reviewed: list[dict[str, Any]] = []
    for spec in _PRIVATE_INTERNAL_ONLY_PROVENANCE_SPECS:
        matches = frame.loc[
            frame["dataset"].astype(str).eq(spec["dataset"])
        ]
        _require(
            len(matches) <= 1,
            "Private/internal-only reviewed provenance row is duplicated: "
            + spec["dataset"],
        )
        if matches.empty:
            continue

        row = matches.iloc[0].to_dict()
        for field in (
            "source",
            "source_url",
            "source_hash",
            "object_path",
            "content_type",
        ):
            _require(
                row.get(field) == spec[field],
                "Private/internal-only reviewed provenance changed: "
                f"{spec['dataset']}.{field}",
            )
        _require(
            row.get("archive_id") == spec["source_hash"],
            "Private/internal-only reviewed provenance archive_id changed: "
            + spec["dataset"],
        )
        try:
            archive_id = validate_source_archive_id(
                row["archive_id"],
                source=row["source"],
                source_url=row["source_url"],
                source_hash=row["source_hash"],
            )
        except ValueError as exc:
            raise RuntimeError(
                "Private/internal-only reviewed provenance identity mismatch: "
                + spec["dataset"]
            ) from exc

        object_path = str(row["object_path"])
        path = _safe_object_path(repository.root, object_path)
        _require(
            path.is_file(),
            "Missing private/internal-only reviewed provenance payload: "
            + object_path,
        )
        compressed = path.read_bytes()
        try:
            raw = gzip.decompress(compressed)
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                "Invalid private/internal-only reviewed provenance gzip: "
                + object_path
            ) from exc
        _require(
            sha256_bytes(raw) == spec["source_hash"],
            "Private/internal-only reviewed provenance hash mismatch: "
            + object_path,
        )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Invalid private/internal-only reviewed provenance JSON: "
                + object_path
            ) from exc
        _require(
            isinstance(payload, dict)
            and payload.get("schema") == spec["schema"]
            and payload.get("license_policy") == spec["license_policy"],
            "Private/internal-only reviewed provenance policy changed: "
            + spec["dataset"],
        )
        _require(
            spec["warning"] in warning_set,
            "Private/internal-only release warning is missing for: "
            + spec["dataset"],
        )
        reviewed.append(
            {
                "archive_id": archive_id,
                "compressed_sha256": sha256_bytes(compressed),
                "dataset": spec["dataset"],
                "license_policy_sha256": _json_sha256(spec["license_policy"]),
                "object_path": object_path,
                "raw_sha256": spec["source_hash"],
                "schema": spec["schema"],
                "source": spec["source"],
            }
        )

    reviewed.sort(key=lambda item: item["dataset"])
    evidence = {
        "restricted": bool(reviewed or warning_hits),
        "source_archive_version": archive_version,
        "reviewed_provenance": reviewed,
        "release_warning_restrictions": warning_hits,
    }
    return {**evidence, "evidence_sha256": _json_sha256(evidence)}


def _require_private_internal_only_publisher_ack(
    restrictions: dict[str, Any],
    *,
    acknowledged: bool,
) -> dict[str, Any]:
    restricted = restrictions.get("restricted") is True
    _require(
        not restricted or acknowledged,
        "Private/internal-only source archives require explicit publisher "
        "acknowledgement; pass "
        "--ack-private-internal-only-source-archives. This acknowledgement "
        "does not replace the mandatory Cloudflare private-state verification.",
    )
    return {
        **restrictions,
        "ack_flag_supplied": bool(acknowledged),
        "publisher_acknowledged": bool(restricted and acknowledged),
    }


def _verify_archive_payloads(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    version = release.dataset_versions.get("source_archive")
    if not version:
        return {
            "payloads": 0,
            "compressed_bytes": 0,
            "raw_bytes": 0,
            "snapshot_sha256": _json_sha256([]),
        }

    frame = repository.read_frame("source_archive", version)
    provenance_columns = ("source", "source_url", "source_hash")
    required_columns = {
        "archive_id",
        "object_path",
        *provenance_columns,
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    _require(
        not missing_columns,
        "Source archive lacks publication identity columns: "
        + ", ".join(missing_columns),
    )
    duplicated_provenance = frame.duplicated(
        subset=list(provenance_columns), keep=False
    )
    _require(
        not bool(duplicated_provenance.any()),
        "Source archive contains duplicate provenance tuples: "
        f"rows={int(duplicated_provenance.sum())}",
    )
    compressed_bytes = 0
    raw_bytes = 0
    fingerprints: list[dict[str, Any]] = []
    ordered = frame.sort_values(["object_path", "archive_id"], kind="stable")
    for row in ordered.itertuples(index=False):
        object_path = str(row.object_path)
        path = _safe_object_path(repository.root, object_path)
        _require(path.is_file(), f"Missing source archive payload: {object_path}")
        payload = path.read_bytes()
        compressed_bytes += len(payload)
        raw = gzip.decompress(payload) if object_path.endswith(".gz") else payload
        raw_bytes += len(raw)
        digest = sha256_bytes(raw)
        archive_id = row.archive_id
        source_hash = row.source_hash
        source = row.source
        source_url = row.source_url
        try:
            archive_id = validate_source_archive_id(
                archive_id,
                source=source,
                source_url=source_url,
                source_hash=source_hash,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Source archive row identity mismatch: {object_path}"
            ) from exc
        _require(
            digest == source_hash,
            f"Source archive hash mismatch: {object_path}",
        )
        _require(
            Path(object_path).name.startswith(f"{source_hash}."),
            f"Source archive object path is not content-addressed: {object_path}",
        )
        fingerprints.append(
            {
                "archive_id": archive_id,
                "object_path": object_path,
                "compressed_sha256": sha256_bytes(payload),
                "raw_sha256": digest,
                "compressed_bytes": len(payload),
                "raw_bytes": len(raw),
            }
        )
    return {
        "payloads": len(frame),
        "compressed_bytes": compressed_bytes,
        "raw_bytes": raw_bytes,
        "snapshot_sha256": _json_sha256(fingerprints),
    }


class _ArchiveObjectPathPublishRepository(LocalDatasetRepository):
    """Expose each current archive object once to the storage upload loop.

    The source_archive Parquet and manifest still contain every provenance row.
    ``publish_repository`` reads the current frame without a version only when
    it enumerates immutable payload objects, so this view removes redundant R2
    conditional PUT/GET attempts without changing any dataset bytes or merge
    inputs (which are always read by explicit version).
    """

    def read_frame(self, dataset: str, version: str | None = None):
        frame = super().read_frame(dataset, version)
        if (
            dataset == "source_archive"
            and version is None
            and "object_path" in frame.columns
        ):
            return frame.drop_duplicates("object_path", keep="first").reset_index(
                drop=True
            )
        return frame


def _snapshot_fingerprint(
    repository: LocalDatasetRepository,
    release: DataRelease,
    dataset_chains: dict[str, tuple[DatasetManifest, ...]],
    archive_stats: dict[str, Any],
) -> str:
    payload = {
        "release_sha256": sha256_bytes(release.to_bytes()),
        "source_archive_sha256": archive_stats["snapshot_sha256"],
        "datasets": {
            dataset: [
                {
                    "version": manifest.version,
                    "manifest_sha256": sha256_bytes(
                        repository.objects.get(
                            f"{repository.version_prefix(dataset, manifest.version)}"
                            "/manifest.json"
                        ).data
                    ),
                    "files": [
                        {
                            "path": item.path,
                            "sha256": item.sha256,
                            "size_bytes": item.size_bytes,
                            "row_count": item.row_count,
                        }
                        for item in manifest.files
                    ],
                }
                for manifest in chain
            ]
            for dataset, chain in sorted(dataset_chains.items())
        },
    }
    return _json_sha256(payload)


def _current_state_fingerprint(
    repository: LocalDatasetRepository,
    release: DataRelease,
    snapshot_sha256: str,
) -> str:
    """Fingerprint exact release and current-pointer bytes for TOCTOU checks."""

    current_release = repository.objects.get("releases/current.json").data
    immutable_release = repository.objects.get(
        f"releases/{release.version}.json"
    ).data
    _require(
        current_release == release.to_bytes(),
        f"Local current release bytes changed: {release.version}",
    )
    _require(
        immutable_release == current_release,
        f"Local release current/immutable mismatch: {release.version}",
    )

    pointers: dict[str, Any] = {}
    for dataset, version in sorted(release.dataset_versions.items()):
        pointer_bytes = repository.objects.get(repository.current_key(dataset)).data
        pointer = CurrentPointer.from_bytes(pointer_bytes)
        expected_manifest_path = (
            f"{repository.version_prefix(dataset, version)}/manifest.json"
        )
        _require(
            pointer.dataset == dataset
            and pointer.version == version
            and pointer.manifest_path == expected_manifest_path,
            f"Local current pointer changed: {dataset}/{version}",
        )
        manifest_bytes = repository.objects.get(expected_manifest_path).data
        _require(
            sha256_bytes(manifest_bytes) == pointer.manifest_sha256,
            f"Local current manifest hash changed: {dataset}/{version}",
        )
        pointers[dataset] = {
            "pointer_sha256": sha256_bytes(pointer_bytes),
            "manifest_sha256": sha256_bytes(manifest_bytes),
        }

    return _json_sha256(
        {
            "release_current_sha256": sha256_bytes(current_release),
            "release_immutable_sha256": sha256_bytes(immutable_release),
            "snapshot_sha256": snapshot_sha256,
            "pointers": pointers,
        }
    )


def _fingerprint_release_state(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str]:
    """Rehash immutable content and exact pointers without provider access."""

    chains = {
        dataset: repository.manifest_chain(dataset, version)
        for dataset, version in release.dataset_versions.items()
    }
    for dataset, chain in chains.items():
        for manifest in chain:
            version_root = repository.root / repository.version_prefix(
                dataset, manifest.version
            )
            validate_manifest_files(version_root, manifest).raise_for_errors()
    archive_stats = _verify_archive_payloads(repository, release)
    snapshot_sha256 = _snapshot_fingerprint(
        repository,
        release,
        chains,
        archive_stats,
    )
    return {
        "snapshot_sha256": snapshot_sha256,
        "state_sha256": _current_state_fingerprint(
            repository,
            release,
            snapshot_sha256,
        ),
    }


def _validate_cross_dataset_snapshot(
    repository: LocalDatasetRepository,
    release: DataRelease,
):
    reviewed = _terminal_tail_identity_gap_fingerprints(repository, release)
    report = validate_repository_snapshot(
        repository,
        allowed_index_identity_gap_fingerprints=reviewed,
    )
    report.raise_for_errors()
    return report, reviewed


def validate_release_snapshot(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    """Strictly validate one release and every inherited local file."""

    _require(
        str(release.quality) != str(DataQuality.BLOCKED),
        f"Release is blocked: {release.version}",
    )
    current_value = repository.objects.get("releases/current.json").data
    immutable_value = repository.objects.get(
        f"releases/{release.version}.json"
    ).data
    _require(
        current_value == release.to_bytes(),
        f"Local current release changed while validating: {release.version}",
    )
    _require(
        immutable_value == current_value,
        f"Local release current/immutable mismatch: {release.version}",
    )

    dataset_stats: dict[str, Any] = {}
    dataset_chains: dict[str, tuple[DatasetManifest, ...]] = {}
    manifest_file_count = 0
    manifest_bytes = 0
    logical_rows = 0
    validation_warnings: list[dict[str, Any]] = []

    for dataset, version in release.dataset_versions.items():
        pointer_value = repository.objects.get(repository.current_key(dataset)).data
        pointer = CurrentPointer.from_bytes(pointer_value)
        _require(
            pointer.dataset == dataset and pointer.version == version,
            f"Release/current pointer mismatch: {dataset} expected {version}, "
            f"found {pointer.version}",
        )
        expected_manifest_path = (
            f"{repository.version_prefix(dataset, version)}/manifest.json"
        )
        _require(
            pointer.manifest_path == expected_manifest_path,
            f"Unexpected current manifest path for {dataset}: {pointer.manifest_path}",
        )
        current_manifest_bytes = repository.objects.get(pointer.manifest_path).data
        _require(
            sha256_bytes(current_manifest_bytes) == pointer.manifest_sha256,
            f"Local current manifest hash mismatch: {dataset}/{version}",
        )
        latest = DatasetManifest.from_bytes(current_manifest_bytes)
        _require(
            latest.dataset == dataset and latest.version == version,
            f"Malformed current manifest identity: {dataset}/{version}",
        )
        _require(
            str(latest.quality) != str(DataQuality.BLOCKED),
            f"Dataset manifest is blocked: {dataset}/{version}",
        )
        _require(
            latest.unresolved_action_count == 0,
            f"Dataset has unresolved corporate actions: {dataset}/{version} "
            f"count={latest.unresolved_action_count}",
        )
        _require(
            latest.conflict_count == 0,
            f"Dataset manifest contains conflicts: {dataset}/{version} "
            f"count={latest.conflict_count}",
        )

        chain = repository.manifest_chain(dataset, version)
        dataset_chains[dataset] = chain
        dataset_file_count = 0
        dataset_size = 0
        for manifest in chain:
            version_root = repository.root / repository.version_prefix(
                dataset, manifest.version
            )
            validate_manifest_files(version_root, manifest).raise_for_errors()
            dataset_file_count += len(manifest.files)
            dataset_size += sum(item.size_bytes for item in manifest.files)

        frame = repository.read_frame(dataset, version)
        report = validate_dataset(
            dataset,
            frame,
            incomplete_action_policy="block",
            completed_session=latest.completed_session,
        )
        report.raise_for_errors()
        validation_warnings.extend(
            {
                "dataset": dataset,
                "code": issue.code,
                "message": issue.message,
                "row_count": issue.row_count,
            }
            for issue in report.issues
            if issue.severity != "error"
        )
        rows = len(frame)
        logical_rows += rows
        manifest_file_count += dataset_file_count
        manifest_bytes += dataset_size
        dataset_stats[dataset] = {
            "version": version,
            "quality": latest.quality,
            "chain_depth": len(chain),
            "files": dataset_file_count,
            "size_bytes": dataset_size,
            "logical_rows": rows,
            "warnings": list(latest.warnings),
        }
        del frame

    cross_report, reviewed_identity_gap_fingerprints = (
        _validate_cross_dataset_snapshot(repository, release)
    )
    validation_warnings.extend(
        {
            "dataset": cross_report.dataset,
            "code": issue.code,
            "message": issue.message,
            "row_count": issue.row_count,
        }
        for issue in cross_report.issues
        if issue.severity != "error"
    )
    archive_stats = _verify_archive_payloads(repository, release)
    snapshot_sha256 = _snapshot_fingerprint(
        repository,
        release,
        dataset_chains,
        archive_stats,
    )
    state_sha256 = _current_state_fingerprint(
        repository,
        release,
        snapshot_sha256,
    )
    return {
        "version": release.version,
        "completed_session": release.completed_session,
        "quality": release.quality,
        "release_sha256": sha256_bytes(release.to_bytes()),
        "snapshot_sha256": snapshot_sha256,
        "state_sha256": state_sha256,
        "datasets": dataset_stats,
        "dataset_count": len(dataset_stats),
        "manifest_files": manifest_file_count,
        "manifest_bytes": manifest_bytes,
        "logical_rows": logical_rows,
        "source_archive": archive_stats,
        "release_warnings": list(release.warnings),
        "reviewed_index_identity_gap_fingerprints": list(
            reviewed_identity_gap_fingerprints
        ),
        "validation_warnings": validation_warnings,
    }


def validate_remote_release(
    store: ObjectStore,
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    """Read back remote pointers and compare them with the final local release."""

    remote_current = store.get("releases/current.json").data
    remote_immutable = store.get(f"releases/{release.version}.json").data
    _require(
        remote_current == release.to_bytes(),
        f"Remote current release differs from local final release: {release.version}",
    )
    _require(
        remote_immutable == remote_current,
        f"Remote release current/immutable mismatch: {release.version}",
    )

    pointers: dict[str, Any] = {}
    manifest_versions = 0
    for dataset, version in release.dataset_versions.items():
        pointer_value = store.get(DatasetPublisher.current_key(dataset)).data
        pointer = CurrentPointer.from_bytes(pointer_value)
        _require(
            pointer.dataset == dataset and pointer.version == version,
            f"Remote dataset pointer mismatch: {dataset} expected {version}, "
            f"found {pointer.version}",
        )
        expected_manifest_path = (
            f"{DatasetPublisher.version_prefix(dataset, version)}/manifest.json"
        )
        _require(
            pointer.manifest_path == expected_manifest_path,
            f"Unexpected remote manifest path for {dataset}: {pointer.manifest_path}",
        )
        manifest_value = store.get(expected_manifest_path).data
        _require(
            sha256_bytes(manifest_value) == pointer.manifest_sha256,
            f"Remote manifest pointer hash mismatch: {dataset}/{version}",
        )
        local_latest_manifest = repository.objects.get(expected_manifest_path).data
        _require(
            manifest_value == local_latest_manifest,
            f"Remote/local raw manifest mismatch: {dataset}/{version}",
        )
        manifest_hashes: dict[str, str] = {}
        for local_manifest in repository.manifest_chain(dataset, version):
            manifest_path = (
                f"{DatasetPublisher.version_prefix(dataset, local_manifest.version)}"
                "/manifest.json"
            )
            local_manifest_bytes = repository.objects.get(manifest_path).data
            remote_manifest_bytes = (
                manifest_value
                if local_manifest.version == version
                else store.get(manifest_path).data
            )
            _require(
                remote_manifest_bytes == local_manifest_bytes,
                f"Remote/local raw manifest mismatch: "
                f"{dataset}/{local_manifest.version}",
            )
            manifest_hashes[local_manifest.version] = sha256_bytes(
                remote_manifest_bytes
            )
            manifest_versions += 1
        pointers[dataset] = {
            "version": version,
            "manifest_path": pointer.manifest_path,
            "manifest_sha256": pointer.manifest_sha256,
            "manifest_chain_sha256": manifest_hashes,
        }
    return {
        "release_sha256": sha256_bytes(remote_current),
        "pointer_count": len(pointers),
        "manifest_versions": manifest_versions,
        "pointers": pointers,
    }


def compare_repository_snapshots(
    local: LocalDatasetRepository,
    fresh: LocalDatasetRepository,
    release: DataRelease,
    local_stats: dict[str, Any],
    fresh_stats: dict[str, Any],
) -> dict[str, Any]:
    _require(
        local_stats["snapshot_sha256"] == fresh_stats["snapshot_sha256"],
        "Local/fresh snapshot fingerprint mismatch.",
    )
    _require(
        local.objects.get("releases/current.json").data
        == fresh.objects.get("releases/current.json").data,
        "Local/fresh current release bytes differ.",
    )

    manifest_versions = 0
    for dataset, version in release.dataset_versions.items():
        local_chain = local.manifest_chain(dataset, version)
        fresh_chain = fresh.manifest_chain(dataset, version)
        _require(
            tuple(item.version for item in local_chain)
            == tuple(item.version for item in fresh_chain),
            f"Local/fresh manifest lineage mismatch: {dataset}/{version}",
        )
        for local_manifest, _fresh_manifest in zip(local_chain, fresh_chain):
            manifest_path = (
                f"{local.version_prefix(dataset, local_manifest.version)}"
                "/manifest.json"
            )
            _require(
                local.objects.get(manifest_path).data
                == fresh.objects.get(manifest_path).data,
                f"Local/fresh raw manifest bytes differ: "
                f"{dataset}/{local_manifest.version}",
            )
            manifest_versions += 1

    archive_payloads = 0
    source_version = release.dataset_versions.get("source_archive")
    if source_version:
        archive = local.read_frame("source_archive", source_version)
        for row in archive.itertuples(index=False):
            object_path = str(row.object_path)
            local_path = _safe_object_path(local.root, object_path)
            fresh_path = _safe_object_path(fresh.root, object_path)
            _require(
                local_path.is_file() and fresh_path.is_file(),
                f"Local/fresh archive payload missing: {object_path}",
            )
            _require(
                sha256_file(local_path) == sha256_file(fresh_path),
                f"Local/fresh compressed archive bytes differ: {object_path}",
            )
            archive_payloads += 1
    return {
        "snapshot_sha256": local_stats["snapshot_sha256"],
        "manifest_versions_compared": manifest_versions,
        "archive_payloads_compared": archive_payloads,
    }


_LOCAL_PREFLIGHT_GATE_NAMES = (
    "current_release",
    "private_archive_restrictions",
    "private_archive_publisher_ack",
    "lifecycle_coverage",
    "cross_validation",
    "terminal_transition_readiness",
    "release_snapshot",
    "release_state_fingerprint",
    "release_state_stability",
)


def _preflight_failure(exc: Exception) -> dict[str, str]:
    return {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def run_local_preflight(
    repository: LocalDatasetRepository,
    *,
    ack_private_internal_only_source_archives: bool = False,
) -> dict[str, Any]:
    """Run every publication gate that requires no network or R2 access.

    Failures are accumulated so an operator can see independent blockers in
    one offline run.  Remote privacy verification, publication, and cold-cache
    verification are intentionally not attempted here.
    """

    gates: dict[str, dict[str, Any]] = {}
    blockers: list[dict[str, str]] = []

    def run_gate(name: str, operation):
        try:
            details = operation()
        except Exception as exc:
            failure = _preflight_failure(exc)
            gates[name] = failure
            blockers.append(
                {
                    "gate": name,
                    "error_type": failure["error_type"],
                    "error": failure["error"],
                }
            )
            return None
        gates[name] = {"status": "passed", "details": details}
        return details

    release = run_gate(
        "current_release",
        lambda: _release_from_current(repository),
    )
    if release is None:
        for name in _LOCAL_PREFLIGHT_GATE_NAMES[1:]:
            gates[name] = {
                "status": "skipped",
                "reason": "current_release gate failed",
            }
        return {
            "status": "blocked",
            "mode": "preflight_only",
            "local_only": True,
            "eodhd_accessed": False,
            "release": None,
            "gates": gates,
            "blockers": blockers,
            "remaining_remote_gates": [
                "cloudflare_private_state",
                "r2_conflict_aware_publication",
                "cold_cache_redownload_and_hash_verification",
            ],
        }

    # Replace the full dataclass in operator output with its stable identity.
    gates["current_release"]["details"] = {
        "version": release.version,
        "completed_session": release.completed_session,
        "quality": release.quality,
        "dataset_count": len(release.dataset_versions),
    }

    restrictions = run_gate(
        "private_archive_restrictions",
        lambda: _private_internal_only_source_archive_restrictions(
            repository,
            release,
        ),
    )
    if restrictions is None:
        gates["private_archive_publisher_ack"] = {
            "status": "skipped",
            "reason": "private_archive_restrictions gate failed",
        }
    else:
        run_gate(
            "private_archive_publisher_ack",
            lambda: _require_private_internal_only_publisher_ack(
                restrictions,
                acknowledged=ack_private_internal_only_source_archives,
            ),
        )

    run_gate(
        "lifecycle_coverage",
        lambda: _validate_release_lifecycle_coverage(repository, release),
    )
    run_gate(
        "cross_validation",
        lambda: validate_cross_validation_gate(repository, release),
    )
    run_gate(
        "terminal_transition_readiness",
        lambda: _validate_terminal_transition_readiness(repository, release),
    )
    snapshot = run_gate(
        "release_snapshot",
        lambda: validate_release_snapshot(repository, release),
    )
    fingerprint = run_gate(
        "release_state_fingerprint",
        lambda: _fingerprint_release_state(repository, release),
    )

    if snapshot is None or fingerprint is None or restrictions is None:
        dependencies = [
            name
            for name, value in (
                ("release_snapshot", snapshot),
                ("release_state_fingerprint", fingerprint),
                ("private_archive_restrictions", restrictions),
            )
            if value is None
        ]
        gates["release_state_stability"] = {
            "status": "skipped",
            "reason": "dependent gate failed: " + ", ".join(dependencies),
        }
    else:
        def validate_stability() -> dict[str, str]:
            _require(
                fingerprint["snapshot_sha256"] == snapshot["snapshot_sha256"]
                and fingerprint["state_sha256"] == snapshot["state_sha256"],
                "Local release content or current pointers changed during "
                "offline preflight.",
            )
            _require(
                _private_internal_only_source_archive_restrictions(
                    repository,
                    release,
                )
                == restrictions,
                "Private/internal-only source archive restrictions changed "
                "during offline preflight.",
            )
            return {
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "state_sha256": snapshot["state_sha256"],
            }

        run_gate("release_state_stability", validate_stability)

    return {
        "status": "ready" if not blockers else "blocked",
        "mode": "preflight_only",
        "local_only": True,
        "eodhd_accessed": False,
        "release": {
            "version": release.version,
            "completed_session": release.completed_session,
        },
        "gates": gates,
        "blockers": blockers,
        "remaining_remote_gates": [
            "cloudflare_private_state",
            "r2_conflict_aware_publication",
            "cold_cache_redownload_and_hash_verification",
        ],
    }


def publish_and_verify(
    repository: LocalDatasetRepository,
    store: ObjectStore,
    *,
    verify_only: bool,
    keep_verify_cache: bool,
    ack_private_internal_only_source_archives: bool = False,
) -> dict[str, Any]:
    """Execute the complete no-provider publication and cold-read audit."""

    frozen_release = _release_from_current(repository)
    frozen_private_restrictions = (
        _private_internal_only_source_archive_restrictions(
            repository,
            frozen_release,
        )
    )
    private_archive_gate = _require_private_internal_only_publisher_ack(
        frozen_private_restrictions,
        acknowledged=ack_private_internal_only_source_archives,
    )
    frozen_coverage = _validate_release_lifecycle_coverage(
        repository,
        frozen_release,
    )
    frozen_cross_validation = validate_cross_validation_gate(
        repository,
        frozen_release,
    )
    frozen_terminal_readiness = _validate_terminal_transition_readiness(
        repository,
        frozen_release,
    )
    frozen_stats = validate_release_snapshot(repository, frozen_release)
    frozen_stats["lifecycle_coverage"] = frozen_coverage
    frozen_stats["cross_validation"] = frozen_cross_validation
    frozen_stats["terminal_transition_readiness"] = frozen_terminal_readiness
    frozen_fingerprint = _fingerprint_release_state(repository, frozen_release)
    _require(
        frozen_fingerprint["snapshot_sha256"] == frozen_stats["snapshot_sha256"]
        and frozen_fingerprint["state_sha256"] == frozen_stats["state_sha256"],
        "Local release content or current pointers changed after "
        "pre-publication validation.",
    )
    _require(
        _private_internal_only_source_archive_restrictions(
            repository,
            frozen_release,
        )
        == frozen_private_restrictions,
        "Private/internal-only source archive restrictions changed after "
        "pre-publication validation.",
    )

    # This is the last gate before publish_repository can issue its first R2
    # write. R2ObjectStore.put independently enforces the same cached result.
    privacy_verification = _verify_remote_private_access(store)

    publish_results: list[dict[str, Any]] = []
    if not verify_only:
        publication_repository = _ArchiveObjectPathPublishRepository(
            repository.root
        )
        supersede_versions = _validated_remote_release_supersede_versions(
            store,
            frozen_release,
        )
        results = publish_repository(
            publication_repository,
            store,
            tuple(frozen_release.dataset_versions),
            supersede_versions=supersede_versions,
        )
        publish_results = [dict(item.__dict__) for item in results]
        conflicts = [item for item in results if item.conflict]
        _require(
            not conflicts,
            "R2 publication conflict: "
            + "; ".join(
                f"{item.dataset}: {item.detail or item.version}" for item in conflicts
            ),
        )

    final_release = _release_from_current(repository)
    final_private_restrictions = (
        _private_internal_only_source_archive_restrictions(
            repository,
            final_release,
        )
    )
    _require(
        final_private_restrictions == frozen_private_restrictions,
        "Private/internal-only source archive restrictions changed during "
        "publication.",
    )
    final_coverage = _validate_release_lifecycle_coverage(
        repository,
        final_release,
    )
    final_cross_validation = validate_cross_validation_gate(
        repository,
        final_release,
    )
    final_terminal_readiness = _validate_terminal_transition_readiness(
        repository,
        final_release,
    )
    final_stats = validate_release_snapshot(repository, final_release)
    final_stats["lifecycle_coverage"] = final_coverage
    final_stats["cross_validation"] = final_cross_validation
    final_stats["terminal_transition_readiness"] = final_terminal_readiness
    final_fingerprint = _fingerprint_release_state(repository, final_release)
    _require(
        final_fingerprint["snapshot_sha256"] == final_stats["snapshot_sha256"]
        and final_fingerprint["state_sha256"] == final_stats["state_sha256"],
        "Local release content or current pointers changed after final validation.",
    )
    remote_stats = validate_remote_release(store, repository, final_release)

    verify_root = Path(
        tempfile.mkdtemp(prefix="stq-r2-verify-", dir="/tmp")
    )
    verify_root_removed = False
    try:
        downloaded_release = DatasetCache(verify_root, store).sync_release()
        _require(
            downloaded_release.to_bytes() == final_release.to_bytes(),
            "Cold-cache release differs from the final local release.",
        )
        fresh_repository = LocalDatasetRepository(verify_root)
        fresh_private_restrictions = (
            _private_internal_only_source_archive_restrictions(
                fresh_repository,
                downloaded_release,
            )
        )
        _require(
            fresh_private_restrictions == final_private_restrictions,
            "Private/internal-only source archive restrictions changed in the "
            "cold-cache download.",
        )
        fresh_coverage = _validate_release_lifecycle_coverage(
            fresh_repository,
            downloaded_release,
        )
        fresh_cross_validation = validate_cross_validation_gate(
            fresh_repository,
            downloaded_release,
        )
        fresh_terminal_readiness = _validate_terminal_transition_readiness(
            fresh_repository,
            downloaded_release,
        )
        fresh_stats = validate_release_snapshot(
            fresh_repository,
            downloaded_release,
        )
        fresh_stats["lifecycle_coverage"] = fresh_coverage
        fresh_stats["cross_validation"] = fresh_cross_validation
        fresh_stats["terminal_transition_readiness"] = fresh_terminal_readiness
        comparison = compare_repository_snapshots(
            repository,
            fresh_repository,
            final_release,
            final_stats,
            fresh_stats,
        )
    finally:
        if not keep_verify_cache:
            shutil.rmtree(verify_root)
            verify_root_removed = True

    return {
        "status": "ok",
        "mode": "verify_only" if verify_only else "publish_and_verify",
        "eodhd_accessed": False,
        "private_internal_only_source_archives": private_archive_gate,
        "r2_privacy": privacy_verification,
        "frozen_release": frozen_stats,
        "publish_results": publish_results,
        "final_release": final_stats,
        "remote": remote_stats,
        "cold_cache": {
            "path": str(verify_root),
            "kept": keep_verify_cache,
            "removed": verify_root_removed,
            "validation": fresh_stats,
        },
        "comparison": comparison,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate the current local release, optionally publish it "
            "to R2, then redownload and verify it from a new /tmp cache. "
            "This command never calls EODHD."
        )
    )
    parser.add_argument(
        "--data-config",
        default=str(DEFAULT_DATA_CONFIG_PATH),
        help="Market-data configuration containing local cache and R2 settings.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not write to R2; verify its current release against the local release.",
    )
    mode.add_argument(
        "--privacy-check-only",
        action="store_true",
        help=(
            "Perform only the fail-closed S3 and Cloudflare R2 public-domain "
            "checks. Do not read or write repository objects."
        ),
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run and report every local lifecycle, cross-validation, archive "
            "acknowledgement, snapshot, and stability gate. Do not construct "
            "an R2 client or use the network."
        ),
    )
    parser.add_argument(
        "--keep-verify-cache",
        action="store_true",
        help="Keep the newly downloaded /tmp cache after successful verification.",
    )
    parser.add_argument(
        "--ack-private-internal-only-source-archives",
        action="store_true",
        help=(
            "Explicitly acknowledge that hash-verified Unknown-license source "
            "archives may be copied only to the private/internal R2 bucket. "
            "This does not bypass the Cloudflare private-state check."
        ),
    )
    return parser


def main() -> None:
    load_env()
    args = _parser().parse_args()
    config = load_data_store_config(args.data_config)
    if getattr(args, "preflight_only", False):
        repository = LocalDatasetRepository(config.local_cache_dir)
        summary = run_local_preflight(
            repository,
            ack_private_internal_only_source_archives=(
                getattr(
                    args,
                    "ack_private_internal_only_source_archives",
                    False,
                )
            ),
        )
        if config.r2.enabled:
            summary["gates"] = {
                "r2_configuration": {
                    "status": "passed",
                    "details": {"enabled": True},
                },
                **summary["gates"],
            }
        else:
            config_blocker = {
                "gate": "r2_configuration",
                "error_type": "RuntimeError",
                "error": "R2 must be enabled in the selected data configuration.",
            }
            summary["status"] = "blocked"
            summary["gates"] = {
                "r2_configuration": {
                    "status": "failed",
                    "error_type": config_blocker["error_type"],
                    "error": config_blocker["error"],
                },
                **summary["gates"],
            }
            summary["blockers"] = [config_blocker, *summary["blockers"]]
        summary["r2"] = {
            "bucket": config.r2.bucket,
            "prefix": config.r2.prefix,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        if summary["status"] != "ready":
            raise SystemExit(1)
        return
    if not config.r2.enabled:
        raise RuntimeError("R2 must be enabled in the selected data configuration.")
    store = R2ObjectStore(config.r2)
    if args.privacy_check_only:
        summary = {
            "status": "ok",
            "mode": "privacy_check_only",
            "eodhd_accessed": False,
            "r2_privacy": store.verify_private_access(force=True),
        }
    else:
        repository = LocalDatasetRepository(config.local_cache_dir)
        summary = publish_and_verify(
            repository,
            store,
            verify_only=args.verify_only,
            keep_verify_cache=args.keep_verify_cache,
            ack_private_internal_only_source_archives=(
                getattr(
                    args,
                    "ack_private_internal_only_source_archives",
                    False,
                )
            ),
        )
    summary["r2"] = {
        "bucket": config.r2.bucket,
        "prefix": config.r2.prefix,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "eodhd_accessed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
