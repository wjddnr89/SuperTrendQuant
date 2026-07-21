#!/usr/bin/env python3
"""Attach hash-pinned issuer basis evidence to four 2019 spin events.

This repair is intentionally offline and fail-closed:

* Dow and Corteva receive the exact issuer Form 8937 basis percentages.
* The 2019 FOX/FOXA exchange is *not* assigned a fabricated percentage.  The
  SEC information statement says the distribution was reported as taxable and
  that the received FOX stock takes fair-market-value basis.  That result
  cannot be represented by ``cost_basis_fraction`` (a fraction of the old
  position's historical cost), so those two rows are explicitly annotated as
  unsupported and remain unresolved whenever the old parent is actually held.
* the default command only validates a plan; only ``--apply`` writes;
* official PDFs/HTML are accepted only at their reviewed SHA-256 and size;
* apply is protected by a repository lock, pointer/release CAS, a transaction
  journal, and rollback; there is no network, EODHD, or R2 code path.

The repair changes ``corporate_actions`` metadata and ``source_archive`` only.
Adjustment factors are asserted unchanged because tax-lot metadata has no
price-adjustment effect.
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
from typing import Any, Iterable, Mapping

import pandas as pd

from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_EVIDENCE_DIR = DEFAULT_CACHE_ROOT / "state/issuer_lifecycle"
WRITE_DATASETS = ("corporate_actions", "source_archive")
REQUIRED_DATASETS = (
    "corporate_actions",
    "source_archive",
    "adjustment_factors",
)


@dataclass(frozen=True)
class EvidenceSpec:
    key: str
    url: str
    sha256: str
    size: int
    content_type: str
    retrieved_at: str

    @property
    def extension(self) -> str:
        return "pdf" if self.content_type == "application/pdf" else "html"

    @property
    def filename(self) -> str:
        return f"{self.sha256}.{self.extension}"


EVIDENCE_SPECS = (
    EvidenceSpec(
        key="dow_form_8937",
        url=(
            "https://s23.q4cdn.com/981382065/files/doc_downloads/"
            "spinoff_faq/2019/DowDuPont-Form-8937.pdf"
        ),
        sha256="3e695bbe934f646c50068812bfb320e02f0605cf881a42a4125521611f7c6bb0",
        size=84_960,
        content_type="application/pdf",
        retrieved_at="2026-07-18T07:46:24Z",
    ),
    EvidenceSpec(
        key="dow_form_8937_attachment",
        url=(
            "https://s23.q4cdn.com/981382065/files/doc_downloads/"
            "spinoff_faq/2019/Attachment-to-DowDuPont-Form-8937.pdf"
        ),
        sha256="12011c41e79e0fa694e60e284f1cec02a947d6894a3ecd926fc2db1ae513a9b0",
        size=223_384,
        content_type="application/pdf",
        retrieved_at="2026-07-18T07:46:25Z",
    ),
    EvidenceSpec(
        key="ctva_form_8937",
        url=(
            "https://s23.q4cdn.com/116192123/files/doc_downloads/2019/07/"
            "Form-8937-Comined-Corteva-Spin-and-Rev-Stk-Splt-Final-Signed.pdf"
        ),
        sha256="a89852823b8daadad6e79dbbfdce56417f70a8e24fd1387964689d5764301783",
        size=67_821,
        content_type="application/pdf",
        retrieved_at="2026-07-18T07:46:26Z",
    ),
    EvidenceSpec(
        key="ctva_form_8937_attachment",
        url=(
            "https://s23.q4cdn.com/116192123/files/doc_downloads/2019/07/"
            "Form-8937-Attachment-Combined-Corteva-Spin-and-Rev-Stk-Splt-Final.pdf"
        ),
        sha256="3fdb48bde93e6642c9e53bf5243f279530c3f9c1b75d5410b968b9a768ccfb3e",
        size=143_930,
        content_type="application/pdf",
        retrieved_at="2026-07-18T07:46:27Z",
    ),
    EvidenceSpec(
        key="fox_information_statement",
        url=(
            "https://www.sec.gov/Archives/edgar/data/1754301/"
            "000119312519003906/d624266dex991.htm"
        ),
        sha256="3b4f08910d595f667c0e272d53adfe6b17165b1b46b40a678f0461237bcf331f",
        size=2_211_838,
        content_type="text/html",
        retrieved_at="2026-07-18T07:46:28Z",
    ),
    EvidenceSpec(
        key="fox_distribution_8k",
        url=(
            "https://www.sec.gov/Archives/edgar/data/1308161/"
            "000119312519079034/d721340d8k.htm"
        ),
        sha256="1d4f1c52c717d0845982fcb2f7afc8f86692bdd47044f50d9ba7d4d6bdcb60ae",
        size=23_019,
        content_type="text/html",
        retrieved_at="2026-07-18T07:46:28Z",
    ),
)


@dataclass(frozen=True)
class ActionSpec:
    event_id: str
    security_id: str
    effective_date: str
    new_security_id: str
    new_symbol: str
    existing_source_hash: str
    metadata: Mapping[str, Any]


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _evidence_by_key() -> dict[str, EvidenceSpec]:
    return {spec.key: spec for spec in EVIDENCE_SPECS}


def _action_specs() -> tuple[ActionSpec, ...]:
    evidence = _evidence_by_key()
    dow = evidence["dow_form_8937_attachment"]
    ctva = evidence["ctva_form_8937_attachment"]
    fox_basis = evidence["fox_information_statement"]
    fox_terms = evidence["fox_distribution_8k"]
    dow_metadata = {
        "cost_basis_fraction": 0.3356,
        "distribution_ratio": 1 / 3,
        "method": "relative_fair_market_value_average_high_low",
        "parent_basis_fraction": 0.6644,
        "prices": {"DOW": 56.36, "DWDP": 37.19},
        "source_hash": dow.sha256,
        "source_url": dow.url,
        "valuation_date": "2019-04-03",
    }
    ctva_metadata = {
        "cost_basis_fraction": 0.2586805,
        "distribution_ratio": 1 / 3,
        "method": "relative_fair_market_value_average_high_low",
        "parent_basis_fraction": 0.7413195,
        "prices": {"CTVA": 26.15, "DWDP_PRE_REVERSE_SPLIT": 24.98},
        "source_hash": ctva.sha256,
        "source_url": ctva.url,
        "valuation_date": "2019-06-04",
    }
    # This deliberately contains no cost_basis_fraction.  The SEC statement
    # specifies an FMV basis reset, which depends on the holder's old basis and
    # the market value at distribution and cannot be encoded as one constant
    # fraction of historical cost.
    fox_metadata = {
        "basis_status": "unsupported_taxable_fmv_reset",
        "child_tax_basis_method": "fair_market_value_at_distribution",
        "distribution_ratio": 1 / 3,
        "exchanged_parent_share_fraction": 0.263183,
        "holding_period_begins": "2019-03-20",
        "source_hash": fox_basis.sha256,
        "source_url": fox_basis.url,
        "terms_source_hash": fox_terms.sha256,
        "terms_source_url": fox_terms.url,
    }
    return (
        ActionSpec(
            event_id="c4352ac7ce4b7de9ed1a0bcdabd77f1822e5b3c908cd80e6c8074c066cecc5f9",
            security_id="US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3",
            effective_date="2019-04-01",
            new_security_id="US:EODHD:97d908f3-f2ea-52f4-b179-a5fc616014b6",
            new_symbol="DOW",
            existing_source_hash="d2b6f7fb864a447dbfedb007219123aa0ab020115df51908ff44d07a521398c3",
            metadata=dow_metadata,
        ),
        ActionSpec(
            event_id="3f84b5bc6aee3da9d1fd8c955445d9f7abd8c4cc0097f5c65058a406ab589288",
            security_id="US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3",
            effective_date="2019-06-01",
            new_security_id="US:EODHD:80f98754-6fdc-5892-bd6f-3361a23e5fc8",
            new_symbol="CTVA",
            existing_source_hash="ae9343609e64dcd8421f11462b8782cc8db38a130e03c983714f3c10ba8db311",
            metadata=ctva_metadata,
        ),
        ActionSpec(
            event_id="07f971ad0675a970f2c456d314bcadf7fd2cc25211759977fe9b1627a7fa7421",
            security_id="US:EODHD:acd9ed55-bf0c-5b15-b624-1a917bf6078e",
            effective_date="2019-03-19",
            new_security_id="US:EODHD:bf3b55f9-c8af-5738-9772-2aa3f5b689b8",
            new_symbol="FOX",
            existing_source_hash="8a6a25526fbbb0b147f4fdad28aca4a75aa864563c61d724d81d7ecc7a067509",
            metadata=fox_metadata,
        ),
        ActionSpec(
            event_id="655fd418ed06bc94c3fa632bf00cb05197f1afb633e2a3255febff46696eaf60",
            security_id="US:EODHD:9398e16f-425d-5a51-8720-35fba7433f28",
            effective_date="2019-03-19",
            new_security_id="US:EODHD:5c7fb4cf-793f-582b-9002-a5aa62819933",
            new_symbol="FOXA",
            existing_source_hash="8a6a25526fbbb0b147f4fdad28aca4a75aa864563c61d724d81d7ecc7a067509",
            metadata=fox_metadata,
        ),
    )


ACTION_SPECS = _action_specs()


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    frames: Mapping[str, pd.DataFrame]
    artifacts: tuple[SourceArtifact, ...]
    summary: Mapping[str, Any]


def _read_artifacts(evidence_dir: Path) -> tuple[SourceArtifact, ...]:
    artifacts: list[SourceArtifact] = []
    missing: list[str] = []
    for spec in EVIDENCE_SPECS:
        path = evidence_dir / spec.filename
        if not path.is_file():
            missing.append(str(path))
            continue
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != spec.sha256 or len(content) != spec.size:
            raise ValueError(
                f"Pinned evidence hash/size mismatch: {spec.key}; "
                f"sha256={digest}; size={len(content)}."
            )
        artifacts.append(
            SourceArtifact(
                source=f"official_spinoff_basis_{spec.key}",
                source_url=spec.url,
                retrieved_at=spec.retrieved_at,
                content=content,
                content_type=spec.content_type,
            )
        )
    if missing:
        raise FileNotFoundError(
            "Pinned official basis evidence is missing: " + ", ".join(missing)
        )
    return tuple(artifacts)


def _validate_artifacts(
    artifacts: Iterable[SourceArtifact],
) -> dict[str, SourceArtifact]:
    by_url = {artifact.source_url: artifact for artifact in artifacts}
    if len(by_url) != len(EVIDENCE_SPECS):
        raise ValueError("Official basis evidence set is incomplete or duplicated.")
    for spec in EVIDENCE_SPECS:
        artifact = by_url.get(spec.url)
        if artifact is None:
            raise ValueError(f"Official basis evidence is absent: {spec.key}.")
        if artifact.source_hash != spec.sha256 or len(artifact.content) != spec.size:
            raise ValueError(f"Official basis evidence changed: {spec.key}.")
        if artifact.content_type != spec.content_type:
            raise ValueError(f"Official basis evidence content type changed: {spec.key}.")
    return by_url


def _archive_extension(artifact: SourceArtifact) -> str:
    return "pdf" if artifact.content_type == "application/pdf" else "html"


def _append_source_archive(
    source_archive: pd.DataFrame,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> tuple[pd.DataFrame, int]:
    output = source_archive.copy()
    added = 0
    for artifact in artifacts:
        existing = output.loc[
            output["archive_id"].astype(str).eq(artifact.source_hash)
        ]
        if not existing.empty:
            exact = existing["source_url"].astype(str).eq(artifact.source_url)
            if len(existing) != 1 or not bool(exact.iloc[0]):
                raise ValueError(f"Source archive collision: {artifact.source_hash}.")
            continue
        row = {
            "archive_id": artifact.source_hash,
            "dataset": artifact.source,
            "object_path": (
                f"archives/{completed_session}/{artifact.source_hash}."
                f"{_archive_extension(artifact)}.gz"
            ),
            "content_type": artifact.content_type,
            "effective_date": completed_session,
            "source": artifact.source,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
            "source_url": artifact.source_url,
        }
        output = pd.concat([output, pd.DataFrame([row])], ignore_index=True, sort=False)
        added += 1
    return output.reset_index(drop=True), added


def _metadata_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, Mapping):
        return _canonical_json(value)
    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Existing spin-off metadata is not valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("Existing spin-off metadata is not a JSON object.")
    return _canonical_json(parsed)


def _rewrite_actions(actions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    output = actions.copy()
    if "metadata" not in output.columns:
        output["metadata"] = ""
    changed = 0
    for spec in ACTION_SPECS:
        matches = output["event_id"].astype(str).eq(spec.event_id)
        if int(matches.sum()) != 1:
            raise ValueError(
                f"Target spin-off action is not unique: {spec.new_symbol}."
            )
        index = output.index[matches][0]
        row = output.loc[index]
        ratio = pd.to_numeric(pd.Series([row.get("ratio")]), errors="coerce").iloc[0]
        exact = (
            str(row.get("security_id")) == spec.security_id
            and str(row.get("action_type")) == "spinoff"
            and str(row.get("effective_date")) == spec.effective_date
            and str(row.get("ex_date")) == spec.effective_date
            and str(row.get("new_security_id")) == spec.new_security_id
            and str(row.get("new_symbol")) == spec.new_symbol
            and bool(row.get("official"))
            and str(row.get("source_hash")) == spec.existing_source_hash
            and math.isclose(float(ratio), 1 / 3, rel_tol=0, abs_tol=1e-12)
        )
        if not exact:
            raise ValueError(f"Target spin-off terms changed: {spec.new_symbol}.")
        expected = _canonical_json(spec.metadata)
        if _metadata_text(row.get("metadata")) != expected:
            output.at[index, "metadata"] = expected
            changed += 1
    output = output.sort_values(
        ["security_id", "effective_date", "event_id"]
    ).reset_index(drop=True)
    return output, changed


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        overrides: Mapping[str, pd.DataFrame],
    ):
        self.base = base
        self.versions = dict(versions)
        self.overrides = dict(overrides)

    def current_manifest(self, dataset: str):
        return self.base.manifest_for_version(dataset, self.versions[dataset])

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        return self.base.read_frame(dataset, self.versions[dataset])


def _capture_pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}.")
        output[dataset] = etag
    return output


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    evidence_dir: Path,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    artifacts = _read_artifacts(evidence_dir)
    _validate_artifacts(artifacts)
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    factors = repository.read_frame(
        "adjustment_factors", release.dataset_versions["adjustment_factors"]
    )
    rewritten_actions, action_changes = _rewrite_actions(actions)
    rewritten_archive, archive_additions = _append_source_archive(
        archive, artifacts, completed_session=release.completed_session
    )
    frames = {
        "corporate_actions": rewritten_actions,
        "source_archive": rewritten_archive,
    }
    for dataset, frame in frames.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=release.completed_session,
            incomplete_action_policy="warn",
        ).raise_for_errors()
    candidate = _CandidateRepository(repository, release.dataset_versions, frames)
    validate_repository_snapshot(candidate).raise_for_errors()
    # Basis metadata cannot affect price adjustment factors.  Keep the exact
    # version and row inventory rather than manufacturing a redundant rewrite.
    factor_version = release.dataset_versions["adjustment_factors"]
    if factors.empty:
        raise ValueError("Current adjustment-factor dataset is unexpectedly empty.")
    status = (
        "already_applied"
        if action_changes == 0 and archive_additions == 0
        else "validated_offline_plan"
    )
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=_capture_pointer_etags(repository, release),
        frames=frames,
        artifacts=artifacts,
        summary={
            "status": status,
            "base_release_version": release.version,
            "action_metadata_rows_changed": action_changes,
            "archive_rows_added": archive_additions,
            "exact_cost_basis_fractions": {"DOW": 0.3356, "CTVA": 0.2586805},
            "unsupported_taxable_fmv_actions": ["FOX", "FOXA"],
            "adjustment_factor_version": factor_version,
            "adjustment_factors_changed": False,
            "official_evidence_count": len(artifacts),
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


def _persist_archive_payloads(
    repository: LocalDatasetRepository,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> None:
    for artifact in artifacts:
        path = (
            repository.root
            / "archives"
            / completed_session
            / f"{artifact.source_hash}.{_archive_extension(artifact)}.gz"
        )
        if path.is_file():
            if gzip.decompress(path.read_bytes()) != artifact.content:
                raise RuntimeError(f"Conflicting immutable archive payload: {path}.")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, gzip.compress(artifact.content, mtime=0))
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"Immutable archive verification failed: {path}.")


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / "recovery/us-official-spinoff-basis"
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved spin-off basis recovery marker blocks writes.")
        transactions = repository.root / "transactions/us-official-spinoff-basis"
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = str(json.loads(journal.read_bytes()).get("status", ""))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted spin-off basis transaction blocks writes: {journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_release_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, etag = repository.current_release()
    if (
        release is None
        or release.version != prepared.release.version
        or etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed during basis validation.")


def _restore_transaction_state(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            release = DataRelease.from_bytes(current.data)
            belongs = (
                bool(committed_release_version)
                and release.version == committed_release_version
            ) or all(
                release.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not belongs:
                raise RuntimeError(f"unexpected release during rollback: {release.version}")
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            old = old_pointer_bytes[dataset]
            if current.data != old:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected pointer during rollback: {pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_applied":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(repository, prepared)
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before basis apply.")
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        planned = {
            dataset: (
                "official-spinoff-basis-"
                f"{prepared.release.completed_session.replace('-', '')}-"
                f"{transaction_id}-{dataset}"
            )
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/us-official-spinoff-basis"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "us_official_spinoff_basis_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": planned,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            _persist_archive_payloads(
                repository,
                prepared.artifacts,
                completed_session=prepared.release.completed_session,
            )
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="warn",
                    metadata={
                        "operation": "repair_us_official_spinoff_basis",
                        "network_accessed": False,
                        "eodhd_calls": 0,
                        "r2_accessed": False,
                    },
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                versions[dataset] = result.manifest.version
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=(
                    DataQuality.DEGRADED
                    if prepared.release.warnings
                    else DataQuality.VALID
                ),
                warnings=prepared.release.warnings,
                expected_etag=prepared.release_etag,
            )
            validate_repository_snapshot(repository).raise_for_errors()
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
                "new_release_version": committed.version,
                "transaction_id": transaction_id,
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
                "writes_performed": True,
            }
        except BaseException as original:
            errors = _restore_transaction_state(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=committed.version if committed else "",
            )
            journal.update(
                {
                    "status": "rollback_failed" if errors else "rolled_back",
                    "original_error": f"{type(original).__name__}: {original}",
                    "rollback_errors": list(errors),
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            if errors:
                recovery = (
                    repository.root
                    / "recovery/us-official-spinoff-basis"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "Spin-off basis rollback was incomplete; recovery marker blocks "
                    f"writes: {recovery}; errors={errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach reviewed official basis metadata to 2019 US spin-offs."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository, evidence_dir=args.evidence_dir)
    result = (
        apply_repair(repository, prepared)
        if args.apply
        else {**prepared.summary, "mode": "plan", "writes_performed": False}
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
