#!/usr/bin/env python3
"""Rebind the 2016 AA spin-off to confirmatory SEC evidence, offline.

The stored AA spin-off economics are correct, but its action provenance points
to a pre-event 2016 press release whose 1-for-3 ratio was still conditional on
shareholder approval.  A hash-pinned 2018 Form 10-K already present in the
local source archive confirms that the separation completed on 2016-11-01 and
that holders received one Alcoa Corporation share for every three parent
shares.

This repair changes only ``corporate_actions`` and ``adjustment_factors``:

* the AA action's URL/hash are rebound to the confirmatory Form 10-K;
* the complete adjustment-factor inventory is rebuilt against the planned
  corporate-actions version;
* factor keys and both economic values must remain bit-for-bit unchanged;
* the confirmatory source-archive row and decompressed payload hash are verified;
* plan mode is read-only and is the default;
* apply uses one writer lock, release/pointer CAS, immutable versions, a durable
  journal, verified rollback, and a recovery marker on rollback failure;
* there is no network, EODHD, or R2 code path.
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

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
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
WRITE_DATASETS = ("corporate_actions", "adjustment_factors")
REQUIRED_DATASETS = (
    "corporate_actions",
    "adjustment_factors",
    "daily_price_raw",
    "source_archive",
)
TRANSACTION_DIR = "transactions/us-aa-spinoff-provenance"
RECOVERY_DIR = "recovery/us-aa-spinoff-provenance"
OPERATION = "repair_us_aa_spinoff_provenance"
REPAIR_REVIEWED_AT = "2026-07-18T11:30:00Z"

AA_EVENT_ID = "fd8e8f6bc37de73342db04090624676b30b75765657bd2867bd41fd62fffd187"
PARENT_SECURITY_ID = "US:EODHD:f5daeed5-d1a2-5279-aa49-8c06c902b97f"
AA_SECURITY_ID = "US:EODHD:a0eebd04-b9d4-54bc-8682-899643216993"
EFFECTIVE_DATE = "2016-11-01"
RECORD_DATE = "2016-10-20"
RATIO = 1.0 / 3.0
OLD_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/4281/"
    "000119312516731663/d249430dex991.htm"
)
OLD_SOURCE_HASH = (
    "3e79667b1b4efd2981b0aa2137ec9e83239661960b8ed88d7ecca4f594d18e49"
)
CONFIRMATORY_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/4281/"
    "000000428119000031/form10k_4q18.htm"
)
CONFIRMATORY_SOURCE_HASH = (
    "91d56adf0fbb9e55403527282600c22c3ad140838e0a037b5ff4df1d3a686a3f"
)
ACTION_RETRIEVED_AT = "2026-07-18T02:25:18.591534Z"
EVIDENCE_RETRIEVED_AT = "2026-07-18T02:19:25.834601Z"
EXPECTED_METADATA = json.dumps(
    {
        "average_prices": {"AA": 22.67, "ARNC": 20.59},
        "cost_basis_fraction": 0.2686,
        "currency": "USD",
        "method": "relative_fair_market_value_average_high_low",
        "parent_cost_basis_fraction": 0.7314,
        "source_hash": (
            "527dfea5f6529bd0154cc351c7fb9066c7b6e7b7db5fac39456c2d372593b9ec"
        ),
        "source_url": (
            "https://www.howmet.com/wp-content/uploads/sites/3/2023/05/"
            "Tax-Basis-Information-for-Shares-after-the-Separation.pdf"
        ),
        "valuation_date": EFFECTIVE_DATE,
    },
    sort_keys=True,
    separators=(",", ":"),
)


@dataclass(frozen=True)
class EvidenceSpec:
    old_url: str
    old_hash: str
    confirmatory_url: str
    confirmatory_hash: str
    archive_dataset: str
    archive_source: str
    content_type: str
    retrieved_at: str
    extension: str

    def object_path(self, completed_session: str) -> str:
        return (
            f"archives/{completed_session}/{self.confirmatory_hash}."
            f"{self.extension}.gz"
        )


DEFAULT_EVIDENCE = EvidenceSpec(
    old_url=OLD_SOURCE_URL,
    old_hash=OLD_SOURCE_HASH,
    confirmatory_url=CONFIRMATORY_SOURCE_URL,
    confirmatory_hash=CONFIRMATORY_SOURCE_HASH,
    archive_dataset="official_identity_evidence_raw",
    archive_source="official_identity_evidence_raw",
    content_type="text/html",
    retrieved_at=EVIDENCE_RETRIEVED_AT,
    extension="html",
)


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    planned_versions: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    evidence: EvidenceSpec
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


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


def _canonical_metadata(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("AA spin-off metadata is not valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("AA spin-off metadata must be a JSON object.")
    return json.dumps(
        dict(parsed), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _safe_archive_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    target = (base / object_path).resolve()
    if target == base or base not in target.parents:
        raise ValueError(f"AA evidence archive path escapes repository: {object_path}.")
    return target


def _verify_confirmatory_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
    evidence: EvidenceSpec,
) -> bytes:
    expected_path = evidence.object_path(completed_session)
    related = (
        archive["archive_id"].astype(str).eq(evidence.confirmatory_hash)
        | archive["source_hash"].astype(str).eq(evidence.confirmatory_hash)
        | archive["object_path"].astype(str).eq(expected_path)
        | archive.get(
            "source_url", pd.Series(index=archive.index, dtype="object")
        ).astype(str).eq(evidence.confirmatory_url)
    )
    rows = archive.loc[related]
    if len(rows) != 1:
        raise ValueError(
            "Current release must contain exactly one confirmatory AA evidence row; "
            f"found {len(rows)}."
        )
    row = rows.iloc[0]
    expected = {
        "archive_id": evidence.confirmatory_hash,
        "dataset": evidence.archive_dataset,
        "object_path": expected_path,
        "content_type": evidence.content_type,
        "effective_date": completed_session,
        "source": evidence.archive_source,
        "retrieved_at": evidence.retrieved_at,
        "source_hash": evidence.confirmatory_hash,
        "source_url": evidence.confirmatory_url,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if (
            _date(row.get(key)) != value
            if key == "effective_date"
            else _text(row.get(key)) != value
        )
    ]
    if mismatches:
        raise ValueError(
            "Confirmatory AA source_archive row changed: "
            + ", ".join(mismatches)
            + "."
        )
    path = _safe_archive_path(repository.root, expected_path)
    if not path.is_file():
        raise FileNotFoundError(f"Confirmatory AA evidence payload is missing: {path}.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError("Confirmatory AA evidence payload is invalid gzip.") from exc
    observed = hashlib.sha256(payload).hexdigest()
    if observed != evidence.confirmatory_hash:
        raise ValueError(
            "Confirmatory AA evidence payload hash changed: "
            f"expected={evidence.confirmatory_hash}; observed={observed}."
        )
    return payload


def _exact_action_terms(row: Mapping[str, Any]) -> bool:
    ratio = pd.to_numeric(pd.Series([row.get("ratio")]), errors="coerce").iloc[0]
    return (
        _text(row.get("event_id")) == AA_EVENT_ID
        and _text(row.get("security_id")) == PARENT_SECURITY_ID
        and _text(row.get("action_type")) == "spinoff"
        and _date(row.get("effective_date")) == EFFECTIVE_DATE
        and _date(row.get("ex_date")) == EFFECTIVE_DATE
        and _date(row.get("announcement_date")) == ""
        and _date(row.get("record_date")) == RECORD_DATE
        and _date(row.get("payment_date")) == EFFECTIVE_DATE
        and _text(row.get("cash_amount")) == ""
        and pd.notna(ratio)
        and float(ratio) == RATIO
        and _text(row.get("currency")) == "USD"
        and _text(row.get("new_security_id")) == AA_SECURITY_ID
        and _text(row.get("new_symbol")) == "AA"
        and isinstance(row.get("official"), (bool, np.bool_))
        and bool(row.get("official"))
        and _text(row.get("source_kind")) == "official_filing"
        and _text(row.get("source")) == "official_identity_repair"
        and _text(row.get("retrieved_at")) == ACTION_RETRIEVED_AT
        and _canonical_metadata(row.get("metadata")) == EXPECTED_METADATA
    )


def _rewrite_action_provenance(
    actions: pd.DataFrame,
    *,
    evidence: EvidenceSpec,
) -> tuple[pd.DataFrame, bool]:
    matches = actions["event_id"].astype(str).eq(AA_EVENT_ID)
    if int(matches.sum()) != 1:
        raise ValueError(
            f"Current release must contain exactly one AA spin-off event; found {int(matches.sum())}."
        )
    index = actions.index[matches][0]
    row = actions.loc[index]
    if not _exact_action_terms(row):
        raise ValueError("AA spin-off economic/identity terms changed.")
    observed = (_text(row.get("source_url")), _text(row.get("source_hash")))
    old_pair = (evidence.old_url, evidence.old_hash)
    new_pair = (evidence.confirmatory_url, evidence.confirmatory_hash)
    if observed not in {old_pair, new_pair}:
        raise ValueError(
            "AA spin-off provenance is neither the reviewed conditional source nor "
            f"the confirmatory source: {observed}."
        )
    changed = observed == old_pair
    output = actions.copy(deep=True)
    output.at[index, "source_url"] = evidence.confirmatory_url
    output.at[index, "source_hash"] = evidence.confirmatory_hash
    if len(output) != len(actions) or list(output.columns) != list(actions.columns):
        raise AssertionError("AA action repair changed corporate-actions topology.")
    for column in actions.columns:
        if column in {"source_url", "source_hash"}:
            continue
        if not actions[column].equals(output[column]):
            raise AssertionError(f"AA action repair changed non-provenance field: {column}.")
    changed_rows = (
        actions[["source_url", "source_hash"]]
        .fillna("")
        .astype(str)
        .ne(output[["source_url", "source_hash"]].fillna("").astype(str))
        .any(axis=1)
    )
    if int(changed_rows.sum()) != int(changed):
        raise AssertionError("AA action repair changed an unexpected action row.")
    return output, changed


def _adjustment_source_version(
    daily_price_version: str, corporate_actions_version: str
) -> str:
    if not daily_price_version or not corporate_actions_version:
        raise RuntimeError(
            "AA factor provenance requires exact price and action versions."
        )
    return f"{daily_price_version}+{corporate_actions_version}"


def _new_planned_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"aa-spinoff-provenance-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


def _expected_factors(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
    columns: list[str],
) -> pd.DataFrame:
    output = build_adjustment_factors(
        prices, actions, source_version=source_version
    ).reindex(columns=columns)
    output["source_version"] = source_version
    output["calculated_at"] = REPAIR_REVIEWED_AT
    output["source"] = "derived"
    output["retrieved_at"] = REPAIR_REVIEWED_AT
    output["source_hash"] = source_version
    return output


def _assert_factor_economics_unchanged(
    current: pd.DataFrame, expected: pd.DataFrame
) -> None:
    keys = ["security_id", "session"]
    values = ["split_factor", "total_return_factor"]
    left = current[keys + values].copy()
    right = expected[keys + values].copy()
    for frame in (left, right):
        frame["security_id"] = frame["security_id"].astype(str)
        frame["session"] = pd.to_datetime(
            frame["session"], errors="raise"
        ).dt.normalize()
        frame.sort_values(keys, inplace=True, ignore_index=True)
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise ValueError("AA provenance repair would change factor key inventory.")
    changed = np.zeros(len(left), dtype=bool)
    for column in values:
        old = pd.to_numeric(left[column], errors="raise").to_numpy(dtype=float)
        new = pd.to_numeric(right[column], errors="raise").to_numpy(dtype=float)
        changed |= ~((old == new) | (np.isnan(old) & np.isnan(new)))
    if changed.any():
        sample = left.loc[changed, keys].head(10).to_dict("records")
        raise ValueError(
            "AA provenance-only repair would change adjustment economics: "
            + json.dumps(sample, default=str, sort_keys=True)
        )


def _rebind_factors(
    current: pd.DataFrame,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
) -> tuple[pd.DataFrame, bool]:
    expected = _expected_factors(
        prices,
        actions,
        source_version=source_version,
        columns=list(current.columns),
    )
    _assert_factor_economics_unchanged(current, expected)
    exact_provenance = (
        set(current["source_version"].astype(str)) == {source_version}
        and set(current["source_hash"].astype(str)) == {source_version}
        and set(current["source"].astype(str)) == {"derived"}
    )
    return (
        current.reset_index(drop=True)
        if exact_provenance
        else expected.reset_index(drop=True),
        not exact_provenance,
    )


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
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        version = self.versions.get(dataset)
        return self.base.read_frame(dataset, version) if version else pd.DataFrame()


def _capture_pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}.")
        output[dataset] = etag
    return output


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    evidence: EvidenceSpec = DEFAULT_EVIDENCE,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    pointer_etags = _capture_pointer_etags(repository, release)
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    _verify_confirmatory_evidence(
        repository,
        frames["source_archive"],
        completed_session=release.completed_session,
        evidence=evidence,
    )
    actions, action_changed = _rewrite_action_provenance(
        frames["corporate_actions"], evidence=evidence
    )
    current_lineage = _adjustment_source_version(
        release.dataset_versions["daily_price_raw"],
        release.dataset_versions["corporate_actions"],
    )
    planned_versions: dict[str, str] = {}
    if action_changed:
        planned_versions = _new_planned_versions(release)
        factor_lineage = _adjustment_source_version(
            release.dataset_versions["daily_price_raw"],
            planned_versions["corporate_actions"],
        )
        factors, factor_rebound = _rebind_factors(
            frames["adjustment_factors"],
            frames["daily_price_raw"],
            actions,
            source_version=factor_lineage,
        )
        if not factor_rebound:
            raise RuntimeError(
                "Changed AA action must rebind every factor row to planned actions."
            )
    else:
        factor_lineage = current_lineage
        factors, factor_rebound = _rebind_factors(
            frames["adjustment_factors"],
            frames["daily_price_raw"],
            actions,
            source_version=factor_lineage,
        )
        if factor_rebound:
            raise RuntimeError(
                "AA action is already confirmatory but factor provenance is stale."
            )
    overrides = {
        "corporate_actions": actions,
        "adjustment_factors": factors,
    }
    for dataset, frame in overrides.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=release.completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    validate_repository_snapshot(
        _CandidateRepository(repository, release.dataset_versions, overrides)
    ).raise_for_errors()
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frames=overrides if action_changed else {},
        evidence=evidence,
        summary={
            "status": "validated_offline_plan" if action_changed else "already_repaired",
            "base_release_version": release.version,
            "event_id": AA_EVENT_ID,
            "security_id": PARENT_SECURITY_ID,
            "new_security_id": AA_SECURITY_ID,
            "new_symbol": "AA",
            "effective_date": EFFECTIVE_DATE,
            "ratio": RATIO,
            "old_source_url": evidence.old_url,
            "old_source_hash": evidence.old_hash,
            "confirmatory_source_url": evidence.confirmatory_url,
            "confirmatory_source_hash": evidence.confirmatory_hash,
            "corporate_action_rows_changed": int(action_changed),
            "adjustment_factor_rows": len(factors),
            "adjustment_factor_economic_rows_changed": 0,
            "adjustment_factor_provenance_rows_rebound": (
                len(factors) if action_changed else 0
            ),
            "factor_source_version": factor_lineage,
            "source_daily_price_version": release.dataset_versions[
                "daily_price_raw"
            ],
            "source_corporate_actions_version": (
                planned_versions.get("corporate_actions")
                or release.dataset_versions["corporate_actions"]
            ),
            "planned_versions": dict(planned_versions),
            "write_datasets": list(WRITE_DATASETS),
            "other_dataset_versions_unchanged": True,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved AA provenance recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted AA provenance transaction blocks writes: {journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_inputs_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, release_etag = repository.current_release()
    if (
        release is None
        or release.version != prepared.release.version
        or release_etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after AA provenance planning.")
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or pointer.version != prepared.release.dataset_versions[dataset]
            or etag != prepared.pointer_etags[dataset]
        ):
            raise RuntimeError(f"{dataset} pointer changed after AA planning.")


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
) -> dict[str, Any]:
    current = repository.manifest_for_version(
        dataset, prepared.release.dataset_versions[dataset]
    )
    metadata = dict(current.metadata)
    metadata.update(
        {
            "operation": OPERATION,
            "aa_spinoff_event_id": AA_EVENT_ID,
            "confirmatory_source_url": prepared.evidence.confirmatory_url,
            "confirmatory_source_hash": prepared.evidence.confirmatory_hash,
            "input_release_version": prepared.release.version,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    input_versions = dict(prepared.release.dataset_versions)
    input_versions["corporate_actions"] = prepared.planned_versions[
        "corporate_actions"
    ]
    metadata["input_versions"] = input_versions
    if dataset == "adjustment_factors":
        daily_version = prepared.release.dataset_versions["daily_price_raw"]
        action_version = prepared.planned_versions["corporate_actions"]
        source_version = _adjustment_source_version(daily_version, action_version)
        factors = prepared.frames["adjustment_factors"]
        if (
            set(factors["source_version"].astype(str)) != {source_version}
            or set(factors["source_hash"].astype(str)) != {source_version}
            or set(factors["source"].astype(str)) != {"derived"}
        ):
            raise RuntimeError("Prepared AA factors have stale planned lineage.")
        metadata.update(
            {
                "source_version": source_version,
                "source_daily_price_version": daily_version,
                "source_corporate_actions_version": action_version,
            }
        )
    return metadata


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
    old_versions: Mapping[str, str],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            expected_versions = {**dict(old_versions), **dict(planned_versions)}
            belongs = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or observed.dataset_versions == expected_versions
            if not belongs:
                raise RuntimeError(
                    f"unexpected release during AA rollback: {observed.version}"
                )
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
                        f"unexpected {dataset} pointer during AA rollback: {pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository, release: DataRelease
) -> None:
    current, _ = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes():
        raise RuntimeError("Committed AA provenance release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, _ = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"Applied AA release pointer mismatch: {dataset}.")
    daily_version = release.dataset_versions["daily_price_raw"]
    action_version = release.dataset_versions["corporate_actions"]
    factor_version = release.dataset_versions["adjustment_factors"]
    lineage = _adjustment_source_version(daily_version, action_version)
    manifest = repository.manifest_for_version(
        "adjustment_factors", factor_version
    )
    expected_metadata = {
        "source_version": lineage,
        "source_daily_price_version": daily_version,
        "source_corporate_actions_version": action_version,
    }
    if any(
        _text(manifest.metadata.get(key)) != value
        for key, value in expected_metadata.items()
    ):
        raise RuntimeError("AA factor manifest lineage is not release-exact.")
    factors = repository.read_frame("adjustment_factors", factor_version)
    if (
        set(factors["source_version"].astype(str)) != {lineage}
        or set(factors["source_hash"].astype(str)) != {lineage}
        or set(factors["source"].astype(str)) != {"derived"}
    ):
        raise RuntimeError("AA factor rows are not release-exact.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_repaired":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        current_archive = repository.read_frame(
            "source_archive",
            prepared.release.dataset_versions["source_archive"],
        )
        _verify_confirmatory_evidence(
            repository,
            current_archive,
            completed_session=prepared.release.completed_session,
            evidence=prepared.evidence,
        )
        planned = dict(prepared.planned_versions)
        if set(planned) != set(WRITE_DATASETS) or len(set(planned.values())) != 2:
            raise RuntimeError("Prepared AA repair has invalid planned versions.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before AA apply.")
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_aa_spinoff_provenance_transaction/v1",
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
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(repository, prepared, dataset),
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                if result.manifest.version != planned[dataset]:
                    raise RuntimeError(f"Unexpected {dataset} version was written.")
                versions[dataset] = result.manifest.version
                inject(f"after_write:{dataset}")
            if any(
                versions.get(dataset) != version
                for dataset, version in prepared.release.dataset_versions.items()
                if dataset not in WRITE_DATASETS
            ):
                raise RuntimeError("AA repair changed an out-of-scope dataset version.")
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=prepared.release.quality,
                warnings=prepared.release.warnings,
                expected_etag=prepared.release_etag,
            )
            inject("after_release_commit")
            validate_repository_snapshot(repository).raise_for_errors()
            _assert_applied_release(repository, committed)
            replay = prepare_repair(repository, evidence=prepared.evidence)
            if replay.summary["status"] != "already_repaired":
                raise RuntimeError("AA provenance repair is not idempotent.")
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
                "transaction_id": transaction_id,
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
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
                    "AA provenance rollback was incomplete; recovery marker blocks "
                    f"writes: {recovery}; errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebind the 2016 AA spin-off to confirmatory SEC evidence."
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
