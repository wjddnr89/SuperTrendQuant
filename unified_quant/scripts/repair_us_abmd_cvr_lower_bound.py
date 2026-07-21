#!/usr/bin/env python3
"""Install the reviewed ABMD cash-plus-CVR lower-bound model offline.

ABIOMED holders received $380 cash and one non-tradeable contractual CVR per
share on 2022-12-22.  The CVR can pay up to $35 only if future milestones are
met.  The market-store action schema can execute the guaranteed cash leg but
cannot carry a non-tradeable contingent instrument as a priced position.

This repair therefore records the guaranteed $380 as a ``cash_merger`` and
marks the CVR at zero under an explicit, as-of-2026-07-15 lower-bound policy.
The omitted upside is never silent: the exact policy and official SEC hash are
stored on the action, the lifecycle resolution points to that action, and a
release warning is propagated into every backtest result.

There is deliberately no network, EODHD, or R2 code path.  The default command
is a read-only plan.  ``--apply`` uses a writer lock, release/pointer CAS,
immutable versions, a transaction journal, and rollback on failure.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import html
import json
import math
import re
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from supertrend_quant.market_store.lifecycle import (
    build_lifecycle_candidates,
    canonical_lifecycle_event_id,
)
from supertrend_quant.market_store.lifecycle_coverage import (
    LifecycleCoverageReport,
    validate_lifecycle_coverage,
)
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.official_lifecycle_evidence import (
    OfficialLifecycleExceptionEvidenceSpec,
    include_bound_official_applied_event_candidates,
    load_official_lifecycle_exception_evidence,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_ISSUER_EVIDENCE_DIR = DEFAULT_CACHE_ROOT / "state/issuer_lifecycle"
DEFAULT_HINTS = Path(__file__).resolve().parents[1] / "configs/us_lifecycle_hints.yaml"
POLICY_AS_OF = "2026-07-15"
POLICY_VERSION = "abmd_nontradeable_cvr_lower_bound/v1"
REVIEWED_BY = "abmd_cvr_lower_bound_policy_v1"
REVIEWED_AT = "2026-07-18T00:00:00Z"

ABMD_SECURITY_ID = "US:EODHD:faece1b7-4b1a-5c1f-951b-b1178ed57161"
ABMD_SYMBOL = "ABMD"
LAST_REAL_SESSION = "2022-12-21"
LAST_REAL_CLOSE = 381.02
EFFECTIVE_DATE = "2022-12-22"
ANNOUNCEMENT_DATE = "2022-11-01"
GUARANTEED_CASH = 380.0
CVR_QUANTITY = 1.0
CVR_MAXIMUM_CASH = 35.0
CVR_MARK = 0.0
EVENT_ID = canonical_lifecycle_event_id(
    ABMD_SECURITY_ID, "cash_merger", EFFECTIVE_DATE
)


@dataclass(frozen=True)
class EvidenceSpec:
    source_url: str
    source_hash: str
    uncompressed_size: int
    archive_object_path: str
    retrieved_at: str
    required_text_groups: tuple[tuple[str, ...], ...]


EVIDENCE = EvidenceSpec(
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/815094/"
        "000119312522311074/0001193125-22-311074.txt"
    ),
    source_hash=(
        "f98bc807432739e4f2447ffbc6a70f7651bd8982b901989fc31dcffaa56ec593"
    ),
    uncompressed_size=331_457,
    archive_object_path=(
        "archives/2026-07-15/"
        "f98bc807432739e4f2447ffbc6a70f7651bd8982b901989fc31dcffaa56ec593"
        ".txt.gz"
    ),
    retrieved_at="2026-07-18T07:26:22.110643Z",
    required_text_groups=(
        ("December 22, 2022",),
        ("$380.00 per Company Share",),
        ("one non-tradeable contractual contingent value right per Company Share",),
        ("up to $35.00 per Company Share",),
    ),
)


@dataclass(frozen=True)
class ValuationEvidenceSpec:
    source_url: str
    source_hash: str
    size: int
    retrieved_at: str
    report_as_of: str
    remaining_cvr_liability_usd_billions: float

    @property
    def filename(self) -> str:
        return f"{self.source_hash}.pdf"

    @property
    def archive_object_path(self) -> str:
        return f"archives/{POLICY_AS_OF}/{self.source_hash}.pdf.gz"


VALUATION_EVIDENCE = ValuationEvidenceSpec(
    source_url=(
        "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000200406/"
        "e09b8882-48b1-4fea-a818-66acddf84c4f.pdf"
    ),
    source_hash=(
        "65710a85a1f27aa581c1cddce22cab62bec0a3b5848283e163bbdcc1aa67b5e8"
    ),
    size=2_184_778,
    retrieved_at="2026-07-18T08:15:00Z",
    report_as_of="2025-12-28",
    remaining_cvr_liability_usd_billions=0.4,
)

LOWER_BOUND_WARNING = (
    "ABMD 2022-12-22 merger models only the guaranteed $380/share cash leg; "
    "one non-tradeable CVR per share (up to $35 contingent cash) is marked at "
    "$0 under the as-of-2026-07-15 lower-bound policy, so returns may be "
    "understated; J&J still reported about $0.4bn of aggregate ABMD CVR "
    "liability at 2025 year-end, so $0 is not a fair-value estimate."
)

REQUIRED_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "index_constituent_anchors",
    "index_membership_events",
    "lifecycle_resolutions",
    "source_archive",
)
WRITE_DATASETS = (
    "corporate_actions",
    "lifecycle_resolutions",
    "source_archive",
)


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    frames: Mapping[str, pd.DataFrame]
    coverage: LifecycleCoverageReport
    official_evidence_specs: Mapping[
        str, OfficialLifecycleExceptionEvidenceSpec
    ]
    source_candidate_drift: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: Mapping[str, Any]


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _metadata_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return _canonical_json(value)
    raw = _text(value)
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("ABMD action metadata is not valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("ABMD action metadata must be a JSON object.")
    return _canonical_json(parsed)


def _policy_metadata() -> dict[str, Any]:
    return {
        "consideration_model": POLICY_VERSION,
        "guaranteed_cash_per_share": GUARANTEED_CASH,
        "last_real_session": LAST_REAL_SESSION,
        "last_real_close": LAST_REAL_CLOSE,
        "cvr": {
            "quantity_per_share": CVR_QUANTITY,
            "tradeable": False,
            "maximum_contingent_cash_per_right": CVR_MAXIMUM_CASH,
            "mark_per_right": CVR_MARK,
            "mark_as_of": POLICY_AS_OF,
            "valuation_policy": "zero_mark_lower_bound",
            "future_contingent_upside_included": False,
        },
        "audit_warning": LOWER_BOUND_WARNING,
        "official_evidence": {
            "source_url": EVIDENCE.source_url,
            "source_hash": EVIDENCE.source_hash,
        },
        "lower_bound_rationale": {
            "zero_is_fair_value_estimate": False,
            "report_as_of": VALUATION_EVIDENCE.report_as_of,
            "remaining_aggregate_cvr_liability_usd_billions": (
                VALUATION_EVIDENCE.remaining_cvr_liability_usd_billions
            ),
            "source_url": VALUATION_EVIDENCE.source_url,
            "source_hash": VALUATION_EVIDENCE.source_hash,
        },
    }


def _normalized_document_text(content: bytes) -> str:
    raw = content.decode("utf-8", errors="replace")
    without_tags = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(html.unescape(without_tags).replace("\xa0", " ").split())


def _verified_evidence(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
) -> tuple[bytes, str]:
    matches = source_archive.loc[
        source_archive["archive_id"].astype(str).eq(EVIDENCE.source_hash)
    ]
    if len(matches) != 1:
        raise ValueError("Exact ABMD SEC source_archive row is missing or duplicated.")
    row = matches.iloc[0]
    expected = {
        "dataset": "sec_edgar_filing",
        "object_path": EVIDENCE.archive_object_path,
        "content_type": "text/plain",
        "source": "sec_edgar_filing",
        "source_hash": EVIDENCE.source_hash,
        "source_url": EVIDENCE.source_url,
    }
    changed = [key for key, value in expected.items() if _text(row.get(key)) != value]
    if changed:
        raise ValueError(
            "ABMD SEC archive provenance changed: " + ", ".join(sorted(changed))
        )
    retrieved_at = _text(row.get("retrieved_at"))
    if not retrieved_at:
        raise ValueError("ABMD SEC archive row lacks retrieved_at provenance.")
    path = (repository.root / EVIDENCE.archive_object_path).resolve()
    root = repository.root.resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError("Pinned ABMD SEC archive payload is missing.")
    try:
        content = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError("Pinned ABMD SEC archive payload is not valid gzip.") from exc
    digest = hashlib.sha256(content).hexdigest()
    if digest != EVIDENCE.source_hash or len(content) != EVIDENCE.uncompressed_size:
        raise ValueError(
            "Pinned ABMD SEC payload hash/size mismatch: "
            f"sha256={digest}; size={len(content)}."
        )
    text = _normalized_document_text(content)
    for alternatives in EVIDENCE.required_text_groups:
        if not any(value in text for value in alternatives):
            raise ValueError(
                "Pinned ABMD SEC payload lacks reviewed term: "
                + " | ".join(alternatives)
            )
    return content, retrieved_at


def _verified_valuation_evidence(
    evidence_dir: Path = DEFAULT_ISSUER_EVIDENCE_DIR,
) -> bytes:
    path = evidence_dir / VALUATION_EVIDENCE.filename
    if not path.is_file():
        raise FileNotFoundError(
            "Pinned J&J 2025 annual-report evidence is missing: " + str(path)
        )
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != VALUATION_EVIDENCE.source_hash or len(content) != VALUATION_EVIDENCE.size:
        raise ValueError(
            "Pinned J&J 2025 annual report hash/size mismatch: "
            f"sha256={digest}; size={len(content)}."
        )
    if not content.startswith(b"%PDF-"):
        raise ValueError("Pinned J&J valuation evidence is not a PDF.")
    return content


def _rewrite_source_archive(
    source_archive: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    output = source_archive.copy()
    matches = output["archive_id"].astype(str).eq(VALUATION_EVIDENCE.source_hash)
    expected = {
        "archive_id": VALUATION_EVIDENCE.source_hash,
        "dataset": "official_abmd_cvr_valuation_evidence",
        "object_path": VALUATION_EVIDENCE.archive_object_path,
        "content_type": "application/pdf",
        "effective_date": POLICY_AS_OF,
        "source": "jnj_2025_annual_report",
        "retrieved_at": VALUATION_EVIDENCE.retrieved_at,
        "source_hash": VALUATION_EVIDENCE.source_hash,
        "source_url": VALUATION_EVIDENCE.source_url,
    }
    if matches.any():
        rows = output.loc[matches]
        if len(rows) != 1 or any(
            _text(rows.iloc[0].get(key)) != value for key, value in expected.items()
        ):
            raise ValueError("J&J valuation source_archive row conflicts with policy.")
        return output.reset_index(drop=True), False
    output = pd.concat(
        [output, pd.DataFrame([expected])], ignore_index=True, sort=False
    )
    return output.reset_index(drop=True), True


def _verify_persisted_valuation_archive(
    repository: LocalDatasetRepository,
    expected_content: bytes,
) -> None:
    path = repository.root / VALUATION_EVIDENCE.archive_object_path
    if not path.is_file():
        raise FileNotFoundError("Referenced J&J valuation archive payload is missing.")
    try:
        content = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError("Referenced J&J valuation archive is invalid gzip.") from exc
    if content != expected_content:
        raise ValueError("Referenced J&J valuation archive differs from pinned PDF.")


def _expected_action(*, retrieved_at: str | None = None) -> dict[str, Any]:
    return {
        "event_id": EVENT_ID,
        "security_id": ABMD_SECURITY_ID,
        "action_type": "cash_merger",
        "effective_date": EFFECTIVE_DATE,
        "ex_date": EFFECTIVE_DATE,
        "announcement_date": ANNOUNCEMENT_DATE,
        "record_date": "",
        "payment_date": EFFECTIVE_DATE,
        "cash_amount": GUARANTEED_CASH,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": True,
        "source_url": EVIDENCE.source_url,
        "source_kind": "official_lower_bound_policy",
        "source": "sec_edgar+audited_lower_bound_policy",
        "retrieved_at": retrieved_at or EVIDENCE.retrieved_at,
        "source_hash": EVIDENCE.source_hash,
        "metadata": _canonical_json(_policy_metadata()),
    }


def _verify_terminal_identity(
    master: pd.DataFrame,
    history: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    master_rows = master.loc[
        master["security_id"].astype(str).eq(ABMD_SECURITY_ID)
    ]
    history_rows = history.loc[
        history["security_id"].astype(str).eq(ABMD_SECURITY_ID)
        & history["symbol"].astype(str).str.upper().eq(ABMD_SYMBOL)
    ]
    if len(master_rows) != 1 or len(history_rows) != 1:
        raise ValueError("ABMD terminal identity rows are missing or duplicated.")
    if (
        _text(master_rows.iloc[0].get("primary_symbol")).upper() != ABMD_SYMBOL
        or _text(master_rows.iloc[0].get("active_to")) != LAST_REAL_SESSION
        or _text(history_rows.iloc[0].get("effective_to")) != LAST_REAL_SESSION
    ):
        raise ValueError("ABMD terminal identity boundary is not the reviewed date.")
    target = prices.loc[prices["security_id"].astype(str).eq(ABMD_SECURITY_ID)].copy()
    if target.empty:
        raise ValueError("ABMD has no terminal price history.")
    target["_session"] = pd.to_datetime(target["session"], errors="coerce")
    if target["_session"].isna().any():
        raise ValueError("ABMD terminal price history contains invalid sessions.")
    last = target.sort_values("_session").iloc[-1]
    close = pd.to_numeric(pd.Series([last.get("close")]), errors="coerce").iloc[0]
    if (
        pd.Timestamp(last["_session"]).date().isoformat() != LAST_REAL_SESSION
        or pd.isna(close)
        or not math.isclose(float(close), LAST_REAL_CLOSE, rel_tol=0, abs_tol=1e-8)
    ):
        raise ValueError("ABMD last real session/close changed from 2022-12-21/381.02.")


def _action_is_exact(row: Mapping[str, Any], *, retrieved_at: str) -> bool:
    expected = _expected_action(retrieved_at=retrieved_at)
    text_columns = (
        "event_id",
        "security_id",
        "action_type",
        "effective_date",
        "ex_date",
        "announcement_date",
        "record_date",
        "payment_date",
        "currency",
        "new_security_id",
        "new_symbol",
        "source_url",
        "source_kind",
        "source",
        "retrieved_at",
        "source_hash",
    )
    if any(_text(row.get(key)) != _text(expected[key]) for key in text_columns):
        return False
    cash = pd.to_numeric(pd.Series([row.get("cash_amount")]), errors="coerce").iloc[0]
    ratio = pd.to_numeric(pd.Series([row.get("ratio")]), errors="coerce").iloc[0]
    return bool(
        _text(row.get("official")).lower() == "true"
        and not pd.isna(cash)
        and math.isclose(float(cash), GUARANTEED_CASH, rel_tol=0, abs_tol=1e-12)
        and pd.isna(ratio)
        and _metadata_text(row.get("metadata")) == expected["metadata"]
    )


def _rewrite_actions(
    actions: pd.DataFrame,
    *,
    retrieved_at: str,
) -> tuple[pd.DataFrame, bool]:
    output = actions.copy()
    if "metadata" not in output.columns:
        output["metadata"] = ""
    same_event = output["event_id"].astype(str).eq(EVENT_ID)
    same_terminal = (
        output["security_id"].astype(str).eq(ABMD_SECURITY_ID)
        & output["action_type"].astype(str).eq("cash_merger")
        & output["effective_date"].astype(str).eq(EFFECTIVE_DATE)
    )
    matches = output.loc[same_event | same_terminal]
    if not matches.empty:
        if len(matches) != 1 or not _action_is_exact(
            matches.iloc[0].to_dict(), retrieved_at=retrieved_at
        ):
            raise ValueError("Conflicting or partially modeled ABMD merger action exists.")
        return output.reset_index(drop=True), False
    output = pd.concat(
        [output, pd.DataFrame([_expected_action(retrieved_at=retrieved_at)])],
        ignore_index=True,
        sort=False,
    )
    output = output.sort_values(
        ["effective_date", "security_id", "action_type", "event_id"]
    ).reset_index(drop=True)
    return output, True


def _exception_is_exact(row: Mapping[str, Any]) -> bool:
    return bool(
        _text(row.get("security_id")) == ABMD_SECURITY_ID
        and _text(row.get("symbol")).upper() == ABMD_SYMBOL
        and _text(row.get("last_price_date")) == LAST_REAL_SESSION
        and _text(row.get("resolution")) == "exception"
        and _text(row.get("event_id")) == ""
        and _text(row.get("exception_code")) == "unsupported_consideration"
        and _text(row.get("source_url")) == EVIDENCE.source_url
        and _text(row.get("source_hash")) == EVIDENCE.source_hash
    )


def _applied_resolution_is_exact(row: Mapping[str, Any]) -> bool:
    return bool(
        _text(row.get("security_id")) == ABMD_SECURITY_ID
        and _text(row.get("symbol")).upper() == ABMD_SYMBOL
        and _text(row.get("last_price_date")) == LAST_REAL_SESSION
        and _text(row.get("resolution")) == "applied"
        and _text(row.get("event_id")) == EVENT_ID
        and all(
            _text(row.get(column)) == ""
            for column in ("exception_code", "exception_reason", "recheck_after")
        )
        and _text(row.get("successor_security_id")) == ""
        and _text(row.get("successor_symbol")) == ""
        and _text(row.get("source_url")) == EVIDENCE.source_url
        and _text(row.get("source_hash")) == EVIDENCE.source_hash
    )


def _rewrite_resolutions(
    resolutions: pd.DataFrame,
    *,
    action_present: bool,
    retrieved_at: str,
) -> tuple[pd.DataFrame, bool]:
    output = resolutions.copy()
    matches = output["security_id"].astype(str).eq(ABMD_SECURITY_ID)
    if int(matches.sum()) != 1:
        raise ValueError("ABMD lifecycle resolution is missing or duplicated.")
    index = output.index[matches][0]
    row = output.loc[index].to_dict()
    if _applied_resolution_is_exact(row):
        if not action_present:
            raise ValueError("ABMD applied resolution exists without its exact action.")
        return output.reset_index(drop=True), False
    if not _exception_is_exact(row):
        raise ValueError("ABMD lifecycle resolution is not the reviewed exception.")
    values = {
        "resolution": "applied",
        "event_id": EVENT_ID,
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": REVIEWED_BY,
        "reviewed_at": REVIEWED_AT,
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        "source_url": EVIDENCE.source_url,
        "source": "abmd_cvr_lower_bound_repair",
        "retrieved_at": retrieved_at,
        "source_hash": EVIDENCE.source_hash,
    }
    for column, value in values.items():
        output.at[index, column] = value
    return output.reset_index(drop=True), True


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
        return self.base.read_frame(dataset, self.versions[dataset])


def _candidate_frame(
    repository: LocalDatasetRepository,
    release: DataRelease,
    official_evidence_specs: Mapping[
        str, OfficialLifecycleExceptionEvidenceSpec
    ],
) -> pd.DataFrame:
    values = include_bound_official_applied_event_candidates(
        build_lifecycle_candidates(repository, release=release),
        repository,
        release,
        official_evidence_specs,
    )
    rows = [asdict(item) for item in values]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=("security_id", "last_price_date")
    )


def _resolution_candidate_frame(resolutions: pd.DataFrame) -> pd.DataFrame:
    """Recover the finalizer-bound candidate set without changing its identity.

    A different identity repair may advance security/index datasets before the
    lifecycle finalizer is replayed.  Planning this ABMD-only repair remains
    useful in that transient state, but apply must stay blocked.  The stored
    resolution set plus its manifest hash is the immutable prior candidate
    inventory; the freshly rebuilt set is compared separately below.
    """

    return resolutions[["candidate_id", "security_id", "last_price_date"]].copy()


def _candidate_drift(
    stored: pd.DataFrame,
    rebuilt: pd.DataFrame,
) -> tuple[str, ...]:
    def keys(frame: pd.DataFrame) -> set[str]:
        return {
            f"{_text(row.security_id)}|{_text(row.last_price_date)}"
            for row in frame.itertuples(index=False)
        }

    return tuple(sorted(keys(stored) ^ keys(rebuilt)))


def _validate_coverage(
    candidates: pd.DataFrame,
    resolutions: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    completed_session: str,
) -> LifecycleCoverageReport:
    report = validate_lifecycle_coverage(
        candidates,
        resolutions,
        actions,
        completed_session=completed_session,
    )
    report.raise_for_errors()
    if not report.valid or report.open_count:
        raise ValueError("ABMD repair did not close lifecycle coverage.")
    return report


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


def _dedupe_warnings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*values, LOWER_BOUND_WARNING)))


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    evidence_dir: Path = DEFAULT_ISSUER_EVIDENCE_DIR,
    hints_path: Path = DEFAULT_HINTS,
    official_evidence_specs: Mapping[
        str, OfficialLifecycleExceptionEvidenceSpec
    ] | None = None,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    if release.completed_session != POLICY_AS_OF:
        raise RuntimeError(
            "ABMD zero-mark policy is frozen to completed_session "
            f"{POLICY_AS_OF}; found {release.completed_session}."
        )
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))

    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    _content, evidence_retrieved_at = _verified_evidence(
        repository, frames["source_archive"]
    )
    valuation_content = _verified_valuation_evidence(evidence_dir)
    _verify_terminal_identity(
        frames["security_master"],
        frames["symbol_history"],
        frames["daily_price_raw"],
    )
    resolved_official_evidence_specs = (
        load_official_lifecycle_exception_evidence(hints_path)
        if official_evidence_specs is None
        else dict(official_evidence_specs)
    )
    candidates = _resolution_candidate_frame(frames["lifecycle_resolutions"])
    rebuilt_candidates = _candidate_frame(
        repository,
        release,
        resolved_official_evidence_specs,
    )
    source_candidate_drift = _candidate_drift(candidates, rebuilt_candidates)
    current_coverage = _validate_coverage(
        candidates,
        frames["lifecycle_resolutions"],
        frames["corporate_actions"],
        completed_session=release.completed_session,
    )
    manifest = repository.manifest_for_version(
        "lifecycle_resolutions", release.dataset_versions["lifecycle_resolutions"]
    )
    if manifest.metadata.get("candidate_set_sha256") != current_coverage.candidate_set_sha256:
        raise ValueError("Current lifecycle candidate manifest hash is stale.")

    actions, action_added = _rewrite_actions(
        frames["corporate_actions"], retrieved_at=evidence_retrieved_at
    )
    action_present = bool(
        actions["event_id"].astype(str).eq(EVENT_ID).sum() == 1
    )
    resolutions, resolution_changed = _rewrite_resolutions(
        frames["lifecycle_resolutions"],
        action_present=action_present,
        retrieved_at=evidence_retrieved_at,
    )
    source_archive, archive_added = _rewrite_source_archive(frames["source_archive"])
    if not archive_added:
        _verify_persisted_valuation_archive(repository, valuation_content)
    # cash_merger is intentionally absent from both RATIO_ACTIONS and
    # CASH_DISTRIBUTION_ACTIONS in adjustments.py.  Preserve the exact factor
    # version and values; rewriting 2.1M neutral rows would add cost without
    # changing a single adjusted price.
    if frames["adjustment_factors"].empty:
        raise ValueError("Current adjustment-factor dataset is unexpectedly empty.")
    overrides = {
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "source_archive": source_archive,
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
    coverage = _validate_coverage(
        candidates,
        resolutions,
        actions,
        completed_session=release.completed_session,
    )
    warnings = _dedupe_warnings(release.warnings)
    data_changed = action_added or resolution_changed or archive_added
    warning_changed = warnings != release.warnings
    status = (
        "validated_offline_plan"
        if data_changed
        else "warning_only"
        if warning_changed
        else "already_applied"
    )
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=(
            _capture_pointer_etags(repository, release) if data_changed else {}
        ),
        frames=overrides if data_changed else {},
        coverage=coverage,
        official_evidence_specs=resolved_official_evidence_specs,
        source_candidate_drift=source_candidate_drift,
        warnings=warnings,
        summary={
            "status": status,
            "base_release_version": release.version,
            "event_id": EVENT_ID,
            "guaranteed_cash_per_share": GUARANTEED_CASH,
            "last_real_close": LAST_REAL_CLOSE,
            "cvr_quantity_per_share": CVR_QUANTITY,
            "cvr_maximum_contingent_cash": CVR_MAXIMUM_CASH,
            "cvr_mark_per_right": CVR_MARK,
            "cvr_policy": POLICY_VERSION,
            "policy_as_of": POLICY_AS_OF,
            "action_added": action_added,
            "resolution_changed_to_applied": resolution_changed,
            "valuation_archive_added": archive_added,
            "adjustment_factor_values_changed": False,
            "adjustment_factor_rows_rebound": 0,
            "adjustment_factor_version_retained": release.dataset_versions[
                "adjustment_factors"
            ],
            "lifecycle_applied_count": coverage.applied_count,
            "lifecycle_exception_count": coverage.exception_count,
            "lifecycle_open_count": coverage.open_count,
            "source_candidate_drift_count": len(source_candidate_drift),
            "source_candidate_drift": list(source_candidate_drift),
            "apply_blocked_until_lifecycle_refresh": bool(source_candidate_drift),
            "official_source_hash": EVIDENCE.source_hash,
            "valuation_source_hash": VALUATION_EVIDENCE.source_hash,
            "remaining_cvr_liability_usd_billions_at_2025_year_end": (
                VALUATION_EVIDENCE.remaining_cvr_liability_usd_billions
            ),
            "release_warning_added": warning_changed,
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
        recovery = repository.root / "recovery/us-abmd-cvr-lower-bound"
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved ABMD CVR recovery marker blocks writes.")
        transactions = repository.root / "transactions/us-abmd-cvr-lower-bound"
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = str(json.loads(journal.read_bytes()).get("status", ""))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted ABMD CVR transaction blocks writes: {journal}."
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
        raise RuntimeError("Current release changed during ABMD CVR validation.")


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


def _release_quality_with_warning(release: DataRelease) -> str:
    return (
        str(DataQuality.BLOCKED)
        if str(release.quality) == str(DataQuality.BLOCKED)
        else str(DataQuality.DEGRADED)
    )


def _persist_valuation_evidence(
    repository: LocalDatasetRepository,
    *,
    evidence_dir: Path,
) -> None:
    content = _verified_valuation_evidence(evidence_dir)
    path = repository.root / VALUATION_EVIDENCE.archive_object_path
    if path.is_file():
        _verify_persisted_valuation_archive(repository, content)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, gzip.compress(content, mtime=0))
    if gzip.decompress(path.read_bytes()) != content:
        raise RuntimeError("J&J valuation archive verification failed after write.")


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
    planned: Mapping[str, str],
) -> dict[str, Any]:
    current = repository.manifest_for_version(
        dataset, prepared.release.dataset_versions[dataset]
    )
    metadata = dict(current.metadata)
    metadata.update(
        {
            "operation": "repair_us_abmd_cvr_lower_bound",
            "policy": POLICY_VERSION,
            "policy_as_of": POLICY_AS_OF,
            "official_source_hash": EVIDENCE.source_hash,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
            "adjustment_factor_delta_policy": (
                "cash_merger_is_adjustment_neutral; exact factor version retained"
            ),
            "retained_adjustment_factors_version": (
                prepared.release.dataset_versions["adjustment_factors"]
            ),
        }
    )
    if dataset == "corporate_actions":
        metadata.update(
            {
                "abmd_event_id": EVENT_ID,
                "cvr_mark_per_right": CVR_MARK,
                "_logical_quality": str(DataQuality.DEGRADED),
                "_logical_warnings": (LOWER_BOUND_WARNING,),
            }
        )
    elif dataset == "lifecycle_resolutions":
        metadata.update(prepared.coverage.manifest_metadata())
        metadata.update(
            {
                "output_versions": {
                    **dict(metadata.get("output_versions") or {}),
                    **dict(planned),
                },
            }
        )
    elif dataset == "source_archive":
        metadata.update(
            {
                "abmd_valuation_evidence_archive_id": (
                    VALUATION_EVIDENCE.source_hash
                ),
            }
        )
    return metadata


def _apply_metadata_only(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> dict[str, Any]:
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(repository, prepared)
        committed = repository.commit_release(
            prepared.release.completed_session,
            prepared.release.dataset_versions,
            quality=_release_quality_with_warning(prepared.release),
            warnings=prepared.warnings,
            expected_etag=prepared.release_etag,
        )
        validate_repository_snapshot(repository).raise_for_errors()
        return {
            **prepared.summary,
            "status": "applied_metadata_only",
            "new_release_version": committed.version,
            "quality": str(committed.quality),
            "warnings": list(committed.warnings),
            "dataset_writes_performed": False,
            "writes_performed": True,
        }


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    evidence_dir: Path = DEFAULT_ISSUER_EVIDENCE_DIR,
    inject_failure: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if prepared.source_candidate_drift:
        raise RuntimeError(
            "Lifecycle candidate sources changed after the stored finalizer run; "
            "rerun the lifecycle finalizer before ABMD CVR apply: "
            + ", ".join(prepared.source_candidate_drift)
        )
    if prepared.summary["status"] == "already_applied":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    if prepared.summary["status"] == "warning_only":
        return _apply_metadata_only(repository, prepared)
    inject_failure = inject_failure or (lambda _stage: None)
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
                raise RuntimeError(f"{dataset} pointer changed before ABMD CVR apply.")
            old_pointers[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        planned = {
            dataset: (
                "abmd-cvr-lower-bound-"
                f"{prepared.release.completed_session.replace('-', '')}-"
                f"{transaction_id}-{dataset}"
            )
            for dataset in WRITE_DATASETS
        }
        frames = dict(prepared.frames)
        for dataset, frame in frames.items():
            validate_dataset(
                dataset,
                frame,
                completed_session=prepared.release.completed_session,
                incomplete_action_policy="block",
            ).raise_for_errors()
        validate_repository_snapshot(
            _CandidateRepository(repository, prepared.release.dataset_versions, frames)
        ).raise_for_errors()

        journal_path = (
            repository.root
            / "transactions/us-abmd-cvr-lower-bound"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "us_abmd_cvr_lower_bound_transaction/v1",
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
            _persist_valuation_evidence(repository, evidence_dir=evidence_dir)
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(
                        repository,
                        prepared,
                        dataset,
                        planned,
                    ),
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                versions[dataset] = result.manifest.version
                inject_failure(f"after_write:{dataset}")
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=_release_quality_with_warning(prepared.release),
                warnings=prepared.warnings,
                expected_etag=prepared.release_etag,
            )
            inject_failure("after_release_commit")
            validate_repository_snapshot(repository).raise_for_errors()
            candidates = _candidate_frame(
                repository,
                committed,
                prepared.official_evidence_specs,
            )
            coverage = _validate_coverage(
                candidates,
                repository.read_frame(
                    "lifecycle_resolutions",
                    committed.dataset_versions["lifecycle_resolutions"],
                ),
                repository.read_frame(
                    "corporate_actions",
                    committed.dataset_versions["corporate_actions"],
                ),
                completed_session=committed.completed_session,
            )
            if coverage.manifest_metadata() != prepared.coverage.manifest_metadata():
                raise RuntimeError("Committed ABMD lifecycle coverage changed after apply.")
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
                    / "recovery/us-abmd-cvr-lower-bound"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "ABMD CVR rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the reviewed ABMD cash-plus-CVR lower-bound model offline."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--evidence-dir", type=Path, default=DEFAULT_ISSUER_EVIDENCE_DIR
    )
    parser.add_argument("--hints", type=Path, default=DEFAULT_HINTS)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(
        repository,
        evidence_dir=args.evidence_dir,
        hints_path=args.hints,
    )
    result = (
        apply_repair(repository, prepared, evidence_dir=args.evidence_dir)
        if args.apply
        else {**prepared.summary, "mode": "plan", "writes_performed": False}
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
