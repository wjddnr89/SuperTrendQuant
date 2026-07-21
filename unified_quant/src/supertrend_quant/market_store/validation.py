from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .manifest import DatasetManifest, sha256_file
from .models import CorporateActionType, DataQuality
from .schemas import DatasetSpec, dataset_spec


INDEX_MEMBER_PRICE_EDGE_GRACE_DAYS = 14


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"
    row_count: int = 0
    fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def quality(self) -> DataQuality:
        if not self.valid:
            return DataQuality.BLOCKED
        if self.issues:
            return DataQuality.DEGRADED
        return DataQuality.VALID

    def raise_for_errors(self) -> None:
        errors = [issue.message for issue in self.issues if issue.severity == "error"]
        if errors:
            raise ValueError("; ".join(errors))


def index_member_identity_gap_fingerprint(
    *,
    index_id: str,
    replay_date: str,
    security_id: str,
    next_remove_event_id: str,
    next_remove_effective_date: str,
    next_remove_source: str,
    next_remove_source_hash: str,
) -> str:
    """Fingerprint one exact replay gap including its pending removal lineage."""

    payload = {
        "code": "index_member_missing_active_symbol",
        "index_id": str(index_id).strip(),
        "replay_date": str(replay_date).strip(),
        "security_id": str(security_id).strip(),
        "next_remove_event_id": str(next_remove_event_id).strip(),
        "next_remove_effective_date": str(next_remove_effective_date).strip(),
        "next_remove_source": str(next_remove_source).strip(),
        "next_remove_source_hash": str(next_remove_source_hash).strip().lower(),
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_dataset(
    name: str,
    frame: pd.DataFrame,
    *,
    incomplete_action_policy: str = "warn",
    completed_session: str | None = None,
) -> ValidationReport:
    spec = dataset_spec(name)
    issues: list[ValidationIssue] = []
    missing = tuple(column for column in spec.required_columns if column not in frame.columns)
    if missing:
        issues.append(
            ValidationIssue("missing_columns", f"{name} is missing columns: {', '.join(missing)}")
        )
        return ValidationReport(name, tuple(issues))

    _validate_primary_key(frame, spec, issues)
    _validate_dates(frame, spec, issues)
    _validate_source_metadata(frame, issues)
    if name == "daily_price_raw":
        _validate_prices(frame, issues)
        _validate_price_sessions(frame, issues, completed_session)
    elif name == "symbol_history":
        _validate_symbol_history(frame, issues)
    elif name == "corporate_actions":
        _validate_actions(frame, issues, incomplete_action_policy)
    elif name == "lifecycle_resolutions":
        _validate_lifecycle_resolutions(frame, issues, completed_session)
    elif name == "adjustment_factors":
        _validate_adjustment_factors(frame, issues)
    elif name in {"index_membership_events", "custom_universe_overlays"}:
        _validate_operations(frame, issues)
    return ValidationReport(name, tuple(issues))


def validate_manifest_files(root: str | Path, manifest: DatasetManifest) -> ValidationReport:
    base = Path(root)
    issues: list[ValidationIssue] = []
    for item in manifest.files:
        path = base / item.path
        if not path.is_file():
            issues.append(ValidationIssue("missing_file", f"Missing manifest file: {item.path}"))
            continue
        if path.stat().st_size != item.size_bytes:
            issues.append(ValidationIssue("size_mismatch", f"Size mismatch: {item.path}"))
        if sha256_file(path) != item.sha256:
            issues.append(ValidationIssue("hash_mismatch", f"SHA-256 mismatch: {item.path}"))
    return ValidationReport(manifest.dataset, tuple(issues))


def validate_repository_snapshot(
    repository,
    *,
    allowed_index_price_gap_ids: Iterable[str] = (),
    allowed_index_identity_gap_fingerprints: Iterable[str] = (),
) -> ValidationReport:
    """Cross-dataset checks run before publication and during quant-data validate."""
    issues: list[ValidationIssue] = []
    master = _optional_repository_frame(repository, "security_master")
    prices = _optional_repository_frame(repository, "daily_price_raw")
    factors = _optional_repository_frame(repository, "adjustment_factors")
    actions = _optional_repository_frame(repository, "corporate_actions")
    history = _optional_repository_frame(repository, "symbol_history")
    anchors = _optional_repository_frame(repository, "index_constituent_anchors")
    events = _optional_repository_frame(repository, "index_membership_events")

    if not prices.empty and not factors.empty:
        price_keys = set(zip(prices["security_id"].astype(str), pd.to_datetime(prices["session"]).dt.date))
        factor_keys = set(zip(factors["security_id"].astype(str), pd.to_datetime(factors["session"]).dt.date))
        missing = price_keys - factor_keys
        if missing:
            issues.append(
                ValidationIssue(
                    "missing_adjustment_factors",
                    "Raw price rows are missing adjustment factors.",
                    row_count=len(missing),
                )
            )
        issues.extend(_factor_action_issues(prices, actions, factors))
    if not actions.empty and not history.empty:
        unknown = set(actions["security_id"].astype(str)) - set(history["security_id"].astype(str))
        if unknown:
            issues.append(
                ValidationIssue(
                    "unknown_action_security",
                    "Corporate actions reference unknown security_ids.",
                    row_count=len(unknown),
                )
            )
    if not actions.empty:
        issues.extend(_lifecycle_successor_issues(actions, master, history))
    if not events.empty:
        issues.extend(_index_transition_issues(anchors, events))
        if not history.empty:
            unknown = set(events["security_id"].astype(str)) - set(history["security_id"].astype(str))
            if unknown:
                issues.append(
                    ValidationIssue(
                        "unknown_index_security",
                        "Index events reference unknown security_ids.",
                        row_count=len(unknown),
                    )
                )
    if not anchors.empty:
        issues.extend(
            _index_member_coverage_issues(
                anchors,
                events,
                history,
                prices,
                allowed_price_gap_ids=allowed_index_price_gap_ids,
                allowed_identity_gap_fingerprints=(
                    allowed_index_identity_gap_fingerprints
                ),
            )
        )
    return ValidationReport("repository_snapshot", tuple(issues))


def validate_revisions(
    previous: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    primary_key: tuple[str, ...],
    value_columns: tuple[str, ...],
    allow_revisions: bool = False,
) -> ValidationReport:
    joined = previous.merge(candidate, on=list(primary_key), suffixes=("_old", "_new"))
    changed = pd.Series(False, index=joined.index)
    for column in value_columns:
        old = joined[f"{column}_old"]
        new = joined[f"{column}_new"]
        changed |= ~(old.eq(new) | (old.isna() & new.isna()))
    if not changed.any():
        return ValidationReport("revisions")
    return ValidationReport(
        "revisions",
        (
            ValidationIssue(
                "source_revision",
                "Previously stored source values changed.",
                severity="warning" if allow_revisions else "error",
                row_count=int(changed.sum()),
            ),
        ),
    )


def _validate_primary_key(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    issues: list[ValidationIssue],
) -> None:
    nulls = frame.loc[:, list(spec.primary_key)].isna().any(axis=1)
    if nulls.any():
        issues.append(
            ValidationIssue(
                "null_primary_key",
                f"{spec.name} has null primary-key values.",
                row_count=int(nulls.sum()),
            )
        )
    duplicates = frame.duplicated(list(spec.primary_key), keep=False)
    if duplicates.any():
        issues.append(
            ValidationIssue(
                "duplicate_primary_key",
                f"{spec.name} has duplicate primary keys.",
                row_count=int(duplicates.sum()),
            )
        )


def _validate_dates(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    issues: list[ValidationIssue],
) -> None:
    for column in spec.date_columns:
        values = frame[column]
        nonempty = values.notna() & values.astype(str).ne("")
        invalid = nonempty & pd.to_datetime(values, errors="coerce").isna()
        if invalid.any():
            issues.append(
                ValidationIssue(
                    "invalid_date",
                    f"{spec.name}.{column} has invalid dates.",
                    row_count=int(invalid.sum()),
                )
            )


def _validate_source_metadata(frame: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    columns = ["source", "retrieved_at", "source_hash"]
    if "source_kind" in frame:
        columns.append("source_kind")
    for column in columns:
        missing = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        if missing.any():
            issues.append(
                ValidationIssue(
                    "incomplete_source_metadata",
                    f"{column} is required for provenance.",
                    row_count=int(missing.sum()),
                )
            )


def _validate_prices(frame: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_numeric = numeric.isna().any(axis=1)
    if invalid_numeric.any():
        issues.append(
            ValidationIssue("invalid_numeric", "Price rows contain non-numeric values.", row_count=int(invalid_numeric.sum()))
        )
    prices = numeric[["open", "high", "low", "close"]]
    nonpositive = prices.le(0).any(axis=1)
    if nonpositive.any():
        issues.append(
            ValidationIssue("nonpositive_price", "OHLC values must be positive.", row_count=int(nonpositive.sum()))
        )
    invalid_high = numeric["high"] < prices.max(axis=1)
    invalid_low = numeric["low"] > prices.min(axis=1)
    if invalid_high.any() or invalid_low.any():
        issues.append(
            ValidationIssue(
                "invalid_ohlc",
                "High/low values violate OHLC bounds.",
                row_count=int((invalid_high | invalid_low).sum()),
            )
        )
    negative_volume = numeric["volume"] < 0
    if negative_volume.any():
        issues.append(
            ValidationIssue("negative_volume", "Volume must be non-negative.", row_count=int(negative_volume.sum()))
        )


def _validate_price_sessions(
    frame: pd.DataFrame,
    issues: list[ValidationIssue],
    completed_session: str | None,
) -> None:
    sessions = pd.to_datetime(frame["session"], errors="coerce").dropna().dt.normalize()
    if sessions.empty:
        return
    try:
        import exchange_calendars as xcals
    except ModuleNotFoundError:
        xcals = None
    if xcals is not None:
        calendar = xcals.get_calendar("XNYS")
        invalid = [value for value in sessions.drop_duplicates() if not calendar.is_session(value)]
        if invalid:
            issues.append(
                ValidationIssue(
                    "non_trading_session",
                    "US daily prices contain non-XNYS sessions.",
                    row_count=int(sessions.isin(invalid).sum()),
                )
            )
    if not completed_session:
        return
    expected = pd.Timestamp(completed_session).normalize()
    future = sessions > expected
    if future.any():
        issues.append(
            ValidationIssue(
                "future_session",
                f"Price sessions extend beyond completed session {expected.date()}.",
                row_count=int(future.sum()),
            )
        )
    if sessions.max() < expected:
        issues.append(
            ValidationIssue(
                "missing_completed_session",
                f"Price delta does not contain expected completed session {expected.date()}.",
            )
        )


def _validate_symbol_history(frame: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    working = frame.copy()
    working["_start"] = pd.to_datetime(working["effective_from"], errors="coerce")
    working["_end"] = pd.to_datetime(working["effective_to"], errors="coerce").fillna(pd.Timestamp.max)
    overlaps = 0
    for _, group in working.sort_values("_start").groupby("security_id"):
        prior_end: pd.Timestamp | None = None
        for _, row in group.iterrows():
            start = row["_start"]
            end = row["_end"]
            if prior_end is not None and start <= prior_end:
                overlaps += 1
            prior_end = max(prior_end, end) if prior_end is not None else end
    if overlaps:
        issues.append(
            ValidationIssue("overlapping_symbol_history", "Symbol-history intervals overlap.", row_count=overlaps)
        )


def _validate_actions(
    frame: pd.DataFrame,
    issues: list[ValidationIssue],
    incomplete_action_policy: str,
) -> None:
    allowed = {str(item) for item in CorporateActionType}
    invalid_type = ~frame["action_type"].astype(str).isin(allowed)
    if invalid_type.any():
        issues.append(
            ValidationIssue("invalid_action_type", "Unknown corporate-action type.", row_count=int(invalid_type.sum()))
        )
    action_type = frame["action_type"].astype(str)
    cash_amount = pd.to_numeric(frame["cash_amount"], errors="coerce")
    ratio = pd.to_numeric(frame["ratio"], errors="coerce")
    cash_present = (
        frame["cash_amount"].notna()
        & frame["cash_amount"].astype(str).str.strip().ne("")
    )
    ratio_present = (
        frame["ratio"].notna()
        & frame["ratio"].astype(str).str.strip().ne("")
    )
    cash_finite = cash_amount.notna() & cash_amount.map(
        lambda value: bool(pd.notna(value) and isfinite(float(value)))
    )
    ratio_finite = ratio.notna() & ratio.map(
        lambda value: bool(pd.notna(value) and isfinite(float(value)))
    )
    cash_missing = action_type.isin(
        {"cash_dividend", "special_dividend", "cash_merger"}
    ) & ~cash_present
    ratio_missing = action_type.isin(
        {"split", "capital_reduction", "stock_dividend", "spinoff", "stock_merger"}
    ) & ~ratio_present
    lifecycle = action_type.isin(
        {"cash_merger", "stock_merger", "ticker_change", "delisting"}
    )
    effective_missing = lifecycle & (
        frame["effective_date"].isna()
        | frame["effective_date"].astype(str).str.strip().eq("")
    )
    successor_missing = action_type.isin({"stock_merger", "ticker_change"}) & (
        frame["new_security_id"].isna()
        | frame["new_security_id"].astype(str).str.strip().eq("")
        | frame["new_symbol"].isna()
        | frame["new_symbol"].astype(str).str.strip().eq("")
    )
    delisting_recovery_missing = action_type.eq("delisting") & ~cash_present
    lifecycle_cash = (
        action_type.isin({"cash_merger", "delisting"})
        | (action_type.eq("stock_merger") & cash_present)
    )
    currency_missing = lifecycle_cash & (
        frame["currency"].isna()
        | frame["currency"].astype(str).str.strip().eq("")
    )
    incomplete = (
        cash_missing
        | ratio_missing
        | effective_missing
        | successor_missing
        | delisting_recovery_missing
        | currency_missing
    )
    if incomplete.any():
        severity = "error" if incomplete_action_policy == "block" else "warning"
        issues.append(
            ValidationIssue(
                "incomplete_corporate_action",
                "Corporate-action terms are incomplete.",
                severity=severity,
                row_count=int(incomplete.sum()),
            )
        )

    invalid_lifecycle_terms = (
        (
            action_type.eq("cash_merger")
            & cash_present
            & (~cash_finite | cash_amount.le(0))
        )
        | (
            action_type.eq("stock_merger")
            & ratio_present
            & (~ratio_finite | ratio.le(0))
        )
        | (
            action_type.eq("stock_merger")
            & cash_present
            & (~cash_finite | cash_amount.lt(0))
        )
        | (
            action_type.eq("delisting")
            & cash_present
            & (~cash_finite | cash_amount.lt(0))
        )
    )
    if invalid_lifecycle_terms.any():
        issues.append(
            ValidationIssue(
                "invalid_lifecycle_terms",
                "Lifecycle cash amounts and exchange ratios must be finite and within their allowed ranges.",
                row_count=int(invalid_lifecycle_terms.sum()),
            )
        )

    official = frame["official"].eq(True).fillna(False)  # noqa: E712
    source_url_present = (
        frame["source_url"].notna()
        & frame["source_url"].astype(str).str.strip().ne("")
        & frame["source_url"].astype(str).str.match(r"^https?://", case=False)
    )
    unverified_lifecycle = lifecycle & (~official | ~source_url_present)
    if unverified_lifecycle.any():
        issues.append(
            ValidationIssue(
                "unverified_lifecycle_source",
                "Lifecycle actions require an official source and HTTP(S) source URL.",
                row_count=int(unverified_lifecycle.sum()),
            )
        )


def _validate_lifecycle_resolutions(
    frame: pd.DataFrame,
    issues: list[ValidationIssue],
    completed_session: str | None,
) -> None:
    """Validate one resolution row without performing full candidate coverage."""

    # Keep the coverage module out of validation's import graph.  The publisher
    # invokes its cross-dataset gate separately after rebuilding the candidates.
    from .lifecycle_coverage import (
        LifecycleExceptionCode,
        LifecycleResolutionKind,
        TEMPORARY_EXCEPTION_CODES,
    )

    def present(column: str) -> pd.Series:
        values = frame[column]
        return values.notna() & values.astype(str).str.strip().ne("")

    identity_missing = ~(
        present("candidate_id")
        & present("security_id")
        & present("symbol")
        & present("last_price_date")
        & present("source_url")
    )
    if identity_missing.any():
        issues.append(
            ValidationIssue(
                "incomplete_lifecycle_resolution_identity",
                "Lifecycle resolutions require candidate, security, symbol, date, and source URL identity.",
                row_count=int(identity_missing.sum()),
            )
        )

    resolution = frame["resolution"].fillna("").astype(str).str.strip().str.lower()
    allowed_resolutions = {str(value) for value in LifecycleResolutionKind}
    invalid_resolution = ~resolution.isin(allowed_resolutions)
    if invalid_resolution.any():
        issues.append(
            ValidationIssue(
                "invalid_lifecycle_resolution",
                "Resolution must be applied or exception.",
                row_count=int(invalid_resolution.sum()),
            )
        )

    event_present = present("event_id")
    exception_code_present = present("exception_code")
    exception_reason_present = present("exception_reason")
    reviewed_by_present = present("reviewed_by")
    reviewed_at_present = present("reviewed_at")
    recheck_present = present("recheck_after")

    applied = resolution.eq(str(LifecycleResolutionKind.APPLIED))
    invalid_applied = applied & (
        ~event_present
        | exception_code_present
        | exception_reason_present
        | recheck_present
    )
    if invalid_applied.any():
        issues.append(
            ValidationIssue(
                "mixed_applied_exception_resolution",
                "Applied resolutions require event_id and cannot contain exception fields.",
                row_count=int(invalid_applied.sum()),
            )
        )

    exception = resolution.eq(str(LifecycleResolutionKind.EXCEPTION))
    allowed_exception_codes = {str(value) for value in LifecycleExceptionCode}
    exception_code = frame["exception_code"].fillna("").astype(str).str.strip()
    invalid_exception_code = exception & ~exception_code.isin(allowed_exception_codes)
    if invalid_exception_code.any():
        issues.append(
            ValidationIssue(
                "invalid_lifecycle_exception_code",
                "Exception resolutions require an allowed exception_code.",
                row_count=int(invalid_exception_code.sum()),
            )
        )

    incomplete_exception = exception & (
        event_present
        | ~exception_code_present
        | ~exception_reason_present
        | ~reviewed_by_present
        | ~reviewed_at_present
    )
    if incomplete_exception.any():
        issues.append(
            ValidationIssue(
                "incomplete_lifecycle_exception",
                "Exception resolutions cannot contain event_id and require code, reason, reviewer, and review time.",
                row_count=int(incomplete_exception.sum()),
            )
        )

    temporary_codes = {str(value) for value in TEMPORARY_EXCEPTION_CODES}
    temporary = exception & exception_code.isin(temporary_codes)
    temporary_without_recheck = temporary & ~recheck_present
    if temporary_without_recheck.any():
        issues.append(
            ValidationIssue(
                "temporary_exception_requires_recheck_after",
                "Temporary lifecycle exceptions require recheck_after.",
                row_count=int(temporary_without_recheck.sum()),
            )
        )

    if completed_session and temporary.any():
        completed = pd.to_datetime(completed_session, errors="coerce")
        recheck = pd.to_datetime(frame["recheck_after"], errors="coerce")
        expired = temporary & recheck.notna() & (
            recheck.dt.normalize() <= completed.normalize()
        )
        if expired.any():
            issues.append(
                ValidationIssue(
                    "expired_lifecycle_exception",
                    "Temporary lifecycle exception recheck dates must be after the completed session.",
                    row_count=int(expired.sum()),
                )
            )


def _lifecycle_successor_issues(
    actions: pd.DataFrame,
    master: pd.DataFrame,
    history: pd.DataFrame,
) -> list[ValidationIssue]:
    transfers = actions.loc[
        actions["action_type"].astype(str).isin({"stock_merger", "ticker_change"})
    ].copy()
    if transfers.empty:
        return []

    issues: list[ValidationIssue] = []
    successor_ids = transfers["new_security_id"].fillna("").astype(str).str.strip()
    missing_successor = successor_ids.eq("")
    if missing_successor.any():
        issues.append(
            ValidationIssue(
                "missing_action_successor_security",
                "Stock mergers and ticker changes require a successor security_id.",
                row_count=int(missing_successor.sum()),
            )
        )

    known_master_ids = (
        set(master["security_id"].astype(str)) if not master.empty else set()
    )
    unknown_successor = ~missing_successor & ~successor_ids.isin(known_master_ids)
    if unknown_successor.any():
        issues.append(
            ValidationIssue(
                "unknown_action_successor_security",
                "Lifecycle actions reference successor security_ids absent from security_master.",
                row_count=int(unknown_successor.sum()),
            )
        )

    mismatches = 0
    if not history.empty:
        working = history.copy()
        working["_security_id"] = working["security_id"].astype(str)
        working["_symbol"] = working["symbol"].astype(str).str.upper()
        working["_start"] = pd.to_datetime(working["effective_from"], errors="coerce")
        working["_end"] = pd.to_datetime(working["effective_to"], errors="coerce")
    else:
        working = pd.DataFrame()
    for index, row in transfers.iterrows():
        successor_id = successor_ids.loc[index]
        raw_symbol = row.get("new_symbol")
        new_symbol = "" if pd.isna(raw_symbol) else str(raw_symbol).strip().upper()
        effective = pd.to_datetime(row.get("effective_date"), errors="coerce")
        if (
            not successor_id
            or successor_id not in known_master_ids
            or not new_symbol
            or pd.isna(effective)
        ):
            continue
        if working.empty:
            mismatches += 1
            continue
        matches = working.loc[
            working["_security_id"].eq(successor_id)
            & working["_symbol"].eq(new_symbol)
            & working["_start"].notna()
            & working["_start"].le(effective)
            & (working["_end"].isna() | working["_end"].ge(effective))
        ]
        if matches.empty:
            mismatches += 1
    if mismatches:
        issues.append(
            ValidationIssue(
                "action_successor_symbol_mismatch",
                "Lifecycle successor security_ids do not match symbol history on the effective date.",
                row_count=mismatches,
            )
        )
    return issues


def _validate_adjustment_factors(frame: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    factors = frame[["split_factor", "total_return_factor"]].apply(pd.to_numeric, errors="coerce")
    invalid = factors.isna().any(axis=1) | factors.le(0).any(axis=1)
    if invalid.any():
        issues.append(
            ValidationIssue("invalid_adjustment_factor", "Adjustment factors must be positive.", row_count=int(invalid.sum()))
        )


def _validate_operations(frame: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    invalid = ~frame["operation"].astype(str).str.upper().isin({"ADD", "REMOVE"})
    if invalid.any():
        issues.append(
            ValidationIssue("invalid_membership_operation", "Operation must be ADD or REMOVE.", row_count=int(invalid.sum()))
        )


def _optional_repository_frame(repository, dataset: str) -> pd.DataFrame:
    return (
        repository.read_frame(dataset)
        if repository.current_manifest(dataset) is not None
        else pd.DataFrame()
    )


def _factor_action_issues(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    factors: pd.DataFrame,
) -> list[ValidationIssue]:
    if actions.empty:
        return []
    issues: list[ValidationIssue] = []
    split_actions = actions.loc[
        actions["action_type"].astype(str).isin({"split", "capital_reduction", "stock_dividend"})
        & actions["ratio"].notna()
    ]
    inconsistent = 0
    for action in split_actions.itertuples(index=False):
        security_factors = factors.loc[
            factors["security_id"].astype(str) == str(action.security_id)
        ].copy()
        security_factors["_session"] = pd.to_datetime(security_factors["session"]).dt.normalize()
        raw_ex_date = action.ex_date
        raw_effective = (
            raw_ex_date
            if pd.notna(raw_ex_date) and str(raw_ex_date).strip()
            else action.effective_date
        )
        effective = pd.Timestamp(raw_effective).normalize()
        before = security_factors.loc[security_factors["_session"] < effective].sort_values("_session")
        on_or_after = security_factors.loc[security_factors["_session"] >= effective].sort_values("_session")
        if before.empty or on_or_after.empty:
            continue
        observed = float(before.iloc[-1]["split_factor"]) / float(on_or_after.iloc[0]["split_factor"])
        expected = 1.0 / float(action.ratio)
        if abs(observed - expected) > 1e-8:
            inconsistent += 1
    if inconsistent:
        issues.append(
            ValidationIssue(
                "action_factor_mismatch",
                "Split-like actions do not match adjacent adjustment factors.",
                row_count=inconsistent,
            )
        )
    return issues


def _index_transition_issues(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    impossible = 0
    for index_id, index_events in events.groupby("index_id"):
        index_anchors = anchors.loc[anchors["index_id"].astype(str) == str(index_id)].copy()
        if index_anchors.empty:
            issues.append(
                ValidationIssue(
                    "missing_index_anchor",
                    f"Index events have no anchor: {index_id}",
                )
            )
            continue
        index_anchors["_date"] = pd.to_datetime(index_anchors["anchor_date"]).dt.normalize()
        anchor_date = index_anchors["_date"].min()
        members = set(
            index_anchors.loc[index_anchors["_date"] == anchor_date, "security_id"].astype(str)
        )
        working = index_events.copy()
        working["_date"] = pd.to_datetime(working["effective_date"]).dt.normalize()
        for row in working.loc[working["_date"] > anchor_date].sort_values(["_date", "event_id"]).itertuples(index=False):
            security_id = str(row.security_id)
            if str(row.operation).upper() == "ADD":
                if security_id in members:
                    impossible += 1
                members.add(security_id)
            else:
                if security_id not in members:
                    impossible += 1
                members.discard(security_id)
    if impossible:
        issues.append(
            ValidationIssue(
                "impossible_index_transition",
                "Index event sequence contains duplicate ADD or missing REMOVE transitions.",
                row_count=impossible,
            )
        )
    return issues


def _index_member_coverage_issues(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    history: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    allowed_price_gap_ids: Iterable[str] = (),
    allowed_identity_gap_fingerprints: Iterable[str] = (),
) -> list[ValidationIssue]:
    """Reject index identities whose active dates and price history disagree.

    Community constituent histories can retroactively replace an old issuer's
    ticker with a later issuer that reused the same ticker.  Primary-key and
    ADD/REMOVE checks cannot detect that collision.  Replaying membership
    intervals and comparing their edges with symbol/price coverage makes the
    mismatch fail closed before publication.  A short edge grace covers normal
    announcement/effective-date conventions and terminal trading halts.
    """

    if anchors.empty:
        return []

    prepared_history = history.copy()
    history_ranges: dict[
        str,
        tuple[tuple[pd.Timestamp, pd.Timestamp | None], ...],
    ] = {}
    if not prepared_history.empty:
        prepared_history["_security_id"] = prepared_history["security_id"].astype(str)
        prepared_history["_start"] = pd.to_datetime(
            prepared_history["effective_from"], errors="coerce"
        ).dt.normalize()
        prepared_history["_end"] = pd.to_datetime(
            prepared_history["effective_to"], errors="coerce"
        ).dt.normalize()
        history_ranges = {
            str(security_id): tuple(
                (
                    pd.Timestamp(start),
                    None if pd.isna(end) else pd.Timestamp(end),
                )
                for start, end in group[["_start", "_end"]].itertuples(
                    index=False, name=None
                )
                if pd.notna(start)
            )
            for security_id, group in prepared_history.groupby(
                "_security_id", sort=False
            )
        }

    price_sessions: dict[str, np.ndarray] = {}
    completed_session: pd.Timestamp | None = None
    if not prices.empty:
        prepared_prices = prices[["security_id", "session"]].copy()
        prepared_prices["_security_id"] = prepared_prices["security_id"].astype(str)
        prepared_prices["_session"] = pd.to_datetime(
            prepared_prices["session"], errors="coerce"
        ).dt.normalize()
        prepared_prices = prepared_prices.loc[prepared_prices["_session"].notna()]
        if not prepared_prices.empty:
            price_sessions = {
                str(security_id): np.sort(
                    group["_session"].drop_duplicates().to_numpy(dtype="datetime64[ns]")
                )
                for security_id, group in prepared_prices.groupby(
                    "_security_id", sort=False
                )
            }
            completed_session = pd.Timestamp(prepared_prices["_session"].max())

    missing_active_symbols: dict[tuple[str, str, str], str] = {}
    no_price_overlap: set[tuple[str, str, str, str]] = set()
    late_price_start: set[tuple[str, str, str, str]] = set()
    early_price_end: set[tuple[str, str, str, str]] = set()
    allowed_price_gaps = {
        str(security_id).strip()
        for security_id in allowed_price_gap_ids
        if str(security_id).strip()
    }
    allowed_identity_gaps = {
        str(value).strip().lower()
        for value in allowed_identity_gap_fingerprints
        if str(value).strip()
    }
    removal_rows: dict[tuple[str, str], tuple[dict[str, str], ...]] = {}
    if not events.empty:
        prepared_events = events.copy()
        prepared_events["_date"] = pd.to_datetime(
            prepared_events["effective_date"], errors="coerce"
        ).dt.normalize()
        prepared_events = prepared_events.loc[
            prepared_events["_date"].notna()
            & prepared_events["operation"].astype(str).str.upper().eq("REMOVE")
        ].sort_values(["_date", "event_id"], kind="stable")
        for (index_id, security_id), group in prepared_events.groupby(
            [
                prepared_events["index_id"].astype(str),
                prepared_events["security_id"].astype(str),
            ],
            sort=False,
        ):
            removal_rows[(str(index_id), str(security_id))] = tuple(
                {
                    "event_id": str(row.get("event_id", "")),
                    "effective_date": pd.Timestamp(row["_date"]).date().isoformat(),
                    "source": str(row.get("source", "") or ""),
                    "source_hash": str(row.get("source_hash", "") or ""),
                }
                for row in group.to_dict(orient="records")
            )
    grace = pd.Timedelta(days=INDEX_MEMBER_PRICE_EDGE_GRACE_DAYS)

    def check_active_symbols(
        index_id: str,
        effective_date: pd.Timestamp,
        members: set[str],
    ) -> None:
        if prepared_history.empty:
            return
        for security_id in members:
            ranges = history_ranges.get(security_id, ())
            active = any(
                start <= effective_date
                and (end is None or end >= effective_date)
                for start, end in ranges
            )
            if not active:
                replay_date = effective_date.date().isoformat()
                removals = removal_rows.get((index_id, security_id), ())
                next_remove = next(
                    (
                        item
                        for item in removals
                        if item["effective_date"] > replay_date
                    ),
                    {
                        "event_id": "",
                        "effective_date": "",
                        "source": "",
                        "source_hash": "",
                    },
                )
                key = (index_id, replay_date, security_id)
                missing_active_symbols[key] = index_member_identity_gap_fingerprint(
                    index_id=index_id,
                    replay_date=replay_date,
                    security_id=security_id,
                    next_remove_event_id=next_remove["event_id"],
                    next_remove_effective_date=next_remove["effective_date"],
                    next_remove_source=next_remove["source"],
                    next_remove_source_hash=next_remove["source_hash"],
                )

    for raw_index_id in sorted(anchors["index_id"].astype(str).unique()):
        index_id = str(raw_index_id)
        index_anchors = anchors.loc[
            anchors["index_id"].astype(str).eq(index_id)
        ].copy()
        index_anchors["_date"] = pd.to_datetime(
            index_anchors["anchor_date"], errors="coerce"
        ).dt.normalize()
        anchor_dates = index_anchors["_date"].dropna()
        if anchor_dates.empty:
            continue
        anchor_date = pd.Timestamp(anchor_dates.min())
        members = set(
            index_anchors.loc[
                index_anchors["_date"].eq(anchor_date), "security_id"
            ].astype(str)
        )
        starts = {security_id: anchor_date for security_id in members}
        intervals: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
        check_active_symbols(index_id, anchor_date, members)

        if events.empty:
            index_events = events.copy()
            index_events["_date"] = pd.Series(dtype="datetime64[ns]")
        else:
            index_events = events.loc[
                events["index_id"].astype(str).eq(index_id)
            ].copy()
            index_events["_date"] = pd.to_datetime(
                index_events["effective_date"], errors="coerce"
            ).dt.normalize()
            index_events = index_events.loc[
                index_events["_date"].notna()
                & index_events["_date"].gt(anchor_date)
            ].sort_values(["_date", "event_id"], kind="stable")

        for effective_date, date_events in index_events.groupby("_date", sort=True):
            date = pd.Timestamp(effective_date)
            for row in date_events.itertuples(index=False):
                security_id = str(row.security_id)
                operation = str(row.operation).upper()
                if operation == "REMOVE" and security_id in members:
                    intervals.append(
                        (
                            security_id,
                            starts.pop(security_id),
                            date - pd.Timedelta(days=1),
                        )
                    )
                    members.remove(security_id)
                elif operation == "ADD" and security_id not in members:
                    members.add(security_id)
                    starts[security_id] = date
            check_active_symbols(index_id, date, members)

        if completed_session is not None:
            intervals.extend(
                (security_id, starts[security_id], completed_session)
                for security_id in members
                if starts[security_id] <= completed_session
            )

        if not price_sessions:
            continue
        for security_id, start, end in intervals:
            if security_id in allowed_price_gaps:
                continue
            key = (
                index_id,
                security_id,
                start.date().isoformat(),
                end.date().isoformat(),
            )
            sessions = price_sessions.get(security_id)
            if sessions is None or len(sessions) == 0:
                no_price_overlap.add(key)
                continue
            first_position = int(
                np.searchsorted(sessions, np.datetime64(start), side="left")
            )
            after_position = int(
                np.searchsorted(sessions, np.datetime64(end), side="right")
            )
            if first_position >= after_position:
                no_price_overlap.add(key)
                continue
            first_price = pd.Timestamp(sessions[first_position])
            last_price = pd.Timestamp(sessions[after_position - 1])
            if first_price > start + grace:
                late_price_start.add(key)
            if last_price < end - grace:
                early_price_end.add(key)

    issues: list[ValidationIssue] = []
    unexpected_identity_gaps = {
        key: fingerprint
        for key, fingerprint in missing_active_symbols.items()
        if fingerprint not in allowed_identity_gaps
    }
    if unexpected_identity_gaps:
        issues.append(
            ValidationIssue(
                "index_member_missing_active_symbol",
                "Index members lack an active symbol identity on a replay date.",
                row_count=len(unexpected_identity_gaps),
                fingerprints=tuple(sorted(unexpected_identity_gaps.values())),
            )
        )
    if no_price_overlap:
        issues.append(
            ValidationIssue(
                "index_member_no_price_overlap",
                "Index membership intervals have no overlapping daily prices.",
                row_count=len(no_price_overlap),
            )
        )
    if late_price_start:
        issues.append(
            ValidationIssue(
                "index_member_price_starts_late",
                "Index member prices begin more than the allowed edge grace after membership starts.",
                row_count=len(late_price_start),
            )
        )
    if early_price_end:
        issues.append(
            ValidationIssue(
                "index_member_price_ends_early",
                "Index member prices end more than the allowed edge grace before membership ends.",
                row_count=len(early_price_end),
            )
        )
    return issues
