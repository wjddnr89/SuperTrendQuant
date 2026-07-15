from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .manifest import DatasetManifest, sha256_file
from .models import CorporateActionType, DataQuality
from .schemas import DatasetSpec, dataset_spec


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"
    row_count: int = 0


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


def validate_repository_snapshot(repository) -> ValidationReport:
    """Cross-dataset checks run before publication and during quant-data validate."""
    issues: list[ValidationIssue] = []
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
    cash_missing = action_type.isin(
        {"cash_dividend", "special_dividend", "cash_merger"}
    ) & frame["cash_amount"].isna()
    ratio_missing = action_type.isin(
        {"split", "capital_reduction", "stock_dividend", "spinoff", "stock_merger"}
    ) & frame["ratio"].isna()
    incomplete = cash_missing | ratio_missing
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
