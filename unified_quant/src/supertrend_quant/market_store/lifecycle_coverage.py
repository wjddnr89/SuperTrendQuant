from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Iterable

import pandas as pd


COVERAGE_GATE_VERSION = 1
DEFAULT_SELECTION_RULE = "us_terminal_v1"
LIFECYCLE_ACTION_TYPES = frozenset(
    {"cash_merger", "stock_merger", "ticker_change", "delisting"}
)


class LifecycleResolutionKind(StrEnum):
    APPLIED = "applied"
    EXCEPTION = "exception"


class LifecycleExceptionCode(StrEnum):
    NOT_LIFECYCLE_EVENT = "not_lifecycle_event"
    ALREADY_REPRESENTED = "already_represented"
    UNSUPPORTED_CONSIDERATION = "unsupported_consideration"
    RECOVERY_UNCERTAIN = "recovery_uncertain"
    INSUFFICIENT_OFFICIAL_EVIDENCE = "insufficient_official_evidence"
    SUCCESSOR_UNRESOLVED = "successor_unresolved"
    PRICE_IDENTITY_CONFLICT = "price_identity_conflict"
    CROSSCHECK_FAILED = "crosscheck_failed"


TEMPORARY_EXCEPTION_CODES = frozenset(
    {
        LifecycleExceptionCode.INSUFFICIENT_OFFICIAL_EVIDENCE,
        LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
        LifecycleExceptionCode.PRICE_IDENTITY_CONFLICT,
        LifecycleExceptionCode.CROSSCHECK_FAILED,
    }
)


@dataclass(frozen=True)
class LifecycleCoverageIssue:
    code: str
    message: str
    candidate_ids: tuple[str, ...] = ()

    @property
    def row_count(self) -> int:
        return len(self.candidate_ids)


@dataclass(frozen=True)
class LifecycleCoverageReport:
    candidate_count: int
    resolution_count: int
    applied_count: int
    exception_count: int
    open_count: int
    candidate_set_sha256: str
    resolution_set_sha256: str
    issues: tuple[LifecycleCoverageIssue, ...] = ()
    gate_version: int = COVERAGE_GATE_VERSION
    selection_rule: str = DEFAULT_SELECTION_RULE

    @property
    def valid(self) -> bool:
        return not self.issues and self.open_count == 0

    @property
    def closed_count(self) -> int:
        return self.applied_count + self.exception_count

    def raise_for_errors(self) -> None:
        if self.issues:
            raise ValueError("; ".join(issue.message for issue in self.issues))

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "coverage_gate_version": self.gate_version,
            "selection_rule": self.selection_rule,
            "candidate_set_sha256": self.candidate_set_sha256,
            "resolution_set_sha256": self.resolution_set_sha256,
            "candidate_count": self.candidate_count,
            "resolution_count": self.resolution_count,
            "applied_count": self.applied_count,
            "exception_count": self.exception_count,
            "open_count": self.open_count,
        }


def lifecycle_candidate_id(
    security_id: str,
    last_price_date: str,
    *,
    selection_rule: str = DEFAULT_SELECTION_RULE,
) -> str:
    security = str(security_id).strip()
    rule = str(selection_rule).strip()
    if not security:
        raise ValueError("security_id is required for a lifecycle candidate_id")
    if not rule:
        raise ValueError("selection_rule is required for a lifecycle candidate_id")
    date = _date_iso(last_price_date, "last_price_date")
    return hashlib.sha256(f"{rule}|{security}|{date}".encode()).hexdigest()


def attach_lifecycle_candidate_ids(
    candidates: pd.DataFrame,
    *,
    selection_rule: str = DEFAULT_SELECTION_RULE,
) -> pd.DataFrame:
    _require_columns(candidates, ("security_id", "last_price_date"), "candidates")
    output = candidates.copy()
    output["candidate_id"] = [
        lifecycle_candidate_id(
            row.security_id,
            row.last_price_date,
            selection_rule=selection_rule,
        )
        for row in output.itertuples(index=False)
    ]
    return output


def lifecycle_candidate_set_sha256(
    candidates: pd.DataFrame,
    *,
    selection_rule: str = DEFAULT_SELECTION_RULE,
) -> str:
    identified = attach_lifecycle_candidate_ids(
        candidates,
        selection_rule=selection_rule,
    )
    records = sorted(
        {
            (
                str(row.candidate_id),
                _text(row.security_id),
                _date_iso(row.last_price_date, "last_price_date"),
            )
            for row in identified.itertuples(index=False)
        }
    )
    return _json_sha256(
        [
            {
                "candidate_id": candidate_id,
                "security_id": security_id,
                "last_price_date": last_price_date,
            }
            for candidate_id, security_id, last_price_date in records
        ]
    )


def lifecycle_resolution_set_sha256(resolutions: pd.DataFrame) -> str:
    columns = (
        "candidate_id",
        "security_id",
        "resolution",
        "event_id",
        "exception_code",
        "exception_reason",
        "reviewed_by",
        "reviewed_at",
        "recheck_after",
        "successor_security_id",
        "successor_symbol",
        "source_hash",
    )
    records = [
        {column: _text(row.get(column)) for column in columns}
        for row in resolutions.to_dict(orient="records")
    ]
    records.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    return _json_sha256(records)


def validate_lifecycle_coverage(
    candidates: pd.DataFrame,
    resolutions: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    *,
    completed_session: str,
    selection_rule: str = DEFAULT_SELECTION_RULE,
) -> LifecycleCoverageReport:
    """Validate exhaustive candidate closure without reading files or providers."""

    issues: list[LifecycleCoverageIssue] = []
    expected: dict[str, dict[str, str]] = {}
    candidate_columns = ("security_id", "last_price_date")
    missing_candidate_columns = _missing_columns(candidates, candidate_columns)
    if missing_candidate_columns:
        issues.append(
            LifecycleCoverageIssue(
                "missing_candidate_columns",
                "Lifecycle candidates are missing columns: "
                + ", ".join(missing_candidate_columns),
            )
        )
    else:
        for row in candidates.to_dict(orient="records"):
            try:
                canonical = lifecycle_candidate_id(
                    _text(row.get("security_id")),
                    _text(row.get("last_price_date")),
                    selection_rule=selection_rule,
                )
            except ValueError as exc:
                issues.append(
                    LifecycleCoverageIssue(
                        "invalid_candidate_identity",
                        str(exc),
                    )
                )
                continue
            supplied = _text(row.get("candidate_id"))
            if supplied and supplied != canonical:
                issues.append(
                    LifecycleCoverageIssue(
                        "invalid_candidate_id",
                        f"Candidate_id does not match canonical identity: {supplied}",
                        (canonical,),
                    )
                )
            if canonical in expected:
                issues.append(
                    LifecycleCoverageIssue(
                        "duplicate_lifecycle_candidate",
                        f"Lifecycle candidate appears more than once: {canonical}",
                        (canonical,),
                    )
                )
            expected[canonical] = {
                "security_id": _text(row.get("security_id")),
                "last_price_date": _date_iso(
                    row.get("last_price_date"),
                    "last_price_date",
                ),
            }

    candidate_hash = _json_sha256(
        [
            {
                "candidate_id": candidate_id,
                **expected[candidate_id],
            }
            for candidate_id in sorted(expected)
        ]
    )
    resolution_hash = lifecycle_resolution_set_sha256(resolutions)

    resolution_columns = (
        "candidate_id",
        "security_id",
        "resolution",
        "event_id",
        "exception_code",
        "exception_reason",
        "reviewed_by",
        "reviewed_at",
        "recheck_after",
    )
    missing_resolution_columns = _missing_columns(resolutions, resolution_columns)
    if missing_resolution_columns:
        issues.append(
            LifecycleCoverageIssue(
                "missing_resolution_columns",
                "Lifecycle resolutions are missing columns: "
                + ", ".join(missing_resolution_columns),
            )
        )
        return _report(
            expected,
            resolutions,
            (),
            (),
            candidate_hash,
            resolution_hash,
            issues,
            selection_rule,
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in resolutions.to_dict(orient="records"):
        grouped.setdefault(_text(row.get("candidate_id")), []).append(row)

    blank_candidate_rows = grouped.get("", [])
    if blank_candidate_rows:
        issues.append(
            LifecycleCoverageIssue(
                "invalid_resolution_candidate_id",
                "Lifecycle resolutions require a non-empty candidate_id.",
            )
        )

    expected_ids = set(expected)
    actual_ids = {value for value in grouped if value}
    missing_ids = tuple(sorted(expected_ids - actual_ids))
    extra_ids = tuple(sorted(actual_ids - expected_ids))
    if missing_ids:
        issues.append(
            LifecycleCoverageIssue(
                "missing_lifecycle_resolution",
                f"Lifecycle candidates are missing resolutions: {len(missing_ids)}",
                missing_ids,
            )
        )
    if extra_ids:
        issues.append(
            LifecycleCoverageIssue(
                "unexpected_lifecycle_resolution",
                f"Lifecycle resolutions contain unexpected candidates: {len(extra_ids)}",
                extra_ids,
            )
        )
    for candidate_id, rows in grouped.items():
        if candidate_id and len(rows) > 1:
            issues.append(
                LifecycleCoverageIssue(
                    "duplicate_lifecycle_resolution",
                    f"Lifecycle candidate has multiple resolutions: {candidate_id}",
                    (candidate_id,),
                )
            )

    has_applied_resolution = any(
        _text(row.get("resolution")).lower() == LifecycleResolutionKind.APPLIED
        for rows in grouped.values()
        for row in rows
    )
    action_index, action_issues = (
        _index_actions(corporate_actions)
        if has_applied_resolution
        else ({}, ())
    )
    issues.extend(action_issues)
    applied_references: dict[str, list[str]] = {}
    for candidate_id, rows in grouped.items():
        if len(rows) != 1:
            continue
        row = rows[0]
        if _text(row.get("resolution")).lower() == LifecycleResolutionKind.APPLIED:
            event_id = _text(row.get("event_id"))
            if event_id:
                applied_references.setdefault(event_id, []).append(candidate_id)
    duplicated_event_candidates = {
        candidate_id
        for candidate_ids in applied_references.values()
        if len(candidate_ids) > 1
        for candidate_id in candidate_ids
    }
    for event_id, candidate_ids in applied_references.items():
        if len(candidate_ids) > 1:
            issues.append(
                LifecycleCoverageIssue(
                    "duplicate_applied_event_reference",
                    f"Corporate action is referenced by multiple candidates: {event_id}",
                    tuple(sorted(candidate_ids)),
                )
            )

    completed = _optional_date(completed_session)
    closed_applied: set[str] = set()
    closed_exceptions: set[str] = set()
    for candidate_id in sorted(expected_ids & actual_ids):
        rows = grouped[candidate_id]
        if len(rows) != 1:
            continue
        row = rows[0]
        candidate_valid = True
        candidate = expected[candidate_id]
        resolution_security = _text(row.get("security_id"))
        if resolution_security != candidate["security_id"]:
            issues.append(
                LifecycleCoverageIssue(
                    "resolution_candidate_identity_mismatch",
                    f"Resolution security_id does not match candidate: {candidate_id}",
                    (candidate_id,),
                )
            )
            candidate_valid = False

        kind_value = _text(row.get("resolution")).lower()
        if kind_value not in {str(value) for value in LifecycleResolutionKind}:
            issues.append(
                LifecycleCoverageIssue(
                    "invalid_lifecycle_resolution",
                    f"Unknown lifecycle resolution for {candidate_id}: {kind_value}",
                    (candidate_id,),
                )
            )
            continue

        if kind_value == LifecycleResolutionKind.APPLIED:
            event_id = _text(row.get("event_id"))
            exception_values = tuple(
                _text(row.get(column))
                for column in ("exception_code", "exception_reason", "recheck_after")
            )
            if not event_id or any(exception_values):
                issues.append(
                    LifecycleCoverageIssue(
                        "mixed_applied_exception_resolution",
                        "Applied resolutions require event_id and cannot contain "
                        f"exception fields: {candidate_id}",
                        (candidate_id,),
                    )
                )
                candidate_valid = False
            if candidate_id in duplicated_event_candidates:
                candidate_valid = False
            if event_id:
                action = action_index.get(event_id)
                if action is None:
                    issues.append(
                        LifecycleCoverageIssue(
                            "missing_applied_corporate_action",
                            f"Applied lifecycle event is absent: {event_id}",
                            (candidate_id,),
                        )
                    )
                    candidate_valid = False
                else:
                    action_messages = _applied_action_errors(row, candidate, action)
                    if action_messages:
                        issues.extend(
                            LifecycleCoverageIssue(
                                code,
                                message,
                                (candidate_id,),
                            )
                            for code, message in action_messages
                        )
                        candidate_valid = False
            if candidate_valid:
                closed_applied.add(candidate_id)
            continue

        event_id = _text(row.get("event_id"))
        if event_id:
            issues.append(
                LifecycleCoverageIssue(
                    "mixed_applied_exception_resolution",
                    f"Exception resolution cannot reference event_id: {candidate_id}",
                    (candidate_id,),
                )
            )
            candidate_valid = False
        exception_code = _text(row.get("exception_code"))
        try:
            code = LifecycleExceptionCode(exception_code)
        except ValueError:
            issues.append(
                LifecycleCoverageIssue(
                    "invalid_lifecycle_exception_code",
                    f"Unknown lifecycle exception code for {candidate_id}: {exception_code}",
                    (candidate_id,),
                )
            )
            candidate_valid = False
            code = None
        for column in ("exception_reason", "reviewed_by", "reviewed_at"):
            if not _text(row.get(column)):
                issues.append(
                    LifecycleCoverageIssue(
                        "incomplete_lifecycle_exception",
                        f"Lifecycle exception requires {column}: {candidate_id}",
                        (candidate_id,),
                    )
                )
                candidate_valid = False
        if _text(row.get("reviewed_at")) and _optional_timestamp(
            row.get("reviewed_at")
        ) is None:
            issues.append(
                LifecycleCoverageIssue(
                    "invalid_exception_reviewed_at",
                    f"Lifecycle exception reviewed_at is invalid: {candidate_id}",
                    (candidate_id,),
                )
            )
            candidate_valid = False

        recheck_value = _text(row.get("recheck_after"))
        recheck = _optional_date(recheck_value)
        if code in TEMPORARY_EXCEPTION_CODES:
            if not recheck_value or recheck is None:
                issues.append(
                    LifecycleCoverageIssue(
                        "temporary_exception_requires_recheck_after",
                        f"Temporary lifecycle exception requires recheck_after: {candidate_id}",
                        (candidate_id,),
                    )
                )
                candidate_valid = False
            elif completed is None:
                issues.append(
                    LifecycleCoverageIssue(
                        "completed_session_required",
                        "completed_session is required to validate temporary exceptions.",
                        (candidate_id,),
                    )
                )
                candidate_valid = False
            elif recheck <= completed:
                issues.append(
                    LifecycleCoverageIssue(
                        "expired_lifecycle_exception",
                        f"Lifecycle exception recheck date has expired: {candidate_id}",
                        (candidate_id,),
                    )
                )
                candidate_valid = False
        elif recheck_value and recheck is None:
            issues.append(
                LifecycleCoverageIssue(
                    "invalid_exception_recheck_after",
                    f"Lifecycle exception recheck_after is invalid: {candidate_id}",
                    (candidate_id,),
                )
            )
            candidate_valid = False
        if candidate_valid:
            closed_exceptions.add(candidate_id)

    open_ids = tuple(
        sorted(expected_ids - closed_applied - closed_exceptions)
    )
    if open_ids:
        issues.append(
            LifecycleCoverageIssue(
                "open_lifecycle_candidate",
                f"Lifecycle candidates remain open: {len(open_ids)}",
                open_ids,
            )
        )
    return LifecycleCoverageReport(
        candidate_count=len(expected),
        resolution_count=len(resolutions),
        applied_count=len(closed_applied),
        exception_count=len(closed_exceptions),
        open_count=len(open_ids),
        candidate_set_sha256=candidate_hash,
        resolution_set_sha256=resolution_hash,
        issues=tuple(issues),
        selection_rule=selection_rule,
    )


def _index_actions(
    corporate_actions: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], tuple[LifecycleCoverageIssue, ...]]:
    required = (
        "event_id",
        "security_id",
        "action_type",
        "effective_date",
        "new_security_id",
        "new_symbol",
        "cash_amount",
        "ratio",
        "official",
        "source_url",
        "source_kind",
    )
    missing = _missing_columns(corporate_actions, required)
    if missing:
        return {}, (
            LifecycleCoverageIssue(
                "missing_corporate_action_columns",
                "Corporate actions are missing columns: " + ", ".join(missing),
            ),
        )
    output: dict[str, dict[str, Any]] = {}
    issues: list[LifecycleCoverageIssue] = []
    duplicates: set[str] = set()
    for row in corporate_actions.to_dict(orient="records"):
        event_id = _text(row.get("event_id"))
        if not event_id:
            continue
        if event_id in output:
            duplicates.add(event_id)
        output[event_id] = row
    if duplicates:
        issues.append(
            LifecycleCoverageIssue(
                "duplicate_corporate_action_event_id",
                f"Corporate action event_ids are duplicated: {len(duplicates)}",
                tuple(sorted(duplicates)),
            )
        )
        for event_id in duplicates:
            output.pop(event_id, None)
    return output, tuple(issues)


def _applied_action_errors(
    resolution: dict[str, Any],
    candidate: dict[str, str],
    action: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    errors: list[tuple[str, str]] = []
    event_id = _text(action.get("event_id"))
    if _text(action.get("security_id")) != candidate["security_id"]:
        errors.append(
            (
                "applied_action_identity_mismatch",
                f"Applied action security_id does not match candidate: {event_id}",
            )
        )
    action_type = _text(action.get("action_type")).lower()
    if action_type not in LIFECYCLE_ACTION_TYPES:
        errors.append(
            (
                "applied_action_not_lifecycle",
                f"Applied event is not a supported lifecycle action: {event_id}",
            )
        )
    if not _text(action.get("effective_date")):
        errors.append(
            (
                "applied_action_missing_effective_date",
                f"Applied lifecycle action lacks effective_date: {event_id}",
            )
        )
    if action_type in {"stock_merger", "ticker_change"}:
        successor_id = _text(action.get("new_security_id"))
        successor_symbol = _text(action.get("new_symbol"))
        if not successor_id or not successor_symbol:
            errors.append(
                (
                    "applied_action_missing_successor",
                    f"Applied lifecycle action lacks successor identity: {event_id}",
                )
            )
        expected_id = _text(resolution.get("successor_security_id"))
        expected_symbol = _text(resolution.get("successor_symbol"))
        if expected_id and expected_id != successor_id:
            errors.append(
                (
                    "applied_action_successor_mismatch",
                    f"Applied action successor security_id differs from resolution: {event_id}",
                )
            )
        if expected_symbol and expected_symbol.upper() != successor_symbol.upper():
            errors.append(
                (
                    "applied_action_successor_mismatch",
                    f"Applied action successor symbol differs from resolution: {event_id}",
                )
            )
    if action_type == "cash_merger" and not _positive_number(action.get("cash_amount")):
        errors.append(
            (
                "applied_action_incomplete_terms",
                f"Applied cash merger lacks positive cash consideration: {event_id}",
            )
        )
    if action_type == "stock_merger" and not _positive_number(action.get("ratio")):
        errors.append(
            (
                "applied_action_incomplete_terms",
                f"Applied stock merger lacks a positive exchange ratio: {event_id}",
            )
        )
    if action_type == "delisting" and not _nonnegative_number(
        action.get("cash_amount")
    ):
        errors.append(
            (
                "applied_action_incomplete_terms",
                f"Applied delisting lacks verified recovery terms: {event_id}",
            )
        )
    if _text(action.get("official")).lower() != "true":
        errors.append(
            (
                "unofficial_applied_lifecycle_action",
                f"Applied lifecycle action is not marked official: {event_id}",
            )
        )
    source_url = _text(action.get("source_url")).lower()
    if not source_url.startswith(("http://", "https://")) or not _text(
        action.get("source_kind")
    ):
        errors.append(
            (
                "unverified_applied_lifecycle_source",
                f"Applied lifecycle action lacks official source provenance: {event_id}",
            )
        )
    return tuple(errors)


def _report(
    expected: dict[str, dict[str, str]],
    resolutions: pd.DataFrame,
    closed_applied: Iterable[str],
    closed_exceptions: Iterable[str],
    candidate_hash: str,
    resolution_hash: str,
    issues: list[LifecycleCoverageIssue],
    selection_rule: str,
) -> LifecycleCoverageReport:
    applied = set(closed_applied)
    exceptions = set(closed_exceptions)
    open_ids = set(expected) - applied - exceptions
    return LifecycleCoverageReport(
        candidate_count=len(expected),
        resolution_count=len(resolutions),
        applied_count=len(applied),
        exception_count=len(exceptions),
        open_count=len(open_ids),
        candidate_set_sha256=candidate_hash,
        resolution_set_sha256=resolution_hash,
        issues=tuple(issues),
        selection_rule=selection_rule,
    )


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> tuple[str, ...]:
    return tuple(column for column in required if column not in frame.columns)


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = _missing_columns(frame, required)
    if missing:
        raise ValueError(f"{label} are missing columns: {', '.join(missing)}")


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _date_iso(value: Any, field: str) -> str:
    parsed = _optional_date(value)
    if parsed is None:
        raise ValueError(f"{field} must be a valid date")
    return parsed.date().isoformat()


def _optional_date(value: Any) -> pd.Timestamp | None:
    if not _text(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _optional_timestamp(value: Any) -> pd.Timestamp | None:
    if not _text(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _positive_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(numeric) and numeric > 0


def _nonnegative_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(numeric) and numeric >= 0
