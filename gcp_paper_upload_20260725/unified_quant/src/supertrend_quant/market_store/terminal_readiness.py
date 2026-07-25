"""Strategy-independent readiness checks for terminal security transitions.

The backtest ledger consumes a corporate action on its engine session even when
the source position is not held.  A bad terminal date can therefore either
leave a dead security buyable after its action was consumed or postpone the
settlement beyond the first session where the source has no price.  Share
transitions have an additional requirement: the successor must be priceable on
the exact session where the ledger creates it.

This module reproduces those engine semantics without instantiating a strategy:

* ``ex_date`` wins over ``effective_date``;
* a non-XNYS action date is consumed on the next XNYS session;
* the expected terminal boundary is the next XNYS session after the source's
  final valid close.

The public audit is read-only.  ``TerminalTransitionReport.raise_for_errors``
is the fail-closed publication/backtest gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import exchange_calendars as xcals
import numpy as np
import pandas as pd


TERMINAL_ACTION_TYPES = frozenset(
    {"cash_merger", "stock_merger", "ticker_change", "delisting"}
)
SHARE_TRANSITION_TYPES = frozenset({"stock_merger", "ticker_change"})

_REQUIRED_RELEASE_DATASETS = (
    "corporate_actions",
    "lifecycle_resolutions",
    "daily_price_raw",
    "index_constituent_anchors",
    "index_membership_events",
    "symbol_history",
    "security_master",
)


@dataclass(frozen=True)
class TerminalTransitionIssue:
    """One engine-reproducible terminal-transition blocker."""

    code: str
    message: str
    security_id: str
    symbol: str
    event_id: str = ""
    action_type: str = ""
    last_price_session: str = ""
    expected_transition_session: str = ""
    engine_session: str = ""
    action_date_field: str = ""
    action_date: str = ""
    first_reentry_session: str = ""
    affected_index_ids: tuple[str, ...] = ()
    successor_security_id: str = ""
    successor_symbol: str = ""
    successor_blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "security_id": self.security_id,
            "symbol": self.symbol,
            "event_id": self.event_id,
            "action_type": self.action_type,
            "last_price_session": self.last_price_session,
            "expected_transition_session": self.expected_transition_session,
            "engine_session": self.engine_session,
            "action_date_field": self.action_date_field,
            "action_date": self.action_date,
            "first_reentry_session": self.first_reentry_session,
            "affected_index_ids": list(self.affected_index_ids),
            "successor_security_id": self.successor_security_id,
            "successor_symbol": self.successor_symbol,
            "successor_blockers": list(self.successor_blockers),
        }


@dataclass(frozen=True)
class TerminalTransitionReport:
    """Read-only audit result suitable for a fail-closed gate."""

    release_version: str
    applied_resolution_count: int
    terminal_transition_count: int
    issues: tuple[TerminalTransitionIssue, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.issues

    @property
    def risk_symbols(self) -> tuple[str, ...]:
        return tuple(sorted({issue.symbol for issue in self.issues if issue.symbol}))

    @property
    def risk_security_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({issue.security_id for issue in self.issues if issue.security_id})
        )

    @property
    def issue_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(issue.code for issue in self.issues).items()))

    def raise_for_errors(self) -> None:
        if self.ready:
            return
        preview = ", ".join(
            f"{issue.symbol or issue.security_id}:{issue.code}"
            for issue in self.issues[:8]
        )
        if len(self.issues) > 8:
            preview += f", +{len(self.issues) - 8} more"
        raise RuntimeError(
            "Terminal-transition readiness is blocked"
            + (f" for release {self.release_version}" if self.release_version else "")
            + f": {preview}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "release_version": self.release_version,
            "ready": self.ready,
            "applied_resolution_count": self.applied_resolution_count,
            "terminal_transition_count": self.terminal_transition_count,
            "issue_count": len(self.issues),
            "issue_counts": self.issue_counts,
            "risk_symbols": list(self.risk_symbols),
            "risk_security_ids": list(self.risk_security_ids),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def audit_release_terminal_transitions(
    repository,
    release=None,
    *,
    calendar_name: str = "XNYS",
) -> TerminalTransitionReport:
    """Load one coherent immutable release and audit it without writing.

    Passing an explicit release pins every input version even if another
    process advances ``releases/current.json`` during the audit.
    """

    if release is None:
        release, _ = repository.current_release()
    if release is None:
        raise RuntimeError("No current data release is available for terminal audit.")
    missing = [
        dataset
        for dataset in _REQUIRED_RELEASE_DATASETS
        if not _text(release.dataset_versions.get(dataset))
    ]
    if missing:
        raise RuntimeError(
            f"Release {release.version} lacks terminal-audit datasets: "
            + ", ".join(missing)
        )
    frames = {
        dataset: repository.read_frame(
            dataset,
            release.dataset_versions[dataset],
        )
        for dataset in _REQUIRED_RELEASE_DATASETS
    }
    return audit_terminal_transitions(
        corporate_actions=frames["corporate_actions"],
        lifecycle_resolutions=frames["lifecycle_resolutions"],
        daily_price_raw=frames["daily_price_raw"],
        index_constituent_anchors=frames["index_constituent_anchors"],
        index_membership_events=frames["index_membership_events"],
        symbol_history=frames["symbol_history"],
        security_master=frames["security_master"],
        release_version=str(release.version),
        calendar_name=calendar_name,
    )


def audit_terminal_transitions(
    *,
    corporate_actions: pd.DataFrame,
    lifecycle_resolutions: pd.DataFrame,
    daily_price_raw: pd.DataFrame,
    index_constituent_anchors: pd.DataFrame,
    index_membership_events: pd.DataFrame,
    symbol_history: pd.DataFrame,
    security_master: pd.DataFrame,
    release_version: str = "",
    calendar_name: str = "XNYS",
) -> TerminalTransitionReport:
    """Audit applied terminal actions against prices, identity, and membership."""

    _require_columns(
        "corporate_actions",
        corporate_actions,
        (
            "event_id",
            "security_id",
            "action_type",
            "effective_date",
            "ex_date",
            "new_security_id",
            "new_symbol",
        ),
    )
    _require_columns(
        "lifecycle_resolutions",
        lifecycle_resolutions,
        (
            "security_id",
            "symbol",
            "last_price_date",
            "resolution",
            "event_id",
        ),
    )
    _require_columns(
        "daily_price_raw", daily_price_raw, ("security_id", "session", "close")
    )
    _require_columns(
        "index_constituent_anchors",
        index_constituent_anchors,
        ("index_id", "anchor_date", "security_id"),
    )
    _require_columns(
        "index_membership_events",
        index_membership_events,
        ("event_id", "index_id", "effective_date", "operation", "security_id"),
    )
    _require_columns(
        "symbol_history",
        symbol_history,
        ("security_id", "symbol", "effective_from", "effective_to"),
    )
    _require_columns(
        "security_master",
        security_master,
        ("security_id", "primary_symbol", "active_from", "active_to"),
    )

    calendar = xcals.get_calendar(calendar_name)
    price_index = _PriceIndex(daily_price_raw)
    membership = _MembershipResolver(
        index_constituent_anchors,
        index_membership_events,
    )
    identities = _IdentityIndex(symbol_history, security_master)

    action_rows = _unique_rows_by_key(corporate_actions, "event_id")
    applied = lifecycle_resolutions.loc[
        lifecycle_resolutions["resolution"].astype(str).str.lower().eq("applied")
    ].copy()
    issues: list[TerminalTransitionIssue] = []
    terminal_count = 0

    for resolution in applied.sort_values(
        ["last_price_date", "security_id"], kind="stable"
    ).to_dict("records"):
        security_id = _text(resolution.get("security_id"))
        symbol = _text(resolution.get("symbol")).upper()
        event_id = _text(resolution.get("event_id"))
        action = action_rows.get(event_id)
        if action is None:
            issues.append(
                TerminalTransitionIssue(
                    code="applied_resolution_action_missing",
                    message=(
                        "Applied lifecycle resolution has no matching corporate action."
                    ),
                    security_id=security_id,
                    symbol=symbol,
                    event_id=event_id,
                )
            )
            continue

        action_type = _text(action.get("action_type")).lower()
        if action_type not in TERMINAL_ACTION_TYPES:
            issues.append(
                TerminalTransitionIssue(
                    code="applied_resolution_not_terminal",
                    message=(
                        "Applied lifecycle resolution points to a non-terminal action."
                    ),
                    security_id=security_id,
                    symbol=symbol,
                    event_id=event_id,
                    action_type=action_type,
                )
            )
            continue
        terminal_count += 1

        if _text(action.get("security_id")) != security_id:
            issues.append(
                TerminalTransitionIssue(
                    code="terminal_action_security_mismatch",
                    message=(
                        "Lifecycle resolution and terminal action reference different "
                        "security IDs."
                    ),
                    security_id=security_id,
                    symbol=symbol,
                    event_id=event_id,
                    action_type=action_type,
                )
            )
            continue

        last_price = _date(resolution.get("last_price_date"))
        observed_last = price_index.last_session(security_id)
        if last_price is None or observed_last is None or last_price != observed_last:
            issues.append(
                TerminalTransitionIssue(
                    code="resolution_terminal_price_mismatch",
                    message=(
                        "Lifecycle last_price_date does not equal the final valid source "
                        "close in daily_price_raw."
                    ),
                    security_id=security_id,
                    symbol=symbol,
                    event_id=event_id,
                    action_type=action_type,
                    last_price_session=_iso(last_price),
                )
            )
            # Expected transition and re-entry checks cannot be trusted without
            # an exact terminal price boundary.  Continue fail-closed.
            continue

        if not calendar.is_session(last_price):
            issues.append(
                TerminalTransitionIssue(
                    code="terminal_price_not_xnys_session",
                    message="Lifecycle terminal price is not an XNYS session.",
                    security_id=security_id,
                    symbol=symbol,
                    event_id=event_id,
                    action_type=action_type,
                    last_price_session=_iso(last_price),
                )
            )
            continue

        expected_transition = _normal_session(calendar.next_session(last_price))
        action_date_field, raw_action_date = _engine_action_date(action)
        parsed_action_date = _date(raw_action_date)
        if parsed_action_date is None:
            issues.append(
                TerminalTransitionIssue(
                    code="terminal_action_date_invalid",
                    message=(
                        "Terminal action has no valid engine date; ex_date is preferred "
                        "when present."
                    ),
                    security_id=security_id,
                    symbol=symbol,
                    event_id=event_id,
                    action_type=action_type,
                    last_price_session=_iso(last_price),
                    expected_transition_session=_iso(expected_transition),
                    action_date_field=action_date_field,
                    action_date=_text(raw_action_date),
                )
            )
            continue
        try:
            engine_session = _normal_session(
                calendar.date_to_session(parsed_action_date, direction="next")
            )
        except (ValueError, IndexError) as exc:
            issues.append(
                TerminalTransitionIssue(
                    code="terminal_engine_session_unavailable",
                    message=f"Terminal action cannot be mapped to XNYS: {exc}",
                    security_id=security_id,
                    symbol=symbol,
                    event_id=event_id,
                    action_type=action_type,
                    last_price_session=_iso(last_price),
                    expected_transition_session=_iso(expected_transition),
                    action_date_field=action_date_field,
                    action_date=_iso(parsed_action_date),
                )
            )
            continue

        common = {
            "security_id": security_id,
            "symbol": symbol,
            "event_id": event_id,
            "action_type": action_type,
            "last_price_session": _iso(last_price),
            "expected_transition_session": _iso(expected_transition),
            "engine_session": _iso(engine_session),
            "action_date_field": action_date_field,
            "action_date": _iso(parsed_action_date),
        }

        if engine_session > expected_transition:
            issues.append(
                TerminalTransitionIssue(
                    code="terminal_action_after_expected_session",
                    message=(
                        "The engine consumes the terminal action after the first XNYS "
                        "session following the source's final price."
                    ),
                    **common,
                )
            )

        for candidate_session in price_index.sessions_after(
            security_id, engine_session
        ):
            active_indices = membership.member_indices_on(
                security_id,
                candidate_session,
            )
            if not active_indices:
                continue
            if not identities.symbol_active(
                security_id,
                symbol,
                candidate_session,
            ):
                continue
            issues.append(
                TerminalTransitionIssue(
                    code="source_reentry_after_terminal_action",
                    message=(
                        "After the action is consumed, the source can again be selected "
                        "because it still has an active index membership, symbol, and "
                        "valid close."
                    ),
                    first_reentry_session=_iso(candidate_session),
                    affected_index_ids=active_indices,
                    **common,
                )
            )
            break

        if action_type in SHARE_TRANSITION_TYPES:
            successor_security_id = _text(action.get("new_security_id"))
            successor_symbol = _text(action.get("new_symbol")).upper()
            blockers: list[str] = []
            if not successor_security_id:
                blockers.append("missing_successor_security_id")
            if not successor_symbol:
                blockers.append("missing_successor_symbol")
            if successor_security_id and not price_index.has_session(
                successor_security_id,
                engine_session,
            ):
                blockers.append("missing_valid_close")
            if successor_security_id and not identities.security_active(
                successor_security_id,
                engine_session,
            ):
                blockers.append("security_master_inactive")
            if (
                successor_security_id
                and successor_symbol
                and not identities.symbol_active(
                    successor_security_id,
                    successor_symbol,
                    engine_session,
                )
            ):
                blockers.append("symbol_history_inactive")
            if blockers:
                issues.append(
                    TerminalTransitionIssue(
                        code="successor_not_ready_on_transition",
                        message=(
                            "A stock merger or ticker change creates a successor that "
                            "cannot be valued and resolved on the engine session."
                        ),
                        successor_security_id=successor_security_id,
                        successor_symbol=successor_symbol,
                        successor_blockers=tuple(blockers),
                        **common,
                    )
                )

    return TerminalTransitionReport(
        release_version=release_version,
        applied_resolution_count=len(applied),
        terminal_transition_count=terminal_count,
        issues=tuple(
            sorted(
                issues,
                key=lambda issue: (
                    issue.engine_session or issue.last_price_session,
                    issue.security_id,
                    issue.code,
                ),
            )
        ),
    )


class _PriceIndex:
    def __init__(self, frame: pd.DataFrame):
        working = frame[["security_id", "session", "close"]].copy()
        working["_security_id"] = working["security_id"].map(_text)
        working["_session"] = pd.to_datetime(
            working["session"], errors="coerce"
        ).dt.normalize()
        working["_close"] = pd.to_numeric(working["close"], errors="coerce")
        valid = (
            working["_security_id"].ne("")
            & working["_session"].notna()
            & working["_close"].notna()
            & np.isfinite(working["_close"])
        )
        working = working.loc[valid, ["_security_id", "_session"]].drop_duplicates()
        self._sessions = {
            str(security_id): tuple(
                _normal_session(value)
                for value in sorted(group["_session"].unique())
            )
            for security_id, group in working.groupby("_security_id", sort=False)
        }
        self._sets = {
            security_id: frozenset(sessions)
            for security_id, sessions in self._sessions.items()
        }

    def last_session(self, security_id: str) -> pd.Timestamp | None:
        sessions = self._sessions.get(security_id, ())
        return sessions[-1] if sessions else None

    def sessions_after(
        self,
        security_id: str,
        session: pd.Timestamp,
    ) -> tuple[pd.Timestamp, ...]:
        return tuple(
            candidate
            for candidate in self._sessions.get(security_id, ())
            if candidate > session
        )

    def has_session(self, security_id: str, session: pd.Timestamp) -> bool:
        return session in self._sets.get(security_id, frozenset())


class _IdentityIndex:
    def __init__(self, history: pd.DataFrame, master: pd.DataFrame):
        prepared_history = history[
            ["security_id", "symbol", "effective_from", "effective_to"]
        ].copy()
        prepared_history["_security_id"] = prepared_history["security_id"].map(
            _text
        )
        prepared_history["_symbol"] = prepared_history["symbol"].map(
            lambda value: _text(value).upper()
        )
        prepared_history["_start"] = pd.to_datetime(
            prepared_history["effective_from"], errors="coerce"
        ).dt.normalize()
        prepared_history["_end"] = pd.to_datetime(
            prepared_history["effective_to"], errors="coerce"
        ).dt.normalize()
        self._history = prepared_history

        prepared_master = master[
            ["security_id", "active_from", "active_to"]
        ].copy()
        prepared_master["_security_id"] = prepared_master["security_id"].map(
            _text
        )
        prepared_master["_start"] = pd.to_datetime(
            prepared_master["active_from"], errors="coerce"
        ).dt.normalize()
        prepared_master["_end"] = pd.to_datetime(
            prepared_master["active_to"], errors="coerce"
        ).dt.normalize()
        self._master = prepared_master

    def symbol_active(
        self,
        security_id: str,
        symbol: str,
        session: pd.Timestamp,
    ) -> bool:
        rows = self._history.loc[
            self._history["_security_id"].eq(security_id)
            & self._history["_symbol"].eq(symbol.upper())
        ]
        return _has_active_interval(rows, session)

    def security_active(self, security_id: str, session: pd.Timestamp) -> bool:
        rows = self._master.loc[self._master["_security_id"].eq(security_id)]
        return _has_active_interval(rows, session)


class _MembershipResolver:
    """Resolve membership for one security without rebuilding full universes."""

    def __init__(self, anchors: pd.DataFrame, events: pd.DataFrame):
        prepared_anchors = anchors[["index_id", "anchor_date", "security_id"]].copy()
        prepared_anchors["_index_id"] = prepared_anchors["index_id"].map(_text)
        prepared_anchors["_date"] = pd.to_datetime(
            prepared_anchors["anchor_date"], errors="coerce"
        ).dt.normalize()
        if prepared_anchors["_index_id"].eq("").any() or prepared_anchors[
            "_date"
        ].isna().any():
            raise ValueError("Index anchors contain blank IDs or invalid dates.")
        self._anchor_dates: dict[str, tuple[pd.Timestamp, ...]] = {}
        self._anchor_members: dict[tuple[str, pd.Timestamp], frozenset[str]] = {}
        for (index_id, date), group in prepared_anchors.groupby(
            ["_index_id", "_date"], sort=True
        ):
            normalized = _normal_session(date)
            self._anchor_members[(str(index_id), normalized)] = frozenset(
                group["security_id"].map(_text)
            )
        for index_id, group in prepared_anchors.groupby("_index_id", sort=True):
            self._anchor_dates[str(index_id)] = tuple(
                _normal_session(value) for value in sorted(group["_date"].unique())
            )

        event_columns = [
            "event_id",
            "index_id",
            "effective_date",
            "operation",
            "security_id",
        ]
        if "official" in events.columns:
            event_columns.append("official")
        prepared_events = events[event_columns].copy()
        prepared_events["_index_id"] = prepared_events["index_id"].map(_text)
        prepared_events["_security_id"] = prepared_events["security_id"].map(
            _text
        )
        prepared_events["_date"] = pd.to_datetime(
            prepared_events["effective_date"], errors="coerce"
        ).dt.normalize()
        prepared_events["_operation"] = prepared_events["operation"].map(
            lambda value: _text(value).upper()
        )
        invalid = (
            prepared_events["_index_id"].eq("")
            | prepared_events["_security_id"].eq("")
            | prepared_events["_date"].isna()
            | ~prepared_events["_operation"].isin({"ADD", "REMOVE"})
        )
        if invalid.any():
            raise ValueError("Index membership events contain invalid transition rows.")

        transitions: dict[
            tuple[str, str], list[tuple[pd.Timestamp, str]]
        ] = {}
        for (index_id, security_id, date), group in prepared_events.groupby(
            ["_index_id", "_security_id", "_date"], sort=True
        ):
            operations = set(group["_operation"])
            official_operations: set[str] = set()
            if "official" in group.columns:
                official = group["official"].map(_truthy)
                official_operations = set(group.loc[official, "_operation"])
            if len(operations) > 1:
                if len(official_operations) != 1:
                    raise ValueError(
                        "Unresolved index membership conflict for "
                        f"{security_id}/{index_id}/{_iso(_normal_session(date))}."
                    )
                operation = next(iter(official_operations))
            else:
                operation = next(iter(operations))
            transitions.setdefault((str(index_id), str(security_id)), []).append(
                (_normal_session(date), operation)
            )
        self._transitions = {
            key: tuple(sorted(values)) for key, values in transitions.items()
        }

    def member_indices_on(
        self,
        security_id: str,
        session: pd.Timestamp,
    ) -> tuple[str, ...]:
        active: list[str] = []
        for index_id, anchor_dates in self._anchor_dates.items():
            eligible = [date for date in anchor_dates if date <= session]
            if not eligible:
                continue
            anchor_date = eligible[-1]
            state = security_id in self._anchor_members[(index_id, anchor_date)]
            for effective_date, operation in self._transitions.get(
                (index_id, security_id), ()
            ):
                if effective_date <= anchor_date:
                    continue
                if effective_date > session:
                    break
                state = operation == "ADD"
            if state:
                active.append(index_id)
        return tuple(sorted(active))


def _has_active_interval(rows: pd.DataFrame, session: pd.Timestamp) -> bool:
    if rows.empty:
        return False
    return bool(
        (
            rows["_start"].notna()
            & rows["_start"].le(session)
            & (rows["_end"].isna() | rows["_end"].ge(session))
        ).any()
    )


def _engine_action_date(action: Mapping[str, Any]) -> tuple[str, Any]:
    ex_date = action.get("ex_date")
    if _text(ex_date):
        return "ex_date", ex_date
    return "effective_date", action.get("effective_date")


def _unique_rows_by_key(
    frame: pd.DataFrame,
    key: str,
) -> dict[str, dict[str, Any]]:
    normalized = frame[key].map(_text)
    duplicates = normalized.ne("") & normalized.duplicated(keep=False)
    if duplicates.any():
        values = ", ".join(sorted(set(normalized.loc[duplicates])))
        raise ValueError(f"Duplicate {key} rows are not auditable: {values}")
    return {
        _text(row.get(key)): row
        for row in frame.to_dict("records")
        if _text(row.get(key))
    }


def _require_columns(
    dataset: str,
    frame: pd.DataFrame,
    required: tuple[str, ...],
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset} is missing columns: {', '.join(missing)}")


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "yes"}


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _date(value: Any) -> pd.Timestamp | None:
    if not _text(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return _normal_session(parsed)


def _normal_session(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _iso(value: pd.Timestamp | None) -> str:
    return value.date().isoformat() if value is not None else ""


__all__ = [
    "SHARE_TRANSITION_TYPES",
    "TERMINAL_ACTION_TYPES",
    "TerminalTransitionIssue",
    "TerminalTransitionReport",
    "audit_release_terminal_transitions",
    "audit_terminal_transitions",
]
