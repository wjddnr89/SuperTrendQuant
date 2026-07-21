from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from supertrend_quant.market_store.terminal_readiness import (
    audit_release_terminal_transitions,
    audit_terminal_transitions,
)


def _frames() -> dict[str, pd.DataFrame]:
    actions = pd.DataFrame(
        [
            {
                "event_id": "late-event",
                "security_id": "LATE-ID",
                "action_type": "cash_merger",
                "effective_date": "2024-01-05",
                "ex_date": "",
                "new_security_id": "",
                "new_symbol": "",
            },
            {
                "event_id": "reentry-event",
                "security_id": "REENTRY-ID",
                "action_type": "cash_merger",
                # The engine must prefer the earlier ex-date.
                "effective_date": "2024-01-08",
                "ex_date": "2024-01-02",
                "new_security_id": "",
                "new_symbol": "",
            },
            {
                "event_id": "successor-event",
                "security_id": "SOURCE-ID",
                "action_type": "stock_merger",
                # Saturday rolls to Monday, 2024-01-08.
                "effective_date": "2024-01-06",
                "ex_date": "",
                "new_security_id": "SUCCESSOR-ID",
                "new_symbol": "NEXT",
            },
        ]
    )
    resolutions = pd.DataFrame(
        [
            {
                "security_id": "LATE-ID",
                "symbol": "LATE",
                "last_price_date": "2024-01-02",
                "resolution": "applied",
                "event_id": "late-event",
            },
            {
                "security_id": "REENTRY-ID",
                "symbol": "REENTRY",
                "last_price_date": "2024-01-05",
                "resolution": "applied",
                "event_id": "reentry-event",
            },
            {
                "security_id": "SOURCE-ID",
                "symbol": "SOURCE",
                "last_price_date": "2024-01-05",
                "resolution": "applied",
                "event_id": "successor-event",
            },
        ]
    )
    prices = pd.DataFrame(
        [
            {"security_id": "LATE-ID", "session": "2024-01-02", "close": 10.0},
            *(
                {
                    "security_id": "REENTRY-ID",
                    "session": session,
                    "close": 20.0,
                }
                for session in (
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                )
            ),
            {"security_id": "SOURCE-ID", "session": "2024-01-05", "close": 30.0},
            # The successor is one session late and cannot value a converted
            # position on the 2024-01-08 engine session.
            {
                "security_id": "SUCCESSOR-ID",
                "session": "2024-01-09",
                "close": 40.0,
            },
        ]
    )
    anchors = pd.DataFrame(
        [
            {
                "index_id": "test-index",
                "anchor_date": "2024-01-01",
                "security_id": "REENTRY-ID",
            }
        ]
    )
    events = pd.DataFrame(
        columns=(
            "event_id",
            "index_id",
            "effective_date",
            "operation",
            "security_id",
            "official",
        )
    )
    history = pd.DataFrame(
        [
            {
                "security_id": "LATE-ID",
                "symbol": "LATE",
                "effective_from": "2024-01-01",
                "effective_to": "2024-01-02",
            },
            {
                "security_id": "REENTRY-ID",
                "symbol": "REENTRY",
                "effective_from": "2024-01-01",
                "effective_to": "2024-01-05",
            },
            {
                "security_id": "SOURCE-ID",
                "symbol": "SOURCE",
                "effective_from": "2024-01-01",
                "effective_to": "2024-01-05",
            },
            {
                "security_id": "SUCCESSOR-ID",
                "symbol": "NEXT",
                "effective_from": "2024-01-09",
                "effective_to": "",
            },
        ]
    )
    master = pd.DataFrame(
        [
            {
                "security_id": "LATE-ID",
                "primary_symbol": "LATE",
                "active_from": "2024-01-01",
                "active_to": "2024-01-02",
            },
            {
                "security_id": "REENTRY-ID",
                "primary_symbol": "REENTRY",
                "active_from": "2024-01-01",
                "active_to": "2024-01-05",
            },
            {
                "security_id": "SOURCE-ID",
                "primary_symbol": "SOURCE",
                "active_from": "2024-01-01",
                "active_to": "2024-01-05",
            },
            {
                "security_id": "SUCCESSOR-ID",
                "primary_symbol": "NEXT",
                "active_from": "2024-01-09",
                "active_to": "",
            },
        ]
    )
    return {
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "daily_price_raw": prices,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "symbol_history": history,
        "security_master": master,
    }


def _audit(frames: dict[str, pd.DataFrame]):
    return audit_terminal_transitions(**frames, release_version="fixture-v1")


def test_audit_reproduces_all_three_engine_failure_modes() -> None:
    report = _audit(_frames())

    assert not report.ready
    assert report.applied_resolution_count == 3
    assert report.terminal_transition_count == 3
    assert report.issue_counts == {
        "source_reentry_after_terminal_action": 1,
        "successor_not_ready_on_transition": 1,
        "terminal_action_after_expected_session": 1,
    }
    assert report.risk_symbols == ("LATE", "REENTRY", "SOURCE")

    by_symbol = {issue.symbol: issue for issue in report.issues}
    late = by_symbol["LATE"]
    assert late.engine_session == "2024-01-05"
    assert late.expected_transition_session == "2024-01-03"
    assert late.action_date_field == "effective_date"

    reentry = by_symbol["REENTRY"]
    assert reentry.engine_session == "2024-01-02"
    assert reentry.action_date_field == "ex_date"
    assert reentry.first_reentry_session == "2024-01-03"
    assert reentry.affected_index_ids == ("test-index",)

    successor = by_symbol["SOURCE"]
    assert successor.engine_session == "2024-01-08"
    assert successor.expected_transition_session == "2024-01-08"
    assert successor.successor_blockers == (
        "missing_valid_close",
        "security_master_inactive",
        "symbol_history_inactive",
    )
    with pytest.raises(RuntimeError, match="fixture-v1"):
        report.raise_for_errors()


def test_exact_boundary_and_ready_successor_pass() -> None:
    frames = _frames()
    actions = frames["corporate_actions"].set_index("event_id")
    actions.loc["late-event", "effective_date"] = "2024-01-03"
    actions.loc["reentry-event", "ex_date"] = "2024-01-08"
    frames["corporate_actions"] = actions.reset_index()

    successor_price = frames["daily_price_raw"]["security_id"].eq("SUCCESSOR-ID")
    frames["daily_price_raw"].loc[successor_price, "session"] = "2024-01-08"
    successor_history = frames["symbol_history"]["security_id"].eq("SUCCESSOR-ID")
    frames["symbol_history"].loc[
        successor_history, "effective_from"
    ] = "2024-01-08"
    successor_master = frames["security_master"]["security_id"].eq("SUCCESSOR-ID")
    frames["security_master"].loc[
        successor_master, "active_from"
    ] = "2024-01-08"

    report = _audit(frames)

    assert report.ready
    assert report.issues == ()
    report.raise_for_errors()


def test_official_remove_prevents_reentry_and_resolves_lower_grade_conflict() -> None:
    frames = _frames()
    frames["index_membership_events"] = pd.DataFrame(
        [
            {
                "event_id": "unofficial-add",
                "index_id": "test-index",
                "effective_date": "2024-01-03",
                "operation": "ADD",
                "security_id": "REENTRY-ID",
                "official": False,
            },
            {
                "event_id": "official-remove",
                "index_id": "test-index",
                "effective_date": "2024-01-03",
                "operation": "REMOVE",
                "security_id": "REENTRY-ID",
                "official": True,
            },
        ]
    )

    report = _audit(frames)

    assert not any(
        issue.code == "source_reentry_after_terminal_action"
        and issue.symbol == "REENTRY"
        for issue in report.issues
    )


def test_unresolved_same_grade_membership_conflict_fails_closed() -> None:
    frames = _frames()
    frames["index_membership_events"] = pd.DataFrame(
        [
            {
                "event_id": "add",
                "index_id": "test-index",
                "effective_date": "2024-01-03",
                "operation": "ADD",
                "security_id": "REENTRY-ID",
                "official": True,
            },
            {
                "event_id": "remove",
                "index_id": "test-index",
                "effective_date": "2024-01-03",
                "operation": "REMOVE",
                "security_id": "REENTRY-ID",
                "official": True,
            },
        ]
    )

    with pytest.raises(ValueError, match="Unresolved index membership conflict"):
        _audit(frames)


def test_invalid_present_ex_date_does_not_fall_back_to_effective_date() -> None:
    frames = _frames()
    actions = frames["corporate_actions"]
    target = actions["event_id"].eq("late-event")
    actions.loc[target, "ex_date"] = "not-a-date"
    actions.loc[target, "effective_date"] = "2024-01-03"

    report = _audit(frames)

    issue = next(issue for issue in report.issues if issue.symbol == "LATE")
    assert issue.code == "terminal_action_date_invalid"
    assert issue.action_date_field == "ex_date"
    assert issue.action_date == "not-a-date"


def test_terminal_boundary_must_equal_final_valid_close() -> None:
    frames = _frames()
    target = frames["lifecycle_resolutions"]["security_id"].eq("LATE-ID")
    frames["lifecycle_resolutions"].loc[
        target, "last_price_date"
    ] = "2024-01-03"

    report = _audit(frames)

    issue = next(issue for issue in report.issues if issue.symbol == "LATE")
    assert issue.code == "resolution_terminal_price_mismatch"


def test_release_loader_pins_every_dataset_version() -> None:
    frames = _frames()
    versions = {dataset: f"{dataset}-v1" for dataset in frames}
    release = SimpleNamespace(version="pinned-release", dataset_versions=versions)

    class Repository:
        def __init__(self):
            self.reads: list[tuple[str, str]] = []

        def current_release(self):  # pragma: no cover - explicit release is used
            raise AssertionError("current pointer must not be reread")

        def read_frame(self, dataset, version):
            self.reads.append((dataset, version))
            return deepcopy(frames[dataset])

    repository = Repository()
    report = audit_release_terminal_transitions(repository, release)

    assert report.release_version == "pinned-release"
    assert set(repository.reads) == set(versions.items())


def test_missing_required_column_fails_closed() -> None:
    frames = _frames()
    frames["daily_price_raw"] = frames["daily_price_raw"].drop(columns="close")

    with pytest.raises(ValueError, match="daily_price_raw is missing columns: close"):
        _audit(frames)
