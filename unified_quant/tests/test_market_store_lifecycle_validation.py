from __future__ import annotations

import unittest

import pandas as pd

from supertrend_quant.market_store.lifecycle_coverage import LifecycleExceptionCode
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


def _action(event_id: str, action_type: str, **overrides) -> dict:
    values = {
        "event_id": event_id,
        "security_id": "OLD",
        "action_type": action_type,
        "effective_date": "2024-03-18",
        "ex_date": "2024-03-18",
        "announcement_date": "2024-03-18",
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": True,
        "source": "sec_edgar",
        "source_url": "https://www.sec.gov/Archives/example.txt",
        "source_kind": "official_crosscheck",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "abc123",
    }
    values.update(overrides)
    return values


def _master(*security_ids: str) -> pd.DataFrame:
    return pd.DataFrame({"security_id": list(security_ids)})


def _history(
    security_id: str,
    symbol: str,
    *,
    effective_from: str = "2015-01-01",
    effective_to: str = "",
) -> dict:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "effective_from": effective_from,
        "effective_to": effective_to,
    }


def _resolution(resolution: str, **overrides) -> dict:
    values = {
        "candidate_id": "candidate-1",
        "security_id": "OLD",
        "symbol": "OLD",
        "last_price_date": "2024-03-15",
        "resolution": resolution,
        "event_id": "",
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": "",
        "reviewed_at": "",
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        "source_url": "https://www.sec.gov/Archives/evidence.txt",
        "source": "lifecycle_review",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "evidence-sha256",
    }
    values.update(overrides)
    return values


class _SnapshotRepository:
    def __init__(self, **frames: pd.DataFrame):
        self.frames = frames

    def current_manifest(self, dataset: str):
        return object() if dataset in self.frames else None

    def read_frame(self, dataset: str):
        return self.frames[dataset].copy()


class LifecycleDatasetValidationTest(unittest.TestCase):
    def test_all_supported_lifecycle_types_accept_complete_verified_terms(self):
        frame = pd.DataFrame(
            [
                _action("cash", "cash_merger", cash_amount=95.0),
                _action(
                    "stock",
                    "stock_merger",
                    cash_amount=2.0,
                    ratio=0.5,
                    new_security_id="NEW-STOCK",
                    new_symbol="NSTK",
                ),
                _action(
                    "ticker",
                    "ticker_change",
                    new_security_id="NEW-TICKER",
                    new_symbol="NEW",
                ),
                _action("delist", "delisting", cash_amount=0.0),
            ]
        )

        report = validate_dataset(
            "corporate_actions",
            frame,
            incomplete_action_policy="block",
        )

        self.assertTrue(report.valid, report.issues)

    def test_missing_lifecycle_terms_follow_incomplete_action_policy(self):
        frame = pd.DataFrame(
            [
                _action("cash", "cash_merger", cash_amount=None),
                _action("stock", "stock_merger", ratio=0.5),
                _action("ticker", "ticker_change"),
                _action("delist", "delisting", cash_amount=None),
            ]
        )

        warning = validate_dataset("corporate_actions", frame)
        blocked = validate_dataset(
            "corporate_actions",
            frame,
            incomplete_action_policy="block",
        )

        self.assertTrue(warning.valid)
        self.assertFalse(blocked.valid)
        issue = next(
            item for item in blocked.issues if item.code == "incomplete_corporate_action"
        )
        self.assertEqual(issue.row_count, 4)

    def test_nonfinite_and_out_of_range_lifecycle_terms_are_rejected(self):
        frame = pd.DataFrame(
            [
                _action("cash", "cash_merger", cash_amount=0.0),
                _action(
                    "stock",
                    "stock_merger",
                    ratio=float("inf"),
                    new_security_id="NEW-STOCK",
                    new_symbol="NSTK",
                ),
                _action("delist", "delisting", cash_amount=-0.01),
            ]
        )

        report = validate_dataset("corporate_actions", frame)

        self.assertFalse(report.valid)
        issue = next(
            item for item in report.issues if item.code == "invalid_lifecycle_terms"
        )
        self.assertEqual(issue.row_count, 3)

    def test_cash_lifecycle_terms_require_currency(self):
        frame = pd.DataFrame(
            [_action("cash", "cash_merger", cash_amount=95.0, currency="")]
        )

        report = validate_dataset(
            "corporate_actions",
            frame,
            incomplete_action_policy="block",
        )

        self.assertFalse(report.valid)
        self.assertIn(
            "incomplete_corporate_action",
            {issue.code for issue in report.issues},
        )

    def test_lifecycle_actions_require_official_source_and_url(self):
        frame = pd.DataFrame(
            [
                _action(
                    "cash",
                    "cash_merger",
                    cash_amount=95.0,
                    official=False,
                    source_url="",
                )
            ]
        )

        report = validate_dataset("corporate_actions", frame)

        self.assertFalse(report.valid)
        issue = next(
            item for item in report.issues if item.code == "unverified_lifecycle_source"
        )
        self.assertEqual(issue.row_count, 1)

        nonofficial_url = pd.DataFrame(
            [
                _action(
                    "cash-memory",
                    "cash_merger",
                    cash_amount=95.0,
                    source_url="memory://parsed-fixture",
                )
            ]
        )
        report = validate_dataset("corporate_actions", nonofficial_url)
        self.assertIn(
            "unverified_lifecycle_source",
            {item.code for item in report.issues},
        )


class LifecycleResolutionRowValidationTest(unittest.TestCase):
    def test_schema_has_candidate_primary_key_and_all_audit_fields(self):
        spec = dataset_spec("lifecycle_resolutions")

        self.assertEqual(spec.primary_key, ("candidate_id",))
        self.assertTrue(
            {
                "candidate_id",
                "security_id",
                "symbol",
                "last_price_date",
                "resolution",
                "event_id",
                "exception_code",
                "exception_reason",
                "reviewed_by",
                "reviewed_at",
                "recheck_after",
                "successor_security_id",
                "successor_symbol",
                "source_url",
                "source",
                "retrieved_at",
                "source_hash",
            }.issubset(spec.required_columns)
        )

    def test_valid_applied_and_exception_rows_pass_format_validation(self):
        frame = pd.DataFrame(
            [
                _resolution("applied", event_id="event-1"),
                _resolution(
                    "exception",
                    candidate_id="candidate-2",
                    security_id="OTHER",
                    symbol="OTHER",
                    exception_code=LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
                    exception_reason="The consideration is not representable.",
                    reviewed_by="reviewer",
                    reviewed_at="2026-07-18T00:00:00Z",
                ),
            ]
        )

        report = validate_dataset(
            "lifecycle_resolutions",
            frame,
            completed_session="2026-07-18",
        )

        self.assertTrue(report.valid, report.issues)

    def test_resolution_kinds_and_fields_are_mutually_exclusive(self):
        frame = pd.DataFrame(
            [
                _resolution(
                    "applied",
                    event_id="event-1",
                    exception_reason="mixed",
                ),
                _resolution(
                    "exception",
                    candidate_id="candidate-2",
                    event_id="event-2",
                    exception_code="free-form",
                ),
                _resolution("unknown", candidate_id="candidate-3"),
            ]
        )

        report = validate_dataset("lifecycle_resolutions", frame)
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.valid)
        self.assertIn("mixed_applied_exception_resolution", codes)
        self.assertIn("invalid_lifecycle_exception_code", codes)
        self.assertIn("incomplete_lifecycle_exception", codes)
        self.assertIn("invalid_lifecycle_resolution", codes)

    def test_temporary_exception_requires_future_recheck_and_valid_dates(self):
        missing = pd.DataFrame(
            [
                _resolution(
                    "exception",
                    exception_code=LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
                    exception_reason="The successor has not been identified.",
                    reviewed_by="reviewer",
                    reviewed_at="2026-07-18T00:00:00Z",
                )
            ]
        )
        expired = missing.copy()
        expired.loc[0, "recheck_after"] = "2026-07-18"
        invalid = missing.copy()
        invalid.loc[0, "reviewed_at"] = "not-a-time"
        invalid.loc[0, "recheck_after"] = "not-a-date"

        missing_report = validate_dataset(
            "lifecycle_resolutions",
            missing,
            completed_session="2026-07-18",
        )
        expired_report = validate_dataset(
            "lifecycle_resolutions",
            expired,
            completed_session="2026-07-18",
        )
        invalid_report = validate_dataset(
            "lifecycle_resolutions",
            invalid,
            completed_session="2026-07-18",
        )

        self.assertIn(
            "temporary_exception_requires_recheck_after",
            {issue.code for issue in missing_report.issues},
        )
        self.assertIn(
            "expired_lifecycle_exception",
            {issue.code for issue in expired_report.issues},
        )
        self.assertIn(
            "invalid_date",
            {issue.code for issue in invalid_report.issues},
        )


class LifecycleSnapshotValidationTest(unittest.TestCase):
    def _repository(self, action: dict, target_history: dict | None = None):
        history_rows = [_history("OLD", "OLD")]
        if target_history is not None:
            history_rows.append(target_history)
        return _SnapshotRepository(
            corporate_actions=pd.DataFrame([action]),
            security_master=_master("OLD", "TARGET"),
            symbol_history=pd.DataFrame(history_rows),
        )

    def test_successor_must_exist_and_match_symbol_history_on_effective_date(self):
        valid_action = _action(
            "stock",
            "stock_merger",
            ratio=0.5,
            new_security_id="TARGET",
            new_symbol="NEW",
        )
        valid = validate_repository_snapshot(
            self._repository(valid_action, _history("TARGET", "NEW"))
        )
        self.assertTrue(valid.valid, valid.issues)

        unknown = validate_repository_snapshot(
            self._repository(
                {**valid_action, "new_security_id": "MISSING"},
                _history("TARGET", "NEW"),
            )
        )
        self.assertIn(
            "unknown_action_successor_security",
            {issue.code for issue in unknown.issues},
        )

        wrong_symbol = validate_repository_snapshot(
            self._repository(valid_action, _history("TARGET", "WRONG"))
        )
        self.assertIn(
            "action_successor_symbol_mismatch",
            {issue.code for issue in wrong_symbol.issues},
        )

        out_of_range = validate_repository_snapshot(
            self._repository(
                valid_action,
                _history("TARGET", "NEW", effective_from="2025-01-01"),
            )
        )
        self.assertIn(
            "action_successor_symbol_mismatch",
            {issue.code for issue in out_of_range.issues},
        )

    def test_snapshot_reports_missing_successor_id(self):
        action = _action("ticker", "ticker_change", new_symbol="NEW")
        report = validate_repository_snapshot(self._repository(action))

        self.assertIn(
            "missing_action_successor_security",
            {issue.code for issue in report.issues},
        )

    def test_snapshot_rejects_index_identity_and_price_range_collisions(self):
        anchors = pd.DataFrame(
            [
                {
                    "index_id": "sp500",
                    "anchor_date": "2015-01-07",
                    "security_id": "OLD",
                }
            ]
        )
        events = pd.DataFrame(
            [
                {
                    "event_id": "add-other",
                    "index_id": "sp500",
                    "effective_date": "2016-01-04",
                    "operation": "ADD",
                    "security_id": "OTHER",
                    "official": True,
                },
                {
                    "event_id": "remove-old",
                    "index_id": "sp500",
                    "effective_date": "2020-01-02",
                    "operation": "REMOVE",
                    "security_id": "OLD",
                    "official": True,
                },
            ]
        )
        prices = pd.DataFrame(
            [
                {"security_id": "OLD", "session": "2019-01-02"},
                {"security_id": "OLD", "session": "2019-12-31"},
                {"security_id": "OTHER", "session": "2016-01-04"},
                {"security_id": "OTHER", "session": "2019-12-31"},
            ]
        )
        history = pd.DataFrame(
            [
                _history("OLD", "OLD", effective_to="2015-12-31"),
                _history("OTHER", "OTHER"),
            ]
        )
        report = validate_repository_snapshot(
            _SnapshotRepository(
                index_constituent_anchors=anchors,
                index_membership_events=events,
                daily_price_raw=prices,
                symbol_history=history,
            )
        )

        codes = {issue.code for issue in report.issues}
        self.assertIn("index_member_missing_active_symbol", codes)
        self.assertIn("index_member_price_starts_late", codes)

    def test_snapshot_allows_short_terminal_price_gap(self):
        anchors = pd.DataFrame(
            [
                {
                    "index_id": "nasdaq100",
                    "anchor_date": "2025-01-02",
                    "security_id": "OLD",
                }
            ]
        )
        events = pd.DataFrame(
            [
                {
                    "event_id": "remove-old",
                    "index_id": "nasdaq100",
                    "effective_date": "2025-07-28",
                    "operation": "REMOVE",
                    "security_id": "OLD",
                    "official": True,
                }
            ]
        )
        prices = pd.DataFrame(
            [
                {"security_id": "OLD", "session": "2025-01-02"},
                {"security_id": "OLD", "session": "2025-07-17"},
            ]
        )
        history = pd.DataFrame([_history("OLD", "OLD", effective_to="2025-07-17")])

        report = validate_repository_snapshot(
            _SnapshotRepository(
                index_constituent_anchors=anchors,
                index_membership_events=events,
                daily_price_raw=prices,
                symbol_history=history,
            )
        )

        codes = {issue.code for issue in report.issues}
        self.assertNotIn("index_member_price_ends_early", codes)
        self.assertNotIn("index_member_missing_active_symbol", codes)

    def test_snapshot_checks_each_reentry_interval_not_global_price_bounds(self):
        anchors = pd.DataFrame(
            [
                {
                    "index_id": "sp500",
                    "anchor_date": "2015-01-02",
                    "security_id": "REUSED",
                }
            ]
        )
        events = pd.DataFrame(
            [
                {
                    "event_id": "remove-legacy",
                    "index_id": "sp500",
                    "effective_date": "2015-02-02",
                    "operation": "REMOVE",
                    "security_id": "REUSED",
                    "official": True,
                },
                {
                    "event_id": "add-reused",
                    "index_id": "sp500",
                    "effective_date": "2023-01-03",
                    "operation": "ADD",
                    "security_id": "REUSED",
                    "official": True,
                },
            ]
        )
        prices = pd.DataFrame(
            [
                {"security_id": "REUSED", "session": "2015-01-02"},
                {"security_id": "REUSED", "session": "2015-01-30"},
                # A later row keeps global min/max wide, but there are no prices
                # anywhere near the 2023 re-entry edge.
                {"security_id": "REUSED", "session": "2026-07-15"},
            ]
        )
        history = pd.DataFrame([_history("REUSED", "REUSED")])

        report = validate_repository_snapshot(
            _SnapshotRepository(
                index_constituent_anchors=anchors,
                index_membership_events=events,
                daily_price_raw=prices,
                symbol_history=history,
            )
        )

        self.assertIn(
            "index_member_price_starts_late",
            {issue.code for issue in report.issues},
        )

    def test_migration_can_allow_only_an_exact_preexisting_price_gap(self):
        anchors = pd.DataFrame(
            [
                {
                    "index_id": "sp500",
                    "anchor_date": "2015-01-02",
                    "security_id": "COV-FIXTURE",
                }
            ]
        )
        history = pd.DataFrame([_history("COV-FIXTURE", "COV")])
        prices = pd.DataFrame(
            [{"security_id": "OTHER", "session": "2026-07-15"}]
        )
        repository = _SnapshotRepository(
            index_constituent_anchors=anchors,
            daily_price_raw=prices,
            symbol_history=history,
        )

        strict = validate_repository_snapshot(repository)
        migration = validate_repository_snapshot(
            repository,
            allowed_index_price_gap_ids={"COV-FIXTURE"},
        )

        self.assertIn(
            "index_member_no_price_overlap",
            {issue.code for issue in strict.issues},
        )
        self.assertNotIn(
            "index_member_no_price_overlap",
            {issue.code for issue in migration.issues},
        )


if __name__ == "__main__":
    unittest.main()
