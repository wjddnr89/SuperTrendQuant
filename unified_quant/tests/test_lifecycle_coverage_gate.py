from __future__ import annotations

import unittest

import pandas as pd

from supertrend_quant.market_store.lifecycle_coverage import (
    LifecycleExceptionCode,
    attach_lifecycle_candidate_ids,
    lifecycle_candidate_id,
    lifecycle_candidate_set_sha256,
    lifecycle_resolution_set_sha256,
    validate_lifecycle_coverage,
)


COMPLETED_SESSION = "2026-07-18"


def _candidates(*security_ids: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "security_id": security_id,
                "symbol": security_id,
                "last_price_date": f"2024-01-{number:02d}",
            }
            for number, security_id in enumerate(security_ids, start=1)
        ]
    )
    return attach_lifecycle_candidate_ids(frame)


def _resolution(
    candidate: pd.Series,
    resolution: str,
    **overrides,
) -> dict:
    values = {
        "candidate_id": candidate["candidate_id"],
        "security_id": candidate["security_id"],
        "resolution": resolution,
        "event_id": "",
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": "",
        "reviewed_at": "",
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        "source_hash": "report-sha256",
    }
    values.update(overrides)
    return values


def _exception(candidate: pd.Series, **overrides) -> dict:
    values = {
        "exception_code": LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "exception_reason": "The consideration cannot be represented safely.",
        "reviewed_by": "reviewer",
        "reviewed_at": "2026-07-18T00:00:00Z",
    }
    values.update(overrides)
    return _resolution(candidate, "exception", **values)


def _action(event_id: str, security_id: str, **overrides) -> dict:
    values = {
        "event_id": event_id,
        "security_id": security_id,
        "action_type": "cash_merger",
        "effective_date": "2024-01-02",
        "new_security_id": "",
        "new_symbol": "",
        "cash_amount": 50.0,
        "ratio": None,
        "official": True,
        "source_url": "https://www.sec.gov/Archives/example.txt",
        "source_kind": "official_crosscheck",
    }
    values.update(overrides)
    return values


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


class LifecycleCandidateIdentityTest(unittest.TestCase):
    def test_candidate_id_and_set_hash_are_deterministic(self):
        first = lifecycle_candidate_id("SEC-A", "2024-01-01")
        second = lifecycle_candidate_id("SEC-A", "2024-01-01T23:30:00")
        self.assertEqual(first, second)
        self.assertNotEqual(first, lifecycle_candidate_id("SEC-B", "2024-01-01"))

        candidates = _candidates("SEC-A", "SEC-B")
        reversed_candidates = candidates.iloc[::-1].reset_index(drop=True)
        self.assertEqual(
            lifecycle_candidate_set_sha256(candidates),
            lifecycle_candidate_set_sha256(reversed_candidates),
        )

    def test_supplied_noncanonical_candidate_id_is_rejected(self):
        candidates = _candidates("SEC-A")
        candidates.loc[0, "candidate_id"] = "wrong"
        resolution = pd.DataFrame([_exception(candidates.iloc[0])])

        report = validate_lifecycle_coverage(
            candidates,
            resolution,
            pd.DataFrame(),
            completed_session=COMPLETED_SESSION,
        )

        self.assertFalse(report.valid)
        self.assertIn("invalid_candidate_id", _codes(report))


class LifecycleCoverageClosureTest(unittest.TestCase):
    def test_applied_and_explicit_exception_close_the_exact_candidate_set(self):
        candidates = _candidates("SEC-A", "SEC-B")
        resolutions = pd.DataFrame(
            [
                _resolution(
                    candidates.iloc[0],
                    "applied",
                    event_id="event-a",
                ),
                _exception(candidates.iloc[1]),
            ]
        )
        actions = pd.DataFrame([_action("event-a", "SEC-A")])

        report = validate_lifecycle_coverage(
            candidates,
            resolutions,
            actions,
            completed_session=COMPLETED_SESSION,
        )

        self.assertTrue(report.valid, report.issues)
        self.assertEqual(report.candidate_count, 2)
        self.assertEqual(report.resolution_count, 2)
        self.assertEqual(report.applied_count, 1)
        self.assertEqual(report.exception_count, 1)
        self.assertEqual(report.open_count, 0)
        self.assertEqual(
            report.candidate_set_sha256,
            lifecycle_candidate_set_sha256(candidates),
        )
        self.assertEqual(
            report.resolution_set_sha256,
            lifecycle_resolution_set_sha256(resolutions),
        )
        self.assertEqual(report.manifest_metadata()["open_count"], 0)

    def test_all_exception_snapshot_does_not_require_action_columns(self):
        candidates = _candidates("SEC-A")
        report = validate_lifecycle_coverage(
            candidates,
            pd.DataFrame([_exception(candidates.iloc[0])]),
            pd.DataFrame(),
            completed_session=COMPLETED_SESSION,
        )

        self.assertTrue(report.valid, report.issues)
        self.assertEqual(report.exception_count, 1)

    def test_applied_and_exception_fields_are_mutually_exclusive(self):
        candidates = _candidates("SEC-A", "SEC-B")
        resolutions = pd.DataFrame(
            [
                _resolution(
                    candidates.iloc[0],
                    "applied",
                    event_id="event-a",
                    exception_code=LifecycleExceptionCode.ALREADY_REPRESENTED,
                    exception_reason="mixed",
                ),
                _exception(candidates.iloc[1], event_id="event-b"),
            ]
        )
        actions = pd.DataFrame(
            [
                _action("event-a", "SEC-A"),
                _action("event-b", "SEC-B"),
            ]
        )

        report = validate_lifecycle_coverage(
            candidates,
            resolutions,
            actions,
            completed_session=COMPLETED_SESSION,
        )

        self.assertFalse(report.valid)
        self.assertEqual(report.open_count, 2)
        self.assertIn("mixed_applied_exception_resolution", _codes(report))

    def test_missing_extra_and_duplicate_resolutions_are_rejected(self):
        candidates = _candidates("SEC-A", "SEC-B")
        first = _exception(candidates.iloc[0])
        extra = {**first, "candidate_id": "not-a-candidate", "security_id": "EXTRA"}
        duplicate = dict(first)
        resolutions = pd.DataFrame([first, duplicate, extra])

        report = validate_lifecycle_coverage(
            candidates,
            resolutions,
            pd.DataFrame(),
            completed_session=COMPLETED_SESSION,
        )

        codes = _codes(report)
        self.assertFalse(report.valid)
        self.assertIn("missing_lifecycle_resolution", codes)
        self.assertIn("unexpected_lifecycle_resolution", codes)
        self.assertIn("duplicate_lifecycle_resolution", codes)


class LifecycleExceptionValidationTest(unittest.TestCase):
    def _report(self, **overrides):
        candidates = _candidates("SEC-A")
        resolution = _exception(candidates.iloc[0], **overrides)
        return validate_lifecycle_coverage(
            candidates,
            pd.DataFrame([resolution]),
            pd.DataFrame(),
            completed_session=COMPLETED_SESSION,
        )

    def test_exception_code_must_be_allowed(self):
        report = self._report(exception_code="free_form_reason")

        self.assertFalse(report.valid)
        self.assertIn("invalid_lifecycle_exception_code", _codes(report))

    def test_temporary_exception_requires_future_recheck_after(self):
        missing = self._report(
            exception_code=LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
            recheck_after="",
        )
        expired = self._report(
            exception_code=LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
            recheck_after=COMPLETED_SESSION,
        )
        valid = self._report(
            exception_code=LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
            recheck_after="2026-08-18",
        )

        self.assertIn(
            "temporary_exception_requires_recheck_after",
            _codes(missing),
        )
        self.assertIn("expired_lifecycle_exception", _codes(expired))
        self.assertTrue(valid.valid, valid.issues)

    def test_recovery_uncertain_is_a_permanent_reviewed_exception(self):
        report = self._report(
            exception_code=LifecycleExceptionCode.RECOVERY_UNCERTAIN,
            exception_reason=(
                "The active receivership has no final per-share stockholder recovery."
            ),
            recheck_after="",
        )

        self.assertTrue(report.valid, report.issues)

    def test_exception_requires_reason_reviewer_and_valid_review_time(self):
        report = self._report(
            exception_reason="",
            reviewed_by="",
            reviewed_at="not-a-time",
        )

        codes = _codes(report)
        self.assertFalse(report.valid)
        self.assertIn("incomplete_lifecycle_exception", codes)
        self.assertIn("invalid_exception_reviewed_at", codes)


class AppliedLifecycleActionValidationTest(unittest.TestCase):
    def _report(self, action: dict | None, **resolution_overrides):
        candidates = _candidates("SEC-A")
        resolution = _resolution(
            candidates.iloc[0],
            "applied",
            event_id="event-a",
            **resolution_overrides,
        )
        action_columns = list(_action("fixture", "fixture"))
        actions = (
            pd.DataFrame([action])
            if action is not None
            else pd.DataFrame(columns=action_columns)
        )
        return validate_lifecycle_coverage(
            candidates,
            pd.DataFrame([resolution]),
            actions,
            completed_session=COMPLETED_SESSION,
        )

    def test_applied_event_must_exist(self):
        report = self._report(None)

        self.assertFalse(report.valid)
        self.assertIn("missing_applied_corporate_action", _codes(report))

    def test_applied_event_must_match_candidate_identity(self):
        report = self._report(_action("event-a", "OTHER"))

        self.assertFalse(report.valid)
        self.assertIn("applied_action_identity_mismatch", _codes(report))

    def test_transfer_requires_matching_successor_identity(self):
        missing = self._report(
            _action(
                "event-a",
                "SEC-A",
                action_type="stock_merger",
                cash_amount=None,
                ratio=0.5,
            )
        )
        mismatch = self._report(
            _action(
                "event-a",
                "SEC-A",
                action_type="ticker_change",
                cash_amount=None,
                new_security_id="NEW-A",
                new_symbol="NEW",
            ),
            successor_security_id="NEW-B",
            successor_symbol="OTHER",
        )

        self.assertIn("applied_action_missing_successor", _codes(missing))
        self.assertIn("applied_action_successor_mismatch", _codes(mismatch))

    def test_applied_event_requires_official_http_source(self):
        unofficial = self._report(
            _action("event-a", "SEC-A", official=False)
        )
        bad_source = self._report(
            _action("event-a", "SEC-A", source_url="memory://fixture")
        )

        self.assertIn("unofficial_applied_lifecycle_action", _codes(unofficial))
        self.assertIn("unverified_applied_lifecycle_source", _codes(bad_source))


if __name__ == "__main__":
    unittest.main()
