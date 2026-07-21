from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from supertrend_quant.market_store.terminal_readiness import (
    TerminalTransitionIssue,
    TerminalTransitionReport,
)
from supertrend_quant.market_store import terminal_readiness_exceptions as reviewed


DRAFT_PATH = (
    Path(__file__).parents[1]
    / "configs/drafts/us_terminal_readiness_reviewed_exceptions.yaml"
)
PUBLISH_PATH = Path(__file__).parents[1] / "scripts/publish_and_verify_r2.py"
PUBLISH_SPEC = importlib.util.spec_from_file_location(
    "publish_and_verify_r2_reviewed_terminal_test",
    PUBLISH_PATH,
)
if PUBLISH_SPEC is None or PUBLISH_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load {PUBLISH_PATH}")
publish_script = importlib.util.module_from_spec(PUBLISH_SPEC)
PUBLISH_SPEC.loader.exec_module(publish_script)


def _fixture():
    policy = yaml.safe_load(DRAFT_PATH.read_text(encoding="utf-8"))
    raw = next(
        item
        for item in policy["reviewed_terminal_readiness_exceptions"]
        if item["symbol"] == "WIN"
    )
    spec = copy.deepcopy(raw)

    action = {
        "event_id": spec["event_id"],
        "security_id": spec["security_id"],
        "action_type": "delisting",
        "effective_date": spec["action_date"],
        "ex_date": spec["action_date"],
        "announcement_date": "2020-09-22",
        "record_date": "",
        "payment_date": "",
        "cash_amount": 0.0,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": True,
        "source_url": spec["action_source_url"],
        "source_kind": spec["action_source_kind"],
        "source": spec["action_source"],
        "retrieved_at": "2026-07-18T10:30:43.912238Z",
        "source_hash": spec["action_source_hash"],
        "metadata": None,
    }
    resolution = {
        "candidate_id": spec["resolution_candidate_id"],
        "security_id": spec["security_id"],
        "symbol": spec["symbol"],
        "last_price_date": spec["last_price_session"],
        "resolution": "applied",
        "event_id": spec["event_id"],
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": "us_lifecycle_finalizer_v1",
        "reviewed_at": "2026-07-18T00:00:00Z",
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        "source_url": spec["resolution_source_url"],
        "source": "lifecycle_finalizer",
        "retrieved_at": "2026-07-18T10:30:43.912238Z",
        "source_hash": spec["resolution_source_hash"],
    }
    removal_spec = spec["index_removals"][0]
    removal = {
        "event_id": removal_spec["event_id"],
        "index_id": removal_spec["index_id"],
        "announcement_date": "",
        "effective_date": removal_spec["effective_date"],
        "operation": "REMOVE",
        "security_id": spec["security_id"],
        "official": False,
        "source": removal_spec["source"],
        "source_url": removal_spec["source_url"],
        "source_kind": removal_spec["source_kind"],
        "retrieved_at": "2026-07-16T15:56:14.474469Z",
        "source_hash": removal_spec["source_hash"],
    }
    add = {
        **removal,
        "event_id": "a" * 64,
        "effective_date": "2015-01-07",
        "operation": "ADD",
    }
    spec["action_row_sha256"] = reviewed.release_row_sha256(action)
    spec["resolution_row_sha256"] = reviewed.release_row_sha256(resolution)
    spec["index_removals"][0]["row_sha256"] = reviewed.release_row_sha256(removal)
    spec = reviewed.canonical_reviewed_terminal_readiness_exception(spec)

    issue = TerminalTransitionIssue(
        code=spec["issue_code"],
        message="delayed",
        security_id=spec["security_id"],
        symbol=spec["symbol"],
        event_id=spec["event_id"],
        action_type=spec["action_type"],
        last_price_session=spec["last_price_session"],
        expected_transition_session=spec["expected_transition_session"],
        engine_session=spec["engine_session"],
        action_date_field=spec["action_date_field"],
        action_date=spec["action_date"],
    )
    report = TerminalTransitionReport(
        release_version="release-v1",
        applied_resolution_count=1,
        terminal_transition_count=1,
        issues=(issue,),
    )
    frames = {
        "corporate_actions": pd.DataFrame([action]),
        "lifecycle_resolutions": pd.DataFrame([resolution]),
        "index_membership_events": pd.DataFrame([add, removal]),
        "index_constituent_anchors": pd.DataFrame(
            [
                {
                    "index_id": "sp500",
                    "anchor_date": "2015-01-07",
                    "security_id": spec["security_id"],
                }
            ]
        ),
        "daily_price_raw": pd.DataFrame(
            [
                {
                    "security_id": spec["security_id"],
                    "session": spec["last_price_session"],
                    "close": 0.07,
                }
            ]
        ),
        "source_archive": pd.DataFrame(
            [
                {
                    "archive_id": spec["action_source_hash"],
                    "source_hash": spec["action_source_hash"],
                    "source_url": spec["action_source_url"],
                },
                {
                    "archive_id": removal_spec["source_hash"],
                    "source_hash": removal_spec["source_hash"],
                    "source_url": removal_spec["source_url"],
                },
            ]
        ),
    }

    class Repository:
        def read_frame(self, dataset, version):
            if version != f"{dataset}-v1":  # pragma: no cover
                raise AssertionError((dataset, version))
            return frames[dataset]

    release = SimpleNamespace(
        version="release-v1",
        dataset_versions={dataset: f"{dataset}-v1" for dataset in frames},
    )
    return Repository(), release, report, spec, frames


class ReviewedTerminalReadinessExceptionTest(unittest.TestCase):
    def test_complete_draft_inventory_is_code_pinned_and_excludes_sivb_avp(self):
        policy = yaml.safe_load(DRAFT_PATH.read_text(encoding="utf-8"))
        registry = reviewed.reviewed_terminal_readiness_exceptions(policy)
        self.assertEqual(
            reviewed.reviewed_terminal_readiness_exception_inventory_sha256(policy),
            reviewed.TRUSTED_REVIEWED_TERMINAL_READINESS_EXCEPTIONS_SHA256,
        )
        self.assertEqual(
            set(registry),
            set(reviewed.TRUSTED_REVIEWED_TERMINAL_READINESS_EXCEPTION_EVENT_IDS),
        )
        self.assertEqual(
            {item["symbol"] for item in registry.values()},
            {"WIN", "CHK", "FTR", "BBBY", "ENDP", "NTCOY"},
        )
        self.assertNotIn("SIVB", {item["symbol"] for item in registry.values()})
        self.assertNotIn("AVP", {item["symbol"] for item in registry.values()})
        self.assertTrue(
            all(
                removal["official"] is False
                for item in registry.values()
                for removal in item["index_removals"]
            )
        )

    def test_exact_exception_is_reported_as_degraded_not_erased(self):
        repository, release, report, spec, _frames = _fixture()
        with patch.object(
            reviewed,
            "load_code_pinned_reviewed_terminal_readiness_exceptions",
            return_value={spec["event_id"]: spec},
        ):
            result = reviewed.validate_publication_terminal_readiness_exceptions(
                repository, release, report
            )
        self.assertTrue(result["ready"])
        self.assertFalse(result["raw_ready"])
        self.assertEqual(result["raw_issue_count"], 1)
        self.assertEqual(result["issue_count"], 0)
        self.assertEqual(result["reviewed_exception_count"], 1)
        self.assertTrue(result["quality_degraded"])
        self.assertEqual(result["quality"], "degraded")
        self.assertEqual(len(result["release_warnings"]), 1)
        self.assertEqual(result["reviewed_exceptions"][0]["symbol"], "WIN")

    def test_action_projection_ignores_only_non_economic_metadata(self):
        repository, release, report, spec, frames = _fixture()
        baseline = reviewed.reviewed_terminal_action_projection_sha256(
            frames["corporate_actions"].iloc[0].to_dict()
        )
        resolution_baseline = (
            reviewed.reviewed_terminal_resolution_projection_sha256(
                frames["lifecycle_resolutions"].iloc[0].to_dict()
            )
        )
        removal_baseline = reviewed.reviewed_terminal_removal_projection_sha256(
            frames["index_membership_events"].iloc[1].to_dict()
        )
        frames["corporate_actions"].loc[0, "retrieved_at"] = (
            "2099-01-01T00:00:00Z"
        )
        frames["corporate_actions"].loc[0, "metadata"] = "schema-extension"
        frames["lifecycle_resolutions"].loc[0, "retrieved_at"] = (
            "2099-01-01T00:00:00Z"
        )
        frames["lifecycle_resolutions"].loc[0, "reviewed_at"] = (
            "2099-01-01T00:00:01Z"
        )
        frames["lifecycle_resolutions"].loc[0, "metadata"] = "schema-extension"
        frames["index_membership_events"].loc[1, "retrieved_at"] = (
            "2099-01-01T00:00:00Z"
        )
        frames["index_membership_events"].loc[1, "metadata"] = "schema-extension"
        self.assertEqual(
            reviewed.reviewed_terminal_action_projection_sha256(
                frames["corporate_actions"].iloc[0].to_dict()
            ),
            baseline,
        )
        self.assertEqual(
            reviewed.reviewed_terminal_resolution_projection_sha256(
                frames["lifecycle_resolutions"].iloc[0].to_dict()
            ),
            resolution_baseline,
        )
        self.assertEqual(
            reviewed.reviewed_terminal_removal_projection_sha256(
                frames["index_membership_events"].iloc[1].to_dict()
            ),
            removal_baseline,
        )
        with patch.object(
            reviewed,
            "load_code_pinned_reviewed_terminal_readiness_exceptions",
            return_value={spec["event_id"]: spec},
        ):
            result = reviewed.validate_publication_terminal_readiness_exceptions(
                repository, release, report
            )
        self.assertTrue(result["ready"])

        frames["corporate_actions"].loc[0, "announcement_date"] = "2099-01-02"
        with (
            patch.object(
                reviewed,
                "load_code_pinned_reviewed_terminal_readiness_exceptions",
                return_value={spec["event_id"]: spec},
            ),
            self.assertRaisesRegex(RuntimeError, "action row drifted"),
        ):
            reviewed.validate_publication_terminal_readiness_exceptions(
                repository, release, report
            )

    def test_unused_reviewed_policy_does_not_block_a_later_exact_repair(self):
        repository, release, _report, spec, _frames = _fixture()
        clean = TerminalTransitionReport(
            release_version="release-v1",
            applied_resolution_count=1,
            terminal_transition_count=1,
            issues=(),
        )
        with patch.object(
            reviewed,
            "load_code_pinned_reviewed_terminal_readiness_exceptions",
            return_value={spec["event_id"]: spec},
        ):
            result = reviewed.validate_publication_terminal_readiness_exceptions(
                repository, release, clean
            )
        self.assertTrue(result["ready"])
        self.assertEqual(result["reviewed_exception_count"], 0)
        self.assertFalse(result["quality_degraded"])
        self.assertEqual(result["quality"], "validated")

    def test_release_and_policy_mutations_fail_closed(self):
        variants = (
            "action",
            "resolution",
            "removal",
            "later_add",
            "later_anchor",
            "later_price",
            "archive",
            "issue",
        )
        for variant in variants:
            with self.subTest(variant=variant):
                repository, release, report, spec, frames = _fixture()
                if variant == "action":
                    frames["corporate_actions"].loc[0, "cash_amount"] = 1.0
                elif variant == "resolution":
                    frames["lifecycle_resolutions"].loc[0, "resolution"] = "exception"
                elif variant == "removal":
                    frames["index_membership_events"].loc[1, "source_hash"] = "f" * 64
                elif variant == "later_add":
                    later = frames["index_membership_events"].iloc[0].copy()
                    later["event_id"] = "b" * 64
                    later["effective_date"] = "2019-01-02"
                    later["operation"] = "ADD"
                    frames["index_membership_events"] = pd.concat(
                        [frames["index_membership_events"], later.to_frame().T],
                        ignore_index=True,
                    )
                elif variant == "later_anchor":
                    later = frames["index_constituent_anchors"].iloc[0].copy()
                    later["anchor_date"] = "2019-01-02"
                    frames["index_constituent_anchors"] = pd.concat(
                        [frames["index_constituent_anchors"], later.to_frame().T],
                        ignore_index=True,
                    )
                elif variant == "later_price":
                    later = frames["daily_price_raw"].iloc[0].copy()
                    later["session"] = "2020-07-13"
                    frames["daily_price_raw"] = pd.concat(
                        [frames["daily_price_raw"], later.to_frame().T],
                        ignore_index=True,
                    )
                elif variant == "archive":
                    frames["source_archive"] = frames["source_archive"].iloc[1:].copy()
                else:
                    changed = copy.deepcopy(report.issues[0])
                    changed = TerminalTransitionIssue(
                        **{
                            **changed.__dict__,
                            "engine_session": "2020-09-22",
                        }
                    )
                    report = TerminalTransitionReport(
                        release_version=report.release_version,
                        applied_resolution_count=1,
                        terminal_transition_count=1,
                        issues=(changed,),
                    )
                with (
                    patch.object(
                        reviewed,
                        "load_code_pinned_reviewed_terminal_readiness_exceptions",
                        return_value={spec["event_id"]: spec},
                    ),
                    self.assertRaises(RuntimeError),
                ):
                    reviewed.validate_publication_terminal_readiness_exceptions(
                        repository, release, report
                    )

        policy = yaml.safe_load(DRAFT_PATH.read_text(encoding="utf-8"))
        policy["reviewed_terminal_readiness_exceptions"][0][
            "required_release_warning"
        ] += " changed"
        with (
            patch.object(yaml, "safe_load", return_value=policy),
            self.assertRaisesRegex(RuntimeError, "not code-pinned"),
        ):
            reviewed.load_code_pinned_reviewed_terminal_readiness_exceptions(
                DRAFT_PATH
            )

    def test_publisher_uses_reviewed_gate_and_keeps_generic_audit_strict(self):
        repository, release, report, _spec, _frames = _fixture()
        accepted = {
            "ready": True,
            "issue_count": 0,
            "issues": [],
            "reviewed_exception_count": 1,
            "quality": "degraded",
            "quality_degraded": True,
            "release_warnings": ["warning"],
        }
        with (
            patch.object(
                publish_script,
                "audit_release_terminal_transitions",
                return_value=report,
            ) as audit,
            patch.object(
                publish_script,
                "validate_publication_terminal_readiness_exceptions",
                return_value=accepted,
            ) as validate,
        ):
            result = publish_script._validate_terminal_transition_readiness(
                repository, release
            )
        self.assertIs(result, accepted)
        audit.assert_called_once_with(repository, release)
        validate.assert_called_once_with(repository, release, report)
        with self.assertRaisesRegex(
            RuntimeError, "Terminal-transition readiness is blocked"
        ):
            with (
                patch.object(
                    publish_script,
                    "audit_release_terminal_transitions",
                    return_value=report,
                ),
                patch.object(
                    publish_script,
                    "validate_publication_terminal_readiness_exceptions",
                    return_value={
                        **accepted,
                        "ready": False,
                        "issue_count": 1,
                        "issues": [report.issues[0].to_dict()],
                    },
                ),
            ):
                publish_script._validate_terminal_transition_readiness(
                    repository, release
                )
        # The generic report itself is still fail-closed and was not changed.
        with self.assertRaisesRegex(RuntimeError, "readiness is blocked"):
            report.raise_for_errors()


if __name__ == "__main__":
    unittest.main()
