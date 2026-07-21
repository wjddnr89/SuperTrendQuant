from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts/audit_us_no_data_date_bindings.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_us_no_data_date_bindings_for_test", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


class NoDataDateBindingAuditTest(unittest.TestCase):
    def test_finite_inventory_and_repair_classification_are_pinned(self):
        by_symbol = {case.symbol: case for case in script.CASES}
        self.assertEqual(len(script.CASES), 18)
        self.assertEqual(len(by_symbol), 18)
        self.assertEqual(
            set(by_symbol),
            {
                "SIVB",
                "BMYRT",
                "WIN",
                "FLT",
                "CDAY",
                "CHK",
                "FTR",
                "XEC",
                "TFCFA",
                "TFCF",
                "HCP",
                "UTX",
                "VAL",
                "COG",
                "CTRP",
                "SYMC",
                "ENDP",
                "ARNC",
            },
        )
        self.assertEqual(
            Counter(case.disposition for case in script.CASES),
            {
                "accepted_exact_reviewed_exception": 4,
                "dataset_repair_required": 13,
                "blocked_independent_successor_price": 1,
            },
        )
        self.assertEqual(
            Counter(case.repair_scope for case in script.CASES),
            {
                "identity_interval_and_old_sid_tail": 8,
                "bankruptcy_otc_or_market_exit_gap": 4,
                "missing_terminal_action_only": 1,
                "none": 4,
                "policy_only_blocked": 1,
            },
        )
        self.assertEqual(
            {
                case.symbol
                for case in script.CASES
                if case.disposition == "accepted_exact_reviewed_exception"
            },
            {"BMYRT", "TFCFA", "TFCF", "VAL"},
        )
        self.assertTrue(
            all(
                len(case.target_id) == 64
                and set(case.target_id) <= set("0123456789abcdef")
                for case in script.CASES
            )
        )

    def test_market_session_relation_is_not_a_generic_date_tolerance(self):
        self.assertEqual(
            script._xnys_relation("2019-11-21", "2019-11-21"),
            ("event_on_terminal_session", 0),
        )
        relation, sessions = script._xnys_relation(
            "2023-03-09", "2023-03-28"
        )
        self.assertEqual(relation, "event_after_multi_session_gap")
        self.assertEqual(sessions, 13)
        relation, sessions = script._xnys_relation(
            "2024-05-24", "2024-03-25"
        )
        self.assertEqual(relation, "event_precedes_terminal_price")
        self.assertLess(sessions, 0)


if __name__ == "__main__":
    unittest.main()
