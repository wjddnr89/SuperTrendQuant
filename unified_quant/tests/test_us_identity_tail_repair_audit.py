from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts/audit_us_identity_tail_repairs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_us_identity_tail_repairs_for_test", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


class IdentityTailRepairAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = script.build_audit(
            script.LocalDatasetRepository(Path("data/cache"))
        )

    def test_finite_inventory_and_hash_pins_are_exact(self):
        self.assertEqual(len(script.CASES), 8)
        self.assertEqual(
            [case.symbol for case in script.CASES],
            ["FLT", "CDAY", "XEC", "HCP", "UTX", "COG", "CTRP", "SYMC"],
        )
        self.assertEqual(sum(case.tail_rows for case in script.CASES), 853)
        self.assertEqual(
            Counter(case.repair_class for case in script.CASES),
            {
                "delete_old_tail_close_identity": 5,
                "delete_synthetic_merger_tail_close_identity": 1,
                "replace_successor_first_session_then_delete_old_tail": 1,
                "canonicalize_nlok_identity_then_retire_old_sid": 1,
            },
        )
        for case in script.CASES:
            for digest in (
                case.old_tail_source_hash,
                case.successor_overlap_source_hash,
                case.official_source_hash,
                case.old_tail_sha256,
                case.successor_overlap_sha256,
                case.action_row_sha256,
                case.membership_inventory_sha256,
                case.identity_inventory_sha256,
            ):
                self.assertEqual(len(digest), 64)
                self.assertLessEqual(set(digest), set("0123456789abcdef"))

    def test_actual_release_audit_is_stable_and_offline(self):
        audit = self.audit
        self.assertEqual(audit["release_version"], script.PINNED_RELEASE_VERSION)
        self.assertEqual(
            audit["summary"],
            {
                "case_count": 8,
                "old_sid_tail_rows": 853,
                "delete_close_cases": 5,
                "synthetic_merger_tail_cases": 1,
                "successor_first_row_replacement_cases": 1,
                "canonical_identity_cases": 1,
                "direct_index_member_tail_session_count": 1,
                "direct_index_member_tail_symbols": ["SYMC"],
            },
        )
        self.assertFalse(audit["network_accessed"])
        self.assertEqual(audit["http_attempts"], 0)
        self.assertEqual(audit["eodhd_calls"], 0)
        self.assertFalse(audit["r2_accessed"])
        self.assertFalse(audit["dataset_writes_performed"])
        self.assertFalse(audit["release_pointer_writes_performed"])
        self.assertFalse(audit["generic_tolerance_added"])
        self.assertEqual(
            script.sha256_bytes(script._canonical_json_bytes(audit)),
            "46c2acd305a92c99560b88b7dae06df7d7c7a85aa99bbe97887a5b58019cd089",
        )

    def test_case_specific_repairs_do_not_apply_a_generic_reassignment(self):
        cases = {case["symbol"]: case for case in self.audit["cases"]}
        for symbol in {"FLT", "CDAY", "UTX", "COG", "CTRP"}:
            repair = cases[symbol]["safe_minimum_repair"]
            self.assertEqual(repair["repair_class"], "delete_old_tail_close_identity")
            self.assertFalse(repair["reassign_all_tail_rows"])
            self.assertEqual(
                cases[symbol]["index_membership"][
                    "old_sid_member_tail_session_count"
                ],
                0,
            )

        xec = cases["XEC"]
        self.assertEqual(xec["action_type"], "stock_merger")
        self.assertEqual(xec["old_tail"]["rows"], 2)
        self.assertFalse(
            xec["hypothetical_full_tail_reassignment_signal_impact"]["evaluated"]
        )

        hcp = cases["HCP"]["safe_minimum_repair"]
        self.assertEqual(hcp["replace_successor_sessions_from_old"], ["2019-11-05"])
        self.assertEqual(hcp["replacement_rows"], 1)

        utx_impact = cases["UTX"][
            "hypothetical_full_tail_reassignment_signal_impact"
        ]["total_return_adjusted"]
        self.assertEqual(utx_impact["TripleBuySignal"]["count"], 2)
        self.assertEqual(utx_impact["TripleSellSignal"]["count"], 2)

    def test_symc_requires_canonical_identity_and_index_rebind(self):
        symc = next(case for case in self.audit["cases"] if case["symbol"] == "SYMC")
        self.assertEqual(
            symc["index_membership"]["old_sid_member_tail_sessions"],
            {"sp500": ["2019-11-04"]},
        )
        repair = symc["safe_minimum_repair"]
        self.assertTrue(repair["retire_entire_old_sid"])
        self.assertTrue(
            repair["rebind_sp500_anchor_and_remove_redundant_2019_11_05_swap"]
        )
        diagnostic = self.audit["symc_full_identity_diagnostic"]
        self.assertEqual(diagnostic["old_sid_rows"], 1455)
        self.assertEqual(diagnostic["old_rows_covered_by_successor"], 1455)
        for mode in ("raw", "total_return_adjusted"):
            self.assertEqual(
                diagnostic["pre_transition_triple_supertrend_diff"][mode][
                    "TripleBuySignal"
                ]["count"],
                0,
            )
            self.assertEqual(
                diagnostic["pre_transition_triple_supertrend_diff"][mode][
                    "TripleSellSignal"
                ]["count"],
                0,
            )

    def test_source_contains_no_dataset_or_external_write_path(self):
        source = SCRIPT_PATH.read_text()
        for forbidden in (
            ".write_frame(",
            ".commit_release(",
            "requests.",
            "httpx.",
            "boto3",
            "EODHD_API_KEY",
            "R2_ACCESS_KEY",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
