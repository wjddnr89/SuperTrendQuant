from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from supertrend_quant.market_store.terminal_policy_exceptions import (
    TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTION_EVENT_IDS,
    TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTIONS_SHA256,
    action_metadata_sha256,
    canonical_reviewed_terminal_policy_exception,
    reviewed_terminal_policy_action_mismatches,
    reviewed_terminal_policy_exception_inventory_sha256,
    reviewed_terminal_policy_release_warning_mismatches,
    reviewed_terminal_policy_report_mismatches,
)


REPORT_HASH = "109813ac49e2be1b05cce11b9f042ae0cf5f23d6c317006ed1a168007768b4f9"
DRAFT_POLICY_PATH = (
    Path(__file__).parents[1]
    / "configs/drafts/us_terminal_policy_exceptions_review.yaml"
)
PRODUCTION_POLICY_PATH = (
    Path(__file__).parents[1] / "configs/us_cross_validation.yaml"
)


CASES = {
    "ABMD": {
        "event_id": "350bb85a7395ef9272e5f2867afdd4e523c99c258752120977ec1f35e36a2c8a",
        "candidate_id": "76a94b17cf5b10aed59bf7e924b1df483e3fcab9a628db9b220d8b22b269278c",
        "security_id": "US:EODHD:faece1b7-4b1a-5c1f-951b-b1178ed57161",
        "action_type": "cash_merger",
        "effective_date": "2022-12-22",
        "last_price_date": "2022-12-21",
        "report_candidate_active_to": "2022-12-21",
        "announcement_date": "2022-11-01",
        "payment_date": "2022-12-22",
        "new_security_id": "",
        "new_symbol": "",
        "ratio": None,
        "cash_amount": 380.0,
        "report_effective_date": "2022-12-22",
        "report_cash_amount": 35.0,
        "source_kind": "official_lower_bound_policy",
        "source_url": "https://www.sec.gov/Archives/edgar/data/815094/000119312522311074/0001193125-22-311074.txt",
        "source_hash": "f98bc807432739e4f2447ffbc6a70f7651bd8982b901989fc31dcffaa56ec593",
        "report_source_url": "https://www.sec.gov/Archives/edgar/data/200406/000119312522311072/0001193125-22-311072.txt",
        "report_source_hash": "2ee8f9067eb842d175b082b80f143cfa77d99a4df037310818b3d441a5a2bdcb",
        "manual_reason": "Consideration included $380 cash plus a contingent value right worth up to $35, which the action schema cannot represent.",
        "cross": (False, True, False),
        "allowed": ["cash_amount", "source_hash", "source_url"],
        "policy_code": "abmd_nontradeable_cvr_lower_bound/v1",
        "warning": "ABMD exact lower-bound warning",
        "filing_accession": "0001193125-22-311072",
        "filing_date": "2022-12-22",
    },
    "CELG": {
        "event_id": "cb355a88e767bd5f557350ddf9c13f1b324da6e8f96c622a2e1f8eeea01fa36a",
        "candidate_id": "dcaad5c3a8565d37f4d045db599f64c56e21a4cd7897aa6fb574e69a023120f1",
        "security_id": "US:EODHD:0337dd23-67ad-5354-b972-50babd1ae5a0",
        "action_type": "stock_merger",
        "effective_date": "2019-11-21",
        "last_price_date": "2019-11-20",
        "report_candidate_active_to": "2019-11-20",
        "announcement_date": "2019-11-20",
        "payment_date": "2019-11-21",
        "new_security_id": "US:EODHD:25d16784-a5a9-5eee-bf6e-81519b64ef0b",
        "new_symbol": "BMY",
        "ratio": 1.0,
        "cash_amount": 50.0,
        "report_effective_date": "2019-11-20",
        "report_cash_amount": 50.0,
        "source_kind": "official_crosscheck",
        "source_url": "https://www.sec.gov/Archives/edgar/data/14272/000114036119021048/0001140361-19-021048.txt",
        "source_hash": "157cae6dae5486f16c63a51e61d79aab2ce2f37d0e8584337fb21d7d0ec6f211",
        "report_source_url": "https://www.sec.gov/Archives/edgar/data/14272/000114036119021048/0001140361-19-021048.txt",
        "report_source_hash": "157cae6dae5486f16c63a51e61d79aab2ce2f37d0e8584337fb21d7d0ec6f211",
        "manual_reason": "Consideration included a separately tradable contingent value right not represented by the action schema.",
        "cross": (True, True, True),
        "allowed": ["effective_date"],
        "policy_code": "celg_next_session_cvr_delivery/v1",
        "warning": "CELG exact unsupported-path warning",
        "filing_accession": "0001140361-19-021048",
        "filing_date": "2019-11-20",
    },
    "PARA": {
        "event_id": "f553d393e8bda37561276fec20d5b9bce5f722609e466e96bc9e199c624891c1",
        "candidate_id": "22590705691d87ce58407ff1748c94c2ede7747b095b2925e34d87dd36327dee",
        "security_id": "US:EODHD:f60b749b-3d84-552a-9dc9-39e742f67537",
        "action_type": "stock_merger",
        "effective_date": "2025-08-07",
        "last_price_date": "2025-08-06",
        "report_candidate_active_to": "2025-08-07",
        "announcement_date": "2025-08-07",
        "payment_date": "",
        "new_security_id": "US:EODHD:fe84848c-624b-5aba-b542-24af3959f97f",
        "new_symbol": "PSKY",
        "ratio": 1.0,
        "cash_amount": None,
        "report_effective_date": "2025-08-06",
        "report_cash_amount": 15.0,
        "source_kind": "sec_filing_default_stock_policy",
        "source_url": "https://www.sec.gov/Archives/edgar/data/813828/000119312525175027/0001193125-25-175027.txt",
        "source_hash": "61ea922a72a55f05b79c2cf00e9c4b0367434c35bf25a78fd4d815d3b20e68be",
        "report_source_url": "https://www.sec.gov/Archives/edgar/data/813828/000119312525175027/0001193125-25-175027.txt",
        "report_source_hash": "61ea922a72a55f05b79c2cf00e9c4b0367434c35bf25a78fd4d815d3b20e68be",
        "manual_reason": "Shareholder election and proration made the $15 cash and one-share alternatives mutually exclusive, not additive.",
        "cross": (False, True, False),
        "allowed": ["cash_amount", "effective_date"],
        "policy_code": "para_no_election_default_stock/v1",
        "warning": "",
        "filing_accession": "0001193125-25-175027",
        "filing_date": "2025-08-07",
    },
}


def _fixture(symbol: str):
    case = CASES[symbol]
    metadata = {"reviewed_case": symbol, "policy_code": case["policy_code"]}
    spec = {
        "event_id": case["event_id"],
        "security_id": case["security_id"],
        "action_type": case["action_type"],
        "effective_date": case["effective_date"],
        "new_security_id": case["new_security_id"],
        "new_symbol": case["new_symbol"],
        "ratio": case["ratio"],
        "cash_amount": case["cash_amount"],
        "currency": "USD",
        "source_kind": case["source_kind"],
        "source_url": case["source_url"],
        "source_hash": case["source_hash"],
        "candidate_id": case["candidate_id"],
        "symbol": symbol,
        "last_price_date": case["last_price_date"],
        "ex_date": case["effective_date"],
        "announcement_date": case["announcement_date"],
        "payment_date": case["payment_date"],
        "action_metadata_sha256": action_metadata_sha256(metadata),
        "report_effective_date": case["report_effective_date"],
        "report_action_type": case["action_type"],
        "report_new_symbol": case["new_symbol"],
        "report_ratio": case["ratio"],
        "report_cash_amount": case["report_cash_amount"],
        "report_source_url": case["report_source_url"],
        "report_source_hash": case["report_source_hash"],
        "report_candidate_active_to": case["report_candidate_active_to"],
        "report_manual_review_reason": case["manual_reason"],
        "report_crosscheck_passed": case["cross"][0],
        "report_crosscheck_date_passed": case["cross"][1],
        "report_crosscheck_economic_terms_passed": case["cross"][2],
        "allowed_report_mismatches": case["allowed"],
        "policy_code": case["policy_code"],
        "required_release_warning": case["warning"],
        "lifecycle_evidence_report_sha256": REPORT_HASH,
        "filing_accession_number": case["filing_accession"],
        "filing_date": case["filing_date"],
    }
    action = {
        key: spec[key]
        for key in (
            "event_id",
            "security_id",
            "action_type",
            "effective_date",
            "new_security_id",
            "new_symbol",
            "ratio",
            "cash_amount",
            "currency",
            "source_kind",
            "source_url",
            "source_hash",
            "ex_date",
            "announcement_date",
            "payment_date",
        )
    }
    action.update({"official": True, "metadata": metadata})
    resolution = {
        "candidate_id": case["candidate_id"],
        "security_id": case["security_id"],
        "symbol": symbol,
        "last_price_date": case["last_price_date"],
        "event_id": case["event_id"],
        "successor_security_id": case["new_security_id"],
        "source_url": case["source_url"],
        "source_hash": case["source_hash"],
    }
    record = {
        "candidate": {
            "security_id": case["security_id"],
            "symbol": symbol,
            "last_price_date": case["last_price_date"],
            "active_to": case["report_candidate_active_to"],
        },
        "eligible_for_apply": False,
        "manual_review_reason": case["manual_reason"],
        "crosscheck": {
            "passed": case["cross"][0],
            "date_passed": case["cross"][1],
            "economic_terms_passed": case["cross"][2],
        },
        "parsed": {
            "action_type": case["action_type"],
            "effective_date": case["report_effective_date"],
            "new_symbol": case["new_symbol"],
            "ratio": case["ratio"],
            "cash_amount": case["report_cash_amount"],
        },
        "source_url": case["report_source_url"],
        "source_hash": case["report_source_hash"],
        "successor_security_id": case["new_security_id"],
        "filing": {
            "accession_number": case["filing_accession"],
            "filing_date": case["filing_date"],
        },
    }
    return spec, action, resolution, record


class TerminalPolicyExceptionTests(unittest.TestCase):
    def test_reviewed_draft_inventory_is_code_pinned(self):
        for path, section in (
            (DRAFT_POLICY_PATH, None),
            (PRODUCTION_POLICY_PATH, "events"),
        ):
            with self.subTest(path=path.name):
                policy = yaml.safe_load(path.read_text(encoding="utf-8"))
                events = policy if section is None else policy[section]
                self.assertEqual(
                    reviewed_terminal_policy_exception_inventory_sha256(events),
                    TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTIONS_SHA256,
                )

    def test_three_exact_policies_are_canonical_and_code_scoped(self):
        registry = []
        for symbol in CASES:
            with self.subTest(symbol=symbol):
                spec, action, resolution, record = _fixture(symbol)
                normalized = canonical_reviewed_terminal_policy_exception(spec)
                self.assertEqual(normalized["event_id"], spec["event_id"])
                self.assertEqual(
                    reviewed_terminal_policy_action_mismatches(action, spec), ()
                )
                self.assertEqual(
                    reviewed_terminal_policy_report_mismatches(
                        action, resolution, record, spec, REPORT_HASH
                    ),
                    (),
                )
                self.assertEqual(
                    reviewed_terminal_policy_release_warning_mismatches(
                        [case["warning"] for case in CASES.values() if case["warning"]],
                        spec,
                    ),
                    (),
                )
                registry.append(spec)
        self.assertEqual(
            {item["event_id"] for item in registry},
            set(TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTION_EVENT_IDS),
        )
        self.assertEqual(
            len(
                reviewed_terminal_policy_exception_inventory_sha256(
                    {"reviewed_terminal_policy_exceptions": registry}
                )
            ),
            64,
        )

    def test_action_report_and_warning_drift_fail_closed(self):
        for symbol in CASES:
            spec, action, resolution, record = _fixture(symbol)
            mutations = {
                "action_metadata": ("action", "metadata", {"changed": True}),
                "action_official": ("action", "official", False),
                "report_cash": ("parsed", "cash_amount", 999.0),
                "report_manual_reason": ("record", "manual_review_reason", "changed"),
                "report_successor": (
                    "record",
                    "successor_security_id",
                    "DIFFERENT",
                ),
                "report_candidate_active_to": (
                    "candidate",
                    "active_to",
                    "2000-01-03",
                ),
                "report_crosscheck": ("crosscheck", "passed", None),
                "filing_accession": ("filing", "accession_number", "changed"),
            }
            for label, (target, field, value) in mutations.items():
                with self.subTest(symbol=symbol, mutation=label):
                    changed_action = copy.deepcopy(action)
                    changed_record = copy.deepcopy(record)
                    if target == "action":
                        changed_action[field] = value
                    elif target == "parsed":
                        changed_record["parsed"][field] = value
                    elif target == "record":
                        changed_record[field] = value
                    else:
                        changed_record[target][field] = value
                    action_mismatches = reviewed_terminal_policy_action_mismatches(
                        changed_action, spec
                    )
                    report_mismatches = reviewed_terminal_policy_report_mismatches(
                        changed_action,
                        resolution,
                        changed_record,
                        spec,
                        REPORT_HASH,
                    )
                    self.assertTrue(action_mismatches or report_mismatches)

            wrong_hash = reviewed_terminal_policy_report_mismatches(
                action, resolution, record, spec, "f" * 64
            )
            self.assertIn("lifecycle_evidence_report_sha256", wrong_hash)
            if spec["required_release_warning"]:
                self.assertEqual(
                    reviewed_terminal_policy_release_warning_mismatches([], spec),
                    ("required_release_warning",),
                )

    def test_frc_self_authored_occ_extraction_is_not_an_approved_exception(self):
        frc_event_id = (
            "e351f774b133eae45d49e0fbe60215e5bbceec540c3386076f4c3f2b6c57d9ea"
        )
        self.assertNotIn(
            frc_event_id, TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTION_EVENT_IDS
        )
        spec, _, _, _ = _fixture("PARA")
        spec["policy_code"] = "frc_self_authored_occ_extraction/v1"
        with self.assertRaisesRegex(RuntimeError, "policy_code"):
            canonical_reviewed_terminal_policy_exception(spec)


if __name__ == "__main__":
    unittest.main()
