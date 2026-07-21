from __future__ import annotations

import base64
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.schemas import dataset_spec


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_frc_para_lifecycle.py"
)
SPEC = importlib.util.spec_from_file_location("repair_us_frc_para_lifecycle", SCRIPT_PATH)
assert SPEC and SPEC.loader
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _frame(dataset: str, rows: list[dict] | None = None) -> pd.DataFrame:
    columns = list(dataset_spec(dataset).required_columns)
    extras = {
        "security_master": [
            "source_url",
            "provider_symbol",
            "action_provider_symbol",
            "isin",
        ],
        "symbol_history": ["source_url"],
        "daily_price_raw": ["source_url"],
        "corporate_actions": ["metadata"],
        "source_archive": ["source_url"],
    }.get(dataset, [])
    columns = list(dict.fromkeys((*columns, *extras)))
    return pd.DataFrame(rows or [], columns=columns)


def _source_defaults() -> dict[str, str]:
    return {
        "source": "fixture",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "f" * 64,
        "source_url": "https://example.test/source",
    }


def _master_row(
    security_id: str,
    symbol: str,
    *,
    active_from: str,
    active_to: str,
) -> dict:
    return {
        "security_id": security_id,
        "primary_symbol": symbol,
        "name": symbol,
        "exchange": "NASDAQ" if symbol in {"PARA", "PSKY"} else "NYSE",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": active_from,
        "active_to": active_to,
        "provider_symbol": f"{symbol}.US",
        "action_provider_symbol": f"{symbol}.US",
        "isin": "",
        **_source_defaults(),
    }


def _history_row(
    security_id: str,
    symbol: str,
    effective_from: str,
    effective_to: str = "",
) -> dict:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "exchange": "NASDAQ" if symbol in {"PARA", "PSKY"} else "NYSE",
        "effective_from": effective_from,
        "effective_to": effective_to,
        **_source_defaults(),
    }


def _price_row(security_id: str, session: str, close: float) -> dict:
    return {
        "security_id": security_id,
        "session": session,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1000,
        "currency": "USD",
        **_source_defaults(),
    }


def _resolution_row(
    candidate_id: str,
    security_id: str,
    symbol: str,
    last_price_date: str,
    code: str,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "security_id": security_id,
        "symbol": symbol,
        "last_price_date": last_price_date,
        "resolution": "exception",
        "event_id": "",
        "exception_code": code,
        "exception_reason": "fixture exception",
        "reviewed_by": "fixture",
        "reviewed_at": "2026-07-18T00:00:00Z",
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        **_source_defaults(),
    }


def _existing() -> dict[str, pd.DataFrame]:
    master = _frame(
        "security_master",
        [
            _master_row(
                script.FRC_SECURITY_ID,
                "FRC",
                active_from="2015-01-02",
                active_to=script.FRC_OLD_LAST,
            ),
            _master_row(
                script.PARA_SECURITY_ID,
                "PARA",
                active_from="2015-01-02",
                active_to=script.PARA_LAST,
            ),
            _master_row(
                script.PSKY_SECURITY_ID,
                "PSKY",
                active_from="2015-01-01",
                active_to="",
            ),
        ],
    )
    history = _frame(
        "symbol_history",
        [
            _history_row(script.FRC_SECURITY_ID, "FRC", "2015-01-01"),
            _history_row(script.PARA_SECURITY_ID, "PARA", "2015-01-01"),
            _history_row(script.PSKY_SECURITY_ID, "PSKY", "2015-01-01"),
        ],
    )
    prices = _frame(
        "daily_price_raw",
        [
            _price_row(script.FRC_SECURITY_ID, script.FRC_OLD_LAST, 3.51),
            _price_row(script.PARA_SECURITY_ID, script.PARA_LAST, 11.04),
            _price_row(script.PSKY_SECURITY_ID, script.PARA_LAST, 11.04),
            _price_row(script.PSKY_SECURITY_ID, script.PARA_TRANSITION, 11.74),
            _price_row(script.PSKY_SECURITY_ID, "2025-08-08", 10.51),
            _price_row(script.PSKY_SECURITY_ID, "2026-07-15", 9.50),
        ],
    )
    actions = _frame("corporate_actions")
    factors = build_adjustment_factors(
        prices,
        actions,
        source_version="fixture",
    )
    resolutions = _frame(
        "lifecycle_resolutions",
        [
            _resolution_row(
                "frc-candidate",
                script.FRC_SECURITY_ID,
                "FRC",
                script.FRC_OLD_LAST,
                "recovery_uncertain",
            ),
            _resolution_row(
                "para-candidate",
                script.PARA_SECURITY_ID,
                "PARA",
                script.PARA_LAST,
                "unsupported_consideration",
            ),
        ],
    )
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "adjustment_factors": factors,
        "source_archive": _frame("source_archive"),
    }


def _bundle() -> script.ProviderBundle:
    retrieved_at = "2026-07-18T00:00:00Z"
    rows = [
        {
            "date": script.FRC_TRANSITION,
            "open": 0.38,
            "high": 0.44,
            "low": 0.30,
            "close": 0.33,
            "volume": 1000000,
        },
        {
            "date": script.FRC_INDEX_EXIT,
            "open": 0.35,
            "high": 0.40,
            "low": 0.25,
            "close": 0.30,
            "volume": 900000,
        },
        {
            "date": "2026-06-01",
            "open": 0.001,
            "high": 0.0012,
            "low": 0.0008,
            "close": 0.001,
            "volume": 50000,
        },
    ]
    artifacts = tuple(
        SourceArtifact(
            source=f"eodhd_{endpoint}",
            source_url=script.REQUEST_URLS[endpoint],
            retrieved_at=retrieved_at,
            content=script._canonical_json_bytes(rows if endpoint == "eod" else []),
            content_type="application/json",
        )
        for endpoint in script.EODHD_ENDPOINTS
    )
    return script.bundle_from_artifacts(
        artifacts,
        original_http_attempts=3,
        budget_used_before=8828,
        budget_used_after=8831,
    )


def _invalid_artifacts() -> tuple[SourceArtifact, ...]:
    rows = [
        {
            "date": script.FRC_TRANSITION,
            "open": 0.38,
            "high": 0.44,
            "low": 0.30,
            "close": 0.33,
            "volume": 1000000,
        },
        {
            "date": script.FRC_INDEX_EXIT,
            "open": 0.35,
            "high": 0.40,
            "low": 0.25,
            "close": 0.30,
            "volume": 900000,
        },
        {
            "date": "2023-12-11",
            "open": 0.01055,
            "high": 0.012,
            "low": 0.0111,
            "close": 0.0115,
            "volume": 12345,
        },
        {
            "date": "2026-06-01",
            "open": 0.001,
            "high": 0.0012,
            "low": 0.0008,
            "close": 0.001,
            "volume": 50000,
        },
    ]
    raw = {
        "eod": json.dumps(rows, indent=1).encode() + b"\n",
        "div": b"[]\n",
        "splits": b"[ ]\n",
    }
    return tuple(
        SourceArtifact(
            source=f"eodhd_{endpoint}",
            source_url=script.REQUEST_URLS[endpoint],
            retrieved_at="2026-07-18T09:00:00Z",
            content=raw[endpoint],
            content_type="application/json; charset=utf-8",
        )
        for endpoint in script.EODHD_ENDPOINTS
    )


class _FakeBudget:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.seed_used = 8835
        self.period = "2026-07-18"
        self.limit = 100000
        self.reserve = 5000
        self.used = 8835

    @property
    def ceiling(self) -> int:
        return self.limit - self.reserve

    def claim(self) -> int:
        self.used += 1
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"period": self.period, "used": self.used}),
            encoding="utf-8",
        )
        return self.used


class _FakeRawClient:
    payloads: dict[str, bytes] = {}

    def __init__(self, *, budget):
        self.budget = budget
        self.attempted_endpoints: list[str] = []

    def get_raw_artifact(self, endpoint, *, params, retrieved_at):
        normalized = endpoint.strip("/")
        short = normalized.split("/", 1)[0]
        self.budget.claim()
        self.attempted_endpoints.append(normalized)
        return SourceArtifact(
            source=f"eodhd_{short}",
            source_url=script.REQUEST_URLS[short],
            retrieved_at=retrieved_at,
            content=self.payloads[short],
            content_type="application/json; charset=utf-8",
        )


class RepairUsFrcParaLifecycleTests(unittest.TestCase):
    def test_requirements_plan_is_exactly_three_frcb_calls_and_zero_psky_calls(self):
        plan = script.requirements_plan()

        self.assertEqual(plan["eodhd_total_calls"], 3)
        self.assertEqual(plan["psky_eodhd_calls"], 0)
        self.assertEqual(
            [(item["endpoint"], item["symbol"]) for item in plan["eodhd_requests"]],
            [("eod", "FRCB.US"), ("div", "FRCB.US"), ("splits", "FRCB.US")],
        )
        self.assertFalse(plan["network_accessed"])
        self.assertFalse(plan["r2_accessed"])

    def test_prepare_repairs_same_frc_identity_and_no_election_para_successor(self):
        frames, summary, artifacts = script.prepare_frames(
            _existing(), _bundle(), completed_session="2026-07-15"
        )

        self.assertTrue(script.identity_is_repaired(frames))
        self.assertEqual(summary["frc_policy"], "same_security_ticker_change_then_exit_only")
        self.assertEqual(summary["para_policy"], "no_election_default_stock_one_for_one")
        self.assertFalse(summary["frc_final_recovery_modeled"])
        self.assertFalse(summary["para_cash_elector_proration_modeled"])
        self.assertEqual(len(artifacts), 4)

        frc = frames["security_master"].loc[
            frames["security_master"]["security_id"].eq(script.FRC_SECURITY_ID)
        ].iloc[0]
        self.assertEqual(frc["primary_symbol"], "FRCB")
        self.assertEqual(frc["provider_symbol"], "FRCB.US")
        self.assertEqual(frc["active_to"], "")

        para = frames["security_master"].loc[
            frames["security_master"]["security_id"].eq(script.PARA_SECURITY_ID)
        ].iloc[0]
        psky = frames["security_master"].loc[
            frames["security_master"]["security_id"].eq(script.PSKY_SECURITY_ID)
        ].iloc[0]
        self.assertEqual(para["active_to"], script.PARA_TRANSITION)
        self.assertEqual(psky["active_from"], script.PARA_TRANSITION)

        history = frames["symbol_history"].set_index("security_id")
        self.assertEqual(
            history.loc[script.PARA_SECURITY_ID, "effective_to"],
            script.PARA_TRANSITION,
        )
        self.assertEqual(
            history.loc[script.PSKY_SECURITY_ID, "effective_from"],
            script.PARA_TRANSITION,
        )

        actions = frames["corporate_actions"].set_index("event_id")
        frc_action = actions.loc[script.FRC_EVENT_ID]
        self.assertEqual(frc_action["action_type"], "ticker_change")
        self.assertEqual(frc_action["new_security_id"], script.FRC_SECURITY_ID)
        para_action = actions.loc[script.PARA_EVENT_ID]
        self.assertEqual(para_action["action_type"], "stock_merger")
        self.assertEqual(float(para_action["ratio"]), 1.0)
        self.assertEqual(para_action["new_security_id"], script.PSKY_SECURITY_ID)

    def test_psky_provider_predecessor_history_and_actions_are_removed(self):
        existing = _existing()
        old_action = {
            "event_id": "old-psky-dividend",
            "security_id": script.PSKY_SECURITY_ID,
            "action_type": "cash_dividend",
            "effective_date": "2024-06-17",
            "ex_date": "2024-06-17",
            "announcement_date": "",
            "record_date": "",
            "payment_date": "",
            "cash_amount": 0.05,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "official": False,
            "source_kind": "provider",
            "metadata": None,
            **_source_defaults(),
        }
        existing["corporate_actions"] = _frame("corporate_actions", [old_action])
        existing["adjustment_factors"] = build_adjustment_factors(
            existing["daily_price_raw"],
            existing["corporate_actions"],
            source_version="fixture",
        )

        frames, summary, _ = script.prepare_frames(
            existing, _bundle(), completed_session="2026-07-15"
        )

        psky_prices = frames["daily_price_raw"].loc[
            frames["daily_price_raw"]["security_id"].eq(script.PSKY_SECURITY_ID)
        ]
        self.assertEqual(psky_prices["session"].astype(str).min(), script.PARA_TRANSITION)
        self.assertNotIn("old-psky-dividend", set(frames["corporate_actions"]["event_id"]))
        self.assertEqual(summary["psky_pre_transition_price_rows_removed"], 1)
        self.assertEqual(summary["psky_pre_transition_action_rows_removed"], 1)

    def test_resolutions_change_from_permanent_exceptions_to_applied_events(self):
        frames, _, _ = script.prepare_frames(
            _existing(), _bundle(), completed_session="2026-07-15"
        )
        rows = frames["lifecycle_resolutions"].set_index("security_id")

        self.assertEqual(rows.loc[script.FRC_SECURITY_ID, "resolution"], "applied")
        self.assertEqual(rows.loc[script.FRC_SECURITY_ID, "event_id"], script.FRC_EVENT_ID)
        self.assertEqual(
            rows.loc[script.FRC_SECURITY_ID, "successor_security_id"],
            script.FRC_SECURITY_ID,
        )
        self.assertEqual(rows.loc[script.PARA_SECURITY_ID, "resolution"], "applied")
        self.assertEqual(rows.loc[script.PARA_SECURITY_ID, "event_id"], script.PARA_EVENT_ID)
        self.assertEqual(
            rows.loc[script.PARA_SECURITY_ID, "successor_security_id"],
            script.PSKY_SECURITY_ID,
        )
        self.assertEqual(rows.loc[script.FRC_SECURITY_ID, "exception_code"], "")
        self.assertEqual(rows.loc[script.PARA_SECURITY_ID, "exception_code"], "")

    def test_bundle_rejects_missing_index_exit_session(self):
        bundle = _bundle()
        broken = script.ProviderBundle(
            prices=bundle.prices.loc[
                bundle.prices["session"].astype(str).ne(script.FRC_INDEX_EXIT)
            ],
            corporate_actions=bundle.corporate_actions,
            artifacts=bundle.artifacts,
            original_http_attempts=3,
            budget_used_before=8828,
            budget_used_after=8831,
        )

        with self.assertRaisesRegex(ValueError, "tradable exit bridge"):
            script.validate_bundle(broken)

    def test_bundle_rejects_budget_delta_other_than_three(self):
        bundle = _bundle()
        broken = script.ProviderBundle(
            prices=bundle.prices,
            corporate_actions=bundle.corporate_actions,
            artifacts=bundle.artifacts,
            original_http_attempts=3,
            budget_used_before=8828,
            budget_used_after=8832,
        )

        with self.assertRaisesRegex(ValueError, "budget delta"):
            script.validate_bundle(broken)

    def test_invalid_bar_reports_exact_non_sensitive_fields_and_values(self):
        with self.assertRaises(script.OhlcvValidationError) as caught:
            script.bundle_from_artifacts(
                _invalid_artifacts(),
                original_http_attempts=3,
                budget_used_before=8835,
                budget_used_after=8838,
            )

        self.assertEqual(len(caught.exception.diagnostics), 1)
        row = caught.exception.diagnostics[0]
        self.assertEqual(row["session"], "2023-12-11")
        self.assertEqual(row["row_values"]["open"], 0.01055)
        self.assertEqual(row["row_values"]["low"], 0.0111)
        self.assertEqual(
            row["violations"],
            [
                {
                    "rule": "open_below_low",
                    "fields": ["open", "low"],
                    "values": {"open": 0.01055, "low": 0.0111},
                }
            ],
        )
        self.assertNotIn("api_token", str(caught.exception))

    def test_only_exact_hash_date_allowlist_can_make_minimal_envelope_expansion(self):
        artifacts = _invalid_artifacts()
        eod_hash = artifacts[0].source_hash
        allowlist = {
            eod_hash: {
                "2023-12-11": (
                    {
                        "field": "low",
                        "observed": 0.0111,
                        "corrected": 0.01055,
                        "observed_row": {
                            "open": 0.01055,
                            "high": 0.012,
                            "low": 0.0111,
                            "close": 0.0115,
                            "volume": 12345,
                        },
                        "justification": "reviewed envelope-only correction",
                    },
                )
            }
        }
        with mock.patch.object(
            script, "FRCB_EOD_ENVELOPE_CORRECTION_ALLOWLIST", allowlist
        ):
            bundle = script.bundle_from_artifacts(
                artifacts,
                original_http_attempts=3,
                budget_used_before=8835,
                budget_used_after=8838,
            )

        row = bundle.prices.loc[bundle.prices["session"].eq("2023-12-11")].iloc[0]
        self.assertEqual(float(row["low"]), 0.01055)
        self.assertEqual(len(bundle.envelope_corrections), 1)
        self.assertEqual(bundle.envelope_corrections[0]["raw_eod_sha256"], eod_hash)

        nonminimal = {
            eod_hash: {
                "2023-12-11": (
                    {
                        "field": "low",
                        "observed": 0.0111,
                        "corrected": 0.0105,
                        "observed_row": {
                            "open": 0.01055,
                            "high": 0.012,
                            "low": 0.0111,
                            "close": 0.0115,
                            "volume": 12345,
                        },
                        "justification": "not minimal",
                    },
                )
            }
        }
        with mock.patch.object(
            script, "FRCB_EOD_ENVELOPE_CORRECTION_ALLOWLIST", nonminimal
        ):
            with self.assertRaisesRegex(ValueError, "minimal envelope"):
                script.bundle_from_artifacts(
                    artifacts,
                    original_http_attempts=3,
                    budget_used_before=8835,
                    budget_used_after=8838,
                )

    def test_validation_failure_preserves_three_raws_and_actual_receipt_in_quarantine(self):
        artifacts = _invalid_artifacts()
        payloads = {
            endpoint: artifact.content
            for endpoint, artifact in zip(
                script.EODHD_ENDPOINTS, artifacts, strict=True
            )
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = _FakeBudget(root / "state/eodhd_call_budget.json")
            with mock.patch.object(_FakeRawClient, "payloads", payloads):
                with self.assertRaises(script.OhlcvValidationError) as caught:
                    script.collect_bundle(
                        root,
                        client_factory=_FakeRawClient,
                        budget_factory=lambda: budget,
                    )

            quarantines = list(
                (root / "state/us-frc-para-lifecycle/quarantine").glob(
                    "*.json.gz"
                )
            )
            diagnostics = list(
                (root / "state/us-frc-para-lifecycle/quarantine").glob(
                    "*.validation.json"
                )
            )
            self.assertEqual(len(quarantines), 1)
            self.assertEqual(len(diagnostics), 1)
            envelope = json.loads(gzip.decompress(quarantines[0].read_bytes()))
            self.assertEqual(envelope["budget_receipt"]["used_before"], 8835)
            self.assertEqual(envelope["budget_receipt"]["used_after"], 8838)
            self.assertEqual(envelope["budget_receipt"]["delta"], 3)
            self.assertNotIn("prior_failed_attempt_note", envelope)
            self.assertEqual(len(envelope["artifacts"]), 3)
            for item, endpoint in zip(
                envelope["artifacts"], script.EODHD_ENDPOINTS, strict=True
            ):
                content = base64.b64decode(item["content_base64"], validate=True)
                self.assertEqual(content, payloads[endpoint])
                self.assertEqual(item["content_sha256"], script.sha256_bytes(content))

            report = json.loads(diagnostics[0].read_text(encoding="utf-8"))
            self.assertEqual(report["invalid_ohlcv_rows"], list(caught.exception.diagnostics))
            note = json.loads(
                script.prior_failed_attempt_note_path(root).read_text(encoding="utf-8")
            )
            self.assertEqual(note["budget_used_before"], 8832)
            self.assertEqual(note["budget_used_after"], 8835)
            self.assertFalse(note["raw_responses_preserved"])
            self.assertIsNone(note["raw_payload_binding"])
            self.assertFalse(script.bundle_cache_path(root).exists())

    def test_production_allowlist_pins_exact_raw_hash_date_and_observed_row(self):
        raw_hash = "3c96be232fb5f567e77a94fe315b67aa61e520d1611842620c04680fb5df6ab3"
        instruction = script.FRCB_EOD_ENVELOPE_CORRECTION_ALLOWLIST[raw_hash][
            "2024-12-30"
        ][0]

        self.assertEqual(instruction["field"], "low")
        self.assertEqual(instruction["observed"], 0.0)
        self.assertEqual(instruction["corrected"], 0.003)
        self.assertEqual(
            instruction["observed_row"],
            {
                "open": 0.003,
                "high": 0.006,
                "low": 0.0,
                "close": 0.004,
                "volume": 629864,
            },
        )

    def test_quarantine_offline_promotion_and_bundle_tamper_detection(self):
        artifacts = _invalid_artifacts()
        eod_hash = artifacts[0].source_hash
        allowlist = {
            eod_hash: {
                "2023-12-11": (
                    {
                        "field": "low",
                        "observed": 0.0111,
                        "corrected": 0.01055,
                        "observed_row": {
                            "open": 0.01055,
                            "high": 0.012,
                            "low": 0.0111,
                            "close": 0.0115,
                            "volume": 12345,
                        },
                        "justification": "reviewed envelope-only correction",
                    },
                )
            }
        }
        receipt = {
            "schema": "eodhd_budget_receipt/v1",
            "period": "2026-07-18",
            "used_before": 8836,
            "used_after": 8839,
            "delta": 3,
            "daily_limit": 100000,
            "reserve": 5000,
            "safety_ceiling": 95000,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(
                script, "FRCB_EOD_ENVELOPE_CORRECTION_ALLOWLIST", allowlist
            ):
                quarantine_id, quarantine_path = script._write_raw_quarantine(
                    root, artifacts, receipt
                )
                bundle, quarantine, bundle_path = script.promote_raw_quarantine(
                    root, quarantine_id
                )
                self.assertEqual(quarantine.path, quarantine_path)
                self.assertEqual(bundle.artifacts[0].source_hash, eod_hash)
                self.assertEqual(bundle.budget_used_before, 8836)
                self.assertEqual(bundle.budget_used_after, 8839)
                self.assertEqual(len(bundle.envelope_corrections), 1)
                replay = script._read_bundle_cache(bundle_path)
                self.assertIsNotNone(replay)

                envelope = json.loads(gzip.decompress(bundle_path.read_bytes()))
                envelope["payload"]["budget_used_after"] = 8840
                bundle_path.write_bytes(
                    gzip.compress(script._canonical_json_bytes(envelope), mtime=0)
                )
                with self.assertRaisesRegex(ValueError, "payload hash"):
                    script._read_bundle_cache(bundle_path)

    def test_quarantine_content_address_rejects_tampered_wrapper(self):
        artifacts = _invalid_artifacts()
        receipt = {
            "schema": "eodhd_budget_receipt/v1",
            "period": "2026-07-18",
            "used_before": 8836,
            "used_after": 8839,
            "delta": 3,
            "daily_limit": 100000,
            "reserve": 5000,
            "safety_ceiling": 95000,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            quarantine_id, path = script._write_raw_quarantine(root, artifacts, receipt)
            envelope = json.loads(gzip.decompress(path.read_bytes()))
            envelope["budget_receipt"]["used_after"] = 8840
            path.write_bytes(gzip.compress(script._canonical_json_bytes(envelope), mtime=0))

            with self.assertRaisesRegex(ValueError, "content-address hash"):
                script.read_raw_quarantine(root, quarantine_id)

    def test_occ_extraction_is_hash_stable_and_binds_same_cusip(self):
        first = script._occ_artifact()
        second = script._occ_artifact()

        self.assertEqual(first.content, second.content)
        self.assertEqual(first.source_hash, second.source_hash)
        self.assertIn(script.FRC_CUSIP.encode(), first.content)
        self.assertIn(b"same First Republic common-share identity", first.content)


if __name__ == "__main__":
    unittest.main()
