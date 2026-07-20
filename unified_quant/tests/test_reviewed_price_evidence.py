from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from supertrend_quant.config import load_data_store_config
from supertrend_quant.market_store.cross_validation import (
    _check_report_rows,
    _validate_policy_contract,
    independent_provider_source_mask,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.reviewed_price_evidence import (
    REVIEWED_PRICE_EVIDENCE_BASIS,
    TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256,
    TRUSTED_REVIEWED_PRICE_EVIDENCE_TARGET_IDS,
    build_reviewed_price_projection,
    reviewed_price_evidence_inventory_sha256,
    reviewed_price_evidence_registry,
    verify_reviewed_price_projection,
)
from supertrend_quant.market_store.yahoo_chart import YahooChartCache


ROOT = Path(__file__).parents[2]
POLICY_PATH = ROOT / "unified_quant/configs/us_cross_validation.yaml"
DATA_CONFIG_PATH = ROOT / "unified_quant/configs/data.yaml"
CACHE_PATH = ROOT / "data/cache/state/us_cross_validation/yahoo_chart"
SPLIT_TYPES = {"split", "capital_reduction", "stock_dividend"}
SCRIPT_PATH = ROOT / "unified_quant/scripts/validate_us_lifecycle_cross_sources.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "validate_us_lifecycle_cross_sources_reviewed_price_test", SCRIPT_PATH
)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
collector = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = collector
SCRIPT_SPEC.loader.exec_module(collector)


def _policy() -> dict:
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


class ReviewedPriceEvidenceContractTest(unittest.TestCase):
    def test_registry_is_exactly_code_pinned_and_excludes_unsafe_cases(self):
        policy = _policy()
        registry = reviewed_price_evidence_registry(policy["prices"])
        self.assertEqual(set(registry), set(TRUSTED_REVIEWED_PRICE_EVIDENCE_TARGET_IDS))
        self.assertEqual(len(registry), 17)
        self.assertEqual(
            reviewed_price_evidence_inventory_sha256(policy["prices"]),
            TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256,
        )
        symbols = {item["symbol"] for item in registry.values()}
        self.assertIn("OVV", symbols)
        self.assertTrue(
            {"GPN", "FRCB", "BBBY", "BBT", "DD", "COL"}.isdisjoint(symbols)
        )

    def test_publication_report_attestation_rejects_tampering(self):
        target_id = next(
            item
            for item in TRUSTED_REVIEWED_PRICE_EVIDENCE_TARGET_IDS
            if item.startswith("d5a785")
        )
        item = {
            "target_id": target_id,
            "status": "passed",
            "validation_basis": REVIEWED_PRICE_EVIDENCE_BASIS,
            "overlap_session_count": 1,
            "independent_internal_price_rows": 1,
            "self_source_rows_excluded": 0,
            "reviewed_price_evidence_applied": True,
            "reviewed_price_evidence_registry_sha256": (
                TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256
            ),
            "reviewed_price_evidence_sha256": "1" * 64,
            "reviewed_price_projection_sha256": "2" * 64,
            "reviewed_internal_ohlcv_sha256": "3" * 64,
            "reviewed_provider_ohlcv_sha256": "4" * 64,
            "reviewed_overlap_ohlcv_sha256": "5" * 64,
            "reviewed_all_null_sessions_sha256": "6" * 64,
            "reviewed_price_mismatch_rows": [],
            "reviewed_triple_supertrend_signal": {},
            "reviewed_provider_metadata": {},
            "reviewed_price_limitation": "exact reviewed limitation",
            "reviewed_official_event_binding_passed": True,
            "all_overlap_sessions_compared": True,
            "scale_stability_passed": True,
            "price_tolerance_passed": True,
            "session_coverage_passed": True,
            "currency_passed": True,
            "identity_boundary_passed": True,
            "provider_adjustment_basis": "reviewed_exact_raw_quote_ohlcv",
            "adjusted_close_used": False,
        }
        report = {"events": [], "permanent_exceptions": [], "prices": [item]}
        self.assertEqual(_check_report_rows(report)["price_pass_count"], 1)
        for field, value in (
            ("reviewed_price_evidence_registry_sha256", "0" * 64),
            ("reviewed_price_projection_sha256", "short"),
            ("reviewed_triple_supertrend_signal", []),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(report)
                changed["prices"][0][field] = value
                with self.assertRaisesRegex(
                    RuntimeError, "Reviewed exact price report is incomplete"
                ):
                    _check_report_rows(changed)

    def test_publication_policy_rejects_registry_drift(self):
        changed = copy.deepcopy(_policy())
        changed["prices"]["reviewed_price_evidence"].pop()
        with self.assertRaisesRegex(
            RuntimeError, "Reviewed price-evidence inventory is not"
        ):
            _validate_policy_contract(changed)


@unittest.skipUnless(
    DATA_CONFIG_PATH.is_file() and CACHE_PATH.is_dir(),
    "local frozen release/cache is required for exact evidence replay",
)
class ReviewedPriceEvidenceFrozenReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = _policy()
        cls.registry = reviewed_price_evidence_registry(cls.policy["prices"])
        config = load_data_store_config(DATA_CONFIG_PATH)
        cls.repository = LocalDatasetRepository(config.local_cache_dir)
        cls.release, _ = cls.repository.current_release()
        if cls.release is None:
            raise unittest.SkipTest("current local release is unavailable")
        cls.prices = cls.repository.read_frame(
            "daily_price_raw", cls.release.dataset_versions["daily_price_raw"]
        )
        cls.actions = cls.repository.read_frame(
            "corporate_actions", cls.release.dataset_versions["corporate_actions"]
        )
        cls.release_end = (
            pd.to_datetime(cls.prices["session"], errors="raise")
            .max()
            .date()
            .isoformat()
        )
        provider = cls.policy["provider"]
        cls.cache = YahooChartCache(
            CACHE_PATH,
            endpoint_template=provider["endpoint_template"],
            max_http_attempts=provider["max_http_attempts"],
            timeout_seconds=provider["timeout_seconds"],
            max_response_bytes=provider["max_response_bytes"],
        )

    @classmethod
    def _replay(cls, spec: dict, *, content: bytes | None = None, prices=None):
        start = pd.Timestamp(spec["identity_active_from"], tz="UTC")
        end = pd.Timestamp(
            spec["identity_active_to"] or cls.release_end, tz="UTC"
        )
        response = cls.cache.get(
            spec["symbol"],
            period1=int(start.timestamp()),
            period2=int((end + pd.Timedelta(days=1)).timestamp()),
        )
        if response is None:
            raise unittest.SkipTest(
                f"frozen Yahoo cache is missing for {spec['symbol']}"
            )
        own = (cls.prices if prices is None else prices).loc[
            (cls.prices if prices is None else prices)["security_id"]
            .astype(str)
            .eq(spec["security_id"])
        ].copy()
        own = own.loc[~independent_provider_source_mask(own)].copy()
        split_dates = [
            str(value)[:10]
            for value in cls.actions.loc[
                cls.actions["security_id"].astype(str).eq(spec["security_id"])
                & cls.actions["action_type"]
                .astype(str)
                .str.lower()
                .isin(SPLIT_TYPES),
                "effective_date",
            ]
        ]
        provider_rows, projection = build_reviewed_price_projection(
            content=response.content if content is None else content,
            spec=spec,
            target={
                "target_id": spec["target_id"],
                "security_id": spec["security_id"],
                "symbol": spec["symbol"],
                "active_from": spec["identity_active_from"],
                "active_to": spec["identity_active_to"],
            },
            internal_prices=own,
            split_dates=split_dates,
            policy_prices=cls.policy["prices"],
        )
        return response, provider_rows, projection

    def test_every_frozen_review_recomputes_to_its_exact_projection(self):
        for target_id, spec in self.registry.items():
            with self.subTest(symbol=spec["symbol"], target_id=target_id):
                response, provider, projection = self._replay(spec)
                self.assertEqual(response.source_hash, spec["source_sha256"])
                self.assertEqual(response.wrapper_hash, spec["cache_wrapper_sha256"])
                self.assertGreater(len(provider), 0)
                self.assertEqual(
                    verify_reviewed_price_projection(projection, spec),
                    spec["expected_projection_sha256"],
                )

    def test_payload_internal_row_and_projection_tampering_fail_closed(self):
        ovv = next(
            item for item in self.registry.values() if item["symbol"] == "OVV"
        )
        response, _, projection = self._replay(ovv)
        tampered_payload = bytearray(response.content)
        tampered_payload[-2] = ord(" ")
        with self.assertRaisesRegex(RuntimeError, "response bytes changed"):
            self._replay(ovv, content=bytes(tampered_payload))

        tampered_prices = self.prices.copy()
        row_index = tampered_prices.index[
            tampered_prices["security_id"].astype(str).eq(ovv["security_id"])
        ][0]
        tampered_prices.loc[row_index, "close"] = (
            float(tampered_prices.loc[row_index, "close"]) * 2.0
        )
        with self.assertRaisesRegex(RuntimeError, "mismatch row inventory changed"):
            self._replay(ovv, prices=tampered_prices)

        tampered_spec = copy.deepcopy(ovv)
        tampered_spec["expected_projection_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "projection hash changed"):
            verify_reviewed_price_projection(projection, tampered_spec)

    def test_collector_uses_exact_ovv_path_and_rejects_changed_bytes(self):
        ovv = next(
            item for item in self.registry.values() if item["symbol"] == "OVV"
        )
        response, _, _ = self._replay(ovv)
        target = collector.PriceTarget(
            security_id=ovv["security_id"],
            symbol=ovv["symbol"],
            origins=("frozen_test",),
            active_from=ovv["identity_active_from"],
            active_to=ovv["identity_active_to"],
            request_start=ovv["identity_active_from"],
            request_end=self.release_end,
        )
        policy = collector.Policy(self.policy)
        passed = collector.build_price_checks(
            [target],
            {target.target_id: response},
            self.prices,
            self.actions,
            [],
            policy,
        )
        self.assertEqual(passed[0]["status"], "passed")
        self.assertTrue(passed[0]["reviewed_price_evidence_applied"])
        self.assertEqual(
            passed[0]["reviewed_price_projection_sha256"],
            ovv["expected_projection_sha256"],
        )

        changed_response = replace(response, content=response.content + b" ")
        failed = collector.build_price_checks(
            [target],
            {target.target_id: changed_response},
            self.prices,
            self.actions,
            [],
            policy,
        )
        self.assertEqual(failed[0]["status"], "mismatch")
        self.assertFalse(failed[0]["reviewed_price_evidence_applied"])


if __name__ == "__main__":
    unittest.main()
