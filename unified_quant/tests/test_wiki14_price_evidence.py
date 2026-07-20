from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

from supertrend_quant.config import load_data_store_config
from supertrend_quant.market_store.cross_validation import (
    _check_report_rows,
    _validate_policy_contract,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.wiki14_price_evidence import (
    REVIEWED_WIKI14_PRICE_ONLY_BASIS,
    TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256,
    TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS,
    WIKI14_ARCHIVE_ARTIFACT_INVENTORY_SHA256,
    WIKI14_PROVENANCE_SHA256,
    verify_wiki14_price_only_evidence,
    wiki14_price_only_inventory_sha256,
    wiki14_price_only_registry,
)


ROOT = Path(__file__).parents[2]
POLICY_PATH = ROOT / "unified_quant/configs/us_cross_validation.yaml"
DATA_CONFIG_PATH = ROOT / "unified_quant/configs/data.yaml"
SCRIPT_PATH = ROOT / "unified_quant/scripts/validate_us_lifecycle_cross_sources.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "validate_us_lifecycle_cross_sources_wiki14_test", SCRIPT_PATH
)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
collector = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = collector
SCRIPT_SPEC.loader.exec_module(collector)


def _policy() -> dict:
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


class Wiki14PriceOnlyContractTest(unittest.TestCase):
    def test_registry_and_policy_are_exactly_code_pinned(self):
        policy = _policy()
        registry = wiki14_price_only_registry(policy["prices"])
        self.assertEqual(
            set(registry), set(TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS)
        )
        self.assertEqual(len(registry), 14)
        self.assertEqual(
            {item["symbol"] for item in registry.values()},
            {
                "ADT", "CAM", "COL", "EMC", "EVHC", "FB", "FOX", "FOXA",
                "INFO", "NFX", "SCG", "SNDK", "STI", "TE",
            },
        )
        self.assertEqual(
            wiki14_price_only_inventory_sha256(policy["prices"]),
            TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256,
        )
        _validate_policy_contract(policy)

    def test_publication_policy_rejects_target_or_license_scope_drift(self):
        changed = copy.deepcopy(_policy())
        changed["prices"]["reviewed_wiki14_price_only_evidence"].pop()
        with self.assertRaisesRegex(RuntimeError, "WIKI14 price-only evidence"):
            _validate_policy_contract(changed)


@unittest.skipUnless(
    DATA_CONFIG_PATH.is_file(),
    "local frozen release is required for exact WIKI14 evidence replay",
)
class Wiki14PriceOnlyFrozenReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = _policy()
        config = load_data_store_config(DATA_CONFIG_PATH)
        cls.repository = LocalDatasetRepository(config.local_cache_dir)
        cls.release, _ = cls.repository.current_release()
        if cls.release is None:
            raise unittest.SkipTest("current local release is unavailable")
        names = (
            "daily_price_raw",
            "adjustment_factors",
            "security_master",
            "symbol_history",
            "corporate_actions",
            "lifecycle_resolutions",
            "source_archive",
        )
        cls.frames = {
            name: cls.repository.read_frame(
                name, cls.release.dataset_versions[name]
            )
            for name in names
        }
        targets = collector.build_price_targets(
            cls.frames["security_master"],
            cls.frames["symbol_history"],
            cls.frames["corporate_actions"],
            cls.frames["lifecycle_resolutions"],
            cls.frames["daily_price_raw"],
        )
        cls.target_objects = {
            target.target_id: target
            for target in targets
            if target.target_id in TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS
        }
        cls.targets = {
            target.target_id: {
                "target_id": target.target_id,
                "security_id": target.security_id,
                "symbol": target.symbol,
                "provider_symbol": target.provider_symbol,
                "active_from": target.active_from,
                "active_to": target.active_to,
                "terminal_event_id": target.terminal_event_id,
            }
            for target in targets
            if target.target_id in TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS
        }
        try:
            cls.evidence = cls._verify()
        except (FileNotFoundError, RuntimeError) as exc:
            raise unittest.SkipTest(f"frozen WIKI14 archive is unavailable: {exc}")

    @classmethod
    def _verify(cls, *, prices=None, archive=None):
        return verify_wiki14_price_only_evidence(
            cls.repository,
            cls.frames["source_archive"] if archive is None else archive,
            prices=cls.frames["daily_price_raw"] if prices is None else prices,
            factors=cls.frames["adjustment_factors"],
            master=cls.frames["security_master"],
            history=cls.frames["symbol_history"],
            actions=cls.frames["corporate_actions"],
            targets=cls.targets,
            prices_policy=cls.policy["prices"],
            release_warnings=cls.release.warnings,
        )

    def test_every_extract_replays_price_relation_and_triple_supertrend(self):
        self.assertEqual(
            set(self.evidence), set(TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS)
        )
        self.assertEqual(
            sum(item["overlap_session_count"] for item in self.evidence.values()),
            8_499,
        )
        for item in self.evidence.values():
            with self.subTest(symbol=item["symbol"]):
                self.assertEqual(
                    item["validation_basis"], REVIEWED_WIKI14_PRICE_ONLY_BASIS
                )
                self.assertTrue(item["private_internal_only"])
                self.assertFalse(item["redistribution_allowed"])
                self.assertFalse(item["public_publication_allowed"])
                self.assertEqual(
                    item["triple_supertrend"]["current_signal_sha256"],
                    item["triple_supertrend"]["substituted_signal_sha256"],
                )
                self.assertFalse(
                    any(item["triple_supertrend"]["field_differences"].values())
                )

    def test_archive_metadata_and_local_price_tampering_fail_closed(self):
        changed_archive = self.frames["source_archive"].copy()
        index = changed_archive.index[
            changed_archive["archive_id"].astype(str).eq(WIKI14_PROVENANCE_SHA256)
        ][0]
        changed_archive.loc[index, "source_url"] = "https://example.invalid/wiki.csv"
        with self.assertRaisesRegex(RuntimeError, "source_archive metadata changed"):
            self._verify(archive=changed_archive)

        adt = next(
            item for item in self.evidence.values() if item["symbol"] == "ADT"
        )
        changed_prices = self.frames["daily_price_raw"].copy()
        row = changed_prices.index[
            changed_prices["security_id"].astype(str).eq(adt["security_id"])
        ][0]
        changed_prices.loc[row, "close"] = float(changed_prices.loc[row, "close"]) + 1.0
        with self.assertRaisesRegex(RuntimeError, "raw price economics changed"):
            self._verify(prices=changed_prices)

    def test_collector_and_report_gate_preserve_price_only_scope(self):
        diagnostic = next(
            item for item in self.evidence.values() if item["symbol"] == "ADT"
        )
        target = self.target_objects[diagnostic["target_id"]]
        checks = collector.build_price_checks(
            [target],
            {},
            self.frames["daily_price_raw"],
            self.frames["corporate_actions"],
            [],
            collector.Policy(self.policy),
            wiki14_price_only_evidence={target.target_id: diagnostic},
        )
        item = checks[0]
        self.assertEqual(item["status"], "explicit_exception")
        self.assertTrue(item["price_only_arbitration_passed"])
        self.assertFalse(item["corporate_actions_validated"])
        self.assertFalse(item["adjustment_factors_validated"])
        self.assertEqual(item["source_sha256"], diagnostic["extract_sha256"])
        self.assertEqual(item["provenance_sha256"], WIKI14_PROVENANCE_SHA256)
        computed = _check_report_rows(
            {"events": [], "permanent_exceptions": [], "prices": checks}
        )
        self.assertEqual(computed["price_exception_count"], 1)

        changed = copy.deepcopy(item)
        changed["reviewed_wiki14_price_only_evidence"][
            "redistribution_allowed"
        ] = True
        with self.assertRaisesRegex(RuntimeError, "WIKI14 price-only exception"):
            _check_report_rows(
                {"events": [], "permanent_exceptions": [], "prices": [changed]}
            )

    def test_artifact_inventory_pin_is_nonempty(self):
        self.assertEqual(len(WIKI14_ARCHIVE_ARTIFACT_INVENTORY_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
