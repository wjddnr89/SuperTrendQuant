from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from supertrend_quant.config import parse_config
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DatasetManifest,
    ManifestFile,
    sha256_bytes,
)
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.realtime import BarSource, FrameQuoteProvider, RealtimeBarOverlay
from supertrend_quant.market_store.storage import (
    ConditionalWriteFailed,
    DatasetCache,
    DatasetPublisher,
    LocalObjectStore,
)
from supertrend_quant.market_store.validation import validate_dataset


def _base_config(**extra):
    return {
        "strategy": {"name": "test", "type": "equal", "params": {}},
        "scoring": {"type": "relative_strength", "params": {"lookback_bars": 20}},
        "market": "US",
        "universe": {"source": "symbols", "symbols": ["AAPL"]},
        **extra,
    }


class DataStoreConfigTest(unittest.TestCase):
    def test_defaults_use_daily_parquet_total_return(self):
        config = parse_config(_base_config())

        self.assertEqual(config.timeframe, "1d")
        self.assertEqual(config.period, "max")
        self.assertEqual(config.data_store.provider, "parquet")
        self.assertTrue(config.data_store.auto_sync)
        self.assertEqual(config.data_store.price_mode, "total_return_adjusted")
        self.assertEqual(config.data_store.signal_price_mode, "total_return_adjusted")
        self.assertEqual(config.data_store.dividend_tax_rate, 0.0)
        self.assertEqual(config.data_store.dividend_withholding_rate, 0.0)
        self.assertEqual(config.data_store.incomplete_action_policy, "warn")
        self.assertEqual(config.data_store.index_source_mode, "best_effort")

    def test_r2_requires_bucket_when_enabled_and_endpoint_is_environment_only(self):
        with self.assertRaisesRegex(ValueError, "endpoint_env and bucket"):
            parse_config(_base_config(data_store={"r2": {"enabled": True}}))
        config = parse_config(
            _base_config(
                data_store={
                    "r2": {
                        "enabled": True,
                        "bucket": "market-data",
                        "endpoint_env": "PRIVATE_R2_ENDPOINT",
                    }
                }
            )
        )
        self.assertEqual(config.data_store.r2.endpoint_env, "PRIVATE_R2_ENDPOINT")

    def test_public_data_store_names_and_legacy_aliases_are_compatible(self):
        config = parse_config(
            _base_config(
                data_store={
                    "signal_price_mode": "split_adjusted",
                    "dividend_withholding_rate": 0.15,
                }
            )
        )
        self.assertEqual(config.data_store.price_mode, "split_adjusted")
        self.assertEqual(config.data_store.dividend_tax_rate, 0.15)
        with self.assertRaisesRegex(ValueError, "conflicts with legacy"):
            parse_config(
                _base_config(
                    data_store={
                        "signal_price_mode": "raw",
                        "price_mode": "split_adjusted",
                    }
                )
            )

    def test_index_event_universe_accepts_russell3000(self):
        config = parse_config(
            _base_config(
                universe={
                    "source": "index_events",
                    "profiles": {"US": ["russell3000"]},
                }
            )
        )
        self.assertEqual(config.universe.source, "index_events")


class DatasetValidationTest(unittest.TestCase):
    def test_valid_raw_price_frame(self):
        frame = pd.DataFrame(
            [
                {
                    "security_id": "US:037833100",
                    "session": "2026-07-14",
                    "open": 100.0,
                    "high": 103.0,
                    "low": 99.0,
                    "close": 102.0,
                    "volume": 1_000,
                    "currency": "USD",
                    "source": "test",
                    "retrieved_at": "2026-07-15T00:00:00Z",
                    "source_hash": "abc",
                }
            ]
        )
        report = validate_dataset("daily_price_raw", frame)
        self.assertTrue(report.valid)
        self.assertEqual(report.quality, DataQuality.VALID)

    def test_incomplete_action_warns_by_default_and_can_block(self):
        frame = pd.DataFrame(
            [
                {
                    "event_id": "div-1",
                    "security_id": "US:037833100",
                    "action_type": "cash_dividend",
                    "effective_date": "2026-07-14",
                    "ex_date": "2026-07-14",
                    "announcement_date": "",
                    "record_date": "",
                    "payment_date": "",
                    "cash_amount": None,
                    "ratio": None,
                    "currency": "USD",
                    "new_security_id": "",
                    "new_symbol": "",
                    "official": False,
                    "source": "test",
                    "source_url": "memory://test",
                    "source_kind": "fixture",
                    "retrieved_at": "2026-07-15T00:00:00Z",
                    "source_hash": "abc",
                }
            ]
        )

        warning = validate_dataset("corporate_actions", frame)
        blocked = validate_dataset("corporate_actions", frame, incomplete_action_policy="block")

        self.assertTrue(warning.valid)
        self.assertEqual(warning.quality, DataQuality.DEGRADED)
        self.assertFalse(blocked.valid)
        self.assertEqual(blocked.quality, DataQuality.BLOCKED)

    def test_daily_prices_reject_holidays_future_rows_and_missing_latest_session(self):
        base = {
            "security_id": "US:037833100",
            "open": 100.0,
            "high": 103.0,
            "low": 99.0,
            "close": 102.0,
            "volume": 1_000,
            "currency": "USD",
            "source": "test",
            "retrieved_at": "2026-07-15T00:00:00Z",
            "source_hash": "abc",
        }
        holiday = validate_dataset(
            "daily_price_raw",
            pd.DataFrame([{**base, "session": "2026-07-04"}]),
            completed_session="2026-07-06",
        )
        future = validate_dataset(
            "daily_price_raw",
            pd.DataFrame([{**base, "session": "2026-07-15"}]),
            completed_session="2026-07-14",
        )
        stale = validate_dataset(
            "daily_price_raw",
            pd.DataFrame([{**base, "session": "2026-07-13"}]),
            completed_session="2026-07-14",
        )

        self.assertIn("non_trading_session", {issue.code for issue in holiday.issues})
        self.assertIn("future_session", {issue.code for issue in future.issues})
        self.assertIn("missing_completed_session", {issue.code for issue in stale.issues})


class IntradayV2SeamTest(unittest.TestCase):
    def test_fake_bar_source_and_memory_overlay_never_require_persistence(self):
        class FakeBarSource:
            def load(self, symbols, timeframe, period):
                index = pd.date_range("2026-07-14 09:30", periods=2, freq="5min")
                return {
                    symbol: pd.DataFrame({"Close": [10.0, 11.0]}, index=index)
                    for symbol in symbols
                }

        source: BarSource = FakeBarSource()
        historical = source.load(["AAA"], "5m", "1d")
        overlay = RealtimeBarOverlay()
        overlay.replace(
            "AAA",
            pd.DataFrame(
                {"Close": [12.0]},
                index=[pd.Timestamp("2026-07-14 09:40")],
            ),
        )

        merged = overlay.merge(historical)
        quotes = FrameQuoteProvider(merged).quotes(["AAA"])
        self.assertEqual(merged["AAA"]["Close"].tolist(), [10.0, 11.0, 12.0])
        self.assertEqual(quotes["AAA"].price, 12.0)
        self.assertFalse(hasattr(overlay, "save"))
        self.assertFalse(hasattr(overlay, "publish"))


class VersionedStorageTest(unittest.TestCase):
    def test_manifest_roundtrips_audit_and_quality_counters(self):
        manifest = DatasetManifest.create(
            "corporate_actions",
            "v1",
            "2026-07-14",
            (),
            published_by="tester",
            source_mode="best_effort",
            official_coverage_start="2020-01-01",
            official_coverage_end="2026-07-14",
            unresolved_action_count=2,
            conflict_count=1,
        )
        restored = DatasetManifest.from_bytes(manifest.to_bytes())
        self.assertEqual(restored.published_by, "tester")
        self.assertEqual(restored.source_mode, "best_effort")
        self.assertEqual(restored.unresolved_action_count, 2)
        self.assertEqual(restored.conflict_count, 1)

    def _version(self, root: Path, dataset: str, version: str, value: bytes):
        version_root = root / version
        file_path = version_root / "year=2026" / "part-0.parquet"
        file_path.parent.mkdir(parents=True)
        file_path.write_bytes(value)
        item = ManifestFile(
            path="year=2026/part-0.parquet",
            sha256=sha256_bytes(value),
            size_bytes=len(value),
            row_count=1,
            min_session="2026-07-14",
            max_session="2026-07-14",
        )
        return version_root, DatasetManifest.create(
            dataset,
            version,
            "2026-07-14",
            (item,),
        )

    def test_publish_and_cache_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "remote")
            version_root, manifest = self._version(root / "build", "daily_price_raw", "v1", b"parquet")

            published = DatasetPublisher(store).publish(
                version_root,
                manifest,
                expected_pointer_etag=None,
            )
            cached = DatasetCache(root / "cache", store).sync("daily_price_raw")

            self.assertFalse(published.conflict)
            self.assertEqual(cached.version, "v1")
            self.assertEqual(
                (root / "cache/datasets/daily_price_raw/versions/v1/year=2026/part-0.parquet").read_bytes(),
                b"parquet",
            )

    def test_cache_hit_skips_parquet_download_but_hash_mismatch_repairs_it(self):
        class CountingStore(LocalObjectStore):
            def __init__(self, root):
                super().__init__(root)
                self.gets = []

            def get(self, key):
                self.gets.append(key)
                return super().get(key)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = CountingStore(root / "remote")
            version_root, manifest = self._version(
                root / "build", "daily_price_raw", "v1", b"parquet"
            )
            DatasetPublisher(store).publish(
                version_root,
                manifest,
                expected_pointer_etag=None,
            )
            cache = DatasetCache(root / "cache", store)
            cache.sync("daily_price_raw")
            parquet_key = (
                "datasets/daily_price_raw/versions/v1/year=2026/part-0.parquet"
            )
            first_downloads = store.gets.count(parquet_key)

            cache.sync("daily_price_raw")
            self.assertEqual(store.gets.count(parquet_key), first_downloads)

            local = root / "cache" / parquet_key
            local.write_bytes(b"corrupt")
            cache.sync("daily_price_raw")
            self.assertEqual(store.gets.count(parquet_key), first_downloads + 1)
            self.assertEqual(local.read_bytes(), b"parquet")

    def test_stale_pointer_is_quarantined_as_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "remote")
            publisher = DatasetPublisher(store)
            v1_root, v1 = self._version(root / "build", "daily_price_raw", "v1", b"one")
            first = publisher.publish(v1_root, v1, expected_pointer_etag=None)
            _, stale_etag = publisher.current("daily_price_raw")
            self.assertEqual(first.pointer_etag, stale_etag)

            # Another publisher advances current before this writer does.
            v2_root, v2 = self._version(root / "build", "daily_price_raw", "v2", b"two")
            publisher.publish(v2_root, v2, expected_pointer_etag=stale_etag)
            v3_root, v3 = self._version(root / "build", "daily_price_raw", "v3", b"three")
            conflict = publisher.publish(v3_root, v3, expected_pointer_etag=stale_etag)

            current, _ = publisher.current("daily_price_raw")
            self.assertTrue(conflict.conflict)
            self.assertEqual(current.version, "v2")
            self.assertIn(
                "conflicts/daily_price_raw/v3/manifest.json",
                store.list("conflicts"),
            )

    def test_local_store_enforces_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalObjectStore(directory)
            etag = store.put("current.json", b"v1", if_none_match=True)
            with self.assertRaises(ConditionalWriteFailed):
                store.put("current.json", b"v2", if_match="stale")
            store.put("current.json", b"v2", if_match=etag)
            self.assertEqual(store.get("current.json").data, b"v2")


if __name__ == "__main__":
    unittest.main()
