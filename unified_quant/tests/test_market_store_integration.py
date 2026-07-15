from __future__ import annotations

import tempfile
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from supertrend_quant.config import parse_config
from supertrend_quant.data import MarketData
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.preflight import DailyPreflight, expected_completed_us_session
from supertrend_quant.market_store.provider import (
    ParquetMarketDataProvider,
    load_configured_market_data,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.storage import DatasetCache, LocalObjectStore, publish_repository
from supertrend_quant.universe import resolve_universe


def _source(row):
    return {
        **row,
        "source": "fixture",
        "retrieved_at": "2026-07-15T23:00:00Z",
        "source_hash": "fixture-hash",
    }


class ParquetDuckDBIntegrationTest(unittest.TestCase):
    def test_unresolved_action_count_survives_later_delta_and_status_reports_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = LocalDatasetRepository(directory)
            incomplete = pd.DataFrame(
                [
                    _source(
                        {
                            "event_id": "missing-cash",
                            "security_id": "SEC-A",
                            "action_type": "cash_dividend",
                            "effective_date": "2026-07-14",
                            "ex_date": "2026-07-14",
                            "cash_amount": None,
                            "ratio": None,
                            "currency": "USD",
                            "new_security_id": "",
                            "new_symbol": "",
                            "official": False,
                        }
                    )
                ]
            )
            complete = incomplete.copy()
            complete["event_id"] = "complete-cash"
            complete["effective_date"] = "2026-07-15"
            complete["ex_date"] = "2026-07-15"
            complete["cash_amount"] = 1.0
            repository.write_frame(
                "corporate_actions", incomplete, completed_session="2026-07-14"
            )
            repository.append_frame(
                "corporate_actions", complete, completed_session="2026-07-15"
            )

            manifest = repository.current_manifest("corporate_actions")
            status = {row["dataset"]: row for row in repository.status()}
            self.assertEqual(manifest.unresolved_action_count, 1)
            self.assertEqual(status["corporate_actions"]["unresolved_action_count"], 1)
            self.assertGreater(status["corporate_actions"]["size_bytes"], 0)

    def test_top_level_preflight_falls_back_to_one_provider_sync_when_local_is_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            config = parse_config(
                {
                    "strategy": {"name": "test", "type": "equal", "params": {}},
                    "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
                    "market": "US",
                    "universe": {"source": "symbols", "symbols": ["AAA"]},
                    "data_store": {
                        "provider": "parquet",
                        "local_cache_dir": directory,
                        "auto_sync": True,
                    },
                }
            )
            loaded = MarketData(
                bars={"AAA": pd.DataFrame({"Close": [1.0]}, index=[pd.Timestamp("2026-07-15")])},
                completed_session="2026-07-15",
            )
            with (
                patch(
                    "supertrend_quant.market_store.preflight.expected_completed_us_session",
                    return_value="2026-07-15",
                ),
                patch(
                    "supertrend_quant.market_store.ingest.DailyDataSynchronizer"
                ) as synchronizer_type,
                patch.object(ParquetMarketDataProvider, "load", return_value=loaded),
            ):
                synchronizer_type.return_value.sync.return_value = SimpleNamespace(
                    completed_session="2026-07-15"
                )

                result = load_configured_market_data(config, ["AAA"])

            self.assertIs(result, loaded)
            synchronizer_type.return_value.sync.assert_called_once_with(
                "2026-07-15",
                refresh_security_master=True,
            )

    def test_release_is_not_published_until_every_referenced_dataset_is_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = LocalObjectStore(root / "remote")
            repository = LocalDatasetRepository(root / "local")
            row = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": "SEC-A",
                            "session": "2026-07-14",
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.0,
                            "close": 10.0,
                            "volume": 100.0,
                            "currency": "USD",
                        }
                    )
                ]
            )
            repository.write_frame(
                "daily_price_raw", row, completed_session="2026-07-14", version="prices-v1"
            )
            repository.commit_release(
                "2026-07-14",
                {"daily_price_raw": "prices-v1", "adjustment_factors": "factors-v1"},
                quality="valid",
            )

            published = publish_repository(repository, remote, ("daily_price_raw",))

            self.assertTrue(published[0].published)
            self.assertTrue(published[-1].conflict)
            self.assertIn("adjustment_factors=missing", published[-1].detail)
            self.assertFalse((root / "remote" / "releases" / "current.json").exists())

    def test_r2_style_publish_rebases_disjoint_writers_and_updates_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = LocalObjectStore(root / "remote")
            first_repo = LocalDatasetRepository(root / "first")
            row = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": "SEC-A",
                            "session": "2026-07-14",
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.0,
                            "close": 10.0,
                            "volume": 100.0,
                            "currency": "USD",
                        }
                    )
                ]
            )
            first_repo.write_frame("daily_price_raw", row, completed_session="2026-07-14", version="v1")
            first_repo.commit_release(
                "2026-07-14",
                {"daily_price_raw": "v1"},
                quality="valid",
            )
            published = publish_repository(first_repo, remote, ("daily_price_raw",))
            self.assertTrue(published[0].published)
            self.assertTrue(published[-1].published)

            DatasetCache(root / "second", remote).sync_release()
            DatasetCache(root / "third", remote).sync_release()
            second_repo = LocalDatasetRepository(root / "second")
            third_repo = LocalDatasetRepository(root / "third")
            row2 = row.copy()
            row2["session"] = "2026-07-15"
            second_repo.append_frame("daily_price_raw", row2, completed_session="2026-07-15", version="v2-a")
            second_repo.commit_release(
                "2026-07-15",
                {"daily_price_raw": "v2-a"},
                quality="valid",
            )
            row3 = row.copy()
            row3["session"] = "2026-07-16"
            third_repo.append_frame("daily_price_raw", row3, completed_session="2026-07-16", version="v2-b")
            third_repo.commit_release(
                "2026-07-16",
                {"daily_price_raw": "v2-b"},
                quality="valid",
            )

            winner = publish_repository(second_repo, remote, ("daily_price_raw",))
            rebased = publish_repository(third_repo, remote, ("daily_price_raw",))

            self.assertTrue(winner[0].published)
            self.assertTrue(rebased[0].published)
            self.assertFalse(rebased[0].conflict)
            self.assertIn("rebased", rebased[0].detail)
            DatasetCache(root / "reader", remote).sync_release()
            reader = LocalDatasetRepository(root / "reader")
            self.assertEqual(len(reader.read_frame("daily_price_raw")), 3)
            self.assertNotEqual(reader.current_manifest("daily_price_raw").version, "v2-a")
            self.assertEqual(reader.current_release()[0].completed_session, "2026-07-16")

    def test_r2_style_publish_quarantines_same_key_different_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = LocalObjectStore(root / "remote")
            first_repo = LocalDatasetRepository(root / "first")
            row = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": "SEC-A",
                            "session": "2026-07-14",
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.0,
                            "close": 10.0,
                            "volume": 100.0,
                            "currency": "USD",
                        }
                    )
                ]
            )
            first_repo.write_frame(
                "daily_price_raw", row, completed_session="2026-07-14", version="v1"
            )
            first_repo.commit_release("2026-07-14", {"daily_price_raw": "v1"}, quality="valid")
            publish_repository(first_repo, remote, ("daily_price_raw",))
            DatasetCache(root / "writer-a", remote).sync_release()
            DatasetCache(root / "writer-b", remote).sync_release()
            writer_a = LocalDatasetRepository(root / "writer-a")
            writer_b = LocalDatasetRepository(root / "writer-b")
            a = row.copy()
            a["session"] = "2026-07-15"
            a["close"] = 12.0
            a["high"] = 13.0
            b = a.copy()
            b["close"] = 14.0
            b["high"] = 15.0
            writer_a.append_frame("daily_price_raw", a, completed_session="2026-07-15", version="a")
            writer_a.commit_release("2026-07-15", {"daily_price_raw": "a"}, quality="valid")
            writer_b.append_frame("daily_price_raw", b, completed_session="2026-07-15", version="b")
            writer_b.commit_release("2026-07-15", {"daily_price_raw": "b"}, quality="valid")

            publish_repository(writer_a, remote, ("daily_price_raw",))
            conflict = publish_repository(writer_b, remote, ("daily_price_raw",))

            self.assertTrue(conflict[0].conflict)
            self.assertIn("same-key value conflict", conflict[0].detail)
            self.assertIn(
                "conflicts/daily_price_raw/b/manifest.json",
                remote.list("conflicts/daily_price_raw"),
            )
            self.assertEqual(writer_b.conflicts()[0]["version"], "b")

    def test_daily_append_creates_delta_chain_and_compaction_collapses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = LocalDatasetRepository(directory)
            first = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": "SEC-A",
                            "session": "2026-07-14",
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.0,
                            "close": 10.0,
                            "volume": 100.0,
                            "currency": "USD",
                        }
                    )
                ]
            )
            second = first.copy()
            second["session"] = "2026-07-15"
            second["close"] = 12.0
            second["high"] = 13.0
            repository.write_frame("daily_price_raw", first, completed_session="2026-07-14", version="base")
            repository.append_frame("daily_price_raw", second, completed_session="2026-07-15", version="delta")

            chained = repository.read_frame("daily_price_raw")
            self.assertEqual(len(chained), 2)
            self.assertEqual(len(repository.manifest_chain("daily_price_raw")), 2)

            compacted = repository.compact("daily_price_raw")
            self.assertFalse(compacted.conflict)
            self.assertEqual(len(repository.manifest_chain("daily_price_raw")), 1)
            self.assertEqual(len(repository.read_frame("daily_price_raw")), 2)

    def test_identical_collection_rerun_is_logically_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = LocalDatasetRepository(directory)
            row = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": "SEC-A",
                            "session": "2026-07-14",
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.0,
                            "close": 10.0,
                            "volume": 100.0,
                            "currency": "USD",
                        }
                    )
                ]
            )
            repository.write_frame(
                "daily_price_raw", row, completed_session="2026-07-14", version="first"
            )
            repository.append_frame(
                "daily_price_raw", row, completed_session="2026-07-14", version="rerun"
            )

            logical = repository.read_frame("daily_price_raw")
            self.assertEqual(len(logical), 1)
            self.assertFalse(logical.duplicated(["security_id", "session"]).any())

    def test_changed_overlap_is_quarantined_without_advancing_current(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = LocalDatasetRepository(directory)
            first = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": "SEC-A",
                            "session": "2026-07-14",
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.0,
                            "close": 10.0,
                            "volume": 100.0,
                            "currency": "USD",
                        }
                    )
                ]
            )
            changed = first.copy()
            changed["close"] = 10.5
            repository.write_frame(
                "daily_price_raw", first, completed_session="2026-07-14", version="base"
            )

            result = repository.append_frame(
                "daily_price_raw", changed, completed_session="2026-07-14", version="revision"
            )

            self.assertTrue(result.conflict)
            self.assertEqual(repository.current_manifest("daily_price_raw").version, "base")
            self.assertTrue((Path(directory) / result.conflict_path).is_file())

    def test_repository_writes_partitioned_parquet_and_provider_reads_dual_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = LocalDatasetRepository(root)
            history = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": "SEC-AAPL",
                            "symbol": "AAPL",
                            "exchange": "XNAS",
                            "effective_from": "1980-12-12",
                            "effective_to": "",
                        }
                    ),
                    _source(
                        {
                            "security_id": "SEC-QQQ",
                            "symbol": "QQQ",
                            "exchange": "XNAS",
                            "effective_from": "1999-03-10",
                            "effective_to": "",
                        }
                    ),
                ]
            )
            repository.write_frame("symbol_history", history, completed_session="2026-07-15", version="symbols-v1")
            prices = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": security_id,
                            "session": session,
                            "open": close - 1,
                            "high": close + 1,
                            "low": close - 2,
                            "close": close,
                            "volume": 1000.0,
                            "currency": "USD",
                        }
                    )
                    for security_id, closes in (
                        ("SEC-AAPL", (80.0, 100.0, 50.0)),
                        ("SEC-QQQ", (400.0, 500.0, 501.0)),
                    )
                    for session, close in zip(
                        ("2025-01-02", "2026-07-14", "2026-07-15"),
                        closes,
                    )
                ]
            )
            repository.write_frame("daily_price_raw", prices, completed_session="2026-07-15", version="prices-v1")
            actions = pd.DataFrame(
                [
                    _source(
                        {
                            "event_id": "split-aapl",
                            "security_id": "SEC-AAPL",
                            "action_type": "split",
                            "effective_date": "2026-07-15",
                            "ex_date": "2026-07-15",
                            "cash_amount": None,
                            "ratio": 2.0,
                            "currency": "USD",
                            "new_security_id": "",
                            "new_symbol": "",
                            "official": True,
                        }
                    )
                ]
            )
            repository.write_frame("corporate_actions", actions, completed_session="2026-07-15", version="actions-v1")
            factors = build_adjustment_factors(prices, actions, source_version="prices-v1+actions-v1")
            repository.write_frame("adjustment_factors", factors, completed_session="2026-07-15", version="factors-v1")
            config = parse_config(
                {
                    "strategy": {"name": "test", "type": "equal", "params": {}},
                    "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
                    "market": "US",
                    "universe": {"source": "symbols", "symbols": ["AAPL"]},
                    "timeframe": "1d",
                    "period": "1d",
                    "data_store": {
                        "provider": "parquet",
                        "local_cache_dir": str(root),
                        "price_mode": "total_return_adjusted",
                    },
                }
            )

            provider = ParquetMarketDataProvider(root, capture_query_plans=True)
            market_data = provider.load(config, ["AAPL"])

            self.assertEqual(market_data.execution_bars["AAPL"].iloc[0]["Close"], 100.0)
            self.assertEqual(market_data.bars["AAPL"].iloc[0]["Close"], 50.0)
            self.assertEqual(market_data.benchmark["AAPL"].iloc[-1]["Close"], 501.0)
            self.assertEqual(market_data.completed_session, "2026-07-15")
            self.assertEqual(market_data.validated_session, "2026-07-15")
            self.assertEqual(market_data.data_quality, "valid")
            self.assertIn("daily_price_raw=prices-v1", market_data.data_version)
            self.assertEqual(market_data.corporate_actions[0]["event_id"], "split-aapl")
            manifest = repository.current_manifest("daily_price_raw")
            self.assertTrue(
                any(item.path.startswith("year=2026/month=07/") for item in manifest.files)
            )
            self.assertEqual(
                provider.query_counts,
                {
                    "symbol_history": 1,
                    "daily_price_raw": 1,
                    "adjustment_factors": 1,
                    "corporate_actions": 1,
                },
            )
            self.assertEqual(list(root.rglob("*.duckdb")), [])
            self.assertEqual(len(provider.query_files["daily_price_raw"]), 1)
            self.assertIn("year=2026", provider.query_files["daily_price_raw"][0])
            self.assertNotIn("year=2025", provider.query_files["daily_price_raw"][0])
            self.assertIn("Filters", provider.query_plans["daily_price_raw"])
            self.assertIn("session", provider.query_plans["daily_price_raw"])
            expected_signal = pd.DataFrame(
                {
                    "Open": [49.5, 49.0],
                    "High": [50.5, 51.0],
                    "Low": [49.0, 48.0],
                    "Close": [50.0, 50.0],
                    "Volume": [2000.0, 1000.0],
                },
                index=pd.to_datetime(["2026-07-14", "2026-07-15"]),
            )
            expected_signal.index.name = "Date"
            pd.testing.assert_frame_equal(
                market_data.bars["AAPL"],
                expected_signal,
                check_freq=False,
            )

    def test_index_event_universe_resolves_stable_ids_into_point_in_time_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = LocalDatasetRepository(directory)
            master = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": security_id,
                            "primary_symbol": symbol,
                            "name": symbol,
                            "exchange": "XNAS",
                            "asset_type": "STOCK",
                            "currency": "USD",
                            "country": "US",
                            "active_from": "2000-01-01",
                            "active_to": "",
                        }
                    )
                    for security_id, symbol in (("SEC-A", "AAA"), ("SEC-B", "BBB"), ("SEC-C", "CCC"))
                ]
            )
            history = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": security_id,
                            "symbol": symbol,
                            "exchange": "XNAS",
                            "effective_from": "2000-01-01",
                            "effective_to": "",
                        }
                    )
                    for security_id, symbol in (("SEC-A", "AAA"), ("SEC-B", "BBB"), ("SEC-C", "CCC"))
                ]
            )
            anchors = pd.DataFrame(
                [
                    _source(
                        {
                            "index_id": "sp500",
                            "anchor_date": "2026-07-14",
                            "security_id": security_id,
                            "official": True,
                        }
                    )
                    for security_id in ("SEC-A", "SEC-B")
                ]
            )
            events = pd.DataFrame(
                [
                    _source(
                        {
                            "event_id": "remove-a",
                            "index_id": "sp500",
                            "effective_date": "2026-07-15",
                            "operation": "REMOVE",
                            "security_id": "SEC-A",
                            "official": True,
                        }
                    ),
                    _source(
                        {
                            "event_id": "add-c",
                            "index_id": "sp500",
                            "effective_date": "2026-07-15",
                            "operation": "ADD",
                            "security_id": "SEC-C",
                            "official": True,
                        }
                    ),
                ]
            )
            repository.write_frame("security_master", master, completed_session="2026-07-15", version="master-v1")
            repository.write_frame("symbol_history", history, completed_session="2026-07-15", version="symbols-v1")
            repository.write_frame("index_constituent_anchors", anchors, completed_session="2026-07-15", version="anchors-v1")
            repository.write_frame("index_membership_events", events, completed_session="2026-07-15", version="events-v1")
            config = parse_config(
                {
                    "strategy": {"name": "test", "type": "equal", "params": {}},
                    "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
                    "market": "US",
                    "universe": {"source": "index_events", "profiles": {"US": ["sp500"]}},
                    "data_store": {"local_cache_dir": directory},
                }
            )

            resolved = resolve_universe(config, as_of=pd.Timestamp("2026-07-15").date())

            self.assertEqual(resolved.eligible_symbols, ("BBB", "CCC"))
            self.assertEqual(resolved.member_for("CCC").security_id, "SEC-C")
            self.assertEqual([entry.effective_date for entry in resolved.schedule], ["2026-07-14", "2026-07-15"])
            self.assertEqual(resolved.schedule[0].symbols, ("AAA", "BBB"))


class DailyPreflightTest(unittest.TestCase):
    def test_expected_session_waits_for_ninety_minute_publication_delay(self):
        before_delay = datetime(2026, 7, 15, 20, 30, tzinfo=UTC)
        after_delay = datetime(2026, 7, 15, 22, 0, tzinfo=UTC)
        self.assertEqual(expected_completed_us_session(before_delay), "2026-07-14")
        self.assertEqual(expected_completed_us_session(after_delay), "2026-07-15")

    def test_only_one_automatic_attempt_per_expected_session_unless_forced(self):
        with tempfile.TemporaryDirectory() as directory:
            attempts = []

            def sync(expected):
                attempts.append(expected)
                return "2026-07-14"

            preflight = DailyPreflight(Path(directory) / "preflight.json")
            now = datetime(2026, 7, 15, 22, 0, tzinfo=UTC)
            first = preflight.run("2026-07-14", auto_sync=True, sync=sync, now=now)
            second = preflight.run("2026-07-14", auto_sync=True, sync=sync, now=now)
            forced = preflight.run("2026-07-14", auto_sync=True, sync=sync, force=True, now=now)

            self.assertTrue(first.sync_attempted)
            self.assertFalse(second.sync_attempted)
            self.assertTrue(forced.sync_attempted)
            self.assertEqual(len(attempts), 2)
            state = json.loads((Path(directory) / "preflight.json").read_text())
            self.assertEqual(state["last_attempted_session"], "2026-07-15")
            self.assertNotIn("last_validated_session", state)

    def test_success_records_last_validated_session_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            preflight = DailyPreflight(path)
            now = datetime(2026, 7, 15, 22, 0, tzinfo=UTC)

            result = preflight.run(
                "2026-07-14",
                auto_sync=True,
                sync=lambda expected: expected,
                now=now,
            )

            self.assertTrue(result.ready)
            state = json.loads(path.read_text())
            self.assertEqual(state["last_attempted_session"], "2026-07-15")
            self.assertEqual(state["last_validated_session"], "2026-07-15")


if __name__ == "__main__":
    unittest.main()
