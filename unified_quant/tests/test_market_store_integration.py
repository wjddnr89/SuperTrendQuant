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
from supertrend_quant.indicators import calculate_supertrend
from supertrend_quant.ledger import PortfolioLedger
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.preflight import DailyPreflight, expected_completed_us_session
from supertrend_quant.market_store.provider import (
    ParquetMarketDataProvider,
    _filter_actions_to_requested_intervals,
    _frames_by_identity_intervals,
    _scheduled_symbols_on,
    load_configured_market_data,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.storage import DatasetCache, LocalObjectStore, publish_repository
from supertrend_quant.portfolio import Position
from supertrend_quant.ranking import RelativeStrengthScorer
from supertrend_quant.universe import resolve_universe


def _source(row):
    return {
        **row,
        "source": "fixture",
        "retrieved_at": "2026-07-15T23:00:00Z",
        "source_hash": "fixture-hash",
    }


class ParquetDuckDBIntegrationTest(unittest.TestCase):
    def test_terminal_action_after_weekend_remains_in_requested_identity(self):
        actions = pd.DataFrame(
            [
                {
                    "event_id": "wfm-monday-close",
                    "security_id": "SEC-WFM",
                    "action_type": "cash_merger",
                    "effective_date": "2017-08-28",
                    "ex_date": "2017-08-28",
                },
                {
                    "event_id": "unrelated-late-action",
                    "security_id": "SEC-WFM",
                    "action_type": "cash_merger",
                    "effective_date": "2017-09-08",
                    "ex_date": "2017-09-08",
                },
            ]
        )
        # The old symbol's last price/history date is Friday 2017-08-25;
        # identity intervals represent that with an exclusive 2017-08-26 end.
        intervals = {
            "WFM": (
                (
                    "SEC-WFM",
                    pd.Timestamp("2015-01-01"),
                    pd.Timestamp("2017-08-26"),
                ),
            )
        }

        selected = _filter_actions_to_requested_intervals(
            actions,
            intervals,
            ("SEC-WFM",),
        )

        self.assertEqual(selected["event_id"].tolist(), ["wfm-monday-close"])

    def test_query_unifies_optional_column_types_across_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.parquet"
            second = root / "second.parquet"
            pd.DataFrame(
                {"security_id": ["SEC-A"], "metadata": [None]}
            ).to_parquet(first, index=False)
            pd.DataFrame(
                {"security_id": ["SEC-A"], "metadata": ['{"kind":"ticker_change"}']}
            ).to_parquet(second, index=False)

            provider = ParquetMarketDataProvider(root)
            provider._release_versions = {"corporate_actions": "mixed-v1"}
            provider.repository = SimpleNamespace(
                manifest_for_version=lambda dataset, version: SimpleNamespace(
                    version=version
                ),
                parquet_paths=lambda dataset, version, **kwargs: (first, second),
            )

            result = provider._query_dataset(
                "corporate_actions",
                where_column="security_id",
                values=("SEC-A",),
                columns=("security_id", "metadata"),
            )

            self.assertEqual(result["security_id"].tolist(), ["SEC-A", "SEC-A"])
            self.assertTrue(pd.isna(result.iloc[0]["metadata"]))
            self.assertEqual(result.iloc[1]["metadata"], '{"kind":"ticker_change"}')

    def test_completed_session_uses_latest_index_schedule_for_freshness(self):
        schedule = (
            {"effective_date": "2026-03-23", "symbols": ["AAPL", "SATS"]},
            {"effective_date": "2026-06-24", "symbols": ["AAPL"]},
        )

        self.assertEqual(
            _scheduled_symbols_on(schedule, "2026-07-15"),
            ("AAPL",),
        )

    def test_actions_outside_a_ticker_identity_interval_are_excluded(self):
        provider = ParquetMarketDataProvider("unused")
        provider._resolved_symbol_history = pd.DataFrame(
            [
                {
                    "security_id": "SEC-PEAK-SOURCE",
                    "symbol": "PEAK",
                    "_start": pd.Timestamp("2015-01-01"),
                    "_end": pd.Timestamp("2019-09-16"),
                }
            ]
        )
        actions = pd.DataFrame(
            [
                {
                    "event_id": "inside",
                    "security_id": "SEC-PEAK-SOURCE",
                    "action_type": "cash_dividend",
                    "effective_date": "2019-08-02",
                },
                {
                    "event_id": "outside",
                    "security_id": "SEC-PEAK-SOURCE",
                    "action_type": "cash_dividend",
                    "effective_date": "2021-02-19",
                },
            ]
        )

        selected = provider._actions_in_symbol_intervals(actions)

        self.assertEqual(selected["event_id"].tolist(), ["inside"])
        self.assertEqual(selected["symbol"].tolist(), ["PEAK"])

    def test_same_day_action_created_child_can_use_its_exit_mark(self):
        provider = ParquetMarketDataProvider("unused")
        provider._resolved_symbol_history = pd.DataFrame(
            [
                {
                    "security_id": "SEC-BMYRT",
                    "symbol": "BMYRT",
                    "_start": pd.Timestamp("2019-11-21"),
                    "_end": pd.Timestamp("2019-11-21"),
                }
            ]
        )
        actions = pd.DataFrame(
            [
                {
                    "event_id": "create-bmyrt",
                    "security_id": "SEC-CELG",
                    "action_type": "spinoff",
                    "effective_date": "2019-11-21",
                    "ex_date": "2019-11-21",
                    "new_security_id": "SEC-BMYRT",
                    "new_symbol": "BMYRT",
                },
                {
                    "event_id": "bmyrt-first-session-exit",
                    "security_id": "SEC-BMYRT",
                    "action_type": "delisting",
                    "effective_date": "2019-11-21",
                    "ex_date": "2019-11-21",
                }
            ]
        )

        selected = provider._actions_in_symbol_intervals(actions)

        self.assertEqual(
            selected[["event_id", "symbol"]].to_dict("records"),
            [
                {
                    "event_id": "bmyrt-first-session-exit",
                    "symbol": "BMYRT",
                }
            ],
        )

    def test_missing_ticker_predecessor_does_not_self_map_to_successor(self):
        provider = ParquetMarketDataProvider("unused")
        provider._resolved_symbol_history = pd.DataFrame(
            [
                {
                    "security_id": "SEC-ISSUER",
                    "symbol": "NEW",
                    "_start": pd.Timestamp("2020-03-02"),
                    "_end": pd.NaT,
                }
            ]
        )
        actions = pd.DataFrame(
            [
                {
                    "event_id": "rename-with-missing-old-history",
                    "security_id": "SEC-ISSUER",
                    "action_type": "ticker_change",
                    "effective_date": "2020-03-02",
                    "ex_date": "2020-03-02",
                    "new_security_id": "SEC-ISSUER",
                    "new_symbol": "NEW",
                }
            ]
        )

        selected = provider._actions_in_symbol_intervals(actions)

        self.assertTrue(selected.empty)

    def test_same_day_split_precedes_ticker_change_on_the_old_alias(self):
        provider = ParquetMarketDataProvider("unused")
        provider._resolved_symbol_history = pd.DataFrame(
            [
                {
                    "security_id": "SEC-ISSUER",
                    "symbol": "OLD",
                    "_start": pd.Timestamp("2015-01-01"),
                    "_end": pd.Timestamp("2020-03-01"),
                },
                {
                    "security_id": "SEC-ISSUER",
                    "symbol": "NEW",
                    "_start": pd.Timestamp("2020-03-02"),
                    "_end": pd.NaT,
                },
            ]
        )
        actions = pd.DataFrame(
            [
                {
                    "event_id": "same-day-split",
                    "security_id": "SEC-ISSUER",
                    "action_type": "split",
                    "effective_date": "2020-03-02",
                    "ex_date": "2020-03-02",
                },
                {
                    "event_id": "same-day-rename",
                    "security_id": "SEC-ISSUER",
                    "action_type": "ticker_change",
                    "effective_date": "2020-03-02",
                    "ex_date": "2020-03-02",
                    "new_security_id": "SEC-ISSUER",
                    "new_symbol": "NEW",
                },
            ]
        )

        selected = provider._actions_in_symbol_intervals(actions)

        self.assertEqual(
            selected.set_index("event_id")["symbol"].to_dict(),
            {
                "same-day-split": "OLD",
                "same-day-rename": "OLD",
            },
        )

    def test_distinct_id_ticker_successor_entitlements_use_predecessor(self):
        provider = ParquetMarketDataProvider("unused")
        provider._resolved_symbol_history = pd.DataFrame(
            [
                {
                    "security_id": "SEC-LB",
                    "symbol": "LB",
                    "_start": pd.Timestamp("2015-01-01"),
                    "_end": pd.Timestamp("2021-08-02"),
                },
                {
                    "security_id": "SEC-BBWI",
                    "symbol": "BBWI",
                    "_start": pd.Timestamp("2021-08-03"),
                    "_end": pd.NaT,
                },
            ]
        )
        actions = pd.DataFrame(
            [
                {
                    "event_id": "lb-to-bbwi",
                    "security_id": "SEC-LB",
                    "action_type": "ticker_change",
                    "effective_date": "2021-08-03",
                    "ex_date": "2021-08-03",
                    "new_security_id": "SEC-BBWI",
                    "new_symbol": "BBWI",
                },
                {
                    "event_id": "bbwi-split",
                    "security_id": "SEC-BBWI",
                    "action_type": "split",
                    "effective_date": "2021-08-03",
                    "ex_date": "2021-08-03",
                    "ratio": 1.237,
                },
                {
                    "event_id": "bbwi-cash",
                    "security_id": "SEC-BBWI",
                    "action_type": "cash_dividend",
                    "effective_date": "2021-08-03",
                    "ex_date": "2021-08-03",
                    "cash_amount": 18.96321,
                },
            ]
        )

        selected = provider._actions_in_symbol_intervals(actions)

        self.assertEqual(
            selected.set_index("event_id")["symbol"].to_dict(),
            {
                "lb-to-bbwi": "LB",
                "bbwi-split": "LB",
                "bbwi-cash": "LB",
            },
        )

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

    def test_top_level_loader_does_not_sync_when_local_release_is_stale(self):
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
                    "supertrend_quant.market_store.ingest.DailyDataSynchronizer"
                ) as synchronizer_type,
                patch.object(
                    LocalDatasetRepository,
                    "current_release",
                    return_value=(
                        SimpleNamespace(completed_session="2026-07-14"),
                        Path(directory) / "releases" / "current.json",
                    ),
                ),
                patch.object(ParquetMarketDataProvider, "load", return_value=loaded),
            ):
                result = load_configured_market_data(config, ["AAA"])

            self.assertIs(result, loaded)
            synchronizer_type.assert_not_called()

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

    def test_equal_remote_pointer_backfills_missing_parent_manifest(self):
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
                "daily_price_raw",
                row,
                completed_session="2026-07-14",
                version="parent-v1",
            )
            row2 = row.copy()
            row2["session"] = "2026-07-15"
            repository.append_frame(
                "daily_price_raw",
                row2,
                completed_session="2026-07-15",
                version="child-v1",
            )
            repository.commit_release(
                "2026-07-15",
                {"daily_price_raw": "child-v1"},
                quality="valid",
            )
            publish_repository(repository, remote, ("daily_price_raw",))
            missing = (
                root
                / "remote"
                / "datasets"
                / "daily_price_raw"
                / "versions"
                / "parent-v1"
                / "manifest.json"
            )
            missing.unlink()

            repeated = publish_repository(
                repository, remote, ("daily_price_raw",)
            )

            self.assertTrue(missing.is_file())
            self.assertFalse(repeated[0].conflict)
            self.assertIn("lineage reconciled", repeated[0].detail)

    def test_divergent_publish_repairs_missing_shared_parent_manifest(self):
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
                "daily_price_raw",
                row,
                completed_session="2026-07-14",
                version="shared-v1",
            )
            first_repo.commit_release(
                "2026-07-14",
                {"daily_price_raw": "shared-v1"},
                quality="valid",
            )
            publish_repository(first_repo, remote, ("daily_price_raw",))
            DatasetCache(root / "writer-a", remote).sync_release()
            DatasetCache(root / "writer-b", remote).sync_release()
            writer_a = LocalDatasetRepository(root / "writer-a")
            writer_b = LocalDatasetRepository(root / "writer-b")
            row_a = row.copy()
            row_a["session"] = "2026-07-15"
            writer_a.append_frame(
                "daily_price_raw",
                row_a,
                completed_session="2026-07-15",
                version="writer-a-v2",
            )
            writer_a.commit_release(
                "2026-07-15",
                {"daily_price_raw": "writer-a-v2"},
                quality="valid",
            )
            row_b = row.copy()
            row_b["session"] = "2026-07-16"
            writer_b.append_frame(
                "daily_price_raw",
                row_b,
                completed_session="2026-07-16",
                version="writer-b-v2",
            )
            writer_b.commit_release(
                "2026-07-16",
                {"daily_price_raw": "writer-b-v2"},
                quality="valid",
            )
            publish_repository(writer_a, remote, ("daily_price_raw",))
            missing = (
                root
                / "remote"
                / "datasets"
                / "daily_price_raw"
                / "versions"
                / "shared-v1"
                / "manifest.json"
            )
            missing.unlink()

            repaired = publish_repository(writer_b, remote, ("daily_price_raw",))

            self.assertTrue(missing.is_file())
            self.assertTrue(repaired[0].published)
            self.assertFalse(repaired[0].conflict)
            self.assertIn("rebased", repaired[0].detail)

    def test_explicit_remote_release_version_can_be_superseded_without_deleting_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = LocalObjectStore(root / "remote")
            repository = LocalDatasetRepository(root / "local")
            rows = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": "SEC-A",
                            "session": session,
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.0,
                            "close": 10.0,
                            "volume": 100.0,
                            "currency": "USD",
                        }
                    )
                    for session in ("2026-07-14", "2026-07-15")
                ]
            )
            repository.write_frame(
                "daily_price_raw",
                rows,
                completed_session="2026-07-15",
                version="old-release-v1",
            )
            repository.commit_release(
                "2026-07-15",
                {"daily_price_raw": "old-release-v1"},
                quality="valid",
            )
            publish_repository(repository, remote, ("daily_price_raw",))
            old_manifest = (
                root
                / "remote"
                / "datasets"
                / "daily_price_raw"
                / "versions"
                / "old-release-v1"
                / "manifest.json"
            )
            repository.write_frame(
                "daily_price_raw",
                rows.iloc[[1]].copy(),
                completed_session="2026-07-15",
                version="validated-replacement-v2",
            )
            repository.commit_release(
                "2026-07-15",
                {"daily_price_raw": "validated-replacement-v2"},
                quality="valid",
            )

            result = publish_repository(
                repository,
                remote,
                ("daily_price_raw",),
                supersede_versions={"daily_price_raw": "old-release-v1"},
            )

            self.assertTrue(result[0].published)
            self.assertFalse(result[0].conflict)
            self.assertIn("superseded remote release version", result[0].detail)
            self.assertTrue(old_manifest.is_file())
            DatasetCache(root / "reader", remote).sync_release()
            reader = LocalDatasetRepository(root / "reader")
            self.assertEqual(len(reader.read_frame("daily_price_raw")), 1)

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
            self.assertNotIn("year", chained.columns)
            self.assertNotIn("month", chained.columns)
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
                    "symbol_history": 2,
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

    def test_static_provider_does_not_stale_block_an_exit_only_action_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = LocalDatasetRepository(root)
            history = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": security_id,
                            "symbol": symbol,
                            "exchange": "XNYS",
                            "effective_from": start,
                            "effective_to": end,
                        }
                    )
                    for security_id, symbol, start, end in (
                        ("SEC-PARENT", "PARENT", "2026-01-02", ""),
                        ("SEC-CHILD", "CHILD", "2026-01-05", "2026-01-05"),
                        ("SEC-QQQ", "QQQ", "2026-01-02", ""),
                    )
                ]
            )
            repository.write_frame(
                "symbol_history",
                history,
                completed_session="2026-01-06",
                version="symbols-v1",
            )
            prices = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": security_id,
                            "session": session,
                            "open": close,
                            "high": close,
                            "low": close,
                            "close": close,
                            "volume": 1_000.0,
                            "currency": "USD",
                        }
                    )
                    for security_id, points in {
                        "SEC-PARENT": (("2026-01-02", 100.0), ("2026-01-06", 101.0)),
                        "SEC-CHILD": (("2026-01-05", 2.30),),
                        "SEC-QQQ": (("2026-01-02", 500.0), ("2026-01-06", 501.0)),
                    }.items()
                    for session, close in points
                ]
            )
            repository.write_frame(
                "daily_price_raw",
                prices,
                completed_session="2026-01-06",
                version="prices-v1",
            )
            actions = pd.DataFrame(
                [
                    _source(
                        {
                            "event_id": "parent-spin-exit-only-child",
                            "security_id": "SEC-PARENT",
                            "action_type": "spinoff",
                            "effective_date": "2026-01-05",
                            "ex_date": "2026-01-05",
                            "announcement_date": "",
                            "record_date": "",
                            "payment_date": "2026-01-05",
                            "cash_amount": None,
                            "ratio": 1.0,
                            "currency": "USD",
                            "new_security_id": "SEC-CHILD",
                            "new_symbol": "CHILD",
                            "official": True,
                            "source_url": "https://example.test/exit-only-child",
                            "source_kind": "fixture",
                            "metadata": {"cost_basis_fraction": 0.02},
                        }
                    )
                ]
            )
            repository.write_frame(
                "corporate_actions",
                actions,
                completed_session="2026-01-06",
                version="actions-v1",
            )
            factors = build_adjustment_factors(
                prices,
                actions,
                source_version="prices-v1+actions-v1",
            )
            repository.write_frame(
                "adjustment_factors",
                factors,
                completed_session="2026-01-06",
                version="factors-v1",
            )
            config = parse_config(
                {
                    "strategy": {"name": "test", "type": "equal", "params": {}},
                    "scoring": {
                        "type": "relative_strength",
                        "params": {"lookback_bars": 1},
                    },
                    "market": "US",
                    "universe": {"source": "symbols", "symbols": ["PARENT"]},
                    "period": "max",
                    "data_store": {
                        "provider": "parquet",
                        "local_cache_dir": str(root),
                        "price_mode": "total_return_adjusted",
                    },
                }
            )

            market_data = ParquetMarketDataProvider(root).load(config, ["PARENT"])

            self.assertEqual(set(market_data.bars), {"PARENT", "CHILD"})
            self.assertEqual(market_data.entry_symbols, ("PARENT",))
            self.assertEqual(market_data.data_quality, "valid")
            self.assertFalse(
                any(
                    "Selected universe is incomplete" in warning
                    and "CHILD" in warning
                    for warning in market_data.warnings
                )
            )

    def test_provider_recursively_loads_action_successors_without_making_them_entry_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = LocalDatasetRepository(root)
            history = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": security_id,
                            "symbol": symbol,
                            "exchange": "XNAS",
                            "effective_from": effective_from,
                            "effective_to": effective_to,
                        }
                    )
                    for security_id, symbol, effective_from, effective_to in (
                        ("SEC-PARENT", "OLD", "2000-01-01", "2026-01-02"),
                        ("SEC-PARENT", "NEW", "2026-01-05", ""),
                        ("SEC-CHILD", "CHILD", "2026-01-05", ""),
                        ("SEC-GRAND", "GRAND", "2026-01-06", ""),
                        ("SEC-QQQ", "QQQ", "1999-03-10", ""),
                    )
                ]
            )
            repository.write_frame(
                "symbol_history",
                history,
                completed_session="2026-01-07",
                version="symbols-v1",
            )
            sessions = ("2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07")
            available = {
                "SEC-PARENT": sessions,
                "SEC-CHILD": sessions[1:],
                "SEC-GRAND": sessions[2:],
                "SEC-QQQ": sessions,
            }
            prices = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": security_id,
                            "session": session,
                            "open": price,
                            "high": price,
                            "low": price,
                            "close": price,
                            "volume": 1000.0,
                            "currency": "USD",
                        }
                    )
                    for security_id, security_sessions in available.items()
                    for session in security_sessions
                    for price in (100.0 if security_id == "SEC-PARENT" else 20.0,)
                ]
            )
            repository.write_frame(
                "daily_price_raw",
                prices,
                completed_session="2026-01-07",
                version="prices-v1",
            )

            def action(
                event_id: str,
                security_id: str,
                action_type: str,
                effective_date: str,
                new_security_id: str,
                new_symbol: str,
                ratio: float,
            ) -> dict[str, object]:
                return _source(
                    {
                        "event_id": event_id,
                        "security_id": security_id,
                        "action_type": action_type,
                        "effective_date": effective_date,
                        "ex_date": effective_date,
                        "announcement_date": "",
                        "record_date": "",
                        "payment_date": "",
                        "cash_amount": None,
                        "ratio": ratio,
                        "currency": "USD",
                        "new_security_id": new_security_id,
                        "new_symbol": new_symbol,
                        "official": True,
                        "source_url": "https://example.test/action",
                        "source_kind": "fixture",
                        "metadata": (
                            {"cost_basis_fraction": 0.4}
                            if action_type == "spinoff"
                            else {}
                        ),
                    }
                )

            actions = pd.DataFrame(
                [
                    action(
                        "parent-spin",
                        "SEC-PARENT",
                        "spinoff",
                        "2026-01-05",
                        "SEC-CHILD",
                        "CHILD",
                        1.0,
                    ),
                    action(
                        "parent-rename",
                        "SEC-PARENT",
                        "ticker_change",
                        "2026-01-05",
                        "SEC-PARENT",
                        "NEW",
                        1.0,
                    ),
                    action(
                        "child-spin",
                        "SEC-CHILD",
                        "spinoff",
                        "2026-01-06",
                        "SEC-GRAND",
                        "GRAND",
                        0.5,
                    ),
                ]
            )
            repository.write_frame(
                "corporate_actions",
                actions,
                completed_session="2026-01-07",
                version="actions-v1",
            )
            factors = build_adjustment_factors(
                prices,
                actions,
                source_version="prices-v1+actions-v1",
            )
            repository.write_frame(
                "adjustment_factors",
                factors,
                completed_session="2026-01-07",
                version="factors-v1",
            )
            config = parse_config(
                {
                    "strategy": {"name": "test", "type": "equal", "params": {}},
                    "scoring": {
                        "type": "relative_strength",
                        "params": {"lookback_bars": 1},
                    },
                    "market": "US",
                    "universe": {"source": "symbols", "symbols": ["OLD"]},
                    "period": "max",
                    "data_store": {
                        "provider": "parquet",
                        "local_cache_dir": str(root),
                        "price_mode": "total_return_adjusted",
                    },
                }
            )

            market_data = ParquetMarketDataProvider(root).load(config, ["OLD"])

            self.assertEqual(
                set(market_data.bars),
                {"OLD", "NEW", "CHILD", "GRAND"},
            )
            self.assertEqual(set(market_data.execution_bars), set(market_data.bars))
            self.assertEqual(market_data.entry_symbols, ("OLD",))
            actions_by_id = {
                item["event_id"]: item for item in market_data.corporate_actions
            }
            self.assertEqual(set(actions_by_id), {"parent-spin", "parent-rename", "child-spin"})
            self.assertEqual(actions_by_id["parent-spin"]["symbol"], "OLD")
            self.assertEqual(actions_by_id["parent-rename"]["symbol"], "OLD")
            self.assertEqual(actions_by_id["child-spin"]["symbol"], "CHILD")

            # A static configuration may name only the post-change ticker.
            # Its exact predecessor transition must still expose the same-day
            # spin-off entitlement, without making the old alias buyable.
            current_market_data = ParquetMarketDataProvider(root).load(config, ["NEW"])
            self.assertEqual(
                set(current_market_data.bars),
                {"NEW", "CHILD", "GRAND"},
            )
            self.assertNotIn("OLD", current_market_data.bars)
            self.assertEqual(current_market_data.entry_symbols, ("NEW",))
            current_actions = {
                item["event_id"]: item
                for item in current_market_data.corporate_actions
            }
            self.assertEqual(
                set(current_actions),
                {"parent-spin", "parent-rename", "child-spin"},
            )
            self.assertEqual(current_actions["parent-spin"]["symbol"], "NEW")
            self.assertEqual(current_actions["parent-rename"]["symbol"], "NEW")
            self.assertEqual(current_actions["parent-rename"]["new_symbol"], "NEW")
            ledger = PortfolioLedger(
                cash=0.0,
                positions={"NEW": Position("NEW", 10.0, 100.0)},
            )
            ledger.apply_actions(
                current_market_data.corporate_actions,
                through="2026-01-05",
            )
            self.assertEqual(ledger.positions["NEW"].avg_price, 60.0)
            self.assertEqual(ledger.positions["CHILD"].quantity, 10.0)
            self.assertEqual(ledger.positions["CHILD"].avg_price, 40.0)

            # Dynamic entry eligibility changes at the alias/index boundaries,
            # but the same issuer's new ticker keeps prior-alias feature
            # warm-up.  A requested spin child also needs its action-day bar,
            # one session before it becomes an index member.
            index_config = parse_config(
                {
                    "strategy": {"name": "test", "type": "equal", "params": {}},
                    "scoring": {
                        "type": "relative_strength",
                        "params": {"lookback_bars": 1},
                    },
                    "market": "US",
                    "universe": {
                        "source": "index_events",
                        "profiles": {"US": ["nasdaq100"]},
                    },
                    "period": "max",
                    "data_store": {
                        "provider": "parquet",
                        "local_cache_dir": str(root),
                        "price_mode": "total_return_adjusted",
                    },
                }
            )
            dynamic_schedule = (
                {
                    "effective_date": "2026-01-02",
                    "members": [
                        {"symbol": "OLD", "security_id": "SEC-PARENT"}
                    ],
                },
                {
                    "effective_date": "2026-01-05",
                    "members": [
                        {"symbol": "NEW", "security_id": "SEC-PARENT"}
                    ],
                },
                {
                    "effective_date": "2026-01-06",
                    "members": [
                        {"symbol": "NEW", "security_id": "SEC-PARENT"},
                        {"symbol": "CHILD", "security_id": "SEC-CHILD"},
                    ],
                },
            )
            dynamic_market_data = ParquetMarketDataProvider(root).load(
                index_config,
                ["OLD", "NEW", "CHILD"],
                universe_schedule=dynamic_schedule,
            )
            new_frame = dynamic_market_data.bars["NEW"]
            self.assertIn(pd.Timestamp("2026-01-02"), new_frame.index)
            self.assertNotIn("IdentitySegment", new_frame)
            featured = calculate_supertrend(new_frame, period=2, multiplier=1.0)
            self.assertTrue(
                pd.notna(featured.loc[pd.Timestamp("2026-01-05"), "ATR"])
            )
            scored = RelativeStrengthScorer(
                {"lookback_bars": 1},
                "US",
            ).add_scores(
                {"NEW": new_frame},
                dynamic_market_data.benchmark,
            )["NEW"]
            self.assertTrue(
                pd.notna(scored.loc[pd.Timestamp("2026-01-05"), "Score"])
            )
            self.assertEqual(
                dynamic_market_data.execution_bars["CHILD"].index.min(),
                pd.Timestamp("2026-01-05"),
            )

            missing_child_factor = factors.loc[
                ~(
                    factors["security_id"].astype(str).eq("SEC-CHILD")
                    & pd.to_datetime(factors["session"]).eq(
                        pd.Timestamp("2026-01-05")
                    )
                )
            ].copy()
            repository.write_frame(
                "adjustment_factors",
                missing_child_factor,
                completed_session="2026-01-07",
                version="factors-missing-child-session",
            )
            with self.assertRaisesRegex(
                ValueError,
                "SEC-CHILD/2026-01-05",
            ):
                ParquetMarketDataProvider(root).load(
                    index_config,
                    ["OLD", "NEW", "CHILD"],
                    universe_schedule=dynamic_schedule,
                )

    def test_reused_ticker_uses_scheduled_issuer_intervals_and_resets_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = LocalDatasetRepository(root)
            history = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": security_id,
                            "symbol": symbol,
                            "exchange": "XNAS",
                            "effective_from": start,
                            "effective_to": end,
                        }
                    )
                    for security_id, symbol, start, end in (
                        ("SEC-DUP-OLD", "DUP", "2000-01-01", "2020-01-03"),
                        ("SEC-DUP-NEW", "DUP", "2020-01-01", ""),
                        ("SEC-QQQ", "QQQ", "1999-03-10", ""),
                    )
                ]
            )
            repository.write_frame(
                "symbol_history",
                history,
                completed_session="2026-01-05",
                version="symbols-v1",
            )
            price_points = {
                "SEC-DUP-OLD": (
                    ("2009-01-02", 10.0),
                    ("2009-01-05", 11.0),
                    ("2020-01-02", 12.0),
                ),
                "SEC-DUP-NEW": (
                    ("2020-01-02", 98.0),
                    ("2025-12-31", 99.0),
                    ("2026-01-02", 100.0),
                    ("2026-01-05", 101.0),
                ),
                "SEC-QQQ": (
                    ("2009-01-02", 40.0),
                    ("2009-01-05", 41.0),
                    ("2025-12-31", 49.0),
                    ("2026-01-02", 50.0),
                    ("2026-01-05", 51.0),
                ),
            }
            prices = pd.DataFrame(
                [
                    _source(
                        {
                            "security_id": security_id,
                            "session": session,
                            "open": close,
                            "high": close,
                            "low": close,
                            "close": close,
                            "volume": 1000.0,
                            "currency": "USD",
                        }
                    )
                    for security_id, points in price_points.items()
                    for session, close in points
                ]
            )
            repository.write_frame(
                "daily_price_raw",
                prices,
                completed_session="2026-01-05",
                version="prices-v1",
            )
            actions = pd.DataFrame(
                [
                    _source(
                        {
                            "event_id": "old-issuer-after-removal-dividend",
                            "security_id": "SEC-DUP-OLD",
                            "action_type": "cash_dividend",
                            "effective_date": "2010-06-01",
                            "ex_date": "2010-06-01",
                            "announcement_date": "",
                            "record_date": "",
                            "payment_date": "",
                            "cash_amount": 1.0,
                            "ratio": None,
                            "currency": "USD",
                            "new_security_id": "",
                            "new_symbol": "",
                            "official": True,
                            "source_url": "https://example.test/old-dividend",
                            "source_kind": "fixture",
                        }
                    )
                ]
            )
            repository.write_frame(
                "corporate_actions",
                actions,
                completed_session="2026-01-05",
                version="actions-v1",
            )
            factors = build_adjustment_factors(
                prices,
                actions,
                source_version="prices-v1",
            )
            repository.write_frame(
                "adjustment_factors",
                factors,
                completed_session="2026-01-05",
                version="factors-v1",
            )
            static_config = parse_config(
                {
                    "strategy": {"name": "test", "type": "equal", "params": {}},
                    "scoring": {
                        "type": "relative_strength",
                        "params": {"lookback_bars": 1},
                    },
                    "market": "US",
                    "universe": {"source": "symbols", "symbols": ["DUP"]},
                    "period": "max",
                    "data_store": {
                        "provider": "parquet",
                        "local_cache_dir": str(root),
                        "price_mode": "total_return_adjusted",
                    },
                }
            )
            with self.assertRaisesRegex(ValueError, "Ambiguous ticker identity"):
                ParquetMarketDataProvider(root).load(static_config, ["DUP"])

            index_config = parse_config(
                {
                    "strategy": {"name": "test", "type": "equal", "params": {}},
                    "scoring": {
                        "type": "relative_strength",
                        "params": {"lookback_bars": 1},
                    },
                    "market": "US",
                    "universe": {
                        "source": "index_events",
                        "profiles": {"US": ["nasdaq100"]},
                    },
                    "period": "max",
                    "data_store": {
                        "provider": "parquet",
                        "local_cache_dir": str(root),
                        "price_mode": "total_return_adjusted",
                    },
                }
            )
            schedule = (
                {
                    "effective_date": "2009-01-02",
                    "members": [
                        {"symbol": "DUP", "security_id": "SEC-DUP-OLD"}
                    ],
                },
                {"effective_date": "2010-01-01", "members": []},
                {
                    "effective_date": "2026-01-02",
                    "members": [
                        {"symbol": "DUP", "security_id": "SEC-DUP-NEW"}
                    ],
                },
            )
            market_data = ParquetMarketDataProvider(root).load(
                index_config,
                ["DUP"],
                universe_schedule=schedule,
            )

            frame = market_data.bars["DUP"]
            self.assertEqual(
                [action["event_id"] for action in market_data.corporate_actions],
                ["old-issuer-after-removal-dividend"],
            )
            self.assertEqual(
                frame["IdentitySegment"].tolist(),
                [
                    "SEC-DUP-OLD",
                    "SEC-DUP-OLD",
                    "SEC-DUP-NEW",
                    "SEC-DUP-NEW",
                ],
            )
            self.assertNotIn(pd.Timestamp("2020-01-02"), frame.index)
            self.assertNotIn(pd.Timestamp("2025-12-31"), frame.index)
            featured = calculate_supertrend(frame, period=2, multiplier=1.0)
            self.assertTrue(pd.isna(featured.loc[pd.Timestamp("2026-01-02"), "ATR"]))
            self.assertTrue(pd.notna(featured.loc[pd.Timestamp("2026-01-05"), "ATR"]))
            scored = RelativeStrengthScorer(
                {"lookback_bars": 1},
                "US",
            ).add_scores({"DUP": frame}, market_data.benchmark)["DUP"]
            self.assertTrue(pd.isna(scored.loc[pd.Timestamp("2026-01-02"), "Score"]))
            self.assertTrue(
                pd.notna(scored.loc[pd.Timestamp("2026-01-05"), "Score"]),
                scored.to_string(),
            )
            overlapping = pd.DataFrame(
                [
                    {
                        "security_id": security_id,
                        "session": "2026-01-02",
                        "open": close,
                        "high": close,
                        "low": close,
                        "close": close,
                        "volume": 1.0,
                    }
                    for security_id, close in (
                        ("SEC-DUP-OLD", 10.0),
                        ("SEC-DUP-NEW", 100.0),
                    )
                ]
            )
            with self.assertRaisesRegex(ValueError, "price histories overlap"):
                _frames_by_identity_intervals(
                    overlapping,
                    ("DUP",),
                    {
                        "DUP": (
                            ("SEC-DUP-OLD", None, None),
                            ("SEC-DUP-NEW", None, None),
                        )
                    },
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
                            "effective_from": effective_from,
                            "effective_to": effective_to,
                        }
                    )
                    for security_id, symbol, effective_from, effective_to in (
                        ("SEC-A", "AAA", "2000-01-01", ""),
                        ("SEC-B", "BBB", "2000-01-01", "2026-07-15"),
                        ("SEC-B", "BBX", "2026-07-16", ""),
                        ("SEC-C", "CCC", "2000-01-01", ""),
                    )
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

            resolved = resolve_universe(config, as_of=pd.Timestamp("2026-07-16").date())

            self.assertEqual(resolved.eligible_symbols, ("BBX", "CCC"))
            self.assertEqual(resolved.member_for("CCC").security_id, "SEC-C")
            self.assertEqual(
                [entry.effective_date for entry in resolved.schedule],
                ["2026-07-14", "2026-07-15", "2026-07-16"],
            )
            self.assertEqual(resolved.schedule[0].symbols, ("AAA", "BBB"))
            self.assertEqual(resolved.schedule[1].symbols, ("BBB", "CCC"))
            self.assertEqual(resolved.schedule[2].symbols, ("BBX", "CCC"))


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
