from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from supertrend_quant.config import load_split_config
from supertrend_quant.data import _yf_download, extract_ohlc
from supertrend_quant.research.export import config_to_split_dicts, strict_split_roundtrip
from supertrend_quant.research.data_resolver import data_request_key
from supertrend_quant.results import save_backtest_result
from supertrend_quant.universe import (
    UniverseMember,
    available_universe_profiles,
    register_universe_provider,
    resolve_universe,
)


STRATEGY = "configs/strategies/simple_supertrend.yaml"
SIMULATION = "configs/runtimes/simulation.yaml"


def member(symbol: str, *, profile: str = "sp500", name: str = "", security_type: str = "STOCK") -> UniverseMember:
    return UniverseMember(
        symbol=symbol,
        market="US",
        exchange="US",
        name=name,
        security_type=security_type,
        yfinance_symbol=symbol.replace(".", "-"),
        profiles=(profile,),
    )


def history(close: float = 10.0, volume: float = 2_000_000.0, bars: int = 130) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [close] * bars, "Volume": [volume] * bars},
        index=pd.date_range("2025-01-02", periods=bars, freq="B"),
    )


def profile_config(tmp: str, profiles=("sp500",), *, filters_enabled: bool = False):
    base = load_split_config(STRATEGY, SIMULATION)
    filters = replace(base.universe.filters, enabled=filters_enabled)
    universe = replace(
        base.universe,
        source="profiles",
        profiles={"US": tuple(profiles)},
        snapshot_dir=str(Path(tmp) / "snapshots"),
        filters=filters,
    )
    return replace(base, market="US", universe=universe, symbols=())


def kr_profile_config(tmp: str, profiles=("kospi200",), *, filters_enabled: bool = True):
    base = load_split_config(STRATEGY, "configs/runtimes/research_kr.yaml")
    universe = replace(
        base.universe,
        source="profiles",
        profiles={"KR": tuple(profiles)},
        snapshot_dir=str(Path(tmp) / "snapshots"),
        filters=replace(base.universe.filters, enabled=filters_enabled),
    )
    return replace(base, market="KR", universe=universe, symbols=())


class UniverseRegistryAndResolutionTest(unittest.TestCase):
    def test_builtin_profiles_are_registered_and_duplicates_rejected(self):
        self.assertEqual(
            available_universe_profiles(),
            ("dow30", "kosdaq150", "kospi200", "nasdaq100", "sp500"),
        )
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_universe_provider("sp500", lambda as_of: ())
        with self.assertRaisesRegex(ValueError, "Unknown universe profile"):
            register_universe_provider("missing", lambda as_of: ())

    def test_same_market_profiles_union_deduplicates_and_uses_spy(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = profile_config(tmp, ("sp500", "dow30"))
            providers = {
                "sp500": lambda as_of: (member("BBB"), member("AAA")),
                "dow30": lambda as_of: (member("AAA", profile="dow30"), member("CCC", profile="dow30")),
            }
            with patch.dict("supertrend_quant.universe._PROVIDERS", providers, clear=False):
                resolved = resolve_universe(config, as_of=date(2026, 7, 14))

            self.assertEqual(resolved.eligible_symbols, ("AAA", "BBB", "CCC"))
            self.assertEqual(resolved.benchmark_for("AAA"), "SPY")
            self.assertEqual(resolved.member_for("AAA").profiles, ("dow30", "sp500"))

    def test_single_profile_benchmarks_are_profile_specific(self):
        expectations = {"nasdaq100": "QQQ", "sp500": "SPY", "dow30": "DIA"}
        for profile, expected in expectations.items():
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                config = profile_config(tmp, (profile,))
                provider = lambda as_of, selected=profile: (member("AAA", profile=selected),)
                with patch.dict("supertrend_quant.universe._PROVIDERS", {profile: provider}, clear=False):
                    resolved = resolve_universe(config, as_of=date(2026, 7, 14))
                self.assertEqual(resolved.benchmark_for("AAA"), expected)

    def test_kr_union_keeps_board_specific_yahoo_symbols_and_benchmarks(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = kr_profile_config(tmp, ("kospi200", "kosdaq150"), filters_enabled=False)
            kospi = UniverseMember("005930", "KR", "KOSPI", profiles=("kospi200",))
            kosdaq = UniverseMember("091990", "KR", "KOSDAQ", profiles=("kosdaq150",))
            with patch.dict(
                "supertrend_quant.universe._PROVIDERS",
                {
                    "kospi200": lambda as_of: (kospi,),
                    "kosdaq150": lambda as_of: (kosdaq,),
                },
                clear=False,
            ):
                resolved = resolve_universe(config, as_of=date(2026, 7, 14))

            self.assertEqual(resolved.yfinance_symbol_for("005930"), "005930.KS")
            self.assertEqual(resolved.yfinance_symbol_for("091990"), "091990.KQ")
            self.assertEqual(resolved.benchmark_for("005930"), "^KS11")
            self.assertEqual(resolved.benchmark_for("091990"), "^KQ11")

    def test_daily_snapshot_is_reused_without_calling_provider_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = profile_config(tmp)
            calls = []

            def provider(as_of):
                calls.append(as_of)
                return (member("AAA"),)

            with patch.dict("supertrend_quant.universe._PROVIDERS", {"sp500": provider}, clear=False):
                first = resolve_universe(config, as_of=date(2026, 7, 14))
                second = resolve_universe(config, as_of=date(2026, 7, 14))

            self.assertEqual(len(calls), 1)
            self.assertEqual(first.snapshot.selection_hash, second.snapshot.selection_hash)
            self.assertEqual(first.eligible_symbols, second.eligible_symbols)

    def test_refresh_failure_blocks_entries_in_live_and_fails_backtest(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = profile_config(tmp)
            with patch.dict(
                "supertrend_quant.universe._PROVIDERS",
                {"sp500": lambda as_of: (member("AAA"),)},
                clear=False,
            ):
                resolve_universe(config, as_of=date(2026, 7, 14))

            def failed(as_of):
                raise RuntimeError("provider down")

            with patch.dict("supertrend_quant.universe._PROVIDERS", {"sp500": failed}, clear=False):
                fallback = resolve_universe(
                    config,
                    as_of=date(2026, 7, 14),
                    force_refresh=True,
                    held_symbols=("AAA",),
                    mode="live",
                )
            self.assertFalse(fallback.entries_allowed)
            self.assertEqual(fallback.symbols, ("AAA",))
            self.assertIn("provider down", fallback.refresh_error)

        with tempfile.TemporaryDirectory() as empty_tmp:
            config = profile_config(empty_tmp)
            with patch.dict("supertrend_quant.universe._PROVIDERS", {"sp500": failed}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "Universe refresh failed"):
                    resolve_universe(config, as_of=date(2026, 7, 14), mode="backtest")


class UniverseConfigTest(unittest.TestCase):
    def test_active_runtime_defaults_and_balanced_thresholds(self):
        us = load_split_config(STRATEGY, "configs/runtimes/research_us.yaml")
        kr = load_split_config(STRATEGY, "configs/runtimes/research_kr.yaml")
        live = load_split_config(STRATEGY, "configs/runtimes/live_toss.yaml")

        self.assertEqual(us.universe.profiles, {"US": ("sp500",)})
        self.assertEqual(kr.universe.profiles, {"KR": ("kospi200", "kosdaq150")})
        self.assertEqual(us.universe.filters.min_price, {"US": 5.0, "KR": 1_000.0})
        self.assertEqual(
            us.universe.filters.min_avg_turnover,
            {"US": 10_000_000.0, "KR": 1_000_000_000.0},
        )
        self.assertEqual(us.universe.filters.min_history_daily_bars, 120)
        self.assertEqual(live.universe.source, "file")
        self.assertFalse(live.universe.filters.enabled)

    def test_split_export_uses_nested_universe_and_roundtrips(self):
        config = load_split_config(STRATEGY, SIMULATION)
        _, runtime = config_to_split_dicts(config)
        reloaded = strict_split_roundtrip(config)

        self.assertIn("universe", runtime)
        self.assertNotIn("universe_file", runtime)
        self.assertNotIn("symbols", runtime)
        self.assertEqual(reloaded.universe, config.universe)

    def test_research_cache_key_changes_with_universe_filter(self):
        config = load_split_config(STRATEGY, SIMULATION)
        changed = replace(
            config,
            universe=replace(
                config.universe,
                filters=replace(
                    config.universe.filters,
                    min_price={"US": 10.0, "KR": 1_000.0},
                ),
            ),
        )
        self.assertNotEqual(data_request_key(config), data_request_key(changed))

    def test_backtest_result_persists_universe_snapshot_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = profile_config(tmp)
            with patch.dict(
                "supertrend_quant.universe._PROVIDERS",
                {"sp500": lambda as_of: (member("AAA"),)},
                clear=False,
            ):
                resolved = resolve_universe(config, as_of=date(2026, 7, 14))
            result = SimpleNamespace(
                equity=pd.Series([10_000.0, 10_100.0], index=pd.date_range("2026-01-01", periods=2)),
                metrics={"total_return": 0.01},
                trades=[],
                trade_records=(),
                skipped=(),
                universe_snapshot=resolved.snapshot.to_dict(),
            )
            run_dir = save_backtest_result(result, config, Path(tmp) / "results", run_id="universe")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertTrue((run_dir / "universe_snapshot.json").exists())
            self.assertEqual(summary["universe"]["profiles"], ["sp500"])
            self.assertEqual(summary["universe"]["as_of"], "2026-07-14")

    def test_invalid_profile_and_filter_values_are_rejected_during_load(self):
        cases = (
            (
                "unknown profile",
                "profiles:\n    US: [missing]",
                "Unsupported universe profile",
            ),
            (
                "wrong market",
                "profiles:\n    US: [kospi200]",
                "belongs to KR",
            ),
            (
                "negative price",
                "profiles:\n    US: [sp500]\n  filters:\n    min_price:\n      US: -1",
                "non-negative",
            ),
            (
                "zero history",
                "profiles:\n    US: [sp500]\n  filters:\n    min_history_daily_bars: 0",
                "positive integer",
            ),
            (
                "unknown filter",
                "profiles:\n    US: [sp500]\n  filters:\n    mystery: true",
                "Unsupported keys",
            ),
        )
        for label, universe_body, error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                runtime = Path(tmp) / "runtime.yaml"
                runtime.write_text(
                    "name: invalid\n"
                    "market: US\n"
                    "universe:\n"
                    "  source: profiles\n"
                    f"  {universe_body}\n"
                    "data:\n"
                    "  timeframe: 1d\n"
                    "  period: 60d\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, error):
                    load_split_config(STRATEGY, runtime)


class UniverseFilterTest(unittest.TestCase):
    def test_balanced_filters_record_each_rejection_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = profile_config(tmp, filters_enabled=True)
            members = (
                member("PASS"),
                member("LOW"),
                member("ILLIQ"),
                member("NEW"),
                member("STALE"),
                member("ETF", name="Example ETF", security_type="ETF"),
                member("SPAC", name="Example Acquisition Corp"),
                member("PREF", name="Example Preferred"),
            )
            frames = {
                "PASS": history(),
                "LOW": history(close=4.0),
                "ILLIQ": history(volume=100.0),
                "NEW": history(bars=100),
                "STALE": history().iloc[:-1],
                "ETF": history(),
                "SPAC": history(),
                "PREF": history(),
            }
            with patch.dict(
                "supertrend_quant.universe._PROVIDERS",
                {"sp500": lambda as_of: members},
                clear=False,
            ):
                resolved = resolve_universe(
                    config,
                    as_of=date(2026, 7, 14),
                    price_loader=lambda selected, required: frames,
                )

            self.assertEqual(resolved.eligible_symbols, ("PASS",))
            rejected = {item["symbol"]: item["reasons"] for item in resolved.snapshot.rejected}
            self.assertIn("min_price", rejected["LOW"])
            self.assertIn("min_avg_turnover", rejected["ILLIQ"])
            self.assertIn("insufficient_history", rejected["NEW"])
            self.assertIn("suspended_or_stale", rejected["STALE"])
            self.assertIn("etf_etn", rejected["ETF"])
            self.assertIn("spac", rejected["SPAC"])
            self.assertIn("preferred", rejected["PREF"])

    def test_rejected_held_symbol_is_exit_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = profile_config(tmp, filters_enabled=True)
            with patch.dict(
                "supertrend_quant.universe._PROVIDERS",
                {"sp500": lambda as_of: (member("LOW"),)},
                clear=False,
            ):
                resolved = resolve_universe(
                    config,
                    as_of=date(2026, 7, 14),
                    held_symbols=("LOW",),
                    price_loader=lambda selected, required: {"LOW": history(close=4.0)},
                )
            self.assertEqual(resolved.eligible_symbols, ())
            self.assertEqual(resolved.exit_only_symbols, ("LOW",))

    def test_missing_and_infinite_history_are_excluded(self):
        bad = history()
        bad.loc[bad.index[-1], "Close"] = float("inf")
        with tempfile.TemporaryDirectory() as tmp:
            config = profile_config(tmp, filters_enabled=True)
            with patch.dict(
                "supertrend_quant.universe._PROVIDERS",
                {"sp500": lambda as_of: (member("MISSING"), member("INF"))},
                clear=False,
            ):
                resolved = resolve_universe(
                    config,
                    as_of=date(2026, 7, 14),
                    price_loader=lambda selected, required: {"INF": bad},
                )
            self.assertEqual(resolved.eligible_symbols, ())
            reasons = {item["symbol"]: item["reasons"] for item in resolved.snapshot.rejected}
            self.assertIn("missing_history", reasons["MISSING"])
            self.assertTrue(reasons["INF"])

    def test_kr_managed_and_delisting_lists_exclude_new_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = kr_profile_config(tmp)
            members = tuple(
                UniverseMember(symbol, "KR", "KOSPI", yfinance_symbol=f"{symbol}.KS")
                for symbol in ("PASS", "MANAGED", "DELIST")
            )
            frames = {symbol: history(close=50_000.0, volume=100_000.0) for symbol in ("PASS", "MANAGED", "DELIST")}
            with patch.dict(
                "supertrend_quant.universe._PROVIDERS",
                {"kospi200": lambda as_of: members},
                clear=False,
            ):
                resolved = resolve_universe(
                    config,
                    as_of=date(2026, 7, 14),
                    price_loader=lambda selected, required: frames,
                    status_loader=lambda as_of: ({"MANAGED"}, {"DELIST"}),
                )

            self.assertEqual(resolved.eligible_symbols, ("PASS",))
            reasons = {item["symbol"]: item["reasons"] for item in resolved.snapshot.rejected}
            self.assertIn("managed", reasons["MANAGED"])
            self.assertIn("delisting", reasons["DELIST"])


class YahooBatchDownloadTest(unittest.TestCase):
    def test_downloads_in_batches_of_one_hundred_and_merges(self):
        class FakeYF:
            def __init__(self):
                self.calls = []

            def download(self, tickers, **kwargs):
                selected = list(tickers)
                self.calls.append(selected)
                index = pd.date_range("2026-01-01", periods=2, freq="D")
                columns = pd.MultiIndex.from_product(
                    [selected, ["Open", "High", "Low", "Close"]]
                )
                return pd.DataFrame(1.0, index=index, columns=columns)

        yf = FakeYF()
        tickers = [f"S{index:03d}" for index in range(205)]
        raw = _yf_download(yf, tickers, "60d", "1d")

        self.assertEqual([len(call) for call in yf.calls], [100, 100, 5])
        self.assertFalse(extract_ohlc(raw, "S204").empty)


if __name__ == "__main__":
    unittest.main()
