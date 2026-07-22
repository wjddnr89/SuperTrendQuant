import tempfile
import unittest
from pathlib import Path

import pandas as pd

from supertrend_quant.brokers import PaperBroker
from supertrend_quant.config import benchmark_for_symbol, load_split_config
from supertrend_quant.portfolio import AccountSnapshot
from supertrend_quant.strategies import (
    _trend_down_confirmed,
    available_strategies,
    build_order_plan,
    create_strategy,
    register_strategy,
)


UNIFIED_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = UNIFIED_ROOT / "configs"
PACKAGE_ROOT = UNIFIED_ROOT / "src" / "supertrend_quant"


class ConfigAndStrategyTest(unittest.TestCase):
    def test_strategy_examples_load_with_sp500_runtime(self):
        simple_config = load_split_config("configs/strategies/simple_supertrend.yaml", "configs/runtimes/research_sp500.yaml")
        leader_config = load_split_config("configs/strategies/leader_rotation.yaml", "configs/runtimes/research_sp500.yaml")

        self.assertEqual(simple_config.strategy.type, "simple_supertrend")
        self.assertEqual(leader_config.strategy.type, "leader_rotation")

    def test_all_strategy_files_load_with_a_runtime(self):
        config_paths = [
            path
            for path in (CONFIG_ROOT / "strategies").glob("*.yaml")
        ]
        self.assertTrue(config_paths)
        for path in config_paths:
            with self.subTest(path=str(path)):
                runtime = CONFIG_ROOT / "runtimes" / (
                    "live_toss.yaml" if "main_jo" in path.name else "research_sp500.yaml"
                )
                config = load_split_config(path, runtime)
                self.assertIn(config.strategy.type, {"simple_supertrend", "leader_rotation", "triple_filters"})
                self.assertGreater(config.capital.initial_cash, 0)

    def test_split_config_composes_strategy_universe_and_runtime(self):
        config = load_split_config(
            "configs/strategies/leader_rotation.yaml",
            "configs/runtimes/research_sp500.yaml",
        )

        self.assertEqual(config.strategy.name, "leader_rotation_default")
        self.assertEqual(config.strategy.type, "leader_rotation")
        self.assertEqual(config.market, "US")
        self.assertEqual(config.universe_file, "universe.json")
        self.assertEqual(config.symbols, ())
        self.assertEqual(config.timeframe, "1d")
        self.assertEqual(config.period, "max")
        self.assertFalse(hasattr(config, "benchmark"))
        self.assertEqual(config.market_trend_filter.enabled, True)
        self.assertEqual(config.market_trend_filter.timeframe, "1d")
        self.assertEqual(config.scoring.type, "relative_strength")
        self.assertEqual(config.execution.broker, "paper")
        self.assertEqual(config.paper.state_file, "state/research_sp500_paper.json")
        self.assertEqual(config.paper.results_dir, "results/research/sp500/paper")
        self.assertEqual(config.backtest.results_dir, "results/research/sp500/backtests")
        self.assertNotIn("relative_strength", {component.type for component in config.components})

    def test_strategy_registry_creates_registered_strategy(self):
        config = load_split_config("configs/strategies/simple_supertrend.yaml", "configs/runtimes/research_sp500.yaml")

        strategy = create_strategy(config)

        self.assertEqual(strategy.strategy_type, "simple_supertrend")
        self.assertGreater(strategy.warmup_bars(), 0)
        self.assertIn("leader_rotation", available_strategies())
        self.assertIn("simple_supertrend", available_strategies())
        self.assertIn("triple_filters", available_strategies())

    def test_unknown_strategy_type_lists_available_strategies(self):
        config = load_split_config("configs/strategies/simple_supertrend.yaml", "configs/runtimes/research_sp500.yaml")
        bad_config = config.__class__(
            **{
                **config.__dict__,
                "strategy": config.strategy.__class__(
                    name="missing",
                    type="missing_strategy",
                    params={},
                ),
            }
        )

        with self.assertRaisesRegex(ValueError, "Available strategies: .*simple_supertrend"):
            create_strategy(bad_config)

    def test_duplicate_strategy_registration_is_rejected(self):
        class DuplicateSimpleSupertrend:
            strategy_type = "simple_supertrend"

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_strategy(DuplicateSimpleSupertrend)

    def test_strategy_specific_params_reject_unknown_keys(self):
        config = load_split_config("configs/strategies/simple_supertrend.yaml", "configs/runtimes/research_sp500.yaml")
        bad_config = config.__class__(
            **{
                **config.__dict__,
                "strategy": config.strategy.__class__(
                    name=config.strategy.name,
                    type=config.strategy.type,
                    params={"typo": 1},
                ),
            }
        )

        with self.assertRaisesRegex(ValueError, "Unsupported params for simple_supertrend"):
            create_strategy(bad_config)

    def test_backtest_engine_has_no_strategy_type_dispatch(self):
        runner_source = (PACKAGE_ROOT / "runners.py").read_text(encoding="utf-8")

        self.assertNotIn("config.strategy.type ==", runner_source)
        self.assertNotIn("if config.strategy.type", runner_source)

    def test_split_live_config_uses_remaining_leader_rotation_strategy(self):
        config = load_split_config(
            "configs/strategies/leader_rotation.yaml",
            "configs/runtimes/live_toss.yaml",
        )

        self.assertEqual(config.strategy.name, "leader_rotation_default")
        self.assertEqual(config.market, "AUTO")
        self.assertEqual(config.universe_file, "universe.json")
        self.assertEqual(config.timeframe, "1d")
        self.assertEqual(config.period, "max")
        self.assertEqual(config.data_store.provider, "parquet")
        self.assertEqual(config.supertrend.atr_method, "wilder")
        self.assertEqual(config.market_trend_filter.timeframe, "1d")
        self.assertEqual(config.scoring.type, "relative_strength")
        self.assertEqual(config.execution.broker, "toss")
        self.assertEqual(config.execution.live_confirm_required, True)

    def test_benchmark_is_auto_mapped_by_symbol(self):
        self.assertEqual(benchmark_for_symbol("SOXL", "US", "universe.json"), "QQQ")
        self.assertEqual(benchmark_for_symbol("005930", "KR", "universe.json"), "^KS11")
        self.assertEqual(benchmark_for_symbol("010170", "KR", "universe.json"), "^KQ11")

    def test_legacy_relative_strength_filter_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_strategy.yaml"
            path.write_text(
                """
name: bad
type: leader_rotation
portfolio:
  max_positions: 1
scoring:
  type: relative_strength
  params:
    lookback_bars: 100
signals:
  entries:
    - type: supertrend
  filters:
    - type: relative_strength
      enabled: true
      lookback_bars: 100
  exits:
    - type: supertrend_flip
      confirm_bars: 1
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported component"):
                load_split_config(path, "configs/runtimes/research_sp500.yaml")

    def test_scoring_section_is_required(self):
        source = (CONFIG_ROOT / "strategies/simple_supertrend.yaml").read_text(encoding="utf-8")
        missing = source.replace(
            "scoring:\n  type: relative_strength\n  params:\n    lookback_bars: 100\n\n",
            "",
            1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing_scoring.yaml"
            path.write_text(missing, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scoring"):
                load_split_config(path, "configs/runtimes/research_sp500.yaml")

    def test_scoring_type_and_params_are_validated_during_load(self):
        source = (CONFIG_ROOT / "strategies/simple_supertrend.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            unknown_path = Path(tmp) / "unknown_scoring.yaml"
            unknown_path.write_text(
                source.replace("type: relative_strength", "type: missing_scorer", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Available scorers"):
                load_split_config(unknown_path, "configs/runtimes/research_sp500.yaml")

            invalid_path = Path(tmp) / "invalid_scoring.yaml"
            invalid_path.write_text(
                source.replace("lookback_bars: 100", "lookback_bars: 0", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "positive integer"):
                load_split_config(invalid_path, "configs/runtimes/research_sp500.yaml")

    def test_runtime_cannot_override_strategy_owned_position_sizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_path = Path(tmp) / "bad_runtime.yaml"
            runtime_path.write_text(
                """
name: bad_runtime
market: US
universe_file: universe.json
symbols: []
data:
  timeframe: 30m
  period: 60d
execution:
  order_type: market
  allocation_pct: 0.5
  broker: paper
  live_confirm_required: true
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "runtime.execution"):
                load_split_config("configs/strategies/simple_supertrend.yaml", runtime_path)

    def test_confirm_bars_requires_consecutive_down_trend(self):
        one_down = pd.DataFrame({"Trend": [1, 1, -1]})
        two_down = pd.DataFrame({"Trend": [1, -1, -1]})

        self.assertFalse(_trend_down_confirmed(one_down, 2))
        self.assertTrue(_trend_down_confirmed(two_down, 2))

    def test_simple_supertrend_returns_order_plan(self):
        config = load_split_config("configs/strategies/simple_supertrend.yaml", "configs/runtimes/research_sp500.yaml")
        config = config.__class__(
            **{
                **config.__dict__,
                "supertrend": config.supertrend.__class__(period=2, multiplier=1.0),
                "risk": config.risk.__class__(max_position_count=1),
            }
        )
        df = pd.DataFrame(
            {
                "Open": [10, 9, 8, 9, 11, 12],
                "High": [11, 10, 9, 10, 12, 13],
                "Low": [9, 8, 7, 8, 10, 11],
                "Close": [10, 9, 8, 9, 11, 12],
            },
            index=pd.date_range("2026-01-01", periods=6, freq="30min"),
        )

        plan = build_order_plan(config, {"AAA": df}, AccountSnapshot(cash=10_000), mode="paper")
        self.assertEqual(plan.strategy_name, "simple_supertrend_default")
        self.assertIsInstance(plan.orders, tuple)

    def test_paper_broker_writes_json_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.json"
            broker = PaperBroker(path, initial_cash=1_000)
            account = broker.get_account()

            self.assertEqual(account.cash, 1_000)
            self.assertTrue(path.parent.exists())


if __name__ == "__main__":
    unittest.main()
