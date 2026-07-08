import tempfile
import unittest
from pathlib import Path

import pandas as pd

from supertrend_quant.brokers import PaperBroker
from supertrend_quant.config import benchmark_for_symbol, load_split_config
from supertrend_quant.portfolio import AccountSnapshot
from supertrend_quant.strategies import _trend_down_confirmed, build_order_plan, effective_rs_period


class ConfigAndStrategyTest(unittest.TestCase):
    def test_strategy_examples_load_with_simulation_runtime(self):
        simple_config = load_split_config("configs/strategies/simple_supertrend.yaml", "configs/runtimes/simulation.yaml")
        leader_config = load_split_config("configs/strategies/leader_rotation.yaml", "configs/runtimes/simulation.yaml")

        self.assertEqual(simple_config.strategy.type, "simple_supertrend")
        self.assertEqual(leader_config.strategy.type, "leader_rotation")
        self.assertEqual(simple_config.supertrend.multiplier, 3.0)
        self.assertEqual(simple_config.risk.max_position_count, 3)
        self.assertEqual(simple_config.execution.allocation_pct, 0.9)

    def test_all_strategy_files_load_with_a_runtime(self):
        config_paths = [
            path
            for path in Path("configs/strategies").glob("*.yaml")
        ]
        self.assertGreater(len(config_paths), 1)
        for path in config_paths:
            with self.subTest(path=str(path)):
                runtime = "configs/runtimes/live_toss.yaml" if "main_jo" in path.name else "configs/runtimes/simulation.yaml"
                config = load_split_config(path, runtime)
                self.assertIn(config.strategy.type, {"simple_supertrend", "leader_rotation"})
                self.assertGreater(config.capital.initial_cash, 0)

    def test_split_config_composes_strategy_universe_and_runtime(self):
        config = load_split_config(
            "configs/strategies/leader_rotation.yaml",
            "configs/runtimes/simulation.yaml",
        )

        self.assertEqual(config.strategy.name, "leader_rotation_default")
        self.assertEqual(config.strategy.type, "leader_rotation")
        self.assertEqual(config.market, "US")
        self.assertEqual(config.universe_file, "universe.json")
        self.assertEqual(config.symbols, ())
        self.assertEqual(config.timeframe, "30m")
        self.assertEqual(config.period, "30d")
        self.assertFalse(hasattr(config, "benchmark"))
        self.assertEqual(config.supertrend.period, 10)
        self.assertEqual(config.supertrend.multiplier, 3.0)
        self.assertEqual(config.market_trend_filter.enabled, True)
        self.assertEqual(config.market_trend_filter.timeframe, "1d")
        self.assertEqual(config.leader_rotation.rs_period, 100)
        self.assertEqual(config.leader_rotation.max_slots, 1)
        self.assertEqual(config.exit.sell_confirm_bars, 1)
        self.assertEqual(config.execution.broker, "paper")
        self.assertEqual(config.paper.state_file, "state/paper.json")
        self.assertEqual(config.paper.results_dir, "results/paper")
        self.assertEqual(config.backtest.results_dir, "results/backtests")
        self.assertIn("relative_strength", {component.type for component in config.components})
        self.assertEqual(len(config.components), 4)

    def test_split_live_config_matches_main_jo_runtime_settings(self):
        config = load_split_config(
            "configs/strategies/main_jo_leader_rotation.yaml",
            "configs/runtimes/live_toss.yaml",
        )

        self.assertEqual(config.strategy.name, "main_jo_live_leader_rotation")
        self.assertEqual(config.market, "AUTO")
        self.assertEqual(config.universe_file, "universe.json")
        self.assertEqual(config.period, "30d")
        self.assertEqual(config.supertrend.period, 7)
        self.assertEqual(config.supertrend.atr_method, "ewm")
        self.assertEqual(config.supertrend.symbol_multipliers["SOXL"], 4.5)
        self.assertEqual(config.market_trend_filter.timeframe, "1h")
        self.assertEqual(config.leader_rotation.rs_period, 100)
        self.assertEqual(config.leader_rotation.rs_period_by_market, {"US": 130, "KR": 100})
        self.assertEqual(effective_rs_period(config.__class__(**{**config.__dict__, "market": "US"})), 130)
        self.assertEqual(effective_rs_period(config.__class__(**{**config.__dict__, "market": "KR"})), 100)
        self.assertEqual(config.execution.broker, "toss")
        self.assertEqual(config.execution.live_confirm_required, True)

    def test_benchmark_is_auto_mapped_by_symbol(self):
        self.assertEqual(benchmark_for_symbol("SOXL", "US", "universe.json"), "QQQ")
        self.assertEqual(benchmark_for_symbol("005930", "KR", "universe.json"), "^KS11")
        self.assertEqual(benchmark_for_symbol("010170", "KR", "universe.json"), "^KQ11")

    def test_unsupported_component_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_strategy.yaml"
            path.write_text(
                """
name: bad
type: leader_rotation
portfolio:
  mode: leader_rotation
  max_positions: 1
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

            with self.assertRaisesRegex(ValueError, "Unsupported keys"):
                load_split_config(path, "configs/runtimes/simulation.yaml")

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
        config = load_split_config("configs/strategies/simple_supertrend.yaml", "configs/runtimes/simulation.yaml")
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
