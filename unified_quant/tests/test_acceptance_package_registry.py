from __future__ import annotations

import inspect
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

import supertrend_quant
from supertrend_quant import cli
from supertrend_quant.config import StrategyIdentity, load_split_config
from supertrend_quant.data import MarketData
from supertrend_quant.portfolio import OrderPlan
from supertrend_quant.runners import run_backtest_on_data
from supertrend_quant.strategies import available_strategies, create_strategy, register_strategy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_ROOT = REPOSITORY_ROOT / "unified_quant"
STRATEGY_PATH = UNIFIED_ROOT / "configs/strategies/simple_supertrend.yaml"
RUNTIME_PATH = UNIFIED_ROOT / "configs/runtimes/research_sp500.yaml"


class ExtensionProbeStrategy:
    strategy_type = "acceptance_extension_probe"
    seen_modes: list[str] = []

    def __init__(self, config):
        self.config = config

    @classmethod
    def validate_config(cls, config) -> None:
        return None

    def warmup_bars(self) -> int:
        return 0

    def build_order_plan(
        self,
        bars,
        account,
        mode,
        benchmark=None,
        filter_benchmark=None,
    ) -> OrderPlan:
        self.__class__.seen_modes.append(mode)
        return OrderPlan(self.config.strategy.name, mode, ())


class PackageAndRegistryAcceptanceTest(unittest.TestCase):
    def test_root_package_and_all_cli_functions_exist(self):
        package_file = Path(supertrend_quant.__file__).resolve()
        expected_source = (UNIFIED_ROOT / "src/supertrend_quant").resolve()
        self.assertTrue(
            package_file.is_relative_to(expected_source),
            f"supertrend_quant resolved outside unified source: {package_file}",
        )

        expected_scripts = {
            "quant-backtest": "backtest_main",
            "quant-paper": "paper_main",
            "quant-live": "live_main",
            "quant-compare": "compare_main",
            "quant-compare-strategies": "compare_strategies_main",
            "quant-search": "search_main",
            "quant-optimize": "optimize_main",
        }
        project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = project["project"]["scripts"]
        for command, function_name in expected_scripts.items():
            with self.subTest(command=command):
                self.assertEqual(scripts.get(command), f"supertrend_quant.cli:{function_name}")
                self.assertTrue(callable(getattr(cli, function_name, None)))

    def test_registered_extension_runs_without_runner_type_dispatch(self):
        if ExtensionProbeStrategy.strategy_type not in available_strategies():
            register_strategy(ExtensionProbeStrategy)

        base = load_split_config(STRATEGY_PATH, RUNTIME_PATH)
        config = replace(
            base,
            strategy=StrategyIdentity(
                name="acceptance_extension",
                type=ExtensionProbeStrategy.strategy_type,
            ),
        )
        self.assertIsInstance(create_strategy(config), ExtensionProbeStrategy)

        index = pd.date_range("2026-01-05 09:30", periods=3, freq="30min")
        frame = pd.DataFrame(
            {
                "Open": [10.0, 10.0, 10.0],
                "High": [11.0, 11.0, 11.0],
                "Low": [9.0, 9.0, 9.0],
                "Close": [10.0, 10.0, 10.0],
            },
            index=index,
        )
        ExtensionProbeStrategy.seen_modes.clear()
        result = run_backtest_on_data(config, MarketData(bars={"AAA": frame}))

        self.assertFalse(result.equity.empty)
        self.assertEqual(ExtensionProbeStrategy.seen_modes, ["backtest", "backtest"])
        runner_source = inspect.getsource(run_backtest_on_data)
        self.assertNotIn("leader_rotation", runner_source)
        self.assertNotIn("simple_supertrend", runner_source)
        self.assertNotIn("config.strategy.type ==", runner_source)


if __name__ == "__main__":
    unittest.main()
