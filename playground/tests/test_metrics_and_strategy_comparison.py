import json
import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from supertrend_quant import cli, research
from supertrend_quant.config import AppConfig, ScoringConfig, StrategyIdentity
from supertrend_quant.data import MarketData
from supertrend_quant.metrics import annualization_factor, calculate_metrics
from supertrend_quant.research import comparison
from supertrend_quant.runners import BacktestResult


class MetricsTest(unittest.TestCase):
    def test_cagr_calmar_and_sortino_use_standard_risk_adjusted_formulas(self):
        index = pd.to_datetime(
            ["2025-01-01", "2025-04-01", "2025-07-01", "2026-01-01"]
        )
        equity = pd.Series([100.0, 120.0, 90.0, 121.0], index=index)

        metrics = calculate_metrics(equity, [0.2, -0.25, 0.1], "1d")

        elapsed_years = (index[-1] - index[0]).total_seconds() / (
            365.25 * 24 * 60 * 60
        )
        expected_cagr = (121.0 / 100.0) ** (1.0 / elapsed_years) - 1.0
        returns = equity.pct_change().dropna()
        downside = np.minimum(returns.to_numpy(), 0.0)
        expected_sortino = (
            returns.mean()
            / np.sqrt(np.mean(np.square(downside)))
            * math.sqrt(annualization_factor("1d"))
        )
        self.assertAlmostEqual(metrics["cagr"], expected_cagr)
        self.assertAlmostEqual(metrics["mdd"], -0.25)
        self.assertAlmostEqual(metrics["calmar"], expected_cagr / 0.25)
        self.assertAlmostEqual(metrics["sortino"], expected_sortino)
        self.assertEqual(metrics["trade_count"], 3)

    def test_metric_edge_cases_are_stable(self):
        empty = calculate_metrics(pd.Series(dtype=float), [], "30m")
        self.assertEqual(
            set(empty),
            {
                "total_return",
                "cagr",
                "mdd",
                "calmar",
                "sharpe",
                "sortino",
                "win_rate",
                "payoff_ratio",
                "trade_count",
            },
        )
        single = calculate_metrics(
            pd.Series([100.0], index=pd.to_datetime(["2025-01-01"])), [], "1d"
        )
        self.assertEqual(single["cagr"], 0.0)
        self.assertEqual(single["calmar"], 0.0)
        self.assertEqual(single["sortino"], 0.0)

        rising = calculate_metrics(
            pd.Series(
                [100.0, 105.0, 110.0],
                index=pd.to_datetime(["2025-01-01", "2025-07-01", "2026-01-01"]),
            ),
            [0.05, 0.05],
            "1d",
        )
        self.assertTrue(math.isinf(rising["calmar"]))
        self.assertTrue(math.isinf(rising["sortino"]))
        self.assertTrue(math.isinf(rising["payoff_ratio"]))

        wiped_out = calculate_metrics(
            pd.Series(
                [100.0, 0.0],
                index=pd.to_datetime(["2025-01-01", "2026-01-01"]),
            ),
            [-1.0],
            "1d",
        )
        self.assertAlmostEqual(wiped_out["cagr"], -1.0)
        self.assertEqual(wiped_out["mdd"], -1.0)


class StrategyComparisonTest(unittest.TestCase):
    def _config(self, name: str) -> AppConfig:
        return AppConfig(
            strategy=StrategyIdentity(name=name, type="simple_supertrend"),
            scoring=ScoringConfig(type="relative_strength", params={"lookback_bars": 1}),
        )

    def _market_data(self) -> MarketData:
        index = pd.date_range("2026-01-05 09:30", periods=8, freq="30min")
        frame = pd.DataFrame(
            {
                "Open": np.linspace(10.0, 11.0, len(index)),
                "High": np.linspace(10.5, 11.5, len(index)),
                "Low": np.linspace(9.5, 10.5, len(index)),
                "Close": np.linspace(10.0, 11.0, len(index)),
            },
            index=index,
        )
        return MarketData(bars={"AAA": frame}, benchmark={"AAA": frame})

    def _metrics(self, calmar: float) -> dict[str, float | int]:
        return {
            "total_return": calmar / 10,
            "cagr": calmar / 5,
            "mdd": -0.1,
            "calmar": calmar,
            "sharpe": calmar,
            "sortino": calmar,
            "win_rate": 0.5,
            "payoff_ratio": calmar,
            "trade_count": 2,
        }

    def test_discovers_nested_yaml_uses_common_warmup_and_continues_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "strategies"
            nested = root / "nested"
            nested.mkdir(parents=True)
            for path in (root / "fast.yaml", nested / "slow.yaml", root / "bad.yaml"):
                path.write_text("placeholder: true\n", encoding="utf-8")
            runtime = Path(tmp) / "runtime.yaml"
            runtime.write_text("placeholder: true\n", encoding="utf-8")
            data = self._market_data()
            seen_indexes = []

            def fake_load(path, _runtime):
                if Path(path).stem == "bad":
                    raise ValueError("invalid strategy")
                return self._config(Path(path).stem)

            def fake_create(config):
                warmup = 3 if config.strategy.name == "slow" else 1
                return type("FakeStrategy", (), {"warmup_bars": lambda self: warmup})()

            def fake_run(config, _data, run_index=None):
                selected = pd.Index(run_index)
                seen_indexes.append(selected)
                calmar = 2.0 if config.strategy.name == "slow" else 1.0
                equity = pd.Series([100.0] * len(selected), index=selected, name="equity")
                return BacktestResult(equity, self._metrics(calmar), [], ())

            with patch.object(comparison, "load_split_config", side_effect=fake_load), patch.object(
                comparison, "create_strategy", side_effect=fake_create
            ), patch.object(comparison, "run_backtest_on_data", side_effect=fake_run):
                result = comparison.compare_strategies(
                    root,
                    runtime,
                    market_data=lambda _config: data,
                )

            self.assertEqual([row.strategy_name for row in result.rows], ["slow", "fast"])
            self.assertEqual(result.winner.strategy_name, "slow")
            self.assertEqual(len(result.errors), 1)
            self.assertEqual(result.errors[0].strategy_path, "bad.yaml")
            self.assertEqual(len(result.common_index), 5)
            for selected in seen_indexes:
                pd.testing.assert_index_equal(selected, result.common_index)

    def test_composite_ranking_single_candidate_and_saved_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "strategies"
            root.mkdir()
            strategy_path = root / "only.yaml"
            strategy_path.write_text("placeholder: true\n", encoding="utf-8")
            runtime = Path(tmp) / "runtime.yaml"
            runtime.write_text("placeholder: true\n", encoding="utf-8")
            config = self._config("only")
            data = self._market_data()

            fake_strategy = type("FakeStrategy", (), {"warmup_bars": lambda self: 1})()

            def fake_run(_config, _data, run_index=None):
                selected = pd.Index(run_index)
                equity = pd.Series(
                    np.linspace(100.0, 110.0, len(selected)),
                    index=selected,
                    name="equity",
                )
                return BacktestResult(equity, self._metrics(1.0), [0.1], ())

            with patch.object(comparison, "load_split_config", return_value=config), patch.object(
                comparison, "create_strategy", return_value=fake_strategy
            ), patch.object(comparison, "run_backtest_on_data", side_effect=fake_run):
                result = comparison.compare_strategies(
                    root,
                    runtime,
                    rank_by="composite",
                    market_data=lambda _config: data,
                )

            self.assertTrue(result.winner.is_best)
            self.assertEqual(result.winner.composite_score, 100.0)
            table = comparison.format_comparison_table(result)
            self.assertIn("Sortino", table)
            self.assertIn("BEST", table)

            run_dir = comparison.save_comparison_result(
                result,
                Path(tmp) / "results",
                run_id="comparison-test",
            )
            self.assertTrue((run_dir / "comparison.csv").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "strategies" / "001_only" / "summary.json").exists())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["winner"]["strategy_name"], "only")
            self.assertEqual(summary["rank_by"], "composite")

            output = io.StringIO()
            with patch("sys.argv", ["quant-compare-strategies", "--no-save"]), patch.object(
                research, "compare_strategies", return_value=result
            ), patch.object(
                research, "format_comparison_table", return_value="comparison table"
            ), patch.object(research, "save_comparison_result") as save_result, redirect_stdout(
                output
            ):
                cli.compare_strategies_main()
            save_result.assert_not_called()
            self.assertIn("Best strategy: only", output.getvalue())


if __name__ == "__main__":
    unittest.main()
