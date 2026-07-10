import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from supertrend_quant.config import load_split_config
from supertrend_quant.paper_runtime import PaperRuntime
from supertrend_quant.results import PaperRunRecorder, compare_paper_to_backtest, save_backtest_result


class FakeCache:
    def __init__(self, bars, benchmark):
        self.bars = bars
        self.benchmark = benchmark
        self.sync_count = 0

    def sync(self, symbols, market, universe_file, benchmarks, current_candle_base=None):
        self.sync_count += 1

    def retry_missing(self, market, universe_file, market_tz, current_candle_base):
        return []

    def fresh_stock_bars(self, symbols, market_tz, current_candle_base):
        return self.bars, []

    def fresh_benchmark_map(self, symbols, market, universe_file, source, market_tz, current_base):
        return {symbol: self.benchmark for symbol in symbols}


class FakeBacktestResult:
    def __init__(self, equity):
        self.equity = equity
        self.metrics = {
            "total_return": 0.1,
            "mdd": -0.02,
            "sharpe": 1.2,
            "win_rate": 0.5,
            "payoff_ratio": 1.5,
            "trade_count": 2,
        }
        self.trades = [0.1, -0.02]
        self.skipped = ()


class PaperRuntimeResultsTest(unittest.TestCase):
    def test_paper_once_records_cycle_equity_and_skips_duplicate_candle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = load_split_config("configs/strategies/simple_supertrend.yaml", "configs/runtimes/simulation.yaml")
            config = config.__class__(
                **{
                    **config.__dict__,
                    "symbols": ("AAA",),
                    "paper": config.paper.__class__(
                        state_file=str(tmp_path / "paper.json"),
                        results_dir=str(tmp_path / "paper-results"),
                        loop_interval_seconds=1,
                        run_once_per_candle=True,
                    ),
                }
            )
            idx = pd.date_range("2026-01-01", periods=8, freq="30min")
            df = pd.DataFrame(
                {
                    "Open": [10, 10, 10, 10, 10, 10, 10, 10],
                    "High": [11, 11, 11, 11, 11, 11, 11, 11],
                    "Low": [9, 9, 9, 9, 9, 9, 9, 9],
                    "Close": [10, 10, 10, 10, 10, 10, 10, 10],
                },
                index=idx,
            )
            recorder = PaperRunRecorder(config.paper.results_dir, config.strategy.name, run_id="paper-test")
            runtime = PaperRuntime(
                config,
                data_cache=FakeCache({"AAA": df}, df),
                recorder=recorder,
            )

            runtime.run_once(ignore_schedule=True)
            second_plan, _ = runtime.run_once(ignore_schedule=True)

            run_dir = tmp_path / "paper-results" / "paper-test"
            self.assertTrue((run_dir / "cycles.jsonl").exists())
            self.assertTrue((run_dir / "equity.csv").exists())
            self.assertIn("Candle already processed", second_plan.notes[0])
            self.assertEqual(len((run_dir / "cycles.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            state = json.loads((tmp_path / "paper.json").read_text(encoding="utf-8"))
            self.assertIn("metadata", state)

    def test_saved_backtest_and_paper_outputs_can_be_compared(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = load_split_config("configs/strategies/leader_rotation.yaml", "configs/runtimes/simulation.yaml")
            backtest_equity = pd.Series(
                [10_000, 10_500, 11_000],
                index=pd.date_range("2026-01-01", periods=3, freq="30min"),
                name="equity",
            )
            backtest_dir = save_backtest_result(
                FakeBacktestResult(backtest_equity),
                config,
                tmp_path / "backtests",
                run_id="backtest-test",
            )
            paper_dir = tmp_path / "paper" / "paper-test"
            paper_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01", periods=3, freq="30min").astype(str),
                    "market": ["US", "US", "US"],
                    "candle_base": pd.date_range("2026-01-01", periods=3, freq="30min").astype(str),
                    "equity": [10_000, 10_100, 10_200],
                    "cash": [10_000, 10_100, 10_200],
                    "positions_value": [0, 0, 0],
                    "position_count": [0, 0, 0],
                    "order_count": [0, 0, 0],
                    "fill_count": [0, 0, 0],
                }
            ).to_csv(paper_dir / "equity.csv", index=False)

            comparison = compare_paper_to_backtest(paper_dir, backtest_dir, "30m")

            self.assertEqual(comparison["paper_points"], 3)
            self.assertIn("total_return", comparison["diff"])


if __name__ == "__main__":
    unittest.main()
