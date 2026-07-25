import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from supertrend_quant.config import load_split_config, parse_config
from supertrend_quant.data import MarketData
from supertrend_quant.market_store.realtime import TossRealtimeQuoteProvider
from supertrend_quant.paper_runtime import PaperRuntime
from supertrend_quant.portfolio import AccountSnapshot, Position
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


class EmptyQuoteProvider:
    def quotes(self, symbols):
        return {}


class RecordingNotifier:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)
        return True


class PaperRuntimeResultsTest(unittest.TestCase):
    def test_toss_quote_source_keeps_paper_broker_and_selects_toss_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = load_split_config(
                "configs/strategies/leader_rotation_dual_momentum_nasdaq100.yaml",
                "configs/runtimes/paper_toss_nasdaq100_canonical.yaml",
            )
            config = config.__class__(
                **{
                    **config.__dict__,
                    "paper": config.paper.__class__(
                        state_file=str(tmp_path / "paper.json"),
                        results_dir=str(tmp_path / "results"),
                        loop_interval_seconds=60,
                        run_once_per_candle=True,
                        quote_source="toss",
                    ),
                }
            )

            runtime = PaperRuntime(config)

            self.assertEqual(config.execution.broker, "paper")
            self.assertIsInstance(runtime.quote_provider, TossRealtimeQuoteProvider)

    def test_paper_telegram_reports_start_fills_summary_and_errors_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = parse_config(
                {
                    "strategy": {"name": "telegram-test", "type": "equal", "params": {}},
                    "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
                    "market": "US",
                    "universe": {"source": "symbols", "symbols": ["AAA"]},
                    "capital": {"initial_cash": 10_000},
                    "paper": {
                        "state_file": str(root / "paper.json"),
                        "results_dir": str(root / "results"),
                        "telegram_enabled": True,
                    },
                }
            )
            notifier = RecordingNotifier()
            runtime = PaperRuntime(config, notifier=notifier)

            runtime._notify_start_once()
            runtime._notify_start_once()
            runtime._notify_fills(["BUY AAA 2 @ 100.0000", "SKIP BBB: no price"])
            account = AccountSnapshot(
                cash=9_800,
                positions={"AAA": Position("AAA", 2, 100)},
            )
            runtime._notify_summary(
                session_market="US",
                candle_value="2026-07-24T00:00:00-04:00",
                account=account,
                prices={"AAA": 110},
                fills=["BUY AAA 2 @ 100.0000"],
            )
            runtime._notify_summary(
                session_market="US",
                candle_value="2026-07-24T00:00:00-04:00",
                account=account,
                prices={"AAA": 110},
                fills=[],
            )

            with patch.object(runtime, "_run_once", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    runtime.run_once(ignore_schedule=True)

            self.assertEqual(len(notifier.messages), 4)
            self.assertIn("가상계좌 운용 시작", notifier.messages[0])
            self.assertIn("BUY AAA 2", notifier.messages[1])
            self.assertNotIn("SKIP BBB", notifier.messages[1])
            self.assertIn("총자산: $10,020.00", notifier.messages[2])
            self.assertIn("누적수익률: +0.20%", notifier.messages[2])
            self.assertIn("RuntimeError: boom", notifier.messages[3])

    def test_stale_daily_history_blocks_all_paper_strategy_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = parse_config(
                {
                    "strategy": {"name": "test", "type": "equal", "params": {}},
                    "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
                    "market": "US",
                    "universe": {"source": "symbols", "symbols": ["AAA"]},
                    "paper": {
                        "state_file": str(root / "paper.json"),
                        "results_dir": str(root / "results"),
                    },
                    "data_store": {
                        "provider": "parquet",
                        "local_cache_dir": str(root / "cache"),
                    },
                }
            )
            frame = pd.DataFrame(
                {"Open": [10.0], "High": [10.0], "Low": [10.0], "Close": [10.0]},
                index=[pd.Timestamp("2026-07-14")],
            )
            stale = MarketData(
                bars={"AAA": frame},
                execution_bars={"AAA": frame},
                completed_session="2026-07-15",
            )
            runtime = PaperRuntime(config)
            with (
                patch(
                    "supertrend_quant.paper_runtime.ensure_configured_data_ready"
                ),
                patch(
                    "supertrend_quant.paper_runtime.load_configured_market_data",
                    return_value=stale,
                ),
            ):
                plan, results = runtime.run_once(ignore_schedule=True)

            self.assertEqual(plan.orders, ())
            self.assertIn("Historical data is incomplete", plan.notes[0])
            self.assertEqual(results, ["Paper orders blocked by historical data gap."])

    def test_missing_paper_quote_blocks_the_entire_strategy_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = load_split_config(
                "configs/strategies/simple_supertrend.yaml",
                "configs/runtimes/research_sp500.yaml",
            )
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
            idx = pd.date_range("2026-01-01", periods=3, freq="D")
            frame = pd.DataFrame(
                {
                    "Open": [10.0] * 3,
                    "High": [10.0] * 3,
                    "Low": [10.0] * 3,
                    "Close": [10.0] * 3,
                },
                index=idx,
            )
            runtime = PaperRuntime(
                config,
                data_cache=FakeCache({"AAA": frame}, frame),
                quote_provider=EmptyQuoteProvider(),
            )

            plan, results = runtime.run_once(ignore_schedule=True)

            self.assertEqual(plan.orders, ())
            self.assertIn("Paper quote gap", plan.notes[0])
            self.assertEqual(results, ["Paper orders blocked by quote gap."])

    def test_paper_once_records_cycle_equity_and_skips_duplicate_candle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = load_split_config("configs/strategies/simple_supertrend.yaml", "configs/runtimes/research_sp500.yaml")
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
            config = load_split_config("configs/strategies/leader_rotation.yaml", "configs/runtimes/research_sp500.yaml")
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
