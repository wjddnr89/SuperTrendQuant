from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from supertrend_quant.config import parse_config
from supertrend_quant.data import MarketData
from supertrend_quant.portfolio import OrderIntent, OrderPlan
from supertrend_quant.runners import run_backtest_on_data


class _BuyThenSellStrategy:
    def __init__(self):
        self.signal_closes = []

    def warmup_bars(self):
        return 0

    def build_order_plan(self, bars, account, mode, **kwargs):
        self.signal_closes.append(float(bars["AAA"]["Close"].iloc[-1]))
        if not account.positions:
            return OrderPlan("test", mode, (OrderIntent("AAA", "buy", 1),))
        return OrderPlan("test", mode, (OrderIntent("AAA", "sell", 1),))


class _BuyAndHoldStrategy:
    def warmup_bars(self):
        return 0

    def build_order_plan(self, bars, account, mode, **kwargs):
        if not account.positions:
            return OrderPlan("test", mode, (OrderIntent("AAA", "buy", 1),))
        return OrderPlan("test", mode, ())


class DualPriceStreamBacktestTest(unittest.TestCase):
    def test_blocked_market_data_fails_closed_before_strategy_execution(self):
        config = parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["AAA"]},
            }
        )
        index = pd.date_range("2026-01-01", periods=2, freq="D")
        frame = pd.DataFrame(
            {"Open": [1.0, 1.0], "High": [1.0, 1.0], "Low": [1.0, 1.0], "Close": [1.0, 1.0]},
            index=index,
        )
        market_data = MarketData(
            bars={"AAA": frame},
            data_quality="blocked",
            warnings=("missing active member",),
        )
        with self.assertRaisesRegex(RuntimeError, "missing active member"):
            run_backtest_on_data(config, market_data)

    def test_signals_use_adjusted_bars_while_fills_ledger_and_value_use_raw_bars(self):
        config = parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["AAA"]},
                "timeframe": "1d",
                "period": "max",
                "capital": {"initial_cash": 1000.0},
                "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
            }
        )
        index = pd.date_range("2026-01-01", periods=3, freq="D")
        adjusted = pd.DataFrame(
            {"Open": [10.0, 20.0, 30.0], "High": [11.0, 21.0, 31.0], "Low": [9.0, 19.0, 29.0], "Close": [10.0, 20.0, 30.0]},
            index=index,
        )
        raw = pd.DataFrame(
            {"Open": [100.0, 200.0, 300.0], "High": [101.0, 201.0, 301.0], "Low": [99.0, 199.0, 299.0], "Close": [100.0, 200.0, 300.0]},
            index=index,
        )
        market_data = MarketData(
            bars={"AAA": adjusted},
            execution_bars={"AAA": raw},
            corporate_actions=(
                {
                    "event_id": "div-1",
                    "action_type": "cash_dividend",
                    "symbol": "AAA",
                    "effective_date": "2026-01-02",
                    "cash_amount": 5.0,
                },
            ),
            data_version="fixture-v1",
            completed_session="2026-01-03",
        )
        strategy = _BuyThenSellStrategy()

        with patch("supertrend_quant.runners.create_strategy", return_value=strategy):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(strategy.signal_closes, [10.0, 20.0])
        self.assertEqual(result.trade_records[0]["entry_price"], 200.0)
        self.assertEqual(result.trade_records[0]["exit_price"], 300.0)
        self.assertEqual(result.equity.iloc[-1], 1105.0)
        self.assertAlmostEqual(result.trade_records[0]["pnl_pct"], 0.525)
        self.assertEqual(result.trade_records[0]["corporate_action_cash"], 5.0)
        self.assertEqual(result.corporate_action_cash, 5.0)
        self.assertEqual(result.processed_corporate_action_ids, ("div-1",))
        self.assertEqual(result.data_version, "fixture-v1")

    def test_split_bookkeeping_does_not_create_drawdown_or_return(self):
        config = parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["AAA"]},
                "capital": {"initial_cash": 1000.0},
                "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
            }
        )
        index = pd.date_range("2026-01-01", periods=4, freq="D")
        adjusted = pd.DataFrame(
            {"Open": [50.0] * 4, "High": [50.0] * 4, "Low": [50.0] * 4, "Close": [50.0] * 4},
            index=index,
        )
        raw = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 50.0, 50.0],
                "High": [100.0, 100.0, 50.0, 50.0],
                "Low": [100.0, 100.0, 50.0, 50.0],
                "Close": [100.0, 100.0, 50.0, 50.0],
            },
            index=index,
        )
        market_data = MarketData(
            bars={"AAA": adjusted},
            execution_bars={"AAA": raw},
            corporate_actions=(
                {
                    "event_id": "split",
                    "action_type": "split",
                    "symbol": "AAA",
                    "effective_date": "2026-01-03",
                    "ratio": 2.0,
                },
            ),
        )
        with patch("supertrend_quant.runners.create_strategy", return_value=_BuyAndHoldStrategy()):
            result = run_backtest_on_data(config, market_data)

        self.assertTrue((result.equity == 1000.0).all())
        self.assertEqual(result.metrics["mdd"], 0.0)
        self.assertEqual(result.metrics["total_return"], 0.0)
        self.assertEqual(result.metrics["sharpe"], 0.0)

    def test_incomplete_action_is_unapplied_and_marks_result_degraded(self):
        config = parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["AAA"]},
                "capital": {"initial_cash": 1000.0},
                "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
            }
        )
        index = pd.date_range("2026-01-01", periods=3, freq="D")
        frame = pd.DataFrame(
            {"Open": [100.0] * 3, "High": [100.0] * 3, "Low": [100.0] * 3, "Close": [100.0] * 3},
            index=index,
        )
        market_data = MarketData(
            bars={"AAA": frame},
            execution_bars={"AAA": frame},
            corporate_actions=(
                {
                    "event_id": "incomplete",
                    "action_type": "cash_dividend",
                    "symbol": "AAA",
                    "effective_date": "2026-01-02",
                    "cash_amount": None,
                },
            ),
        )
        with patch("supertrend_quant.runners.create_strategy", return_value=_BuyAndHoldStrategy()):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(result.data_quality, "degraded")
        self.assertEqual(result.unresolved_corporate_action_ids, ("incomplete",))
        self.assertTrue(any("left unapplied" in warning for warning in result.data_warnings))


if __name__ == "__main__":
    unittest.main()
