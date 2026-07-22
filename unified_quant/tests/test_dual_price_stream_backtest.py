from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from supertrend_quant.config import parse_config
from supertrend_quant.data import MarketData
from supertrend_quant.portfolio import AccountSnapshot, OrderIntent, OrderPlan, Position
from supertrend_quant.runners import (
    _reactivate_retired_symbol,
    _scheduled_corporate_actions,
    run_backtest_on_data,
    run_live_once,
    run_paper_once,
)


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


class _BuyThenDelayedSellStrategy:
    def __init__(self):
        self.signal_count = 0
        self.bought = False

    def warmup_bars(self):
        return 0

    def build_order_plan(self, bars, account, mode, **kwargs):
        self.signal_count += 1
        if not self.bought:
            self.bought = True
            return OrderPlan("test", mode, (OrderIntent("AAA", "buy", 1),))
        position = account.positions.get("AAA")
        if position is not None and self.signal_count >= 4:
            return OrderPlan(
                "test",
                mode,
                (OrderIntent("AAA", "sell", position.quantity, reason="DelayedExit"),),
            )
        return OrderPlan("test", mode, ())


class _BuyAndHoldStrategy:
    def warmup_bars(self):
        return 0

    def build_order_plan(self, bars, account, mode, **kwargs):
        if not account.positions:
            return OrderPlan("test", mode, (OrderIntent("AAA", "buy", 1),))
        return OrderPlan("test", mode, ())


class _BuyOldAndHoldStrategy:
    def __init__(self):
        self.bought = False

    def warmup_bars(self):
        return 0

    def build_order_plan(self, bars, account, mode, **kwargs):
        if not self.bought:
            self.bought = True
            return OrderPlan("test", mode, (OrderIntent("OLD", "buy", 1),))
        return OrderPlan("test", mode, ())


class _RetryOldBuyStrategy:
    def warmup_bars(self):
        return 0

    def build_order_plan(self, bars, account, mode, **kwargs):
        if not account.positions:
            return OrderPlan("test", mode, (OrderIntent("OLD", "buy", 1),))
        return OrderPlan("test", mode, ())


class _BuyActiveSymbolStrategy:
    def warmup_bars(self):
        return 0

    def build_order_plan(self, bars, account, mode, **kwargs):
        if account.positions or not bars:
            return OrderPlan("test", mode, ())
        symbol = sorted(bars)[0]
        return OrderPlan("test", mode, (OrderIntent(symbol, "buy", 1),))


class _BuyOldThenSellStrategy:
    def __init__(self):
        self.bought = False

    def warmup_bars(self):
        return 0

    def build_order_plan(self, bars, account, mode, **kwargs):
        if not self.bought:
            self.bought = True
            return OrderPlan("test", mode, (OrderIntent("OLD", "buy", 1),))
        position = account.positions.get("OLD")
        if position is not None:
            return OrderPlan(
                "test",
                mode,
                (OrderIntent("OLD", "sell", position.quantity, reason="OldExit"),),
            )
        return OrderPlan("test", mode, ())


class _BuyParentAndHoldStrategy:
    def __init__(self):
        self.bought = False

    def warmup_bars(self):
        return 0

    def build_order_plan(self, bars, account, mode, **kwargs):
        if not self.bought:
            self.bought = True
            return OrderPlan("test", mode, (OrderIntent("PARENT", "buy", 1),))
        return OrderPlan("test", mode, ())


class _SpinLifecycleStrategy:
    def __init__(self):
        self.bought = False
        self.seen_symbols = []

    def warmup_bars(self):
        return 0

    def build_order_plan(self, bars, account, mode, **kwargs):
        self.seen_symbols.append(set(bars))
        if not self.bought:
            self.bought = True
            return OrderPlan("test", mode, (OrderIntent("PARENT", "buy", 1),))
        child = account.positions.get("CHILD")
        if child is not None:
            return OrderPlan(
                "test",
                mode,
                (OrderIntent("CHILD", "sell", child.quantity, reason="ChildExit"),),
            )
        return OrderPlan("test", mode, ())


class DualPriceStreamBacktestTest(unittest.TestCase):
    @staticmethod
    def _operational_fixture():
        index = pd.date_range("2026-01-01", periods=2, freq="D")

        def frame(price):
            return pd.DataFrame(
                {
                    "Open": [price, price],
                    "High": [price, price],
                    "Low": [price, price],
                    "Close": [price, price],
                },
                index=index,
            )

        return MarketData(
            bars={
                "AAA": frame(10.0),
                "HELD": frame(20.0),
                "ACTION_LINKED_ONLY": frame(30.0),
            },
            execution_bars={
                "AAA": frame(10.0),
                "HELD": frame(20.0),
                "ACTION_LINKED_ONLY": frame(30.0),
            },
            entry_symbols=("AAA", "HELD"),
        )

    @staticmethod
    def _operational_config():
        return parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {
                    "type": "relative_strength",
                    "params": {"lookback_bars": 1},
                },
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["AAA"]},
                "capital": {"initial_cash": 1000.0},
                "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
            }
        )

    def test_paper_strategy_hides_action_linked_only_symbol_but_keeps_held_exit(self):
        config = self._operational_config()
        market_data = self._operational_fixture()
        resolved = SimpleNamespace(
            symbols=("AAA",),
            entries_allowed=True,
            exit_only_symbols=(),
        )
        captured = {}

        def build(config, bars, account, mode, **kwargs):
            captured["bars"] = set(bars)
            return OrderPlan(
                "test",
                mode,
                (OrderIntent("HELD", "sell", 1.0, reason="HeldExit"),),
            )

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "paper.json"
            state_path.write_text(
                json.dumps(
                    {
                        "cash": 1000.0,
                        "positions": {
                            "HELD": {"quantity": 1.0, "avg_price": 20.0}
                        },
                        "metadata": {},
                    }
                )
            )
            with (
                patch("supertrend_quant.runners.ensure_configured_data_ready"),
                patch("supertrend_quant.runners.resolve_universe", return_value=resolved),
                patch(
                    "supertrend_quant.runners.load_configured_market_data",
                    return_value=market_data,
                ),
                patch("supertrend_quant.runners.build_order_plan", side_effect=build),
            ):
                plan, fills = run_paper_once(config, str(state_path))

        self.assertEqual(captured["bars"], {"AAA", "HELD"})
        self.assertEqual(plan.orders[0].symbol, "HELD")
        self.assertEqual(fills, ["SELL HELD 1 @ 20.0000"])

    def test_live_strategy_hides_action_linked_only_symbol_but_keeps_held_exit(self):
        config = self._operational_config()
        market_data = self._operational_fixture()
        resolved = SimpleNamespace(
            symbols=("AAA",),
            entries_allowed=True,
            exit_only_symbols=(),
        )
        captured = {}

        class Broker:
            def __init__(self):
                self.orders = []

            def get_account(self, market):
                return AccountSnapshot(
                    cash=1000.0,
                    positions={"HELD": Position("HELD", 1.0, 20.0)},
                )

            def place_order(self, order):
                self.orders.append(order)
                return True

        broker = Broker()

        def build(config, bars, account, mode, **kwargs):
            captured["bars"] = set(bars)
            return OrderPlan(
                "test",
                mode,
                (OrderIntent("HELD", "sell", 1.0, reason="HeldExit"),),
            )

        with (
            patch("supertrend_quant.runners.ensure_configured_data_ready"),
            patch("supertrend_quant.runners.TossBroker", return_value=broker),
            patch("supertrend_quant.runners.resolve_universe", return_value=resolved),
            patch(
                "supertrend_quant.runners.load_configured_market_data",
                return_value=market_data,
            ),
            patch("supertrend_quant.runners.build_order_plan", side_effect=build),
        ):
            plan, results = run_live_once(config, assume_yes=True)

        self.assertEqual(captured["bars"], {"AAA", "HELD"})
        self.assertEqual(plan.orders[0].symbol, "HELD")
        self.assertEqual([order.symbol for order in broker.orders], ["HELD"])
        self.assertEqual(results, ["SENT SELL HELD 1"])

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
        # The position is opened at the ex-date open, after entitlement was
        # fixed, so this buyer must not receive the distribution.
        self.assertEqual(result.equity.iloc[-1], 1100.0)
        self.assertAlmostEqual(result.trade_records[0]["pnl_pct"], 0.5)
        self.assertEqual(result.trade_records[0]["corporate_action_cash"], 0.0)
        self.assertEqual(result.corporate_action_cash, 0.0)
        self.assertEqual(result.processed_corporate_action_ids, ("div-1",))
        self.assertEqual(result.data_version, "fixture-v1")

    def test_prior_close_holder_gets_dividend_before_ex_date_open_sale(self):
        config = parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {
                    "type": "relative_strength",
                    "params": {"lookback_bars": 1},
                },
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["AAA"]},
                "capital": {"initial_cash": 1000.0},
                "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
            }
        )
        index = pd.date_range("2026-01-01", periods=3, freq="D")
        frame = pd.DataFrame(
            {
                "Open": [100.0, 200.0, 300.0],
                "High": [100.0, 200.0, 300.0],
                "Low": [100.0, 200.0, 300.0],
                "Close": [100.0, 200.0, 300.0],
            },
            index=index,
        )
        market_data = MarketData(
            bars={"AAA": frame},
            execution_bars={"AAA": frame},
            corporate_actions=(
                {
                    "event_id": "holder-dividend",
                    "action_type": "cash_dividend",
                    "symbol": "AAA",
                    "ex_date": "2026-01-03",
                    "effective_date": "2026-01-03",
                    "cash_amount": 5.0,
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyThenSellStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        trade = result.trade_records[0]
        self.assertEqual(trade["exit_time"], index[2])
        self.assertEqual(trade["corporate_action_cash"], 5.0)
        self.assertAlmostEqual(trade["pnl_pct"], 0.525)
        self.assertEqual(result.corporate_action_cash, 5.0)
        self.assertEqual(result.equity.iloc[-1], 1105.0)

    def test_dividend_payment_does_not_double_count_entitlement_in_trade_pnl(self):
        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=5, freq="D")
        frame = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [100.0] * 5,
                "Low": [100.0] * 5,
                "Close": [100.0] * 5,
            },
            index=index,
        )
        market_data = MarketData(
            bars={"AAA": frame},
            execution_bars={"AAA": frame},
            corporate_actions=(
                {
                    "event_id": "paid-dividend",
                    "action_type": "cash_dividend",
                    "symbol": "AAA",
                    "ex_date": "2026-01-03",
                    "effective_date": "2026-01-03",
                    "payment_date": "2026-01-04",
                    "cash_amount": 5.0,
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyThenDelayedSellStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(len(result.trade_records), 1)
        trade = result.trade_records[0]
        self.assertEqual(trade["exit_time"], index[4])
        self.assertEqual(trade["corporate_action_cash"], 5.0)
        self.assertAlmostEqual(trade["pnl_pct"], 0.05)
        self.assertEqual(result.corporate_action_cash, 5.0)
        self.assertEqual(result.metrics["trade_count"], 1)
        self.assertEqual(result.metrics["win_rate"], 1.0)

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

    def test_preopen_split_scales_full_sell_to_post_split_holding(self):
        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=3, freq="D")
        adjusted = pd.DataFrame(
            {
                "Open": [50.0] * 3,
                "High": [50.0] * 3,
                "Low": [50.0] * 3,
                "Close": [50.0] * 3,
            },
            index=index,
        )
        raw = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 50.0],
                "High": [100.0, 100.0, 50.0],
                "Low": [100.0, 100.0, 50.0],
                "Close": [100.0, 100.0, 50.0],
            },
            index=index,
        )
        market_data = MarketData(
            bars={"AAA": adjusted},
            execution_bars={"AAA": raw},
            corporate_actions=(
                {
                    "event_id": "split-before-full-sell",
                    "action_type": "split",
                    "symbol": "AAA",
                    "effective_date": "2026-01-03",
                    "ratio": 2.0,
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyThenSellStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(len(result.trade_records), 1)
        trade = result.trade_records[0]
        self.assertEqual(trade["quantity"], 2.0)
        self.assertEqual(trade["exit_time"], index[2])
        self.assertEqual(result.metrics["trade_count"], 1)
        self.assertEqual(result.equity.iloc[-1], 1000.0)

    def test_preopen_split_buy_quantity_depends_on_signal_price_mode(self):
        index = pd.date_range("2026-01-01", periods=3, freq="D")
        adjusted = pd.DataFrame(
            {
                "Open": [50.0] * 3,
                "High": [50.0] * 3,
                "Low": [50.0] * 3,
                "Close": [50.0] * 3,
            },
            index=index,
        )
        raw = pd.DataFrame(
            {
                "Open": [100.0, 50.0, 50.0],
                "High": [100.0, 50.0, 50.0],
                "Low": [100.0, 50.0, 50.0],
                "Close": [100.0, 50.0, 50.0],
            },
            index=index,
        )
        action = {
            "event_id": "split-before-buy",
            "action_type": "split",
            "symbol": "AAA",
            "effective_date": "2026-01-02",
            "ratio": 2.0,
        }
        for price_mode, signal, expected_quantity in (
            ("total_return_adjusted", adjusted, 1.0),
            ("raw", raw, 2.0),
        ):
            with self.subTest(price_mode=price_mode):
                config = parse_config(
                    {
                        "strategy": {
                            "name": "test",
                            "type": "equal",
                            "params": {},
                        },
                        "scoring": {
                            "type": "relative_strength",
                            "params": {"lookback_bars": 1},
                        },
                        "market": "US",
                        "universe": {"source": "symbols", "symbols": ["AAA"]},
                        "capital": {"initial_cash": 1000.0},
                        "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
                        "data_store": {"price_mode": price_mode},
                    }
                )
                market_data = MarketData(
                    bars={"AAA": signal},
                    execution_bars={"AAA": raw},
                    corporate_actions=(action,),
                )
                with patch(
                    "supertrend_quant.runners.create_strategy",
                    return_value=_BuyAndHoldStrategy(),
                ):
                    result = run_backtest_on_data(config, market_data)

                self.assertEqual(len(result.trade_records), 1)
                self.assertEqual(
                    result.trade_records[0]["quantity"],
                    expected_quantity,
                )

    def test_terminal_action_cancels_prior_close_buy_without_a_holding(self):
        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=3, freq="D")
        frame = pd.DataFrame(
            {
                "Open": [100.0] * 3,
                "High": [100.0] * 3,
                "Low": [100.0] * 3,
                "Close": [100.0] * 3,
            },
            index=index,
        )
        for action_type in ("cash_merger", "delisting"):
            with self.subTest(action_type=action_type):
                market_data = MarketData(
                    bars={"OLD": frame},
                    execution_bars={"OLD": frame},
                    corporate_actions=(
                        {
                            "event_id": f"retire-before-buy-{action_type}",
                            "action_type": action_type,
                            "symbol": "OLD",
                            "effective_date": "2026-01-02",
                            "cash_amount": 100.0,
                        },
                    ),
                )
                with patch(
                    "supertrend_quant.runners.create_strategy",
                    return_value=_BuyOldAndHoldStrategy(),
                ):
                    result = run_backtest_on_data(config, market_data)

                self.assertEqual(result.trade_records, ())
                self.assertEqual(result.metrics["trade_count"], 0)
                self.assertTrue((result.equity == 1000.0).all())

    def test_terminal_action_persistently_blocks_stale_schedule_reentry(self):
        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=5, freq="D")
        frame = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [100.0] * 5,
                "Low": [100.0] * 5,
                "Close": [100.0] * 5,
            },
            index=index,
        )
        market_data = MarketData(
            bars={"OLD": frame},
            execution_bars={"OLD": frame},
            universe_schedule=(
                {
                    "effective_date": "2026-01-01",
                    "members": [
                        {"symbol": "OLD", "security_id": "SEC-OLD"}
                    ],
                },
            ),
            corporate_actions=(
                {
                    "event_id": "retire-old",
                    "security_id": "SEC-OLD",
                    "action_type": "cash_merger",
                    "symbol": "OLD",
                    "effective_date": "2026-01-02",
                    "cash_amount": 100.0,
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_RetryOldBuyStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(result.trade_records, ())
        self.assertEqual(result.metrics["trade_count"], 0)
        self.assertTrue((result.equity == 1000.0).all())

    def test_terminal_tombstone_allows_a_later_distinct_ticker_identity(self):
        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=5, freq="D")
        frame = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [100.0] * 5,
                "Low": [100.0] * 5,
                "Close": [100.0] * 5,
            },
            index=index,
        )
        market_data = MarketData(
            bars={"OLD": frame},
            execution_bars={"OLD": frame},
            universe_schedule=(
                {
                    "effective_date": "2026-01-01",
                    "members": [
                        {"symbol": "OLD", "security_id": "SEC-OLD"}
                    ],
                },
                {
                    "effective_date": "2026-01-03",
                    "members": [
                        {"symbol": "OLD", "security_id": "SEC-NEW"}
                    ],
                },
            ),
            corporate_actions=(
                {
                    "event_id": "retire-old-issuer",
                    "security_id": "SEC-OLD",
                    "action_type": "cash_merger",
                    "symbol": "OLD",
                    "effective_date": "2026-01-02",
                    "cash_amount": 100.0,
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_RetryOldBuyStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(len(result.trade_records), 1)
        # The Jan-2 signal belongs to SEC-OLD and cannot fill SEC-NEW at the
        # Jan-3 open.  SEC-NEW first becomes buyable from its own Jan-3 signal.
        self.assertEqual(result.trade_records[0]["entry_time"], index[3])
        self.assertEqual(result.metrics["trade_count"], 1)

    def test_same_day_ticker_successor_is_created_before_terminal_action(self):
        actions = (
            {
                "event_id": "terminal-hash-sorts-first",
                "action_type": "delisting",
                "symbol": "NEW",
                "effective_date": "2026-01-03",
                "cash_amount": 120.0,
            },
            {
                "event_id": "rename-hash-sorts-last",
                "action_type": "ticker_change",
                "symbol": "OLD",
                "effective_date": "2026-01-03",
                "new_symbol": "NEW",
            },
        )

        scheduled = _scheduled_corporate_actions(actions)

        self.assertEqual(
            [item[2]["event_id"] for item in scheduled],
            ["rename-hash-sorts-last", "terminal-hash-sorts-first"],
        )

        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=4, freq="D")

        def frame(price):
            return pd.DataFrame(
                {
                    "Open": [price] * len(index),
                    "High": [price] * len(index),
                    "Low": [price] * len(index),
                    "Close": [price] * len(index),
                },
                index=index,
            )

        market_data = MarketData(
            bars={"OLD": frame(100.0), "NEW": frame(120.0)},
            execution_bars={"OLD": frame(100.0), "NEW": frame(120.0)},
            corporate_actions=actions,
        )
        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyOldAndHoldStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0]["exit_reason"], "Delisting")
        self.assertEqual(result.trade_records[0]["exit_time"], index[2])
        self.assertEqual(result.corporate_action_cash, 120.0)

    def test_same_symbol_identity_successor_precedes_its_terminal_action(self):
        actions = (
            {
                "event_id": "new-identity-terminal",
                "action_type": "delisting",
                "symbol": "SAME",
                "security_id": "SEC-NEW",
                "effective_date": "2026-01-03",
                "cash_amount": 120.0,
            },
            {
                "event_id": "same-symbol-identity-change",
                "action_type": "ticker_change",
                "symbol": "SAME",
                "security_id": "SEC-OLD",
                "effective_date": "2026-01-03",
                "new_symbol": "SAME",
                "new_security_id": "SEC-NEW",
            },
        )

        scheduled = _scheduled_corporate_actions(actions)

        self.assertEqual(
            [item[2]["event_id"] for item in scheduled],
            ["same-symbol-identity-change", "new-identity-terminal"],
        )

    def test_same_day_ticker_dependency_cycle_fails_closed(self):
        actions = (
            {
                "event_id": "a-to-b",
                "action_type": "ticker_change",
                "symbol": "A",
                "effective_date": "2026-01-03",
                "new_symbol": "B",
            },
            {
                "event_id": "b-to-a",
                "action_type": "ticker_change",
                "symbol": "B",
                "effective_date": "2026-01-03",
                "new_symbol": "A",
            },
        )

        with self.assertRaisesRegex(ValueError, "dependency is cyclic"):
            _scheduled_corporate_actions(actions)

    def test_same_day_ticker_chain_uses_successor_topology_not_event_hash(self):
        actions = (
            {
                "event_id": "00-b-to-c",
                "action_type": "ticker_change",
                "symbol": "B",
                "effective_date": "2026-01-03",
                "new_symbol": "C",
            },
            {
                "event_id": "99-a-to-b",
                "action_type": "ticker_change",
                "symbol": "A",
                "effective_date": "2026-01-03",
                "new_symbol": "B",
            },
        )

        scheduled = _scheduled_corporate_actions(actions)

        self.assertEqual(
            [item[2]["event_id"] for item in scheduled],
            ["99-a-to-b", "00-b-to-c"],
        )

    def test_fisv_round_trip_reactivates_only_the_exact_successor_identity(self):
        retired = {"FISV": {"SEC-ANCIENT", "SEC-FISV"}}
        _reactivate_retired_symbol(retired, "FISV", "SEC-FISV")
        self.assertEqual(retired, {"FISV": {"SEC-ANCIENT"}})

        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=5, freq="D")

        def frame(price):
            return pd.DataFrame(
                {
                    "Open": [price] * len(index),
                    "High": [price] * len(index),
                    "Low": [price] * len(index),
                    "Close": [price] * len(index),
                },
                index=index,
            )

        market_data = MarketData(
            bars={"FISV": frame(100.0), "FI": frame(100.0)},
            execution_bars={"FISV": frame(100.0), "FI": frame(100.0)},
            universe_schedule=(
                {
                    "effective_date": "2026-01-01",
                    "members": [
                        {"symbol": "FISV", "security_id": "SEC-FISV"}
                    ],
                },
                {
                    "effective_date": "2026-01-02",
                    "members": [
                        {"symbol": "FI", "security_id": "SEC-FISV"}
                    ],
                },
                {
                    "effective_date": "2026-01-03",
                    "members": [
                        {"symbol": "FISV", "security_id": "SEC-FISV"}
                    ],
                },
            ),
            corporate_actions=(
                {
                    "event_id": "fisv-to-fi",
                    "security_id": "SEC-FISV",
                    "action_type": "ticker_change",
                    "symbol": "FISV",
                    "effective_date": "2026-01-02",
                    "new_security_id": "SEC-FISV",
                    "new_symbol": "FI",
                },
                {
                    "event_id": "fi-to-fisv",
                    "security_id": "SEC-FISV",
                    "action_type": "ticker_change",
                    "symbol": "FI",
                    "effective_date": "2026-01-03",
                    "new_security_id": "SEC-FISV",
                    "new_symbol": "FISV",
                },
            ),
        )
        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyActiveSymbolStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0]["symbol"], "FISV")
        self.assertEqual(result.trade_records[0]["entry_time"], index[3])

    def test_ticker_and_stock_merger_transfer_entry_bookkeeping(self):
        config = parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["OLD"]},
                "capital": {"initial_cash": 1000.0},
                "costs": {"fee_rate": 0.01, "slippage_rate": 0.0},
            }
        )
        index = pd.date_range("2026-01-01", periods=5, freq="D")

        def frame(price):
            return pd.DataFrame(
                {
                    "Open": [price] * len(index),
                    "High": [price] * len(index),
                    "Low": [price] * len(index),
                    "Close": [price] * len(index),
                },
                index=index,
            )

        bars = {"OLD": frame(100.0), "MID": frame(100.0), "NEW": frame(200.0)}
        market_data = MarketData(
            bars=bars,
            execution_bars=bars,
            corporate_actions=(
                {
                    "event_id": "dividend",
                    "action_type": "cash_dividend",
                    "symbol": "OLD",
                    "effective_date": "2026-01-03",
                    "cash_amount": 2.0,
                },
                {
                    "event_id": "rename",
                    "action_type": "ticker_change",
                    "symbol": "OLD",
                    "effective_date": "2026-01-03",
                    "new_symbol": "MID",
                },
                {
                    "event_id": "mixed-merger",
                    "action_type": "stock_merger",
                    "symbol": "MID",
                    "effective_date": "2026-01-04",
                    "new_symbol": "NEW",
                    "ratio": 0.5,
                    "cash_amount": 10.0,
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyOldAndHoldStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        trade = result.trade_records[0]
        self.assertEqual(trade["symbol"], "NEW")
        self.assertEqual(trade["entry_time"], index[1])
        self.assertEqual(trade["corporate_action_cash"], 12.0)
        self.assertAlmostEqual(trade["pnl_pct"], 111.0 / 101.0 - 1.0)
        self.assertEqual(result.corporate_action_cash, 12.0)
        self.assertEqual(
            result.processed_corporate_action_ids,
            ("dividend", "mixed-merger", "rename"),
        )

    def test_same_day_spinoff_and_dividend_precede_ticker_change(self):
        config = parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {
                    "type": "relative_strength",
                    "params": {"lookback_bars": 1},
                },
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["OLD"]},
                "capital": {"initial_cash": 1000.0},
                "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
            }
        )
        index = pd.date_range("2026-01-01", periods=5, freq="D")

        def frame(values, selected_index=index):
            return pd.DataFrame(
                {
                    "Open": values,
                    "High": values,
                    "Low": values,
                    "Close": values,
                },
                index=selected_index,
            )

        bars = {
            "OLD": frame([100.0] * 5),
            "NEW": frame([80.0] * 5),
            "CHILD": frame([20.0] * 3, index[2:]),
        }
        market_data = MarketData(
            bars=bars,
            execution_bars=bars,
            entry_symbols=("OLD",),
            corporate_actions=(
                {
                    "event_id": "a-rename",
                    "action_type": "ticker_change",
                    "symbol": "OLD",
                    "effective_date": "2026-01-03",
                    "new_symbol": "NEW",
                },
                {
                    "event_id": "m-dividend",
                    "action_type": "cash_dividend",
                    "symbol": "OLD",
                    "ex_date": "2026-01-03",
                    "effective_date": "2026-01-03",
                    "cash_amount": 2.0,
                },
                {
                    "event_id": "z-spin",
                    "action_type": "spinoff",
                    "symbol": "OLD",
                    "effective_date": "2026-01-03",
                    "new_symbol": "CHILD",
                    "ratio": 1.0,
                    "metadata": {"cost_basis_fraction": 0.2},
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyOldAndHoldStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        trades = {trade["symbol"]: trade for trade in result.trade_records}
        self.assertEqual(set(trades), {"NEW", "CHILD"})
        self.assertEqual(trades["NEW"]["entry_time"], index[1])
        self.assertEqual(trades["CHILD"]["entry_time"], index[1])
        self.assertEqual(trades["NEW"]["corporate_action_cash"], 2.0)
        self.assertEqual(trades["NEW"]["entry_price"], 80.0)
        self.assertEqual(trades["CHILD"]["entry_price"], 20.0)
        self.assertEqual(result.corporate_action_cash, 2.0)
        self.assertEqual(result.equity.iloc[-1], 1002.0)

    def test_preopen_ticker_change_remaps_old_sell_to_new_symbol(self):
        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=4, freq="D")

        def frame(price, selected_index):
            return pd.DataFrame(
                {
                    "Open": [price] * len(selected_index),
                    "High": [price] * len(selected_index),
                    "Low": [price] * len(selected_index),
                    "Close": [price] * len(selected_index),
                },
                index=selected_index,
            )

        bars = {
            "OLD": frame(100.0, index[:2]),
            "NEW": frame(100.0, index[2:]),
        }
        market_data = MarketData(
            bars=bars,
            execution_bars=bars,
            universe_schedule=(
                {"effective_date": "2026-01-01", "symbols": ["OLD"]},
                {"effective_date": "2026-01-03", "symbols": ["NEW"]},
            ),
            corporate_actions=(
                {
                    "event_id": "rename-at-open",
                    "action_type": "ticker_change",
                    "symbol": "OLD",
                    "effective_date": "2026-01-03",
                    "new_symbol": "NEW",
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyOldThenSellStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0]["symbol"], "NEW")
        self.assertEqual(result.trade_records[0]["exit_time"], index[2])
        self.assertEqual(result.trade_records[0]["exit_reason"], "OldExit")

    def test_preopen_stock_merger_scales_full_sell_to_converted_quantity(self):
        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=4, freq="D")

        def frame(price, selected_index):
            return pd.DataFrame(
                {
                    "Open": [price] * len(selected_index),
                    "High": [price] * len(selected_index),
                    "Low": [price] * len(selected_index),
                    "Close": [price] * len(selected_index),
                },
                index=selected_index,
            )

        bars = {
            "OLD": frame(100.0, index[:2]),
            "NEW": frame(50.0, index[2:]),
        }
        market_data = MarketData(
            bars=bars,
            execution_bars=bars,
            universe_schedule=(
                {"effective_date": "2026-01-01", "symbols": ["OLD"]},
                {"effective_date": "2026-01-03", "symbols": ["NEW"]},
            ),
            corporate_actions=(
                {
                    "event_id": "two-for-one-merger",
                    "action_type": "stock_merger",
                    "symbol": "OLD",
                    "effective_date": "2026-01-03",
                    "new_symbol": "NEW",
                    "ratio": 2.0,
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyOldThenSellStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(len(result.trade_records), 1)
        trade = result.trade_records[0]
        self.assertEqual(trade["symbol"], "NEW")
        self.assertEqual(trade["quantity"], 2.0)
        self.assertEqual(trade["exit_time"], index[2])
        self.assertEqual(trade["exit_reason"], "OldExit")
        self.assertEqual(result.metrics["trade_count"], 1)
        self.assertEqual(result.equity.iloc[-1], 1000.0)

    def test_cash_merger_and_delisting_create_terminal_trade_metrics(self):
        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=4, freq="D")
        frame = pd.DataFrame(
            {
                "Open": [100.0] * 4,
                "High": [100.0] * 4,
                "Low": [100.0] * 4,
                "Close": [100.0] * 4,
            },
            index=index,
        )
        cases = (
            ("cash_merger", 120.0, "CashMerger", 0.2, 1.0),
            ("delisting", 80.0, "Delisting", -0.2, 0.0),
        )
        for action_type, settlement, reason, expected_pnl, win_rate in cases:
            with self.subTest(action_type=action_type):
                market_data = MarketData(
                    bars={"OLD": frame},
                    execution_bars={"OLD": frame},
                    corporate_actions=(
                        {
                            "event_id": f"terminal-{action_type}",
                            "action_type": action_type,
                            "symbol": "OLD",
                            "effective_date": "2026-01-03",
                            "cash_amount": settlement,
                        },
                    ),
                )
                with patch(
                    "supertrend_quant.runners.create_strategy",
                    return_value=_BuyOldAndHoldStrategy(),
                ):
                    result = run_backtest_on_data(
                        config, market_data, capture_artifacts=True
                    )

                self.assertEqual(len(result.trade_records), 1)
                trade = result.trade_records[0]
                self.assertEqual(trade["exit_time"], index[2])
                self.assertEqual(trade["exit_price"], settlement)
                self.assertEqual(trade["exit_reason"], reason)
                self.assertEqual(
                    trade["corporate_action_event_id"],
                    f"terminal-{action_type}",
                )
                self.assertAlmostEqual(trade["pnl_pct"], expected_pnl)
                self.assertEqual(result.metrics["trade_count"], 1)
                self.assertEqual(result.metrics["win_rate"], win_rate)
                self.assertEqual(
                    result.metrics["payoff_ratio"],
                    float("inf") if expected_pnl > 0 else 0.0,
                )
                self.assertEqual(result.corporate_action_cash, settlement)
                terminal_fill = result.artifacts.fills[-1]
                self.assertEqual(terminal_fill["event_type"], "corporate_action")
                self.assertEqual(terminal_fill["side"], "sell")
                self.assertEqual(terminal_fill["fill_price"], settlement)

    def test_preopen_ticker_change_cancels_stale_old_buy(self):
        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=3, freq="D")

        def frame(symbol_index):
            return pd.DataFrame(
                {
                    "Open": [100.0] * len(symbol_index),
                    "High": [100.0] * len(symbol_index),
                    "Low": [100.0] * len(symbol_index),
                    "Close": [100.0] * len(symbol_index),
                },
                index=symbol_index,
            )

        market_data = MarketData(
            bars={"OLD": frame(index[:1]), "NEW": frame(index[1:])},
            execution_bars={"OLD": frame(index[:1]), "NEW": frame(index[1:])},
            universe_schedule=(
                {"effective_date": "2026-01-01", "symbols": ["OLD"]},
                {"effective_date": "2026-01-02", "symbols": ["NEW"]},
            ),
            corporate_actions=(
                {
                    "event_id": "rename-before-buy",
                    "action_type": "ticker_change",
                    "symbol": "OLD",
                    "effective_date": "2026-01-02",
                    "new_symbol": "NEW",
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyOldAndHoldStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(result.trade_records, ())
        self.assertTrue((result.equity == config.capital.initial_cash).all())

    def test_spinoff_child_is_exit_only_and_inherits_cost_basis_and_entry_time(self):
        config = parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {
                    "type": "relative_strength",
                    "params": {"lookback_bars": 1},
                },
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["PARENT"]},
                "capital": {"initial_cash": 1000.0},
                "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
            }
        )
        index = pd.date_range("2026-01-01", periods=5, freq="D")

        def frame(values, frame_index=index):
            return pd.DataFrame(
                {
                    "Open": values,
                    "High": values,
                    "Low": values,
                    "Close": values,
                },
                index=frame_index,
            )

        bars = {
            "PARENT": frame([100.0, 100.0, 80.0, 80.0, 80.0]),
            # The child has no pre-distribution history.  It must not truncate
            # the static-universe backtest timeline or become a buy candidate.
            "CHILD": frame([20.0, 25.0, 25.0], index[2:]),
        }
        market_data = MarketData(
            bars=bars,
            execution_bars=bars,
            entry_symbols=("PARENT",),
            corporate_actions=(
                {
                    "event_id": "spin",
                    "action_type": "spinoff",
                    "symbol": "PARENT",
                    "effective_date": "2026-01-03",
                    "new_symbol": "CHILD",
                    "ratio": 1.0,
                    "metadata": {"cost_basis_fraction": 0.2},
                },
            ),
        )
        strategy = _SpinLifecycleStrategy()

        with patch("supertrend_quant.runners.create_strategy", return_value=strategy):
            result = run_backtest_on_data(config, market_data)

        self.assertEqual(strategy.seen_symbols[0], {"PARENT"})
        self.assertEqual(strategy.seen_symbols[1], {"PARENT"})
        self.assertEqual(strategy.seen_symbols[2], {"PARENT", "CHILD"})
        child_trade = next(
            trade for trade in result.trade_records if trade["symbol"] == "CHILD"
        )
        self.assertEqual(child_trade["entry_time"], index[1])
        self.assertEqual(child_trade["entry_price"], 20.0)
        self.assertEqual(child_trade["exit_price"], 25.0)
        self.assertAlmostEqual(child_trade["pnl_pct"], 0.25)
        self.assertEqual(result.equity.index[0], index[0])
        self.assertEqual(result.processed_corporate_action_ids, ("spin",))

    def test_official_exit_mark_closes_child_once_and_settles_cash_next_session(self):
        config = parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {
                    "type": "relative_strength",
                    "params": {"lookback_bars": 1},
                },
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["PARENT"]},
                "capital": {"initial_cash": 1000.0},
                "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
            }
        )
        index = pd.to_datetime(
            ["2019-11-19", "2019-11-20", "2019-11-21", "2019-11-22"]
        )

        def frame(values, selected_index=index):
            return pd.DataFrame(
                {
                    "Open": values,
                    "High": values,
                    "Low": values,
                    "Close": values,
                },
                index=selected_index,
            )

        bars = {
            # Keep the static entry timeline through settlement; the merger
            # action retires the held parent before the later placeholder bars
            # can be traded or valued.
            "PARENT": frame([100.0, 100.0, 100.0, 100.0]),
            "NEW": frame([56.48, 56.48], index[2:]),
            "CHILD": frame([2.30], index[2:3]),
        }
        market_data = MarketData(
            bars=bars,
            execution_bars=bars,
            entry_symbols=("PARENT",),
            corporate_actions=(
                {
                    "event_id": "cvr-spin",
                    "action_type": "spinoff",
                    "symbol": "PARENT",
                    "effective_date": "2019-11-21",
                    "new_symbol": "CHILD",
                    "ratio": 1.0,
                    "metadata": {"cost_basis_fraction": 2.30 / 108.78},
                },
                {
                    "event_id": "stock-merger",
                    "action_type": "stock_merger",
                    "symbol": "PARENT",
                    "effective_date": "2019-11-21",
                    "new_symbol": "NEW",
                    "ratio": 1.0,
                    "cash_amount": 50.0,
                },
                {
                    "event_id": "official-exit-mark",
                    "action_type": "delisting",
                    "symbol": "CHILD",
                    "effective_date": "2019-11-21",
                    "payment_date": "2019-11-22",
                    "cash_amount": 2.30,
                    "metadata": {
                        "mode": "official_exit_mark",
                        "exit_only": True,
                        "execution_timing": "first_tradable_session_close",
                        "cash_available_session": "2019-11-22",
                    },
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyParentAndHoldStrategy(),
        ):
            result = run_backtest_on_data(config, market_data)

        child_trades = [
            trade for trade in result.trade_records if trade["symbol"] == "CHILD"
        ]
        self.assertEqual(len(child_trades), 1)
        self.assertEqual(child_trades[0]["exit_time"], index[2])
        self.assertEqual(child_trades[0]["exit_price"], 2.30)
        self.assertEqual(child_trades[0]["exit_reason"], "Delisting")
        self.assertAlmostEqual(result.corporate_action_cash, 52.30)
        self.assertEqual(
            result.processed_corporate_action_ids,
            ("cvr-spin", "official-exit-mark", "stock-merger"),
        )
        self.assertAlmostEqual(float(result.equity.loc[index[2]]), 1008.78)
        self.assertAlmostEqual(float(result.equity.loc[index[3]]), 1008.78)

    def test_stale_held_spinoff_price_fails_closed_on_exact_session(self):
        config = parse_config(
            {
                "strategy": {"name": "test", "type": "equal", "params": {}},
                "scoring": {
                    "type": "relative_strength",
                    "params": {"lookback_bars": 1},
                },
                "market": "US",
                "universe": {"source": "symbols", "symbols": ["PARENT"]},
                "capital": {"initial_cash": 1000.0},
                "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
            }
        )
        index = pd.date_range("2026-01-01", periods=5, freq="D")
        parent = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [100.0] * 5,
                "Low": [100.0] * 5,
                "Close": [100.0] * 5,
            },
            index=index,
        )
        child_signal = pd.DataFrame(
            {
                "Open": [20.0, 20.0, 20.0],
                "High": [20.0, 20.0, 20.0],
                "Low": [20.0, 20.0, 20.0],
                "Close": [20.0, 20.0, 20.0],
            },
            index=index[2:],
        )
        market_data = MarketData(
            bars={"PARENT": parent, "CHILD": child_signal},
            execution_bars={
                "PARENT": parent,
                "CHILD": child_signal.iloc[:1],
            },
            entry_symbols=("PARENT",),
            corporate_actions=(
                {
                    "event_id": "spin-without-price",
                    "action_type": "spinoff",
                    "symbol": "PARENT",
                    "effective_date": "2026-01-03",
                    "new_symbol": "CHILD",
                    "ratio": 1.0,
                    "metadata": {"cost_basis_fraction": 0.2},
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyParentAndHoldStrategy(),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "CHILD/2026-01-04",
            ):
                run_backtest_on_data(config, market_data)

    def test_incomplete_held_action_stops_backtest(self):
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
                    "effective_date": "2026-01-03",
                    "cash_amount": None,
                },
            ),
        )
        with patch("supertrend_quant.runners.create_strategy", return_value=_BuyAndHoldStrategy()):
            with self.assertRaisesRegex(
                RuntimeError,
                "incomplete/cash_dividend",
            ):
                run_backtest_on_data(config, market_data)

    def test_missing_held_spinoff_cost_basis_stops_backtest(self):
        config = self._operational_config()
        index = pd.date_range("2026-01-01", periods=4, freq="D")

        def frame(price):
            return pd.DataFrame(
                {
                    "Open": [price] * 4,
                    "High": [price] * 4,
                    "Low": [price] * 4,
                    "Close": [price] * 4,
                },
                index=index,
            )

        market_data = MarketData(
            bars={"PARENT": frame(100.0), "CHILD": frame(20.0)},
            execution_bars={"PARENT": frame(100.0), "CHILD": frame(20.0)},
            entry_symbols=("PARENT",),
            corporate_actions=(
                {
                    "event_id": "spin-missing-basis",
                    "action_type": "spinoff",
                    "symbol": "PARENT",
                    "effective_date": "2026-01-03",
                    "new_symbol": "CHILD",
                    "ratio": 1.0,
                },
            ),
        )

        with patch(
            "supertrend_quant.runners.create_strategy",
            return_value=_BuyParentAndHoldStrategy(),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "spin-missing-basis/spinoff",
            ):
                run_backtest_on_data(config, market_data)


if __name__ == "__main__":
    unittest.main()
