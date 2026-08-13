from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "unified_quant" / "src"))
sys.path.insert(0, str(ROOT))

from supertrend_quant.portfolio import AccountSnapshot, OrderPlan, Position  # noqa: E402
from regime_overlay import (  # noqa: E402
    RegimeManagedPreparedBacktest,
    RegimePolicy,
    build_regime_states,
)


class EmptyDelegate:
    def build_order_plan(self, signal_ts, account, mode="backtest"):
        return OrderPlan("delegate", mode, ())


class RegimeStateTests(unittest.TestCase):
    def benchmark(self, count: int = 12) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=count, freq="D")
        return pd.DataFrame({"Close": [100.0 + value for value in range(count)]}, index=index)

    def test_st_confirmed_uses_three_down_and_two_up_bars(self):
        benchmark = self.benchmark()
        trend = pd.Series([1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1], index=benchmark.index)
        policy = RegimePolicy("test", "st_confirmed", "full_exit", bear_confirm_bars=3, bull_confirm_bars=2)
        states = build_regime_states(trend, benchmark, policy)
        self.assertEqual(
            states.tolist(),
            ["neutral", "bull", "neutral", "neutral", "bear", "bear", "neutral", "bull", "bull", "neutral", "neutral", "bear"],
        )

    def test_three_state_distinguishes_mixed_regime(self):
        benchmark = self.benchmark()
        trend = pd.Series([1] * 6 + [-1] * 6, index=benchmark.index)
        policy = RegimePolicy("test", "three_state", "full_exit")
        states = build_regime_states(trend, benchmark, policy)
        self.assertEqual(states.iloc[0], "neutral")
        self.assertTrue((states.iloc[1:6] == "bull").all())
        self.assertTrue((states.iloc[6:] == "neutral").all())

    def test_future_trend_change_does_not_change_past_state(self):
        benchmark = self.benchmark()
        trend = pd.Series([1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1], index=benchmark.index)
        policy = RegimePolicy("test", "st_confirmed", "full_exit", bear_confirm_bars=3, bull_confirm_bars=2)
        original = build_regime_states(trend, benchmark, policy)
        changed = trend.copy()
        changed.iloc[8:] = 1
        mutated = build_regime_states(changed, benchmark, policy)
        pd.testing.assert_series_equal(original.iloc[:8], mutated.iloc[:8])

    def test_bear_full_exit_sells_all_positions(self):
        index = pd.date_range("2024-01-01", periods=2, freq="D")
        states = pd.Series(["bull", "bear"], index=index)
        policy = RegimePolicy("exit", "st_immediate", "full_exit")
        wrapper = RegimeManagedPreparedBacktest(EmptyDelegate(), policy, states)
        account = AccountSnapshot(100.0, {"AAA": Position("AAA", 7, 10.0)})
        plan = wrapper.build_order_plan(index[1], account)
        self.assertEqual(len(plan.orders), 1)
        self.assertEqual(plan.orders[0].side, "sell")
        self.assertEqual(plan.orders[0].quantity, 7)

    def test_trim_transition_is_sell_then_half_rebuy(self):
        index = pd.date_range("2024-01-01", periods=2, freq="D")
        states = pd.Series(["bull", "bear"], index=index)
        policy = RegimePolicy("trim", "st_immediate", "rebalance_50")
        wrapper = RegimeManagedPreparedBacktest(EmptyDelegate(), policy, states)
        account = AccountSnapshot(100.0, {"AAA": Position("AAA", 7, 10.0)})
        plan = wrapper.build_order_plan(index[1], account)
        self.assertEqual([order.side for order in plan.orders], ["sell", "buy"])
        self.assertEqual(plan.orders[1].cash_allocation_pct, 0.5)
        self.assertEqual(plan.orders[1].required_sell_symbols, ("AAA",))


if __name__ == "__main__":
    unittest.main()
