from __future__ import annotations

import sys
import unittest
from pathlib import Path

from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "unified_quant" / "src"))
sys.path.insert(0, str(ROOT))

from supertrend_quant.portfolio import (  # noqa: E402
    AccountSnapshot,
    OrderPlan,
    Position,
    PositionEconomics,
)
from risk_overlay import (  # noqa: E402
    PortfolioRiskPolicy,
    PortfolioRiskPreparedBacktest,
    atr_target_weight,
)


class EmptyDelegate:
    def __init__(self):
        self.calls = 0
        self.strategy = SimpleNamespace(
            config=SimpleNamespace(
                costs=SimpleNamespace(fee_rate=0.001, slippage_rate=0.0005)
            )
        )
        self.prepared = {}

    def build_order_plan(self, signal_ts, account, mode="backtest"):
        self.calls += 1
        return OrderPlan("delegate", mode, ())


class AtrWeightTests(unittest.TestCase):
    def test_max_one_scales_high_atr_to_half(self):
        self.assertAlmostEqual(
            atr_target_weight(0.05, max_positions=1, target_portfolio_atr_pct=0.025),
            0.5,
        )

    def test_max_two_caps_each_slot_at_half(self):
        self.assertAlmostEqual(
            atr_target_weight(0.02, max_positions=2, target_portfolio_atr_pct=0.025),
            0.5,
        )

    def test_max_two_scales_high_atr_to_quarter(self):
        self.assertAlmostEqual(
            atr_target_weight(0.05, max_positions=2, target_portfolio_atr_pct=0.025),
            0.25,
        )

    def test_invalid_atr_gets_zero_weight(self):
        self.assertEqual(
            atr_target_weight(float("nan"), max_positions=1, target_portfolio_atr_pct=0.025),
            0.0,
        )

    def test_drawdown_brake_liquidates_and_waits_twenty_sessions(self):
        delegate = EmptyDelegate()
        policy = PortfolioRiskPolicy(
            "brake",
            1,
            drawdown_stop_pct=0.15,
            cooldown_sessions=20,
        )
        wrapper = PortfolioRiskPreparedBacktest(delegate, policy)
        wrapper.build_order_plan(0, AccountSnapshot(100.0))
        account = AccountSnapshot(
            0.0,
            {"AAA": Position("AAA", 1, 100.0)},
            position_economics={
                "AAA": PositionEconomics(
                    entry_cost=100.0,
                    raw_mark=84.0,
                    estimated_exit_proceeds=84.0,
                )
            },
        )
        plan = wrapper.build_order_plan(1, account)
        self.assertEqual(wrapper.drawdown_stop_count, 1)
        self.assertEqual(len(plan.orders), 1)
        self.assertEqual(plan.orders[0].side, "sell")
        for signal_number in range(2, 22):
            cooldown_plan = wrapper.build_order_plan(signal_number, AccountSnapshot(84.0))
            self.assertFalse(cooldown_plan.orders)
        wrapper.build_order_plan(22, AccountSnapshot(84.0))
        self.assertEqual(delegate.calls, 2)


if __name__ == "__main__":
    unittest.main()
