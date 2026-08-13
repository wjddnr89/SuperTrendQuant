from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLAYGROUND_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "unified_quant" / "src"))
sys.path.insert(0, str(PLAYGROUND_ROOT))

from research_extensions.seasonal_overlay import (  # noqa: E402
    HalloweenPreparedBacktest,
    halloween_allows_execution,
)
from scripts.compare_nasdaq100_market_filters_halloween_seasons import (  # noqa: E402
    period_return,
)
from supertrend_quant.portfolio import (  # noqa: E402
    AccountSnapshot,
    OrderPlan,
    Position,
)


class Delegate:
    def __init__(self) -> None:
        self.calls = 0
        self.delegate = SimpleNamespace(
            strategy=SimpleNamespace(
                config=SimpleNamespace(
                    strategy=SimpleNamespace(name="test_strategy")
                )
            )
        )

    def report_frames(self, symbols):
        return {}

    def build_order_plan(self, signal_ts, account, mode="backtest"):
        self.calls += 1
        return OrderPlan("test_strategy", mode, ())


class HalloweenOverlayTests(unittest.TestCase):
    def test_active_month_definition(self):
        self.assertTrue(halloween_allows_execution("2024-04-01"))
        self.assertTrue(halloween_allows_execution("2024-11-01"))
        self.assertFalse(halloween_allows_execution("2024-05-01"))
        self.assertFalse(halloween_allows_execution("2024-10-31"))

    def test_last_april_signal_liquidates_at_first_may_open(self):
        delegate = Delegate()
        index = pd.DatetimeIndex(["2024-04-30", "2024-05-01"])
        overlay = HalloweenPreparedBacktest(delegate, index)
        account = AccountSnapshot(
            cash=0.0,
            positions={"A": Position("A", 3.0, 100.0)},
        )

        plan = overlay.build_order_plan(index[0], account)

        self.assertEqual(delegate.calls, 0)
        self.assertEqual(len(plan.orders), 1)
        self.assertEqual(plan.orders[0].side, "sell")
        self.assertEqual(plan.orders[0].reason, "Halloween seasonal exit")

    def test_last_october_signal_delegates_for_first_november_open(self):
        delegate = Delegate()
        index = pd.DatetimeIndex(["2024-10-31", "2024-11-01"])
        overlay = HalloweenPreparedBacktest(delegate, index)

        overlay.build_order_plan(index[0], AccountSnapshot(cash=1000.0))

        self.assertEqual(delegate.calls, 1)

    def test_cycle_return_uses_pre_november_and_post_may_boundaries(self):
        index = pd.DatetimeIndex(
            [
                "2023-10-31",
                "2023-11-01",
                "2024-04-30",
                "2024-05-01",
            ]
        )
        equity = pd.Series([100.0, 99.0, 109.0, 110.0], index=index)

        result = period_return(
            equity,
            start_boundary=pd.Timestamp("2023-10-31"),
            end_boundary=pd.Timestamp("2024-05-01"),
        )

        self.assertIsNotNone(result)
        start_ts, end_ts, total_return, _ = result
        self.assertEqual(start_ts, pd.Timestamp("2023-10-31"))
        self.assertEqual(end_ts, pd.Timestamp("2024-05-01"))
        self.assertAlmostEqual(total_return, 0.10)


if __name__ == "__main__":
    unittest.main()
