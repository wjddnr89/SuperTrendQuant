from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLAYGROUND_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "unified_quant" / "src"))
sys.path.insert(0, str(PLAYGROUND_ROOT))

from research_extensions.cold_start_overlay import (  # noqa: E402
    ColdStartPreparedBacktest,
    ColdStartRule,
)
from scripts.compare_nasdaq_three_accounts_cold_start import (  # noqa: E402
    window_market_data,
)
from supertrend_quant.data import MarketData  # noqa: E402
from supertrend_quant.portfolio import (  # noqa: E402
    AccountSnapshot,
    OrderIntent,
    OrderPlan,
    Position,
)


class Delegate:
    def __init__(self, symbols):
        self.symbols = iter(symbols)
        self.calls = 0

    def report_frames(self, symbols):
        return {}

    def build_order_plan(self, signal_ts, account, mode="backtest"):
        self.calls += 1
        symbol = next(self.symbols, None)
        orders = () if symbol is None else (
            OrderIntent(symbol=symbol, side="buy", quantity=1),
        )
        return OrderPlan("test", mode, orders)


class ColdStartOverlayTests(unittest.TestCase):
    def test_window_slice_retains_requested_warmup_sessions(self):
        index = pd.bdate_range("2026-01-01", periods=10)
        frame = pd.DataFrame(
            {"Open": range(10), "Close": range(10)},
            index=index,
        )
        data = MarketData(bars={"TEST": frame}, execution_bars={"TEST": frame})

        sliced = window_market_data(
            data,
            index[6:9],
            index,
            warmup_sessions=3,
        )

        self.assertEqual(sliced.bars["TEST"].index[0], index[3])
        self.assertEqual(sliced.bars["TEST"].index[-1], index[8])
        self.assertEqual(sliced.execution_bars["TEST"].index[0], index[3])

    def test_skip_first_leader_waits_until_leader_changes(self):
        normal = Delegate(["AMD", "AMD", "FTNT", "NVDA"])
        overlay = ColdStartPreparedBacktest(
            normal,
            ColdStartRule("skip_first_leader", "Skip first leader"),
        )
        flat = AccountSnapshot(cash=10_000.0)

        first = overlay.build_order_plan("2026-01-05", flat)
        second = overlay.build_order_plan("2026-01-06", flat)
        third = overlay.build_order_plan("2026-01-07", flat)

        self.assertFalse(first.orders)
        self.assertFalse(second.orders)
        self.assertEqual(third.orders[0].symbol, "FTNT")

    def test_guarded_delegate_is_used_only_for_initial_entry(self):
        normal = Delegate(["NVDA"])
        guarded = Delegate([None, "FTNT"])
        overlay = ColdStartPreparedBacktest(
            normal,
            ColdStartRule("fresh_only", "Fresh only"),
            guarded_delegate=guarded,
        )
        flat = AccountSnapshot(cash=10_000.0)

        waiting = overlay.build_order_plan("2026-01-05", flat)
        entry = overlay.build_order_plan("2026-01-06", flat)
        held = AccountSnapshot(
            cash=100.0,
            positions={"FTNT": Position("FTNT", 1, 100.0)},
        )
        normal_plan = overlay.build_order_plan("2026-01-07", held)

        self.assertFalse(waiting.orders)
        self.assertEqual(entry.orders[0].symbol, "FTNT")
        self.assertEqual(normal_plan.orders[0].symbol, "NVDA")
        self.assertEqual(guarded.calls, 2)
        self.assertEqual(normal.calls, 1)


if __name__ == "__main__":
    unittest.main()
