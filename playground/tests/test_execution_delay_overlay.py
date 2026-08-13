from __future__ import annotations

import sys
import unittest
from pathlib import Path


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLAYGROUND_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "unified_quant" / "src"))
sys.path.insert(0, str(PLAYGROUND_ROOT))

from research_extensions.execution_delay_overlay import (  # noqa: E402
    OneSessionDelayedPreparedBacktest,
)
from supertrend_quant.portfolio import (  # noqa: E402
    AccountSnapshot,
    OrderIntent,
    OrderPlan,
)


class Delegate:
    def __init__(self) -> None:
        self.calls = []

    def report_frames(self, symbols):
        return {}

    def build_order_plan(self, signal_ts, account, mode="backtest"):
        self.calls.append(signal_ts)
        return OrderPlan(
            "test",
            mode,
            (
                OrderIntent(
                    symbol="A",
                    side="buy",
                    quantity=1,
                    reason=f"signal-{signal_ts}",
                ),
            ),
        )


class ExecutionDelayTests(unittest.TestCase):
    def test_nonempty_plan_is_released_on_next_signal_session(self):
        delegate = Delegate()
        delayed = OneSessionDelayedPreparedBacktest(delegate)
        account = AccountSnapshot(cash=1000.0)

        queued = delayed.build_order_plan("2026-01-05", account)
        released = delayed.build_order_plan("2026-01-06", account)
        next_queued = delayed.build_order_plan("2026-01-07", account)

        self.assertFalse(queued.orders)
        self.assertEqual(len(released.orders), 1)
        self.assertEqual(released.orders[0].reason, "signal-2026-01-05")
        self.assertFalse(next_queued.orders)
        self.assertEqual(delegate.calls, ["2026-01-05", "2026-01-07"])

    def test_empty_plan_is_not_queued(self):
        delegate = Delegate()
        delegate.build_order_plan = lambda signal_ts, account, mode="backtest": (
            OrderPlan("test", mode, ())
        )
        delayed = OneSessionDelayedPreparedBacktest(delegate)

        result = delayed.build_order_plan(
            "2026-01-05",
            AccountSnapshot(cash=1000.0),
        )

        self.assertFalse(result.orders)
        self.assertIsNone(delayed.pending_plan)


if __name__ == "__main__":
    unittest.main()
