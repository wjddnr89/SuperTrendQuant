from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from supertrend_quant.config import load_split_config
from supertrend_quant.holdings import HoldingsStore
from supertrend_quant.live_runtime import HybridLiveRuntime
from supertrend_quant.portfolio import AccountSnapshot, OrderIntent, OrderPlan, Position
from supertrend_quant.universe import ResolvedUniverse, UniverseMember, UniverseSnapshot


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_ROOT = REPOSITORY_ROOT / "unified_quant"
STRATEGY_PATH = UNIFIED_ROOT / "configs/strategies/leader_rotation.yaml"
RUNTIME_PATH = UNIFIED_ROOT / "configs/runtimes/live_toss.yaml"


class RecordingBroker:
    def __init__(self, account: AccountSnapshot):
        self.account = account
        self.open_orders: list[dict] = []
        self.prices: dict[str, float] = {}
        self.placed: list[OrderIntent] = []
        self.sell_succeeds = True
        self.buy_succeeds = True

    def get_account(self, market: str) -> AccountSnapshot:
        return self.account

    def list_open_orders(self) -> list[dict]:
        return list(self.open_orders)

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        return {symbol: self.prices[symbol] for symbol in symbols if symbol in self.prices}

    def place_order(self, order: OrderIntent) -> bool:
        self.placed.append(order)
        return self.sell_succeeds if order.side.lower() == "sell" else self.buy_succeeds


class SilentNotifier:
    def send(self, message: str) -> bool:
        return True


class StaticCache:
    def __init__(self):
        index = pd.date_range("2026-01-05 09:30", periods=3, freq="30min")
        self.frame = pd.DataFrame(
            {
                "Open": [10.0, 10.0, 10.0],
                "High": [11.0, 11.0, 11.0],
                "Low": [9.0, 9.0, 9.0],
                "Close": [10.0, 10.0, 10.0],
            },
            index=index,
        )

    def configure(self, *args, **kwargs) -> None:
        return None

    def sync(self, *args, **kwargs) -> None:
        return None

    def retry_missing(self, *args, **kwargs) -> list[str]:
        return []

    def fresh_stock_bars(self, symbols, *args, **kwargs):
        return {symbol: self.frame for symbol in symbols}, []

    def fresh_benchmark_map(self, symbols, *args, **kwargs):
        return {symbol: self.frame for symbol in symbols}


def live_config():
    base = load_split_config(STRATEGY_PATH, RUNTIME_PATH)
    return replace(
        base,
        market="US",
        symbols=("AAA", "BBB"),
        execution=replace(base.execution, live_confirm_required=False),
    )


def runtime_for(broker: RecordingBroker) -> HybridLiveRuntime:
    holdings_path = Path(tempfile.mkdtemp()) / "holdings.json"
    return HybridLiveRuntime(
        live_config(),
        broker=broker,
        notifier=SilentNotifier(),
        holdings=HoldingsStore(holdings_path),
        data_cache=StaticCache(),
    )


class LiveSafetyRegressionTest(unittest.TestCase):
    def test_unmanaged_holding_can_never_generate_or_send_a_sell(self):
        broker = RecordingBroker(
            AccountSnapshot(
                cash=1_000.0,
                positions={"OUT": Position("OUT", 5, 20.0)},
            )
        )
        broker.prices = {"AAA": 10.0, "BBB": 20.0, "OUT": 21.0}
        runtime = runtime_for(broker)
        unsafe_plan = OrderPlan(
            "leader",
            "live",
            (OrderIntent("OUT", "sell", 5, reason="Held symbol missing from strategy data"),),
        )

        with patch("supertrend_quant.live_runtime.build_order_plan", return_value=unsafe_plan):
            plan, _ = runtime.run_once(ignore_schedule=True, assume_yes=True)

        self.assertEqual(plan.orders, ())
        self.assertEqual(broker.placed, [])

    def test_guarded_sell_removes_its_post_sell_buy(self):
        account = AccountSnapshot(
            cash=1_000.0,
            positions={"AAA": Position("AAA", 5, 100.0)},
        )
        broker = RecordingBroker(account)
        broker.open_orders = [{"symbol": "AAA", "side": "SELL"}]
        broker.prices = {"AAA": 120.0, "BBB": 50.0}
        runtime = runtime_for(broker)
        plan = OrderPlan(
            "leader",
            "live",
            (
                OrderIntent("AAA", "sell", 5, reason="Supertrend down"),
                OrderIntent("BBB", "buy", 10, reason="Post-sell leader entry"),
            ),
        )

        guarded = runtime._apply_live_guards(live_config(), plan, account, ["AAA", "BBB"])

        self.assertEqual(guarded.orders, ())

    def test_failed_sell_never_executes_its_post_sell_buy(self):
        account = AccountSnapshot(
            cash=1_000.0,
            positions={"AAA": Position("AAA", 5, 100.0)},
        )
        broker = RecordingBroker(account)
        broker.prices = {"AAA": 120.0, "BBB": 50.0}
        broker.sell_succeeds = False
        runtime = runtime_for(broker)
        plan = OrderPlan(
            "leader",
            "live",
            (
                OrderIntent("AAA", "sell", 5, reason="Supertrend down"),
                OrderIntent("BBB", "buy", 10, reason="Post-sell leader entry"),
            ),
        )

        with patch("supertrend_quant.live_runtime.build_order_plan", return_value=plan):
            _, results = runtime.run_once(ignore_schedule=True, assume_yes=True)

        self.assertEqual([order.side.lower() for order in broker.placed], ["sell"])
        self.assertTrue(any(result.startswith("FAILED SELL AAA") for result in results))
        self.assertFalse(any(result.startswith("SENT BUY BBB") for result in results))

    def test_missing_realtime_price_drops_independent_buy(self):
        account = AccountSnapshot(cash=1_000.0)
        broker = RecordingBroker(account)
        broker.prices = {}
        runtime = runtime_for(broker)
        plan = OrderPlan(
            "leader",
            "live",
            (OrderIntent("AAA", "buy", 50, reason="Top-ranked leader"),),
        )

        guarded = runtime._apply_live_guards(live_config(), plan, account, ["AAA", "BBB"])

        self.assertEqual(guarded.orders, ())

    def test_missing_realtime_price_drops_rotation_sell_and_dependent_buy(self):
        account = AccountSnapshot(
            cash=1_000.0,
            positions={"AAA": Position("AAA", 5, 100.0)},
        )
        broker = RecordingBroker(account)
        broker.prices = {"BBB": 50.0}
        runtime = runtime_for(broker)
        plan = OrderPlan(
            "leader",
            "live",
            (
                OrderIntent("AAA", "sell", 5, reason="Leader rotation"),
                OrderIntent("BBB", "buy", 10, reason="Post-sell leader entry"),
            ),
        )

        guarded = runtime._apply_live_guards(live_config(), plan, account, ["AAA", "BBB"])

        self.assertEqual(guarded.orders, ())

    def test_refresh_failure_allows_known_holding_sell_but_blocks_buy(self):
        account = AccountSnapshot(
            cash=1_000.0,
            positions={"AAA": Position("AAA", 5, 100.0)},
        )
        broker = RecordingBroker(account)
        broker.prices = {"AAA": 120.0, "BBB": 50.0}
        runtime = runtime_for(broker)
        aaa = UniverseMember("AAA", "US", "US", yfinance_symbol="AAA", benchmark="SPY")
        bbb = UniverseMember("BBB", "US", "US", yfinance_symbol="BBB", benchmark="SPY")
        snapshot = UniverseSnapshot(
            schema_version=1,
            as_of="2026-07-14",
            created_at="2026-07-14T00:00:00+00:00",
            market="US",
            source="profiles",
            profiles=("sp500",),
            selection_hash="test",
            raw_members=(aaa, bbb),
            eligible_members=(aaa, bbb),
            rejected=(),
            filters={},
        )
        resolved = ResolvedUniverse(
            eligible_members=(aaa, bbb),
            exit_only_members=(),
            raw_members=(aaa, bbb),
            snapshot=snapshot,
            entries_allowed=False,
            refresh_error="provider down",
        )
        unsafe_plan = OrderPlan(
            "leader",
            "live",
            (
                OrderIntent("AAA", "sell", 5, reason="Supertrend down"),
                OrderIntent("BBB", "buy", 10, reason="Top-ranked leader"),
            ),
        )

        with (
            patch("supertrend_quant.live_runtime.resolve_universe", return_value=resolved),
            patch("supertrend_quant.live_runtime.build_order_plan", return_value=unsafe_plan),
        ):
            plan, _ = runtime.run_once(ignore_schedule=True, assume_yes=True)

        self.assertEqual([(order.symbol, order.side) for order in plan.orders], [("AAA", "sell")])
        self.assertEqual([(order.symbol, order.side) for order in broker.placed], [("AAA", "sell")])


if __name__ == "__main__":
    unittest.main()
