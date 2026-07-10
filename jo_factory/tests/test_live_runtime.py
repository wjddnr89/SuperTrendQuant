import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from supertrend_quant.config import benchmark_for_symbol, load_split_config
from supertrend_quant.data_cache import YahooStateCache, align_stock_to_benchmark_history
from supertrend_quant.holdings import HoldingsStore
from supertrend_quant.live_runtime import HybridLiveRuntime
from supertrend_quant.portfolio import AccountSnapshot, OrderIntent, OrderPlan, Position
from supertrend_quant.runtime import check_market_schedule


def load_live_config():
    return load_split_config("configs/strategies/leader_rotation.yaml", "configs/runtimes/live_toss.yaml")


class FakeBroker:
    def __init__(self):
        self.open_orders = []
        self.prices = {}
        self.account = AccountSnapshot(cash=10_000)
        self.orders = []

    def list_open_orders(self):
        return self.open_orders

    def get_prices(self, symbols):
        return {symbol: self.prices[symbol] for symbol in symbols if symbol in self.prices}

    def get_account(self, market):
        return self.account

    def place_order(self, order):
        self.orders.append(order)
        if order.side == "sell":
            self.account = AccountSnapshot(cash=12_000, positions={})
        return True


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)
        return True


class FakeCache:
    def __init__(self, bars, benchmark, filter_benchmark):
        self._bars = bars
        self._benchmark = benchmark
        self._filter_benchmark = filter_benchmark
        self.sync_count = 0

    def sync(self, symbols, market, universe_file, benchmarks, current_candle_base=None):
        self.sync_count += 1

    def retry_missing(self, market, universe_file, market_tz, current_candle_base):
        return []

    def fresh_stock_bars(self, symbols, market_tz, current_candle_base):
        return self._bars, []

    def fresh_benchmark_map(self, symbols, market, universe_file, source, market_tz, current_base):
        return self._filter_benchmark if source == "1h" else self._benchmark


class LiveRuntimeTest(unittest.TestCase):
    def test_market_schedule_matches_main_jo_windows(self):
        kr_open = datetime(2026, 7, 8, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))
        us_closed = datetime(2026, 7, 7, 21, 0, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(check_market_schedule(now_kr=kr_open, now_us=us_closed).state, "KR")

        kr_closed = datetime(2026, 7, 8, 20, 0, tzinfo=ZoneInfo("Asia/Seoul"))
        us_open = datetime(2026, 7, 8, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(check_market_schedule(now_kr=kr_closed, now_us=us_open).state, "US")

    def test_holdings_store_syncs_real_account_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HoldingsStore(Path(tmp) / "holding.json")
            account = AccountSnapshot(
                cash=100,
                positions={
                    "SOXL": Position("SOXL", 3, 10),
                    "OUT": Position("OUT", 1, 5),
                },
            )

            synced = store.sync_market("US", account, ["SOXL"])
            self.assertEqual(synced, {"SOXL": {"qty": 3, "buy_price": 10}})

    def test_live_guards_skip_open_orders_and_low_profit_rotation(self):
        config = load_live_config()
        fake_broker = FakeBroker()
        fake_broker.open_orders = [{"symbol": "AMD", "side": "buy"}]
        fake_broker.prices = {"SOXL": 100, "AMD": 50}
        runtime = HybridLiveRuntime(config, broker=fake_broker)
        account = AccountSnapshot(
            cash=10_000,
            positions={"SOXL": Position("SOXL", 10, 100)},
        )
        plan = OrderPlan(
            strategy_name="test",
            mode="live",
            orders=(
                OrderIntent("AMD", "buy", 10, reason="Top RS leader"),
                OrderIntent("SOXL", "sell", 10, reason="Leader rotation"),
            ),
        )

        guarded = runtime._apply_live_guards(config, plan, account, ["SOXL", "AMD"])
        self.assertEqual(guarded.orders, ())

    def test_live_guards_keep_post_sell_buy_for_refreshed_cash(self):
        config = load_live_config()
        fake_broker = FakeBroker()
        fake_broker.prices = {"SOXL": 120, "AMD": 50}
        runtime = HybridLiveRuntime(config, broker=fake_broker)
        account = AccountSnapshot(
            cash=0,
            positions={"SOXL": Position("SOXL", 10, 100)},
        )
        plan = OrderPlan(
            strategy_name="test",
            mode="live",
            orders=(
                OrderIntent("SOXL", "sell", 10, reason="Supertrend down"),
                OrderIntent("AMD", "buy", 20, reason="Post-sell leader entry"),
            ),
        )

        guarded = runtime._apply_live_guards(config, plan, account, ["SOXL", "AMD"])
        self.assertEqual([order.symbol for order in guarded.orders], ["SOXL", "AMD"])

    def test_kr_benchmark_mapping_and_freshness(self):
        self.assertEqual(benchmark_for_symbol("005930", "KR", "universe.json"), "^KS11")
        self.assertEqual(benchmark_for_symbol("010170", "KR", "universe.json"), "^KQ11")

        tz = ZoneInfo("Asia/Seoul")
        current_base = pd.Timestamp("2026-07-08 10:00:00", tz=tz)
        cache = YahooStateCache()
        fresh_df = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1]},
            index=[current_base],
        )
        stale_df = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1]},
            index=[pd.Timestamp("2026-07-08 09:30:00", tz=tz)],
        )
        cache.stock_bars["A"] = fresh_df
        cache.stock_bars["B"] = stale_df

        bars, stale = cache.fresh_stock_bars(["A", "B", "C"], tz, current_base)
        self.assertEqual(list(bars), ["A"])
        self.assertEqual(stale, ["B", "C"])

    def test_stock_history_aligns_to_benchmark_like_main_jo(self):
        tz = ZoneInfo("America/New_York")
        idx = pd.date_range("2026-07-08 09:30", periods=4, freq="30min", tz=tz)
        benchmark = pd.DataFrame(
            {"Open": [1, 1, 1, 1], "High": [1, 1, 1, 1], "Low": [1, 1, 1, 1], "Close": [1, 1, 1, 1]},
            index=idx,
        )
        stock = pd.DataFrame(
            {"Open": [10, 12], "High": [10, 12], "Low": [10, 12], "Close": [10, 12]},
            index=[idx[0], idx[-1]],
        )

        aligned = align_stock_to_benchmark_history(stock, benchmark, idx[-1])

        self.assertEqual(list(aligned.index), list(idx))
        self.assertEqual(aligned.loc[idx[1], "Close"], 10)
        self.assertEqual(aligned.loc[idx[-1], "Close"], 12)

    def test_live_cycle_replay_sends_sell_then_post_sell_buy(self):
        config = load_live_config()
        config = config.__class__(
            **{
                **config.__dict__,
                "market": "US",
                "leader_rotation": config.leader_rotation.__class__(
                    rs_period=130,
                    max_slots=1,
                    hurdle_atr_mult=0.0,
                    allow_late_chase=True,
                    min_rotation_profit_pct=0.01,
                ),
                "execution": config.execution.__class__(
                    order_type="market",
                    allocation_pct=0.9,
                    broker="toss",
                    live_confirm_required=False,
                ),
            }
        )
        idx = pd.date_range("2026-07-08 09:30", periods=135, freq="30min", tz=ZoneInfo("America/New_York"))
        held_close = [100 + (i * 0.02) for i in range(len(idx))]
        leader_close = [50 + (i * 1.2) for i in range(len(idx))]
        held = pd.DataFrame(
            {
                "Open": held_close,
                "High": [value + 1 for value in held_close],
                "Low": [value - 1 for value in held_close],
                "Close": held_close,
            },
            index=idx,
        )
        leader = pd.DataFrame(
            {
                "Open": leader_close,
                "High": [value + 1 for value in leader_close],
                "Low": [value - 1 for value in leader_close],
                "Close": leader_close,
            },
            index=idx,
        )
        benchmark_df = pd.DataFrame(
            {"Open": [100] * len(idx), "High": [101] * len(idx), "Low": [99] * len(idx), "Close": [100] * len(idx)},
            index=idx,
        )
        filter_df = pd.DataFrame(
            {
                "Open": [100] * len(idx),
                "High": [110] * len(idx),
                "Low": [90] * len(idx),
                "Close": [100 + i * 0.1 for i in range(len(idx))],
            },
            index=idx,
        )

        broker = FakeBroker()
        broker.prices = {"SOXL": 120, "AMD": 70}
        broker.account = AccountSnapshot(cash=0, positions={"SOXL": Position("SOXL", 10, 100)})
        runtime = HybridLiveRuntime(
            config,
            broker=broker,
            notifier=FakeNotifier(),
            holdings=HoldingsStore(Path(tempfile.mkdtemp()) / "holding.json"),
            data_cache=FakeCache(
                bars={"SOXL": held, "AMD": leader},
                benchmark={"SOXL": benchmark_df, "AMD": benchmark_df},
                filter_benchmark={"SOXL": filter_df, "AMD": filter_df},
            ),
        )

        plan, results = runtime.run_once(ignore_schedule=True, assume_yes=True)

        self.assertGreaterEqual(len(plan.orders), 1)
        self.assertEqual([order.side for order in broker.orders], ["sell", "buy"])
        self.assertIn("SENT SELL SOXL", results[0])
        self.assertIn("SENT BUY AMD", results[1])

    def test_live_runtime_syncs_once_per_candle_base(self):
        config = load_live_config()
        config = config.__class__(
            **{
                **config.__dict__,
                "market": "US",
                "execution": config.execution.__class__(
                    order_type="market",
                    allocation_pct=0.9,
                    broker="toss",
                    live_confirm_required=False,
                ),
            }
        )
        idx = pd.date_range("2026-07-08 09:30", periods=135, freq="30min", tz=ZoneInfo("America/New_York"))
        df = pd.DataFrame(
            {
                "Open": [100 + i for i in range(len(idx))],
                "High": [101 + i for i in range(len(idx))],
                "Low": [99 + i for i in range(len(idx))],
                "Close": [100 + i for i in range(len(idx))],
            },
            index=idx,
        )
        broker = FakeBroker()
        broker.account = AccountSnapshot(cash=10_000)
        cache = FakeCache(
            bars={"SOXL": df},
            benchmark={"SOXL": df},
            filter_benchmark={"SOXL": df},
        )
        runtime = HybridLiveRuntime(
            config,
            broker=broker,
            notifier=FakeNotifier(),
            holdings=HoldingsStore(Path(tempfile.mkdtemp()) / "holding.json"),
            data_cache=cache,
        )

        runtime.run_once(ignore_schedule=True, assume_yes=True)
        runtime.run_once(ignore_schedule=True, assume_yes=True)

        self.assertEqual(cache.sync_count, 1)


if __name__ == "__main__":
    unittest.main()
