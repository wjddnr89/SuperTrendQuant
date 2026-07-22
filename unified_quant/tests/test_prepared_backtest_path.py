from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from supertrend_quant.config import StrategyIdentity, load_split_config
from supertrend_quant.data import MarketData
from supertrend_quant.portfolio import AccountSnapshot, OrderPlan, Position, PositionEconomics
from supertrend_quant.research import search_configs
from supertrend_quant.runners import run_backtest_on_data
from supertrend_quant.strategies import available_strategies, create_strategy, register_strategy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_ROOT = REPOSITORY_ROOT / "unified_quant"
LEADER_PATH = UNIFIED_ROOT / "configs/strategies/leader_rotation.yaml"
SIMPLE_PATH = UNIFIED_ROOT / "configs/strategies/simple_supertrend.yaml"
RUNTIME_PATH = UNIFIED_ROOT / "configs/runtimes/research_sp500.yaml"


def trend_frame(index: pd.Index, start: float, step: float) -> pd.DataFrame:
    close = [start + step * position for position in range(len(index))]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [value + 1.0 for value in close],
            "Low": [value - 1.0 for value in close],
            "Close": close,
        },
        index=index,
    )


def confirmed_uptrend_frame(index: pd.Index, start: float, step: float) -> pd.DataFrame:
    """Create an explicit prior-upper-band breakout before strategy assertions."""

    close = [start, start]
    close.extend(start + 4.0 + step * (position - 2) for position in range(2, len(index)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": [value + 1.0 for value in close],
            "Low": [value - 1.0 for value in close],
            "Close": close,
        },
        index=index,
    )


def leader_fixture():
    config = load_split_config(LEADER_PATH, RUNTIME_PATH)
    config = replace(
        config,
        symbols=("AAA", "BBB"),
        supertrend=replace(
            config.supertrend,
            period=2,
            multiplier=1.0,
            symbol_multipliers={},
        ),
        market_trend_filter=replace(
            config.market_trend_filter,
            enabled=True,
            timeframe="30m",
        ),
        leader_rotation=replace(
            config.leader_rotation,
            hurdle_atr_mult=0.0,
            min_rotation_profit_pct=0.0,
        ),
        scoring=replace(
            config.scoring,
            params={"lookback_bars": 2},
        ),
    )
    index = pd.date_range("2026-01-05 09:30", periods=24, freq="30min")
    bars = {
        "AAA": confirmed_uptrend_frame(index, 100.0, 0.5),
        "BBB": confirmed_uptrend_frame(index, 50.0, 1.5),
    }
    shared_benchmark = confirmed_uptrend_frame(index, 100.0, 0.0)
    benchmarks = {"AAA": shared_benchmark, "BBB": shared_benchmark}
    return config, MarketData(
        bars=bars,
        benchmark=benchmarks,
        filter_benchmark=benchmarks,
    )


class InstrumentedPreparedBacktest:
    def __init__(self, owner):
        self.owner = owner

    def build_order_plan(self, signal_ts, account, mode="backtest") -> OrderPlan:
        self.owner.prepared_timestamps.append(signal_ts)
        self.owner.prepared_accounts.append(account)
        return OrderPlan(self.owner.config.strategy.name, mode, ())


class InstrumentedPreparableStrategy:
    strategy_type = "acceptance_instrumented_preparable"
    prepare_calls = 0
    legacy_calls = 0
    prepared_timestamps: list[object] = []
    prepared_accounts: list[AccountSnapshot] = []
    prepared_bar_lengths: dict[str, int] = {}

    def __init__(self, config):
        self.config = config

    @classmethod
    def validate_config(cls, config) -> None:
        return None

    @classmethod
    def reset(cls) -> None:
        cls.prepare_calls = 0
        cls.legacy_calls = 0
        cls.prepared_timestamps = []
        cls.prepared_accounts = []
        cls.prepared_bar_lengths = {}

    def warmup_bars(self) -> int:
        return 1

    def prepare_backtest(self, bars, benchmark=None, filter_benchmark=None):
        self.__class__.prepare_calls += 1
        self.__class__.prepared_bar_lengths = {
            symbol: len(frame) for symbol, frame in bars.items()
        }
        return InstrumentedPreparedBacktest(self)

    def build_order_plan(
        self,
        bars,
        account,
        mode,
        benchmark=None,
        filter_benchmark=None,
    ) -> OrderPlan:
        self.__class__.legacy_calls += 1
        return OrderPlan(self.config.strategy.name, mode, ())


class PreparedBacktestAcceptanceTest(unittest.TestCase):
    def test_prepared_leader_matches_normal_plan_at_multiple_timestamps_and_accounts(self):
        config, market_data = leader_fixture()
        strategy = create_strategy(config)
        prepared = strategy.prepare_backtest(
            market_data.bars,
            benchmark=market_data.benchmark,
            filter_benchmark=market_data.filter_benchmark,
        )
        index = market_data.bars["AAA"].index
        scenarios = (
            (
                "new leader entry",
                index[4],
                AccountSnapshot(cash=10_000.0),
                ("buy",),
            ),
            (
                "leader rotation",
                index[10],
                AccountSnapshot(
                    cash=500.0,
                    positions={"AAA": Position("AAA", 5, 90.0)},
                    position_economics={
                        "AAA": PositionEconomics(450.0, net_return_pct=0.1)
                    },
                ),
                ("sell", "buy"),
            ),
            (
                "hold current leader",
                index[15],
                AccountSnapshot(
                    cash=500.0,
                    positions={"BBB": Position("BBB", 5, 50.0)},
                ),
                (),
            ),
        )

        for label, signal_ts, account, expected_sides in scenarios:
            with self.subTest(label=label):
                sliced_bars = {
                    symbol: frame.loc[:signal_ts].copy()
                    for symbol, frame in market_data.bars.items()
                }
                sliced_benchmark = {
                    symbol: frame.loc[:signal_ts].copy()
                    for symbol, frame in market_data.benchmark.items()
                }
                sliced_filter = {
                    symbol: frame.loc[:signal_ts].copy()
                    for symbol, frame in market_data.filter_benchmark.items()
                }
                normal = strategy.build_order_plan(
                    sliced_bars,
                    account,
                    mode="backtest",
                    benchmark=sliced_benchmark,
                    filter_benchmark=sliced_filter,
                )
                fast = prepared.build_order_plan(signal_ts, account, mode="backtest")

                self.assertEqual(fast, normal)
                self.assertEqual(
                    tuple(order.side for order in fast.orders),
                    expected_sides,
                )

    def test_leader_rotation_can_fill_multiple_top_slots(self):
        config, market_data = leader_fixture()
        config = replace(
            config,
            risk=replace(config.risk, max_position_count=2),
            leader_rotation=replace(config.leader_rotation, max_slots=2),
            execution=replace(config.execution, allocation_pct=1.0),
        )
        strategy = create_strategy(config)
        prepared = strategy.prepare_backtest(
            market_data.bars,
            benchmark=market_data.benchmark,
            filter_benchmark=market_data.filter_benchmark,
        )
        signal_ts = market_data.bars["AAA"].index[4]

        plan = prepared.build_order_plan(
            signal_ts,
            AccountSnapshot(cash=10_000.0),
            mode="backtest",
        )

        self.assertEqual(tuple(order.side for order in plan.orders), ("buy", "buy"))
        self.assertEqual(len({order.symbol for order in plan.orders}), 2)

    def test_canonical_runner_prepares_once_and_never_calls_legacy_plan_path(self):
        if InstrumentedPreparableStrategy.strategy_type not in available_strategies():
            register_strategy(InstrumentedPreparableStrategy)
        base = load_split_config(SIMPLE_PATH, RUNTIME_PATH)
        config = replace(
            base,
            strategy=StrategyIdentity(
                "instrumented_preparable",
                InstrumentedPreparableStrategy.strategy_type,
            ),
        )
        index = pd.date_range("2026-01-05 09:30", periods=6, freq="30min")
        market_data = MarketData(
            bars={"AAA": trend_frame(index, 10.0, 0.1)},
        )
        InstrumentedPreparableStrategy.reset()

        result = run_backtest_on_data(config, market_data)

        self.assertFalse(result.equity.empty)
        self.assertEqual(InstrumentedPreparableStrategy.prepare_calls, 1)
        self.assertEqual(InstrumentedPreparableStrategy.legacy_calls, 0)
        self.assertEqual(InstrumentedPreparableStrategy.prepared_bar_lengths, {"AAA": 6})
        self.assertEqual(
            InstrumentedPreparableStrategy.prepared_timestamps,
            list(index[1:-1]),
        )
        self.assertTrue(
            all(account.positions == {} for account in InstrumentedPreparableStrategy.prepared_accounts)
        )

    def test_two_configuration_synthetic_search_completes_with_holdout(self):
        config, market_data = leader_fixture()

        result = search_configs(
            config,
            market_data,
            {"st_period": (2, 3)},
            min_segment_bars=3,
            evaluate_test_for_best=True,
            limit=2,
        )

        self.assertEqual(len(result.rows), 2)
        self.assertEqual([row.rank for row in result.rows], [1, 2])
        self.assertEqual(result.errors, ())
        self.assertIsNotNone(result.best_evaluation.validation)
        self.assertIsNotNone(result.best_evaluation.test)
        self.assertFalse(result.rows[0].evaluation.is_partial)
        self.assertTrue(result.rows[1].evaluation.is_partial)


if __name__ == "__main__":
    unittest.main()
