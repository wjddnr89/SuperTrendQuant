from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from supertrend_quant.config import CapitalConfig, CostsConfig, StrategyIdentity, load_split_config
from supertrend_quant.data import MarketData, delay_daily_until_next_session, resample_ohlc
from supertrend_quant.portfolio import OrderIntent, OrderPlan, estimate_quantity
from supertrend_quant.runners import BacktestResult, _slice_benchmark_frame, run_backtest_on_data
from supertrend_quant.strategies import available_strategies, register_strategy

try:
    from supertrend_quant.research import evaluation
except ImportError:  # pragma: no cover - documents optional API during migration
    evaluation = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_ROOT = REPOSITORY_ROOT / "unified_quant"
STRATEGY_PATH = UNIFIED_ROOT / "configs/strategies/simple_supertrend.yaml"
RUNTIME_PATH = UNIFIED_ROOT / "configs/runtimes/research_sp500.yaml"


class DeterministicRoundTripStrategy:
    strategy_type = "acceptance_deterministic_round_trip"

    def __init__(self, config):
        self.config = config

    @classmethod
    def validate_config(cls, config) -> None:
        return None

    def warmup_bars(self) -> int:
        return 0

    def build_order_plan(
        self,
        bars,
        account,
        mode,
        benchmark=None,
        filter_benchmark=None,
    ) -> OrderPlan:
        position = account.positions.get("AAA")
        if position is not None:
            order = OrderIntent("AAA", "sell", position.quantity, reason="deterministic exit")
            return OrderPlan(self.config.strategy.name, mode, (order,))
        price = float(bars["AAA"]["Close"].iloc[-1])
        quantity = estimate_quantity(account.cash, price, self.config.execution.allocation_pct)
        order = OrderIntent("AAA", "buy", quantity, reason="deterministic entry")
        return OrderPlan(self.config.strategy.name, mode, (order,))


def canonical_config():
    if DeterministicRoundTripStrategy.strategy_type not in available_strategies():
        register_strategy(DeterministicRoundTripStrategy)
    base = load_split_config(STRATEGY_PATH, RUNTIME_PATH)
    return replace(
        base,
        strategy=StrategyIdentity(
            name="deterministic_round_trip",
            type=DeterministicRoundTripStrategy.strategy_type,
        ),
        capital=CapitalConfig(initial_cash=1_000.0),
        costs=CostsConfig(fee_rate=0.01, slippage_rate=0.02),
        execution=replace(base.execution, allocation_pct=0.5),
    )


def canonical_market_data(periods: int = 3) -> MarketData:
    index = pd.date_range("2026-01-05 09:30", periods=periods, freq="30min")
    if periods == 3:
        opens = [10.0, 11.0, 13.0]
        closes = [10.0, 12.0, 13.0]
    else:
        opens = [10.0 + index for index in range(periods)]
        closes = [value + 0.5 for value in opens]
    frame = pd.DataFrame(
        {
            "Open": opens,
            "High": [value + 1.0 for value in closes],
            "Low": [value - 1.0 for value in opens],
            "Close": closes,
        },
        index=index,
    )
    return MarketData(bars={"AAA": frame})


class CanonicalEngineAcceptanceTest(unittest.TestCase):
    def test_backtest_on_data_is_deterministic_and_uses_next_open_integer_and_costs(self):
        config = canonical_config()
        market_data = canonical_market_data()

        first = run_backtest_on_data(config, market_data)
        second = run_backtest_on_data(config, market_data)

        pd.testing.assert_series_equal(first.equity, second.equity)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.trade_records, second.trade_records)
        self.assertEqual(len(first.trade_records), 1)

        trade = first.trade_records[0]
        index = market_data.bars["AAA"].index
        expected_entry = 11.0 * 1.02
        expected_exit = 13.0 * 0.98
        expected_cost = 50 * expected_entry * 1.01
        expected_proceeds = 50 * expected_exit * 0.99
        expected_final_cash = 1_000.0 - expected_cost + expected_proceeds

        self.assertIsInstance(trade["quantity"], int)
        self.assertEqual(trade["quantity"], 50)
        self.assertEqual(trade["entry_time"], index[1])
        self.assertEqual(trade["exit_time"], index[2])
        self.assertAlmostEqual(trade["entry_price"], expected_entry)
        self.assertAlmostEqual(trade["exit_price"], expected_exit)
        self.assertAlmostEqual(first.equity.iloc[-1], expected_final_cash)

    @unittest.skipUnless(evaluation is not None, "canonical research evaluation API is unavailable")
    def test_research_split_is_disjoint_and_evaluation_calls_canonical_runner(self):
        config = canonical_config()
        market_data = canonical_market_data(15)
        full_index = market_data.bars["AAA"].index
        splits = evaluation.split_index(full_index, 0.6, 0.2, min_segment_bars=3)

        self.assertEqual(set(splits), {"overall", "train", "validation", "test"})
        self.assertTrue(splits["train"][-1] < splits["validation"][0])
        self.assertTrue(splits["validation"][-1] < splits["test"][0])
        self.assertEqual(
            len(splits["train"].intersection(splits["validation"])),
            0,
        )
        self.assertEqual(len(splits["validation"].intersection(splits["test"])), 0)

        def fake_canonical_runner(config_arg, market_data_arg, run_index=None):
            selected = pd.Index(run_index)
            equity = pd.Series(
                [config_arg.capital.initial_cash] * len(selected),
                index=selected,
                name="equity",
                dtype=float,
            )
            return BacktestResult(
                equity=equity,
                metrics={
                    "total_return": 0.0,
                    "mdd": 0.0,
                    "sharpe": 0.0,
                    "win_rate": 0.0,
                    "payoff_ratio": 0.0,
                    "trade_count": 0,
                },
                trades=[],
                skipped=(),
            )

        with patch.object(
            evaluation,
            "run_backtest_on_data",
            side_effect=fake_canonical_runner,
        ) as canonical_runner, patch.object(
            evaluation,
            "build_benchmark_report",
            return_value={},
        ):
            result = evaluation.evaluate_config(
                config,
                market_data,
                train_ratio=0.6,
                validation_ratio=0.2,
                min_segment_bars=3,
            )

        self.assertEqual(canonical_runner.call_count, 4)
        self.assertEqual(set(result.segments), {"overall", "train", "validation", "test"})
        self.assertEqual(result.ranking_segment().name, "validation")
        for call, expected_name in zip(
            canonical_runner.call_args_list,
            ("overall", "train", "validation", "test"),
        ):
            self.assertIs(call.args[0], config)
            self.assertIs(call.args[1], market_data)
            pd.testing.assert_index_equal(call.kwargs["run_index"], splits[expected_name])


class DataAvailabilityAcceptanceTest(unittest.TestCase):
    def test_daily_filter_is_hidden_until_the_next_date(self):
        daily = pd.DataFrame(
            {"Open": [100.0], "High": [102.0], "Low": [99.0], "Close": [101.0]},
            index=[pd.Timestamp("2026-01-05")],
        )
        delayed = delay_daily_until_next_session(daily)

        self.assertEqual(delayed.index[0], pd.Timestamp("2026-01-06"))
        self.assertTrue(_slice_benchmark_frame(delayed, pd.Timestamp("2026-01-05 23:59")).empty)
        self.assertEqual(
            float(_slice_benchmark_frame(delayed, pd.Timestamp("2026-01-06 09:30"))["Close"].iloc[-1]),
            101.0,
        )

    def test_higher_timeframe_resample_is_labeled_at_right_edge(self):
        index = pd.date_range("2026-01-05 09:30", periods=4, freq="30min")
        frame = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0, 13.0],
                "High": [11.0, 12.0, 13.0, 14.0],
                "Low": [9.0, 10.0, 11.0, 12.0],
                "Close": [10.5, 11.5, 12.5, 13.5],
            },
            index=index,
        )
        hourly = resample_ohlc(frame, "1h", "US")

        self.assertEqual(hourly.index.tolist(), [pd.Timestamp("2026-01-05 10:30"), pd.Timestamp("2026-01-05 11:30")])
        self.assertEqual(float(hourly.iloc[0]["Open"]), 10.0)
        self.assertEqual(float(hourly.iloc[0]["Close"]), 11.5)
        self.assertEqual(float(hourly.iloc[0]["High"]), 12.0)
        self.assertEqual(float(hourly.iloc[0]["Low"]), 9.0)


if __name__ == "__main__":
    unittest.main()
