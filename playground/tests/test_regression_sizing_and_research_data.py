from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from supertrend_quant.brokers import PaperBroker
from supertrend_quant.config import CapitalConfig, CostsConfig, RiskConfig, load_split_config
from supertrend_quant.data import MarketData
from supertrend_quant.portfolio import AccountSnapshot, estimate_quantity
from supertrend_quant.research import MarketDataCache, MarketDataMismatchError, resolve_market_data, with_timeframe
from supertrend_quant.runners import run_backtest_on_data
from supertrend_quant.strategies import build_order_plan


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_ROOT = REPOSITORY_ROOT / "unified_quant"
STRATEGY_PATH = UNIFIED_ROOT / "configs/strategies/simple_supertrend.yaml"
RUNTIME_PATH = UNIFIED_ROOT / "configs/runtimes/simulation.yaml"


def all_in_config():
    base = load_split_config(STRATEGY_PATH, RUNTIME_PATH)
    return replace(
        base,
        symbols=("AAA",),
        capital=CapitalConfig(initial_cash=1_000.0),
        costs=CostsConfig(fee_rate=0.01, slippage_rate=0.02),
        risk=RiskConfig(max_position_count=1),
        execution=replace(base.execution, allocation_pct=1.0),
        scoring=replace(base.scoring, params={"lookback_bars": 1}),
    )


def flat_market_data(periods: int = 5) -> MarketData:
    index = pd.date_range("2026-01-05 09:30", periods=periods, freq="30min")
    frame = pd.DataFrame(
        {
            "Open": [10.0] * periods,
            "High": [11.0] * periods,
            "Low": [9.0] * periods,
            "Close": [10.0] * periods,
        },
        index=index,
    )
    return MarketData(bars={"AAA": frame}, benchmark={"AAA": frame})


def forced_buy_signal(config, symbol, frame):
    out = frame.copy()
    out["Trend"] = 1
    out["ATR_pct"] = 0.1
    out["BuySignal"] = True
    out["SellSignal"] = False
    return out


class CostAwareSizingRegressionTest(unittest.TestCase):
    def test_central_sizing_keeps_all_in_order_affordable(self):
        quantity = estimate_quantity(
            1_000.0,
            10.0,
            1.0,
            fee_rate=0.01,
            slippage_rate=0.02,
        )
        self.assertEqual(quantity, 97)
        self.assertLessEqual(quantity * 10.0 * 1.02 * 1.01, 1_000.0)
        self.assertGreater((quantity + 1) * 10.0 * 1.02 * 1.01, 1_000.0)

    @patch("supertrend_quant.strategies.simple_supertrend.with_supertrend", forced_buy_signal)
    def test_strategy_sizes_allocation_one_with_costs(self):
        config = all_in_config()
        market_data = flat_market_data(3)
        frame = market_data.bars["AAA"]

        plan = build_order_plan(
            config,
            {"AAA": frame},
            AccountSnapshot(cash=1_000.0),
            mode="paper",
            benchmark=market_data.benchmark,
        )

        self.assertEqual(len(plan.orders), 1)
        self.assertEqual(plan.orders[0].quantity, 97)

    @patch("supertrend_quant.strategies.simple_supertrend.with_supertrend", forced_buy_signal)
    def test_backtest_executes_allocation_one_instead_of_silently_skipping(self):
        result = run_backtest_on_data(all_in_config(), flat_market_data())

        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0]["quantity"], 97)
        self.assertAlmostEqual(result.trade_records[0]["entry_price"], 10.2)

    @patch("supertrend_quant.strategies.simple_supertrend.with_supertrend", forced_buy_signal)
    def test_paper_executes_the_same_affordable_allocation_one_plan(self):
        config = all_in_config()
        market_data = flat_market_data(3)
        frame = market_data.bars["AAA"]
        plan = build_order_plan(
            config,
            {"AAA": frame},
            AccountSnapshot(cash=1_000.0),
            mode="paper",
            benchmark=market_data.benchmark,
        )
        with tempfile.TemporaryDirectory() as tmp:
            broker = PaperBroker(Path(tmp) / "paper.json", initial_cash=1_000.0)
            fills = broker.execute_plan(
                plan,
                {"AAA": 10.0},
                fee_rate=config.costs.fee_rate,
                slippage_rate=config.costs.slippage_rate,
            )
            account = broker.get_account()

        self.assertEqual(fills, ["BUY AAA 97 @ 10.2000"])
        self.assertEqual(account.positions["AAA"].quantity, 97)
        self.assertGreaterEqual(account.cash, 0.0)


class ResearchDataResolutionRegressionTest(unittest.TestCase):
    def test_fixed_market_data_rejects_timeframe_overlay(self):
        base = all_in_config()
        candidate = with_timeframe(base, "1h")
        fixed = flat_market_data()

        with self.assertRaises(MarketDataMismatchError):
            resolve_market_data(fixed, candidate, fixed_config=base)

    def test_market_data_cache_resolves_and_caches_each_distinct_request(self):
        calls = []

        def loader(config):
            calls.append((config.timeframe, config.market_trend_filter.enabled, config.market_trend_filter.timeframe))
            return flat_market_data()

        base = all_in_config()
        hourly = with_timeframe(base, "1h")
        hourly_four_hour_filter = replace(
            hourly,
            market_trend_filter=replace(
                hourly.market_trend_filter,
                enabled=True,
                timeframe="4h",
            ),
        )
        cache = MarketDataCache(loader=loader)

        base_data = cache(base)
        self.assertIs(cache(base), base_data)
        hourly_data = cache(hourly)
        self.assertIs(cache(hourly), hourly_data)
        filtered_data = cache(hourly_four_hour_filter)
        self.assertIs(cache(hourly_four_hour_filter), filtered_data)

        self.assertEqual(
            calls,
            [
                ("1d", False, "1d"),
                ("1h", False, "1d"),
                ("1h", True, "4h"),
            ],
        )
        self.assertEqual(len(cache.cached_requests), 3)


if __name__ == "__main__":
    unittest.main()
