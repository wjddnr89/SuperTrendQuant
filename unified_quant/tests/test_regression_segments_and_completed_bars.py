from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from supertrend_quant.config import StrategyIdentity, load_split_config
from supertrend_quant.data import MarketData
from supertrend_quant.data_cache import trim_to_completed
from supertrend_quant.portfolio import OrderPlan
from supertrend_quant.runners import run_backtest_on_data
from supertrend_quant.runtime import last_completed_bar_end
from supertrend_quant.strategies import available_strategies, register_strategy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_ROOT = REPOSITORY_ROOT / "unified_quant"
STRATEGY_PATH = UNIFIED_ROOT / "configs/strategies/simple_supertrend.yaml"
RUNTIME_PATH = UNIFIED_ROOT / "configs/runtimes/research_sp500.yaml"


class NoOrderSegmentStrategy:
    strategy_type = "regression_no_order_segment"

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
        return OrderPlan(self.config.strategy.name, mode, ())


def segment_fixture():
    if NoOrderSegmentStrategy.strategy_type not in available_strategies():
        register_strategy(NoOrderSegmentStrategy)
    base = load_split_config(STRATEGY_PATH, RUNTIME_PATH)
    config = replace(
        base,
        strategy=StrategyIdentity("no_order_segment", NoOrderSegmentStrategy.strategy_type),
    )
    index = pd.date_range("2026-01-05 09:30", periods=6, freq="30min")
    frame = pd.DataFrame(
        {
            "Open": [10.0] * len(index),
            "High": [11.0] * len(index),
            "Low": [9.0] * len(index),
            "Close": [10.0] * len(index),
        },
        index=index,
    )
    return config, MarketData(bars={"AAA": frame}), index


class RunIndexRegressionTest(unittest.TestCase):
    def test_noncontiguous_run_index_is_explicitly_rejected(self):
        config, market_data, index = segment_fixture()
        requested = pd.Index([index[0], index[2], index[3], index[4]])

        with self.assertRaises((ValueError, RuntimeError)) as raised:
            run_backtest_on_data(config, market_data, run_index=requested)

        self.assertTrue(str(raised.exception).strip())


class CompletedCandleRegressionTest(unittest.TestCase):
    def test_active_30m_candle_is_trimmed_until_its_close(self):
        timezone = ZoneInfo("America/New_York")
        now = datetime(2026, 1, 5, 10, 15, tzinfo=timezone)
        completed_end = last_completed_bar_end(now, "US", "30m")
        index = pd.date_range("2026-01-05 09:30", periods=3, freq="30min", tz=timezone)
        frame = pd.DataFrame({"Close": [10.0, 11.0, 12.0]}, index=index)

        trimmed = trim_to_completed(
            frame,
            "30m",
            completed_end,
            timezone,
            daily_available_on_index=False,
        )

        self.assertEqual(completed_end, datetime(2026, 1, 5, 10, 0, tzinfo=timezone))
        self.assertEqual(trimmed.index.tolist(), [index[0]])

    def test_right_labeled_hourly_candle_is_visible_only_at_completed_edge(self):
        timezone = ZoneInfo("America/New_York")
        now = datetime(2026, 1, 5, 10, 45, tzinfo=timezone)
        completed_end = last_completed_bar_end(now, "US", "1h")
        index = pd.DatetimeIndex(
            [
                datetime(2026, 1, 5, 10, 30, tzinfo=timezone),
                datetime(2026, 1, 5, 11, 30, tzinfo=timezone),
            ]
        )
        frame = pd.DataFrame({"Close": [10.0, 11.0]}, index=index)

        trimmed = trim_to_completed(
            frame,
            "1h",
            completed_end,
            timezone,
            daily_available_on_index=False,
        )

        self.assertEqual(completed_end, datetime(2026, 1, 5, 10, 30, tzinfo=timezone))
        self.assertEqual(trimmed.index.tolist(), [index[0]])

    def test_current_daily_candle_is_excluded_but_delayed_daily_index_is_available(self):
        timezone = ZoneInfo("America/New_York")
        now = datetime(2026, 1, 6, 12, 0, tzinfo=timezone)
        completed_end = last_completed_bar_end(now, "US", "1d")
        raw_index = pd.DatetimeIndex(
            [
                datetime(2026, 1, 5, 0, 0, tzinfo=timezone),
                datetime(2026, 1, 6, 0, 0, tzinfo=timezone),
            ]
        )
        raw = pd.DataFrame({"Close": [100.0, 101.0]}, index=raw_index)
        delayed = pd.DataFrame({"Close": [100.0]}, index=[raw_index[1]])

        completed_raw = trim_to_completed(
            raw,
            "1d",
            completed_end,
            timezone,
            daily_available_on_index=False,
        )
        completed_delayed = trim_to_completed(
            delayed,
            "1d",
            completed_end,
            timezone,
            daily_available_on_index=True,
        )

        self.assertEqual(completed_raw.index.tolist(), [raw_index[0]])
        self.assertEqual(completed_delayed.index.tolist(), [raw_index[1]])


if __name__ == "__main__":
    unittest.main()
