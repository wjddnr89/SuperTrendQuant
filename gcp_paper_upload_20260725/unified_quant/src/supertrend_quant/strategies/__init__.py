from __future__ import annotations

import pandas as pd

from ..config import AppConfig
from ..portfolio import AccountSnapshot, OrderPlan
from .base import BenchmarkInput, Strategy
from .common import trend_down_confirmed as _trend_down_confirmed
from .leader_rotation import LeaderRotationStrategy
from .registry import available_strategies, create_strategy, get_strategy_class, register_strategy
from .simple_supertrend import SimpleSupertrendStrategy
from .triple_filters import TripleFiltersStrategy


def build_order_plan(
    config: AppConfig,
    bars: dict[str, pd.DataFrame],
    account: AccountSnapshot,
    mode: str,
    benchmark: BenchmarkInput = None,
    filter_benchmark: BenchmarkInput = None,
) -> OrderPlan:
    strategy = create_strategy(config)
    return strategy.build_order_plan(
        bars,
        account,
        mode=mode,
        benchmark=benchmark,
        filter_benchmark=filter_benchmark,
    )


__all__ = [
    "BenchmarkInput",
    "LeaderRotationStrategy",
    "SimpleSupertrendStrategy",
    "Strategy",
    "TripleFiltersStrategy",
    "_trend_down_confirmed",
    "available_strategies",
    "build_order_plan",
    "create_strategy",
    "get_strategy_class",
    "register_strategy",
]
