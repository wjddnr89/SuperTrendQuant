from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, TYPE_CHECKING

import pandas as pd

from ..portfolio import AccountSnapshot, OrderPlan

if TYPE_CHECKING:
    from ..config import AppConfig


BenchmarkInput = pd.DataFrame | dict[str, pd.DataFrame] | None


class PreparedBacktest(Protocol):
    """Timestamp-addressable strategy state prepared once for a backtest."""

    def build_order_plan(
        self,
        signal_ts: Any,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        ...

    def report_frames(self, symbols: set[str]) -> dict[str, pd.DataFrame]:
        """Return the exact precomputed strategy features used by the run."""
        ...


def build_prepared_report_frames(
    prepared: Mapping[str, pd.DataFrame],
    market_filter_trends: Mapping[str, pd.Series],
    symbols: set[str],
) -> dict[str, pd.DataFrame]:
    """Project prepared strategy state into report-ready symbol frames."""
    result: dict[str, pd.DataFrame] = {}
    for symbol in sorted(symbols):
        source = prepared.get(symbol)
        if source is None or source.empty:
            continue
        frame = source.copy()
        trend = market_filter_trends.get(symbol)
        if trend is not None and not trend.empty:
            aligned = trend.reindex(frame.index, method="ffill")
            frame["MarketFilterTrend"] = aligned
        result[symbol] = frame
    return result


class BacktestPreparableStrategy(Protocol):
    """Optional strategy extension consumed by the canonical runner."""

    def prepare_backtest(
        self,
        bars: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput = None,
        filter_benchmark: BenchmarkInput = None,
        universe_schedule: tuple[Mapping[str, Any], ...] = (),
    ) -> PreparedBacktest:
        ...


class Strategy(Protocol):
    strategy_type: ClassVar[str]

    def __init__(self, config: AppConfig):
        ...

    @classmethod
    def validate_config(cls, config: AppConfig) -> None:
        ...

    def warmup_bars(self) -> int:
        ...

    def build_order_plan(
        self,
        bars: dict[str, pd.DataFrame],
        account: AccountSnapshot,
        mode: str,
        benchmark: BenchmarkInput = None,
        filter_benchmark: BenchmarkInput = None,
    ) -> OrderPlan:
        ...


def reject_unknown_params(params: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unsupported params for {label}: {', '.join(sorted(unknown))}")
