from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from ..config import AppConfig
from ..data import MarketData
from ..metrics import calculate_metrics


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    equity: pd.Series
    metrics: Mapping[str, float | int]


def _bounded_close(frame: pd.DataFrame, run_index: pd.Index | None) -> pd.Series:
    if frame.empty or "Close" not in frame:
        return pd.Series(dtype=float)
    close = frame["Close"].dropna().sort_index().astype(float)
    if close.empty or run_index is None or len(run_index) == 0:
        return close

    start, end = run_index[0], run_index[-1]
    try:
        return close.loc[start:end]
    except (KeyError, TypeError, ValueError):
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()
        mask = [start_date <= pd.Timestamp(item).date() <= end_date for item in close.index]
        return close.loc[mask]


def _equal_buy_and_hold(
    frames: Iterable[tuple[str, pd.DataFrame]],
    *,
    initial_cash: float,
    run_index: pd.Index | None,
    name: str,
) -> pd.Series:
    closes = [
        close.rename(label)
        for label, frame in frames
        if not (close := _bounded_close(frame, run_index)).empty
    ]
    if not closes:
        return pd.Series(dtype=float, name=name)
    table = pd.concat(closes, axis=1, join="inner").dropna()
    if table.empty:
        return pd.Series(dtype=float, name=name)
    equity = table.div(table.iloc[0]).mean(axis=1) * float(initial_cash)
    equity.name = name
    return equity


def build_benchmark_report(
    config: AppConfig,
    market_data: MarketData,
    run_index: pd.Index | None = None,
) -> dict[str, BenchmarkResult]:
    """Build point-range-matched B&H reports from canonical MarketData.

    ``market_data.benchmark`` is keyed per tradable symbol. Averaging those
    normalized series preserves the production universe's market exposure
    (including KOSPI/KOSDAQ membership weights) without assuming ticker names.
    """

    report: dict[str, BenchmarkResult] = {}
    equal = _equal_buy_and_hold(
        sorted(market_data.bars.items()),
        initial_cash=config.capital.initial_cash,
        run_index=run_index,
        name="equal_buy_and_hold",
    )
    if not equal.empty:
        report["equal"] = BenchmarkResult(
            name="equal",
            equity=equal,
            metrics=calculate_metrics(equal, [], config.timeframe),
        )

    benchmark = market_data.benchmark or {}
    market = _equal_buy_and_hold(
        sorted(benchmark.items()),
        initial_cash=config.capital.initial_cash,
        run_index=run_index,
        name="market_buy_and_hold",
    )
    if not market.empty:
        result = BenchmarkResult(
            name="market",
            equity=market,
            metrics=calculate_metrics(market, [], config.timeframe),
        )
        report["market"] = result
        if config.market.upper() == "US":
            report["qqq"] = BenchmarkResult(
                name="qqq",
                equity=market.rename("qqq_buy_and_hold"),
                metrics=result.metrics,
            )
    return report


def benchmark_report_as_dict(report: Mapping[str, BenchmarkResult]) -> dict[str, dict]:
    return {
        name: {"equity": result.equity, "metrics": dict(result.metrics)}
        for name, result in report.items()
    }


__all__ = [
    "BenchmarkResult",
    "benchmark_report_as_dict",
    "build_benchmark_report",
]
