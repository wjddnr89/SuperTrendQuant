from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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


def _rolling_equal_hold(
    frames: Mapping[str, pd.DataFrame],
    schedule: tuple[Mapping[str, Any], ...],
    *,
    initial_cash: float,
    run_index: pd.Index | None,
    name: str,
) -> pd.Series:
    if not schedule:
        return pd.Series(dtype=float, name=name)
    index = pd.Index(run_index) if run_index is not None else _schedule_index(frames, schedule)
    if len(index) == 0:
        return pd.Series(dtype=float, name=name)
    closes = {
        symbol: frame["Close"].dropna().sort_index().astype(float)
        for symbol, frame in frames.items()
        if "Close" in frame and not frame.empty
    }
    equity = float(initial_cash)
    points: list[tuple[object, float]] = []
    previous_prices: dict[str, float] = {}
    for timestamp in index:
        active = _active_symbols(schedule, timestamp)
        current_prices = {
            symbol: price
            for symbol in active
            if (price := _price_at_or_before(closes.get(symbol), timestamp)) is not None
        }
        returns = [
            current_prices[symbol] / previous_prices[symbol] - 1.0
            for symbol in current_prices
            if symbol in previous_prices and previous_prices[symbol] > 0
        ]
        if returns:
            equity *= 1.0 + float(pd.Series(returns).mean())
        points.append((timestamp, equity))
        previous_prices = current_prices
    return pd.Series([value for _, value in points], index=[ts for ts, _ in points], name=name)


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
    if market_data.universe_schedule:
        equal = _rolling_equal_hold(
            market_data.bars,
            market_data.universe_schedule,
            initial_cash=config.capital.initial_cash,
            run_index=run_index,
            name="rolling_equal_hold",
        )
    else:
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


def _schedule_index(
    frames: Mapping[str, pd.DataFrame],
    schedule: tuple[Mapping[str, Any], ...],
) -> pd.Index:
    pieces = []
    for entry in schedule:
        for symbol in _entry_symbols(entry):
            frame = frames.get(symbol)
            if frame is not None and not frame.empty:
                pieces.append(pd.Index(frame.index))
    if not pieces:
        return pd.Index([])
    index = pieces[0]
    for piece in pieces[1:]:
        index = index.union(piece)
    return index.sort_values().drop_duplicates()


def _active_symbols(schedule: tuple[Mapping[str, Any], ...], timestamp) -> set[str]:
    timestamp_date = pd.Timestamp(timestamp).date()
    selected: set[str] = set()
    for entry in sorted(schedule, key=lambda item: str(item.get("effective_date", ""))):
        if pd.Timestamp(entry["effective_date"]).date() > timestamp_date:
            break
        selected = _entry_symbols(entry)
    return selected


def _entry_symbols(entry: Mapping[str, Any]) -> set[str]:
    symbols = entry.get("symbols")
    if isinstance(symbols, (list, tuple)):
        return {str(symbol) for symbol in symbols if str(symbol)}
    members = entry.get("members", ())
    if isinstance(members, (list, tuple)):
        return {
            str(member.get("symbol"))
            for member in members
            if isinstance(member, Mapping) and member.get("symbol")
        }
    return set()


def _price_at_or_before(series: pd.Series | None, timestamp) -> float | None:
    if series is None or series.empty:
        return None
    try:
        available = series.loc[:timestamp]
    except TypeError:
        signal_date = pd.Timestamp(timestamp).date()
        available = series.loc[[pd.Timestamp(idx).date() <= signal_date for idx in series.index]]
    if available.empty:
        return None
    return float(available.iloc[-1])


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
