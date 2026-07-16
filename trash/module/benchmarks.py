# -*- coding: utf-8 -*-
"""
Benchmark builders for modular strategy evaluation.

Benchmarks are calculated from daily reference data so Equal B&H, QQQ B&H, and
KR market-index B&H stay stable while intraday strategy timeframes change.
"""

from typing import Dict, Iterable, Optional

import pandas as pd

from module.data import MarketDataBundle, US_BENCHMARK_SYMBOL
from module.metrics import calculate_metrics


def _date_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.to_datetime([pd.Timestamp(item).date() for item in index]), index=index)


def clip_by_date(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    if df.empty:
        return df
    dates = _date_series(df.index)
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= dates >= pd.Timestamp(pd.Timestamp(start).date())
    if end is not None:
        mask &= dates <= pd.Timestamp(pd.Timestamp(end).date())
    return df.loc[mask]


def build_equal_weight_benchmark(
    data: Dict[str, pd.DataFrame],
    initial_cash: float,
    symbols: Optional[Iterable[str]] = None,
    start=None,
    end=None,
) -> pd.Series:
    wanted = set(symbols) if symbols is not None else set(data)
    parts = []
    for symbol in sorted(wanted):
        if symbol not in data:
            continue
        df = clip_by_date(data[symbol], start, end)
        if df.empty:
            continue
        parts.append(df["Close"].rename(symbol))

    if not parts:
        raise ValueError("No symbol data available for Equal B&H benchmark.")

    close_df = pd.concat(parts, axis=1, join="inner").dropna()
    if close_df.empty:
        raise ValueError("No common daily benchmark timeline for Equal B&H.")

    equity = close_df.div(close_df.iloc[0]).mean(axis=1) * initial_cash
    equity.name = "equal_buy_and_hold"
    return equity


def build_single_asset_benchmark(
    df: pd.DataFrame,
    initial_cash: float,
    start=None,
    end=None,
    name: str = "single_asset_buy_and_hold",
) -> pd.Series:
    clipped = clip_by_date(df, start, end)
    if clipped.empty:
        raise ValueError(f"No data available for {name}.")
    equity = clipped["Close"].div(clipped["Close"].iloc[0]) * initial_cash
    equity.name = name
    return equity


def build_kr_market_blend_benchmark(
    index_daily: Dict[str, pd.DataFrame],
    symbol_markets: Dict[str, str],
    symbols: Iterable[str],
    initial_cash: float,
    start=None,
    end=None,
) -> pd.Series:
    parts = []
    for symbol in sorted(symbols):
        market = symbol_markets.get(symbol)
        if market not in index_daily:
            continue
        df = clip_by_date(index_daily[market], start, end)
        if df.empty:
            continue
        parts.append(df["Close"].div(df["Close"].iloc[0]).rename(symbol))

    if not parts:
        raise ValueError("No KOSPI/KOSDAQ index data available for Market B&H.")

    normalized = pd.concat(parts, axis=1, join="inner").dropna()
    if normalized.empty:
        raise ValueError("No common daily benchmark timeline for KR Market B&H.")
    equity = normalized.mean(axis=1) * initial_cash
    equity.name = "kr_market_blend_buy_and_hold"
    return equity


def build_benchmark_report(
    bundle: MarketDataBundle,
    symbols: Iterable[str],
    initial_cash: float,
    interval: str,
    start=None,
    end=None,
) -> Dict[str, dict]:
    report = {}

    equal_equity = build_equal_weight_benchmark(
        bundle.stock_daily,
        initial_cash=initial_cash,
        symbols=symbols,
        start=start,
        end=end,
    )
    report["equal"] = {
        "equity": equal_equity,
        "metrics": calculate_metrics(equal_equity, [], interval),
    }

    if bundle.market == "us" and US_BENCHMARK_SYMBOL in bundle.index_daily:
        qqq_equity = build_single_asset_benchmark(
            bundle.index_daily[US_BENCHMARK_SYMBOL],
            initial_cash=initial_cash,
            start=start,
            end=end,
            name="qqq_buy_and_hold",
        )
        qqq_metrics = calculate_metrics(qqq_equity, [], interval)
        report["qqq"] = {"equity": qqq_equity, "metrics": qqq_metrics}
        report["market"] = {"equity": qqq_equity, "metrics": qqq_metrics}

    if bundle.market == "kr":
        market_equity = build_kr_market_blend_benchmark(
            bundle.index_daily,
            bundle.symbol_markets,
            symbols=symbols,
            initial_cash=initial_cash,
            start=start,
            end=end,
        )
        report["market"] = {
            "equity": market_equity,
            "metrics": calculate_metrics(market_equity, [], interval),
        }

    return report

