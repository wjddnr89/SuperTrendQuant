from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import AppConfig, benchmark_for_symbol, to_yfinance_symbol


BenchmarkData = dict[str, pd.DataFrame]


@dataclass(frozen=True)
class MarketData:
    bars: dict[str, pd.DataFrame]
    benchmark: BenchmarkData | None = None
    filter_benchmark: BenchmarkData | None = None
    skipped: tuple[str, ...] = ()


def download_market_data(config: AppConfig, symbols: list[str]) -> MarketData:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError("yfinance is required. Install project dependencies first.") from exc

    yf_to_symbol = {
        to_yfinance_symbol(symbol, config.market, config.universe_file): symbol
        for symbol in symbols
    }
    benchmark_by_symbol = {
        symbol: benchmark_for_symbol(symbol, config.market, config.universe_file)
        for symbol in symbols
    }
    benchmark_tickers = sorted(set(benchmark_by_symbol.values()))
    tickers = sorted(set(yf_to_symbol) | set(benchmark_tickers))
    raw = yf.download(
        tickers=tickers,
        period=config.period,
        interval=config.timeframe,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )

    bars: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for yf_symbol, symbol in yf_to_symbol.items():
        df = extract_ohlc(raw, yf_symbol)
        if df.empty:
            skipped.append(symbol)
        else:
            bars[symbol] = df

    benchmark = _extract_benchmark_map(raw, benchmark_by_symbol)
    filter_benchmark = benchmark
    if config.market_trend_filter.enabled and config.market_trend_filter.timeframe != config.timeframe:
        filter_raw = yf.download(
            tickers=benchmark_tickers,
            period=config.period,
            interval=config.market_trend_filter.timeframe,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        filter_benchmark = _extract_benchmark_map(filter_raw, benchmark_by_symbol)

    return MarketData(
        bars=bars,
        benchmark=benchmark or None,
        filter_benchmark=filter_benchmark or None,
        skipped=tuple(skipped),
    )


def _extract_benchmark_map(raw_data: pd.DataFrame, benchmark_by_symbol: dict[str, str]) -> BenchmarkData:
    benchmark_frames = {
        benchmark: extract_ohlc(raw_data, benchmark)
        for benchmark in sorted(set(benchmark_by_symbol.values()))
    }
    return {
        symbol: benchmark_frames[benchmark]
        for symbol, benchmark in benchmark_by_symbol.items()
        if not benchmark_frames[benchmark].empty
    }


def extract_ohlc(raw_data: pd.DataFrame, yf_symbol: str) -> pd.DataFrame:
    if raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        if yf_symbol in raw_data.columns.get_level_values(0):
            df = raw_data[yf_symbol].copy()
        elif yf_symbol in raw_data.columns.get_level_values(1):
            df = raw_data.xs(yf_symbol, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        df = raw_data.copy()

    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    df = df[required].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=required)


def common_index(bars: dict[str, pd.DataFrame]) -> pd.Index:
    close_df = pd.concat(
        [df["Close"].rename(symbol) for symbol, df in bars.items()],
        axis=1,
        join="inner",
    ).dropna()
    if close_df.empty:
        raise ValueError("No common timeline across symbols.")
    return close_df.index
