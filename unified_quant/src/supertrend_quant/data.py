from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

import pandas as pd

from .config import AppConfig, benchmark_for_symbol, to_yfinance_symbol

if TYPE_CHECKING:
    from .universe import ResolvedUniverse


BenchmarkData = dict[str, pd.DataFrame]


@dataclass(frozen=True)
class MarketData:
    bars: dict[str, pd.DataFrame]
    benchmark: BenchmarkData | None = None
    filter_benchmark: BenchmarkData | None = None
    skipped: tuple[str, ...] = ()
    universe_snapshot: dict[str, object] | None = None


class MarketDataProvider(Protocol):
    """Data-provider seam shared by backtests and research orchestration."""

    def load(self, config: AppConfig, symbols: list[str]) -> MarketData:
        ...


class YahooMarketDataProvider:
    def load(self, config: AppConfig, symbols: list[str]) -> MarketData:
        return _download_yahoo_market_data(config, symbols)


def download_market_data(
    config: AppConfig,
    symbols: list[str],
    provider: MarketDataProvider | None = None,
    resolved_universe: ResolvedUniverse | None = None,
) -> MarketData:
    if provider is not None:
        return provider.load(config, symbols)
    return _download_yahoo_market_data(config, symbols, resolved_universe)


def _download_yahoo_market_data(
    config: AppConfig,
    symbols: list[str],
    resolved_universe: ResolvedUniverse | None = None,
) -> MarketData:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError("yfinance is required. Install project dependencies first.") from exc

    yf_to_symbol = {
        (
            resolved_universe.yfinance_symbol_for(symbol)
            if resolved_universe is not None
            else to_yfinance_symbol(symbol, config.market, config.universe_file)
        ): symbol
        for symbol in symbols
    }
    benchmark_by_symbol = {
        symbol: (
            resolved_universe.benchmark_for(symbol)
            if resolved_universe is not None
            else benchmark_for_symbol(symbol, config.market, config.universe_file)
        )
        for symbol in symbols
    }
    benchmark_tickers = sorted(set(benchmark_by_symbol.values()))
    source_interval = _source_interval(config.timeframe)
    raw = _yf_download(yf, sorted(yf_to_symbol), config.period, source_interval)
    benchmark_raw = _yf_download(yf, benchmark_tickers, config.period, source_interval)

    bars: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for yf_symbol, symbol in yf_to_symbol.items():
        df = extract_ohlc(raw, yf_symbol)
        if df.empty:
            skipped.append(symbol)
        else:
            bars[symbol] = resample_ohlc(df, config.timeframe, config.market)

    benchmark = _resample_benchmark_map(
        _extract_benchmark_map(benchmark_raw, benchmark_by_symbol),
        config.timeframe,
        config.market,
    )
    filter_benchmark = benchmark
    if config.market_trend_filter.enabled and config.market_trend_filter.timeframe != config.timeframe:
        filter_timeframe = config.market_trend_filter.timeframe
        filter_raw = _yf_download(
            yf,
            benchmark_tickers,
            config.period,
            _source_interval(filter_timeframe),
        )
        filter_benchmark = _resample_benchmark_map(
            _extract_benchmark_map(filter_raw, benchmark_by_symbol),
            filter_timeframe,
            config.market,
        )
        if filter_timeframe == "1d" and config.timeframe != "1d":
            filter_benchmark = {
                symbol: delay_daily_until_next_session(df)
                for symbol, df in filter_benchmark.items()
            }

    return MarketData(
        bars=bars,
        benchmark=benchmark or None,
        filter_benchmark=filter_benchmark or None,
        skipped=tuple(skipped),
        universe_snapshot=(
            resolved_universe.snapshot.to_dict()
            if resolved_universe is not None
            else None
        ),
    )


def _yf_download(
    yf,
    tickers: list[str],
    period: str,
    interval: str,
    batch_size: int = 100,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start : start + batch_size]
        frame = yf.download(
            tickers=batch,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        if frame.empty:
            continue
        if not isinstance(frame.columns, pd.MultiIndex) and len(batch) == 1:
            frame = frame.copy()
            frame.columns = pd.MultiIndex.from_product([batch, frame.columns])
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()].sort_index()


def _source_interval(timeframe: str) -> str:
    return "30m" if timeframe in {"1h", "2h", "4h"} else timeframe


def _resample_benchmark_map(
    frames: BenchmarkData,
    timeframe: str,
    market: str,
) -> BenchmarkData:
    return {
        symbol: resample_ohlc(df, timeframe, market)
        for symbol, df in frames.items()
    }


def resample_ohlc(df: pd.DataFrame, timeframe: str, market: str) -> pd.DataFrame:
    """Resample intraday OHLC at the market-open anchor, labeling at bar close.

    Right-edge labels ensure a higher-timeframe bar is not visible to a strategy
    before every source bar in that candle has closed.
    """
    rule = {"1h": "1h", "2h": "2h", "4h": "4h"}.get(timeframe)
    if rule is None or df.empty:
        return df.copy()
    offset = "9h" if market.upper() == "KR" else "9h30min"
    resample_kwargs = {
        "rule": rule,
        "closed": "left",
        "label": "right",
        "origin": "start_day",
        "offset": offset,
    }
    result = (
        df.resample(**resample_kwargs)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    # A regular session can end midway through a 1h/2h/4h bucket.  Label such
    # a partial final bucket at the last contributing 30m candle's close, not
    # at a future theoretical bucket edge after the market has closed.
    source_starts = pd.Series(df.index, index=df.index)
    availability = source_starts.resample(**resample_kwargs).max() + pd.Timedelta(minutes=30)
    result.index = pd.DatetimeIndex([availability.loc[index] for index in result.index])
    return result


def delay_daily_until_next_session(df: pd.DataFrame) -> pd.DataFrame:
    """Make a completed daily candle available no earlier than the next date."""
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index) + pd.offsets.BDay(1)
    return out


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
