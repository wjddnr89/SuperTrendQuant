from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import benchmark_for_symbol, to_yfinance_symbol
from .data import delay_daily_until_next_session, extract_ohlc, resample_ohlc


@dataclass
class YahooStateCache:
    stock_bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    stock_timeframe: str = "30m"
    filter_timeframe: str = "1h"
    period: str = "30d"
    benchmark_bars: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    benchmark_bars_30m: dict[str, pd.DataFrame] = field(default_factory=dict)
    benchmark_close_30m: dict[str, pd.Series] = field(default_factory=dict)
    benchmark_bars_1h: dict[str, pd.DataFrame] = field(default_factory=dict)
    missing_stock_targets: dict[str, pd.Timestamp] = field(default_factory=dict)

    def configure(self, stock_timeframe: str, filter_timeframe: str, period: str = "30d") -> None:
        self.stock_timeframe = stock_timeframe
        self.filter_timeframe = filter_timeframe
        self.period = period

    def sync(
        self,
        symbols: list[str],
        market: str,
        universe_file: str,
        benchmarks: list[str],
        current_candle_base=None,
    ) -> None:
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise RuntimeError("yfinance is required.") from exc

        yf_symbols = [to_yfinance_symbol(symbol, market, universe_file) for symbol in symbols]
        stock_source = _source_interval(self.stock_timeframe)
        raw_stock = yf.download(
            yf_symbols + benchmarks,
            period=self.period,
            interval=stock_source,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
        filter_source = _source_interval(self.filter_timeframe)
        raw_filter = raw_stock if filter_source == stock_source else yf.download(
            benchmarks,
            period=self.period,
            interval=filter_source,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )

        benchmark_frames_stock = {
            benchmark: resample_ohlc(extract_ohlc(raw_stock, benchmark), self.stock_timeframe, market)
            for benchmark in benchmarks
        }
        benchmark_frames_filter = {
            benchmark: resample_ohlc(extract_ohlc(raw_filter, benchmark), self.filter_timeframe, market)
            for benchmark in benchmarks
        }
        if self.filter_timeframe == "1d" and self.stock_timeframe != "1d":
            benchmark_frames_filter = {
                benchmark: delay_daily_until_next_session(df)
                for benchmark, df in benchmark_frames_filter.items()
            }
        self.benchmark_bars[self.stock_timeframe] = benchmark_frames_stock
        self.benchmark_bars[self.filter_timeframe] = benchmark_frames_filter

        for symbol, yf_symbol in zip(symbols, yf_symbols):
            df = resample_ohlc(extract_ohlc(raw_stock, yf_symbol), self.stock_timeframe, market)
            if not df.empty:
                benchmark = benchmark_for_symbol(symbol, market, universe_file)
                bench_df = benchmark_frames_stock.get(benchmark)
                if (
                    self.stock_timeframe == "30m"
                    and current_candle_base is not None
                    and bench_df is not None
                    and not bench_df.empty
                ):
                    df = align_stock_to_benchmark_history(df, bench_df, current_candle_base)
                self.stock_bars[symbol] = df

        for benchmark in benchmarks:
            bench_df = benchmark_frames_stock[benchmark]
            if not bench_df.empty:
                if self.stock_timeframe == "30m":
                    self.benchmark_bars_30m[benchmark] = bench_df
                    self.benchmark_close_30m[benchmark] = bench_df["Close"].dropna()
            filter_df = benchmark_frames_filter[benchmark]
            if not filter_df.empty and self.filter_timeframe == "1h":
                self.benchmark_bars_1h[benchmark] = filter_df

    def fresh_stock_bars(
        self,
        symbols: list[str],
        market_tz,
        current_candle_base,
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        fresh = {}
        stale = []
        current_base = pd.Timestamp(current_candle_base)
        for symbol in symbols:
            df = self.stock_bars.get(symbol)
            if df is None or df.empty:
                stale.append(symbol)
                self.missing_stock_targets[symbol] = current_base
                continue
            completed = trim_to_completed(
                df,
                self.stock_timeframe,
                current_base,
                market_tz,
                daily_available_on_index=False,
            )
            if completed.empty or not _latest_is_fresh(
                completed.index[-1],
                self.stock_timeframe,
                current_base,
                market_tz,
                daily_available_on_index=False,
            ):
                stale.append(symbol)
                self.missing_stock_targets[symbol] = current_base
                continue
            self.missing_stock_targets.pop(symbol, None)
            fresh[symbol] = completed
        return fresh, stale

    def fresh_benchmark_map(
        self,
        symbols: list[str],
        market: str,
        universe_file: str,
        source: str,
        market_tz,
        current_base,
    ) -> dict[str, pd.DataFrame]:
        source_map = self.benchmark_bars.get(source)
        if source_map is None:
            source_map = self.benchmark_bars_1h if source == "1h" else self.benchmark_bars_30m
        out = {}
        for symbol in symbols:
            benchmark = benchmark_for_symbol(symbol, market, universe_file)
            df = source_map.get(benchmark)
            if df is None or df.empty:
                continue
            daily_available_on_index = bool(
                source == "1d"
                and self.filter_timeframe == "1d"
                and self.stock_timeframe != "1d"
            )
            completed = trim_to_completed(
                df,
                source,
                pd.Timestamp(current_base),
                market_tz,
                daily_available_on_index=daily_available_on_index,
            )
            if completed.empty:
                continue
            if _latest_is_fresh(
                completed.index[-1],
                source,
                pd.Timestamp(current_base),
                market_tz,
                daily_available_on_index=daily_available_on_index,
            ):
                out[symbol] = completed
        return out

    def retry_missing(self, market: str, universe_file: str, market_tz, current_candle_base) -> list[str]:
        retry_symbols = [
            symbol
            for symbol, target_base in self.missing_stock_targets.items()
            if pd.Timestamp(target_base) == pd.Timestamp(current_candle_base)
        ]
        if not retry_symbols:
            return []

        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise RuntimeError("yfinance is required.") from exc

        refreshed = []
        yf_symbols = [to_yfinance_symbol(symbol, market, universe_file) for symbol in retry_symbols]
        source_interval = _source_interval(self.stock_timeframe)
        raw_stock = yf.download(
            yf_symbols,
            period=self.period,
            interval=source_interval,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
        for symbol, yf_symbol in zip(retry_symbols, yf_symbols):
            df = resample_ohlc(extract_ohlc(raw_stock, yf_symbol), self.stock_timeframe, market)
            if df.empty:
                continue
            current_base = pd.Timestamp(current_candle_base)
            completed = trim_to_completed(
                df,
                self.stock_timeframe,
                current_base,
                market_tz,
                daily_available_on_index=False,
            )
            if not completed.empty and _latest_is_fresh(
                completed.index[-1],
                self.stock_timeframe,
                current_base,
                market_tz,
                daily_available_on_index=False,
            ):
                self.stock_bars[symbol] = df
                self.missing_stock_targets.pop(symbol, None)
                refreshed.append(symbol)
        return refreshed


def normalize_timestamp(timestamp, market_tz) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(market_tz)
    else:
        ts = ts.tz_localize(market_tz)
    return ts.replace(second=0, microsecond=0)


def align_stock_to_benchmark_history(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    current_candle_base,
) -> pd.DataFrame:
    ffill_until = pd.Timestamp(current_candle_base) - pd.Timedelta(minutes=30)
    bench_index = benchmark_df.dropna(subset=["Close"]).index
    historical_index = bench_index[bench_index <= ffill_until]
    actual_index = stock_df.index[stock_df.index > ffill_until]
    combined_index = historical_index.union(actual_index)
    if historical_index.empty or combined_index.empty:
        return stock_df
    return stock_df.reindex(combined_index).ffill().dropna().copy()


def _source_interval(timeframe: str) -> str:
    return "30m" if timeframe in {"1h", "2h", "4h"} else timeframe


def trim_to_completed(
    df: pd.DataFrame,
    timeframe: str,
    completed_end,
    market_tz,
    *,
    daily_available_on_index: bool,
) -> pd.DataFrame:
    """Hide active/incomplete candles before a strategy receives market data."""
    if df.empty:
        return df.copy()
    cutoff = normalize_timestamp(completed_end, market_tz)
    normalized = pd.DatetimeIndex([normalize_timestamp(item, market_tz) for item in df.index])
    if timeframe == "1d":
        mask = normalized <= cutoff if daily_available_on_index else normalized < cutoff
    elif timeframe == "30m":
        mask = normalized + pd.Timedelta(minutes=30) <= cutoff
    else:
        # Resampled 1h/2h/4h frames are labeled at their right/close edge.
        mask = normalized <= cutoff
    return df.loc[mask].copy()


def _latest_is_fresh(
    latest_index,
    timeframe: str,
    completed_end,
    market_tz,
    *,
    daily_available_on_index: bool,
) -> bool:
    cutoff = normalize_timestamp(completed_end, market_tz)
    latest = normalize_timestamp(latest_index, market_tz)
    if timeframe == "1d":
        availability = latest if daily_available_on_index else latest + pd.offsets.BDay(1)
        age = cutoff - availability
        return pd.Timedelta(0) <= age <= pd.Timedelta(days=7)
    if timeframe == "30m":
        latest += pd.Timedelta(minutes=30)
    return latest >= cutoff
