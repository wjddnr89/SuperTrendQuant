from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import benchmark_for_symbol, to_yfinance_symbol
from .data import extract_ohlc


@dataclass
class YahooStateCache:
    stock_bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    benchmark_bars_30m: dict[str, pd.DataFrame] = field(default_factory=dict)
    benchmark_close_30m: dict[str, pd.Series] = field(default_factory=dict)
    benchmark_bars_1h: dict[str, pd.DataFrame] = field(default_factory=dict)
    missing_stock_targets: dict[str, pd.Timestamp] = field(default_factory=dict)

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
        raw_30m = yf.download(
            yf_symbols + benchmarks,
            period="30d",
            interval="30m",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
        raw_1h = yf.download(
            benchmarks,
            period="30d",
            interval="1h",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )

        benchmark_frames_30m = {
            benchmark: extract_ohlc(raw_30m, benchmark)
            for benchmark in benchmarks
        }

        for symbol, yf_symbol in zip(symbols, yf_symbols):
            df = extract_ohlc(raw_30m, yf_symbol)
            if not df.empty:
                benchmark = benchmark_for_symbol(symbol, market, universe_file)
                bench_df = benchmark_frames_30m.get(benchmark)
                if current_candle_base is not None and bench_df is not None and not bench_df.empty:
                    df = align_stock_to_benchmark_history(df, bench_df, current_candle_base)
                self.stock_bars[symbol] = df

        for benchmark in benchmarks:
            bench_df = benchmark_frames_30m[benchmark]
            if not bench_df.empty:
                self.benchmark_bars_30m[benchmark] = bench_df
                self.benchmark_close_30m[benchmark] = bench_df["Close"].dropna()
            bench_1h = extract_ohlc(raw_1h, benchmark)
            if not bench_1h.empty:
                self.benchmark_bars_1h[benchmark] = bench_1h

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
            latest_ts = normalize_timestamp(df.index[-1], market_tz)
            if latest_ts < current_base:
                stale.append(symbol)
                self.missing_stock_targets[symbol] = current_base
                continue
            self.missing_stock_targets.pop(symbol, None)
            fresh[symbol] = df
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
        source_map = self.benchmark_bars_1h if source == "1h" else self.benchmark_bars_30m
        out = {}
        for symbol in symbols:
            benchmark = benchmark_for_symbol(symbol, market, universe_file)
            df = source_map.get(benchmark)
            if df is None or df.empty:
                continue
            latest_ts = normalize_timestamp(df.index[-1], market_tz)
            if latest_ts >= pd.Timestamp(current_base):
                out[symbol] = df
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
        raw_30m = yf.download(
            yf_symbols,
            period="30d",
            interval="30m",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
        for symbol, yf_symbol in zip(retry_symbols, yf_symbols):
            df = extract_ohlc(raw_30m, yf_symbol)
            if df.empty:
                continue
            latest_ts = normalize_timestamp(df.index[-1], market_tz)
            if latest_ts >= pd.Timestamp(current_candle_base):
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
