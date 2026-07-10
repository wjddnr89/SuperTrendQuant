# -*- coding: utf-8 -*-
"""
Data loading and timeframe preparation for the modular SuperTrend research backtest.

The module reads universe.json, downloads Yahoo Finance OHLCV data, resamples
30-minute bars into higher intraday bars, and keeps benchmark/index data in one
MarketDataBundle so every strategy config can reuse the same downloaded data.
"""

import contextlib
import io
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from module.config import BASE_INTRADAY_INTERVAL, DAILY_INTERVAL, TIMEFRAMES


US_BENCHMARK_SYMBOL = "QQQ"
KR_INDEX_ASSETS = {
    "KOSPI": ("KOSPI", "^KS11"),
    "KOSDAQ": ("KOSDAQ", "^KQ11"),
}


@dataclass(frozen=True)
class Asset:
    symbol: str
    yf_symbol: str
    market: str


@dataclass
class MarketDataBundle:
    market: str
    period: str
    assets: List[Asset]
    symbol_markets: Dict[str, str]
    stock_30m: Dict[str, pd.DataFrame]
    stock_daily: Dict[str, pd.DataFrame]
    index_30m: Dict[str, pd.DataFrame]
    index_daily: Dict[str, pd.DataFrame]
    skipped: Dict[str, List[Asset]]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_universe_path() -> Path:
    return project_root() / "universe.json"


def configure_yfinance_cache() -> None:
    try:
        import yfinance as yf
    except ModuleNotFoundError:
        return

    cache_dir = Path(tempfile.gettempdir()) / "trading_bot_yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def load_universe(universe_path: Path, market: str) -> List[Asset]:
    with open(universe_path, "r", encoding="utf-8") as file:
        universe = json.load(file)

    assets: List[Asset] = []
    if market == "kr":
        for symbol, kr_market in universe.get("KR_UNIVERSE_MAP", {}).items():
            if kr_market == "KOSPI":
                yf_symbol = f"{symbol}.KS"
            elif kr_market == "KOSDAQ":
                yf_symbol = f"{symbol}.KQ"
            else:
                yf_symbol = symbol
            assets.append(Asset(symbol=symbol, yf_symbol=yf_symbol, market=kr_market))
    elif market == "us":
        for symbol in universe.get("US_UNIVERSE_LIST", []):
            assets.append(Asset(symbol=symbol, yf_symbol=symbol, market="US"))
    else:
        raise ValueError("market must be 'us' or 'kr'.")

    if not assets:
        raise ValueError(f"No assets found in {universe_path} for market={market}.")
    return assets


def benchmark_assets(market: str) -> List[Asset]:
    if market == "us":
        return [Asset(symbol=US_BENCHMARK_SYMBOL, yf_symbol=US_BENCHMARK_SYMBOL, market="US")]

    return [
        Asset(symbol=symbol, yf_symbol=yf_symbol, market=symbol)
        for symbol, yf_symbol in KR_INDEX_ASSETS.values()
    ]


def extract_ohlcv(raw_data: pd.DataFrame, yf_symbol: str) -> pd.DataFrame:
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
    if not all(column in df.columns for column in required):
        return pd.DataFrame()

    columns = required + (["Volume"] if "Volume" in df.columns else [])
    df = df[columns].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df


def download_data(
    assets: Iterable[Asset],
    period: str,
    interval: str,
    start: str = None,
    end: str = None,
) -> Tuple[Dict[str, pd.DataFrame], List[Asset]]:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yfinance is required. Install dependencies with: "
            "venv\\Scripts\\python.exe -m pip install -r SuperTrendQuant\\requirements.txt"
        ) from exc

    assets = list(assets)
    yf_symbols = sorted({asset.yf_symbol for asset in assets})
    kwargs = {
        "tickers": yf_symbols,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
        "threads": True,
        "group_by": "ticker",
    }
    if start or end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        raw_data = yf.download(**kwargs)

    data: Dict[str, pd.DataFrame] = {}
    skipped: List[Asset] = []
    for asset in assets:
        df = extract_ohlcv(raw_data, asset.yf_symbol)
        if df.empty:
            skipped.append(asset)
            continue
        data[asset.symbol] = df
    return data, skipped


def load_market_data(
    market: str,
    universe_path: Path = None,
    period: str = "60d",
) -> MarketDataBundle:
    universe_path = Path(universe_path) if universe_path else default_universe_path()
    configure_yfinance_cache()

    assets = load_universe(universe_path, market)
    indexes = benchmark_assets(market)

    stock_30m, skipped_stock_30m = download_data(assets, period, BASE_INTRADAY_INTERVAL)
    stock_daily, skipped_stock_daily = download_data(assets, period, DAILY_INTERVAL)
    index_30m, skipped_index_30m = download_data(indexes, period, BASE_INTRADAY_INTERVAL)
    index_daily, skipped_index_daily = download_data(indexes, period, DAILY_INTERVAL)

    if not stock_30m:
        raise RuntimeError(f"No {market.upper()} 30m stock data was downloaded.")
    if not stock_daily:
        raise RuntimeError(f"No {market.upper()} daily stock data was downloaded.")

    return MarketDataBundle(
        market=market,
        period=period,
        assets=assets,
        symbol_markets={asset.symbol: asset.market for asset in assets},
        stock_30m=stock_30m,
        stock_daily=stock_daily,
        index_30m=index_30m,
        index_daily=index_daily,
        skipped={
            "stock_30m": skipped_stock_30m,
            "stock_daily": skipped_stock_daily,
            "index_30m": skipped_index_30m,
            "index_daily": skipped_index_daily,
        },
    )


def resample_offset(market: str) -> str:
    return "9h" if market == "kr" else "9h30min"


def resample_ohlcv(df: pd.DataFrame, rule: str, market: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return (
        df.resample(
            rule,
            closed="left",
            label="left",
            origin="start_day",
            offset=resample_offset(market),
        )
        .agg(agg)
        .dropna(subset=["Open", "High", "Low", "Close"])
    )


def resample_data(data: Dict[str, pd.DataFrame], rule: str, market: str) -> Dict[str, pd.DataFrame]:
    output = {}
    for symbol, df in data.items():
        resampled = resample_ohlcv(df, rule, market)
        if not resampled.empty:
            output[symbol] = resampled
    return output


def select_timeframe_data(
    bundle: MarketDataBundle,
    timeframe: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    tf = TIMEFRAMES[timeframe]
    if tf["source"] == "daily":
        return (
            {symbol: df.copy() for symbol, df in bundle.stock_daily.items()},
            {symbol: df.copy() for symbol, df in bundle.index_daily.items()},
        )

    if tf["rule"] is None:
        return (
            {symbol: df.copy() for symbol, df in bundle.stock_30m.items()},
            {symbol: df.copy() for symbol, df in bundle.index_30m.items()},
        )

    return (
        resample_data(bundle.stock_30m, tf["rule"], bundle.market),
        resample_data(bundle.index_30m, tf["rule"], bundle.market),
    )


def get_common_index(data: Dict[str, pd.DataFrame]) -> pd.Index:
    close_df = pd.concat(
        [df["Close"].rename(symbol) for symbol, df in data.items()],
        axis=1,
        join="inner",
    ).dropna()
    if close_df.empty:
        raise ValueError("No common timeline across selected tickers.")
    return close_df.index


def filter_by_history_coverage(
    data: Dict[str, pd.DataFrame],
    min_coverage: float,
) -> Tuple[Dict[str, pd.DataFrame], List[Tuple[str, int, int]]]:
    if min_coverage <= 0 or not data:
        return data, []

    max_bars = max(len(df) for df in data.values())
    min_bars = math.ceil(max_bars * min_coverage)

    kept = {}
    dropped = []
    for symbol, df in data.items():
        if len(df) >= min_bars:
            kept[symbol] = df
        else:
            dropped.append((symbol, len(df), min_bars))

    if not kept:
        raise ValueError("All symbols were dropped by min_coverage.")
    return kept, dropped
