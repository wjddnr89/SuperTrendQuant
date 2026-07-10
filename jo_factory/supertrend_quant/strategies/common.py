from __future__ import annotations

import pandas as pd

from ..config import AppConfig
from ..indicators import calculate_supertrend
from ..portfolio import OrderIntent
from .base import BenchmarkInput


def market_filter_allows_buy(config: AppConfig, benchmark: pd.DataFrame | None) -> bool:
    if not config.market_trend_filter.enabled:
        return True
    if benchmark is None or benchmark.empty:
        return False
    st_df = calculate_supertrend(
        benchmark,
        period=config.supertrend.period,
        multiplier=config.supertrend.multiplier,
        atr_method=config.supertrend.atr_method,
    )
    if st_df.empty:
        return False
    return int(st_df["Trend"].iloc[-1]) == 1


def benchmark_for_strategy_symbol(symbol: str, benchmark: BenchmarkInput) -> pd.DataFrame | None:
    if benchmark is None:
        return None
    if isinstance(benchmark, dict):
        return benchmark.get(symbol)
    return benchmark


def with_supertrend(config: AppConfig, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    if not config.supertrend.enabled:
        out = df.copy()
        out["Trend"] = 1
        out["BuySignal"] = False
        out["SellSignal"] = False
        out["ATR_pct"] = 0.0
        return out
    multiplier = config.supertrend.symbol_multipliers.get(symbol, config.supertrend.multiplier)
    return calculate_supertrend(
        df,
        period=config.supertrend.period,
        multiplier=multiplier,
        atr_method=config.supertrend.atr_method,
    )


def trend_down_confirmed(df: pd.DataFrame, confirm_bars: int) -> bool:
    bars = max(1, int(confirm_bars))
    if len(df) < bars:
        return False
    return bool((df["Trend"].tail(bars).astype(int) == -1).all())


def sell_all(position, reason: str) -> OrderIntent:
    return OrderIntent(
        symbol=position.symbol,
        side="sell",
        quantity=position.quantity,
        reason=reason,
    )
