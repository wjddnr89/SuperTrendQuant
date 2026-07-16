from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..config import AppConfig, ComponentConfig
from ..indicators import add_ema_trend, add_ichimoku, add_triple_supertrend, calculate_supertrend
from ..portfolio import AccountSnapshot, OrderIntent
from .base import BenchmarkInput


def market_filter_allows_buy(config: AppConfig, benchmark: pd.DataFrame | None) -> bool:
    if not config.market_trend_filter.enabled:
        return True
    trend = market_filter_trend(config, benchmark)
    if trend.empty:
        return False
    return int(trend.iloc[-1]) == 1


def market_filter_trend(config: AppConfig, benchmark: pd.DataFrame | None) -> pd.Series:
    """Calculate the causal benchmark trend used by the market filter."""
    if benchmark is None or benchmark.empty:
        return pd.Series(dtype="int64")
    st_df = calculate_supertrend(
        benchmark,
        period=config.supertrend.period,
        multiplier=config.supertrend.multiplier,
        atr_method=config.supertrend.atr_method,
    )
    if st_df.empty or "Trend" not in st_df:
        return pd.Series(dtype="int64")
    return st_df["Trend"].astype("int64")


def precompute_market_filter_trends(
    config: AppConfig,
    symbols: list[str],
    benchmark: BenchmarkInput,
) -> dict[str, pd.Series]:
    """Precompute per-symbol market-filter trends, sharing duplicate frames."""
    if not config.market_trend_filter.enabled or benchmark is None:
        return {}
    frame_cache: dict[int, pd.Series] = {}
    trends: dict[str, pd.Series] = {}
    for symbol in symbols:
        frame = benchmark_for_strategy_symbol(symbol, benchmark)
        if frame is None or frame.empty:
            continue
        cache_key = id(frame)
        trend = frame_cache.get(cache_key)
        if trend is None:
            trend = market_filter_trend(config, frame)
            frame_cache[cache_key] = trend
        trends[symbol] = trend
    return trends


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


def enabled_component(config: AppConfig, group: str, component_type: str) -> ComponentConfig | None:
    return next(
        (
            component
            for component in config.components
            if component.group == group and component.type == component_type and component.enabled
        ),
        None,
    )


def with_strategy_components(config: AppConfig, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Compose configured entry/filter features without coupling the engine to a strategy type."""
    out = with_supertrend(config, symbol, df)
    triple_entry = enabled_component(config, "entries", "triple_supertrend")
    triple_exit = enabled_component(config, "exits", "triple_supertrend_flip")
    if triple_entry is not None or triple_exit is not None:
        params = triple_entry.params if triple_entry is not None else {}
        exit_params = triple_exit.params if triple_exit is not None else {}
        out = add_triple_supertrend(
            out,
            settings=params.get("settings", ((10, 1.0), (11, 2.0), (12, 3.0))),
            atr_method=str(params.get("atr_method", config.supertrend.atr_method)),
            exit_down_count=int(exit_params.get("down_count", 2)),
        )

    ichimoku = enabled_component(config, "filters", "ichimoku_cloud")
    if ichimoku is not None:
        out = add_ichimoku(
            out,
            tenkan=int(ichimoku.params.get("tenkan", 9)),
            kijun=int(ichimoku.params.get("kijun", 26)),
            span_b=int(ichimoku.params.get("span_b", 52)),
            shift=int(ichimoku.params.get("shift", 26)),
        )

    ema = enabled_component(config, "filters", "ema_trend")
    if ema is not None:
        out = add_ema_trend(out, period=int(ema.params.get("period", 200)))
    return out


def entry_state_allows_buy(config: AppConfig, row: pd.Series) -> bool:
    if enabled_component(config, "entries", "triple_supertrend") is not None:
        if not bool(row.get("TripleAllUp", False)):
            return False
        return bool(config.leader_rotation.allow_late_chase or row.get("TripleBuySignal", False))
    if int(row.get("Trend", 0)) != 1:
        return False
    return bool(config.leader_rotation.allow_late_chase or row.get("BuySignal", False))


def asset_filters_allow_buy(config: AppConfig, row: pd.Series) -> bool:
    if enabled_component(config, "filters", "ichimoku_cloud") is not None:
        if not bool(row.get("Ichimoku_LongOk", False)):
            return False
    if enabled_component(config, "filters", "ema_trend") is not None:
        if not bool(row.get("EMA_LongOk", False)):
            return False
    return True


def configured_exit_down_confirmed(config: AppConfig, df: pd.DataFrame) -> bool:
    triple_exit = enabled_component(config, "exits", "triple_supertrend_flip")
    if triple_exit is None:
        return trend_down_confirmed(df, config.exit.sell_confirm_bars)

    confirm_bars = max(1, int(triple_exit.params.get("confirm_bars", config.exit.sell_confirm_bars)))
    down_count = int(triple_exit.params.get("down_count", 2))
    if len(df) < confirm_bars or "TripleDownCount" not in df:
        return False
    return bool((df["TripleDownCount"].tail(confirm_bars).astype(int) >= down_count).all())


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


def scheduled_prepared_slice(
    prepared: dict[str, pd.DataFrame],
    signal_ts,
    account: AccountSnapshot,
    universe_schedule: tuple[Mapping[str, Any], ...],
) -> dict[str, pd.DataFrame]:
    active = active_universe_symbols(universe_schedule, signal_ts)
    allowed = None if active is None else active | set(account.positions)
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in prepared.items():
        if allowed is not None and symbol not in allowed:
            continue
        try:
            selected = frame.loc[:signal_ts]
        except TypeError:
            signal_date = pd.Timestamp(signal_ts).date()
            selected = frame.loc[[pd.Timestamp(idx).date() <= signal_date for idx in frame.index]]
        if not selected.empty:
            sliced[symbol] = selected
    return sliced


def active_universe_symbols(
    universe_schedule: tuple[Mapping[str, Any], ...],
    signal_ts,
) -> set[str] | None:
    if not universe_schedule:
        return None
    signal_date = pd.Timestamp(signal_ts).date()
    selected: set[str] = set()
    for entry in sorted(universe_schedule, key=lambda item: str(item.get("effective_date", ""))):
        try:
            effective_date = pd.Timestamp(entry["effective_date"]).date()
        except Exception:
            continue
        if effective_date > signal_date:
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
