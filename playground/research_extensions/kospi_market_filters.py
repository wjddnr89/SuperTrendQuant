from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from supertrend_quant.config import AppConfig
from supertrend_quant.data import MarketData
from supertrend_quant.indicators import calculate_supertrend
from supertrend_quant.strategies.leader_rotation import PreparedLeaderBacktest


@dataclass(frozen=True)
class FilterVariantResult:
    config: AppConfig
    market_filter_trends: dict[str, pd.Series]
    regime: pd.Series
    diagnostics: pd.DataFrame
    synthetic_benchmark: pd.DataFrame | None = None


def scheduled_membership_mask(
    schedule: tuple[dict[str, Any], ...],
    index: pd.Index,
    symbols: list[str],
) -> pd.DataFrame:
    """Build a causal point-in-time membership mask."""

    dates = pd.DatetimeIndex(index)
    symbol_positions = {symbol: position for position, symbol in enumerate(symbols)}
    mask = np.zeros((len(dates), len(symbols)), dtype=bool)
    entries = sorted(
        (
            (pd.Timestamp(str(entry["effective_date"])), entry)
            for entry in schedule
            if entry.get("effective_date")
        ),
        key=lambda item: item[0],
    )
    if not entries:
        raise ValueError("Point-in-time universe schedule is required.")

    entry_position = -1
    active: set[str] = set()
    for row_position, session in enumerate(dates):
        while (
            entry_position + 1 < len(entries)
            and entries[entry_position + 1][0] <= session
        ):
            entry_position += 1
            active = _entry_symbols(entries[entry_position][1])
        for symbol in active:
            column = symbol_positions.get(symbol)
            if column is not None:
                mask[row_position, column] = True
    return pd.DataFrame(mask, index=dates, columns=symbols)


def constituent_breadth_regime(
    data: MarketData,
    config: AppConfig,
    index: pd.Index,
    *,
    threshold: float,
    minimum_coverage: float,
) -> tuple[pd.Series, pd.DataFrame]:
    symbols = sorted(data.bars)
    membership = scheduled_membership_mask(
        data.universe_schedule,
        index,
        symbols,
    )
    trends = pd.DataFrame(index=pd.DatetimeIndex(index), columns=symbols, dtype=float)
    for symbol in symbols:
        frame = data.bars[symbol]
        featured = calculate_supertrend(
            frame,
            period=config.supertrend.period,
            multiplier=config.supertrend.multiplier,
            atr_method=config.supertrend.atr_method,
        )
        trends[symbol] = (
            featured["Trend"]
            .reindex(trends.index)
            .ffill()
            .astype(float)
        )

    active_trends = trends.where(membership)
    member_count = membership.sum(axis=1).astype(float)
    valid_count = active_trends.notna().sum(axis=1).astype(float)
    up_count = active_trends.eq(1).sum(axis=1).astype(float)
    breadth = up_count.div(valid_count.replace(0.0, np.nan))
    coverage = valid_count.div(member_count.replace(0.0, np.nan))
    usable = coverage.ge(float(minimum_coverage)) & breadth.notna()
    regime = pd.Series(
        np.where(usable & breadth.ge(float(threshold)), 1, -1),
        index=trends.index,
        dtype="int64",
        name="breadth_regime",
    )
    diagnostics = pd.DataFrame(
        {
            "member_count": member_count,
            "valid_member_count": valid_count,
            "coverage": coverage,
            "breadth_ratio": breadth,
            "regime": regime,
        }
    )
    return regime, diagnostics


def equal_weight_synthetic_benchmark(
    data: MarketData,
    index: pd.Index,
    *,
    minimum_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a daily rebalanced equal-weight OHLC index causally."""

    symbols = sorted(data.bars)
    dates = pd.DatetimeIndex(index)
    membership = scheduled_membership_mask(
        data.universe_schedule,
        dates,
        symbols,
    )
    ratio_frames = {
        column: pd.DataFrame(index=dates, columns=symbols, dtype=float)
        for column in ("Open", "High", "Low", "Close")
    }
    volumes = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    for symbol in symbols:
        frame = data.bars[symbol].sort_index()
        previous_close = frame["Close"].shift(1)
        for column in ratio_frames:
            ratio_frames[column][symbol] = (
                frame[column].div(previous_close).reindex(dates)
            )
        volumes[symbol] = frame["Volume"].reindex(dates)

    active_close = ratio_frames["Close"].where(membership)
    member_count = membership.sum(axis=1).astype(float)
    valid_count = active_close.notna().sum(axis=1).astype(float)
    coverage = valid_count.div(member_count.replace(0.0, np.nan))
    usable = coverage.ge(float(minimum_coverage))
    mean_ratios = {
        column: frame.where(membership).mean(axis=1).where(usable)
        for column, frame in ratio_frames.items()
    }
    total_volume = volumes.where(membership).sum(axis=1).where(usable)

    records: list[dict[str, float]] = []
    record_dates: list[pd.Timestamp] = []
    prior_level = 100.0
    for session in dates:
        close_ratio = mean_ratios["Close"].get(session)
        if not np.isfinite(close_ratio) or float(close_ratio) <= 0.0:
            continue
        open_level = prior_level * float(mean_ratios["Open"].get(session))
        high_level = prior_level * float(mean_ratios["High"].get(session))
        low_level = prior_level * float(mean_ratios["Low"].get(session))
        close_level = prior_level * float(close_ratio)
        high_level = max(high_level, open_level, close_level)
        low_level = min(low_level, open_level, close_level)
        records.append(
            {
                "Open": open_level,
                "High": high_level,
                "Low": low_level,
                "Close": close_level,
                "Volume": float(total_volume.get(session) or 0.0),
            }
        )
        record_dates.append(session)
        prior_level = close_level

    synthetic = pd.DataFrame(
        records,
        index=pd.DatetimeIndex(record_dates, name=dates.name),
    )
    diagnostics = pd.DataFrame(
        {
            "member_count": member_count,
            "valid_member_count": valid_count,
            "coverage": coverage,
        }
    )
    return synthetic, diagnostics


def build_filter_variant(
    variant: dict[str, Any],
    *,
    base_config: AppConfig,
    data: MarketData,
    canonical_prepared: PreparedLeaderBacktest,
    full_index: pd.Index,
) -> FilterVariantResult:
    variant_type = str(variant["type"])
    prepared_symbols = sorted(canonical_prepared.prepared)

    if variant_type == "cap_weight":
        trends = dict(canonical_prepared.market_filter_trends)
        regime = _representative_regime(trends, full_index)
        diagnostics = pd.DataFrame({"regime": regime})
        return FilterVariantResult(
            config=base_config,
            market_filter_trends=trends,
            regime=regime,
            diagnostics=diagnostics,
        )

    if variant_type == "none":
        config = replace(
            base_config,
            market_trend_filter=replace(
                base_config.market_trend_filter,
                enabled=False,
            ),
        )
        regime = pd.Series(
            1,
            index=pd.DatetimeIndex(full_index),
            dtype="int64",
            name="no_filter_regime",
        )
        return FilterVariantResult(
            config=config,
            market_filter_trends={},
            regime=regime,
            diagnostics=pd.DataFrame({"regime": regime}),
        )

    if variant_type == "breadth":
        regime, diagnostics = constituent_breadth_regime(
            data,
            base_config,
            full_index,
            threshold=float(variant["threshold"]),
            minimum_coverage=float(variant["minimum_coverage"]),
        )
        trends = {symbol: regime for symbol in prepared_symbols}
        return FilterVariantResult(
            config=base_config,
            market_filter_trends=trends,
            regime=regime,
            diagnostics=diagnostics,
        )

    if variant_type == "equal_weight":
        synthetic, diagnostics = equal_weight_synthetic_benchmark(
            data,
            full_index,
            minimum_coverage=float(variant["minimum_coverage"]),
        )
        featured = calculate_supertrend(
            synthetic,
            period=base_config.supertrend.period,
            multiplier=base_config.supertrend.multiplier,
            atr_method=base_config.supertrend.atr_method,
        )
        regime = (
            featured["Trend"]
            .reindex(pd.DatetimeIndex(full_index))
            .ffill()
            .fillna(-1)
            .astype("int64")
            .rename("equal_weight_regime")
        )
        diagnostics = diagnostics.copy()
        diagnostics["regime"] = regime
        trends = {symbol: regime for symbol in prepared_symbols}
        return FilterVariantResult(
            config=base_config,
            market_filter_trends=trends,
            regime=regime,
            diagnostics=diagnostics,
            synthetic_benchmark=synthetic,
        )

    raise ValueError(f"Unsupported market filter variant: {variant_type}")


def _entry_symbols(entry: dict[str, Any]) -> set[str]:
    symbols = entry.get("symbols")
    if isinstance(symbols, (list, tuple)):
        return {str(symbol) for symbol in symbols if str(symbol)}
    return {
        str(member["symbol"])
        for member in entry.get("members", ())
        if isinstance(member, dict) and member.get("symbol")
    }


def _representative_regime(
    trends: dict[str, pd.Series],
    index: pd.Index,
) -> pd.Series:
    if not trends:
        return pd.Series(-1, index=pd.DatetimeIndex(index), dtype="int64")
    source = next(iter(trends.values()))
    return (
        source.reindex(pd.DatetimeIndex(index))
        .ffill()
        .fillna(-1)
        .astype("int64")
        .rename("cap_weight_regime")
    )
