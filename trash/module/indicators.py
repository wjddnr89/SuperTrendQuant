# -*- coding: utf-8 -*-
"""
Indicator calculations used by the modular strategy engine.

The functions here are intentionally pure: they receive OHLCV data and return
new DataFrames or Series.  Strategy modules can combine supertrend_quant-style
entry and filter components such as SuperTrend, Triple SuperTrend, Ichimoku
cloud, EMA trend, ATR percentage, and RS without duplicating indicator code.
"""

import numpy as np
import pandas as pd

from module.config import StrategyConfig, asset_filter_list, normalize_signal


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    if not tr.empty:
        tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def rma(series: pd.Series, length: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < length:
        return pd.Series(out, index=series.index)

    first_idx = length - 1
    out[first_idx] = np.nanmean(values[:length])
    alpha = 1.0 / length
    for i in range(first_idx + 1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    atr_method: str = "wilder",
) -> pd.DataFrame:
    out = df.copy()
    tr = true_range(out)
    if atr_method == "sma":
        atr = tr.rolling(period, min_periods=period).mean()
    else:
        atr = rma(tr, period)

    src = (out["High"] + out["Low"]) / 2.0
    close = out["Close"]

    up = src - multiplier * atr
    dn = src + multiplier * atr
    final_up = up.copy()
    final_dn = dn.copy()
    trend = pd.Series(1, index=out.index, dtype="int64")

    for i in range(1, len(out)):
        prev_up = final_up.iloc[i - 1]
        prev_dn = final_dn.iloc[i - 1]
        up1 = prev_up if not pd.isna(prev_up) else up.iloc[i]
        dn1 = prev_dn if not pd.isna(prev_dn) else dn.iloc[i]

        if not pd.isna(up.iloc[i]) and not pd.isna(up1) and close.iloc[i - 1] > up1:
            final_up.iloc[i] = max(up.iloc[i], up1)
        if not pd.isna(dn.iloc[i]) and not pd.isna(dn1) and close.iloc[i - 1] < dn1:
            final_dn.iloc[i] = min(dn.iloc[i], dn1)

        prev_trend = trend.iloc[i - 1]
        if prev_trend == -1 and not pd.isna(dn1) and close.iloc[i] > dn1:
            trend.iloc[i] = 1
        elif prev_trend == 1 and not pd.isna(up1) and close.iloc[i] < up1:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prev_trend

    out["ATR"] = atr
    out["Supertrend_Up"] = final_up
    out["Supertrend_Down"] = final_dn
    out["Trend"] = trend
    out["BuySignal"] = (out["Trend"] == 1) & (out["Trend"].shift(1) == -1)
    out["SellSignal"] = (out["Trend"] == -1) & (out["Trend"].shift(1) == 1)
    return out


def add_triple_supertrend(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    trend_columns = []
    for idx, (period, multiplier) in enumerate(config.triple_settings, start=1):
        st = calculate_supertrend(
            df,
            period=period,
            multiplier=multiplier,
            atr_method=config.atr_method,
        )
        trend_col = f"TripleST{idx}_Trend"
        out[trend_col] = st["Trend"]
        out[f"TripleST{idx}_ATR"] = st["ATR"]
        trend_columns.append(trend_col)

    out["TripleAllUp"] = out[trend_columns].eq(1).all(axis=1)
    out["TripleDownCount"] = out[trend_columns].eq(-1).sum(axis=1)
    out["TripleBuySignal"] = out["TripleAllUp"] & ~out["TripleAllUp"].shift(1, fill_value=False)
    out["TripleSellSignal"] = (
        out["TripleDownCount"] >= config.triple_exit_down_count
    ) & (
        out["TripleDownCount"].shift(1, fill_value=0) < config.triple_exit_down_count
    )
    return out


def add_ichimoku(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    high = out["High"]
    low = out["Low"]

    tenkan = (
        high.rolling(config.ichimoku_tenkan).max()
        + low.rolling(config.ichimoku_tenkan).min()
    ) / 2.0
    kijun = (
        high.rolling(config.ichimoku_kijun).max()
        + low.rolling(config.ichimoku_kijun).min()
    ) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(config.ichimoku_shift)
    span_b = (
        (
            high.rolling(config.ichimoku_span_b).max()
            + low.rolling(config.ichimoku_span_b).min()
        )
        / 2.0
    ).shift(config.ichimoku_shift)

    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
    out["Ichimoku_Tenkan"] = tenkan
    out["Ichimoku_Kijun"] = kijun
    out["Ichimoku_SpanA"] = span_a
    out["Ichimoku_SpanB"] = span_b
    out["Ichimoku_LongOk"] = out["Close"] > cloud_top
    out["Ichimoku_ShortOk"] = out["Close"] < cloud_bottom
    return out


def add_ema_trend(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["EMA"] = out["Close"].ewm(span=config.ema_period, adjust=False).mean()
    out["EMA_LongOk"] = out["Close"] > out["EMA"]
    return out


def add_strategy_features(
    df: pd.DataFrame,
    config: StrategyConfig,
    rs_period: int,
) -> pd.DataFrame:
    out = calculate_supertrend(
        df,
        period=config.st_period,
        multiplier=config.st_multiplier,
        atr_method=config.atr_method,
    )
    out["ATR_pct"] = out["ATR"] / out["Close"]
    out["RS"] = out["Close"].pct_change(rs_period)

    signal = normalize_signal(config.signal)
    if signal == "triple_supertrend":
        out = add_triple_supertrend(out, config)
    filters = asset_filter_list(config.asset_filter)
    if "ichimoku_cloud" in filters:
        out = add_ichimoku(out, config)
    if "ema_trend" in filters:
        out = add_ema_trend(out, config)
    return out


def entry_state(row: pd.Series, config: StrategyConfig) -> bool:
    if normalize_signal(config.signal) == "triple_supertrend":
        if bool(row.get("TripleAllUp", False)):
            return bool(config.allow_late_chase or row.get("TripleBuySignal", False))
        return False

    if int(row.get("Trend", 0)) == 1:
        return bool(config.allow_late_chase or row.get("BuySignal", False))
    return False


def exit_state(row: pd.Series, config: StrategyConfig) -> bool:
    if normalize_signal(config.signal) == "triple_supertrend":
        return int(row.get("TripleDownCount", 0)) >= config.triple_exit_down_count
    return int(row.get("Trend", 0)) == -1


def raw_uptrend_state(row: pd.Series, config: StrategyConfig) -> bool:
    if normalize_signal(config.signal) == "triple_supertrend":
        return bool(row.get("TripleAllUp", False))
    return int(row.get("Trend", 0)) == 1
