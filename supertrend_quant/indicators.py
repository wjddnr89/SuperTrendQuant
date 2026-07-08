from __future__ import annotations

import numpy as np
import pandas as pd


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
    df = df.copy()
    if df.empty:
        return df

    tr = true_range(df)
    if atr_method == "sma":
        atr = tr.rolling(period, min_periods=period).mean()
    elif atr_method == "ewm":
        return calculate_main_jo_supertrend(df, period=period, multiplier=multiplier)
    else:
        atr = rma(tr, period)

    src = (df["High"] + df["Low"]) / 2.0
    close = df["Close"]
    up = src - multiplier * atr
    dn = src + multiplier * atr
    final_up = up.copy()
    final_dn = dn.copy()
    trend = pd.Series(1, index=df.index, dtype="int64")

    for i in range(1, len(df)):
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

    df["ATR"] = atr
    df["ATR_pct"] = atr / close
    df["Supertrend_Up"] = final_up
    df["Supertrend_Down"] = final_dn
    df["Trend"] = trend
    df["BuySignal"] = (df["Trend"] == 1) & (df["Trend"].shift(1) == -1)
    df["SellSignal"] = (df["Trend"] == -1) & (df["Trend"].shift(1) == 1)
    return df


def calculate_main_jo_supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
    """Compatibility path for the Supertrend implementation in main_jo.py."""
    df = df.copy()
    if df.empty or len(df) < period:
        df["Trend"] = 1
        df["ATR_pct"] = 0.02
        df["BuySignal"] = False
        df["SellSignal"] = False
        return df

    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    df["ATR"] = atr
    df["ATR_pct"] = atr / close

    hl2 = (high + low) / 2
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    trend = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if basic_ub.iloc[i] < final_ub.iloc[i - 1] or close.iloc[i - 1] > final_ub.iloc[i - 1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i - 1]

        if basic_lb.iloc[i] > final_lb.iloc[i - 1] or close.iloc[i - 1] < final_lb.iloc[i - 1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i - 1]

    for i in range(1, len(df)):
        if trend.iloc[i - 1] == 1:
            trend.iloc[i] = -1 if close.iloc[i] < final_lb.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if close.iloc[i] > final_ub.iloc[i] else -1

    df["Supertrend_Up"] = final_lb
    df["Supertrend_Down"] = final_ub
    df["Trend"] = trend.astype("int64")
    df["BuySignal"] = (df["Trend"] == 1) & (df["Trend"].shift(1) == -1)
    df["SellSignal"] = (df["Trend"] == -1) & (df["Trend"].shift(1) == 1)
    return df
