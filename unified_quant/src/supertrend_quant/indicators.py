from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_TRIPLE_SUPERTREND_SETTINGS: tuple[tuple[int, float], ...] = (
    (10, 1.0),
    (11, 2.0),
    (12, 3.0),
)


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


def add_triple_supertrend(
    df: pd.DataFrame,
    settings=DEFAULT_TRIPLE_SUPERTREND_SETTINGS,
    atr_method: str = "wilder",
    exit_down_count: int = 2,
) -> pd.DataFrame:
    """Return a copy with three independently calculated SuperTrend states.

    ``settings`` accepts ``(period, multiplier)`` pairs or mappings with those
    keys.  Keeping the function config-free makes it usable by both the live
    strategy runtime and research workflows.
    """
    normalized = _normalize_triple_settings(settings)
    down_count = int(exit_down_count)
    if down_count < 1 or down_count > len(normalized):
        raise ValueError("exit_down_count must be between 1 and the number of SuperTrend settings.")

    out = df.copy()
    trend_columns: list[str] = []
    for index, (period, multiplier) in enumerate(normalized, start=1):
        st = calculate_supertrend(
            df,
            period=period,
            multiplier=multiplier,
            atr_method=atr_method,
        )
        trend_column = f"TripleST{index}_Trend"
        out[trend_column] = st.get("Trend", pd.Series(index=df.index, dtype="int64"))
        out[f"TripleST{index}_ATR"] = st.get("ATR", pd.Series(np.nan, index=df.index, dtype=float))
        trend_columns.append(trend_column)

    out["TripleAllUp"] = out[trend_columns].eq(1).all(axis=1)
    out["TripleDownCount"] = out[trend_columns].eq(-1).sum(axis=1)
    out["TripleBuySignal"] = out["TripleAllUp"] & ~out["TripleAllUp"].shift(1, fill_value=False)
    threshold_reached = out["TripleDownCount"] >= down_count
    out["TripleSellSignal"] = threshold_reached & ~threshold_reached.shift(1, fill_value=False)
    return out


def calculate_triple_supertrend(
    df: pd.DataFrame,
    settings=DEFAULT_TRIPLE_SUPERTREND_SETTINGS,
    atr_method: str = "wilder",
    exit_down_count: int = 2,
) -> pd.DataFrame:
    """Descriptive alias for :func:`add_triple_supertrend`."""
    return add_triple_supertrend(df, settings, atr_method, exit_down_count)


def add_ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    span_b: int = 52,
    shift: int = 26,
) -> pd.DataFrame:
    """Return a copy with Ichimoku values and long/short cloud states."""
    tenkan = _positive_period(tenkan, "tenkan")
    kijun = _positive_period(kijun, "kijun")
    span_b = _positive_period(span_b, "span_b")
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be non-negative.")

    out = df.copy()
    high = out["High"]
    low = out["Low"]
    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    kijun_line = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0
    span_a_line = ((tenkan_line + kijun_line) / 2.0).shift(shift)
    span_b_line = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2.0).shift(shift)
    cloud_top = pd.concat([span_a_line, span_b_line], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a_line, span_b_line], axis=1).min(axis=1)

    out["Ichimoku_Tenkan"] = tenkan_line
    out["Ichimoku_Kijun"] = kijun_line
    out["Ichimoku_SpanA"] = span_a_line
    out["Ichimoku_SpanB"] = span_b_line
    out["Ichimoku_LongOk"] = out["Close"] > cloud_top
    out["Ichimoku_ShortOk"] = out["Close"] < cloud_bottom
    return out


def calculate_ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    span_b: int = 52,
    shift: int = 26,
) -> pd.DataFrame:
    """Descriptive alias for :func:`add_ichimoku`."""
    return add_ichimoku(df, tenkan, kijun, span_b, shift)


def add_ema_trend(df: pd.DataFrame, period: int = 200) -> pd.DataFrame:
    """Return a copy with an EMA and a close-above-EMA long filter."""
    period = _positive_period(period, "period")
    out = df.copy()
    out["EMA"] = out["Close"].ewm(span=period, adjust=False).mean()
    out["EMA_LongOk"] = out["Close"] > out["EMA"]
    return out


def calculate_ema_trend(df: pd.DataFrame, period: int = 200) -> pd.DataFrame:
    """Descriptive alias for :func:`add_ema_trend`."""
    return add_ema_trend(df, period)


def _normalize_triple_settings(settings) -> tuple[tuple[int, float], ...]:
    try:
        raw_settings = tuple(settings)
    except TypeError as exc:
        raise ValueError("settings must contain exactly three period/multiplier pairs.") from exc
    if len(raw_settings) != 3:
        raise ValueError("settings must contain exactly three period/multiplier pairs.")

    normalized: list[tuple[int, float]] = []
    for index, item in enumerate(raw_settings, start=1):
        if isinstance(item, dict):
            if "period" not in item or "multiplier" not in item:
                raise ValueError(f"settings[{index}] requires period and multiplier.")
            period = item["period"]
            multiplier = item["multiplier"]
        else:
            try:
                period, multiplier = item
            except (TypeError, ValueError) as exc:
                raise ValueError(f"settings[{index}] must be a period/multiplier pair.") from exc
        period = _positive_period(period, f"settings[{index}].period")
        multiplier = float(multiplier)
        if multiplier <= 0:
            raise ValueError(f"settings[{index}].multiplier must be positive.")
        normalized.append((period, multiplier))
    return tuple(normalized)


def _positive_period(value, label: str) -> int:
    period = int(value)
    if period < 1:
        raise ValueError(f"{label} must be positive.")
    return period


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
