import unittest

import pandas as pd

from supertrend_quant.indicators import calculate_supertrend


def main_jo_supertrend_reference(df, period=7, multiplier=3.0):
    df = df.copy()
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
    hl2 = (high + low) / 2
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        final_ub.iloc[i] = (
            basic_ub.iloc[i]
            if basic_ub.iloc[i] < final_ub.iloc[i - 1] or close.iloc[i - 1] > final_ub.iloc[i - 1]
            else final_ub.iloc[i - 1]
        )
        final_lb.iloc[i] = (
            basic_lb.iloc[i]
            if basic_lb.iloc[i] > final_lb.iloc[i - 1] or close.iloc[i - 1] < final_lb.iloc[i - 1]
            else final_lb.iloc[i - 1]
        )
    for i in range(1, len(df)):
        if trend.iloc[i - 1] == 1:
            trend.iloc[i] = -1 if close.iloc[i] < final_lb.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if close.iloc[i] > final_ub.iloc[i] else -1
    return trend.astype("int64")


class MainJoIndicatorCompatTest(unittest.TestCase):
    def test_ewm_supertrend_matches_main_jo_reference(self):
        df = pd.DataFrame(
            {
                "Open": [10, 11, 12, 11, 9, 8, 10, 12, 13, 11],
                "High": [11, 12, 13, 12, 10, 9, 11, 13, 14, 12],
                "Low": [9, 10, 11, 10, 8, 7, 9, 11, 12, 10],
                "Close": [10, 11, 12, 11, 9, 8, 10, 12, 13, 11],
            },
            index=pd.date_range("2026-01-01", periods=10, freq="30min"),
        )

        migrated = calculate_supertrend(df, period=3, multiplier=2.0, atr_method="ewm")
        expected = main_jo_supertrend_reference(df, period=3, multiplier=2.0)

        pd.testing.assert_series_equal(migrated["Trend"], expected, check_names=False)


if __name__ == "__main__":
    unittest.main()
