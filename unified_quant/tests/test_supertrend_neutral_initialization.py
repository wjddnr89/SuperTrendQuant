import pandas as pd
import pytest

from supertrend_quant.indicators import add_triple_supertrend, calculate_supertrend


def _ohlc(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float).to_numpy()
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
        },
        index=pd.date_range("2026-01-01", periods=len(close), freq="D"),
    )


@pytest.mark.parametrize("atr_method", ["wilder", "sma"])
def test_supertrend_stays_neutral_until_a_previous_valid_band_is_crossed(atr_method):
    featured = calculate_supertrend(
        _ohlc([10, 10, 10, 13, 7, 15]),
        period=3,
        multiplier=1.0,
        atr_method=atr_method,
    )

    # ATR first becomes valid on the third row, but that alone establishes no
    # direction.  The following close crosses the prior upper band.
    assert featured["ATR"].iloc[:2].isna().all()
    assert pd.notna(featured["ATR"].iloc[2])
    assert featured["Trend"].iloc[:3].tolist() == [0, 0, 0]
    assert featured["Trend"].iloc[3] == 1

    # 0 -> +1 establishes the first direction; it is not a reversal signal.
    assert not bool(featured["BuySignal"].iloc[3])
    assert bool(featured["SellSignal"].iloc[4])
    assert bool(featured["BuySignal"].iloc[5])


def test_supertrend_can_remain_neutral_after_atr_is_ready_without_a_breakout():
    featured = calculate_supertrend(
        _ohlc([10] * 8),
        period=3,
        multiplier=2.0,
        atr_method="wilder",
    )

    assert featured["ATR"].iloc[2:].notna().all()
    assert featured["Trend"].eq(0).all()
    assert not featured["BuySignal"].any()
    assert not featured["SellSignal"].any()


def test_triple_supertrend_is_not_all_up_before_each_direction_is_established():
    featured = add_triple_supertrend(
        _ohlc([10, 10, 10, 13, 14]),
        settings=((2, 1.0), (3, 1.0), (3, 1.0)),
        atr_method="wilder",
    )

    assert featured["TripleST1_Trend"].iloc[0] == 0
    assert featured["TripleST2_Trend"].iloc[2] == 0
    assert featured["TripleST3_Trend"].iloc[2] == 0
    assert not featured["TripleAllUp"].iloc[:3].any()
    assert bool(featured["TripleAllUp"].iloc[3])
    assert bool(featured["TripleBuySignal"].iloc[3])


def test_each_identity_segment_restarts_from_neutral():
    first = _ohlc([10, 10, 13, 14])
    second = _ohlc([20, 20, 23, 24])
    second.index = pd.date_range("2026-02-01", periods=len(second), freq="D")
    frame = pd.concat([first, second])
    frame["IdentitySegment"] = ["OLD"] * len(first) + ["NEW"] * len(second)

    featured = calculate_supertrend(frame, period=2, multiplier=1.0)

    assert featured.loc[first.index[2], "Trend"] == 1
    assert featured.loc[second.index[0], "Trend"] == 0
    assert featured.loc[second.index[1], "Trend"] == 0
    assert not bool(featured.loc[second.index[2], "BuySignal"])
