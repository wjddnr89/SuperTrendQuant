from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLAYGROUND_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "unified_quant" / "src"))
sys.path.insert(0, str(PLAYGROUND_ROOT))

from research_extensions.kospi_market_filters import (  # noqa: E402
    equal_weight_synthetic_benchmark,
    scheduled_membership_mask,
)
from supertrend_quant.data import MarketData  # noqa: E402


def bars(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    close = pd.Series(closes, index=index)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 100.0,
        },
        index=index,
    )


class MembershipTests(unittest.TestCase):
    def test_membership_changes_only_on_effective_date(self):
        index = pd.date_range("2020-01-01", periods=4, freq="B")
        schedule = (
            {
                "effective_date": "2020-01-01",
                "members": [{"symbol": "A"}],
            },
            {
                "effective_date": "2020-01-03",
                "members": [{"symbol": "B"}],
            },
        )

        result = scheduled_membership_mask(schedule, index, ["A", "B"])

        self.assertEqual(result["A"].tolist(), [True, True, False, False])
        self.assertEqual(result["B"].tolist(), [False, False, True, True])

    def test_equal_weight_close_uses_constituent_mean_return(self):
        frame_a = bars([100.0, 110.0, 121.0, 121.0])
        frame_b = bars([100.0, 100.0, 110.0, 121.0])
        index = frame_a.index
        schedule = (
            {
                "effective_date": "2020-01-01",
                "members": [{"symbol": "A"}, {"symbol": "B"}],
            },
        )
        data = MarketData(
            bars={"A": frame_a, "B": frame_b},
            universe_schedule=schedule,
        )

        synthetic, diagnostics = equal_weight_synthetic_benchmark(
            data,
            index,
            minimum_coverage=1.0,
        )

        self.assertAlmostEqual(float(synthetic.iloc[0]["Close"]), 105.0)
        self.assertAlmostEqual(float(synthetic.iloc[1]["Close"]), 115.5)
        self.assertEqual(float(diagnostics.iloc[1]["coverage"]), 1.0)


if __name__ == "__main__":
    unittest.main()
