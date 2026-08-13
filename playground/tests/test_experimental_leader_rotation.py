from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_SRC = PROJECT_ROOT / "unified_quant" / "src"
PLAYGROUND_ROOT = PROJECT_ROOT / "playground"
sys.path.insert(0, str(UNIFIED_SRC))
sys.path.insert(0, str(PLAYGROUND_ROOT))

from research_extensions.experimental_leader_rotation import (  # noqa: E402
    ExperimentalLeaderPolicy,
    ExperimentalSignalCache,
    late_chase_allows_entry,
)
from scripts.nested_walk_forward_nasdaq_structure import is_eligible  # noqa: E402


class ExperimentalLeaderPolicyTests(unittest.TestCase):
    def test_rotation_gate_values_are_explicit(self) -> None:
        self.assertEqual(
            ExperimentalLeaderPolicy(rotation_profit_gate="off").rotation_profit_gate,
            "off",
        )
        with self.assertRaises(ValueError):
            ExperimentalLeaderPolicy(rotation_profit_gate="minus_three")

    def test_fixed_stop_magnitude_is_validated(self) -> None:
        ExperimentalLeaderPolicy(stop_loss_pct=0.08)
        with self.assertRaises(ValueError):
            ExperimentalLeaderPolicy(stop_loss_pct=-0.08)
        with self.assertRaises(ValueError):
            ExperimentalLeaderPolicy(stop_loss_pct=1.0)

    def test_fresh_only_requires_buy_signal(self) -> None:
        stale = pd.Series({"Trend": 1, "BuySignal": False})
        fresh = pd.Series({"Trend": 1, "BuySignal": True})
        self.assertFalse(late_chase_allows_entry(stale, mode="fresh_only"))
        self.assertTrue(late_chase_allows_entry(fresh, mode="fresh_only"))

    def test_unlimited_allows_any_uptrend(self) -> None:
        stale = pd.Series({"Trend": 1, "BuySignal": False})
        down = pd.Series({"Trend": -1, "BuySignal": True})
        self.assertTrue(late_chase_allows_entry(stale, mode="unlimited"))
        self.assertFalse(late_chase_allows_entry(down, mode="unlimited"))

    def test_two_of_three_positive_years_passes_decimal_threshold(self) -> None:
        row = {
            "positive_year_ratio": 2.0 / 3.0,
            "total_trades": 10,
            "worst_mdd": -0.40,
            "top_trade_gross_profit_share": 0.50,
        }
        rules = {
            "min_positive_year_ratio": 0.6666666667,
            "min_total_trades": 10,
            "max_abs_mdd": 0.55,
            "max_top_trade_gross_profit_share": 0.60,
        }
        self.assertTrue(is_eligible(row, rules))

    def test_inactive_held_symbol_remains_a_ranked_candidate(self) -> None:
        cache = ExperimentalSignalCache.__new__(ExperimentalSignalCache)
        cache.active_by_position = [{"ACTIVE"}]
        cache.symbol_arrays = {
            symbol: {
                "score": np.array([score]),
                "atr_pct": np.array([0.02]),
                "price": np.array([100.0]),
            }
            for symbol, score in (("HELD", 2.0), ("ACTIVE", 1.0))
        }
        cache._rankings = {
            ("unlimited", None): [("HELD", "ACTIVE")]
        }
        policy = ExperimentalLeaderPolicy(late_chase_mode="unlimited")

        without_holding = cache.candidates_at(0, policy, set(), set())
        with_holding = cache.candidates_at(0, policy, set(), {"HELD"})

        self.assertEqual(
            [row["symbol"] for row in without_holding],
            ["ACTIVE"],
        )
        self.assertEqual(
            [row["symbol"] for row in with_holding],
            ["HELD", "ACTIVE"],
        )

    def test_kijun_atr_cap_allows_pullback_and_rejects_extension(self) -> None:
        near = pd.Series(
            {
                "Trend": 1,
                "BuySignal": False,
                "Close": 105.0,
                "Ichimoku_Kijun": 100.0,
                "ATR": 4.0,
            }
        )
        far = near.copy()
        far["Close"] = 115.0
        self.assertTrue(
            late_chase_allows_entry(
                near,
                mode="kijun_atr_capped",
                max_extension_atr=1.5,
            )
        )
        self.assertFalse(
            late_chase_allows_entry(
                far,
                mode="kijun_atr_capped",
                max_extension_atr=1.5,
            )
        )

    def test_fresh_signal_bypasses_extension_cap(self) -> None:
        row = pd.Series(
            {
                "Trend": 1,
                "BuySignal": True,
                "Close": 150.0,
                "Ichimoku_Kijun": 100.0,
                "ATR": 2.0,
            }
        )
        self.assertTrue(
            late_chase_allows_entry(
                row,
                mode="kijun_atr_capped",
                max_extension_atr=1.5,
            )
        )


if __name__ == "__main__":
    unittest.main()
