from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from supertrend_quant.config import ScoringConfig, load_split_config
from supertrend_quant.portfolio import AccountSnapshot, Position
from supertrend_quant.ranking import (
    DualMomentumScorer,
    RelativeStrengthScorer,
    available_scorers,
    create_scorer,
    effective_relative_strength_lookback,
    rank_scores,
    register_scorer,
)
from supertrend_quant.strategies import create_strategy


def score_frame(score: float, *, buy: bool = True, trend: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Close": [100.0, 100.0],
            "Trend": [1, trend],
            "ATR_pct": [0.01, 0.01],
            "BuySignal": [False, buy],
            "Score": [float("nan"), score],
        }
    )


class RankingTest(unittest.TestCase):
    def test_dual_momentum_requires_positive_absolute_and_excess_return(self):
        index = pd.date_range("2026-01-01", periods=3, freq="D")
        benchmark = pd.DataFrame({"Close": [100.0, 102.0, 104.04]}, index=index)
        frames = {
            "LEADER": pd.DataFrame({"Close": [100.0, 110.0, 121.0]}, index=index),
            "LAGGER": pd.DataFrame({"Close": [100.0, 101.0, 102.01]}, index=index),
            "LOSER": pd.DataFrame({"Close": [100.0, 90.0, 81.0]}, index=index),
        }
        scorer = DualMomentumScorer({"lookback_bars": 1}, "US")

        scored = scorer.add_scores(
            frames,
            {symbol: benchmark for symbol in frames},
        )

        self.assertAlmostEqual(scored["LEADER"]["Score"].iloc[-1], 0.08)
        self.assertTrue(scored["LAGGER"]["Score"].isna().all())
        self.assertTrue(scored["LOSER"]["Score"].isna().all())

    def test_dual_momentum_resets_at_identity_segment_boundary(self):
        index = pd.date_range("2026-01-01", periods=4, freq="D")
        asset = pd.DataFrame(
            {
                "Close": [100.0, 110.0, 50.0, 60.0],
                "IdentitySegment": ["old", "old", "new", "new"],
            },
            index=index,
        )
        benchmark = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0]}, index=index)

        scored = DualMomentumScorer({"lookback_bars": 1}, "US").add_scores(
            {"AAA": asset},
            {"AAA": benchmark},
        )["AAA"]

        self.assertTrue(pd.isna(scored["Score"].iloc[2]))
        self.assertGreater(scored["Score"].iloc[3], 0.0)

    def test_relative_strength_adds_causal_score_without_mutating_input(self):
        index = pd.date_range("2026-01-05 09:30", periods=4, freq="30min")
        asset = pd.DataFrame({"Close": [100.0, 110.0, 121.0, 133.1]}, index=index)
        benchmark = pd.DataFrame({"Close": [100.0, 105.0, 110.25, 115.7625]}, index=index)
        original = asset.copy(deep=True)
        scorer = RelativeStrengthScorer({"lookback_bars": 1}, "US")

        scored = scorer.add_scores({"AAA": asset}, {"AAA": benchmark})["AAA"]

        pd.testing.assert_frame_equal(asset, original)
        self.assertTrue(np.isnan(scored["Score"].iloc[0]))
        self.assertTrue(np.allclose(scored["Score"].iloc[1:], 0.05))

    def test_missing_benchmark_preserves_frames_with_nan_score(self):
        frame = pd.DataFrame({"Close": [100.0, 101.0]})
        scorer = RelativeStrengthScorer({"lookback_bars": 1}, "US")

        scored = scorer.add_scores({"AAA": frame}, None)

        self.assertIn("AAA", scored)
        self.assertTrue(scored["AAA"]["Score"].isna().all())

    def test_market_lookback_and_warmup_are_resolved_by_scorer(self):
        params = {"lookback_bars": {"default": 20, "KR": 30}}

        self.assertEqual(effective_relative_strength_lookback(params, "US"), 20)
        self.assertEqual(effective_relative_strength_lookback(params, "KR"), 30)
        scorer = create_scorer(ScoringConfig("relative_strength", params), "KR")
        self.assertEqual(scorer.warmup_bars(), 31)

    def test_rank_scores_is_deterministic_and_excludes_non_finite_values(self):
        ranked = rank_scores(
            {
                "ZZZ": 0.5,
                "AAA": 0.5,
                "LOW": -0.1,
                "NAN": float("nan"),
                "INF": float("inf"),
                "NONE": None,
            }
        )

        self.assertEqual(ranked, ("AAA", "ZZZ", "LOW"))

    def test_registry_rejects_duplicate_and_lists_relative_strength(self):
        class DuplicateRelativeStrength:
            scoring_type = "relative_strength"

        self.assertIn("relative_strength", available_scorers())
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_scorer(DuplicateRelativeStrength)

    def test_relative_strength_rejects_unknown_or_invalid_params(self):
        with self.assertRaisesRegex(ValueError, "Unsupported params"):
            create_scorer(
                ScoringConfig("relative_strength", {"lookback_bars": 10, "typo": 1}),
                "US",
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            create_scorer(ScoringConfig("relative_strength", {"lookback_bars": 0}), "US")
        with self.assertRaisesRegex(ValueError, "Available scorers"):
            create_scorer(ScoringConfig("missing", {"lookback_bars": 10}), "US")

    def test_simple_supertrend_fills_open_slots_in_score_order(self):
        config = load_split_config(
            "configs/strategies/simple_supertrend.yaml",
            "configs/runtimes/research_sp500.yaml",
        )
        config = replace(
            config,
            risk=replace(config.risk, max_position_count=2),
            market_trend_filter=replace(config.market_trend_filter, enabled=False),
        )
        strategy = create_strategy(config)

        plan = strategy._build_order_plan_from_prepared(
            {
                "LOW": score_frame(0.1),
                "HIGH": score_frame(0.3),
                "MID": score_frame(0.2),
            },
            AccountSnapshot(cash=10_000.0),
            "backtest",
        )

        self.assertEqual([order.symbol for order in plan.orders], ["HIGH", "MID"])
        self.assertTrue(all(order.reason == "Top-ranked Supertrend entry" for order in plan.orders))

    def test_leader_exit_still_runs_without_a_valid_score(self):
        config = load_split_config(
            "configs/strategies/leader_rotation.yaml",
            "configs/runtimes/research_sp500.yaml",
        )
        config = replace(
            config,
            market_trend_filter=replace(config.market_trend_filter, enabled=False),
        )
        strategy = create_strategy(config)
        prepared = {"HELD": score_frame(float("nan"), buy=False, trend=-1)}

        plan = strategy._build_order_plan_from_prepared(
            prepared,
            AccountSnapshot(
                cash=0.0,
                positions={"HELD": Position("HELD", quantity=10, avg_price=100.0)},
            ),
            "backtest",
        )

        self.assertEqual([(order.symbol, order.side) for order in plan.orders], [("HELD", "sell")])


if __name__ == "__main__":
    unittest.main()
