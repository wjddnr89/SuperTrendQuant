from __future__ import annotations

import unittest
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_SRC = PROJECT_ROOT / "unified_quant" / "src"
PLAYGROUND_ROOT = PROJECT_ROOT / "playground"
sys.path.insert(0, str(UNIFIED_SRC))
sys.path.insert(0, str(PLAYGROUND_ROOT))

from supertrend_quant.config import ScoringConfig
from supertrend_quant.ranking import available_scorers, create_scorer

from research_extensions.momentum_scorer_adapter import (
    register_research_momentum_scorers,
    with_research_momentum_scoring,
)
from scripts.nested_walk_forward_nasdaq_structure import (
    base_config,
    load_json,
)


class MomentumScorerAdapterTests(unittest.TestCase):
    def test_registers_four_research_scorers_idempotently(self) -> None:
        expected = {
            "vol_adjusted_relative_strength",
            "composite_relative_strength",
            "skip_recent_relative_strength",
            "beta_adjusted_alpha",
        }
        self.assertEqual(
            set(register_research_momentum_scorers()),
            expected,
        )
        self.assertTrue(expected.issubset(set(available_scorers())))
        self.assertEqual(
            set(register_research_momentum_scorers()),
            expected,
        )

    def test_all_research_scorers_produce_causal_scores(self) -> None:
        index = pd.date_range("2020-01-01", periods=80, freq="B")
        frames = {
            "A": pd.DataFrame(
                {"Close": [100.0 * (1.01**i) for i in range(len(index))]},
                index=index,
            )
        }
        benchmark = pd.DataFrame(
            {"Close": [100.0 * (1.003**i) for i in range(len(index))]},
            index=index,
        )
        configs = (
            ScoringConfig(
                "vol_adjusted_relative_strength",
                {"lookback_bars": 20},
            ),
            ScoringConfig(
                "composite_relative_strength",
                {"lookback_bars": 20},
            ),
            ScoringConfig(
                "skip_recent_relative_strength",
                {"lookback_bars": 20, "skip_bars": 5},
            ),
            ScoringConfig(
                "beta_adjusted_alpha",
                {"lookback_bars": 20},
            ),
        )
        for config in configs:
            with self.subTest(scoring_type=config.type):
                scorer = create_scorer(config, "KR")
                score = scorer.add_scores(frames, benchmark)["A"]["Score"]
                self.assertTrue(score.notna().any())
                self.assertTrue(pd.isna(score.iloc[0]))

    def test_transfer_config_preserves_dual_momentum(self) -> None:
        raw = load_json(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "rolling_expanding_kospi200_transfer.json"
        )
        config = base_config(raw)
        self.assertEqual(config.scoring.type, "dual_momentum")
        self.assertEqual(config.scoring.params["lookback_bars"], 150)

    def test_method_aliases_are_applied_without_canonical_changes(self) -> None:
        raw = load_json(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "rolling_expanding_kospi200_transfer.json"
        )
        config = base_config(raw)
        composite = with_research_momentum_scoring(
            config,
            method="composite",
            period=125,
        )
        self.assertEqual(
            composite.scoring.type,
            "composite_relative_strength",
        )
        self.assertEqual(
            composite.scoring.params,
            {"lookback_bars": 125},
        )
        skipped = with_research_momentum_scoring(
            config,
            method="skip_recent",
            period=150,
            skip_bars=21,
        )
        self.assertEqual(
            skipped.scoring.params,
            {"lookback_bars": 150, "skip_bars": 21},
        )


if __name__ == "__main__":
    unittest.main()
