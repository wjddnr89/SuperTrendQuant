from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_ROOT = PROJECT_ROOT / "playground"
sys.path.insert(0, str(PLAYGROUND_ROOT))

from scripts.compare_kospi200_momentum_methods import (  # noqa: E402
    load_json,
    method_study_config,
)
from scripts.rolling_expanding_nasdaq_plateau import (  # noqa: E402
    build_candidate_meta,
)
from scripts.nested_walk_forward_nasdaq_structure import (  # noqa: E402
    base_config,
)


class KospiMomentumComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_json(
            PLAYGROUND_ROOT
            / "configs"
            / "rolling_expanding_kospi200_momentum_comparison.json"
        )

    def test_four_methods_change_only_scoring_method(self) -> None:
        methods = list(self.raw["momentum_methods"])
        self.assertEqual(
            methods,
            [
                "relative_strength",
                "dual_momentum",
                "composite",
                "vol_adjusted",
            ],
        )
        baseline = None
        scoring_types = []
        for method in methods:
            generated = method_study_config(self.raw, method)
            meta = build_candidate_meta(generated)
            self.assertEqual(len(meta), 1)
            config = base_config(generated)
            scoring_types.append(config.scoring.type)
            comparable = {
                **generated,
                "name": "<method>",
                "base_combo": {
                    **generated["base_combo"],
                    "rs_method": "<method>",
                },
            }
            if baseline is None:
                baseline = comparable
            self.assertEqual(comparable, baseline)
            self.assertEqual(
                config.scoring.params["lookback_bars"],
                150,
            )
        self.assertEqual(
            scoring_types,
            [
                "relative_strength",
                "dual_momentum",
                "composite_relative_strength",
                "vol_adjusted_relative_strength",
            ],
        )

    def test_comparison_locks_requested_risk_controls(self) -> None:
        generated = method_study_config(
            self.raw,
            "dual_momentum",
        )
        meta = build_candidate_meta(generated)
        policy = meta[0].candidate.policy
        self.assertEqual(policy.rotation_profit_gate, "off")
        self.assertEqual(policy.stop_loss_pct, 0.12)
        self.assertEqual(policy.late_chase_mode, "unlimited")
        self.assertEqual(generated["base_combo"]["max_positions"], 1)
        self.assertEqual(generated["costs"]["fee_rate"], 0.00225)
        self.assertEqual(generated["costs"]["slippage_rate"], 0.0005)


if __name__ == "__main__":
    unittest.main()
