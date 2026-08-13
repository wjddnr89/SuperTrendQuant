from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_SRC = PROJECT_ROOT / "unified_quant" / "src"
PLAYGROUND_ROOT = PROJECT_ROOT / "playground"
sys.path.insert(0, str(UNIFIED_SRC))
sys.path.insert(0, str(PLAYGROUND_ROOT))

from scripts.rolling_expanding_nasdaq_plateau import (  # noqa: E402
    DEFAULT_CONFIG,
    build_candidate_meta,
    load_json,
    neighbor_ids,
    plateau_sort_key,
    train_years_for,
)


class RollingExpandingPlateauTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_json(DEFAULT_CONFIG)
        cls.meta = build_candidate_meta(cls.raw)

    def test_three_core_families_resolve_to_twenty_one_candidates(self) -> None:
        self.assertEqual(len(self.meta), 21)
        self.assertEqual(sum(item.is_core for item in self.meta), 3)
        self.assertEqual(
            {item.family for item in self.meta if item.is_core},
            {
                "legacy_gate0_unlimited",
                "gateoff_unlimited",
                "gate0_capped",
            },
        )

    def test_rolling_and_expanding_years_do_not_include_outer_year(self) -> None:
        validation = self.raw["validation"]
        self.assertEqual(
            train_years_for("rolling_5y", 2026, validation),
            [2021, 2022, 2023, 2024, 2025],
        )
        self.assertEqual(
            train_years_for("expanding", 2026, validation),
            list(range(2016, 2026)),
        )

    def test_capped_core_has_immediate_stop_and_cap_neighbors(self) -> None:
        core = next(
            item
            for item in self.meta
            if item.family == "gate0_capped" and item.is_core
        )
        neighbors = neighbor_ids(core, self.meta, 1)
        self.assertEqual(len(neighbors), 5)
        self.assertIn(core.candidate.candidate_id, neighbors)

    def test_boundary_penalty_prefers_full_neighborhood(self) -> None:
        common = {
            "plateau_eligible_ratio": 1.0,
            "plateau_min_positive_year_ratio": 1.0,
            "plateau_median_calmar": 1.0,
            "plateau_worst_year_return": 0.1,
            "plateau_worst_mdd": -0.3,
            "plateau_median_compound_return": 1.0,
            "median_calmar": 1.0,
            "compound_return": 1.0,
        }
        boundary = {**common, "plateau_neighbor_coverage": 2 / 3}
        interior = {**common, "plateau_neighbor_coverage": 1.0}
        self.assertGreater(
            plateau_sort_key(interior, penalize_boundary=True),
            plateau_sort_key(boundary, penalize_boundary=True),
        )

    def test_nonselectable_benchmarks_are_marked(self) -> None:
        boundary_config = load_json(
            PLAYGROUND_ROOT
            / "configs"
            / "rolling_expanding_nasdaq_stop_boundary.json"
        )
        boundary_meta = build_candidate_meta(boundary_config)
        self.assertEqual(sum(item.selectable for item in boundary_meta), 6)
        self.assertEqual(sum(not item.selectable for item in boundary_meta), 2)

    def test_kospi_transfer_has_one_locked_candidate(self) -> None:
        transfer_config = load_json(
            PLAYGROUND_ROOT
            / "configs"
            / "rolling_expanding_kospi200_transfer.json"
        )
        transfer_meta = build_candidate_meta(transfer_config)
        selectable = [item for item in transfer_meta if item.selectable]
        self.assertEqual(len(selectable), 1)
        self.assertEqual(selectable[0].candidate.policy.stop_loss_pct, 0.12)
        self.assertEqual(
            selectable[0].candidate.policy.rotation_profit_gate,
            "off",
        )

    def test_kospi_rs_block_changes_only_rs_period(self) -> None:
        configs = [
            load_json(
                PLAYGROUND_ROOT
                / "configs"
                / f"rolling_expanding_kospi200_rs{period}.json"
            )
            for period in (125, 175)
        ]
        self.assertEqual(
            [raw["base_combo"]["rs_period"] for raw in configs],
            [125, 175],
        )
        for raw in configs:
            meta = build_candidate_meta(raw)
            self.assertEqual(len(meta), 1)
            self.assertEqual(meta[0].candidate.policy.stop_loss_pct, 0.12)
            self.assertEqual(
                meta[0].candidate.policy.rotation_profit_gate,
                "off",
            )
            self.assertEqual(
                meta[0].candidate.policy.late_chase_mode,
                "unlimited",
            )


if __name__ == "__main__":
    unittest.main()
