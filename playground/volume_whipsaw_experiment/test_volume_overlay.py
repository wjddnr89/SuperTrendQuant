from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "unified_quant" / "src"))
sys.path.insert(0, str(ROOT))

from volume_overlay import (  # noqa: E402
    CMF_COLUMN,
    OBV_SLOPE_COLUMN,
    RVOL_COLUMN,
    VolumeFilterSpec,
    add_volume_features,
    add_sticky_confirmation,
)


class VolumeFeatureTests(unittest.TestCase):
    def make_frame(self, count: int = 40) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=count, freq="D")
        close = pd.Series(np.arange(100.0, 100.0 + count), index=index)
        return pd.DataFrame(
            {
                "High": close + 2.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": np.arange(100.0, 100.0 + count),
                "IdentitySegment": "A",
            },
            index=index,
        )

    def test_rvol_uses_only_prior_twenty_sessions(self):
        frame = self.make_frame()
        result = add_volume_features(frame)
        expected = frame["Volume"].iloc[20] / frame["Volume"].iloc[:20].mean()
        self.assertAlmostEqual(float(result[RVOL_COLUMN].iloc[20]), float(expected))
        self.assertTrue(result[RVOL_COLUMN].iloc[:20].isna().all())

    def test_future_mutation_does_not_change_past_features(self):
        frame = self.make_frame()
        original = add_volume_features(frame)
        changed = frame.copy()
        changed.iloc[31:, changed.columns.get_loc("Volume")] *= 1000.0
        mutated = add_volume_features(changed)
        pd.testing.assert_frame_equal(
            original.loc[: original.index[30], [RVOL_COLUMN, CMF_COLUMN, OBV_SLOPE_COLUMN]],
            mutated.loc[: mutated.index[30], [RVOL_COLUMN, CMF_COLUMN, OBV_SLOPE_COLUMN]],
        )

    def test_identity_segment_resets_warmup(self):
        frame = self.make_frame()
        frame.loc[frame.index[25]:, "IdentitySegment"] = "B"
        result = add_volume_features(frame)
        self.assertTrue(result.loc[frame.index[25]:, RVOL_COLUMN].iloc[:20].isna().all())

    def test_filter_rejects_nan_and_applies_all_enabled_conditions(self):
        spec = VolumeFilterSpec("combo", rvol_min=1.2, cmf_min=0.0)
        self.assertFalse(spec.allows(pd.Series({RVOL_COLUMN: np.nan, CMF_COLUMN: 0.1})))
        self.assertFalse(spec.allows(pd.Series({RVOL_COLUMN: 1.3, CMF_COLUMN: -0.1})))
        self.assertTrue(spec.allows(pd.Series({RVOL_COLUMN: 1.3, CMF_COLUMN: 0.1})))

    def test_sticky_confirmation_latches_for_the_supertrend_up_leg(self):
        frame = self.make_frame(12)
        frame["Trend"] = [-1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1]
        frame[RVOL_COLUMN] = [0.0, 0.0, 0.8, 1.3, 0.7, 0.6, 0.5, 0.0, 0.0, 1.4, 0.5, 0.4]
        spec = VolumeFilterSpec("sticky", rvol_min=1.2, confirmation_bars=3)
        result = add_sticky_confirmation(frame, spec)
        self.assertEqual(
            result[spec.confirmation_column].tolist(),
            [False, False, False, True, True, True, True, False, False, True, True, True],
        )

    def test_sticky_confirmation_rejects_late_volume_and_resets(self):
        frame = self.make_frame(12)
        frame["Trend"] = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
        frame[RVOL_COLUMN] = [0.0, 0.8, 0.9, 1.0, 1.5, 1.5, 0.0, 1.3, 0.5, 0.5, 0.5, 0.5]
        spec = VolumeFilterSpec("sticky", rvol_min=1.2, confirmation_bars=3)
        result = add_sticky_confirmation(frame, spec)
        self.assertFalse(result[spec.confirmation_column].iloc[1:6].any())
        self.assertTrue(result[spec.confirmation_column].iloc[7:].all())

    def test_sticky_confirmation_is_causal(self):
        frame = self.make_frame(15)
        frame["Trend"] = [-1, -1] + [1] * 13
        frame[RVOL_COLUMN] = [0.0, 0.0, 0.8, 1.3] + [0.5] * 11
        spec = VolumeFilterSpec("sticky", rvol_min=1.2, confirmation_bars=3)
        original = add_sticky_confirmation(frame, spec)
        changed = frame.copy()
        changed.loc[changed.index[10]:, RVOL_COLUMN] = 100.0
        mutated = add_sticky_confirmation(changed, spec)
        pd.testing.assert_series_equal(
            original.loc[: original.index[9], spec.confirmation_column],
            mutated.loc[: mutated.index[9], spec.confirmation_column],
        )


if __name__ == "__main__":
    unittest.main()
