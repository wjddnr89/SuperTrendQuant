from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from supertrend_quant.config import load_split_config
from supertrend_quant.indicators import add_ema_trend, add_ichimoku, add_triple_supertrend
from supertrend_quant.strategies import create_strategy
from supertrend_quant.strategies.common import with_strategy_components


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_ROOT = REPOSITORY_ROOT / "unified_quant"
TRIPLE_PATH = UNIFIED_ROOT / "configs/strategies/triple_filters.yaml"
RUNTIME_PATH = UNIFIED_ROOT / "configs/runtimes/research_us.yaml"


def synthetic_ohlc(periods: int = 240) -> pd.DataFrame:
    index = pd.date_range("2025-01-02 09:30", periods=periods, freq="30min")
    trend = np.linspace(100.0, 180.0, periods)
    close = trend + np.sin(np.arange(periods) / 4.0)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
        },
        index=index,
    )


class TripleConfigAndIndicatorAcceptanceTest(unittest.TestCase):
    def test_strict_triple_filters_config_loads_all_component_parameters(self):
        config = load_split_config(TRIPLE_PATH, RUNTIME_PATH)
        components = {(item.group, item.type): item for item in config.components}

        entry = components[("entries", "triple_supertrend")]
        self.assertEqual(
            entry.params["settings"],
            [
                {"period": 10, "multiplier": 1.0},
                {"period": 11, "multiplier": 2.0},
                {"period": 12, "multiplier": 3.0},
            ],
        )
        self.assertEqual(
            components[("filters", "ichimoku_cloud")].params,
            {"tenkan": 9, "kijun": 26, "span_b": 52, "shift": 26},
        )
        self.assertEqual(components[("filters", "ema_trend")].params["period"], 200)
        self.assertEqual(
            components[("exits", "triple_supertrend_flip")].params,
            {"down_count": 2, "confirm_bars": 3},
        )
        self.assertEqual(config.exit.sell_confirm_bars, 3)
        self.assertEqual(create_strategy(config).strategy_type, "leader_rotation")

    def test_triple_settings_reject_unknown_keys(self):
        malformed = TRIPLE_PATH.read_text(encoding="utf-8").replace(
            "multiplier: 1.0",
            "multipler: 1.0",
            1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_triple.yaml"
            path.write_text(malformed, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported keys"):
                load_split_config(path, RUNTIME_PATH)

    def test_triple_ichimoku_and_ema_smoke_without_input_mutation(self):
        source = synthetic_ohlc(100)
        original = source.copy(deep=True)
        featured = add_triple_supertrend(
            source,
            settings=((3, 1.0), (4, 1.5), (5, 2.0)),
            exit_down_count=2,
        )
        featured = add_ichimoku(featured, tenkan=3, kijun=5, span_b=8, shift=2)
        featured = add_ema_trend(featured, period=10)

        pd.testing.assert_frame_equal(source, original)
        self.assertTrue(
            {
                "TripleST1_Trend",
                "TripleST2_Trend",
                "TripleST3_Trend",
                "TripleAllUp",
                "TripleDownCount",
                "Ichimoku_LongOk",
                "Ichimoku_SpanA",
                "EMA",
                "EMA_LongOk",
            }.issubset(featured.columns)
        )
        self.assertEqual(featured.index.tolist(), source.index.tolist())
        self.assertFalse(featured["EMA"].isna().any())

    def test_configured_strategy_composes_all_research_features(self):
        config = load_split_config(TRIPLE_PATH, RUNTIME_PATH)
        featured = with_strategy_components(config, "AAA", synthetic_ohlc())
        expected = {
            "ATR_pct",
            "TripleAllUp",
            "TripleDownCount",
            "Ichimoku_LongOk",
            "EMA_LongOk",
        }
        self.assertTrue(expected.issubset(featured.columns))
        self.assertFalse(featured["EMA"].tail(1).isna().any())


if __name__ == "__main__":
    unittest.main()
