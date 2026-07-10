import tempfile
import unittest
from pathlib import Path

from supertrend_quant.config import load_split_config
from supertrend_quant.research import save_split_yaml, strict_split_roundtrip


ROOT = Path(__file__).resolve().parents[1]


class ResearchExportTest(unittest.TestCase):
    def test_triple_research_config_promotes_through_strict_parser(self):
        config = load_split_config(
            ROOT / "configs/strategies/triple_filters.yaml",
            ROOT / "configs/runtimes/research_us.yaml",
        )

        promoted = strict_split_roundtrip(config)

        self.assertEqual(promoted.strategy.type, "leader_rotation")
        self.assertEqual(promoted.timeframe, config.timeframe)
        self.assertEqual(promoted.costs, config.costs)
        self.assertIn("triple_supertrend", {item.type for item in promoted.components})
        self.assertIn("ichimoku_cloud", {item.type for item in promoted.components})

    def test_saved_best_pair_reloads_with_public_loader(self):
        config = load_split_config(
            ROOT / "configs/strategies/leader_rotation.yaml",
            ROOT / "configs/runtimes/research_kr.yaml",
        )
        with tempfile.TemporaryDirectory() as tmp:
            strategy_path, runtime_path = save_split_yaml(config, tmp)
            restored = load_split_config(strategy_path, runtime_path)

        self.assertEqual(restored, config)


if __name__ == "__main__":
    unittest.main()
