from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from supertrend_quant.cli import data_main
from supertrend_quant.config import load_data_store_config, load_split_config


ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "configs" / "strategies" / "leader_rotation.yaml"


class SharedDataConfigTest(unittest.TestCase):
    def test_us_runtime_inherits_shared_data_config(self):
        shared = load_data_store_config(ROOT / "configs" / "data.yaml")
        config = load_split_config(
            STRATEGY,
            ROOT / "configs" / "runtimes" / "simulation.yaml",
        )

        self.assertEqual(config.data_store, shared)

    def test_kr_runtime_uses_shared_market_override(self):
        config = load_split_config(
            STRATEGY,
            ROOT / "configs" / "runtimes" / "research_kr.yaml",
        )

        self.assertEqual(config.data_store.provider, "yahoo")
        self.assertFalse(config.data_store.auto_sync)

    def test_quant_data_status_needs_only_data_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.yaml"
            data_path.write_text(
                "data_store:\n"
                "  provider: parquet\n"
                f"  local_cache_dir: {root / 'cache'}\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("sys.argv", ["quant-data", "--data-config", str(data_path), "status"]):
                with redirect_stdout(output):
                    data_main()

        status = json.loads(output.getvalue())
        self.assertEqual(status[0]["dataset"], "__release__")
        self.assertEqual(status[0]["status"], "missing")

    def test_direct_publish_routes_fail_closed_before_cache_or_r2_access(self):
        for command in ("sync", "bootstrap-us"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cache = root / "cache"
                data_path = root / "data.yaml"
                data_path.write_text(
                    "data_store:\n"
                    "  provider: parquet\n"
                    f"  local_cache_dir: {cache}\n",
                    encoding="utf-8",
                )
                errors = io.StringIO()
                with (
                    patch(
                        "sys.argv",
                        [
                            "quant-data",
                            "--data-config",
                            str(data_path),
                            command,
                            "--publish",
                        ],
                    ),
                    redirect_stderr(errors),
                    self.assertRaises(SystemExit) as raised,
                ):
                    data_main()

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("Direct R2 publication is disabled", errors.getvalue())
                self.assertIn("publish_and_verify_r2.py", errors.getvalue())
                self.assertFalse(cache.exists())

    def test_automatic_publish_config_also_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            data_path = root / "data.yaml"
            data_path.write_text(
                "data_store:\n"
                "  provider: parquet\n"
                "  publish_enabled: true\n"
                f"  local_cache_dir: {cache}\n",
                encoding="utf-8",
            )
            errors = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    ["quant-data", "--data-config", str(data_path), "sync"],
                ),
                redirect_stderr(errors),
                self.assertRaises(SystemExit) as raised,
            ):
                data_main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("publish_enabled=true", errors.getvalue())
            self.assertFalse(cache.exists())


if __name__ == "__main__":
    unittest.main()
