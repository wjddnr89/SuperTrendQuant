from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_SRC = REPOSITORY_ROOT / "playground" / "src"


def _run_playground_python(source: str):
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(PLAYGROUND_SRC) + (
        os.pathsep + existing if existing else ""
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


class PlaygroundDataSourceTest(unittest.TestCase):
    def test_parsers_and_source_mapping_preserve_period_and_start(self):
        result = _run_playground_python(
            """
            import contextlib
            import io
            import json
            import sys
            import tempfile
            from pathlib import Path
            from unittest.mock import patch

            sys.path.insert(0, str(Path('playground/scripts').resolve()))
            import evaluate_best_nasdaq100_daily_3y_split as best
            import evaluate_nasdaq100_combo_once as combo
            import optimize_nasdaq100_daily_optuna as optuna
            import search_nasdaq100_daily_3y_grid as grid

            modules = (best, combo, grid, optuna)
            expected = {
                best: ('3y', None),
                combo: ('max', '2010-01-01'),
                grid: ('max', '2010-01-01'),
                optuna: ('max', '2010-01-01'),
            }
            output = {}
            for module in modules:
                defaults = module.build_parser().parse_args([])
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        module.build_parser().parse_args(['--data-source', 'invalid'])
                except SystemExit:
                    rejected_invalid = True
                else:
                    rejected_invalid = False
                output[module.__name__] = {
                    'source': defaults.data_source,
                    'period': defaults.period,
                    'start': getattr(defaults, 'start', None),
                    'rejected_invalid': rejected_invalid,
                }

                class StopLoading(Exception):
                    pass

                argv = [str(module.__file__), '--data-source', 'yahoo', '--period', '2y']
                temporary = None
                if module is optuna:
                    temporary = tempfile.TemporaryDirectory()
                    argv.extend(['--results-dir', temporary.name, '--run-id', 'source-test'])
                with (
                    patch.object(module, 'load_experiment_market_data', side_effect=StopLoading) as loader,
                    patch.object(sys, 'argv', argv),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    try:
                        module.main()
                    except StopLoading:
                        pass
                output[module.__name__]['mapped_source'] = loader.call_args.kwargs['data_source']
                output[module.__name__]['mapped_period'] = loader.call_args.args[0].period
                if temporary is not None:
                    temporary.cleanup()

            print(json.dumps(output))
            """
        )

        expected = {
            "evaluate_best_nasdaq100_daily_3y_split": ("3y", None),
            "evaluate_nasdaq100_combo_once": ("max", "2010-01-01"),
            "search_nasdaq100_daily_3y_grid": ("max", "2010-01-01"),
            "optimize_nasdaq100_daily_optuna": ("max", "2010-01-01"),
        }
        for name, (period, start) in expected.items():
            with self.subTest(script=name):
                self.assertEqual(result[name]["source"], "local")
                self.assertEqual(result[name]["period"], period)
                self.assertEqual(result[name]["start"], start)
                self.assertTrue(result[name]["rejected_invalid"])
                self.assertEqual(result[name]["mapped_source"], "yahoo")
                self.assertEqual(result[name]["mapped_period"], "2y")

    def test_local_adapter_uses_parquet_index_events_and_allows_stale(self):
        result = _run_playground_python(
            """
            import json
            import sys
            from pathlib import Path
            from unittest.mock import patch

            sys.path.insert(0, str(Path('playground/scripts').resolve()))
            import evaluate_best_nasdaq100_daily_3y_split as best
            import market_data_source

            args = best.build_parser().parse_args(['--period', '2y'])
            config = best.best_config(args)
            local = market_data_source._local_config(
                config,
                strategy_path=args.strategy,
                runtime_path=args.runtime,
            )
            _, resolver = market_data_source._unified_modules()
            with patch.object(resolver, 'download_for_config') as loader:
                loader.return_value = type(
                    'Loaded',
                    (),
                    {
                        'bars': {},
                        'benchmark': None,
                        'filter_benchmark': None,
                        'skipped': (),
                        'universe_snapshot': None,
                        'universe_schedule': (),
                    },
                )()
                market_data_source.load_experiment_market_data(
                    config,
                    data_source='local',
                    strategy_path=args.strategy,
                    runtime_path=args.runtime,
                )
            with patch.object(market_data_source, 'download_for_config') as yahoo_loader:
                yahoo_loader.return_value = market_data_source.MarketData(bars={})
                market_data_source.load_experiment_market_data(
                    config,
                    data_source='yahoo',
                    strategy_path=args.strategy,
                    runtime_path=args.runtime,
                )
            print(json.dumps({
                'provider': local.data_store.provider,
                'universe_source': local.universe.source,
                'profiles': local.universe.profiles,
                'period': local.period,
                'allow_stale': loader.call_args.kwargs['allow_stale'],
                'yahoo_calls': yahoo_loader.call_count,
            }))
            """
        )

        self.assertEqual(result["provider"], "parquet")
        self.assertEqual(result["universe_source"], "index_events")
        self.assertEqual(result["profiles"], {"US": ["nasdaq100"]})
        self.assertEqual(result["period"], "2y")
        self.assertTrue(result["allow_stale"])
        self.assertEqual(result["yahoo_calls"], 1)

    def test_optuna_cache_rejects_other_or_unlabeled_sources(self):
        result = _run_playground_python(
            """
            import json
            import sys
            import tempfile
            from pathlib import Path
            from types import SimpleNamespace

            import pandas as pd

            sys.path.insert(0, str(Path('playground/scripts').resolve()))
            import optimize_nasdaq100_daily_optuna as optuna
            import search_nasdaq100_daily_3y_grid as grid

            params = {
                'entry': 'single', 'market_filter': 'none', 'asset_filter': 'none',
                'rs_method': 'relative_strength', 'rs_period': 100,
                'sell_confirm_bars': 1, 'hurdle': 1.25, 'max_positions': 1,
                'st_period': 10, 'st_multiplier': 3.0,
            }
            base = {
                'params': params,
                'metrics': {
                    'total_return': 0.1, 'mdd': -0.1, 'cagr': 0.1,
                    'calmar': 1.0, 'sharpe': 1.0, 'sortino': 1.0,
                    'win_rate': 0.5, 'payoff_ratio': 1.0, 'trade_count': 1,
                },
                'score': 1.0, 'qqq_return': 0.05, 'alpha': 0.05,
                'start': '2024-01-01', 'end': '2024-12-31',
            }
            rows = [
                grid.flat_row({**base, 'data_source': 'local'}),
                grid.flat_row({**base, 'data_source': 'yahoo'}),
                grid.flat_row(base),
            ]
            runner = object.__new__(optuna.OptunaBacktestRunner)
            runner.args = SimpleNamespace(data_source='local')
            runner.cache = {}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'all_results.csv'
                pd.DataFrame(rows).to_csv(path, index=False)
                count = runner.load_cache_files([path], source='imported')
            print(json.dumps({
                'count': count,
                'cache_size': len(runner.cache),
                'source': next(iter(runner.cache.values()))['data_source'],
            }))
            """
        )

        self.assertEqual(result, {"count": 1, "cache_size": 1, "source": "local"})


if __name__ == "__main__":
    unittest.main()
