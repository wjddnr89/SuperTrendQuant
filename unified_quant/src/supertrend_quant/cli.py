from __future__ import annotations

import argparse
import asyncio
import json

from .config import AppConfig, load_config, load_split_config
from .live_runtime import HybridLiveRuntime
from .paper_runtime import PaperRuntime
from .results import PaperRunRecorder, compare_paper_to_backtest, latest_run_dir, save_backtest_result
from .runners import print_backtest_result, run_backtest


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Legacy single-file config path.")
    parser.add_argument("--strategy", default=None, help="Split strategy definition path.")
    parser.add_argument("--runtime", default=None, help="Split runtime definition path.")
    parser.add_argument("--market", choices=["US", "KR", "AUTO"], default=None, help="Override runtime market.")
    parser.add_argument("--universe-file", default=None, help="Override runtime universe file.")
    parser.add_argument("--symbols", default=None, help="Override symbols as a comma-separated list.")


def _load_cli_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> AppConfig:
    split_paths = [args.strategy, args.runtime]
    if args.config:
        if any(split_paths):
            parser.error("--config cannot be combined with --strategy or --runtime.")
        return _apply_config_overrides(load_config(args.config), args)

    missing = [
        name
        for name, value in (
            ("--strategy", args.strategy),
            ("--runtime", args.runtime),
        )
        if not value
    ]
    if missing:
        parser.error("Provide either --config or all split config paths: " + ", ".join(missing))
    return _apply_config_overrides(load_split_config(args.strategy, args.runtime), args)


def _apply_config_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    updates = {}
    if args.market:
        updates["market"] = args.market
    if args.universe_file:
        updates["universe_file"] = args.universe_file
    if args.symbols:
        updates["symbols"] = tuple(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip())
    if not updates:
        return config
    return config.__class__(**{**config.__dict__, **updates})


def backtest_main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured backtest.")
    _add_config_args(parser)
    parser.add_argument("--period", default=None)
    parser.add_argument("--results-dir", default=None, help="Directory where backtest run results are saved.")
    parser.add_argument("--run-id", default=None, help="Optional stable run id for saved results.")
    parser.add_argument("--no-save", action="store_true", help="Do not save backtest summary/equity files.")
    args = parser.parse_args()

    config = _load_cli_config(args, parser)
    if args.period:
        config = config.__class__(**{**config.__dict__, "period": args.period})
    result = run_backtest(config)
    print_backtest_result(result)
    if not args.no_save:
        run_dir = save_backtest_result(result, config, args.results_dir or config.backtest.results_dir, args.run_id)
        print(f"Saved       : {run_dir}")


def paper_main() -> None:
    parser = argparse.ArgumentParser(description="Run configured paper trading.")
    _add_config_args(parser)
    parser.add_argument("--state", default=None, help="Paper account state file. Defaults to runtime paper.state_file.")
    parser.add_argument("--results-dir", default=None, help="Directory where paper run logs are saved.")
    parser.add_argument("--run-id", default=None, help="Optional stable run id for saved results.")
    parser.add_argument("--once", action="store_true", help="Run one paper cycle.")
    parser.add_argument("--ignore-schedule", action="store_true", help="Run even when the configured market is closed.")
    args = parser.parse_args()

    config = _load_cli_config(args, parser)
    if config.execution.broker != "paper":
        parser.error("quant-paper requires runtime execution.broker: paper.")
    recorder = PaperRunRecorder(args.results_dir or config.paper.results_dir, config.strategy.name, run_id=args.run_id)
    runtime = PaperRuntime(config, state_path=args.state, recorder=recorder)
    if not args.once:
        asyncio.run(runtime.run_loop(ignore_schedule=args.ignore_schedule))
        return

    plan, fills = runtime.run_once(ignore_schedule=args.ignore_schedule)
    print(f"Paper Order Plan: {len(plan.orders)} orders")
    for note in plan.notes:
        print(note)
    for order in plan.orders:
        print(f"{order.side.upper():4} {order.symbol:8} qty={order.quantity:g} reason={order.reason}")
    for fill in fills:
        print(fill)
    print(f"Saved       : {runtime.recorder.run_dir}")


def live_main() -> None:
    parser = argparse.ArgumentParser(description="Run the migrated main_jo-style live runtime.")
    _add_config_args(parser)
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    parser.add_argument("--once", action="store_true", help="Run one live cycle.")
    parser.add_argument("--ignore-schedule", action="store_true", help="Run one cycle even when the market is closed.")
    args = parser.parse_args()

    config = _load_cli_config(args, parser)
    if config.execution.broker != "toss":
        parser.error("quant-live requires runtime execution.broker: toss.")
    runtime = HybridLiveRuntime(config)
    if not args.once:
        asyncio.run(runtime.run_loop())
        return

    plan, results = runtime.run_once(ignore_schedule=args.ignore_schedule, assume_yes=args.yes)
    print(f"Live Order Plan: {len(plan.orders)} orders")
    for note in plan.notes:
        print(note)
    for result in results:
        print(result)


def compare_main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved paper results against a saved backtest.")
    parser.add_argument("--paper-dir", default=None, help="Paper run directory. Defaults to latest under --paper-root.")
    parser.add_argument("--backtest-dir", default=None, help="Backtest run directory. Defaults to latest under --backtest-root.")
    parser.add_argument("--paper-root", default="results/paper")
    parser.add_argument("--backtest-root", default="results/backtests")
    parser.add_argument("--interval", default="30m")
    args = parser.parse_args()

    paper_dir = args.paper_dir or latest_run_dir(args.paper_root)
    backtest_dir = args.backtest_dir or latest_run_dir(args.backtest_root)
    comparison = compare_paper_to_backtest(paper_dir, backtest_dir, args.interval)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


def search_main() -> None:
    """Run Cartesian research over the canonical production configuration."""
    from .research import MarketDataCache
    from .research.cli import (
        add_search_arguments,
        emit_best_config,
        print_search_result,
        search_from_namespace,
    )

    parser = argparse.ArgumentParser(description="Search strategy combinations with the canonical backtest engine.")
    _add_config_args(parser)
    parser.add_argument("--period", default=None)
    add_search_arguments(parser)
    args = parser.parse_args()

    config = _load_cli_config(args, parser)
    if args.period:
        config = config.__class__(**{**config.__dict__, "period": args.period})
    result = search_from_namespace(config, MarketDataCache(), args)
    print_search_result(result, top=args.top)
    emit_best_config(result.best_config, args)


def optimize_main() -> None:
    """Optimize a production configuration with validation-only Optuna trials."""
    from .research import MarketDataCache
    from .research.cli import (
        add_optimize_arguments,
        emit_best_config,
        optimize_from_namespace,
        print_optimization_result,
    )

    parser = argparse.ArgumentParser(description="Optimize a strategy with the canonical backtest engine.")
    _add_config_args(parser)
    parser.add_argument("--period", default=None)
    add_optimize_arguments(parser)
    args = parser.parse_args()

    config = _load_cli_config(args, parser)
    if args.period:
        config = config.__class__(**{**config.__dict__, "period": args.period})
    result = optimize_from_namespace(config, MarketDataCache(), args)
    print_optimization_result(result)
    emit_best_config(result.best_config, args)
