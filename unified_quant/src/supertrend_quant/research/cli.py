from __future__ import annotations

import argparse

from ..config import AppConfig
from ..data import MarketData
from ..metrics import format_float, format_pct
from .data_resolver import MarketDataSource
from .export import save_split_yaml, split_yaml_text
from .optimize import OptimizationResult, OptimizationSpace, run_optimize
from .search import SearchResult, run_search


def csv_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def add_search_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--timeframes", default=None, help="Comma-separated candidate timeframes.")
    parser.add_argument("--entries", default="single,triple")
    parser.add_argument("--market-filters", default=None)
    parser.add_argument(
        "--asset-filters",
        default="none,ichimoku_cloud,ema_trend,ichimoku_cloud+ema_trend",
    )
    parser.add_argument("--sell-confirm-bars", default="1,2,3,5,8,13")
    parser.add_argument("--rs-periods", default="50,100")
    parser.add_argument("--max-positions", default=None, help="Comma-separated leader position counts.")
    parser.add_argument("--st-periods", default="10")
    parser.add_argument("--st-multipliers", default="3.0")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--show-best-config",
        action="store_true",
        help="Print the winning strict strategy/runtime YAML documents.",
    )
    parser.add_argument(
        "--save-best-dir",
        default=None,
        help="Save the winning strategy.yaml and runtime.yaml under this directory.",
    )
    return parser


def add_optimize_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeframes", default=None, help="Comma-separated candidate timeframes.")
    parser.add_argument("--min-rs-period", type=int, default=10)
    parser.add_argument("--max-rs-period", type=int, default=200)
    parser.add_argument("--min-positions", type=int, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-sell-confirm-bars", type=int, default=30)
    parser.add_argument(
        "--show-best-config",
        action="store_true",
        help="Print the winning strict strategy/runtime YAML documents.",
    )
    parser.add_argument(
        "--save-best-dir",
        default=None,
        help="Save the winning strategy.yaml and runtime.yaml under this directory.",
    )
    return parser


def search_from_namespace(
    config: AppConfig,
    market_data: MarketDataSource,
    args: argparse.Namespace,
) -> SearchResult:
    if args.market_filters:
        market_filters = csv_values(args.market_filters)
    elif isinstance(market_data, MarketData):
        market_filters = (
            ("none", config.market_trend_filter.timeframe)
            if config.market_trend_filter.enabled
            else ("none",)
        )
    else:
        market_filters = ("none", "1d")
    grid = {
        "timeframe": csv_values(args.timeframes) if args.timeframes else (config.timeframe,),
        "entry": csv_values(args.entries),
        "market_filter": market_filters,
        "asset_filter": csv_values(args.asset_filters),
        "sell_confirm_bars": tuple(int(value) for value in csv_values(args.sell_confirm_bars)),
        "rs_period": tuple(int(value) for value in csv_values(args.rs_periods)),
    }
    if args.max_positions:
        grid["max_positions"] = tuple(int(value) for value in csv_values(args.max_positions))
    # ST knobs are meaningful only for single entries. Keeping a single/triple
    # grid generic would create inactive dimensions, so add them only when the
    # caller searches single entry exclusively.
    entries = set(grid["entry"])
    if entries <= {"single", "single_supertrend", "supertrend"}:
        grid["st_period"] = tuple(int(value) for value in csv_values(args.st_periods))
        grid["st_multiplier"] = tuple(
            float(value) for value in csv_values(args.st_multipliers)
        )
    return run_search(config, market_data, grid, limit=args.limit)


def optimize_from_namespace(
    config: AppConfig,
    market_data: MarketDataSource,
    args: argparse.Namespace,
) -> OptimizationResult:
    market_filters = (
        ("none", config.market_trend_filter.timeframe)
        if config.market_trend_filter.enabled or not isinstance(market_data, MarketData)
        else ("none",)
    )
    space = OptimizationSpace(
        timeframes=csv_values(args.timeframes) if args.timeframes else (config.timeframe,),
        market_filters=market_filters,
        min_rs_period=args.min_rs_period,
        max_rs_period=args.max_rs_period,
        min_leader_positions=(
            config.risk.max_position_count if args.min_positions is None else args.min_positions
        ),
        max_leader_positions=(
            config.risk.max_position_count if args.max_positions is None else args.max_positions
        ),
        max_sell_confirm_bars=args.max_sell_confirm_bars,
    )
    return run_optimize(
        config,
        market_data,
        n_trials=args.n_trials,
        timeout=args.timeout,
        seed=args.seed,
        space=space,
    )


def print_search_result(result: SearchResult, *, top: int = 20) -> None:
    for row in result.rows[: max(1, int(top))]:
        metrics = row.ranking_metrics
        print(
            f"#{row.rank:<3} score={format_float(row.score)} "
            f"return={format_pct(float(metrics.get('total_return', 0.0)))} "
            f"mdd={format_pct(float(metrics.get('mdd', 0.0)))} "
            f"trades={int(metrics.get('trade_count', 0))} params={dict(row.parameters)}"
        )
    if result.errors:
        print(f"Skipped configs: {len(result.errors)}")


def print_optimization_result(result: OptimizationResult) -> None:
    validation = result.best_evaluation.validation or result.best_evaluation.overall
    print(f"Best score: {format_float(result.best_score)}")
    print(f"Best params: {dict(result.best_parameters)}")
    print(
        "Validation: "
        f"return={format_pct(float(validation.metrics.get('total_return', 0.0)))} "
        f"mdd={format_pct(float(validation.metrics.get('mdd', 0.0)))}"
    )
    if result.best_evaluation.test is not None:
        test = result.best_evaluation.test
        print(
            "Test: "
            f"return={format_pct(float(test.metrics.get('total_return', 0.0)))} "
            f"mdd={format_pct(float(test.metrics.get('mdd', 0.0)))}"
        )


def emit_best_config(config: AppConfig, args: argparse.Namespace) -> None:
    if bool(getattr(args, "show_best_config", False)):
        strategy_text, runtime_text, data_text = split_yaml_text(config)
        print("Best Strategy YAML")
        print(strategy_text.rstrip())
        print("Best Runtime YAML")
        print(runtime_text.rstrip())
        print("Best Data YAML")
        print(data_text.rstrip())
    target = getattr(args, "save_best_dir", None)
    if target:
        strategy_path, runtime_path, data_path = save_split_yaml(config, target)
        print(f"Saved strategy: {strategy_path}")
        print(f"Saved runtime : {runtime_path}")
        print(f"Saved data    : {data_path}")


__all__ = [
    "add_optimize_arguments",
    "add_search_arguments",
    "csv_values",
    "emit_best_config",
    "optimize_from_namespace",
    "print_optimization_result",
    "print_search_result",
    "search_from_namespace",
]
