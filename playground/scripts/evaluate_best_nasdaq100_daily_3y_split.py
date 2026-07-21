# Evaluate the current best Nasdaq-100 daily strategy with train/validation/test.
#
# This script fixes the best-return combination found by
# search_nasdaq100_daily_3y_grid.py and evaluates it over the same 3-year
# rolling Nasdaq-100 daily universe.  It chronologically splits the run into
# train/validation/test = 6:2:2 and prints strategy plus benchmark metrics
# directly to the console, including same-period QQQ return and alpha.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PLAYGROUND_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PLAYGROUND_ROOT / "src"))

from market_data_source import load_experiment_market_data
from supertrend_quant.config import AppConfig, load_split_config
from supertrend_quant.metrics import format_float, format_pct
from supertrend_quant.research import apply_config_overlay, evaluate_config


BEST_RETURN_OVERLAY = {
    "entry": "triple",
    "market_filter": "none",
    "asset_filter": "ema_trend",
    "sell_confirm_bars": 8,
    "rs_period": 100,
    "max_positions": 1,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the best Nasdaq-100 daily strategy with 6:2:2 splits."
    )
    parser.add_argument(
        "--strategy",
        default=str(PLAYGROUND_ROOT / "configs" / "strategies" / "leader_rotation.yaml"),
    )
    parser.add_argument(
        "--runtime",
        default=str(PLAYGROUND_ROOT / "configs" / "runtimes" / "research_us_nasdaq100_rolling.yaml"),
    )
    parser.add_argument("--period", default="3y")
    parser.add_argument("--data-source", choices=("local", "yahoo"), default="local")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    return parser


def best_config(args: argparse.Namespace) -> AppConfig:
    base = load_split_config(args.strategy, args.runtime)
    base = base.__class__(**{**base.__dict__, "period": args.period, "timeframe": "1d"})
    return apply_config_overlay(base, BEST_RETURN_OVERLAY)


def metric_line(
    name: str,
    start,
    end,
    bars: int,
    metrics: Mapping[str, float | int],
    qqq_return: float,
) -> str:
    strategy_return = float(metrics.get("total_return", 0.0))
    return (
        f"{name:<10} "
        f"{str(start):<19} {str(end):<19} "
        f"{bars:>4} "
        f"{format_pct(strategy_return):>9} "
        f"{format_pct(qqq_return):>9} "
        f"{format_pct(strategy_return - qqq_return):>9} "
        f"{format_pct(float(metrics.get('mdd', 0.0))):>8} "
        f"{format_float(float(metrics.get('sharpe', 0.0))):>6} "
        f"{format_pct(float(metrics.get('win_rate', 0.0))):>8} "
        f"{format_float(float(metrics.get('payoff_ratio', 0.0))):>6} "
        f"{int(metrics.get('trade_count', 0)):>6}"
    )


def print_strategy_summary(config: AppConfig) -> None:
    print("Fixed Strategy")
    print("Entry         : Triple SuperTrend")
    print("Market Filter : None")
    print("Asset Filter  : EMA trend")
    print("Sell Confirm  : 8 bars")
    print("RS Period     : 100")
    print("Max Positions : 1")
    print(f"Period        : {config.period}")
    print(f"Timeframe     : {config.timeframe}")
    print()


def main() -> None:
    args = build_parser().parse_args()
    config = best_config(args)
    print_strategy_summary(config)
    print(f"Data Source   : {args.data_source}")
    print("Loading shared market data...", flush=True)
    market_data = load_experiment_market_data(
        config,
        data_source=args.data_source,
        strategy_path=args.strategy,
        runtime_path=args.runtime,
    )
    result = evaluate_config(
        config,
        market_data,
        include_test=True,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    print()
    print("Strategy Metrics")
    print(
        "Segment    Start               End                 Bars    Return       QQQ     Alpha      MDD Sharpe  WinRate Payoff Trades"
    )
    for name, segment in result.segments.items():
        qqq = segment.benchmarks.get("qqq")
        qqq_return = float(qqq.metrics.get("total_return", 0.0)) if qqq is not None else 0.0
        print(metric_line(name, segment.start, segment.end, segment.bars, segment.metrics, qqq_return))

    print()
    print("Benchmarks")
    print("Segment    Benchmark  Return      MDD Sharpe")
    for name, segment in result.segments.items():
        for benchmark_name, benchmark in segment.benchmarks.items():
            metrics = benchmark.metrics
            print(
                f"{name:<10} {benchmark_name:<10} "
                f"{format_pct(float(metrics.get('total_return', 0.0))):>9} "
                f"{format_pct(float(metrics.get('mdd', 0.0))):>8} "
                f"{format_float(float(metrics.get('sharpe', 0.0))):>6}"
            )

    if market_data.skipped:
        print()
        print(f"Skipped: {', '.join(market_data.skipped)}")


if __name__ == "__main__":
    main()
