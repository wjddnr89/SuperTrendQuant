# -*- coding: utf-8 -*-
"""
Run a config-grid search over modular SuperTrend strategy features.

The search still uses the lightweight module engine, but it can read and emit
supertrend_quant-style strategy/runtime YAML shapes.  That keeps research
configs close to the live/paper runtime format.
"""

import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from module.config import (
    StrategyConfig,
    config_from_strategy_runtime,
    runtime_dict_from_config,
    strategy_dict_from_config,
)
from module.data import default_universe_path, load_market_data
from module.engine import (
    BacktestResult,
    build_prepared_context,
    evaluate_segment,
    split_index,
)
from module.metrics import format_float, format_pct


def parse_csv(value: str):
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str):
    return [int(item) for item in parse_csv(value)]


def parse_float_csv(value: str):
    return [float(item) for item in parse_csv(value)]


def parse_rs_periods(value: str):
    periods = []
    for item in parse_csv(value):
        periods.append(None if item.lower() == "auto" else int(item))
    return periods


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search modular SuperTrend strategy combinations."
    )
    parser.add_argument("--market", choices=["us", "kr"], default=None)
    parser.add_argument("--universe", default=str(default_universe_path()))
    parser.add_argument("--period", default=None)
    parser.add_argument("--strategy", default=None, help="Optional supertrend_quant-style strategy YAML.")
    parser.add_argument("--runtime", default=None, help="Optional supertrend_quant-style runtime YAML.")
    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument("--timeframes", default="30m,1h,2h,4h,1d")
    parser.add_argument("--signals", default="supertrend,triple_supertrend")
    parser.add_argument("--selectors", default="leader_top1")
    parser.add_argument("--market-filters", default="none,market_1d")
    parser.add_argument("--asset-filters", default="none,ichimoku_cloud,ema_trend")
    parser.add_argument("--sell-confirm-bars", default="1,2,3,5,8,13")
    parser.add_argument("--rs-periods", default="auto,50,100")
    parser.add_argument("--st-periods", default="10")
    parser.add_argument("--st-multipliers", default="3.0")
    parser.add_argument("--fee-rate", type=float, default=None)
    parser.add_argument("--slippage-rate", type=float, default=None)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of configs.")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--no-splits", action="store_true")
    parser.add_argument("--show-best-config", action="store_true", help="Print best config as strategy/runtime YAML.")
    return parser.parse_args()


def metric(result: BacktestResult, key: str, default=0.0):
    return result.metrics.get(key, default) if result else default


def benchmark_return(result: BacktestResult, name: str):
    if not result or name not in result.benchmarks:
        return None
    return result.benchmarks[name]["metrics"]["total_return"]


def fmt_optional_pct(value):
    return "-" if value is None else format_pct(value)


def build_row(results):
    overall = results.get("overall")
    train = results.get("train")
    validation = results.get("validation")
    test = results.get("test")
    rank_result = validation or overall
    config = overall.config

    equal_return = benchmark_return(overall, "equal")
    qqq_return = benchmark_return(overall, "qqq")
    market_return = benchmark_return(overall, "market")
    overall_return = metric(overall, "total_return")

    return {
        "Market": config.market.upper(),
        "Timeframe": config.timeframe,
        "Signal": config.signal,
        "Selector": config.selector,
        "Market Filter": config.market_filter,
        "Asset Filter": config.asset_filter,
        "Sell Confirm": config.sell_confirm_bars,
        "RS": overall.rs_period,
        "ST": f"{config.st_period}/{config.st_multiplier:g}",
        "Score": format_float(rank_result.score),
        "Overall": format_pct(overall_return),
        "Train": format_pct(metric(train, "total_return")) if train else "-",
        "Validation": format_pct(metric(validation, "total_return")) if validation else "-",
        "Test": format_pct(metric(test, "total_return")) if test else "-",
        "MDD": format_pct(metric(rank_result, "mdd")),
        "Sharpe": format_float(metric(rank_result, "sharpe")),
        "Trades": int(metric(rank_result, "trade_count")),
        "Equal B&H": fmt_optional_pct(equal_return),
        "QQQ B&H": fmt_optional_pct(qqq_return),
        "Market B&H": fmt_optional_pct(market_return),
        "Alpha Eq": fmt_optional_pct(overall_return - equal_return if equal_return is not None else None),
        "Alpha Mkt": fmt_optional_pct(overall_return - market_return if market_return is not None else None),
        "Period": f"{overall.start} -> {overall.end}",
        "_config": config,
        "_score": rank_result.score,
        "_overall_return": overall_return,
    }


def generate_configs(args, base_config):
    grids = itertools.product(
        parse_csv(args.timeframes),
        parse_csv(args.signals),
        parse_csv(args.selectors),
        parse_csv(args.market_filters),
        parse_csv(args.asset_filters),
        parse_int_csv(args.sell_confirm_bars),
        parse_rs_periods(args.rs_periods),
        parse_int_csv(args.st_periods),
        parse_float_csv(args.st_multipliers),
    )

    for idx, (
        timeframe,
        signal,
        selector,
        market_filter,
        asset_filter,
        sell_confirm_bars,
        rs_period,
        st_period,
        st_multiplier,
    ) in enumerate(grids, start=1):
        if args.limit is not None and idx > args.limit:
            break
        yield base_config.with_updates(
            market=args.market,
            timeframe=timeframe,
            period=args.period,
            initial_cash=args.initial_cash,
            signal=signal,
            selector=selector,
            market_filter=market_filter,
            asset_filter=asset_filter,
            sell_confirm_bars=sell_confirm_bars,
            rs_period=rs_period,
            st_period=st_period,
            st_multiplier=st_multiplier,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            min_coverage=args.min_coverage,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
        )


def evaluate_config_cached(bundle, config, use_splits, context_cache):
    cache_key = config.with_updates(sell_confirm_bars=1)
    if cache_key not in context_cache:
        context_cache[cache_key] = build_prepared_context(bundle, cache_key)

    context = replace(context_cache[cache_key], config=config)
    if use_splits:
        segments = split_index(
            context.common_index,
            config.train_ratio,
            config.validation_ratio,
        )
    else:
        segments = {"overall": context.common_index}

    return {
        split_name: evaluate_segment(bundle, context, split_name, index)
        for split_name, index in segments.items()
    }


def main():
    args = parse_args()
    if bool(args.strategy) != bool(args.runtime):
        raise ValueError("--strategy and --runtime must be provided together.")
    base_config = (
        config_from_strategy_runtime(args.strategy, args.runtime)
        if args.strategy and args.runtime
        else StrategyConfig()
    )
    args.market = args.market or base_config.market
    args.period = args.period or base_config.period
    args.initial_cash = args.initial_cash if args.initial_cash is not None else base_config.initial_cash
    args.fee_rate = args.fee_rate if args.fee_rate is not None else base_config.fee_rate
    args.slippage_rate = args.slippage_rate if args.slippage_rate is not None else base_config.slippage_rate
    base_config = base_config.with_updates(
        market=args.market,
        period=args.period,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        min_coverage=args.min_coverage,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    universe = Path(args.universe)
    bundle = load_market_data(args.market, universe, args.period)

    rows = []
    errors = []
    context_cache = {}
    configs = list(generate_configs(args, base_config))
    print("=" * 140, flush=True)
    print("Modular SuperTrend Combo Search", flush=True)
    print("=" * 140, flush=True)
    print(f"Market             : {args.market.upper()}", flush=True)
    print(f"Universe           : {universe}", flush=True)
    print(f"Period             : {args.period}", flush=True)
    print(f"Configs            : {len(configs)}", flush=True)
    print(f"Cached contexts    : by config except sell_confirm_bars", flush=True)
    print(f"Train/Val/Test     : {'off' if args.no_splits else f'{args.train_ratio:.0%}/{args.validation_ratio:.0%}/{1 - args.train_ratio - args.validation_ratio:.0%}'}", flush=True)
    print("-" * 140, flush=True)

    for idx, config in enumerate(configs, start=1):
        try:
            results = evaluate_config_cached(
                bundle,
                config,
                use_splits=not args.no_splits,
                context_cache=context_cache,
            )
            rows.append(build_row(results))
        except Exception as exc:
            errors.append((config, str(exc)))

        if idx % 25 == 0 or idx == len(configs):
            print(
                f"Progress           : {idx}/{len(configs)} configs, "
                f"{len(rows)} ok, {len(errors)} skipped, {len(context_cache)} cached contexts",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No config finished successfully.")

    table = pd.DataFrame(rows).sort_values(
        ["_score", "_overall_return"],
        ascending=[False, False],
    )
    best_config = table.iloc[0]["_config"]
    display = table.drop(columns=["_config", "_score", "_overall_return"]).head(args.top)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)
    print("-" * 140)
    print(f"Top {min(args.top, len(display))} Configs")
    print(display.to_string(index=False))

    if errors:
        print("-" * 140)
        print(f"Skipped configs    : {len(errors)}")
        for config, error in errors[:10]:
            print(
                f"- {config.timeframe}/{config.signal}/{config.market_filter}/"
                f"{config.asset_filter}/sell{config.sell_confirm_bars}: {error}"
            )
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more")

    if args.show_best_config:
        print("-" * 140)
        print("Best Strategy YAML")
        print(yaml.safe_dump(strategy_dict_from_config(best_config), sort_keys=False, allow_unicode=True).strip())
        print("-" * 140)
        print("Best Runtime YAML")
        print(
            yaml.safe_dump(
                runtime_dict_from_config(best_config, universe_file=str(universe)),
                sort_keys=False,
                allow_unicode=True,
            ).strip()
        )

    print("=" * 140)


if __name__ == "__main__":
    main()
