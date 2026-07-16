# -*- coding: utf-8 -*-
"""
Optimize modular SuperTrend strategy variables with Optuna.

Optuna proposes controllable variables such as timeframe, RS period,
SuperTrend parameters, sell-confirm bars, market filter, Ichimoku/EMA filters,
and Triple SuperTrend usage.  The objective maximizes validation score while
the test split is kept for final reporting only.  Best results are printed in a
supertrend_quant-style YAML shape so they can be promoted to runtime configs.
"""

import argparse
import sys
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
from module.engine import evaluate_config
from module.metrics import format_float, format_pct
from module.run_combo_search import build_row


def require_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optuna is required for optimization. Install it with: "
            "venv\\Scripts\\python.exe -m pip install -r SuperTrendQuant\\requirements.txt"
        ) from exc
    return optuna


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize modular SuperTrend strategy configs with Optuna."
    )
    parser.add_argument("--market", choices=["us", "kr"], default=None)
    parser.add_argument("--universe", default=str(default_universe_path()))
    parser.add_argument("--period", default=None)
    parser.add_argument("--strategy", default=None, help="Optional supertrend_quant-style strategy YAML.")
    parser.add_argument("--runtime", default=None, help="Optional supertrend_quant-style runtime YAML.")
    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fee-rate", type=float, default=None)
    parser.add_argument("--slippage-rate", type=float, default=None)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--min-rs-period", type=int, default=10)
    parser.add_argument("--max-rs-period", type=int, default=200)
    parser.add_argument("--max-sell-confirm-bars", type=int, default=30)
    return parser.parse_args()


def suggest_config(trial, args, base_config) -> StrategyConfig:
    timeframe = trial.suggest_categorical("timeframe", ["30m", "1h", "2h", "4h", "1d"])
    signal = trial.suggest_categorical(
        "signal",
        ["supertrend", "triple_supertrend"],
    )
    selector = trial.suggest_categorical("selector", ["leader_top1"])
    market_filter = trial.suggest_categorical(
        "market_filter",
        ["none", "market_30m", "market_1h", "market_2h", "market_4h", "market_1d"],
    )
    asset_filter = trial.suggest_categorical(
        "asset_filter",
        ["none", "ichimoku_cloud", "ema_trend", "ichimoku_cloud+ema_trend"],
    )
    rs_period = trial.suggest_int("rs_period", args.min_rs_period, args.max_rs_period)
    sell_confirm_bars = trial.suggest_int(
        "sell_confirm_bars",
        1,
        args.max_sell_confirm_bars,
    )
    st_period = trial.suggest_int("st_period", 5, 30)
    st_multiplier = trial.suggest_float("st_multiplier", 1.0, 5.0, step=0.1)
    hurdle_atr_mult = trial.suggest_float("hurdle_atr_mult", 0.25, 3.0, step=0.25)

    return base_config.with_updates(
        market=args.market,
        timeframe=timeframe,
        period=args.period,
        initial_cash=args.initial_cash,
        signal=signal,
        selector=selector,
        market_filter=market_filter,
        asset_filter=asset_filter,
        rs_period=rs_period,
        sell_confirm_bars=sell_confirm_bars,
        st_period=st_period,
        st_multiplier=st_multiplier,
        hurdle_atr_mult=hurdle_atr_mult,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )


def config_from_params(params, args, base_config) -> StrategyConfig:
    return base_config.with_updates(
        market=args.market,
        timeframe=params["timeframe"],
        period=args.period,
        initial_cash=args.initial_cash,
        signal=params["signal"],
        selector=params["selector"],
        market_filter=params["market_filter"],
        asset_filter=params["asset_filter"],
        rs_period=params["rs_period"],
        sell_confirm_bars=params["sell_confirm_bars"],
        st_period=params["st_period"],
        st_multiplier=params["st_multiplier"],
        hurdle_atr_mult=params["hurdle_atr_mult"],
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )


def main():
    optuna = require_optuna()
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

    def objective(trial):
        config = suggest_config(trial, args, base_config)
        try:
            results = evaluate_config(bundle, config, use_splits=True)
            train = results.get("train")
            validation = results.get("validation") or results["overall"]
            test = results.get("test")

            trial.set_user_attr("overall_return", results["overall"].metrics["total_return"])
            trial.set_user_attr("train_return", train.metrics["total_return"] if train else None)
            trial.set_user_attr("validation_return", validation.metrics["total_return"])
            trial.set_user_attr("test_return", test.metrics["total_return"] if test else None)
            trial.set_user_attr("validation_mdd", validation.metrics["mdd"])
            trial.set_user_attr("validation_sharpe", validation.metrics["sharpe"])
            trial.set_user_attr("validation_trades", validation.metrics["trade_count"])
            return validation.score
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return -999.0

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    print("=" * 140)
    print("Optuna Modular SuperTrend Optimization")
    print("=" * 140)
    print(f"Market             : {args.market.upper()}")
    print(f"Universe           : {universe}")
    print(f"Period             : {args.period}")
    print(f"Trials             : {args.n_trials}")
    print(f"Train/Val/Test     : {args.train_ratio:.0%}/{args.validation_ratio:.0%}/{1 - args.train_ratio - args.validation_ratio:.0%}")
    print("-" * 140)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)

    best_config = config_from_params(study.best_params, args, base_config)
    best_results = evaluate_config(bundle, best_config, use_splits=True)
    best_row = pd.DataFrame([build_row(best_results)]).drop(
        columns=["_config", "_score", "_overall_return"],
        errors="ignore",
    )

    print("Best Trial")
    print(f"Score              : {format_float(study.best_value)}")
    print(f"Params             : {study.best_params}")
    print("-" * 140)
    print("Best Config Report")
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)
    print(best_row.to_string(index=False))
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

    trials = []
    for trial in study.trials:
        if trial.value is None:
            continue
        trials.append(
            {
                "Trial": trial.number,
                "Score": trial.value,
                "Overall": trial.user_attrs.get("overall_return"),
                "Train": trial.user_attrs.get("train_return"),
                "Validation": trial.user_attrs.get("validation_return"),
                "Test": trial.user_attrs.get("test_return"),
                "Val MDD": trial.user_attrs.get("validation_mdd"),
                "Val Sharpe": trial.user_attrs.get("validation_sharpe"),
                "Val Trades": trial.user_attrs.get("validation_trades"),
                **trial.params,
            }
        )

    if trials:
        trial_table = pd.DataFrame(trials).sort_values("Score", ascending=False).head(20)
        for col in ["Overall", "Train", "Validation", "Test", "Val MDD"]:
            if col in trial_table:
                trial_table[col] = trial_table[col].map(
                    lambda value: "-" if value is None else format_pct(value)
                )
        trial_table["Score"] = trial_table["Score"].map(format_float)
        if "Val Sharpe" in trial_table:
            trial_table["Val Sharpe"] = trial_table["Val Sharpe"].map(
                lambda value: "-" if value is None else format_float(value)
            )
        print("-" * 140)
        print("Top 20 Trials")
        print(trial_table.to_string(index=False))

    print("=" * 140)


if __name__ == "__main__":
    main()
