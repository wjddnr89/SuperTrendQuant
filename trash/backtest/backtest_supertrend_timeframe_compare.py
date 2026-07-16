# -*- coding: utf-8 -*-
"""
Supertrend 시간봉 비교 실험 파일입니다.

같은 미국 유니버스를 30분봉, 1시간봉, 2시간봉, 4시간봉, 일봉으로 돌려
성과를 비교합니다. 기준 백테스트와 같은 지표/벤치마크 체계를 사용해
어떤 봉이 유리한지 빠르게 확인하는 용도입니다.
"""

import argparse
from pathlib import Path

import pandas as pd

from backtest_supertrend_universe import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_FEE_RATE,
    DEFAULT_MULTIPLIER,
    DEFAULT_SLIPPAGE_RATE,
    backtest_symbol,
    build_equal_weight_benchmark,
    calculate_metrics,
    calculate_supertrend,
    download_data,
    filter_by_history_coverage,
    format_float,
    format_pct,
    get_common_index,
    load_universe,
)


TIMEFRAMES = {
    "30m": {"source": "intraday", "rule": None},
    "1h": {"source": "intraday", "rule": "1h"},
    "2h": {"source": "intraday", "rule": "2h"},
    "4h": {"source": "intraday", "rule": "4h"},
    "1d": {"source": "daily", "rule": None},
}


def parse_args():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare Supertrend backtest results across 30m, 1h, 2h, 4h, and 1d for US universe."
    )
    parser.add_argument("--universe", default=str(project_dir / "universe.json"))
    parser.add_argument("--period", default="60d")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--atr-period", type=int, default=DEFAULT_ATR_PERIOD)
    parser.add_argument("--multiplier", type=float, default=DEFAULT_MULTIPLIER)
    parser.add_argument(
        "--atr-method",
        choices=["wilder", "sma"],
        default="wilder",
        help="wilder matches TradingView atr(). sma matches the alternate ATR option.",
    )
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="Drop tickers with fewer bars than this fraction of the most complete ticker.",
    )
    parser.add_argument(
        "--top-symbols",
        type=int,
        default=5,
        help="Number of best and worst symbols to print for each timeframe. Use 0 to hide.",
    )
    return parser.parse_args()


def resample_ohlc(data, rule):
    resampled = {}
    for symbol, df in data.items():
        tf_df = (
            df.resample(rule, closed="left", label="left", origin="start_day", offset="9h30min")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna(subset=["Open", "High", "Low", "Close"])
        )
        if not tf_df.empty:
            resampled[symbol] = tf_df
    return resampled


def restrict_to_symbols(data, symbols):
    return {symbol: data[symbol] for symbol in symbols if symbol in data}


def run_backtest_for_timeframe(
    timeframe,
    data,
    initial_cash,
    atr_period,
    multiplier,
    atr_method,
    fee_rate,
    slippage_rate,
):
    common_index = get_common_index(data)
    common_start = common_index[0]
    common_end = common_index[-1]
    per_symbol_cash = initial_cash / len(data)

    equity_parts = []
    all_report_trades = []
    symbol_rows = []

    for symbol, df in data.items():
        st_df = calculate_supertrend(
            df,
            period=atr_period,
            multiplier=multiplier,
            atr_method=atr_method,
        )
        equity, trades = backtest_symbol(
            symbol=symbol,
            df=st_df,
            initial_cash=per_symbol_cash,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        report_equity = equity.loc[common_index]
        if report_equity.empty or report_equity.iloc[0] <= 0:
            continue

        report_equity = (report_equity / report_equity.iloc[0]) * per_symbol_cash
        report_equity.name = symbol
        equity_parts.append(report_equity)

        report_trades = [
            trade for trade in trades
            if common_start <= trade["exit_time"] <= common_end
        ]
        all_report_trades.extend(report_trades)

        symbol_metrics = calculate_metrics(report_equity, report_trades, timeframe)
        symbol_rows.append(
            {
                "Timeframe": timeframe,
                "Symbol": symbol,
                "Return": symbol_metrics["total_return"],
                "MDD": symbol_metrics["mdd"],
                "Sharpe": symbol_metrics["sharpe"],
                "Win Rate": symbol_metrics["win_rate"],
                "Payoff": symbol_metrics["payoff_ratio"],
                "Trades": symbol_metrics["trade_count"],
            }
        )

    if not equity_parts:
        raise ValueError(f"No equity series could be built for {timeframe}.")

    strategy_equity = pd.concat(equity_parts, axis=1, join="inner").sum(axis=1)
    strategy_equity.name = "strategy"
    benchmark_data = {symbol: df.loc[common_index].copy() for symbol, df in data.items()}
    benchmark_equity = build_equal_weight_benchmark(benchmark_data, initial_cash)

    report_index = strategy_equity.index.intersection(benchmark_equity.index)
    strategy_equity = strategy_equity.loc[report_index]
    benchmark_equity = benchmark_equity.loc[report_index]

    strategy_metrics = calculate_metrics(strategy_equity, all_report_trades, timeframe)
    benchmark_metrics = calculate_metrics(benchmark_equity, [], timeframe)
    alpha = strategy_metrics["total_return"] - benchmark_metrics["total_return"]

    return {
        "timeframe": timeframe,
        "start": strategy_equity.index[0],
        "end": strategy_equity.index[-1],
        "symbols": len(data),
        "bars": len(strategy_equity),
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "alpha": alpha,
        "symbol_rows": symbol_rows,
    }


def format_summary_row(result):
    strategy = result["strategy_metrics"]
    benchmark = result["benchmark_metrics"]
    return {
        "TF": result["timeframe"],
        "Period": f"{result['start']} -> {result['end']}",
        "Symbols": result["symbols"],
        "Bars": result["bars"],
        "Strategy": format_pct(strategy["total_return"]),
        "B&H": format_pct(benchmark["total_return"]),
        "Alpha": format_pct(result["alpha"]),
        "MDD": format_pct(strategy["mdd"]),
        "Sharpe": format_float(strategy["sharpe"]),
        "Win Rate": format_pct(strategy["win_rate"]),
        "Payoff": format_float(strategy["payoff_ratio"]),
        "Trades": strategy["trade_count"],
    }


def format_symbol_table(rows, limit):
    table = pd.DataFrame(rows).sort_values("Return", ascending=False)
    if limit > 0 and len(table) > limit * 2:
        table = pd.concat([table.head(limit), table.tail(limit)], ignore_index=True)

    return table.assign(
        Return=table["Return"].map(format_pct),
        MDD=table["MDD"].map(format_pct),
        Sharpe=table["Sharpe"].map(format_float),
        **{
            "Win Rate": table["Win Rate"].map(format_pct),
            "Payoff": table["Payoff"].map(format_float),
        },
    )


def print_results(
    results,
    intraday_dropped,
    daily_dropped,
    skipped_intraday,
    skipped_daily,
    top_symbols,
    fee_rate,
    slippage_rate,
):
    summary = pd.DataFrame([format_summary_row(result) for result in results])
    ranked = summary.copy()
    ranked["_alpha"] = [result["alpha"] for result in results]
    best = ranked.sort_values("_alpha", ascending=False).iloc[0]

    print("=" * 120)
    print("Supertrend US Universe Timeframe Comparison")
    print("=" * 120)
    print("Data source        : Yahoo Finance via yfinance")
    print("Universe           : universe.json / US_UNIVERSE_LIST only")
    print("Lookback           : recent 60 days")
    print("Intraday source    : 30m; 1h/2h/4h are resampled from 30m")
    print(f"Fee / Slippage     : {fee_rate:.4%} / {slippage_rate:.4%} per fill")
    if skipped_intraday:
        print(f"Skipped intraday   : {', '.join(asset.symbol for asset in skipped_intraday)}")
    if skipped_daily:
        print(f"Skipped daily      : {', '.join(asset.symbol for asset in skipped_daily)}")
    if intraday_dropped:
        dropped = ", ".join(f"{symbol}({bars}/{required})" for symbol, bars, required in intraday_dropped)
        print(f"Dropped intraday   : {dropped}")
    if daily_dropped:
        dropped = ", ".join(f"{symbol}({bars}/{required})" for symbol, bars, required in daily_dropped)
        print(f"Dropped daily      : {dropped}")
    print("-" * 120)
    print("Comparison Summary")
    print(summary.drop(columns=["_alpha"], errors="ignore").to_string(index=False))
    print("-" * 120)
    print(
        "Best Alpha         : "
        f"{best['TF']} | Alpha {best['Alpha']} | Strategy {best['Strategy']} | B&H {best['B&H']}"
    )

    if top_symbols > 0:
        for result in results:
            print("-" * 120)
            print(f"By Symbol - {result['timeframe']} (top/bottom {top_symbols})")
            print(format_symbol_table(result["symbol_rows"], top_symbols).to_string(index=False))
    print("=" * 120)


def main():
    args = parse_args()
    assets = load_universe(args.universe, "us")

    intraday_data, skipped_intraday = download_data(
        assets=assets,
        period=args.period,
        interval="30m",
    )
    daily_data, skipped_daily = download_data(
        assets=assets,
        period=args.period,
        interval="1d",
    )

    if not intraday_data:
        raise RuntimeError("No 30m data was downloaded.")
    if not daily_data:
        raise RuntimeError("No daily data was downloaded.")

    intraday_data, intraday_dropped = filter_by_history_coverage(
        intraday_data,
        args.min_coverage,
    )
    daily_data, daily_dropped = filter_by_history_coverage(
        daily_data,
        args.min_coverage,
    )

    common_symbols = sorted(set(intraday_data) & set(daily_data))
    if not common_symbols:
        raise RuntimeError("No common symbols across intraday and daily data.")

    intraday_data = restrict_to_symbols(intraday_data, common_symbols)
    daily_data = restrict_to_symbols(daily_data, common_symbols)

    timeframe_data = {}
    for timeframe, config in TIMEFRAMES.items():
        if config["source"] == "daily":
            timeframe_data[timeframe] = daily_data
        elif config["rule"] is None:
            timeframe_data[timeframe] = intraday_data
        else:
            timeframe_data[timeframe] = resample_ohlc(intraday_data, config["rule"])

    results = []
    for timeframe in TIMEFRAMES:
        result = run_backtest_for_timeframe(
            timeframe=timeframe,
            data=timeframe_data[timeframe],
            initial_cash=args.initial_cash,
            atr_period=args.atr_period,
            multiplier=args.multiplier,
            atr_method=args.atr_method,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        results.append(result)

    print_results(
        results=results,
        intraday_dropped=intraday_dropped,
        daily_dropped=daily_dropped,
        skipped_intraday=skipped_intraday,
        skipped_daily=skipped_daily,
        top_symbols=args.top_symbols,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )


if __name__ == "__main__":
    main()
