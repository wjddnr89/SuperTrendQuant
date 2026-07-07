# -*- coding: utf-8 -*-
"""
기준 백테스트 코드입니다.

universe.json의 미국 종목을 대상으로 최근 60일 30분봉 Supertrend를 돌립니다.
각 종목에 동일 금액을 배분하는 독립 포트폴리오 방식이며,
수수료/슬리피지를 반영하고 Equal B&H와 QQQ B&H를 함께 출력합니다.
"""

import argparse
from pathlib import Path

import pandas as pd

from backtest_supertrend_universe import (
    Asset,
    DEFAULT_ATR_PERIOD,
    DEFAULT_FEE_RATE,
    DEFAULT_MULTIPLIER,
    DEFAULT_SLIPPAGE_RATE,
    backtest_symbol,
    build_display_tables,
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


DEFAULT_PERIOD = "60d"
INTERVAL = "30m"
QQQ_SYMBOL = "QQQ"


def parse_args():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Baseline 30m Supertrend backtest for universe.json US tickers."
    )
    parser.add_argument("--universe", default=str(project_dir / "universe.json"))
    parser.add_argument("--period", default=DEFAULT_PERIOD)
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
        "--hide-trades",
        action="store_true",
        help="Hide the trade-by-trade table.",
    )
    return parser.parse_args()


def run_backtest(args):
    assets = load_universe(args.universe, "us")
    data, skipped = download_data(
        assets=assets,
        period=args.period,
        interval=INTERVAL,
    )
    qqq_data, qqq_skipped = download_data(
        assets=[Asset(symbol=QQQ_SYMBOL, yf_symbol=QQQ_SYMBOL, market="US")],
        period=args.period,
        interval=INTERVAL,
    )

    if not data:
        raise RuntimeError("No 30m price data was downloaded.")
    if qqq_skipped or QQQ_SYMBOL not in qqq_data:
        raise RuntimeError("QQQ benchmark data was not downloaded.")

    data, coverage_dropped = filter_by_history_coverage(data, args.min_coverage)
    common_index = get_common_index(data)
    common_start = common_index[0]
    common_end = common_index[-1]
    per_symbol_cash = args.initial_cash / len(data)

    equity_parts = []
    all_report_trades = []
    symbol_rows = []

    for symbol, df in data.items():
        st_df = calculate_supertrend(
            df,
            period=args.atr_period,
            multiplier=args.multiplier,
            atr_method=args.atr_method,
        )
        equity, trades = backtest_symbol(
            symbol=symbol,
            df=st_df,
            initial_cash=per_symbol_cash,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
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

        symbol_metrics = calculate_metrics(report_equity, report_trades, INTERVAL)
        symbol_rows.append(
            {
                "symbol": symbol,
                "total_return": symbol_metrics["total_return"],
                "mdd": symbol_metrics["mdd"],
                "sharpe": symbol_metrics["sharpe"],
                "win_rate": symbol_metrics["win_rate"],
                "payoff_ratio": symbol_metrics["payoff_ratio"],
                "trade_count": symbol_metrics["trade_count"],
            }
        )

    if not equity_parts:
        raise RuntimeError("No equity series could be built.")

    strategy_equity = pd.concat(equity_parts, axis=1, join="inner").sum(axis=1)
    strategy_equity.name = "strategy"
    benchmark_data = {symbol: df.loc[common_index].copy() for symbol, df in data.items()}
    benchmark_equity = build_equal_weight_benchmark(benchmark_data, args.initial_cash)
    qqq_benchmark_equity = build_single_asset_benchmark(
        qqq_data[QQQ_SYMBOL],
        common_index,
        args.initial_cash,
    )

    report_index = strategy_equity.index.intersection(benchmark_equity.index)
    report_index = report_index.intersection(qqq_benchmark_equity.index)
    strategy_equity = strategy_equity.loc[report_index]
    benchmark_equity = benchmark_equity.loc[report_index]
    qqq_benchmark_equity = qqq_benchmark_equity.loc[report_index]

    strategy_metrics = calculate_metrics(strategy_equity, all_report_trades, INTERVAL)
    benchmark_metrics = calculate_metrics(benchmark_equity, [], INTERVAL)
    qqq_benchmark_metrics = calculate_metrics(qqq_benchmark_equity, [], INTERVAL)
    alpha_vs_equal = strategy_metrics["total_return"] - benchmark_metrics["total_return"]
    alpha_vs_qqq = strategy_metrics["total_return"] - qqq_benchmark_metrics["total_return"]

    summary_table = build_summary_table(
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        qqq_benchmark_metrics=qqq_benchmark_metrics,
        alpha_vs_equal=alpha_vs_equal,
        alpha_vs_qqq=alpha_vs_qqq,
    )
    _, by_symbol_table, trades_table = build_display_tables(
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        alpha=alpha_vs_equal,
        symbol_rows=symbol_rows,
        all_trades=all_report_trades,
    )

    return {
        "summary_table": summary_table,
        "by_symbol_table": by_symbol_table,
        "trades_table": trades_table,
        "strategy_equity": strategy_equity,
        "benchmark_equity": benchmark_equity,
        "qqq_benchmark_equity": qqq_benchmark_equity,
        "skipped": skipped,
        "qqq_skipped": qqq_skipped,
        "coverage_dropped": coverage_dropped,
        "symbol_count": len(data),
    }


def build_single_asset_benchmark(df, common_index, initial_cash):
    close = df["Close"].reindex(common_index, method="ffill").dropna()
    if close.empty:
        raise ValueError("No QQQ benchmark timeline after alignment.")
    equity = close / close.iloc[0] * initial_cash
    equity.name = "qqq_buy_and_hold"
    return equity


def build_summary_table(
    strategy_metrics,
    benchmark_metrics,
    qqq_benchmark_metrics,
    alpha_vs_equal,
    alpha_vs_qqq,
):
    return pd.DataFrame(
        [
            {
                "Name": "Supertrend",
                "Return": format_pct(strategy_metrics["total_return"]),
                "MDD": format_pct(strategy_metrics["mdd"]),
                "Sharpe": format_float(strategy_metrics["sharpe"]),
                "Win Rate": format_pct(strategy_metrics["win_rate"]),
                "Payoff": format_float(strategy_metrics["payoff_ratio"]),
                "Trades": strategy_metrics["trade_count"],
                "Alpha vs Equal": format_pct(alpha_vs_equal),
                "Alpha vs QQQ": format_pct(alpha_vs_qqq),
            },
            {
                "Name": "Equal B&H",
                "Return": format_pct(benchmark_metrics["total_return"]),
                "MDD": format_pct(benchmark_metrics["mdd"]),
                "Sharpe": format_float(benchmark_metrics["sharpe"]),
                "Win Rate": "-",
                "Payoff": "-",
                "Trades": "-",
                "Alpha vs Equal": format_pct(0.0),
                "Alpha vs QQQ": "-",
            },
            {
                "Name": "QQQ B&H",
                "Return": format_pct(qqq_benchmark_metrics["total_return"]),
                "MDD": format_pct(qqq_benchmark_metrics["mdd"]),
                "Sharpe": format_float(qqq_benchmark_metrics["sharpe"]),
                "Win Rate": "-",
                "Payoff": "-",
                "Trades": "-",
                "Alpha vs Equal": "-",
                "Alpha vs QQQ": format_pct(0.0),
            },
        ]
    )


def print_results(args, result):
    strategy_equity = result["strategy_equity"]
    start_label = strategy_equity.index[0]
    end_label = strategy_equity.index[-1]

    print("=" * 100)
    print("Supertrend 30m US Baseline Backtest")
    print("=" * 100)
    print("Data source        : Yahoo Finance via yfinance")
    print("Universe           : universe.json / US_UNIVERSE_LIST only")
    print(f"Bars               : {INTERVAL}, {args.period}")
    print(f"Fee / Slippage     : {args.fee_rate:.4%} / {args.slippage_rate:.4%} per fill")
    print(f"Common period      : {start_label} -> {end_label}")
    print(f"Downloaded symbols : {result['symbol_count']}")
    if result["skipped"]:
        print(f"Skipped symbols    : {', '.join(asset.symbol for asset in result['skipped'])}")
    if result["coverage_dropped"]:
        dropped_text = ", ".join(
            f"{symbol}({bars}/{required})"
            for symbol, bars, required in result["coverage_dropped"]
        )
        print(f"Dropped short data : {dropped_text}")

    print("-" * 100)
    print("Summary")
    print(result["summary_table"].to_string(index=False))
    print("-" * 100)
    print("By Symbol")
    print(result["by_symbol_table"].to_string(index=False))

    trades_table = result["trades_table"]
    if not args.hide_trades and not trades_table.empty:
        print("-" * 100)
        print("Trades")
        print(trades_table.to_string(index=False))
    print("=" * 100)


def main():
    args = parse_args()
    result = run_backtest(args)
    print_results(args, result)


if __name__ == "__main__":
    main()
