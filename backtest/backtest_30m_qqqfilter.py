# -*- coding: utf-8 -*-
"""
기준 30분봉 전략에 QQQ 시장 필터를 붙여 비교하는 파일입니다.

종목 매수/매도 신호는 30분봉 Supertrend를 그대로 쓰고,
신규 매수는 QQQ Supertrend가 상승 추세일 때만 허용합니다.
None, QQQ 30m, QQQ 1h, QQQ 2h, QQQ 4h, QQQ 1d 필터별 성과를 한 번에 출력합니다.
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
STOCK_INTERVAL = "30m"
QQQ_SYMBOL = "QQQ"

QQQ_FILTERS = {
    "None": {"source": "none", "rule": None},
    "QQQ 30m": {"source": "intraday", "rule": None},
    "QQQ 1h": {"source": "intraday", "rule": "1h"},
    "QQQ 2h": {"source": "intraday", "rule": "2h"},
    "QQQ 4h": {"source": "intraday", "rule": "4h"},
    "QQQ 1d": {"source": "daily", "rule": None},
}


def parse_args():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="30m Supertrend backtest with QQQ trend filters by timeframe."
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
        "--top-symbols",
        type=int,
        default=0,
        help="Print top and bottom symbols for each filter. Use 0 to hide.",
    )
    return parser.parse_args()


def resample_ohlc(df, rule):
    return (
        df.resample(rule, closed="left", label="left", origin="start_day", offset="9h30min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna(subset=["Open", "High", "Low", "Close"])
    )


def align_intraday_filter(trend, common_index):
    aligned = trend.sort_index().reindex(common_index, method="ffill")
    return aligned.fillna(-1).astype(int).eq(1)


def align_daily_filter(trend, common_index):
    daily = trend.sort_index()
    daily_by_date = pd.Series(
        daily.to_numpy(),
        index=pd.Index([pd.Timestamp(idx).date() for idx in daily.index]),
    )

    states = []
    for ts in common_index:
        current_date = pd.Timestamp(ts).date()
        past_daily = daily_by_date[daily_by_date.index < current_date]
        states.append(int(past_daily.iloc[-1]) if not past_daily.empty else -1)

    return pd.Series(states, index=common_index).eq(1)


def build_qqq_filter_series(qqq_30m, qqq_daily, common_index, args):
    filter_series = {}

    for filter_name, config in QQQ_FILTERS.items():
        if config["source"] == "none":
            filter_series[filter_name] = pd.Series(True, index=common_index)
            continue

        if config["source"] == "daily":
            qqq_df = qqq_daily.copy()
            aligner = align_daily_filter
        elif config["rule"] is None:
            qqq_df = qqq_30m.copy()
            aligner = align_intraday_filter
        else:
            qqq_df = resample_ohlc(qqq_30m, config["rule"])
            aligner = align_intraday_filter

        qqq_st = calculate_supertrend(
            qqq_df,
            period=args.atr_period,
            multiplier=args.multiplier,
            atr_method=args.atr_method,
        )
        filter_series[filter_name] = aligner(qqq_st["Trend"], common_index)

    return filter_series


def backtest_symbol_with_filter(symbol, df, initial_cash, fee_rate, slippage_rate, buy_filter):
    cash = float(initial_cash)
    qty = 0.0
    entry_price = None
    entry_time = None
    entry_cash = None
    pending_order = None
    trades = []
    equity_points = []

    if len(df) < 3:
        return pd.Series(dtype=float), trades

    for i, (timestamp, row) in enumerate(df.iterrows()):
        if pending_order == "BUY" and qty == 0.0:
            fill_price = row["Open"] * (1.0 + slippage_rate)
            if fill_price > 0 and cash > 0:
                entry_cash = cash
                qty = (cash * (1.0 - fee_rate)) / fill_price
                cash = 0.0
                entry_price = fill_price
                entry_time = timestamp

        elif pending_order == "SELL" and qty > 0.0:
            fill_price = row["Open"] * (1.0 - slippage_rate)
            cash = qty * fill_price * (1.0 - fee_rate)
            pnl_pct = cash / entry_cash - 1.0
            trades.append(
                {
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "entry_price": entry_price,
                    "exit_price": fill_price,
                    "pnl_pct": pnl_pct,
                }
            )
            qty = 0.0
            entry_price = None
            entry_time = None
            entry_cash = None

        pending_order = None

        equity = cash + qty * row["Close"]
        equity_points.append((timestamp, equity))

        if i == len(df) - 1:
            continue

        qqq_allows_buy = bool(buy_filter.get(timestamp, False))
        if qty == 0.0 and bool(row["BuySignal"]) and qqq_allows_buy:
            pending_order = "BUY"
        elif qty > 0.0 and bool(row["SellSignal"]):
            pending_order = "SELL"

    if qty > 0.0:
        timestamp = df.index[-1]
        close_price = df["Close"].iloc[-1] * (1.0 - slippage_rate)
        cash = qty * close_price * (1.0 - fee_rate)
        pnl_pct = cash / entry_cash - 1.0
        trades.append(
            {
                "symbol": symbol,
                "entry_time": entry_time,
                "exit_time": timestamp,
                "entry_price": entry_price,
                "exit_price": close_price,
                "pnl_pct": pnl_pct,
            }
        )
        equity_points[-1] = (timestamp, cash)

    equity = pd.Series(
        [point[1] for point in equity_points],
        index=[point[0] for point in equity_points],
        name=symbol,
    )
    return equity, trades


def run_filter_backtest(filter_name, buy_filter, data, common_index, args):
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
        equity, trades = backtest_symbol_with_filter(
            symbol=symbol,
            df=st_df,
            initial_cash=per_symbol_cash,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            buy_filter=buy_filter,
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

        symbol_metrics = calculate_metrics(report_equity, report_trades, STOCK_INTERVAL)
        symbol_rows.append(
            {
                "Filter": filter_name,
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
        raise RuntimeError(f"No equity series could be built for {filter_name}.")

    strategy_equity = pd.concat(equity_parts, axis=1, join="inner").sum(axis=1)
    strategy_equity.name = "strategy"
    benchmark_data = {symbol: df.loc[common_index].copy() for symbol, df in data.items()}
    benchmark_equity = build_equal_weight_benchmark(benchmark_data, args.initial_cash)

    report_index = strategy_equity.index.intersection(benchmark_equity.index)
    strategy_equity = strategy_equity.loc[report_index]
    benchmark_equity = benchmark_equity.loc[report_index]

    strategy_metrics = calculate_metrics(strategy_equity, all_report_trades, STOCK_INTERVAL)
    benchmark_metrics = calculate_metrics(benchmark_equity, [], STOCK_INTERVAL)

    return {
        "filter": filter_name,
        "start": strategy_equity.index[0],
        "end": strategy_equity.index[-1],
        "symbols": len(data),
        "bars": len(strategy_equity),
        "filter_on_ratio": buy_filter.loc[report_index].mean(),
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "alpha": strategy_metrics["total_return"] - benchmark_metrics["total_return"],
        "symbol_rows": symbol_rows,
    }


def format_summary_row(result):
    strategy = result["strategy_metrics"]
    benchmark = result["benchmark_metrics"]
    return {
        "Filter": result["filter"],
        "Period": f"{result['start']} -> {result['end']}",
        "QQQ On": f"{result['filter_on_ratio'] * 100:.1f}%",
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


def print_results(args, results, skipped, coverage_dropped):
    summary = pd.DataFrame([format_summary_row(result) for result in results])
    ranked = summary.copy()
    ranked["_alpha"] = [result["alpha"] for result in results]
    best = ranked.sort_values("_alpha", ascending=False).iloc[0]

    print("=" * 130)
    print("Supertrend 30m US Backtest With QQQ Market Filters")
    print("=" * 130)
    print("Data source        : Yahoo Finance via yfinance")
    print("Universe           : universe.json / US_UNIVERSE_LIST only")
    print("Stock strategy     : 30m Supertrend")
    print("QQQ filter         : None plus QQQ Supertrend Trend == 1 variants")
    print(f"Lookback           : {args.period}")
    print(f"Fee / Slippage     : {args.fee_rate:.4%} / {args.slippage_rate:.4%} per fill")
    if skipped:
        print(f"Skipped symbols    : {', '.join(asset.symbol for asset in skipped)}")
    if coverage_dropped:
        dropped_text = ", ".join(
            f"{symbol}({bars}/{required})" for symbol, bars, required in coverage_dropped
        )
        print(f"Dropped short data : {dropped_text}")
    print("-" * 130)
    print("QQQ Filter Comparison")
    print(summary.drop(columns=["_alpha"], errors="ignore").to_string(index=False))
    print("-" * 130)
    print(
        "Best Alpha         : "
        f"{best['Filter']} | Alpha {best['Alpha']} | Strategy {best['Strategy']} | B&H {best['B&H']}"
    )

    if args.top_symbols > 0:
        for result in results:
            print("-" * 130)
            print(f"By Symbol - {result['filter']} (top/bottom {args.top_symbols})")
            print(format_symbol_table(result["symbol_rows"], args.top_symbols).to_string(index=False))
    print("=" * 130)


def main():
    args = parse_args()
    assets = load_universe(args.universe, "us")
    qqq_asset = Asset(symbol=QQQ_SYMBOL, yf_symbol=QQQ_SYMBOL, market="US")

    stock_data, skipped = download_data(
        assets=assets,
        period=args.period,
        interval=STOCK_INTERVAL,
    )
    qqq_30m_data, qqq_30m_skipped = download_data(
        assets=[qqq_asset],
        period=args.period,
        interval=STOCK_INTERVAL,
    )
    qqq_daily_data, qqq_daily_skipped = download_data(
        assets=[qqq_asset],
        period=args.period,
        interval="1d",
    )

    if not stock_data:
        raise RuntimeError("No stock 30m data was downloaded.")
    if qqq_30m_skipped or QQQ_SYMBOL not in qqq_30m_data:
        raise RuntimeError("QQQ 30m filter data was not downloaded.")
    if qqq_daily_skipped or QQQ_SYMBOL not in qqq_daily_data:
        raise RuntimeError("QQQ daily filter data was not downloaded.")

    stock_data, coverage_dropped = filter_by_history_coverage(
        stock_data,
        args.min_coverage,
    )
    common_index = get_common_index(stock_data)

    qqq_filters = build_qqq_filter_series(
        qqq_30m=qqq_30m_data[QQQ_SYMBOL],
        qqq_daily=qqq_daily_data[QQQ_SYMBOL],
        common_index=common_index,
        args=args,
    )

    results = []
    for filter_name in QQQ_FILTERS:
        results.append(
            run_filter_backtest(
                filter_name=filter_name,
                buy_filter=qqq_filters[filter_name],
                data=stock_data,
                common_index=common_index,
                args=args,
            )
        )

    print_results(
        args=args,
        results=results,
        skipped=skipped,
        coverage_dropped=coverage_dropped,
    )


if __name__ == "__main__":
    main()
