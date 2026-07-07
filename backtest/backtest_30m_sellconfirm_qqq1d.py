# -*- coding: utf-8 -*-
"""
QQQ 1일봉 필터와 매도 확인 봉 수를 비교하는 실험 파일입니다.

기준 30분봉 Supertrend 전략에 QQQ의 직전 완료 일봉 Supertrend 필터를 적용하고,
종목 Trend == -1 상태가 1~5개 30분봉 연속 확인될 때 매도하는 경우를 비교합니다.
None 행은 QQQ 필터 없이 기존 기준 전략과 같은 조건을 보여줍니다.
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
SELL_CONFIRM_BARS = [1, 2, 3, 4, 5]


def parse_args():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare 30m Supertrend sell confirmation bars with QQQ 1d filter."
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
        help="Print top and bottom symbols for each sell confirmation setting. Use 0 to hide.",
    )
    return parser.parse_args()


def build_single_asset_benchmark(df, common_index, initial_cash):
    close = df["Close"].reindex(common_index, method="ffill").dropna()
    if close.empty:
        raise ValueError("No QQQ benchmark timeline after alignment.")
    equity = close / close.iloc[0] * initial_cash
    equity.name = "qqq_buy_and_hold"
    return equity


def align_qqq_daily_filter(qqq_daily, common_index, args):
    qqq_st = calculate_supertrend(
        qqq_daily.copy(),
        period=args.atr_period,
        multiplier=args.multiplier,
        atr_method=args.atr_method,
    )
    daily_trend = qqq_st["Trend"].sort_index()
    daily_by_date = pd.Series(
        daily_trend.to_numpy(),
        index=pd.Index([pd.Timestamp(idx).date() for idx in daily_trend.index]),
    )

    states = []
    for ts in common_index:
        current_date = pd.Timestamp(ts).date()
        past_daily = daily_by_date[daily_by_date.index < current_date]
        states.append(int(past_daily.iloc[-1]) if not past_daily.empty else -1)

    return pd.Series(states, index=common_index).eq(1)


def backtest_symbol_with_sell_confirm(
    symbol,
    df,
    initial_cash,
    fee_rate,
    slippage_rate,
    qqq_buy_filter,
    sell_confirm_bars,
):
    cash = float(initial_cash)
    qty = 0.0
    entry_price = None
    entry_time = None
    entry_cash = None
    pending_order = None
    sell_streak = 0
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
                sell_streak = 0

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
            sell_streak = 0

        pending_order = None

        equity = cash + qty * row["Close"]
        equity_points.append((timestamp, equity))

        if i == len(df) - 1:
            continue

        if qty == 0.0:
            sell_streak = 0
            qqq_allows_buy = bool(qqq_buy_filter.get(timestamp, False))
            if bool(row["BuySignal"]) and qqq_allows_buy:
                pending_order = "BUY"
            continue

        if int(row["Trend"]) == -1:
            sell_streak += 1
        else:
            sell_streak = 0

        if sell_streak >= sell_confirm_bars:
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


def run_sell_confirm_backtest(
    sell_confirm_bars,
    qqq_buy_filter,
    data,
    common_index,
    qqq_30m,
    args,
    label=None,
):
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
        equity, trades = backtest_symbol_with_sell_confirm(
            symbol=symbol,
            df=st_df,
            initial_cash=per_symbol_cash,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            qqq_buy_filter=qqq_buy_filter,
            sell_confirm_bars=sell_confirm_bars,
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
                "Sell Confirm": sell_confirm_bars,
                "Label": label or f"{sell_confirm_bars} bars",
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
        raise RuntimeError(f"No equity series could be built for sell confirm {sell_confirm_bars}.")

    strategy_equity = pd.concat(equity_parts, axis=1, join="inner").sum(axis=1)
    strategy_equity.name = "strategy"

    benchmark_data = {symbol: df.loc[common_index].copy() for symbol, df in data.items()}
    equal_benchmark_equity = build_equal_weight_benchmark(benchmark_data, args.initial_cash)
    qqq_benchmark_equity = build_single_asset_benchmark(qqq_30m, common_index, args.initial_cash)

    report_index = strategy_equity.index.intersection(equal_benchmark_equity.index)
    report_index = report_index.intersection(qqq_benchmark_equity.index)
    strategy_equity = strategy_equity.loc[report_index]
    equal_benchmark_equity = equal_benchmark_equity.loc[report_index]
    qqq_benchmark_equity = qqq_benchmark_equity.loc[report_index]

    strategy_metrics = calculate_metrics(strategy_equity, all_report_trades, STOCK_INTERVAL)
    equal_benchmark_metrics = calculate_metrics(equal_benchmark_equity, [], STOCK_INTERVAL)
    qqq_benchmark_metrics = calculate_metrics(qqq_benchmark_equity, [], STOCK_INTERVAL)

    return {
        "label": label or f"{sell_confirm_bars} bars",
        "sell_confirm": sell_confirm_bars,
        "start": strategy_equity.index[0],
        "end": strategy_equity.index[-1],
        "symbols": len(data),
        "bars": len(strategy_equity),
        "qqq_on_ratio": qqq_buy_filter.loc[report_index].mean(),
        "strategy_metrics": strategy_metrics,
        "equal_benchmark_metrics": equal_benchmark_metrics,
        "qqq_benchmark_metrics": qqq_benchmark_metrics,
        "alpha_vs_equal": strategy_metrics["total_return"] - equal_benchmark_metrics["total_return"],
        "alpha_vs_qqq": strategy_metrics["total_return"] - qqq_benchmark_metrics["total_return"],
        "symbol_rows": symbol_rows,
    }


def format_summary_row(result):
    strategy = result["strategy_metrics"]
    equal_benchmark = result["equal_benchmark_metrics"]
    qqq_benchmark = result["qqq_benchmark_metrics"]
    return {
        "Sell Confirm": result["label"],
        "Period": f"{result['start']} -> {result['end']}",
        "QQQ On": f"{result['qqq_on_ratio'] * 100:.1f}%",
        "Symbols": result["symbols"],
        "Bars": result["bars"],
        "Strategy": format_pct(strategy["total_return"]),
        "Equal B&H": format_pct(equal_benchmark["total_return"]),
        "QQQ B&H": format_pct(qqq_benchmark["total_return"]),
        "Alpha Eq": format_pct(result["alpha_vs_equal"]),
        "Alpha QQQ": format_pct(result["alpha_vs_qqq"]),
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
    ranked["_alpha_qqq"] = [result["alpha_vs_qqq"] for result in results]
    best = ranked.sort_values("_alpha_qqq", ascending=False).iloc[0]

    print("=" * 150)
    print("Supertrend 30m US Backtest - Sell Confirmation + QQQ 1d Filter")
    print("=" * 150)
    print("Data source        : Yahoo Finance via yfinance")
    print("Universe           : universe.json / US_UNIVERSE_LIST only")
    print("Stock strategy     : 30m Supertrend")
    print("Buy filter         : QQQ previous completed daily Supertrend Trend == 1")
    print("Sell rule          : sell after N consecutive 30m bars with stock Trend == -1")
    print(f"Lookback           : {args.period}")
    print(f"Fee / Slippage     : {args.fee_rate:.4%} / {args.slippage_rate:.4%} per fill")
    if skipped:
        print(f"Skipped symbols    : {', '.join(asset.symbol for asset in skipped)}")
    if coverage_dropped:
        dropped_text = ", ".join(
            f"{symbol}({bars}/{required})" for symbol, bars, required in coverage_dropped
        )
        print(f"Dropped short data : {dropped_text}")
    print("-" * 150)
    print("Sell Confirmation Comparison")
    print(summary.drop(columns=["_alpha_qqq"], errors="ignore").to_string(index=False))
    print("-" * 150)
    print(
        "Best Alpha vs QQQ  : "
        f"{best['Sell Confirm']} | Alpha QQQ {best['Alpha QQQ']} | "
        f"Strategy {best['Strategy']} | MDD {best['MDD']} | Trades {best['Trades']}"
    )

    if args.top_symbols > 0:
        for result in results:
            print("-" * 150)
            print(f"By Symbol - Sell Confirm {result['label']} (top/bottom {args.top_symbols})")
            print(format_symbol_table(result["symbol_rows"], args.top_symbols).to_string(index=False))
    print("=" * 150)


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
        raise RuntimeError("QQQ 30m benchmark data was not downloaded.")
    if qqq_daily_skipped or QQQ_SYMBOL not in qqq_daily_data:
        raise RuntimeError("QQQ daily filter data was not downloaded.")

    stock_data, coverage_dropped = filter_by_history_coverage(
        stock_data,
        args.min_coverage,
    )
    common_index = get_common_index(stock_data)
    qqq_buy_filter = align_qqq_daily_filter(
        qqq_daily=qqq_daily_data[QQQ_SYMBOL],
        common_index=common_index,
        args=args,
    )
    no_filter = pd.Series(True, index=common_index)

    results = []
    results.append(
        run_sell_confirm_backtest(
            sell_confirm_bars=1,
            qqq_buy_filter=no_filter,
            data=stock_data,
            common_index=common_index,
            qqq_30m=qqq_30m_data[QQQ_SYMBOL],
            args=args,
            label="None",
        )
    )
    for sell_confirm_bars in SELL_CONFIRM_BARS:
        results.append(
            run_sell_confirm_backtest(
                sell_confirm_bars=sell_confirm_bars,
                qqq_buy_filter=qqq_buy_filter,
                data=stock_data,
                common_index=common_index,
                qqq_30m=qqq_30m_data[QQQ_SYMBOL],
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
