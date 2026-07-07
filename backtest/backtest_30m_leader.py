# -*- coding: utf-8 -*-
"""
1종목 리더 로테이션 실험 파일입니다.

universe.json의 미국 종목 중 30분봉 Supertrend 상승 추세 종목만 후보로 두고,
상대강도(RS = 종목 최근 N봉 수익률 - QQQ 최근 N봉 수익률)가 가장 강한 1개만 보유합니다.
None 및 QQQ 30m/1h/2h/4h/1d 매수 필터를 비교하고,
성과가 좋았던 QQQ 1d 필터에서는 매도 확인 1봉부터 지정한 최대 봉 수까지 비교합니다.
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
DEFAULT_MAX_SELL_CONFIRM_BARS = 60
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
        description=(
            "30m Supertrend leader-rotation backtest. "
            "Holds at most one strongest US ticker from universe.json."
        )
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
        "--rs-period",
        type=int,
        default=100,
        help="Relative strength lookback in 30m bars.",
    )
    parser.add_argument(
        "--hurdle-atr-mult",
        type=float,
        default=1.25,
        help="Rotate only when new leader's RS edge exceeds ATR_pct * this multiplier.",
    )
    parser.add_argument(
        "--max-sell-confirm-bars",
        type=int,
        default=DEFAULT_MAX_SELL_CONFIRM_BARS,
        help="Test QQQ 1d sell confirmation bars from 1 through this value.",
    )
    parser.add_argument(
        "--late-chase",
        dest="allow_late_chase",
        action="store_true",
        default=True,
        help="Allow buying an already-uptrend leader, not only a fresh BuySignal.",
    )
    parser.add_argument(
        "--no-late-chase",
        dest="allow_late_chase",
        action="store_false",
        help="Buy only on a fresh Supertrend BuySignal.",
    )
    parser.add_argument(
        "--hide-trades",
        action="store_true",
        help="Hide the trade-by-trade table.",
    )
    return parser.parse_args()


def build_single_asset_benchmark(df, common_index, initial_cash):
    close = df["Close"].reindex(common_index, method="ffill").dropna()
    if close.empty:
        raise ValueError("No QQQ benchmark timeline after alignment.")
    equity = close / close.iloc[0] * initial_cash
    equity.name = "qqq_buy_and_hold"
    return equity


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
    daily_trend = trend.sort_index()
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


def prepare_stock_data(stock_data, qqq_30m, common_index, args):
    qqq_close = qqq_30m["Close"].reindex(common_index, method="ffill")
    qqq_return = qqq_close.pct_change(args.rs_period)

    prepared = {}
    for symbol, df in stock_data.items():
        aligned = df.reindex(common_index).ffill()
        st_df = calculate_supertrend(
            aligned,
            period=args.atr_period,
            multiplier=args.multiplier,
            atr_method=args.atr_method,
        )
        st_df["ATR_pct"] = st_df["ATR"] / st_df["Close"]
        st_df["RS"] = st_df["Close"].pct_change(args.rs_period) - qqq_return
        prepared[symbol] = st_df

    return prepared


def portfolio_value(cash, position, prepared, timestamp):
    value = cash
    if position:
        symbol = position["symbol"]
        value += position["qty"] * prepared[symbol].loc[timestamp, "Close"]
    return value


def candidate_rows(prepared, timestamp, buy_allowed, allow_late_chase):
    if not buy_allowed:
        return []

    rows = []
    for symbol, df in prepared.items():
        row = df.loc[timestamp]
        rs_score = row["RS"]
        atr_pct = row["ATR_pct"]
        if pd.isna(rs_score) or pd.isna(atr_pct):
            continue
        if int(row["Trend"]) != 1:
            continue

        rows.append(
            {
                "symbol": symbol,
                "rs": float(rs_score),
                "atr_pct": float(atr_pct),
                "signal_buy": bool(allow_late_chase or row["BuySignal"]),
            }
        )

    return sorted(rows, key=lambda item: item["rs"], reverse=True)


def fill_price(df, exec_ts, signal_ts, side, slippage_rate):
    raw_price = df.loc[exec_ts, "Open"]
    if pd.isna(raw_price):
        raw_price = df.loc[signal_ts, "Close"]

    if side == "buy":
        return raw_price * (1.0 + slippage_rate)
    return raw_price * (1.0 - slippage_rate)


def sell_position(position, prepared, exec_ts, signal_ts, fee_rate, slippage_rate, reason):
    symbol = position["symbol"]
    price = fill_price(prepared[symbol], exec_ts, signal_ts, "sell", slippage_rate)
    cash = position["qty"] * price * (1.0 - fee_rate)
    pnl_pct = cash / position["entry_cash"] - 1.0

    trade = {
        "symbol": symbol,
        "entry_time": position["entry_time"],
        "exit_time": exec_ts,
        "entry_price": position["entry_price"],
        "exit_price": price,
        "pnl_pct": pnl_pct,
        "bars_held": position["bars_held"],
        "exit_reason": reason,
    }
    return cash, trade


def buy_position(symbol, cash, prepared, exec_ts, signal_ts, fee_rate, slippage_rate):
    price = fill_price(prepared[symbol], exec_ts, signal_ts, "buy", slippage_rate)
    if price <= 0 or cash <= 0:
        return cash, None

    entry_cash = cash
    qty = (cash * (1.0 - fee_rate)) / price
    position = {
        "symbol": symbol,
        "qty": qty,
        "entry_price": price,
        "entry_time": exec_ts,
        "entry_cash": entry_cash,
        "bars_held": 0,
        "sell_streak": 0,
    }
    return 0.0, position


def run_leader_rotation(label, buy_filter, sell_confirm_bars, prepared, common_index, args):
    cash = float(args.initial_cash)
    position = None
    trades = []
    equity_points = []
    start_i = args.rs_period + 1

    if len(common_index) <= start_i + 2:
        raise RuntimeError("Not enough bars for the selected RS period.")

    for i in range(start_i):
        equity_points.append((common_index[i], cash))

    for i in range(start_i, len(common_index) - 1):
        signal_ts = common_index[i]
        exec_ts = common_index[i + 1]

        equity_points.append((signal_ts, portfolio_value(cash, position, prepared, signal_ts)))

        if position:
            position["bars_held"] += 1

        buy_allowed = bool(buy_filter.get(signal_ts, False))
        candidates = candidate_rows(
            prepared=prepared,
            timestamp=signal_ts,
            buy_allowed=buy_allowed,
            allow_late_chase=args.allow_late_chase,
        )

        sell_reason = None
        if position:
            held_symbol = position["symbol"]
            held_row = prepared[held_symbol].loc[signal_ts]
            if int(held_row["Trend"]) == -1:
                position["sell_streak"] = position.get("sell_streak", 0) + 1
                if position["sell_streak"] >= sell_confirm_bars:
                    sell_reason = f"TrendDown{sell_confirm_bars}Bars"
            else:
                position["sell_streak"] = 0

            if sell_reason is None and candidates:
                current_rs = held_row["RS"]
                if pd.isna(current_rs):
                    current_rs = -999.0
                best_new = next(
                    (candidate for candidate in candidates if candidate["symbol"] != held_symbol),
                    None,
                )
                if best_new:
                    hurdle = best_new["atr_pct"] * args.hurdle_atr_mult
                    if best_new["rs"] - float(current_rs) > hurdle:
                        sell_reason = "Rotation"

        if position and sell_reason:
            cash, trade = sell_position(
                position=position,
                prepared=prepared,
                exec_ts=exec_ts,
                signal_ts=signal_ts,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
                reason=sell_reason,
            )
            trades.append(trade)
            position = None

        if position is None and candidates:
            for candidate in candidates:
                if not candidate["signal_buy"]:
                    continue
                cash, position = buy_position(
                    symbol=candidate["symbol"],
                    cash=cash,
                    prepared=prepared,
                    exec_ts=exec_ts,
                    signal_ts=signal_ts,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                )
                if position is not None:
                    break

    final_ts = common_index[-1]
    if position:
        cash, trade = sell_position(
            position=position,
            prepared=prepared,
            exec_ts=final_ts,
            signal_ts=final_ts,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            reason="FinalClose",
        )
        trades.append(trade)
        position = None

    equity_points.append((final_ts, cash))
    equity = pd.Series(
        [point[1] for point in equity_points],
        index=[point[0] for point in equity_points],
        name=label,
    )
    return equity, trades


def format_trades_table(trades):
    table = pd.DataFrame(trades)
    if table.empty:
        return table

    return table.assign(
        entry_time=table["entry_time"].astype(str),
        exit_time=table["exit_time"].astype(str),
        entry_price=table["entry_price"].map(lambda value: f"{value:.4f}"),
        exit_price=table["exit_price"].map(lambda value: f"{value:.4f}"),
        pnl_pct=table["pnl_pct"].map(format_pct),
    ).rename(
        columns={
            "symbol": "Symbol",
            "entry_time": "Entry",
            "exit_time": "Exit",
            "entry_price": "Entry Price",
            "exit_price": "Exit Price",
            "pnl_pct": "PnL",
            "bars_held": "Bars Held",
            "exit_reason": "Exit Reason",
        }
    )


def build_result(
    label,
    buy_filter,
    sell_confirm_bars,
    prepared,
    common_index,
    stock_data,
    qqq_30m,
    args,
):
    strategy_equity, trades = run_leader_rotation(
        label=label,
        buy_filter=buy_filter,
        sell_confirm_bars=sell_confirm_bars,
        prepared=prepared,
        common_index=common_index,
        args=args,
    )

    benchmark_data = {symbol: df.loc[common_index].copy() for symbol, df in stock_data.items()}
    equal_benchmark = build_equal_weight_benchmark(benchmark_data, args.initial_cash)
    qqq_benchmark = build_single_asset_benchmark(qqq_30m, common_index, args.initial_cash)

    report_index = strategy_equity.index.intersection(equal_benchmark.index)
    report_index = report_index.intersection(qqq_benchmark.index)
    strategy_equity = strategy_equity.loc[report_index]
    equal_benchmark = equal_benchmark.loc[report_index]
    qqq_benchmark = qqq_benchmark.loc[report_index]

    metrics = calculate_metrics(strategy_equity, trades, STOCK_INTERVAL)
    equal_metrics = calculate_metrics(equal_benchmark, [], STOCK_INTERVAL)
    qqq_metrics = calculate_metrics(qqq_benchmark, [], STOCK_INTERVAL)
    avg_bars_held = (
        sum(trade["bars_held"] for trade in trades) / len(trades)
        if trades
        else 0.0
    )

    return {
        "label": label,
        "sell_confirm_bars": sell_confirm_bars,
        "start": strategy_equity.index[0],
        "end": strategy_equity.index[-1],
        "bars": len(strategy_equity),
        "qqq_on_ratio": buy_filter.loc[report_index].mean(),
        "strategy_equity": strategy_equity,
        "metrics": metrics,
        "equal_metrics": equal_metrics,
        "qqq_metrics": qqq_metrics,
        "alpha_equal": metrics["total_return"] - equal_metrics["total_return"],
        "alpha_qqq": metrics["total_return"] - qqq_metrics["total_return"],
        "avg_bars_held": avg_bars_held,
        "trades": trades,
    }


def format_summary_row(result):
    metrics = result["metrics"]
    return {
        "Filter": result["label"],
        "Sell Confirm": f"{result['sell_confirm_bars']} bars",
        "Period": f"{result['start']} -> {result['end']}",
        "QQQ On": f"{result['qqq_on_ratio'] * 100:.1f}%",
        "Bars": result["bars"],
        "Strategy": format_pct(metrics["total_return"]),
        "Equal B&H": format_pct(result["equal_metrics"]["total_return"]),
        "QQQ B&H": format_pct(result["qqq_metrics"]["total_return"]),
        "Alpha Eq": format_pct(result["alpha_equal"]),
        "Alpha QQQ": format_pct(result["alpha_qqq"]),
        "MDD": format_pct(metrics["mdd"]),
        "Sharpe": format_float(metrics["sharpe"]),
        "Win Rate": format_pct(metrics["win_rate"]),
        "Payoff": format_float(metrics["payoff_ratio"]),
        "Trades": metrics["trade_count"],
        "Avg Hold": f"{result['avg_bars_held']:.1f}",
    }


def print_results(args, filter_results, sell_confirm_results, skipped, coverage_dropped, symbol_count):
    filter_summary = pd.DataFrame([format_summary_row(result) for result in filter_results])
    filter_ranked = filter_summary.copy()
    filter_ranked["_alpha_qqq"] = [result["alpha_qqq"] for result in filter_results]
    best_filter = filter_ranked.sort_values("_alpha_qqq", ascending=False).iloc[0]

    sell_summary = pd.DataFrame([format_summary_row(result) for result in sell_confirm_results])
    sell_ranked = sell_summary.copy()
    sell_ranked["_strategy_return"] = [
        result["metrics"]["total_return"] for result in sell_confirm_results
    ]
    best_sell = sell_ranked.sort_values("_strategy_return", ascending=False).iloc[0]

    print("=" * 160)
    print("Supertrend 30m US Leader Rotation Backtest")
    print("=" * 160)
    print("Data source        : Yahoo Finance via yfinance")
    print("Universe           : universe.json / US_UNIVERSE_LIST only")
    print("Strategy           : hold max 1 ticker with strongest RS among 30m Supertrend uptrends")
    print("Buy filters        : None, QQQ 30m, QQQ 1h, QQQ 2h, QQQ 4h, QQQ 1d")
    print(
        "Sell confirmation  : "
        f"filter comparison uses 1 bar; QQQ 1d also tests 1~{args.max_sell_confirm_bars} bars"
    )
    print("Relative strength  : stock 30m return over RS period - QQQ 30m return over same period")
    print(f"Lookback           : {args.period}")
    print(f"RS period          : {args.rs_period} bars")
    print(f"Rotation hurdle    : new leader RS edge > ATR_pct * {args.hurdle_atr_mult:.2f}")
    print(f"Late chase         : {args.allow_late_chase}")
    print(f"Fee / Slippage     : {args.fee_rate:.4%} / {args.slippage_rate:.4%} per fill")
    print(f"Downloaded symbols : {symbol_count}")
    if skipped:
        print(f"Skipped symbols    : {', '.join(asset.symbol for asset in skipped)}")
    if coverage_dropped:
        dropped_text = ", ".join(
            f"{symbol}({bars}/{required})" for symbol, bars, required in coverage_dropped
        )
        print(f"Dropped short data : {dropped_text}")

    print("-" * 160)
    print("Filter Comparison - Sell Confirm 1 Bar")
    print(filter_summary.drop(columns=["_alpha_qqq"], errors="ignore").to_string(index=False))
    print("-" * 160)
    print(
        "Best Filter Alpha  : "
        f"{best_filter['Filter']} | Alpha QQQ {best_filter['Alpha QQQ']} | "
        f"Strategy {best_filter['Strategy']} | MDD {best_filter['MDD']} | Trades {best_filter['Trades']}"
    )
    print("-" * 160)
    print("QQQ 1d Sell Confirmation Comparison")
    print(sell_summary.drop(columns=["_strategy_return"], errors="ignore").to_string(index=False))
    print("-" * 160)
    print(
        "Best Sell Return   : "
        f"{best_sell['Sell Confirm']} | Strategy {best_sell['Strategy']} | "
        f"Alpha QQQ {best_sell['Alpha QQQ']} | MDD {best_sell['MDD']} | Trades {best_sell['Trades']}"
    )

    if not args.hide_trades:
        trade_results = filter_results + [
            result
            for result in sell_confirm_results
            if not (result["label"] == "QQQ 1d" and result["sell_confirm_bars"] == 1)
        ]
        for result in trade_results:
            trades_table = format_trades_table(result["trades"])
            if trades_table.empty:
                continue
            print("-" * 160)
            print(f"Trades - {result['label']} / Sell Confirm {result['sell_confirm_bars']} bars")
            print(trades_table.to_string(index=False))
    print("=" * 160)


def main():
    args = parse_args()
    if args.max_sell_confirm_bars < 1:
        raise ValueError("--max-sell-confirm-bars must be at least 1.")

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

    stock_data, coverage_dropped = filter_by_history_coverage(stock_data, args.min_coverage)
    common_index = get_common_index(stock_data)
    prepared = prepare_stock_data(
        stock_data=stock_data,
        qqq_30m=qqq_30m_data[QQQ_SYMBOL],
        common_index=common_index,
        args=args,
    )

    qqq_filters = build_qqq_filter_series(
        qqq_30m=qqq_30m_data[QQQ_SYMBOL],
        qqq_daily=qqq_daily_data[QQQ_SYMBOL],
        common_index=common_index,
        args=args,
    )

    filter_results = []
    for filter_name in QQQ_FILTERS:
        filter_results.append(
            build_result(
                label=filter_name,
                buy_filter=qqq_filters[filter_name],
                sell_confirm_bars=1,
                prepared=prepared,
                common_index=common_index,
                stock_data=stock_data,
                qqq_30m=qqq_30m_data[QQQ_SYMBOL],
                args=args,
            )
        )

    sell_confirm_results = []
    for sell_confirm_bars in range(1, args.max_sell_confirm_bars + 1):
        sell_confirm_results.append(
            build_result(
                label="QQQ 1d",
                buy_filter=qqq_filters["QQQ 1d"],
                sell_confirm_bars=sell_confirm_bars,
                prepared=prepared,
                common_index=common_index,
                stock_data=stock_data,
                qqq_30m=qqq_30m_data[QQQ_SYMBOL],
                args=args,
            )
        )

    print_results(
        args=args,
        filter_results=filter_results,
        sell_confirm_results=sell_confirm_results,
        skipped=skipped,
        coverage_dropped=coverage_dropped,
        symbol_count=len(stock_data),
    )


if __name__ == "__main__":
    main()
