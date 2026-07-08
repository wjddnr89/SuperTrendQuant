# -*- coding: utf-8 -*-
"""
미국 종목용 순수 리더 로테이션 백테스트 파일입니다.

universe.json의 미국 주식만 Yahoo Finance로 조회하고, 시장지수/QQQ 매수 필터 없이
Supertrend 상승 상태인 종목 중 최근 수익률 상대강도 1등 하나만 보유합니다.
30분봉, 1시간봉, 2시간봉, 일봉 각각에 매도 확인봉을 적용해 성과를 비교합니다.
QQQ는 매매 조건이 아니라 성과 비교용 Buy & Hold 벤치마크로만 출력합니다.
"""

import argparse
import contextlib
import io
import math
import tempfile
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
BASE_INTRADAY_INTERVAL = "30m"
DAILY_INTERVAL = "1d"
QQQ_SYMBOL = "QQQ"
DEFAULT_MAX_SELL_CONFIRM_BARS = 60
TIMEFRAMES = {
    "30m": {"source": "intraday", "rule": None, "interval": "30m", "scale": 1},
    "1h": {"source": "intraday", "rule": "1h", "interval": "1h", "scale": 2},
    "2h": {"source": "intraday", "rule": "2h", "interval": "2h", "scale": 4},
    "1d": {"source": "daily", "rule": None, "interval": "1d", "scale": 13},
}


def parse_args():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Pure Supertrend leader-rotation backtest. "
            "Compares 30m, 1h, 2h, and 1d bars without a market filter."
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
        "--rs-base-bars",
        type=int,
        default=100,
        help="Base RS lookback in 30m bars. Other timeframes are scaled from this.",
    )
    parser.add_argument(
        "--rs-period",
        type=int,
        default=None,
        help="Optional override: use this same RS lookback bar count for every timeframe.",
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
        help="Test sell confirmation bars from 1 through this value for each timeframe.",
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
        help="Hide the trade-by-trade tables.",
    )
    return parser.parse_args()


def configure_yfinance_cache():
    try:
        import yfinance as yf
    except ModuleNotFoundError:
        return

    cache_dir = Path(tempfile.gettempdir()) / "trading_bot_yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def quiet_download_data(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return download_data(*args, **kwargs)


def resample_ohlc(df, rule):
    return (
        df.resample(rule, closed="left", label="left", origin="start_day", offset="9h30min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna(subset=["Open", "High", "Low", "Close"])
    )


def resample_stock_data(stock_data, rule):
    resampled = {}
    for symbol, df in stock_data.items():
        out = resample_ohlc(df, rule)
        if not out.empty:
            resampled[symbol] = out
    return resampled


def resolve_rs_period(timeframe, args):
    if args.rs_period is not None:
        return args.rs_period
    return max(2, math.ceil(args.rs_base_bars / TIMEFRAMES[timeframe]["scale"]))


def build_single_asset_benchmark(df, common_index, initial_cash):
    close = df["Close"].reindex(common_index, method="ffill").dropna()
    if close.empty:
        raise ValueError("No QQQ benchmark timeline after alignment.")
    equity = close / close.iloc[0] * initial_cash
    equity.name = "qqq_buy_and_hold"
    return equity


def prepare_stock_data(stock_data, common_index, rs_period, args):
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
        st_df["RS"] = st_df["Close"].pct_change(rs_period)
        prepared[symbol] = st_df

    return prepared


def portfolio_value(cash, position, prepared, timestamp):
    value = cash
    if position:
        symbol = position["symbol"]
        value += position["qty"] * prepared[symbol].loc[timestamp, "Close"]
    return value


def candidate_rows(prepared, timestamp, allow_late_chase):
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


def run_leader_rotation(sell_confirm_bars, prepared, common_index, rs_period, args):
    cash = float(args.initial_cash)
    position = None
    trades = []
    equity_points = []
    start_i = rs_period + 1

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

        candidates = candidate_rows(
            prepared=prepared,
            timestamp=signal_ts,
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
        name=f"sell_confirm_{sell_confirm_bars}",
    )
    return equity, trades


def select_timeframe_data(timeframe, stock_30m, qqq_30m, stock_daily, qqq_daily):
    config = TIMEFRAMES[timeframe]
    if config["source"] == "daily":
        return {symbol: df.copy() for symbol, df in stock_daily.items()}, qqq_daily.copy()
    if config["rule"] is None:
        return {symbol: df.copy() for symbol, df in stock_30m.items()}, qqq_30m.copy()
    return resample_stock_data(stock_30m, config["rule"]), resample_ohlc(qqq_30m, config["rule"])


def build_timeframe_context(timeframe, stock_data_raw, qqq_df, args):
    stock_data, coverage_dropped = filter_by_history_coverage(stock_data_raw, args.min_coverage)
    common_index = get_common_index(stock_data)
    rs_period = resolve_rs_period(timeframe, args)
    prepared = prepare_stock_data(
        stock_data=stock_data,
        common_index=common_index,
        rs_period=rs_period,
        args=args,
    )
    benchmark_data = {symbol: df.loc[common_index].copy() for symbol, df in stock_data.items()}
    equal_benchmark = build_equal_weight_benchmark(benchmark_data, args.initial_cash)
    qqq_benchmark = build_single_asset_benchmark(qqq_df, common_index, args.initial_cash)

    return {
        "timeframe": timeframe,
        "rs_period": rs_period,
        "stock_data": stock_data,
        "coverage_dropped": coverage_dropped,
        "common_index": common_index,
        "prepared": prepared,
        "equal_benchmark": equal_benchmark,
        "qqq_benchmark": qqq_benchmark,
    }


def build_result(context, sell_confirm_bars, args):
    timeframe = context["timeframe"]
    common_index = context["common_index"]
    strategy_equity, trades = run_leader_rotation(
        sell_confirm_bars=sell_confirm_bars,
        prepared=context["prepared"],
        common_index=common_index,
        rs_period=context["rs_period"],
        args=args,
    )

    equal_benchmark = context["equal_benchmark"]
    qqq_benchmark = context["qqq_benchmark"]
    report_index = strategy_equity.index.intersection(equal_benchmark.index)
    report_index = report_index.intersection(qqq_benchmark.index)
    strategy_equity = strategy_equity.loc[report_index]
    equal_benchmark = equal_benchmark.loc[report_index]
    qqq_benchmark = qqq_benchmark.loc[report_index]

    interval = TIMEFRAMES[timeframe]["interval"]
    metrics = calculate_metrics(strategy_equity, trades, interval)
    equal_metrics = calculate_metrics(equal_benchmark, [], interval)
    qqq_metrics = calculate_metrics(qqq_benchmark, [], interval)
    avg_bars_held = (
        sum(trade["bars_held"] for trade in trades) / len(trades)
        if trades
        else 0.0
    )

    return {
        "timeframe": timeframe,
        "sell_confirm_bars": sell_confirm_bars,
        "rs_period": context["rs_period"],
        "start": strategy_equity.index[0],
        "end": strategy_equity.index[-1],
        "bars": len(strategy_equity),
        "symbol_count": len(context["stock_data"]),
        "coverage_dropped": context["coverage_dropped"],
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
        "Timeframe": result["timeframe"],
        "Sell Confirm": f"{result['sell_confirm_bars']} bars",
        "Period": f"{result['start']} -> {result['end']}",
        "RS Bars": result["rs_period"],
        "Symbols": result["symbol_count"],
        "Dropped": len(result["coverage_dropped"]),
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


def print_results(args, results, skipped_30m, skipped_daily, qqq_skipped):
    summary = pd.DataFrame([format_summary_row(result) for result in results])
    ranked = summary.copy()
    ranked["_strategy_return"] = [result["metrics"]["total_return"] for result in results]
    ranked["_sell_confirm_bars"] = [result["sell_confirm_bars"] for result in results]
    best = ranked.sort_values(
        ["_strategy_return", "_sell_confirm_bars"],
        ascending=[False, True],
    ).iloc[0]
    best_by_timeframe = []
    for timeframe in TIMEFRAMES:
        timeframe_results = [
            result for result in results if result["timeframe"] == timeframe
        ]
        if timeframe_results:
            best_by_timeframe.append(
                max(timeframe_results, key=lambda result: result["metrics"]["total_return"])
            )
    best_by_timeframe_summary = pd.DataFrame(
        [format_summary_row(result) for result in best_by_timeframe]
    )

    print("=" * 160)
    print("Supertrend US Pure Leader Timeframe Comparison")
    print("=" * 160)
    print("Data source        : Yahoo Finance via yfinance")
    print("Universe           : universe.json / US_UNIVERSE_LIST only")
    print("Strategy           : hold max 1 ticker with strongest RS among Supertrend uptrends")
    print("Buy filter         : None")
    print("Timeframes         : 30m, 1h, 2h, 1d")
    print("Exit rule          : sell after confirmed Supertrend down signal or stronger leader rotation")
    print(f"Lookback           : {args.period}")
    print(f"RS base            : {args.rs_base_bars} bars on 30m")
    if args.rs_period is not None:
        print(f"RS override        : {args.rs_period} bars on every timeframe")
    print(f"Sell confirmation  : 1~{args.max_sell_confirm_bars} bars for each timeframe")
    print(f"Rotation hurdle    : new leader RS edge > ATR_pct * {args.hurdle_atr_mult:.2f}")
    print(f"Late chase         : {args.allow_late_chase}")
    print(f"Fee / Slippage     : {args.fee_rate:.4%} / {args.slippage_rate:.4%} per fill")
    if skipped_30m:
        print(f"Skipped 30m data   : {', '.join(asset.symbol for asset in skipped_30m)}")
    if skipped_daily:
        print(f"Skipped 1d data    : {', '.join(asset.symbol for asset in skipped_daily)}")
    if qqq_skipped:
        print(f"Skipped benchmark  : {', '.join(asset.symbol for asset in qqq_skipped)}")

    dropped_parts = []
    for timeframe in TIMEFRAMES:
        result = next((item for item in results if item["timeframe"] == timeframe), None)
        if result is None or not result["coverage_dropped"]:
            continue
        dropped_text = ", ".join(
            f"{symbol}({bars}/{required})"
            for symbol, bars, required in result["coverage_dropped"]
        )
        dropped_parts.append(f"{result['timeframe']}: {dropped_text}")
    if dropped_parts:
        print(f"Dropped short data : {' | '.join(dropped_parts)}")

    print("-" * 160)
    print("Best Sell Confirmation By Timeframe")
    print(best_by_timeframe_summary.to_string(index=False))
    print("-" * 160)
    print("All Timeframe / Sell Confirmation Results")
    print(
        summary.drop(
            columns=["_strategy_return", "_sell_confirm_bars"],
            errors="ignore",
        ).to_string(index=False)
    )
    print("-" * 160)
    print(
        "Best Return        : "
        f"{best['Timeframe']} / {best['Sell Confirm']} | Strategy {best['Strategy']} | "
        f"Alpha QQQ {best['Alpha QQQ']} | MDD {best['MDD']} | Trades {best['Trades']}"
    )

    if not args.hide_trades:
        for result in results:
            trades_table = format_trades_table(result["trades"])
            if trades_table.empty:
                continue
            print("-" * 160)
            print(f"Trades - {result['timeframe']} / Sell Confirm {result['sell_confirm_bars']} bars")
            print(trades_table.to_string(index=False))
    print("=" * 160)


def main():
    args = parse_args()
    if args.rs_period is not None and args.rs_period < 1:
        raise ValueError("--rs-period must be at least 1.")
    if args.rs_base_bars < 1:
        raise ValueError("--rs-base-bars must be at least 1.")
    if args.max_sell_confirm_bars < 1:
        raise ValueError("--max-sell-confirm-bars must be at least 1.")

    configure_yfinance_cache()

    assets = load_universe(args.universe, "us")
    qqq_asset = Asset(symbol=QQQ_SYMBOL, yf_symbol=QQQ_SYMBOL, market="US")

    stock_30m, skipped_30m = quiet_download_data(
        assets=assets,
        period=args.period,
        interval=BASE_INTRADAY_INTERVAL,
    )
    qqq_30m_data, qqq_30m_skipped = quiet_download_data(
        assets=[qqq_asset],
        period=args.period,
        interval=BASE_INTRADAY_INTERVAL,
    )
    stock_daily, skipped_daily = quiet_download_data(
        assets=assets,
        period=args.period,
        interval=DAILY_INTERVAL,
    )
    qqq_daily_data, qqq_daily_skipped = quiet_download_data(
        assets=[qqq_asset],
        period=args.period,
        interval=DAILY_INTERVAL,
    )
    qqq_skipped = qqq_30m_skipped + qqq_daily_skipped

    if not stock_30m:
        raise RuntimeError("No stock 30m data was downloaded.")
    if not stock_daily:
        raise RuntimeError("No stock daily data was downloaded.")
    if qqq_30m_skipped or QQQ_SYMBOL not in qqq_30m_data:
        raise RuntimeError("QQQ 30m benchmark data was not downloaded.")
    if qqq_daily_skipped or QQQ_SYMBOL not in qqq_daily_data:
        raise RuntimeError("QQQ daily benchmark data was not downloaded.")

    results = []
    for timeframe in TIMEFRAMES:
        stock_data, qqq_df = select_timeframe_data(
            timeframe=timeframe,
            stock_30m=stock_30m,
            qqq_30m=qqq_30m_data[QQQ_SYMBOL],
            stock_daily=stock_daily,
            qqq_daily=qqq_daily_data[QQQ_SYMBOL],
        )
        context = build_timeframe_context(
            timeframe=timeframe,
            stock_data_raw=stock_data,
            qqq_df=qqq_df,
            args=args,
        )
        for sell_confirm_bars in range(1, args.max_sell_confirm_bars + 1):
            results.append(
                build_result(
                    context=context,
                    sell_confirm_bars=sell_confirm_bars,
                    args=args,
                )
            )

    print_results(
        args=args,
        results=results,
        skipped_30m=skipped_30m,
        skipped_daily=skipped_daily,
        qqq_skipped=qqq_skipped,
    )


if __name__ == "__main__":
    main()
