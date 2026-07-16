# -*- coding: utf-8 -*-
"""
국장 종목용 30분봉 리더 로테이션 백테스트 파일입니다.

universe.json의 한국 주식만 Yahoo Finance(.KS/.KQ)로 조회하고, 상대강도 1등 종목
하나만 보유합니다. KOSPI 종목은 ^KS11, KOSDAQ 종목은 ^KQ11 지수 Supertrend를
시장 필터로 쓰며, None/30m/1h/2h/4h/1d 필터 성과를 먼저 비교합니다.
그중 수익률이 가장 좋은 필터에 대해 매도 확인 1봉부터 지정한 최대 봉 수까지
추가 비교합니다.
"""

import argparse
import contextlib
import io
import tempfile
import warnings
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


warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_PERIOD = "60d"
STOCK_INTERVAL = "30m"
DAILY_INTERVAL = "1d"
DEFAULT_INITIAL_CASH = 10_000_000.0
DEFAULT_MAX_SELL_CONFIRM_BARS = 60
MARKET_INDEX_ASSETS = {
    "KOSPI": Asset(symbol="KOSPI", yf_symbol="^KS11", market="KOSPI"),
    "KOSDAQ": Asset(symbol="KOSDAQ", yf_symbol="^KQ11", market="KOSDAQ"),
}
MARKET_FILTERS = {
    "None": {"source": "none", "rule": None},
    "30m": {"source": "intraday", "rule": None},
    "1h": {"source": "intraday", "rule": "1h"},
    "2h": {"source": "intraday", "rule": "2h"},
    "4h": {"source": "intraday", "rule": "4h"},
    "1d": {"source": "daily", "rule": None},
}


def parse_args():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "KR 30m Supertrend leader-rotation backtest using Yahoo Finance. "
            "Holds at most one strongest ticker from universe.json KR universe."
        )
    )
    parser.add_argument("--universe", default=str(project_dir / "universe.json"))
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
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
        help="Test sell confirmation bars from 1 through this value.",
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
        df.resample(rule, closed="left", label="left", origin="start_day", offset="9h")
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


def build_market_filter_sets(index_30m_data, index_daily_data, common_index, args):
    filter_sets = {}

    for filter_name, config in MARKET_FILTERS.items():
        market_filters = {}

        for market in MARKET_INDEX_ASSETS:
            if config["source"] == "none":
                market_filters[market] = pd.Series(True, index=common_index)
                continue

            if config["source"] == "daily":
                index_df = index_daily_data[market].copy()
                aligner = align_daily_filter
            elif config["rule"] is None:
                index_df = index_30m_data[market].copy()
                aligner = align_intraday_filter
            else:
                index_df = resample_ohlc(index_30m_data[market], config["rule"])
                aligner = align_intraday_filter

            st_df = calculate_supertrend(
                index_df,
                period=args.atr_period,
                multiplier=args.multiplier,
                atr_method=args.atr_method,
            )
            market_filters[market] = aligner(st_df["Trend"], common_index)

        filter_sets[filter_name] = market_filters

    return filter_sets


def prepare_stock_data(stock_data, index_data, symbol_markets, common_index, args):
    index_returns = {}
    for market, df in index_data.items():
        close = df["Close"].reindex(common_index, method="ffill")
        index_returns[market] = close.pct_change(args.rs_period)

    prepared = {}
    for symbol, df in stock_data.items():
        aligned = df.reindex(common_index).ffill()
        st_df = calculate_supertrend(
            aligned,
            period=args.atr_period,
            multiplier=args.multiplier,
            atr_method=args.atr_method,
        )
        market = symbol_markets[symbol]
        st_df["ATR_pct"] = st_df["ATR"] / st_df["Close"]
        st_df["RS"] = st_df["Close"].pct_change(args.rs_period) - index_returns[market]
        prepared[symbol] = st_df

    return prepared


def build_market_blend_benchmark(index_data, symbol_markets, common_index, initial_cash):
    parts = []
    for symbol, market in symbol_markets.items():
        close = index_data[market]["Close"].reindex(common_index, method="ffill").dropna()
        if close.empty:
            continue
        parts.append((close / close.iloc[0]).rename(symbol))

    if not parts:
        raise ValueError("No market index benchmark timeline.")

    normalized = pd.concat(parts, axis=1, join="inner")
    benchmark = normalized.mean(axis=1) * initial_cash
    benchmark.name = "market_blend_buy_and_hold"
    return benchmark


def portfolio_value(cash, position, prepared, timestamp):
    value = cash
    if position:
        symbol = position["symbol"]
        value += position["qty"] * prepared[symbol].loc[timestamp, "Close"]
    return value


def candidate_rows(prepared, symbol_markets, market_filters, timestamp, allow_late_chase):
    rows = []
    for symbol, df in prepared.items():
        market = symbol_markets[symbol]
        if not bool(market_filters[market].get(timestamp, False)):
            continue

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
                "market": market,
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
    sale_cash = position["qty"] * price * (1.0 - fee_rate)
    pnl_pct = sale_cash / position["entry_cash"] - 1.0

    trade = {
        "symbol": symbol,
        "market": position["market"],
        "entry_time": position["entry_time"],
        "exit_time": exec_ts,
        "entry_price": position["entry_price"],
        "exit_price": price,
        "pnl_pct": pnl_pct,
        "bars_held": position["bars_held"],
        "exit_reason": reason,
    }
    return sale_cash, trade


def buy_position(symbol, cash, prepared, symbol_markets, exec_ts, signal_ts, fee_rate, slippage_rate):
    price = fill_price(prepared[symbol], exec_ts, signal_ts, "buy", slippage_rate)
    if price <= 0 or cash <= 0:
        return cash, None

    qty = int(cash // (price * (1.0 + fee_rate)))
    if qty <= 0:
        return cash, None

    entry_cash = qty * price * (1.0 + fee_rate)
    cash -= entry_cash
    position = {
        "symbol": symbol,
        "market": symbol_markets[symbol],
        "qty": qty,
        "entry_price": price,
        "entry_time": exec_ts,
        "entry_cash": entry_cash,
        "bars_held": 0,
        "sell_streak": 0,
    }
    return cash, position


def run_leader_rotation(
    sell_confirm_bars,
    prepared,
    symbol_markets,
    market_filters,
    common_index,
    args,
):
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

        candidates = candidate_rows(
            prepared=prepared,
            symbol_markets=symbol_markets,
            market_filters=market_filters,
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
            sale_cash, trade = sell_position(
                position=position,
                prepared=prepared,
                exec_ts=exec_ts,
                signal_ts=signal_ts,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
                reason=sell_reason,
            )
            cash += sale_cash
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
                    symbol_markets=symbol_markets,
                    exec_ts=exec_ts,
                    signal_ts=signal_ts,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                )
                if position is not None:
                    break

    final_ts = common_index[-1]
    if position:
        sale_cash, trade = sell_position(
            position=position,
            prepared=prepared,
            exec_ts=final_ts,
            signal_ts=final_ts,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            reason="FinalClose",
        )
        cash += sale_cash
        trades.append(trade)
        position = None

    equity_points.append((final_ts, cash))
    equity = pd.Series(
        [point[1] for point in equity_points],
        index=[point[0] for point in equity_points],
        name=f"sell_confirm_{sell_confirm_bars}",
    )
    return equity, trades


def market_filter_on_ratio(market_filters, symbol_markets, report_index):
    parts = []
    for symbol, market in symbol_markets.items():
        parts.append(market_filters[market].loc[report_index].astype(float).rename(symbol))
    return pd.concat(parts, axis=1).mean(axis=1).mean()


def build_result(
    filter_name,
    sell_confirm_bars,
    prepared,
    symbol_markets,
    market_filters,
    common_index,
    stock_data,
    index_data,
    args,
):
    strategy_equity, trades = run_leader_rotation(
        sell_confirm_bars=sell_confirm_bars,
        prepared=prepared,
        symbol_markets=symbol_markets,
        market_filters=market_filters,
        common_index=common_index,
        args=args,
    )

    benchmark_data = {symbol: df.loc[common_index].copy() for symbol, df in stock_data.items()}
    equal_benchmark = build_equal_weight_benchmark(benchmark_data, args.initial_cash)
    market_benchmark = build_market_blend_benchmark(
        index_data=index_data,
        symbol_markets=symbol_markets,
        common_index=common_index,
        initial_cash=args.initial_cash,
    )

    report_index = strategy_equity.index.intersection(equal_benchmark.index)
    report_index = report_index.intersection(market_benchmark.index)
    strategy_equity = strategy_equity.loc[report_index]
    equal_benchmark = equal_benchmark.loc[report_index]
    market_benchmark = market_benchmark.loc[report_index]

    metrics = calculate_metrics(strategy_equity, trades, STOCK_INTERVAL)
    equal_metrics = calculate_metrics(equal_benchmark, [], STOCK_INTERVAL)
    market_metrics = calculate_metrics(market_benchmark, [], STOCK_INTERVAL)
    avg_bars_held = (
        sum(trade["bars_held"] for trade in trades) / len(trades)
        if trades
        else 0.0
    )

    return {
        "filter": filter_name,
        "sell_confirm_bars": sell_confirm_bars,
        "start": strategy_equity.index[0],
        "end": strategy_equity.index[-1],
        "bars": len(strategy_equity),
        "market_on_ratio": market_filter_on_ratio(market_filters, symbol_markets, report_index),
        "strategy_equity": strategy_equity,
        "metrics": metrics,
        "equal_metrics": equal_metrics,
        "market_metrics": market_metrics,
        "alpha_equal": metrics["total_return"] - equal_metrics["total_return"],
        "alpha_market": metrics["total_return"] - market_metrics["total_return"],
        "avg_bars_held": avg_bars_held,
        "trades": trades,
    }


def format_summary_row(result):
    metrics = result["metrics"]
    return {
        "Filter": result["filter"],
        "Sell Confirm": f"{result['sell_confirm_bars']} bars",
        "Period": f"{result['start']} -> {result['end']}",
        "Market On": f"{result['market_on_ratio'] * 100:.1f}%",
        "Bars": result["bars"],
        "Strategy": format_pct(metrics["total_return"]),
        "Equal B&H": format_pct(result["equal_metrics"]["total_return"]),
        "Market B&H": format_pct(result["market_metrics"]["total_return"]),
        "Alpha Eq": format_pct(result["alpha_equal"]),
        "Alpha Mkt": format_pct(result["alpha_market"]),
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
        entry_price=table["entry_price"].map(lambda value: f"{value:.2f}"),
        exit_price=table["exit_price"].map(lambda value: f"{value:.2f}"),
        pnl_pct=table["pnl_pct"].map(format_pct),
    ).rename(
        columns={
            "symbol": "Symbol",
            "market": "Market",
            "entry_time": "Entry",
            "exit_time": "Exit",
            "entry_price": "Entry Price",
            "exit_price": "Exit Price",
            "pnl_pct": "PnL",
            "bars_held": "Bars Held",
            "exit_reason": "Exit Reason",
        }
    )


def print_results(
    args,
    filter_results,
    sell_confirm_results,
    skipped,
    coverage_dropped,
    index_skipped,
    symbol_count,
):
    filter_summary = pd.DataFrame([format_summary_row(result) for result in filter_results])
    filter_ranked = filter_summary.copy()
    filter_ranked["_strategy_return"] = [
        result["metrics"]["total_return"] for result in filter_results
    ]
    best_filter = filter_ranked.sort_values("_strategy_return", ascending=False).iloc[0]

    sell_summary = pd.DataFrame([format_summary_row(result) for result in sell_confirm_results])
    sell_ranked = sell_summary.copy()
    sell_ranked["_strategy_return"] = [
        result["metrics"]["total_return"] for result in sell_confirm_results
    ]
    best_sell = sell_ranked.sort_values("_strategy_return", ascending=False).iloc[0]

    print("=" * 160)
    print("Supertrend KR 30m Leader Rotation Backtest")
    print("=" * 160)
    print("Data source        : Yahoo Finance via yfinance")
    print("Universe           : universe.json / KR_UNIVERSE_MAP only")
    print("Bars               : 30m")
    print("Strategy           : hold max 1 ticker with strongest RS among Supertrend uptrends")
    print("Market filters     : None, 30m, 1h, 2h, 4h, 1d")
    print("Market mapping     : KOSPI stocks use ^KS11, KOSDAQ stocks use ^KQ11")
    print("Relative strength  : stock 30m return over RS period - own market index 30m return")
    print(f"Lookback           : {args.period}")
    print(f"RS period          : {args.rs_period} bars")
    print(
        "Sell confirmation  : "
        f"filter comparison uses 1 bar; best filter also tests 1~{args.max_sell_confirm_bars} bars"
    )
    print(f"Rotation hurdle    : new leader RS edge > ATR_pct * {args.hurdle_atr_mult:.2f}")
    print(f"Late chase         : {args.allow_late_chase}")
    print(f"Fee / Slippage     : {args.fee_rate:.4%} / {args.slippage_rate:.4%} per fill")
    print(f"Downloaded symbols : {symbol_count}")
    if skipped:
        print(f"Skipped symbols    : {', '.join(asset.symbol for asset in skipped)}")
    if index_skipped:
        print(f"Skipped indices    : {', '.join(asset.symbol for asset in index_skipped)}")
    if coverage_dropped:
        dropped_text = ", ".join(
            f"{symbol}({bars}/{required})" for symbol, bars, required in coverage_dropped
        )
        print(f"Dropped short data : {dropped_text}")

    print("-" * 160)
    print("Market Filter Timeframe Comparison - Sell Confirm 1 Bar")
    print(filter_summary.drop(columns=["_strategy_return"], errors="ignore").to_string(index=False))
    print("-" * 160)
    print(
        "Best Filter Return : "
        f"{best_filter['Filter']} | Strategy {best_filter['Strategy']} | "
        f"Alpha Mkt {best_filter['Alpha Mkt']} | MDD {best_filter['MDD']} | "
        f"Trades {best_filter['Trades']}"
    )
    print("-" * 160)
    print(f"{best_filter['Filter']} Filter Sell Confirmation Comparison")
    print(sell_summary.drop(columns=["_strategy_return"], errors="ignore").to_string(index=False))
    print("-" * 160)
    print(
        "Best Sell Return   : "
        f"{best_sell['Filter']} / {best_sell['Sell Confirm']} | "
        f"Strategy {best_sell['Strategy']} | Alpha Mkt {best_sell['Alpha Mkt']} | "
        f"MDD {best_sell['MDD']} | Trades {best_sell['Trades']}"
    )

    if not args.hide_trades:
        trade_results = filter_results + [
            result
            for result in sell_confirm_results
            if not (
                result["filter"] == best_filter["Filter"]
                and result["sell_confirm_bars"] == 1
            )
        ]
        for result in trade_results:
            trades_table = format_trades_table(result["trades"])
            if trades_table.empty:
                continue
            print("-" * 160)
            print(f"Trades - Filter {result['filter']} / Sell Confirm {result['sell_confirm_bars']} bars")
            print(trades_table.to_string(index=False))
    print("=" * 160)


def main():
    args = parse_args()
    if args.max_sell_confirm_bars < 1:
        raise ValueError("--max-sell-confirm-bars must be at least 1.")

    configure_yfinance_cache()

    assets = load_universe(args.universe, "kr")
    symbol_markets_all = {asset.symbol: asset.market for asset in assets}
    index_assets = list(MARKET_INDEX_ASSETS.values())

    stock_data, skipped = quiet_download_data(
        assets=assets,
        period=args.period,
        interval=STOCK_INTERVAL,
    )
    index_30m_raw, index_30m_skipped = quiet_download_data(
        assets=index_assets,
        period=args.period,
        interval=STOCK_INTERVAL,
    )
    index_daily_raw, index_daily_skipped = quiet_download_data(
        assets=index_assets,
        period=args.period,
        interval=DAILY_INTERVAL,
    )
    index_skipped = index_30m_skipped + index_daily_skipped

    if not stock_data:
        raise RuntimeError("No KR 30m stock data was downloaded from Yahoo Finance.")
    missing_30m_indices = [market for market in MARKET_INDEX_ASSETS if market not in index_30m_raw]
    missing_daily_indices = [
        market for market in MARKET_INDEX_ASSETS if market not in index_daily_raw
    ]
    if missing_30m_indices or missing_daily_indices:
        missing_text = []
        if missing_30m_indices:
            missing_text.append(
                "30m "
                + ", ".join(
                    f"{market}({MARKET_INDEX_ASSETS[market].yf_symbol})"
                    for market in missing_30m_indices
                )
            )
        if missing_daily_indices:
            missing_text.append(
                "1d "
                + ", ".join(
                    f"{market}({MARKET_INDEX_ASSETS[market].yf_symbol})"
                    for market in missing_daily_indices
                )
            )
        raise RuntimeError(
            "Missing Yahoo Finance market index data: " + " / ".join(missing_text)
        )

    stock_data, coverage_dropped = filter_by_history_coverage(stock_data, args.min_coverage)
    symbol_markets = {
        symbol: symbol_markets_all[symbol]
        for symbol in stock_data
        if symbol in symbol_markets_all
    }
    common_index = get_common_index(stock_data)
    index_data = {
        market: index_30m_raw[market].reindex(common_index).ffill()
        for market in MARKET_INDEX_ASSETS
    }
    market_filter_sets = build_market_filter_sets(
        index_30m_data=index_30m_raw,
        index_daily_data=index_daily_raw,
        common_index=common_index,
        args=args,
    )
    prepared = prepare_stock_data(
        stock_data=stock_data,
        index_data=index_data,
        symbol_markets=symbol_markets,
        common_index=common_index,
        args=args,
    )

    filter_results = []
    for filter_name in MARKET_FILTERS:
        filter_results.append(
            build_result(
                filter_name=filter_name,
                sell_confirm_bars=1,
                prepared=prepared,
                symbol_markets=symbol_markets,
                market_filters=market_filter_sets[filter_name],
                common_index=common_index,
                stock_data=stock_data,
                index_data=index_data,
                args=args,
            )
        )

    best_filter_name = max(
        filter_results,
        key=lambda result: result["metrics"]["total_return"],
    )["filter"]

    sell_confirm_results = []
    for sell_confirm_bars in range(1, args.max_sell_confirm_bars + 1):
        sell_confirm_results.append(
            build_result(
                filter_name=best_filter_name,
                sell_confirm_bars=sell_confirm_bars,
                prepared=prepared,
                symbol_markets=symbol_markets,
                market_filters=market_filter_sets[best_filter_name],
                common_index=common_index,
                stock_data=stock_data,
                index_data=index_data,
                args=args,
            )
        )

    print_results(
        args=args,
        filter_results=filter_results,
        sell_confirm_results=sell_confirm_results,
        skipped=skipped,
        coverage_dropped=coverage_dropped,
        index_skipped=index_skipped,
        symbol_count=len(stock_data),
    )


if __name__ == "__main__":
    main()
