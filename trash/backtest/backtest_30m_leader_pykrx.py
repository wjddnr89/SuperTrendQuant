# -*- coding: utf-8 -*-
"""
국장 종목용 리더 로테이션 백테스트 파일입니다.

backtest_30m_leader.py의 구조를 국장 유니버스에 맞춘 버전입니다.
pykrx가 제공하는 KRX OHLCV 데이터를 사용하며, pykrx 기본 주가 데이터는 일봉 중심이라
이 파일은 실제 데이터 주기에 맞춰 1일봉으로 계산합니다.
KOSPI 종목은 KOSPI 지수 필터, KOSDAQ 종목은 KOSDAQ 지수 필터가 상승일 때만 신규 매수합니다.
pykrx 지수 조회가 실패하면 KOSPI/KOSDAQ ETF 프록시를 필터 데이터로 사용하고 출력에 표시합니다.
"""

import argparse
import contextlib
import io
import os
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from backtest_supertrend_universe import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_FEE_RATE,
    DEFAULT_MULTIPLIER,
    DEFAULT_SLIPPAGE_RATE,
    build_equal_weight_benchmark,
    calculate_metrics,
    calculate_supertrend,
    filter_by_history_coverage,
    format_float,
    format_pct,
    get_common_index,
    load_universe,
)


INTERVAL = "1d"
DEFAULT_LOOKBACK_DAYS = 730
DEFAULT_INITIAL_CASH = 10_000_000.0
DEFAULT_MAX_SELL_CONFIRM_BARS = 60
KOSPI_INDEX = "1001"
KOSDAQ_INDEX = "2001"
MARKET_INDEX = {
    "KOSPI": KOSPI_INDEX,
    "KOSDAQ": KOSDAQ_INDEX,
}
MARKET_PROXY_ETF = {
    "KOSPI": "226490",
    "KOSDAQ": "229200",
}


def parse_args():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "KR leader-rotation backtest using pykrx daily OHLCV. "
            "Holds at most one strongest ticker from universe.json KR universe."
        )
    )
    parser.add_argument("--universe", default=str(project_dir / "universe.json"))
    parser.add_argument("--start", default=None, help="Start date, e.g. 20240101 or 2024-01-01.")
    parser.add_argument("--end", default=None, help="End date, e.g. 20260709 or 2026-07-09.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Used only when --start is omitted.",
    )
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
        help="Relative strength lookback in daily bars.",
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
        "--request-sleep",
        type=float,
        default=0.05,
        help="Seconds to sleep between pykrx ticker requests.",
    )
    parser.add_argument(
        "--hide-trades",
        action="store_true",
        help="Hide the trade-by-trade table.",
    )
    return parser.parse_args()


def parse_date_arg(value):
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y%m%d")


def resolve_date_range(args):
    end = parse_date_arg(args.end) or date.today().strftime("%Y%m%d")
    if args.start:
        start = parse_date_arg(args.start)
    else:
        end_date = pd.Timestamp(end).date()
        start = (end_date - timedelta(days=args.lookback_days)).strftime("%Y%m%d")
    return start, end


def import_pykrx_stock():
    mpl_config = Path(tempfile.gettempdir()) / "trading_bot_matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from pykrx import stock
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pykrx is required for KR price data. Install it with: pip install pykrx"
        ) from exc
    return stock


def quiet_pykrx_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def normalize_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame()

    column_map = {
        "시가": "Open",
        "고가": "High",
        "저가": "Low",
        "종가": "Close",
    }
    renamed = df.rename(columns=column_map)
    required = ["Open", "High", "Low", "Close"]
    if not all(column in renamed.columns for column in required):
        return pd.DataFrame()

    out = renamed[required].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_datetime(out.index)
    return out.dropna(subset=required)


def download_kr_stock_data(assets, start, end, adjusted, request_sleep):
    stock = import_pykrx_stock()
    data = {}
    skipped = []

    for asset in assets:
        try:
            raw = quiet_pykrx_call(
                stock.get_market_ohlcv_by_date,
                start,
                end,
                asset.symbol,
                adjusted=adjusted,
            )
            df = normalize_ohlcv(raw)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            skipped.append(asset)
        else:
            data[asset.symbol] = df

        if request_sleep > 0:
            time.sleep(request_sleep)

    return data, skipped


def download_index_data(start, end):
    stock = import_pykrx_stock()
    index_data = {}
    index_sources = {}

    for market, index_code in MARKET_INDEX.items():
        source = f"Index {index_code}"
        try:
            raw = quiet_pykrx_call(
                stock.get_index_ohlcv_by_date,
                start,
                end,
                index_code,
                name_display=False,
            )
            df = normalize_ohlcv(raw)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            proxy_ticker = MARKET_PROXY_ETF[market]
            source = f"ETF proxy {proxy_ticker}"
            raw = quiet_pykrx_call(
                stock.get_market_ohlcv_by_date,
                start,
                end,
                proxy_ticker,
                adjusted=True,
            )
            df = normalize_ohlcv(raw)

        if df.empty:
            raise RuntimeError(f"No {market} index data was downloaded from pykrx.")
        index_data[market] = df
        index_sources[market] = source

    return index_data, index_sources


def build_market_filter_series(index_data, common_index, args):
    filters = {}
    for market, df in index_data.items():
        st_df = calculate_supertrend(
            df.copy(),
            period=args.atr_period,
            multiplier=args.multiplier,
            atr_method=args.atr_method,
        )
        trend = st_df["Trend"].sort_index().reindex(common_index, method="ffill")
        filters[market] = trend.fillna(-1).astype(int).eq(1)
    return filters


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

    metrics = calculate_metrics(strategy_equity, trades, INTERVAL)
    equal_metrics = calculate_metrics(equal_benchmark, [], INTERVAL)
    market_metrics = calculate_metrics(market_benchmark, [], INTERVAL)
    avg_bars_held = (
        sum(trade["bars_held"] for trade in trades) / len(trades)
        if trades
        else 0.0
    )

    return {
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
        "Sell Confirm": f"{result['sell_confirm_bars']} bars",
        "Period": f"{result['start'].date()} -> {result['end'].date()}",
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


def print_results(args, results, skipped, coverage_dropped, start, end, symbol_count, index_sources):
    summary = pd.DataFrame([format_summary_row(result) for result in results])
    ranked = summary.copy()
    ranked["_strategy_return"] = [result["metrics"]["total_return"] for result in results]
    best = ranked.sort_values("_strategy_return", ascending=False).iloc[0]

    print("=" * 160)
    print("Supertrend KR Leader Rotation Backtest")
    print("=" * 160)
    print("Data source        : KRX via pykrx")
    print("Universe           : universe.json / KR_UNIVERSE_MAP only")
    print("Bars               : 1d (pykrx OHLCV is daily in this script)")
    print("Strategy           : hold max 1 ticker with strongest RS among Supertrend uptrends")
    print("Market filters     : KOSPI stocks use KOSPI index, KOSDAQ stocks use KOSDAQ index")
    print(
        "Filter data        : "
        f"KOSPI {index_sources.get('KOSPI', '-')}, "
        f"KOSDAQ {index_sources.get('KOSDAQ', '-')}"
    )
    print("Relative strength  : stock return over RS period - own market index return")
    print(f"Date range         : {start} -> {end}")
    print(f"RS period          : {args.rs_period} bars")
    print(f"Sell confirmation  : 1~{args.max_sell_confirm_bars} bars")
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
    print("KOSPI/KOSDAQ Filter + Sell Confirmation Comparison")
    print(summary.drop(columns=["_strategy_return"], errors="ignore").to_string(index=False))
    print("-" * 160)
    print(
        "Best Sell Return   : "
        f"{best['Sell Confirm']} | Strategy {best['Strategy']} | "
        f"Alpha Mkt {best['Alpha Mkt']} | MDD {best['MDD']} | Trades {best['Trades']}"
    )

    if not args.hide_trades:
        for result in results:
            trades_table = format_trades_table(result["trades"])
            if trades_table.empty:
                continue
            print("-" * 160)
            print(f"Trades - Sell Confirm {result['sell_confirm_bars']} bars")
            print(trades_table.to_string(index=False))
    print("=" * 160)


def main():
    args = parse_args()
    if args.max_sell_confirm_bars < 1:
        raise ValueError("--max-sell-confirm-bars must be at least 1.")

    start, end = resolve_date_range(args)
    assets = load_universe(args.universe, "kr")
    symbol_markets_all = {asset.symbol: asset.market for asset in assets}

    stock_data, skipped = download_kr_stock_data(
        assets=assets,
        start=start,
        end=end,
        adjusted=True,
        request_sleep=args.request_sleep,
    )
    if not stock_data:
        raise RuntimeError("No KR stock data was downloaded from pykrx.")

    stock_data, coverage_dropped = filter_by_history_coverage(stock_data, args.min_coverage)
    symbol_markets = {
        symbol: symbol_markets_all[symbol]
        for symbol in stock_data
        if symbol in symbol_markets_all
    }
    common_index = get_common_index(stock_data)

    index_data, index_sources = download_index_data(start, end)
    market_filters = build_market_filter_series(
        index_data=index_data,
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

    results = []
    for sell_confirm_bars in range(1, args.max_sell_confirm_bars + 1):
        results.append(
            build_result(
                sell_confirm_bars=sell_confirm_bars,
                prepared=prepared,
                symbol_markets=symbol_markets,
                market_filters=market_filters,
                common_index=common_index,
                stock_data=stock_data,
                index_data=index_data,
                args=args,
            )
        )

    print_results(
        args=args,
        results=results,
        skipped=skipped,
        coverage_dropped=coverage_dropped,
        start=start,
        end=end,
        symbol_count=len(stock_data),
        index_sources=index_sources,
    )


if __name__ == "__main__":
    main()
