# -*- coding: utf-8 -*-
"""
국장 종목용 순수 리더 로테이션 백테스트 파일입니다.

backtest_leader_pure.py와 같은 형태로, universe.json의 한국 주식만 Yahoo Finance
(.KS/.KQ)로 조회합니다. 시장지수 매수 필터 없이 Supertrend 상승 상태인 종목 중
최근 수익률 상대강도 1등 하나만 보유하고, 30분봉/1시간봉/2시간봉/일봉 각각에
매도 확인봉을 적용해 성과를 비교합니다.

KOSPI/KOSDAQ 지수는 매수 조건에는 쓰지 않고, Market B&H 벤치마크 계산에만 씁니다.
1시간봉/2시간봉은 국장 정규장 시작 시각인 09:00에 맞춰 30분봉을 리샘플합니다.
"""

import argparse
from pathlib import Path

import pandas as pd

from backtest_leader_pure import (
    BASE_INTRADAY_INTERVAL,
    DAILY_INTERVAL,
    TIMEFRAMES,
    configure_yfinance_cache,
    format_trades_table,
    prepare_stock_data,
    quiet_download_data,
    resolve_rs_period,
    run_leader_rotation,
)
from backtest_supertrend_universe import (
    Asset,
    DEFAULT_ATR_PERIOD,
    DEFAULT_FEE_RATE,
    DEFAULT_MULTIPLIER,
    DEFAULT_SLIPPAGE_RATE,
    build_equal_weight_benchmark,
    calculate_metrics,
    filter_by_history_coverage,
    format_float,
    format_pct,
    get_common_index,
    load_universe,
)


DEFAULT_PERIOD = "60d"
DEFAULT_INITIAL_CASH = 10_000_000.0
DEFAULT_MAX_SELL_CONFIRM_BARS = 60
KR_RESAMPLE_OFFSET = "9h"
MARKET_INDEX_ASSETS = {
    "KOSPI": Asset(symbol="KOSPI", yf_symbol="^KS11", market="KOSPI"),
    "KOSDAQ": Asset(symbol="KOSDAQ", yf_symbol="^KQ11", market="KOSDAQ"),
}


def parse_args():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Pure KR Supertrend leader-rotation backtest. "
            "Compares 30m, 1h, 2h, and 1d bars without a market filter."
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


def resample_ohlc(df, rule):
    return (
        df.resample(rule, closed="left", label="left", origin="start_day", offset=KR_RESAMPLE_OFFSET)
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


def select_timeframe_data(timeframe, stock_30m, index_30m, stock_daily, index_daily):
    config = TIMEFRAMES[timeframe]
    if config["source"] == "daily":
        stock_data = {symbol: df.copy() for symbol, df in stock_daily.items()}
        index_data = {market: df.copy() for market, df in index_daily.items()}
        return stock_data, index_data

    if config["rule"] is None:
        stock_data = {symbol: df.copy() for symbol, df in stock_30m.items()}
        index_data = {market: df.copy() for market, df in index_30m.items()}
        return stock_data, index_data

    stock_data = resample_stock_data(stock_30m, config["rule"])
    index_data = {
        market: resample_ohlc(df, config["rule"])
        for market, df in index_30m.items()
    }
    return stock_data, index_data


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


def build_timeframe_context(
    timeframe,
    stock_data_raw,
    index_data_raw,
    symbol_markets_all,
    args,
):
    stock_data, coverage_dropped = filter_by_history_coverage(
        stock_data_raw,
        args.min_coverage,
    )
    symbol_markets = {
        symbol: symbol_markets_all[symbol]
        for symbol in stock_data
        if symbol in symbol_markets_all
    }
    common_index = get_common_index(stock_data)
    index_data = {
        market: index_data_raw[market].reindex(common_index).ffill()
        for market in MARKET_INDEX_ASSETS
    }
    rs_period = resolve_rs_period(timeframe, args)
    prepared = prepare_stock_data(
        stock_data=stock_data,
        common_index=common_index,
        rs_period=rs_period,
        args=args,
    )
    benchmark_data = {
        symbol: df.loc[common_index].copy()
        for symbol, df in stock_data.items()
    }
    equal_benchmark = build_equal_weight_benchmark(
        benchmark_data,
        args.initial_cash,
    )
    market_benchmark = build_market_blend_benchmark(
        index_data=index_data,
        symbol_markets=symbol_markets,
        common_index=common_index,
        initial_cash=args.initial_cash,
    )

    return {
        "timeframe": timeframe,
        "rs_period": rs_period,
        "stock_data": stock_data,
        "coverage_dropped": coverage_dropped,
        "common_index": common_index,
        "prepared": prepared,
        "equal_benchmark": equal_benchmark,
        "market_benchmark": market_benchmark,
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
    market_benchmark = context["market_benchmark"]
    report_index = strategy_equity.index.intersection(equal_benchmark.index)
    report_index = report_index.intersection(market_benchmark.index)
    strategy_equity = strategy_equity.loc[report_index]
    equal_benchmark = equal_benchmark.loc[report_index]
    market_benchmark = market_benchmark.loc[report_index]

    interval = TIMEFRAMES[timeframe]["interval"]
    metrics = calculate_metrics(strategy_equity, trades, interval)
    equal_metrics = calculate_metrics(equal_benchmark, [], interval)
    market_metrics = calculate_metrics(market_benchmark, [], interval)
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
        "market_metrics": market_metrics,
        "alpha_equal": metrics["total_return"] - equal_metrics["total_return"],
        "alpha_market": metrics["total_return"] - market_metrics["total_return"],
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


def best_result(results):
    return max(
        results,
        key=lambda result: (
            result["metrics"]["total_return"],
            -result["sell_confirm_bars"],
        ),
    )


def print_results(
    args,
    results,
    skipped_30m,
    skipped_daily,
    index_skipped_30m,
    index_skipped_daily,
):
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
            best_by_timeframe.append(best_result(timeframe_results))
    best_by_timeframe_summary = pd.DataFrame(
        [format_summary_row(result) for result in best_by_timeframe]
    )

    print("=" * 160)
    print("Supertrend KR Pure Leader Timeframe Comparison")
    print("=" * 160)
    print("Data source        : Yahoo Finance via yfinance")
    print("Universe           : universe.json / KR_UNIVERSE_MAP only")
    print("Strategy           : hold max 1 ticker with strongest RS among Supertrend uptrends")
    print("Buy filter         : None")
    print("Timeframes         : 30m, 1h, 2h, 1d")
    print("Market benchmark   : KOSPI stocks use ^KS11, KOSDAQ stocks use ^KQ11")
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
    if index_skipped_30m:
        print(f"Skipped 30m index  : {', '.join(asset.symbol for asset in index_skipped_30m)}")
    if index_skipped_daily:
        print(f"Skipped 1d index   : {', '.join(asset.symbol for asset in index_skipped_daily)}")

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
        f"Alpha Mkt {best['Alpha Mkt']} | MDD {best['MDD']} | Trades {best['Trades']}"
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


def require_market_indices(index_data, interval_label):
    missing = [market for market in MARKET_INDEX_ASSETS if market not in index_data]
    if missing:
        raise RuntimeError(
            f"Missing Yahoo Finance {interval_label} market index data: "
            + ", ".join(
                f"{market}({MARKET_INDEX_ASSETS[market].yf_symbol})"
                for market in missing
            )
        )


def main():
    args = parse_args()
    if args.rs_period is not None and args.rs_period < 1:
        raise ValueError("--rs-period must be at least 1.")
    if args.rs_base_bars < 1:
        raise ValueError("--rs-base-bars must be at least 1.")
    if args.max_sell_confirm_bars < 1:
        raise ValueError("--max-sell-confirm-bars must be at least 1.")

    configure_yfinance_cache()

    assets = load_universe(args.universe, "kr")
    symbol_markets_all = {asset.symbol: asset.market for asset in assets}
    index_assets = list(MARKET_INDEX_ASSETS.values())

    stock_30m, skipped_30m = quiet_download_data(
        assets=assets,
        period=args.period,
        interval=BASE_INTRADAY_INTERVAL,
    )
    index_30m, index_skipped_30m = quiet_download_data(
        assets=index_assets,
        period=args.period,
        interval=BASE_INTRADAY_INTERVAL,
    )
    stock_daily, skipped_daily = quiet_download_data(
        assets=assets,
        period=args.period,
        interval=DAILY_INTERVAL,
    )
    index_daily, index_skipped_daily = quiet_download_data(
        assets=index_assets,
        period=args.period,
        interval=DAILY_INTERVAL,
    )

    if not stock_30m:
        raise RuntimeError("No KR 30m stock data was downloaded from Yahoo Finance.")
    if not stock_daily:
        raise RuntimeError("No KR daily stock data was downloaded from Yahoo Finance.")
    require_market_indices(index_30m, "30m")
    require_market_indices(index_daily, "1d")

    results = []
    for timeframe in TIMEFRAMES:
        stock_data, index_data = select_timeframe_data(
            timeframe=timeframe,
            stock_30m=stock_30m,
            index_30m=index_30m,
            stock_daily=stock_daily,
            index_daily=index_daily,
        )
        context = build_timeframe_context(
            timeframe=timeframe,
            stock_data_raw=stock_data,
            index_data_raw=index_data,
            symbol_markets_all=symbol_markets_all,
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
        index_skipped_30m=index_skipped_30m,
        index_skipped_daily=index_skipped_daily,
    )


if __name__ == "__main__":
    main()
