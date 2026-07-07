# -*- coding: utf-8 -*-
"""
Supertrend 백테스트의 공통 유틸 파일입니다.

TradingView Pine Script의 Supertrend 계산식을 Python으로 옮기고,
universe.json 종목 로드, yfinance 데이터 다운로드, 단일 종목 백테스트,
성과지표 계산과 출력 표 생성에 필요한 공통 함수를 모아둡니다.
다른 실험 파일들은 이 파일의 함수를 import해서 사용합니다.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
DEFAULT_ATR_PERIOD = 10
DEFAULT_MULTIPLIER = 3.0
DEFAULT_FEE_RATE = 0.00225
DEFAULT_SLIPPAGE_RATE = 0.0005


@dataclass(frozen=True)
class Asset:
    symbol: str
    yf_symbol: str
    market: str


def parse_args():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Backtest TradingView Supertrend signals on universe.json."
    )
    parser.add_argument("--universe", default=str(project_dir / "universe.json"))
    parser.add_argument("--market", choices=["all", "kr", "us"], default="us")
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
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
        help="Drop tickers with fewer bars than this fraction of the most complete ticker. Use 0 to keep all.",
    )
    return parser.parse_args()


def load_universe(universe_path, market):
    with open(universe_path, "r", encoding="utf-8") as f:
        universe = json.load(f)

    assets = []

    if market in ("all", "kr"):
        kr_map = universe.get("KR_UNIVERSE_MAP", {})
        for symbol, kr_market in kr_map.items():
            if kr_market == "KOSPI":
                yf_symbol = f"{symbol}.KS"
            elif kr_market == "KOSDAQ":
                yf_symbol = f"{symbol}.KQ"
            else:
                yf_symbol = symbol
            assets.append(Asset(symbol=symbol, yf_symbol=yf_symbol, market=kr_market))

    if market in ("all", "us"):
        for symbol in universe.get("US_UNIVERSE_LIST", []):
            assets.append(Asset(symbol=symbol, yf_symbol=symbol, market="US"))

    if not assets:
        raise ValueError("No tickers found in universe.json for selected market.")

    return assets


def extract_ohlc(raw_data, yf_symbol):
    if raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        if yf_symbol in raw_data.columns.get_level_values(0):
            df = raw_data[yf_symbol].copy()
        elif yf_symbol in raw_data.columns.get_level_values(1):
            df = raw_data.xs(yf_symbol, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        df = raw_data.copy()

    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    df = df[required].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required)
    return df


def download_data(assets, period, interval, start=None, end=None):
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yfinance is required to download price data. "
            "Install it with: pip install yfinance"
        ) from exc

    yf_symbols = sorted({asset.yf_symbol for asset in assets})
    download_kwargs = {
        "tickers": yf_symbols,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
        "threads": True,
        "group_by": "ticker",
    }

    if start or end:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
    else:
        download_kwargs["period"] = period

    raw_data = yf.download(**download_kwargs)

    data = {}
    skipped = []
    for asset in assets:
        df = extract_ohlc(raw_data, asset.yf_symbol)
        if df.empty:
            skipped.append(asset)
            continue
        data[asset.symbol] = df

    return data, skipped


def true_range(df):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def rma(series, length):
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)

    if len(values) < length:
        return pd.Series(out, index=series.index)

    first_idx = length - 1
    out[first_idx] = np.nanmean(values[:length])
    alpha = 1.0 / length

    for i in range(first_idx + 1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]

    return pd.Series(out, index=series.index)


def calculate_supertrend(df, period=10, multiplier=3.0, atr_method="wilder"):
    df = df.copy()
    tr = true_range(df)

    if atr_method == "sma":
        atr = tr.rolling(period, min_periods=period).mean()
    else:
        atr = rma(tr, period)

    src = (df["High"] + df["Low"]) / 2.0
    close = df["Close"]

    up = src - multiplier * atr
    dn = src + multiplier * atr
    final_up = up.copy()
    final_dn = dn.copy()
    trend = pd.Series(1, index=df.index, dtype="int64")

    for i in range(1, len(df)):
        prev_up = final_up.iloc[i - 1]
        prev_dn = final_dn.iloc[i - 1]

        up1 = prev_up if not pd.isna(prev_up) else up.iloc[i]
        dn1 = prev_dn if not pd.isna(prev_dn) else dn.iloc[i]

        if not pd.isna(up.iloc[i]) and not pd.isna(up1) and close.iloc[i - 1] > up1:
            final_up.iloc[i] = max(up.iloc[i], up1)

        if not pd.isna(dn.iloc[i]) and not pd.isna(dn1) and close.iloc[i - 1] < dn1:
            final_dn.iloc[i] = min(dn.iloc[i], dn1)

        prev_trend = trend.iloc[i - 1]
        if prev_trend == -1 and not pd.isna(dn1) and close.iloc[i] > dn1:
            trend.iloc[i] = 1
        elif prev_trend == 1 and not pd.isna(up1) and close.iloc[i] < up1:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prev_trend

    df["ATR"] = atr
    df["Supertrend_Up"] = final_up
    df["Supertrend_Down"] = final_dn
    df["Trend"] = trend
    df["BuySignal"] = (df["Trend"] == 1) & (df["Trend"].shift(1) == -1)
    df["SellSignal"] = (df["Trend"] == -1) & (df["Trend"].shift(1) == 1)
    return df


def backtest_symbol(symbol, df, initial_cash, fee_rate, slippage_rate):
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

        if qty == 0.0 and bool(row["BuySignal"]):
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


def annualization_factor(interval):
    mapping = {
        "1m": 390 * 252,
        "2m": 195 * 252,
        "5m": 78 * 252,
        "15m": 26 * 252,
        "30m": 13 * 252,
        "60m": 6.5 * 252,
        "90m": 4.33 * 252,
        "1h": 6.5 * 252,
        "2h": 3.25 * 252,
        "4h": 1.625 * 252,
        "120m": 3.25 * 252,
        "240m": 1.625 * 252,
        "1d": 252,
        "5d": 52,
        "1wk": 52,
        "1mo": 12,
        "3mo": 4,
    }
    return mapping.get(interval, 252)


def calculate_metrics(equity, trades, interval):
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    drawdown = equity / equity.cummax() - 1.0
    mdd = drawdown.min()

    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(annualization_factor(interval))
    else:
        sharpe = 0.0

    trade_pnl = pd.Series([trade["pnl_pct"] for trade in trades], dtype=float)
    if trade_pnl.empty:
        win_rate = 0.0
        payoff_ratio = 0.0
    else:
        wins = trade_pnl[trade_pnl > 0]
        losses = trade_pnl[trade_pnl <= 0]
        win_rate = len(wins) / len(trade_pnl)
        avg_win = wins.mean() if not wins.empty else 0.0
        avg_loss = abs(losses.mean()) if not losses.empty else 0.0
        payoff_ratio = float("inf") if avg_loss == 0 and avg_win > 0 else (avg_win / avg_loss if avg_loss > 0 else 0.0)

    return {
        "total_return": total_return,
        "mdd": mdd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "payoff_ratio": payoff_ratio,
        "trade_count": len(trades),
    }


def build_equal_weight_benchmark(data, initial_cash):
    close_df = pd.concat(
        [df["Close"].rename(symbol) for symbol, df in data.items()],
        axis=1,
        join="inner",
    ).dropna()

    if close_df.empty:
        raise ValueError("No common benchmark timeline. Try one market at a time or use daily bars.")

    normalized = close_df / close_df.iloc[0]
    benchmark_equity = normalized.mean(axis=1) * initial_cash
    benchmark_equity.name = "buy_and_hold"
    return benchmark_equity


def get_common_index(data):
    close_df = pd.concat(
        [df["Close"].rename(symbol) for symbol, df in data.items()],
        axis=1,
        join="inner",
    ).dropna()

    if close_df.empty:
        raise ValueError("No common timeline across downloaded tickers.")

    return close_df.index


def filter_by_history_coverage(data, min_coverage):
    if min_coverage <= 0 or not data:
        return data, []

    max_bars = max(len(df) for df in data.values())
    min_bars = math.ceil(max_bars * min_coverage)

    kept = {}
    dropped = []
    for symbol, df in data.items():
        if len(df) >= min_bars:
            kept[symbol] = df
        else:
            dropped.append((symbol, len(df), min_bars))

    if not kept:
        raise ValueError("All tickers were dropped by --min-coverage. Lower the threshold.")

    return kept, dropped


def format_pct(value):
    return f"{value * 100:+.2f}%"


def format_float(value):
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def build_display_tables(strategy_metrics, benchmark_metrics, alpha, symbol_rows, all_trades):
    summary_table = pd.DataFrame(
        [
            {
                "Name": "Supertrend",
                "Return": format_pct(strategy_metrics["total_return"]),
                "MDD": format_pct(strategy_metrics["mdd"]),
                "Sharpe": format_float(strategy_metrics["sharpe"]),
                "Win Rate": format_pct(strategy_metrics["win_rate"]),
                "Payoff": format_float(strategy_metrics["payoff_ratio"]),
                "Trades": strategy_metrics["trade_count"],
                "Alpha": format_pct(alpha),
            },
            {
                "Name": "Equal B&H",
                "Return": format_pct(benchmark_metrics["total_return"]),
                "MDD": format_pct(benchmark_metrics["mdd"]),
                "Sharpe": format_float(benchmark_metrics["sharpe"]),
                "Win Rate": "-",
                "Payoff": "-",
                "Trades": "-",
                "Alpha": format_pct(0.0),
            },
        ]
    )

    by_symbol_table = pd.DataFrame(symbol_rows).sort_values("total_return", ascending=False)
    if not by_symbol_table.empty:
        by_symbol_table = by_symbol_table.assign(
            total_return=by_symbol_table["total_return"].map(format_pct),
            mdd=by_symbol_table["mdd"].map(format_pct),
            sharpe=by_symbol_table["sharpe"].map(format_float),
            win_rate=by_symbol_table["win_rate"].map(format_pct),
            payoff_ratio=by_symbol_table["payoff_ratio"].map(format_float),
        )
        by_symbol_table = by_symbol_table.rename(
            columns={
                "symbol": "Symbol",
                "total_return": "Return",
                "mdd": "MDD",
                "sharpe": "Sharpe",
                "win_rate": "Win Rate",
                "payoff_ratio": "Payoff",
                "trade_count": "Trades",
            }
        )

    trades_table = pd.DataFrame(all_trades)
    if not trades_table.empty:
        trades_table = trades_table.assign(
            entry_time=trades_table["entry_time"].astype(str),
            exit_time=trades_table["exit_time"].astype(str),
            entry_price=trades_table["entry_price"].map(lambda value: f"{value:.4f}"),
            exit_price=trades_table["exit_price"].map(lambda value: f"{value:.4f}"),
            pnl_pct=trades_table["pnl_pct"].map(format_pct),
        )
        trades_table = trades_table.rename(
            columns={
                "symbol": "Symbol",
                "entry_time": "Entry",
                "exit_time": "Exit",
                "entry_price": "Entry Price",
                "exit_price": "Exit Price",
                "pnl_pct": "PnL",
            }
        )

    return summary_table, by_symbol_table, trades_table


def main():
    args = parse_args()
    assets = load_universe(args.universe, args.market)
    data, skipped = download_data(
        assets=assets,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )

    if not data:
        raise RuntimeError("No price data was downloaded.")

    data, coverage_dropped = filter_by_history_coverage(data, args.min_coverage)
    common_index = get_common_index(data)
    common_start = common_index[0]
    common_end = common_index[-1]
    per_symbol_cash = args.initial_cash / len(data)

    full_equities = {}
    trades_by_symbol = {}
    symbol_rows = []
    equity_parts = []

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
        full_equities[symbol] = equity
        trades_by_symbol[symbol] = trades

    all_report_trades = []
    for symbol, equity in full_equities.items():
        report_equity = equity.loc[common_index]
        if report_equity.empty or report_equity.iloc[0] <= 0:
            continue

        report_equity = (report_equity / report_equity.iloc[0]) * per_symbol_cash
        report_equity.name = symbol
        equity_parts.append(report_equity)

        report_trades = [
            trade for trade in trades_by_symbol[symbol]
            if common_start <= trade["exit_time"] <= common_end
        ]
        all_report_trades.extend(report_trades)

        symbol_metrics = calculate_metrics(report_equity, report_trades, args.interval)
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

    strategy_equity = pd.concat(equity_parts, axis=1, join="inner").sum(axis=1)
    strategy_equity.name = "strategy"
    benchmark_data = {symbol: df.loc[common_index].copy() for symbol, df in data.items()}
    benchmark_equity = build_equal_weight_benchmark(benchmark_data, args.initial_cash)

    common_index = strategy_equity.index.intersection(benchmark_equity.index)
    strategy_equity = strategy_equity.loc[common_index]
    benchmark_equity = benchmark_equity.loc[common_index]

    strategy_metrics = calculate_metrics(strategy_equity, all_report_trades, args.interval)
    benchmark_metrics = calculate_metrics(benchmark_equity, [], args.interval)
    alpha = strategy_metrics["total_return"] - benchmark_metrics["total_return"]

    summary_table, by_symbol_table, trades_table = build_display_tables(
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        alpha=alpha,
        symbol_rows=symbol_rows,
        all_trades=all_report_trades,
    )

    start_label = strategy_equity.index[0]
    end_label = strategy_equity.index[-1]

    print("=" * 78)
    print("Supertrend Universe Backtest")
    print("=" * 78)
    print(f"Universe file       : {args.universe}")
    print(f"Market             : {args.market.upper()}")
    print(f"Bars               : {args.interval}, {args.period if not args.start else f'{args.start} ~ {args.end}'}")
    print(f"Fee / Slippage     : {args.fee_rate:.4%} / {args.slippage_rate:.4%} per fill")
    print(f"Common period      : {start_label} -> {end_label}")
    print(f"Downloaded symbols : {len(data)}")
    if skipped:
        print(f"Skipped symbols    : {', '.join(asset.symbol for asset in skipped)}")
    if coverage_dropped:
        dropped_text = ", ".join(
            f"{symbol}({bars}/{required})" for symbol, bars, required in coverage_dropped
        )
        print(f"Dropped short data : {dropped_text}")
    if len(common_index) < max(args.atr_period * 3, 30):
        print(
            "Warning            : Common period is short. "
            "The selected universe can be limited by the newest ticker."
        )
    print("-" * 78)
    print("Summary")
    print(summary_table.to_string(index=False))
    print("-" * 78)
    print("By Symbol")
    print(by_symbol_table.to_string(index=False))
    if not trades_table.empty:
        print("-" * 78)
        print("Trades")
        print(trades_table.to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
