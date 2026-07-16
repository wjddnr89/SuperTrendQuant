# -*- coding: utf-8 -*-
"""
Config-driven strategy engine for modular SuperTrend experiments.

The engine prepares indicator features, applies optional market and asset
filters, runs a one-position leader-rotation backtest, splits results into
train/validation/test segments, and attaches stable B&H benchmarks.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from module.benchmarks import build_benchmark_report
from module.config import (
    TIMEFRAMES,
    StrategyConfig,
    asset_filter_list,
    market_filter_timeframe,
    normalize_signal,
    resolve_rs_period,
    validate_config,
)
from module.data import (
    MarketDataBundle,
    filter_by_history_coverage,
    get_common_index,
    select_timeframe_data,
)
from module.indicators import (
    add_strategy_features,
    calculate_supertrend,
    entry_state,
    exit_state,
)
from module.metrics import calculate_metrics, score_metrics


@dataclass
class PreparedContext:
    config: StrategyConfig
    rs_period: int
    interval: str
    stock_data: Dict[str, pd.DataFrame]
    prepared: Dict[str, pd.DataFrame]
    common_index: pd.Index
    market_filter_ok: Dict[str, pd.Series]
    coverage_dropped: List[Tuple[str, int, int]]


@dataclass
class BacktestResult:
    config: StrategyConfig
    split: str
    start: object
    end: object
    bars: int
    symbols: int
    rs_period: int
    equity: pd.Series
    trades: List[dict]
    metrics: Dict[str, float]
    benchmarks: Dict[str, dict]
    score: float
    coverage_dropped: List[Tuple[str, int, int]]

    def flat_config(self) -> Dict[str, object]:
        data = asdict(self.config)
        data["rs_period_resolved"] = self.rs_period
        return data


def align_series_to_index(series: pd.Series, target_index: pd.Index) -> pd.Series:
    series = series.dropna().sort_index()
    if series.empty:
        return pd.Series(False, index=target_index)

    try:
        aligned = series.reindex(target_index, method="ffill")
        return aligned.fillna(False)
    except (TypeError, ValueError):
        source = series.copy()
        source.index = pd.to_datetime([pd.Timestamp(item).date() for item in source.index])
        source = source.groupby(level=0).last().sort_index()
        target_dates = pd.to_datetime([pd.Timestamp(item).date() for item in target_index])
        aligned = source.reindex(target_dates, method="ffill")
        return pd.Series(aligned.to_numpy(), index=target_index).fillna(False)


def build_market_filter(
    bundle: MarketDataBundle,
    config: StrategyConfig,
    common_index: pd.Index,
    symbols: Iterable[str],
) -> Dict[str, pd.Series]:
    filter_tf = market_filter_timeframe(config.market_filter)
    if filter_tf is None:
        return {
            symbol: pd.Series(True, index=common_index)
            for symbol in symbols
        }

    _, index_data = select_timeframe_data(bundle, filter_tf)
    index_trends = {}
    for index_symbol, df in index_data.items():
        st = calculate_supertrend(
            df.reindex(df.index).ffill(),
            period=config.st_period,
            multiplier=config.st_multiplier,
            atr_method=config.atr_method,
        )
        index_trends[index_symbol] = align_series_to_index(st["Trend"].eq(1), common_index)

    filters = {}
    if bundle.market == "us":
        if "QQQ" not in index_trends:
            raise RuntimeError("QQQ data is required for the US market filter.")
        for symbol in symbols:
            filters[symbol] = index_trends["QQQ"]
        return filters

    for symbol in symbols:
        market = bundle.symbol_markets.get(symbol)
        if market not in index_trends:
            raise RuntimeError(f"{market} index data is required for {symbol}.")
        filters[symbol] = index_trends[market]
    return filters


def build_prepared_context(bundle: MarketDataBundle, config: StrategyConfig) -> PreparedContext:
    validate_config(config)
    stock_raw, _ = select_timeframe_data(bundle, config.timeframe)
    stock_data, coverage_dropped = filter_by_history_coverage(stock_raw, config.min_coverage)
    common_index = get_common_index(stock_data)
    rs_period = resolve_rs_period(config)

    prepared = {}
    for symbol, df in stock_data.items():
        aligned = df.reindex(common_index).ffill()
        prepared[symbol] = add_strategy_features(aligned, config, rs_period)

    market_filter_ok = build_market_filter(bundle, config, common_index, prepared.keys())
    return PreparedContext(
        config=config,
        rs_period=rs_period,
        interval=TIMEFRAMES[config.timeframe]["interval"],
        stock_data=stock_data,
        prepared=prepared,
        common_index=common_index,
        market_filter_ok=market_filter_ok,
        coverage_dropped=coverage_dropped,
    )


def fill_price(
    df: pd.DataFrame,
    exec_ts,
    signal_ts,
    side: str,
    slippage_rate: float,
) -> float:
    raw_price = df.loc[exec_ts, "Open"]
    if pd.isna(raw_price):
        raw_price = df.loc[signal_ts, "Close"]
    if side == "buy":
        return float(raw_price) * (1.0 + slippage_rate)
    return float(raw_price) * (1.0 - slippage_rate)


def portfolio_value(cash: float, position: Optional[dict], prepared: Dict[str, pd.DataFrame], ts) -> float:
    if not position:
        return float(cash)
    symbol = position["symbol"]
    return float(cash) + position["qty"] * float(prepared[symbol].loc[ts, "Close"])


def buy_position(
    symbol: str,
    cash: float,
    context: PreparedContext,
    exec_ts,
    signal_ts,
) -> Tuple[float, Optional[dict]]:
    config = context.config
    price = fill_price(context.prepared[symbol], exec_ts, signal_ts, "buy", config.slippage_rate)
    if price <= 0 or cash <= 0:
        return cash, None

    entry_cash = cash * config.allocation_pct
    if entry_cash <= 0:
        return cash, None

    qty = (entry_cash * (1.0 - config.fee_rate)) / price
    remaining_cash = cash - entry_cash
    return remaining_cash, {
        "symbol": symbol,
        "qty": qty,
        "entry_price": price,
        "entry_time": exec_ts,
        "entry_cash": entry_cash,
        "bars_held": 0,
        "sell_streak": 0,
    }


def sell_position(
    position: dict,
    context: PreparedContext,
    exec_ts,
    signal_ts,
    reason: str,
) -> Tuple[float, dict]:
    config = context.config
    symbol = position["symbol"]
    price = fill_price(context.prepared[symbol], exec_ts, signal_ts, "sell", config.slippage_rate)
    cash = position["qty"] * price * (1.0 - config.fee_rate)
    pnl_pct = cash / position["entry_cash"] - 1.0
    return cash, {
        "symbol": symbol,
        "entry_time": position["entry_time"],
        "exit_time": exec_ts,
        "entry_price": position["entry_price"],
        "exit_price": price,
        "pnl_pct": pnl_pct,
        "bars_held": position["bars_held"],
        "exit_reason": reason,
    }


def asset_filter_ok(row: pd.Series, config: StrategyConfig) -> bool:
    for asset_filter in asset_filter_list(config.asset_filter):
        if asset_filter == "none":
            continue
        if asset_filter == "ichimoku_cloud" and not bool(row.get("Ichimoku_LongOk", False)):
            return False
        if asset_filter == "ema_trend" and not bool(row.get("EMA_LongOk", False)):
            return False
    return True


def candidate_rows(context: PreparedContext, timestamp) -> List[dict]:
    config = context.config
    rows = []
    for symbol, df in context.prepared.items():
        row = df.loc[timestamp]
        if not bool(context.market_filter_ok[symbol].loc[timestamp]):
            continue
        if not asset_filter_ok(row, config):
            continue
        if not entry_state(row, config):
            continue

        if config.selector == "leader_top1":
            rs_score = row["RS"]
            atr_pct = row["ATR_pct"]
            if pd.isna(rs_score) or pd.isna(atr_pct):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "rs": float(rs_score),
                    "atr_pct": float(atr_pct),
                    "sort_key": float(rs_score),
                }
            )
        else:
            rows.append(
                {
                    "symbol": symbol,
                    "rs": 0.0,
                    "atr_pct": float(row["ATR_pct"]) if not pd.isna(row["ATR_pct"]) else 0.0,
                    "sort_key": 0.0,
                }
            )

    if config.selector == "leader_top1":
        return sorted(rows, key=lambda item: item["sort_key"], reverse=True)
    return sorted(rows, key=lambda item: item["symbol"])


def run_backtest(
    context: PreparedContext,
    run_index: Optional[pd.Index] = None,
) -> Tuple[pd.Series, List[dict]]:
    config = context.config
    index = pd.Index(run_index if run_index is not None else context.common_index)
    if len(index) < 3:
        raise RuntimeError("Not enough bars to run this segment.")

    cash = float(config.initial_cash)
    position = None
    trades: List[dict] = []
    equity_points = []

    for i in range(0, len(index) - 1):
        signal_ts = index[i]
        exec_ts = index[i + 1]
        equity_points.append((signal_ts, portfolio_value(cash, position, context.prepared, signal_ts)))

        if position:
            position["bars_held"] += 1

        candidates = candidate_rows(context, signal_ts)
        sell_reason = None
        if position:
            held_symbol = position["symbol"]
            held_row = context.prepared[held_symbol].loc[signal_ts]
            if exit_state(held_row, config):
                position["sell_streak"] = position.get("sell_streak", 0) + 1
                if position["sell_streak"] >= config.sell_confirm_bars:
                    sell_reason = f"SignalDown{config.sell_confirm_bars}Bars"
            else:
                position["sell_streak"] = 0

            if (
                sell_reason is None
                and config.selector == "leader_top1"
                and config.rotation_enabled
                and candidates
            ):
                current_rs = held_row["RS"]
                if pd.isna(current_rs):
                    current_rs = -999.0
                best_new = next(
                    (candidate for candidate in candidates if candidate["symbol"] != held_symbol),
                    None,
                )
                if best_new:
                    hurdle = best_new["atr_pct"] * config.hurdle_atr_mult
                    if best_new["rs"] - float(current_rs) > hurdle:
                        profit_pct = (
                            (float(held_row["Close"]) - position["entry_price"]) / position["entry_price"]
                            if position["entry_price"] > 0
                            else 0.0
                        )
                        if profit_pct >= config.min_rotation_profit_pct:
                            sell_reason = "Rotation"

        if position and sell_reason:
            proceeds, trade = sell_position(position, context, exec_ts, signal_ts, sell_reason)
            cash += proceeds
            trades.append(trade)
            position = None

        if position is None and candidates:
            for candidate in candidates:
                cash, position = buy_position(candidate["symbol"], cash, context, exec_ts, signal_ts)
                if position is not None:
                    break

    final_ts = index[-1]
    if position:
        proceeds, trade = sell_position(position, context, final_ts, final_ts, "FinalClose")
        cash += proceeds
        trades.append(trade)
        position = None

    equity_points.append((final_ts, cash))
    equity = pd.Series(
        [point[1] for point in equity_points],
        index=[point[0] for point in equity_points],
        name="strategy_equity",
    )
    return equity, trades


def split_index(index: pd.Index, train_ratio: float, validation_ratio: float) -> Dict[str, pd.Index]:
    n = len(index)
    if n < 12:
        return {"overall": index}

    train_end = max(3, int(n * train_ratio))
    validation_end = max(train_end + 3, int(n * (train_ratio + validation_ratio)))
    if n - validation_end < 3:
        validation_end = n - 3
    if validation_end <= train_end:
        return {"overall": index}

    return {
        "overall": index,
        "train": index[:train_end],
        "validation": index[train_end:validation_end],
        "test": index[validation_end:],
    }


def evaluate_segment(
    bundle: MarketDataBundle,
    context: PreparedContext,
    split_name: str,
    index: pd.Index,
) -> BacktestResult:
    equity, trades = run_backtest(context, index)
    metrics = calculate_metrics(equity, trades, context.interval)
    score = score_metrics(metrics)
    benchmarks = build_benchmark_report(
        bundle,
        symbols=context.prepared.keys(),
        initial_cash=context.config.initial_cash,
        interval=context.interval,
        start=index[0],
        end=index[-1],
    )
    return BacktestResult(
        config=context.config,
        split=split_name,
        start=equity.index[0],
        end=equity.index[-1],
        bars=len(equity),
        symbols=len(context.prepared),
        rs_period=context.rs_period,
        equity=equity,
        trades=trades,
        metrics=metrics,
        benchmarks=benchmarks,
        score=score,
        coverage_dropped=context.coverage_dropped,
    )


def evaluate_config(
    bundle: MarketDataBundle,
    config: StrategyConfig,
    use_splits: bool = True,
) -> Dict[str, BacktestResult]:
    context = build_prepared_context(bundle, config)
    if use_splits:
        segments = split_index(
            context.common_index,
            config.train_ratio,
            config.validation_ratio,
        )
    else:
        segments = {"overall": context.common_index}

    return {
        split_name: evaluate_segment(bundle, context, split_name, index)
        for split_name, index in segments.items()
    }
