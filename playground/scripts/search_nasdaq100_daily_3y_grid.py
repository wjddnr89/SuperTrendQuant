# Exhaustive daily Nasdaq-100 rolling-universe strategy grid search.
#
# This script is a research runner for the conversation-built unified_quant
# engine.  It keeps the daily Nasdaq-100 rolling universe fixed,
# loads the selected market-data source once, then tests combinations of:
# entry type, QQQ market filter, asset filters, relative-strength formula,
# relative-strength lookback, rotation hurdle, and leader position count.  It
# prints and saves Top-N result tables for the requested metrics, including
# same-period QQQ return and alpha.

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLAYGROUND_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PLAYGROUND_ROOT / "src"))

from market_data_source import load_experiment_market_data
from supertrend_quant.config import AppConfig, load_split_config
from supertrend_quant.data import MarketData, market_index
from supertrend_quant.metrics import calculate_metrics, format_float, format_pct
from supertrend_quant.portfolio import AccountSnapshot, OrderIntent, OrderPlan, Position, estimate_quantity
from supertrend_quant.research import apply_config_overlay
from supertrend_quant.research.scoring import score_metrics
from supertrend_quant.runners import (
    BacktestResult,
    _close_at_or_before,
    _portfolio_value,
    _select_run_index,
)
from supertrend_quant.strategies import create_strategy
from supertrend_quant.strategies.common import (
    active_universe_symbols,
    asset_filters_allow_buy,
    configured_exit_down_confirmed,
    enabled_component,
    entry_state_allows_buy,
    sell_all,
)


def csv_values(value: str, cast=str) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search daily Nasdaq-100 strategy combinations.")
    parser.add_argument(
        "--strategy",
        default=str(PLAYGROUND_ROOT / "configs" / "strategies" / "leader_rotation.yaml"),
    )
    parser.add_argument(
        "--runtime",
        default=str(PLAYGROUND_ROOT / "configs" / "runtimes" / "research_us_nasdaq100_rolling.yaml"),
    )
    parser.add_argument("--period", default="max")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--data-source", choices=("local", "yahoo"), default="local")
    parser.add_argument("--entries", default="single")
    parser.add_argument("--market-filters", default="none,1d")
    parser.add_argument("--asset-filters", default="none,ichimoku_cloud,ichimoku_cloud+ema_trend")
    parser.add_argument("--sell-confirm-bars", default="1,2,3,5,8")
    parser.add_argument(
        "--rs-methods",
        default="vol_adjusted,composite,skip_recent,beta_adjusted,dual_momentum",
        help=(
            "Comma-separated RS scoring formulas. Supported aliases include "
            "relative_strength, vol_adjusted, composite, skip_recent, "
            "beta_adjusted, dual_momentum."
        ),
    )
    parser.add_argument("--rs-periods", default="50,100")
    parser.add_argument("--hurdles", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--max-positions", default="3,4,5,6")
    parser.add_argument("--st-periods", default="10")
    parser.add_argument("--st-multipliers", default="3.0")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--results-dir",
        default=str(PLAYGROUND_ROOT / "results" / "research" / "us_nasdaq100_rolling" / "searches"),
    )
    parser.add_argument("--run-id", default="")
    return parser


def group_parameter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries = csv_values(args.entries)
    market_filters = csv_values(args.market_filters)
    asset_filters = csv_values(args.asset_filters)
    rs_methods = csv_values(args.rs_methods)
    rs_periods = csv_values(args.rs_periods, int)
    st_periods = csv_values(args.st_periods, int)
    st_multipliers = csv_values(args.st_multipliers, float)

    groups: list[dict[str, Any]] = []
    common_product = itertools.product(
        entries,
        market_filters,
        asset_filters,
        rs_methods,
        rs_periods,
    )
    for entry, market_filter, asset_filter, rs_method, rs_period in common_product:
        base = {
            "entry": entry,
            "market_filter": market_filter,
            "asset_filter": asset_filter,
            "rs_method": rs_method,
            "rs_period": rs_period,
        }
        if entry in {"single", "single_supertrend", "supertrend"}:
            for st_period, st_multiplier in itertools.product(st_periods, st_multipliers):
                groups.append({**base, "st_period": st_period, "st_multiplier": st_multiplier})
        else:
            groups.append(base)
    return groups


def run_prepared_backtest(
    config: AppConfig,
    market_data: MarketData,
    prepared: dict[str, pd.DataFrame],
    candidates_by_position: list[list[dict[str, float | str]]],
    market_filter_states: dict[str, list[bool]],
    exit_down_states: dict[str, list[bool]],
    active_by_position: list[set[str] | None],
    row_positions: dict[str, Any],
    warmup_bars: int,
    run_index: pd.Index | None = None,
) -> BacktestResult:
    if not market_data.bars:
        raise RuntimeError("No market data was downloaded.")

    full_idx = market_index(market_data)
    idx = _select_run_index(full_idx, run_index)
    if len(idx) < 2:
        raise RuntimeError("Not enough common bars to run a backtest.")

    first_full_position = int(full_idx.get_indexer([idx[0]])[0])
    first_target_position = max(first_full_position, int(warmup_bars))
    idx = idx.intersection(full_idx[first_target_position:], sort=False)
    if len(idx) < 2:
        raise RuntimeError("Not enough bars remain after strategy warm-up.")

    cash = config.capital.initial_cash
    positions: dict[str, Position] = {}
    entry_values: dict[str, float] = {}
    entry_times: dict[str, object] = {}
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_returns: list[float] = []
    trade_records: list[dict[str, object]] = []

    for i in range(0, len(idx) - 1):
        signal_ts = idx[i]
        exec_ts = idx[i + 1]
        full_position = int(full_idx.get_loc(signal_ts))
        exec_position = full_position + 1
        equity_points.append(
            (
                signal_ts,
                portfolio_value_fast(cash, positions, prepared, row_positions, full_position),
            )
        )
        account = AccountSnapshot(cash=cash, positions=positions.copy())
        plan = build_fast_leader_plan(
            config,
            prepared,
            candidates_by_position[full_position],
            market_filter_states,
            exit_down_states,
            active_by_position[full_position],
            row_positions,
            full_position,
            account,
            mode="backtest",
        )

        for order in plan.orders:
            raw_price = open_at_position(prepared, row_positions, order.symbol, exec_position, exec_ts)
            if raw_price is None:
                continue
            if order.side.lower() == "buy":
                affordable_quantity = estimate_quantity(
                    cash,
                    raw_price,
                    1.0,
                    fee_rate=config.costs.fee_rate,
                    slippage_rate=config.costs.slippage_rate,
                )
                quantity = min(order.quantity, affordable_quantity)
                if quantity <= 0:
                    continue
                fill = raw_price * (1.0 + config.costs.slippage_rate)
                cost = quantity * fill * (1.0 + config.costs.fee_rate)
                if cost <= cash:
                    cash -= cost
                    positions[order.symbol] = Position(order.symbol, quantity, fill)
                    entry_values[order.symbol] = cost
                    entry_times[order.symbol] = exec_ts
            else:
                position = positions.get(order.symbol)
                if not position:
                    continue
                qty = min(position.quantity, order.quantity)
                fill = raw_price * (1.0 - config.costs.slippage_rate)
                proceeds = qty * fill * (1.0 - config.costs.fee_rate)
                cash += proceeds
                entry_value = entry_values.pop(order.symbol, qty * position.avg_price)
                pnl_pct = proceeds / entry_value - 1.0 if entry_value else 0.0
                trade_returns.append(pnl_pct)
                trade_records.append(
                    {
                        "symbol": order.symbol,
                        "entry_time": entry_times.pop(order.symbol, None),
                        "exit_time": exec_ts,
                        "entry_price": position.avg_price,
                        "exit_price": fill,
                        "quantity": qty,
                        "entry_value": entry_value,
                        "exit_value": proceeds,
                        "pnl_value": proceeds - entry_value,
                        "pnl_pct": pnl_pct,
                        "exit_reason": order.reason,
                    }
                )
                positions.pop(order.symbol, None)

    if positions:
        final_ts = idx[-1]
        final_full_position = int(full_idx.get_loc(final_ts))
        for symbol, position in list(positions.items()):
            final_close = close_at_or_before_position(
                prepared,
                row_positions,
                symbol,
                final_full_position,
            )
            if final_close is None:
                continue
            final_price = final_close * (1.0 - config.costs.slippage_rate)
            proceeds = position.quantity * final_price * (1.0 - config.costs.fee_rate)
            cash += proceeds
            entry_value = entry_values.pop(symbol, position.quantity * position.avg_price)
            pnl_pct = proceeds / entry_value - 1.0 if entry_value else 0.0
            trade_returns.append(pnl_pct)
            trade_records.append(
                {
                    "symbol": symbol,
                    "entry_time": entry_times.pop(symbol, None),
                    "exit_time": final_ts,
                    "entry_price": position.avg_price,
                    "exit_price": final_price,
                    "quantity": position.quantity,
                    "entry_value": entry_value,
                    "exit_value": proceeds,
                    "pnl_value": proceeds - entry_value,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "FinalClose",
                }
            )
            positions.pop(symbol, None)

    equity_points.append(
        (
            idx[-1],
            portfolio_value_fast(cash, positions, prepared, row_positions, int(full_idx.get_loc(idx[-1]))),
        )
    )
    equity = pd.Series(
        [point[1] for point in equity_points],
        index=[point[0] for point in equity_points],
        name="equity",
    )
    return BacktestResult(
        equity=equity,
        metrics=calculate_metrics(equity, trade_returns, config.timeframe),
        trades=trade_returns,
        skipped=market_data.skipped,
        trade_records=tuple(trade_records),
        universe_snapshot=getattr(market_data, "universe_snapshot", None),
    )


def prepare_row_positions(
    frames: dict[str, pd.DataFrame],
    full_idx: pd.Index,
) -> dict[str, Any]:
    return {
        symbol: frame.index.searchsorted(full_idx, side="right") - 1
        for symbol, frame in frames.items()
    }


def prepare_exit_down_states(
    config: AppConfig,
    prepared: dict[str, pd.DataFrame],
    full_idx: pd.Index,
    row_positions: dict[str, Any],
) -> dict[str, list[bool]]:
    triple_exit = enabled_component(config, "exits", "triple_supertrend_flip")
    if triple_exit is None:
        confirm_bars = max(1, int(config.exit.sell_confirm_bars))
        column = "Trend"
        down_count = None
    else:
        confirm_bars = max(1, int(triple_exit.params.get("confirm_bars", config.exit.sell_confirm_bars)))
        column = "TripleDownCount"
        down_count = int(triple_exit.params.get("down_count", 2))

    states: dict[str, list[bool]] = {}
    for symbol, frame in prepared.items():
        positions = row_positions.get(symbol)
        if positions is None or column not in frame:
            states[symbol] = [False for _ in full_idx]
            continue
        if triple_exit is None:
            raw_down = frame[column].astype(int).eq(-1)
        else:
            raw_down = frame[column].astype(int).ge(int(down_count))
        if confirm_bars > 1:
            confirmed = raw_down.rolling(confirm_bars, min_periods=confirm_bars).sum().eq(confirm_bars)
        else:
            confirmed = raw_down
        values = confirmed.fillna(False).astype(bool).to_numpy()
        states[symbol] = [
            bool(position >= 0 and values[int(position)])
            for position in positions
        ]
    return states


def prepare_market_filter_states(prepared_backtest, full_idx: pd.Index) -> dict[str, list[bool]]:
    states: dict[str, list[bool]] = {}
    for symbol, trend in prepared_backtest.market_filter_trends.items():
        positions = trend.index.searchsorted(full_idx, side="right") - 1
        values = trend.astype("int64").to_numpy()
        states[symbol] = [
            bool(position >= 0 and int(values[position]) == 1)
            for position in positions
        ]
    return states


def prepare_active_universe(
    market_data: MarketData,
    full_idx: pd.Index,
) -> list[set[str] | None]:
    schedule = getattr(market_data, "universe_schedule", ())
    if not schedule:
        return [None for _ in full_idx]
    return [active_universe_symbols(schedule, timestamp) for timestamp in full_idx]


def open_at_position(
    prepared: dict[str, pd.DataFrame],
    row_positions: dict[str, Any],
    symbol: str,
    full_position: int,
    timestamp,
) -> float | None:
    frame = prepared.get(symbol)
    positions = row_positions.get(symbol)
    if frame is None or positions is None or full_position >= len(positions):
        return None
    row_position = int(positions[full_position])
    if row_position < 0 or row_position >= len(frame) or frame.index[row_position] != timestamp:
        return None
    return finite_float(frame["Open"].iat[row_position])


def close_at_or_before_position(
    prepared: dict[str, pd.DataFrame],
    row_positions: dict[str, Any],
    symbol: str,
    full_position: int,
) -> float | None:
    frame = prepared.get(symbol)
    positions = row_positions.get(symbol)
    if frame is None or positions is None or full_position >= len(positions):
        return None
    row_position = int(positions[full_position])
    if row_position < 0 or row_position >= len(frame):
        return None
    return finite_float(frame["Close"].iat[row_position])


def value_at_or_before_position(
    prepared: dict[str, pd.DataFrame],
    row_positions: dict[str, Any],
    symbol: str,
    full_position: int,
    column: str,
) -> float | None:
    frame = prepared.get(symbol)
    positions = row_positions.get(symbol)
    if frame is None or positions is None or column not in frame or full_position >= len(positions):
        return None
    row_position = int(positions[full_position])
    if row_position < 0 or row_position >= len(frame):
        return None
    return finite_float(frame[column].iat[row_position])


def portfolio_value_fast(
    cash: float,
    positions: dict[str, Position],
    prepared: dict[str, pd.DataFrame],
    row_positions: dict[str, Any],
    full_position: int,
) -> float:
    value = cash
    for symbol, position in positions.items():
        close = close_at_or_before_position(prepared, row_positions, symbol, full_position)
        if close is not None:
            value += position.quantity * close
    return value


def build_fast_leader_plan(
    config: AppConfig,
    prepared: dict[str, pd.DataFrame],
    candidates: list[dict[str, float | str]],
    market_filter_states: dict[str, list[bool]],
    exit_down_states: dict[str, list[bool]],
    active_symbols: set[str] | None,
    row_positions: dict[str, Any],
    signal_position: int,
    account: AccountSnapshot,
    mode: str,
) -> OrderPlan:
    if not prepared:
        return OrderPlan(config.strategy.name, mode, (), ("No prepared symbol data.",))

    orders: list[OrderIntent] = []
    max_positions = max(1, int(config.risk.max_position_count))
    target_candidates = candidates[:max_positions]
    target_symbols = {str(candidate["symbol"]) for candidate in target_candidates}
    held_positions = {
        symbol: position
        for symbol, position in account.positions.items()
        if position.quantity > 0
    }
    sell_symbols: set[str] = set()
    estimated_cash = float(account.cash)

    for symbol, held in held_positions.items():
        held_df = prepared.get(symbol)
        held_position = row_positions.get(symbol, [])[signal_position] if symbol in row_positions else -1
        held_close = value_at_or_before_position(prepared, row_positions, symbol, signal_position, "Close")
        if held_df is None or held_df.empty or held_position < 0 or held_close is None:
            orders.append(sell_all(held, "Held symbol missing from strategy data"))
            sell_symbols.add(symbol)
            continue

        sell_reason = None
        exit_states = exit_down_states.get(symbol)
        if exit_states is not None and bool(exit_states[signal_position]):
            sell_reason = (
                "Triple Supertrend down"
                if enabled_component(config, "exits", "triple_supertrend_flip") is not None
                else "Supertrend down"
            )
        elif symbol not in target_symbols:
            replacement = first_replacement_candidate(
                target_candidates,
                held_symbols=set(held_positions),
                sell_symbols=sell_symbols,
            )
            if replacement is None:
                continue
            current_score = value_at_or_before_position(
                prepared,
                row_positions,
                symbol,
                signal_position,
                "Score",
            )
            hurdle = float(replacement["atr_pct"]) * config.leader_rotation.hurdle_atr_mult
            if current_score is not None and float(replacement["score"]) - current_score > hurdle:
                profit_pct = (
                    (held_close - held.avg_price) / held.avg_price
                    if held.avg_price > 0
                    else 0.0
                )
                if profit_pct >= config.leader_rotation.min_rotation_profit_pct:
                    sell_reason = "Leader rotation"
        if sell_reason:
            orders.append(sell_all(held, sell_reason))
            sell_symbols.add(symbol)
            estimated_cash += estimated_sell_proceeds(held, held_close, config)

    kept_symbols = set(held_positions) - sell_symbols
    open_slots = max(0, max_positions - len(kept_symbols))
    buy_candidates = [
        candidate
        for candidate in candidates
        if candidate["symbol"] not in kept_symbols and candidate["symbol"] not in sell_symbols
    ]
    remaining_buy_budget = estimated_cash * config.execution.allocation_pct

    for candidate in buy_candidates:
        if open_slots <= 0 or estimated_cash <= 0 or remaining_buy_budget <= 0:
            break
        slot_budget = remaining_buy_budget / open_slots
        qty = estimate_quantity(
            min(estimated_cash, slot_budget),
            float(candidate["price"]),
            1.0,
            fee_rate=config.costs.fee_rate,
            slippage_rate=config.costs.slippage_rate,
        )
        if qty <= 0:
            continue
        orders.append(
            OrderIntent(
                symbol=str(candidate["symbol"]),
                side="buy",
                quantity=qty,
                order_type=config.execution.order_type,
                reason="Top-ranked leader",
            )
        )
        estimated_cost = estimated_buy_cost(qty, float(candidate["price"]), config)
        estimated_cash = max(0.0, estimated_cash - estimated_cost)
        remaining_buy_budget = max(0.0, remaining_buy_budget - estimated_cost)
        open_slots -= 1

    return OrderPlan(strategy_name=config.strategy.name, mode=mode, orders=tuple(orders))


def prepare_candidate_lists(
    config: AppConfig,
    strategy,
    prepared: dict[str, pd.DataFrame],
    market_filter_states: dict[str, list[bool]],
    active_by_position: list[set[str] | None],
    row_positions: dict[str, Any],
    full_idx: pd.Index,
) -> list[list[dict[str, float | str]]]:
    symbol_arrays = prepare_candidate_arrays(config, prepared, market_filter_states, row_positions, len(full_idx))
    return [
        fast_leader_candidates_from_arrays(
            strategy,
            active_by_position[position],
            symbol_arrays,
            position,
        )
        for position in range(len(full_idx))
    ]


def prepare_candidate_arrays(
    config: AppConfig,
    prepared: dict[str, pd.DataFrame],
    market_filter_states: dict[str, list[bool]],
    row_positions: dict[str, Any],
    length: int,
) -> dict[str, dict[str, np.ndarray]]:
    use_triple_entry = enabled_component(config, "entries", "triple_supertrend") is not None
    use_ichimoku = enabled_component(config, "filters", "ichimoku_cloud") is not None
    use_ema = enabled_component(config, "filters", "ema_trend") is not None
    allow_late_chase = bool(config.leader_rotation.allow_late_chase)

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for symbol, frame in prepared.items():
        positions = row_positions.get(symbol)
        if positions is None:
            continue
        positions_array = np.asarray(positions, dtype=np.int64)
        valid = (positions_array >= 0) & (positions_array < len(frame))

        score = full_float_array(length)
        atr_pct = full_float_array(length)
        price = full_float_array(length)
        ok = valid.copy()

        valid_positions = positions_array[valid]
        fill_float_column(score, valid, frame, valid_positions, "Score")
        fill_float_column(atr_pct, valid, frame, valid_positions, "ATR_pct")
        fill_float_column(price, valid, frame, valid_positions, "Close")

        if config.market_trend_filter.enabled:
            market_states = np.asarray(market_filter_states.get(symbol, [False] * length), dtype=bool)
            if len(market_states) != length:
                fixed = np.zeros(length, dtype=bool)
                fixed[: min(length, len(market_states))] = market_states[: min(length, len(market_states))]
                market_states = fixed
            ok &= market_states

        if use_triple_entry:
            triple_all_up = bool_column_at(frame, valid_positions, "TripleAllUp")
            if allow_late_chase:
                entry_ok = triple_all_up
            else:
                entry_ok = triple_all_up & bool_column_at(frame, valid_positions, "TripleBuySignal")
        else:
            trend = float_column_at(frame, valid_positions, "Trend")
            trend_up = trend == 1.0
            if allow_late_chase:
                entry_ok = trend_up
            else:
                entry_ok = trend_up & bool_column_at(frame, valid_positions, "BuySignal")
        ok[valid] &= entry_ok

        if use_ichimoku:
            ok[valid] &= bool_column_at(frame, valid_positions, "Ichimoku_LongOk")
        if use_ema:
            ok[valid] &= bool_column_at(frame, valid_positions, "EMA_LongOk")

        ok &= np.isfinite(score) & np.isfinite(atr_pct) & np.isfinite(price)
        arrays[symbol] = {
            "ok": ok,
            "score": score,
            "atr_pct": atr_pct,
            "price": price,
        }
    return arrays


def full_float_array(length: int) -> np.ndarray:
    out = np.empty(length, dtype=float)
    out.fill(np.nan)
    return out


def fill_float_column(
    target: np.ndarray,
    valid: np.ndarray,
    frame: pd.DataFrame,
    valid_positions: np.ndarray,
    column: str,
) -> None:
    if column not in frame:
        return
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    target[valid] = values[valid_positions]


def float_column_at(frame: pd.DataFrame, positions: np.ndarray, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(positions), np.nan, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[positions]


def bool_column_at(frame: pd.DataFrame, positions: np.ndarray, column: str) -> np.ndarray:
    if column not in frame:
        return np.zeros(len(positions), dtype=bool)
    return frame[column].fillna(False).astype(bool).to_numpy()[positions]


def fast_leader_candidates_from_arrays(
    strategy,
    active_symbols: set[str] | None,
    symbol_arrays: dict[str, dict[str, np.ndarray]],
    signal_position: int,
) -> list[dict[str, float | str]]:
    candidate_scores: dict[str, float] = {}
    candidates: dict[str, dict[str, float | str]] = {}
    symbols = active_symbols if active_symbols is not None else set(symbol_arrays)
    for symbol in symbols:
        data = symbol_arrays.get(symbol)
        if data is None or not bool(data["ok"][signal_position]):
            continue
        score = float(data["score"][signal_position])
        atr_pct = float(data["atr_pct"][signal_position])
        price = float(data["price"][signal_position])
        candidate_scores[symbol] = score
        candidates[symbol] = {
            "symbol": symbol,
            "score": score,
            "atr_pct": atr_pct,
            "price": price,
        }
    return [candidates[symbol] for symbol in strategy.scorer.rank(candidate_scores)]


def fast_leader_candidates(
    config: AppConfig,
    strategy,
    prepared: dict[str, pd.DataFrame],
    market_filter_states: dict[str, list[bool]],
    active_symbols: set[str] | None,
    row_positions: dict[str, Any],
    signal_position: int,
) -> list[dict[str, float | str]]:
    candidate_scores: dict[str, float] = {}
    candidates: dict[str, dict[str, float | str]] = {}
    symbols = active_symbols if active_symbols is not None else set(prepared)
    for symbol in symbols:
        df = prepared.get(symbol)
        positions = row_positions.get(symbol)
        if df is None or positions is None:
            continue
        row_position = int(positions[signal_position])
        if row_position < 0:
            continue
        if config.market_trend_filter.enabled and not market_filter_states.get(symbol, [False])[signal_position]:
            continue
        row = df.iloc[row_position]
        if not asset_filters_allow_buy(config, row):
            continue
        if not entry_state_allows_buy(config, row):
            continue
        score = finite_float(row.get("Score"))
        atr_pct = finite_float(row.get("ATR_pct"))
        price = finite_float(row.get("Close"))
        if score is None or atr_pct is None or price is None:
            continue
        candidate_scores[symbol] = score
        candidates[symbol] = {
            "symbol": symbol,
            "score": score,
            "atr_pct": atr_pct,
            "price": price,
        }
    return [candidates[symbol] for symbol in strategy.scorer.rank(candidate_scores)]


def finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def first_replacement_candidate(
    candidates: list[dict[str, float | str]],
    *,
    held_symbols: set[str],
    sell_symbols: set[str],
) -> dict[str, float | str] | None:
    for candidate in candidates:
        symbol = str(candidate["symbol"])
        if symbol not in held_symbols or symbol in sell_symbols:
            return candidate
    return None


def estimated_sell_proceeds(position, price: float, config: AppConfig) -> float:
    fill = price * (1.0 - config.costs.slippage_rate)
    return position.quantity * max(0.0, fill) * (1.0 - config.costs.fee_rate)


def estimated_buy_cost(quantity: float, price: float, config: AppConfig) -> float:
    fill = price * (1.0 + config.costs.slippage_rate)
    return quantity * max(0.0, fill) * (1.0 + config.costs.fee_rate)


def rank_value(row: dict[str, Any], rank_by: str) -> float:
    metrics = row["metrics"]
    if rank_by == "score":
        return float(row["score"])
    if rank_by == "sharpe":
        return float(metrics.get("sharpe", 0.0))
    if rank_by == "calmar":
        return float(metrics.get("calmar", 0.0))
    return float(metrics.get("total_return", 0.0))


def run_index_from_start(full_idx: pd.Index, start: str) -> pd.Index:
    if not start:
        return full_idx
    start_date = pd.Timestamp(start).date()
    selected = full_idx[[pd.Timestamp(timestamp).date() >= start_date for timestamp in full_idx]]
    if len(selected) < 2:
        raise RuntimeError(f"Not enough market bars at or after requested start={start}.")
    return selected


def ordered_rows(rows: list[dict[str, Any]], rank_by: str) -> list[dict[str, Any]]:
    def value(row: dict[str, Any]) -> float:
        metrics = row["metrics"]
        if rank_by == "return":
            return float(metrics.get("total_return", 0.0))
        if rank_by == "alpha":
            return float(row.get("alpha", 0.0))
        if rank_by == "mdd":
            return float(metrics.get("mdd", 0.0))
        if rank_by == "sharpe":
            return float(metrics.get("sharpe", 0.0))
        if rank_by == "win_rate":
            return float(metrics.get("win_rate", 0.0))
        if rank_by == "payoff":
            return float(metrics.get("payoff_ratio", 0.0))
        raise ValueError(f"Unsupported rank_by: {rank_by}")

    return sorted(
        rows,
        key=lambda row: (
            value(row),
            float(row["metrics"].get("total_return", 0.0)),
            float(row["metrics"].get("sharpe", 0.0)),
        ),
        reverse=True,
    )


def row_line(rank: int, row: dict[str, Any]) -> str:
    p = row["params"]
    m = row["metrics"]
    return (
        f"{rank:>4} "
        f"{str(p.get('entry')):<6} "
        f"{str(p.get('market_filter')):<3} "
        f"{str(p.get('asset_filter')):<20} "
        f"{str(p.get('rs_method', '-')):<13} "
        f"{int(p.get('sell_confirm_bars')):>4} "
        f"{int(p.get('rs_period')):>3} "
        f"{float(p.get('hurdle', 0.0)):>4.2f} "
        f"{int(p.get('max_positions')):>4} "
        f"{str(p.get('st_period', '-')):>3} "
        f"{str(p.get('st_multiplier', '-')):>3} "
        f"{format_pct(float(m.get('total_return', 0.0))):>8} "
        f"{format_pct(float(row.get('qqq_return', 0.0))):>8} "
        f"{format_pct(float(row.get('alpha', 0.0))):>8} "
        f"{format_pct(float(m.get('mdd', 0.0))):>8} "
        f"{format_float(float(m.get('sharpe', 0.0))):>6} "
        f"{format_pct(float(m.get('win_rate', 0.0))):>7} "
        f"{format_float(float(m.get('payoff_ratio', 0.0))):>6} "
        f"{int(m.get('trade_count', 0)):>6} "
        f"{format_float(float(row['score'])):>5}"
    )


def result_table_lines(title: str, rows: list[dict[str, Any]], top: int) -> list[str]:
    header = (
        "Rank Entry  Mkt Asset                RSMethod      Sell RS  Hurd  Pos STp STm "
        "Return      QQQ    Alpha      MDD Sharpe WinRate Payoff Trades Score"
    )
    selected = rows[:top]
    lines = ["", title, header]
    lines.extend(row_line(rank, row) for rank, row in enumerate(selected, start=1))
    return lines


def qqq_return_for_index(
    market_data: MarketData,
    run_index: pd.Index,
    cache: dict[tuple[object, object], float],
) -> float:
    if len(run_index) == 0:
        return 0.0
    key = (run_index[0], run_index[-1])
    if key in cache:
        return cache[key]
    benchmark = market_data.benchmark or {}
    frame = next((df for df in benchmark.values() if df is not None and not df.empty), None)
    if frame is None or "Close" not in frame:
        cache[key] = 0.0
        return 0.0
    close = frame["Close"].dropna().sort_index().astype(float)
    try:
        bounded = close.loc[run_index[0] : run_index[-1]]
    except (KeyError, TypeError, ValueError):
        start_date = pd.Timestamp(run_index[0]).date()
        end_date = pd.Timestamp(run_index[-1]).date()
        bounded = close.loc[
            [start_date <= pd.Timestamp(timestamp).date() <= end_date for timestamp in close.index]
        ]
    if len(bounded) < 2 or float(bounded.iloc[0]) == 0:
        value = 0.0
    else:
        value = float(bounded.iloc[-1] / bounded.iloc[0] - 1.0)
    cache[key] = value
    return value


def print_rows(rows: list[dict[str, Any]], rank_by: str, top: int) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            rank_value(row, rank_by),
            float(row["metrics"].get("total_return", 0.0)),
            float(row["metrics"].get("sharpe", 0.0)),
        ),
        reverse=True,
    )
    print()
    print(f"Top {min(top, len(ordered))} by {rank_by}")
    header = (
        "Rank Entry  Mkt Asset                RSMethod      Sell RS  Hurd  Pos STp STm "
        "Return      QQQ    Alpha      MDD Sharpe WinRate Payoff Trades Score"
    )
    print(header)
    for rank, row in enumerate(ordered[:top], start=1):
        p = row["params"]
        m = row["metrics"]
        print(
            f"{rank:>4} "
            f"{str(p.get('entry')):<6} "
            f"{str(p.get('market_filter')):<3} "
            f"{str(p.get('asset_filter')):<20} "
            f"{str(p.get('rs_method', '-')):<13} "
            f"{int(p.get('sell_confirm_bars')):>4} "
            f"{int(p.get('rs_period')):>3} "
            f"{float(p.get('hurdle', 0.0)):>4.2f} "
            f"{int(p.get('max_positions')):>4} "
            f"{str(p.get('st_period', '-')):>3} "
            f"{str(p.get('st_multiplier', '-')):>3} "
            f"{format_pct(float(m.get('total_return', 0.0))):>8} "
            f"{format_pct(float(row.get('qqq_return', 0.0))):>8} "
            f"{format_pct(float(row.get('alpha', 0.0))):>8} "
            f"{format_pct(float(m.get('mdd', 0.0))):>8} "
            f"{format_float(float(m.get('sharpe', 0.0))):>6} "
            f"{format_pct(float(m.get('win_rate', 0.0))):>7} "
            f"{format_float(float(m.get('payoff_ratio', 0.0))):>6} "
            f"{int(m.get('trade_count', 0)):>6} "
            f"{format_float(float(row['score'])):>5}"
        )


def flat_row(row: dict[str, Any]) -> dict[str, Any]:
    params = row["params"]
    metrics = row["metrics"]
    return {
        **{f"param_{key}": value for key, value in params.items()},
        "data_source": row.get("data_source", ""),
        "start": str(row["start"]),
        "end": str(row["end"]),
        "total_return": metrics.get("total_return", 0.0),
        "qqq_return": row.get("qqq_return", 0.0),
        "alpha": row.get("alpha", 0.0),
        "mdd": metrics.get("mdd", 0.0),
        "cagr": metrics.get("cagr", 0.0),
        "calmar": metrics.get("calmar", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "sortino": metrics.get("sortino", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
        "payoff_ratio": metrics.get("payoff_ratio", 0.0),
        "trade_count": metrics.get("trade_count", 0),
        "score": row["score"],
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return None
    return value


def write_results(
    rows: list[dict[str, Any]],
    top_by_metric: dict[str, list[dict[str, Any]]],
    report_lines: list[str],
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> Path:
    run_id = args.run_id.strip() or datetime.now().strftime("search_%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(flat_row(row) for row in rows).to_csv(run_dir / "all_results.csv", index=False)
    (run_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary = {
        "metadata": metadata,
        "top": {
            metric: [flat_row(row) for row in selected[: args.top]]
            for metric, selected in top_by_metric.items()
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    args = build_parser().parse_args()
    base = load_split_config(args.strategy, args.runtime)
    base = base.__class__(**{**base.__dict__, "period": args.period, "timeframe": "1d"})
    max_positions = csv_values(args.max_positions, int)
    sell_confirm_values = csv_values(args.sell_confirm_bars, int)
    hurdles = csv_values(args.hurdles, float)
    groups = group_parameter_grid(args)
    total_runs = len(groups) * len(sell_confirm_values) * len(hurdles) * len(max_positions)

    print("Nasdaq-100 Daily Exhaustive Strategy Search")
    print(f"Data source  : {args.data_source}")
    print(f"Groups       : {len(groups)}")
    print(f"Total runs   : {total_runs}")
    print(f"Requested    : start={args.start}, period={args.period}, sell_confirm={args.sell_confirm_bars}")
    print(f"RS methods   : {args.rs_methods}")
    print(f"Hurdles      : {args.hurdles}")
    print(f"Max positions: {args.max_positions}")
    print("Loading shared market data...", flush=True)
    market_data = load_experiment_market_data(
        base,
        data_source=args.data_source,
        strategy_path=args.strategy,
        runtime_path=args.runtime,
    )
    print("Preparing market timeline...", flush=True)
    full_idx = market_index(market_data)
    requested_idx = run_index_from_start(full_idx, args.start)
    active_by_position = prepare_active_universe(market_data, full_idx)
    print(
        f"Market timeline: {full_idx[0]} -> {full_idx[-1]} "
        f"({len(full_idx)} bars), requested bars={len(requested_idx)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    qqq_return_cache: dict[tuple[object, object], float] = {}
    started = time.monotonic()
    completed = 0
    for group_number, group_params in enumerate(groups, start=1):
        print(f"Group {group_number}/{len(groups)} prepare: {group_params}", flush=True)
        group_config = apply_config_overlay(base, group_params)
        strategy = create_strategy(group_config)
        prepared_template = strategy.prepare_backtest(
            market_data.bars,
            benchmark=market_data.benchmark,
            filter_benchmark=market_data.filter_benchmark,
            universe_schedule=market_data.universe_schedule,
        )
        print(f"Group {group_number}/{len(groups)} row positions...", flush=True)
        row_positions = prepare_row_positions(prepared_template.prepared, full_idx)
        print(f"Group {group_number}/{len(groups)} market filters...", flush=True)
        market_filter_states = prepare_market_filter_states(prepared_template, full_idx)
        print(f"Group {group_number}/{len(groups)} candidate lists...", flush=True)
        candidates_by_position = prepare_candidate_lists(
            group_config,
            strategy,
            prepared_template.prepared,
            market_filter_states,
            active_by_position,
            row_positions,
            full_idx,
        )
        print(f"Group {group_number}/{len(groups)} backtest runs...", flush=True)
        for sell_confirm in sell_confirm_values:
            confirm_config = apply_config_overlay(
                group_config,
                {"sell_confirm_bars": sell_confirm},
            )
            exit_down_states = prepare_exit_down_states(
                confirm_config,
                prepared_template.prepared,
                full_idx,
                row_positions,
            )
            for hurdle in hurdles:
                hurdle_config = apply_config_overlay(confirm_config, {"hurdle": hurdle})
                for position_count in max_positions:
                    params = {
                        **group_params,
                        "sell_confirm_bars": sell_confirm,
                        "hurdle": hurdle,
                        "max_positions": position_count,
                    }
                    config = apply_config_overlay(hurdle_config, {"max_positions": position_count})
                    result = run_prepared_backtest(
                        config,
                        market_data,
                        prepared_template.prepared,
                        candidates_by_position,
                        market_filter_states,
                        exit_down_states,
                        active_by_position,
                        row_positions,
                        strategy.warmup_bars(),
                        requested_idx,
                    )
                    qqq_return = qqq_return_for_index(
                        market_data,
                        result.equity.index,
                        qqq_return_cache,
                    )
                    total_return = float(result.metrics.get("total_return", 0.0))
                    rows.append(
                        {
                            "params": params,
                            "metrics": result.metrics,
                            "score": score_metrics(result.metrics),
                            "qqq_return": qqq_return,
                            "alpha": total_return - qqq_return,
                            "data_source": args.data_source,
                            "start": result.equity.index[0],
                            "end": result.equity.index[-1],
                        }
                    )
                    completed += 1
                    if args.progress_every and (
                        completed % args.progress_every == 0 or completed == total_runs
                    ):
                        elapsed = time.monotonic() - started
                        print(
                            f"Progress: {completed}/{total_runs} runs "
                            f"({group_number}/{len(groups)} groups), "
                            f"params={params}, elapsed {elapsed:.1f}s",
                            flush=True,
                        )

    if rows:
        first = rows[0]
        print(f"Evaluation   : {first['start']} -> {first['end']}")
    if market_data.skipped:
        print(f"Skipped      : {', '.join(market_data.skipped)}")

    ranking_specs = {
        "return": "Top 5 by Return",
        "mdd": "Top 5 by MDD",
        "sharpe": "Top 5 by Sharpe",
        "win_rate": "Top 5 by Win Rate",
        "payoff": "Top 5 by Payoff",
    }
    top_by_metric = {metric: ordered_rows(rows, metric) for metric in ranking_specs}
    metadata = {
        "strategy": args.strategy,
        "runtime": args.runtime,
        "data_source": args.data_source,
        "requested_start": args.start,
        "download_period": args.period,
        "timeframe": "1d",
        "sell_confirm_bars": list(sell_confirm_values),
        "rs_methods": list(csv_values(args.rs_methods)),
        "hurdles": list(csv_values(args.hurdles, float)),
        "max_positions": list(max_positions),
        "groups": len(groups),
        "total_runs": total_runs,
        "actual_start": str(rows[0]["start"]) if rows else None,
        "actual_end": str(rows[0]["end"]) if rows else None,
        "universe_schedule_count": len(getattr(market_data, "universe_schedule", ()) or ()),
        "skipped": list(market_data.skipped),
    }
    report_lines = [
        "Nasdaq-100 Daily Search Results",
        f"Data source  : {args.data_source}",
        f"Requested    : start={args.start}, period={args.period}",
        f"Evaluation   : {metadata['actual_start']} -> {metadata['actual_end']}",
        f"Total runs   : {total_runs}",
        f"Sell confirm : {','.join(str(value) for value in sell_confirm_values)}",
        f"RS methods   : {args.rs_methods}",
        f"Hurdles      : {args.hurdles}",
        f"Max positions: {','.join(str(value) for value in max_positions)}",
    ]
    if market_data.skipped:
        report_lines.append(f"Skipped      : {', '.join(market_data.skipped)}")
    for metric, title in ranking_specs.items():
        report_lines.extend(result_table_lines(title, top_by_metric[metric], args.top))

    print("\n".join(report_lines))
    run_dir = write_results(rows, top_by_metric, report_lines, args, metadata)
    print()
    print(f"Saved results: {run_dir}")


if __name__ == "__main__":
    main()
