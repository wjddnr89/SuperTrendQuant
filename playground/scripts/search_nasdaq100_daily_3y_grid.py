# Exhaustive daily Nasdaq-100 rolling-universe strategy grid search.
#
# This script is a research runner for the conversation-built unified_quant
# engine.  It keeps the 3-year daily Nasdaq-100 rolling universe fixed,
# downloads Yahoo Finance data once, then tests combinations of:
# entry type, QQQ market filter, asset filters, sell confirmation bars,
# relative-strength lookback, and leader position count.  It prints results
# directly instead of saving files, including same-period QQQ return and alpha.

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


UNIFIED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UNIFIED_ROOT / "src"))

from supertrend_quant.config import AppConfig, load_split_config
from supertrend_quant.data import MarketData, market_index
from supertrend_quant.metrics import calculate_metrics, format_float, format_pct
from supertrend_quant.portfolio import AccountSnapshot, OrderIntent, OrderPlan, Position, estimate_quantity
from supertrend_quant.research import apply_config_overlay
from supertrend_quant.research.data_resolver import download_for_config
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
        default=str(UNIFIED_ROOT / "configs" / "strategies" / "leader_rotation.yaml"),
    )
    parser.add_argument(
        "--runtime",
        default=str(UNIFIED_ROOT / "configs" / "runtimes" / "research_us_nasdaq100_rolling.yaml"),
    )
    parser.add_argument("--period", default="3y")
    parser.add_argument("--entries", default="single,triple")
    parser.add_argument("--market-filters", default="none,1d")
    parser.add_argument("--asset-filters", default="none,ichimoku_cloud,ema_trend,ichimoku_cloud+ema_trend")
    parser.add_argument("--sell-confirm-bars", default="1,2,3,5,8,13")
    parser.add_argument("--rs-periods", default="50,100")
    parser.add_argument("--max-positions", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--st-periods", default="10")
    parser.add_argument("--st-multipliers", default="3.0")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--rank-by", choices=("score", "return", "sharpe", "calmar"), default="return")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def group_parameter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries = csv_values(args.entries)
    market_filters = csv_values(args.market_filters)
    asset_filters = csv_values(args.asset_filters)
    rs_periods = csv_values(args.rs_periods, int)
    st_periods = csv_values(args.st_periods, int)
    st_multipliers = csv_values(args.st_multipliers, float)

    groups: list[dict[str, Any]] = []
    common_product = itertools.product(
        entries,
        market_filters,
        asset_filters,
        rs_periods,
    )
    for entry, market_filter, asset_filter, rs_period in common_product:
        base = {
            "entry": entry,
            "market_filter": market_filter,
            "asset_filter": asset_filter,
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
    active_by_position: list[set[str] | None],
    row_positions: dict[str, Any],
    warmup_bars: int,
) -> BacktestResult:
    if not market_data.bars:
        raise RuntimeError("No market data was downloaded.")

    full_idx = market_index(market_data)
    idx = _select_run_index(full_idx, None)
    if len(idx) < 2:
        raise RuntimeError("Not enough common bars to run a backtest.")

    first_target_position = max(0, int(warmup_bars))
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
        equity_points.append((signal_ts, _portfolio_value(cash, positions, market_data.bars, signal_ts)))
        account = AccountSnapshot(cash=cash, positions=positions.copy())
        full_position = int(full_idx.get_loc(signal_ts))
        plan = build_fast_leader_plan(
            config,
            prepared,
            candidates_by_position[full_position],
            market_filter_states,
            active_by_position[full_position],
            row_positions,
            full_position,
            account,
            mode="backtest",
        )

        for order in plan.orders:
            df = market_data.bars.get(order.symbol)
            if df is None or exec_ts not in df.index:
                continue
            raw_price = float(df.loc[exec_ts, "Open"])
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
                        "pnl_pct": pnl_pct,
                        "exit_reason": order.reason,
                    }
                )
                positions.pop(order.symbol, None)

    if positions:
        final_ts = idx[-1]
        for symbol, position in list(positions.items()):
            final_close = _close_at_or_before(market_data.bars.get(symbol), final_ts)
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
                    "pnl_pct": pnl_pct,
                    "exit_reason": "FinalClose",
                }
            )
            positions.pop(symbol, None)

    equity_points.append((idx[-1], _portfolio_value(cash, positions, market_data.bars, idx[-1])))
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


def build_fast_leader_plan(
    config: AppConfig,
    prepared: dict[str, pd.DataFrame],
    candidates: list[dict[str, float | str]],
    market_filter_states: dict[str, list[bool]],
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
        if held_df is None or held_df.empty or held_position < 0:
            orders.append(sell_all(held, "Held symbol missing from strategy data"))
            sell_symbols.add(symbol)
            continue

        held_row = held_df.iloc[int(held_position)]
        sell_reason = None
        history = held_df.iloc[: int(held_position) + 1]
        if configured_exit_down_confirmed(config, history):
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
            current_score = finite_float(held_row.get("Score"))
            hurdle = float(replacement["atr_pct"]) * config.leader_rotation.hurdle_atr_mult
            if current_score is not None and float(replacement["score"]) - current_score > hurdle:
                profit_pct = (
                    (float(held_row["Close"]) - held.avg_price) / held.avg_price
                    if held.avg_price > 0
                    else 0.0
                )
                if profit_pct >= config.leader_rotation.min_rotation_profit_pct:
                    sell_reason = "Leader rotation"
        if sell_reason:
            orders.append(sell_all(held, sell_reason))
            sell_symbols.add(symbol)
            estimated_cash += estimated_sell_proceeds(held, float(held_row["Close"]), config)

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
    return [
        fast_leader_candidates(
            config,
            strategy,
            prepared,
            market_filter_states,
            active_by_position[position],
            row_positions,
            position,
        )
        for position in range(len(full_idx))
    ]


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
        "Rank Entry  Mkt Asset                Sell RS  Pos STp STm "
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
            f"{int(p.get('sell_confirm_bars')):>4} "
            f"{int(p.get('rs_period')):>3} "
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


def main() -> None:
    args = build_parser().parse_args()
    base = load_split_config(args.strategy, args.runtime)
    base = base.__class__(**{**base.__dict__, "period": args.period, "timeframe": "1d"})
    max_positions = csv_values(args.max_positions, int)
    sell_confirm_values = csv_values(args.sell_confirm_bars, int)
    groups = group_parameter_grid(args)
    total_runs = len(groups) * len(sell_confirm_values) * len(max_positions)

    print("Nasdaq-100 Daily 3y Exhaustive Strategy Search")
    print(f"Groups       : {len(groups)}")
    print(f"Total runs   : {total_runs}")
    print(f"Rank by      : {args.rank_by}")
    print("Downloading shared market data...", flush=True)
    market_data = download_for_config(base)
    full_idx = market_index(market_data)
    active_by_position = prepare_active_universe(market_data, full_idx)

    rows: list[dict[str, Any]] = []
    qqq_return_cache: dict[tuple[object, object], float] = {}
    started = time.monotonic()
    completed = 0
    for group_number, group_params in enumerate(groups, start=1):
        group_config = apply_config_overlay(base, group_params)
        strategy = create_strategy(group_config)
        prepared_template = strategy.prepare_backtest(
            market_data.bars,
            benchmark=market_data.benchmark,
            filter_benchmark=market_data.filter_benchmark,
            universe_schedule=market_data.universe_schedule,
        )
        row_positions = prepare_row_positions(prepared_template.prepared, full_idx)
        market_filter_states = prepare_market_filter_states(prepared_template, full_idx)
        candidates_by_position = prepare_candidate_lists(
            group_config,
            strategy,
            prepared_template.prepared,
            market_filter_states,
            active_by_position,
            row_positions,
            full_idx,
        )
        for sell_confirm in sell_confirm_values:
            confirm_config = apply_config_overlay(
                group_config,
                {"sell_confirm_bars": sell_confirm},
            )
            for position_count in max_positions:
                params = {
                    **group_params,
                    "sell_confirm_bars": sell_confirm,
                    "max_positions": position_count,
                }
                config = apply_config_overlay(confirm_config, {"max_positions": position_count})
                result = run_prepared_backtest(
                    config,
                    market_data,
                    prepared_template.prepared,
                    candidates_by_position,
                    market_filter_states,
                    active_by_position,
                    row_positions,
                    strategy.warmup_bars(),
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
                        "start": result.equity.index[0],
                        "end": result.equity.index[-1],
                    }
                )
                completed += 1
        if args.progress_every and group_number % args.progress_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"Progress: {completed}/{total_runs} runs "
                f"({group_number}/{len(groups)} groups), elapsed {elapsed:.1f}s",
                flush=True,
            )

    if rows:
        first = rows[0]
        print(f"Evaluation   : {first['start']} -> {first['end']}")
    if market_data.skipped:
        print(f"Skipped      : {', '.join(market_data.skipped)}")
    print_rows(rows, args.rank_by, args.top)


if __name__ == "__main__":
    main()
