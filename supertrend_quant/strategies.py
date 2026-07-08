from __future__ import annotations

import pandas as pd

from .config import AppConfig
from .indicators import calculate_supertrend
from .portfolio import AccountSnapshot, OrderIntent, OrderPlan, estimate_quantity


BenchmarkInput = pd.DataFrame | dict[str, pd.DataFrame] | None


def build_order_plan(
    config: AppConfig,
    bars: dict[str, pd.DataFrame],
    account: AccountSnapshot,
    mode: str,
    benchmark: BenchmarkInput = None,
    filter_benchmark: BenchmarkInput = None,
) -> OrderPlan:
    market_filter_data = filter_benchmark if filter_benchmark is not None else benchmark
    if config.strategy.type == "simple_supertrend":
        return simple_supertrend_plan(config, bars, account, mode, market_filter_data)
    if config.strategy.type == "leader_rotation":
        return leader_rotation_plan(config, bars, account, mode, benchmark, market_filter_data)
    raise ValueError(f"Unsupported strategy type: {config.strategy.type}")


def simple_supertrend_plan(
    config: AppConfig,
    bars: dict[str, pd.DataFrame],
    account: AccountSnapshot,
    mode: str,
    benchmark: BenchmarkInput = None,
) -> OrderPlan:
    orders: list[OrderIntent] = []
    max_positions = config.risk.max_position_count
    open_slots = max(0, max_positions - account.total_position_count)

    for symbol, df in bars.items():
        st_df = _with_supertrend(config, symbol, df)
        if len(st_df) < 2:
            continue
        row = st_df.iloc[-1]
        position = account.positions.get(symbol)
        price = float(row["Close"])

        if position and position.quantity > 0 and _trend_down_confirmed(st_df, config.exit.sell_confirm_bars):
            orders.append(
                OrderIntent(
                    symbol=symbol,
                    side="sell",
                    quantity=position.quantity,
                    order_type=config.execution.order_type,
                    reason="Supertrend SellSignal",
                )
            )
        elif (
            _market_filter_allows_buy(config, benchmark=_benchmark_for_symbol(symbol, benchmark))
            and open_slots > 0
            and not position
            and bool(row["BuySignal"])
        ):
            qty = estimate_quantity(account.cash, price, config.execution.allocation_pct / max(1, open_slots))
            if qty > 0:
                orders.append(
                    OrderIntent(
                        symbol=symbol,
                        side="buy",
                        quantity=qty,
                        order_type=config.execution.order_type,
                        reason="Supertrend BuySignal",
                    )
                )
                open_slots -= 1

    return OrderPlan(strategy_name=config.strategy.name, mode=mode, orders=tuple(orders))


def leader_rotation_plan(
    config: AppConfig,
    bars: dict[str, pd.DataFrame],
    account: AccountSnapshot,
    mode: str,
    benchmark: BenchmarkInput,
    filter_benchmark: BenchmarkInput = None,
) -> OrderPlan:
    prepared = _prepare_leader_data(config, bars, benchmark)
    if not prepared:
        return OrderPlan(config.strategy.name, mode, (), ("No prepared symbol data.",))

    candidates = _leader_candidates(config, prepared, filter_benchmark)
    orders: list[OrderIntent] = []
    held = next(iter(account.positions.values()), None)

    if held and held.quantity > 0:
        held_df = prepared.get(held.symbol)
        if held_df is None or held_df.empty:
            orders.append(_sell_all(held, "Held symbol missing from strategy data"))
        else:
            held_row = held_df.iloc[-1]
            best_new = next((candidate for candidate in candidates if candidate["symbol"] != held.symbol), None)
            sell_reason = None
            follow_up_buy = None
            if _trend_down_confirmed(held_df, config.exit.sell_confirm_bars):
                sell_reason = "Supertrend down"
            elif best_new:
                current_rs = float(held_row["RS"]) if not pd.isna(held_row["RS"]) else -999.0
                hurdle = best_new["atr_pct"] * config.leader_rotation.hurdle_atr_mult
                if best_new["rs"] - current_rs > hurdle:
                    profit_pct = (
                        (float(held_row["Close"]) - held.avg_price) / held.avg_price
                        if held.avg_price > 0
                        else 0.0
                    )
                    if profit_pct >= config.leader_rotation.min_rotation_profit_pct:
                        sell_reason = "Leader rotation"
                        follow_up_buy = best_new
            if sell_reason:
                orders.append(_sell_all(held, sell_reason))
                if follow_up_buy is None and candidates:
                    follow_up_buy = next((candidate for candidate in candidates if candidate["symbol"] != held.symbol), None)
                if follow_up_buy:
                    estimated_cash = account.cash + held.quantity * float(held_row["Close"])
                    qty = estimate_quantity(
                        estimated_cash,
                        follow_up_buy["price"],
                        config.execution.allocation_pct,
                    )
                    if qty > 0:
                        orders.append(
                            OrderIntent(
                                symbol=follow_up_buy["symbol"],
                                side="buy",
                                quantity=qty,
                                order_type=config.execution.order_type,
                                reason="Post-sell leader entry",
                            )
                        )

    if not held and candidates:
        best = candidates[0]
        qty = estimate_quantity(
            account.cash,
            best["price"],
            config.execution.allocation_pct,
        )
        if qty > 0:
            orders.append(
                OrderIntent(
                    symbol=best["symbol"],
                    side="buy",
                    quantity=qty,
                    order_type=config.execution.order_type,
                    reason="Top RS leader",
                )
            )

    return OrderPlan(strategy_name=config.strategy.name, mode=mode, orders=tuple(orders))


def _prepare_leader_data(
    config: AppConfig,
    bars: dict[str, pd.DataFrame],
    benchmark: BenchmarkInput,
) -> dict[str, pd.DataFrame]:
    if benchmark is None:
        return {}
    prepared: dict[str, pd.DataFrame] = {}
    rs_period = effective_rs_period(config)

    for symbol, df in bars.items():
        symbol_benchmark = _benchmark_for_symbol(symbol, benchmark)
        if symbol_benchmark is None or symbol_benchmark.empty:
            continue
        bench_return = symbol_benchmark["Close"].pct_change(rs_period)
        st_df = _with_supertrend(config, symbol, df)
        aligned_bench_return = bench_return.reindex(st_df.index, method="ffill")
        st_df["RS"] = st_df["Close"].pct_change(rs_period) - aligned_bench_return
        prepared[symbol] = st_df.dropna(subset=["RS", "ATR_pct"])
    return prepared


def _leader_candidates(
    config: AppConfig,
    prepared: dict[str, pd.DataFrame],
    filter_benchmark: BenchmarkInput = None,
) -> list[dict[str, float | str]]:
    rows = []
    for symbol, df in prepared.items():
        if df.empty:
            continue
        if not _market_filter_allows_buy(config, _benchmark_for_symbol(symbol, filter_benchmark)):
            continue
        row = df.iloc[-1]
        if int(row["Trend"]) != 1:
            continue
        if not config.leader_rotation.allow_late_chase and not bool(row["BuySignal"]):
            continue
        rows.append(
            {
                "symbol": symbol,
                "rs": float(row["RS"]),
                "atr_pct": float(row["ATR_pct"]),
                "price": float(row["Close"]),
            }
        )
    return sorted(rows, key=lambda item: item["rs"], reverse=True)


def _market_filter_allows_buy(config: AppConfig, benchmark: pd.DataFrame | None) -> bool:
    if not config.market_trend_filter.enabled:
        return True
    if benchmark is None or benchmark.empty:
        return False
    st_df = calculate_supertrend(
        benchmark,
        period=config.supertrend.period,
        multiplier=config.supertrend.multiplier,
        atr_method=config.supertrend.atr_method,
    )
    if st_df.empty:
        return False
    return int(st_df["Trend"].iloc[-1]) == 1


def _benchmark_for_symbol(symbol: str, benchmark: BenchmarkInput) -> pd.DataFrame | None:
    if benchmark is None:
        return None
    if isinstance(benchmark, dict):
        return benchmark.get(symbol)
    return benchmark


def _with_supertrend(config: AppConfig, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    if not config.supertrend.enabled:
        out = df.copy()
        out["Trend"] = 1
        out["BuySignal"] = False
        out["SellSignal"] = False
        out["ATR_pct"] = 0.0
        return out
    multiplier = config.supertrend.symbol_multipliers.get(symbol, config.supertrend.multiplier)
    return calculate_supertrend(
        df,
        period=config.supertrend.period,
        multiplier=multiplier,
        atr_method=config.supertrend.atr_method,
    )


def effective_rs_period(config: AppConfig) -> int:
    return int(config.leader_rotation.rs_period_by_market.get(config.market, config.leader_rotation.rs_period))


def _trend_down_confirmed(df: pd.DataFrame, confirm_bars: int) -> bool:
    bars = max(1, int(confirm_bars))
    if len(df) < bars:
        return False
    return bool((df["Trend"].tail(bars).astype(int) == -1).all())


def _sell_all(position, reason: str) -> OrderIntent:
    return OrderIntent(
        symbol=position.symbol,
        side="sell",
        quantity=position.quantity,
        reason=reason,
    )
