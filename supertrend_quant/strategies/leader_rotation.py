from __future__ import annotations

import pandas as pd

from ..config import AppConfig
from ..portfolio import AccountSnapshot, OrderIntent, OrderPlan, estimate_quantity
from .base import BenchmarkInput, reject_unknown_params
from .common import (
    benchmark_for_strategy_symbol,
    market_filter_allows_buy,
    sell_all,
    trend_down_confirmed,
    with_supertrend,
)
from .registry import register_strategy


@register_strategy
class LeaderRotationStrategy:
    strategy_type = "leader_rotation"

    def __init__(self, config: AppConfig):
        self.config = config

    @classmethod
    def validate_config(cls, config: AppConfig) -> None:
        reject_unknown_params(config.strategy.params, set(), cls.strategy_type)
        if config.risk.max_position_count != 1:
            raise ValueError("leader_rotation currently supports portfolio.max_positions: 1 only.")

    def warmup_bars(self) -> int:
        return effective_rs_period(self.config) + 1

    def build_order_plan(
        self,
        bars: dict[str, pd.DataFrame],
        account: AccountSnapshot,
        mode: str,
        benchmark: BenchmarkInput = None,
        filter_benchmark: BenchmarkInput = None,
    ) -> OrderPlan:
        config = self.config
        market_filter_data = filter_benchmark if filter_benchmark is not None else benchmark
        prepared = self._prepare_leader_data(bars, benchmark)
        if not prepared:
            return OrderPlan(config.strategy.name, mode, (), ("No prepared symbol data.",))

        candidates = self._leader_candidates(prepared, market_filter_data)
        orders: list[OrderIntent] = []
        held = next(iter(account.positions.values()), None)

        if held and held.quantity > 0:
            held_df = prepared.get(held.symbol)
            if held_df is None or held_df.empty:
                orders.append(sell_all(held, "Held symbol missing from strategy data"))
            else:
                held_row = held_df.iloc[-1]
                best_new = next((candidate for candidate in candidates if candidate["symbol"] != held.symbol), None)
                sell_reason = None
                follow_up_buy = None
                if trend_down_confirmed(held_df, config.exit.sell_confirm_bars):
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
                    orders.append(sell_all(held, sell_reason))
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
        self,
        bars: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        if benchmark is None:
            return {}
        prepared: dict[str, pd.DataFrame] = {}
        config = self.config
        rs_period = effective_rs_period(config)

        for symbol, df in bars.items():
            symbol_benchmark = benchmark_for_strategy_symbol(symbol, benchmark)
            if symbol_benchmark is None or symbol_benchmark.empty:
                continue
            bench_return = symbol_benchmark["Close"].pct_change(rs_period)
            st_df = with_supertrend(config, symbol, df)
            aligned_bench_return = bench_return.reindex(st_df.index, method="ffill")
            st_df["RS"] = st_df["Close"].pct_change(rs_period) - aligned_bench_return
            prepared[symbol] = st_df.dropna(subset=["RS", "ATR_pct"])
        return prepared

    def _leader_candidates(
        self,
        prepared: dict[str, pd.DataFrame],
        filter_benchmark: BenchmarkInput = None,
    ) -> list[dict[str, float | str]]:
        config = self.config
        rows = []
        for symbol, df in prepared.items():
            if df.empty:
                continue
            if not market_filter_allows_buy(config, benchmark_for_strategy_symbol(symbol, filter_benchmark)):
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


def effective_rs_period(config: AppConfig) -> int:
    return int(config.leader_rotation.rs_period_by_market.get(config.market, config.leader_rotation.rs_period))
