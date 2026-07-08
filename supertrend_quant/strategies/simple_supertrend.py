from __future__ import annotations

import pandas as pd

from ..config import AppConfig
from ..portfolio import AccountSnapshot, OrderIntent, OrderPlan, estimate_quantity
from .base import BenchmarkInput, reject_unknown_params
from .common import benchmark_for_strategy_symbol, market_filter_allows_buy, trend_down_confirmed, with_supertrend
from .registry import register_strategy


@register_strategy
class SimpleSupertrendStrategy:
    strategy_type = "simple_supertrend"

    def __init__(self, config: AppConfig):
        self.config = config

    @classmethod
    def validate_config(cls, config: AppConfig) -> None:
        reject_unknown_params(config.strategy.params, set(), cls.strategy_type)

    def warmup_bars(self) -> int:
        return 2

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
        orders: list[OrderIntent] = []
        max_positions = config.risk.max_position_count
        open_slots = max(0, max_positions - account.total_position_count)

        for symbol, df in bars.items():
            st_df = with_supertrend(config, symbol, df)
            if len(st_df) < 2:
                continue
            row = st_df.iloc[-1]
            position = account.positions.get(symbol)
            price = float(row["Close"])

            if position and position.quantity > 0 and trend_down_confirmed(st_df, config.exit.sell_confirm_bars):
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
                market_filter_allows_buy(config, benchmark_for_strategy_symbol(symbol, market_filter_data))
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
