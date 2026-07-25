from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..config import AppConfig
from ..portfolio import AccountSnapshot, OrderIntent, OrderPlan, estimate_quantity
from ..ranking import create_scorer
from .base import (
    BenchmarkInput,
    PreparedBacktest,
    build_prepared_report_frames,
    reject_unknown_params,
)
from .common import (
    benchmark_for_strategy_symbol,
    market_filter_allows_buy,
    precompute_market_filter_trends,
    scheduled_prepared_slice,
    trend_down_confirmed,
    with_supertrend,
)
from .registry import register_strategy


@dataclass(frozen=True)
class PreparedSimpleBacktest(PreparedBacktest):
    strategy: "SimpleSupertrendStrategy"
    prepared: dict[str, pd.DataFrame]
    market_filter_trends: dict[str, pd.Series]
    universe_schedule: tuple[Mapping[str, Any], ...] = ()

    def build_order_plan(
        self,
        signal_ts,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        prepared = scheduled_prepared_slice(
            self.prepared,
            signal_ts,
            account,
            self.universe_schedule,
        )
        market_filter_states = {
            symbol: _trend_is_up_at(trend, signal_ts)
            for symbol, trend in self.market_filter_trends.items()
        }
        return self.strategy._build_order_plan_from_prepared(
            prepared,
            account,
            mode,
            market_filter_states=market_filter_states,
        )

    def report_frames(self, symbols: set[str]) -> dict[str, pd.DataFrame]:
        return build_prepared_report_frames(
            self.prepared, self.market_filter_trends, symbols
        )


@register_strategy
class SimpleSupertrendStrategy:
    strategy_type = "simple_supertrend"

    def __init__(self, config: AppConfig):
        self.config = config
        self.scorer = create_scorer(config.scoring, config.market)

    @classmethod
    def validate_config(cls, config: AppConfig) -> None:
        reject_unknown_params(config.strategy.params, set(), cls.strategy_type)

    def warmup_bars(self) -> int:
        return max(2, self.scorer.warmup_bars())

    def prepare_backtest(
        self,
        bars: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput = None,
        filter_benchmark: BenchmarkInput = None,
        universe_schedule: tuple[Mapping[str, Any], ...] = (),
    ) -> PreparedSimpleBacktest:
        market_filter_data = filter_benchmark if filter_benchmark is not None else benchmark
        featured = {
            symbol: with_supertrend(self.config, symbol, frame)
            for symbol, frame in bars.items()
        }
        prepared = self.scorer.add_scores(featured, benchmark)
        market_filter_trends = precompute_market_filter_trends(
            self.config,
            list(bars),
            market_filter_data,
        )
        return PreparedSimpleBacktest(self, prepared, market_filter_trends, universe_schedule)

    def build_order_plan(
        self,
        bars: dict[str, pd.DataFrame],
        account: AccountSnapshot,
        mode: str,
        benchmark: BenchmarkInput = None,
        filter_benchmark: BenchmarkInput = None,
    ) -> OrderPlan:
        market_filter_data = filter_benchmark if filter_benchmark is not None else benchmark
        featured = {
            symbol: with_supertrend(self.config, symbol, frame)
            for symbol, frame in bars.items()
        }
        prepared = self.scorer.add_scores(featured, benchmark)
        return self._build_order_plan_from_prepared(
            prepared,
            account,
            mode,
            filter_benchmark=market_filter_data,
        )

    def _build_order_plan_from_prepared(
        self,
        prepared: dict[str, pd.DataFrame],
        account: AccountSnapshot,
        mode: str,
        filter_benchmark: BenchmarkInput = None,
        market_filter_states: dict[str, bool] | None = None,
    ) -> OrderPlan:
        config = self.config
        orders: list[OrderIntent] = []
        max_positions = config.risk.max_position_count
        open_slots = max(0, max_positions - account.total_position_count)
        per_slot_allocation = config.execution.allocation_pct / max(1, open_slots)
        candidate_scores: dict[str, object] = {}
        candidate_prices: dict[str, float] = {}

        for symbol, st_df in prepared.items():
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
                _market_filter_allows_buy(
                    config,
                    symbol,
                    filter_benchmark,
                    market_filter_states,
                )
                and open_slots > 0
                and not position
                and bool(row["BuySignal"])
            ):
                candidate_scores[symbol] = row.get("Score")
                candidate_prices[symbol] = price

        for symbol in self.scorer.rank(candidate_scores):
            if open_slots <= 0:
                break
            qty = estimate_quantity(
                account.cash,
                candidate_prices[symbol],
                per_slot_allocation,
                fee_rate=config.costs.fee_rate,
                slippage_rate=config.costs.slippage_rate,
            )
            if qty > 0:
                orders.append(
                    OrderIntent(
                        symbol=symbol,
                        side="buy",
                        quantity=qty,
                        order_type=config.execution.order_type,
                        reason="Top-ranked Supertrend entry",
                    )
                )
                open_slots -= 1

        return OrderPlan(strategy_name=config.strategy.name, mode=mode, orders=tuple(orders))


def _market_filter_allows_buy(
    config: AppConfig,
    symbol: str,
    filter_benchmark: BenchmarkInput,
    market_filter_states: dict[str, bool] | None,
) -> bool:
    if market_filter_states is not None:
        return not config.market_trend_filter.enabled or market_filter_states.get(symbol, False)
    return market_filter_allows_buy(
        config,
        benchmark_for_strategy_symbol(symbol, filter_benchmark),
    )


def _trend_is_up_at(trend: pd.Series, signal_ts) -> bool:
    if trend.empty:
        return False
    try:
        available = trend.loc[:signal_ts]
    except TypeError:
        signal_date = pd.Timestamp(signal_ts).date()
        available = trend.loc[
            [pd.Timestamp(timestamp).date() <= signal_date for timestamp in trend.index]
        ]
    if available.empty:
        return False
    return int(available.iloc[-1]) == 1
