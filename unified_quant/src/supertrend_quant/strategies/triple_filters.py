from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
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
    asset_filters_allow_buy,
    active_universe_symbols,
    benchmark_for_strategy_symbol,
    configured_exit_down_confirmed,
    enabled_component,
    market_filter_allows_buy,
    precompute_market_filter_trends,
    scheduled_prepared_slice,
    with_strategy_components,
)
from .registry import register_strategy


@dataclass(frozen=True)
class PreparedTripleFiltersBacktest(PreparedBacktest):
    strategy: "TripleFiltersStrategy"
    prepared: dict[str, pd.DataFrame]
    market_filter_trends: dict[str, pd.Series]
    universe_schedule: tuple[Mapping[str, Any], ...] = ()
    fast_values: dict[str, tuple[pd.Index, dict[str, np.ndarray]]] | None = None

    def build_order_plan(
        self,
        signal_ts,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        market_filter_states = {
            symbol: _trend_is_up_at(trend, signal_ts)
            for symbol, trend in self.market_filter_trends.items()
        }
        if self.fast_values is not None:
            return self.strategy._build_order_plan_fast(
                self.fast_values,
                signal_ts,
                account,
                mode,
                self.universe_schedule,
                market_filter_states,
            )
        triple_exit = enabled_component(self.strategy.config, "exits", "triple_supertrend_flip")
        tail_bars = max(
            1,
            int(self.strategy.config.exit.sell_confirm_bars),
            int(triple_exit.params.get("confirm_bars", 1)) if triple_exit is not None else 1,
        )
        prepared = scheduled_prepared_slice(
            self.prepared,
            signal_ts,
            account,
            self.universe_schedule,
            tail_bars=tail_bars,
        )
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
class TripleFiltersStrategy:
    """Rank eligible entries with the configured scorer without rotating holdings."""

    strategy_type = "triple_filters"

    def __init__(self, config: AppConfig):
        self.config = config
        self.scorer = create_scorer(config.scoring, config.market)

    @classmethod
    def validate_config(cls, config: AppConfig) -> None:
        reject_unknown_params(config.strategy.params, set(), cls.strategy_type)
        if enabled_component(config, "entries", "triple_supertrend") is None:
            raise ValueError("triple_filters requires an enabled triple_supertrend entry.")

    def warmup_bars(self) -> int:
        warmup = self.scorer.warmup_bars()
        triple = enabled_component(self.config, "entries", "triple_supertrend")
        if triple is not None:
            settings = triple.params.get("settings", ())
            periods = [int(item.get("period", 1)) for item in settings if isinstance(item, dict)]
            if periods:
                warmup = max(warmup, max(periods))
        ichimoku = enabled_component(self.config, "filters", "ichimoku_cloud")
        if ichimoku is not None:
            warmup = max(
                warmup,
                int(ichimoku.params.get("span_b", 52))
                + int(ichimoku.params.get("shift", 26)),
            )
        ema = enabled_component(self.config, "filters", "ema_trend")
        if ema is not None:
            warmup = max(warmup, int(ema.params.get("period", 200)))
        return warmup

    def prepare_backtest(
        self,
        bars: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput = None,
        filter_benchmark: BenchmarkInput = None,
        universe_schedule: tuple[Mapping[str, Any], ...] = (),
    ) -> PreparedTripleFiltersBacktest:
        market_filter_data = filter_benchmark if filter_benchmark is not None else benchmark
        prepared = self._prepare_scored_data(bars, benchmark)
        market_filter_trends = precompute_market_filter_trends(
            self.config,
            list(bars),
            market_filter_data,
        )
        fast_values = {
            symbol: (
                frame.index,
                {
                    column: frame[column].to_numpy(copy=False)
                    for column in (
                        "Close",
                        "Score",
                        "TripleAllUp",
                        "TripleDownCount",
                        "Ichimoku_LongOk",
                        "EMA_LongOk",
                    )
                    if column in frame
                },
            )
            for symbol, frame in prepared.items()
        }
        return PreparedTripleFiltersBacktest(
            self, prepared, market_filter_trends, universe_schedule, fast_values
        )

    def _build_order_plan_fast(
        self,
        prepared: dict[str, tuple[pd.Index, dict[str, np.ndarray]]],
        signal_ts,
        account: AccountSnapshot,
        mode: str,
        universe_schedule: tuple[Mapping[str, Any], ...],
        market_filter_states: dict[str, bool],
    ) -> OrderPlan:
        config = self.config
        active = active_universe_symbols(universe_schedule, signal_ts)
        allowed = None if active is None else active | set(account.positions)
        open_slots = max(0, config.risk.max_position_count - account.total_position_count)
        per_slot_allocation = config.execution.allocation_pct / max(1, open_slots)
        triple_exit = enabled_component(config, "exits", "triple_supertrend_flip")
        confirm_bars = max(
            1,
            int(
                triple_exit.params.get("confirm_bars", config.exit.sell_confirm_bars)
                if triple_exit is not None
                else config.exit.sell_confirm_bars
            ),
        )
        down_count = int(triple_exit.params.get("down_count", 2)) if triple_exit else 1
        ichimoku_enabled = enabled_component(config, "filters", "ichimoku_cloud") is not None
        ema_enabled = enabled_component(config, "filters", "ema_trend") is not None
        orders: list[OrderIntent] = []
        candidate_scores: dict[str, object] = {}
        candidate_prices: dict[str, float] = {}
        timestamp = pd.Timestamp(signal_ts)

        for symbol, (index, values) in prepared.items():
            if allowed is not None and symbol not in allowed:
                continue
            position_index = int(index.searchsorted(timestamp, side="right")) - 1
            if position_index < 0:
                continue
            position = account.positions.get(symbol)
            price = float(values["Close"][position_index])
            if position and position.quantity > 0:
                start = position_index - confirm_bars + 1
                down = values.get("TripleDownCount")
                should_exit = (
                    down is not None
                    and start >= 0
                    and bool(np.all(np.asarray(down[start : position_index + 1], dtype=float) >= down_count))
                )
                if should_exit:
                    orders.append(
                        OrderIntent(
                            symbol=symbol,
                            side="sell",
                            quantity=position.quantity,
                            order_type=config.execution.order_type,
                            reason="Triple Supertrend down",
                        )
                    )
                continue
            if config.market_trend_filter.enabled and not market_filter_states.get(symbol, False):
                continue
            if not bool(values["TripleAllUp"][position_index]):
                continue
            if ichimoku_enabled and not bool(values["Ichimoku_LongOk"][position_index]):
                continue
            if ema_enabled and not bool(values["EMA_LongOk"][position_index]):
                continue
            candidate_scores[symbol] = values["Score"][position_index]
            candidate_prices[symbol] = price

        for symbol in self.scorer.rank(candidate_scores):
            if open_slots <= 0:
                break
            quantity = estimate_quantity(
                account.cash,
                candidate_prices[symbol],
                per_slot_allocation,
                fee_rate=config.costs.fee_rate,
                slippage_rate=config.costs.slippage_rate,
            )
            if quantity > 0:
                orders.append(
                    OrderIntent(
                        symbol=symbol,
                        side="buy",
                        quantity=quantity,
                        order_type=config.execution.order_type,
                        reason="Top-ranked triple-filter entry",
                    )
                )
                open_slots -= 1
        return OrderPlan(strategy_name=config.strategy.name, mode=mode, orders=tuple(orders))

    def build_order_plan(
        self,
        bars: dict[str, pd.DataFrame],
        account: AccountSnapshot,
        mode: str,
        benchmark: BenchmarkInput = None,
        filter_benchmark: BenchmarkInput = None,
    ) -> OrderPlan:
        market_filter_data = filter_benchmark if filter_benchmark is not None else benchmark
        prepared = self._prepare_scored_data(bars, benchmark)
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
        open_slots = max(0, config.risk.max_position_count - account.total_position_count)
        per_slot_allocation = config.execution.allocation_pct / max(1, open_slots)

        candidate_scores: dict[str, object] = {}
        candidate_prices: dict[str, float] = {}
        for symbol, feature_df in prepared.items():
            if feature_df.empty:
                continue
            row = feature_df.iloc[-1]
            position = account.positions.get(symbol)
            price = float(row["Close"])

            if position and position.quantity > 0:
                if configured_exit_down_confirmed(config, feature_df):
                    orders.append(
                        OrderIntent(
                            symbol=symbol,
                            side="sell",
                            quantity=position.quantity,
                            order_type=config.execution.order_type,
                            reason="Triple Supertrend down",
                        )
                    )
                continue

            if (
                _market_filter_allows_buy(
                    config,
                    symbol,
                    filter_benchmark,
                    market_filter_states,
                )
                and asset_filters_allow_buy(config, row)
                and bool(row.get("TripleAllUp", False))
            ):
                candidate_scores[symbol] = row.get("Score")
                candidate_prices[symbol] = price

        for symbol in self.scorer.rank(candidate_scores):
            if open_slots <= 0:
                break
            quantity = estimate_quantity(
                account.cash,
                candidate_prices[symbol],
                per_slot_allocation,
                fee_rate=config.costs.fee_rate,
                slippage_rate=config.costs.slippage_rate,
            )
            if quantity > 0:
                orders.append(
                    OrderIntent(
                        symbol=symbol,
                        side="buy",
                        quantity=quantity,
                        order_type=config.execution.order_type,
                        reason="Top-ranked triple-filter entry",
                    )
                )
                open_slots -= 1

        return OrderPlan(strategy_name=config.strategy.name, mode=mode, orders=tuple(orders))

    def _prepare_scored_data(
        self,
        bars: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        featured = {
            symbol: with_strategy_components(self.config, symbol, frame)
            for symbol, frame in bars.items()
        }
        return self.scorer.add_scores(featured, benchmark)


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
