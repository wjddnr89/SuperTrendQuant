from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..config import AppConfig
from ..portfolio import AccountSnapshot, OrderIntent, OrderPlan, estimate_quantity
from ..ranking import create_scorer
from .base import BenchmarkInput, PreparedBacktest, reject_unknown_params
from .common import (
    asset_filters_allow_buy,
    benchmark_for_strategy_symbol,
    configured_exit_down_confirmed,
    enabled_component,
    entry_state_allows_buy,
    market_filter_allows_buy,
    precompute_market_filter_trends,
    scheduled_prepared_slice,
    sell_all,
    with_strategy_components,
)
from .registry import register_strategy


@dataclass(frozen=True)
class PreparedLeaderBacktest(PreparedBacktest):
    strategy: "LeaderRotationStrategy"
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


@register_strategy
class LeaderRotationStrategy:
    strategy_type = "leader_rotation"

    def __init__(self, config: AppConfig):
        self.config = config
        self.scorer = create_scorer(config.scoring, config.market)

    @classmethod
    def validate_config(cls, config: AppConfig) -> None:
        reject_unknown_params(config.strategy.params, set(), cls.strategy_type)
        if config.risk.max_position_count < 1:
            raise ValueError("leader_rotation requires portfolio.max_positions >= 1.")

    def warmup_bars(self) -> int:
        warmup = max(self.scorer.warmup_bars(), self.config.supertrend.period)
        triple = enabled_component(self.config, "entries", "triple_supertrend")
        if triple is not None:
            settings = triple.params.get("settings", ())
            periods = [int(item.get("period", 1)) for item in settings if isinstance(item, dict)]
            if periods:
                warmup = max(warmup, max(periods))
        ichimoku = enabled_component(self.config, "filters", "ichimoku_cloud")
        if ichimoku is not None:
            span_b = int(ichimoku.params.get("span_b", 52))
            shift = int(ichimoku.params.get("shift", 26))
            warmup = max(warmup, span_b + shift)
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
    ) -> PreparedLeaderBacktest:
        market_filter_data = filter_benchmark if filter_benchmark is not None else benchmark
        prepared = self._prepare_leader_data(bars, benchmark)
        market_filter_trends = precompute_market_filter_trends(
            self.config,
            list(bars),
            market_filter_data,
        )
        return PreparedLeaderBacktest(self, prepared, market_filter_trends, universe_schedule)

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
        if not prepared:
            return OrderPlan(config.strategy.name, mode, (), ("No prepared symbol data.",))

        candidates = self._leader_candidates(
            prepared,
            filter_benchmark,
            market_filter_states=market_filter_states,
        )
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
            if held_df is None or held_df.empty:
                orders.append(sell_all(held, "Held symbol missing from strategy data"))
                sell_symbols.add(symbol)
            else:
                held_row = held_df.iloc[-1]
                sell_reason = None
                if configured_exit_down_confirmed(config, held_df):
                    sell_reason = (
                        "Triple Supertrend down"
                        if enabled_component(config, "exits", "triple_supertrend_flip") is not None
                        else "Supertrend down"
                    )
                elif symbol not in target_symbols:
                    replacement = _first_replacement_candidate(
                        target_candidates,
                        held_symbols=set(held_positions),
                        sell_symbols=sell_symbols,
                    )
                    if replacement is None:
                        continue
                    current_score = _finite_float(held_row.get("Score"))
                    hurdle = replacement["atr_pct"] * config.leader_rotation.hurdle_atr_mult
                    if current_score is not None and replacement["score"] - current_score > hurdle:
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
                    estimated_cash += _estimated_sell_proceeds(held, float(held_row["Close"]), config)

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
                candidate["price"],
                1.0,
                fee_rate=config.costs.fee_rate,
                slippage_rate=config.costs.slippage_rate,
            )
            if qty <= 0:
                continue
            orders.append(
                OrderIntent(
                    symbol=candidate["symbol"],
                    side="buy",
                    quantity=qty,
                    order_type=config.execution.order_type,
                    reason=(
                        "Post-sell leader entry"
                        if sell_symbols
                        else "Top-ranked leader"
                    ),
                )
            )
            estimated_cost = _estimated_buy_cost(qty, float(candidate["price"]), config)
            estimated_cash = max(0.0, estimated_cash - estimated_cost)
            remaining_buy_budget = max(0.0, remaining_buy_budget - estimated_cost)
            open_slots -= 1

        return OrderPlan(strategy_name=config.strategy.name, mode=mode, orders=tuple(orders))

    def _prepare_leader_data(
        self,
        bars: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        featured = {
            symbol: with_strategy_components(self.config, symbol, frame)
            for symbol, frame in bars.items()
        }
        return self.scorer.add_scores(featured, benchmark)

    def _leader_candidates(
        self,
        prepared: dict[str, pd.DataFrame],
        filter_benchmark: BenchmarkInput = None,
        market_filter_states: dict[str, bool] | None = None,
    ) -> list[dict[str, float | str]]:
        config = self.config
        candidate_scores: dict[str, float] = {}
        candidates: dict[str, dict[str, float | str]] = {}
        for symbol, df in prepared.items():
            if df.empty:
                continue
            if config.market_trend_filter.enabled:
                if market_filter_states is not None:
                    if not market_filter_states.get(symbol, False):
                        continue
                elif not market_filter_allows_buy(
                    config,
                    benchmark_for_strategy_symbol(symbol, filter_benchmark),
                ):
                    continue
            row = df.iloc[-1]
            if not asset_filters_allow_buy(config, row):
                continue
            if not entry_state_allows_buy(config, row):
                continue
            score = _finite_float(row.get("Score"))
            atr_pct = _finite_float(row.get("ATR_pct"))
            price = _finite_float(row.get("Close"))
            if score is None or atr_pct is None or price is None:
                continue
            candidate_scores[symbol] = score
            candidates[symbol] = {
                "symbol": symbol,
                "score": score,
                "atr_pct": atr_pct,
                "price": price,
            }
        return [candidates[symbol] for symbol in self.scorer.rank(candidate_scores)]


def _finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_replacement_candidate(
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


def _estimated_sell_proceeds(position, price: float, config: AppConfig) -> float:
    fill = price * (1.0 - config.costs.slippage_rate)
    return position.quantity * max(0.0, fill) * (1.0 - config.costs.fee_rate)


def _estimated_buy_cost(quantity: float, price: float, config: AppConfig) -> float:
    fill = price * (1.0 + config.costs.slippage_rate)
    return quantity * max(0.0, fill) * (1.0 + config.costs.fee_rate)


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
