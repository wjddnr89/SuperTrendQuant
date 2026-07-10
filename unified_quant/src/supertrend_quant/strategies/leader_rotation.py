from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import AppConfig
from ..portfolio import AccountSnapshot, OrderIntent, OrderPlan, estimate_quantity
from .base import BenchmarkInput, PreparedBacktest, reject_unknown_params
from .common import (
    asset_filters_allow_buy,
    benchmark_for_strategy_symbol,
    configured_exit_down_confirmed,
    enabled_component,
    entry_state_allows_buy,
    market_filter_allows_buy,
    precompute_market_filter_trends,
    sell_all,
    with_strategy_components,
)
from .registry import register_strategy


@dataclass(frozen=True)
class PreparedLeaderBacktest(PreparedBacktest):
    strategy: "LeaderRotationStrategy"
    prepared: dict[str, pd.DataFrame]
    market_filter_trends: dict[str, pd.Series]

    def build_order_plan(
        self,
        signal_ts,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        prepared = {
            symbol: frame.loc[:signal_ts]
            for symbol, frame in self.prepared.items()
        }
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

    @classmethod
    def validate_config(cls, config: AppConfig) -> None:
        reject_unknown_params(config.strategy.params, set(), cls.strategy_type)
        if config.risk.max_position_count != 1:
            raise ValueError("leader_rotation currently supports portfolio.max_positions: 1 only.")

    def warmup_bars(self) -> int:
        warmup = max(effective_rs_period(self.config) + 1, self.config.supertrend.period)
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
    ) -> PreparedLeaderBacktest:
        market_filter_data = filter_benchmark if filter_benchmark is not None else benchmark
        prepared = self._prepare_leader_data(bars, benchmark)
        market_filter_trends = precompute_market_filter_trends(
            self.config,
            list(bars),
            market_filter_data,
        )
        return PreparedLeaderBacktest(self, prepared, market_filter_trends)

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
                if configured_exit_down_confirmed(config, held_df):
                    sell_reason = (
                        "Triple Supertrend down"
                        if enabled_component(config, "exits", "triple_supertrend_flip") is not None
                        else "Supertrend down"
                    )
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
                        estimated_sell_price = float(held_row["Close"]) * (1.0 - config.costs.slippage_rate)
                        estimated_proceeds = (
                            held.quantity
                            * max(0.0, estimated_sell_price)
                            * (1.0 - config.costs.fee_rate)
                        )
                        estimated_cash = account.cash + max(0.0, estimated_proceeds)
                        qty = estimate_quantity(
                            estimated_cash,
                            follow_up_buy["price"],
                            config.execution.allocation_pct,
                            fee_rate=config.costs.fee_rate,
                            slippage_rate=config.costs.slippage_rate,
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
                fee_rate=config.costs.fee_rate,
                slippage_rate=config.costs.slippage_rate,
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
            feature_df = with_strategy_components(config, symbol, df)
            aligned_bench_return = bench_return.reindex(feature_df.index, method="ffill")
            feature_df["RS"] = feature_df["Close"].pct_change(rs_period) - aligned_bench_return
            prepared[symbol] = feature_df.dropna(subset=["RS", "ATR_pct"])
        return prepared

    def _leader_candidates(
        self,
        prepared: dict[str, pd.DataFrame],
        filter_benchmark: BenchmarkInput = None,
        market_filter_states: dict[str, bool] | None = None,
    ) -> list[dict[str, float | str]]:
        config = self.config
        rows = []
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
