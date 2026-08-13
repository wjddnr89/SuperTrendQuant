from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from supertrend_quant.portfolio import (
    AccountSnapshot,
    OrderIntent,
    OrderPlan,
    estimate_quantity,
)
from supertrend_quant.strategies import leader_rotation as leader_rotation_module
from supertrend_quant.strategies.common import (
    active_universe_symbols,
    enabled_component,
    scheduled_prepared_slice,
    sell_all,
)
from supertrend_quant.strategies.leader_rotation import PreparedLeaderBacktest


LATE_CHASE_MODES = frozenset(
    {
        "fresh_only",
        "unlimited",
        "kijun_atr_capped",
    }
)


@dataclass(frozen=True)
class ExperimentalLeaderPolicy:
    """Research-only controls layered over canonical leader rotation."""

    rotation_profit_gate: str = "nonnegative"
    stop_loss_pct: float | None = None
    late_chase_mode: str = "unlimited"
    max_extension_atr: float | None = None

    def __post_init__(self) -> None:
        if self.rotation_profit_gate not in {"nonnegative", "off"}:
            raise ValueError(
                "rotation_profit_gate must be 'nonnegative' or 'off'."
            )
        if self.stop_loss_pct is not None:
            value = float(self.stop_loss_pct)
            if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
                raise ValueError("stop_loss_pct must be between 0 and 1.")
        if self.late_chase_mode not in LATE_CHASE_MODES:
            raise ValueError(
                "late_chase_mode must be fresh_only, unlimited, or "
                "kijun_atr_capped."
            )
        if self.late_chase_mode == "kijun_atr_capped":
            if self.max_extension_atr is None:
                raise ValueError(
                    "max_extension_atr is required for kijun_atr_capped."
                )
            value = float(self.max_extension_atr)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("max_extension_atr must be positive.")
        elif self.max_extension_atr is not None:
            raise ValueError(
                "max_extension_atr is valid only for kijun_atr_capped."
            )


def late_chase_allows_entry(
    row: pd.Series,
    *,
    mode: str,
    max_extension_atr: float | None = None,
) -> bool:
    """Return whether the single-SuperTrend entry state is allowed."""

    if int(row.get("Trend", 0)) != 1:
        return False

    fresh_signal = bool(row.get("BuySignal", False))
    if mode == "fresh_only":
        return fresh_signal
    if mode == "unlimited":
        return True
    if mode != "kijun_atr_capped":
        raise ValueError(f"Unsupported late-chase mode: {mode}")
    if fresh_signal:
        return True

    close = _finite_float(row.get("Close"))
    kijun = _finite_float(row.get("Ichimoku_Kijun"))
    atr = _finite_float(row.get("ATR"))
    cap = _finite_float(max_extension_atr)
    if close is None or kijun is None or atr is None or atr <= 0.0 or cap is None:
        return False
    extension_atr = max(0.0, close - kijun) / atr
    return extension_atr <= cap


class ExperimentalPreparedLeaderBacktest:
    """Stateful research adapter that leaves fills and accounting canonical.

    A fixed stop is evaluated from marked net liquidation return on the
    completed signal bar.  The canonical runner fills the resulting sell at the
    next session open.  A stopped symbol remains blocked until its next fresh
    SuperTrend buy signal, preventing an immediate re-entry into the same name.
    """

    def __init__(
        self,
        delegate: PreparedLeaderBacktest,
        policy: ExperimentalLeaderPolicy,
    ) -> None:
        self.delegate = delegate
        self.policy = policy
        self.blocked_symbols: set[str] = set()

    def report_frames(self, symbols: set[str]) -> dict[str, pd.DataFrame]:
        return self.delegate.report_frames(symbols)

    def build_order_plan(
        self,
        signal_ts,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        self._release_reset_symbols(signal_ts)
        source = self.delegate
        config = source.strategy.config
        triple_exit = enabled_component(config, "exits", "triple_supertrend_flip")
        tail_bars = max(
            1,
            int(config.exit.sell_confirm_bars),
            int(triple_exit.params.get("confirm_bars", 1))
            if triple_exit is not None
            else 1,
        )
        prepared = scheduled_prepared_slice(
            source.prepared,
            signal_ts,
            account,
            source.universe_schedule,
            tail_bars=tail_bars,
        )

        stopped = self._stopped_symbols(account)
        if stopped:
            self.blocked_symbols.update(stopped)
            orders = tuple(
                sell_all(
                    account.positions[symbol],
                    f"Fixed stop {float(self.policy.stop_loss_pct):.1%}",
                )
                for symbol in sorted(stopped)
            )
            return OrderPlan(
                strategy_name=config.strategy.name,
                mode=mode,
                orders=orders,
                notes=("Experimental fixed stop; replacement deferred one signal bar.",),
            )

        held_symbols = set(account.positions)
        filtered = {
            symbol: frame
            for symbol, frame in prepared.items()
            if symbol not in self.blocked_symbols or symbol in held_symbols
        }
        market_filter_states = {
            symbol: _trend_is_up_at(trend, signal_ts)
            for symbol, trend in source.market_filter_trends.items()
        }

        with patch.object(
            leader_rotation_module,
            "entry_state_allows_buy",
            self._entry_state_allows_buy,
        ):
            return source.strategy._build_order_plan_from_prepared(
                filtered,
                account,
                mode,
                market_filter_states=market_filter_states,
            )

    def _entry_state_allows_buy(self, config, row: pd.Series) -> bool:
        if enabled_component(config, "entries", "triple_supertrend") is not None:
            raise ValueError(
                "The experimental late-chase adapter currently supports only "
                "single SuperTrend entries."
            )
        return late_chase_allows_entry(
            row,
            mode=self.policy.late_chase_mode,
            max_extension_atr=self.policy.max_extension_atr,
        )

    def _stopped_symbols(self, account: AccountSnapshot) -> set[str]:
        threshold = self.policy.stop_loss_pct
        if threshold is None:
            return set()
        stopped: set[str] = set()
        for symbol, position in account.positions.items():
            if position.quantity <= 0:
                continue
            economics = account.position_economics.get(symbol)
            net_return = (
                _finite_float(economics.net_return_pct)
                if economics is not None
                else None
            )
            if net_return is not None and net_return <= -float(threshold):
                stopped.add(symbol)
        return stopped

    def _release_reset_symbols(self, signal_ts) -> None:
        released = {
            symbol
            for symbol in self.blocked_symbols
            if _fresh_buy_signal_at(self.delegate.prepared.get(symbol), signal_ts)
        }
        self.blocked_symbols.difference_update(released)


class ExperimentalSignalCache:
    """Precompute read-only daily signal state for repeated research runs."""

    def __init__(
        self,
        delegate: PreparedLeaderBacktest,
        full_index: pd.Index,
    ) -> None:
        self.delegate = delegate
        self.full_index = pd.Index(full_index)
        self.length = len(self.full_index)
        self.row_positions = {
            symbol: frame.index.searchsorted(self.full_index, side="right") - 1
            for symbol, frame in delegate.prepared.items()
        }
        self.active_by_position = self._prepare_active_universe()
        self.market_filter_states = self._prepare_market_filter_states()
        self.symbol_arrays = self._prepare_symbol_arrays()
        self.exit_down_states = self._prepare_exit_down_states()
        self._rankings: dict[
            tuple[str, float | None],
            list[tuple[str, ...]],
        ] = {}

    def position_at(self, signal_ts) -> int:
        position = int(
            self.full_index.searchsorted(pd.Timestamp(signal_ts), side="right")
        ) - 1
        if position < 0 or position >= self.length:
            raise IndexError(f"Signal timestamp is outside cached index: {signal_ts}")
        return position

    def candidates_at(
        self,
        position: int,
        policy: ExperimentalLeaderPolicy,
        blocked_symbols: set[str],
        held_symbols: set[str],
    ) -> list[dict[str, float | str]]:
        rankings = self._rankings_for(policy)
        active = self.active_by_position[position]
        allowed = (
            None
            if active is None
            else set(active) | set(held_symbols)
        )
        candidates: list[dict[str, float | str]] = []
        for symbol in rankings[position]:
            if allowed is not None and symbol not in allowed:
                continue
            if symbol in blocked_symbols and symbol not in held_symbols:
                continue
            values = self.symbol_arrays[symbol]
            candidates.append(
                {
                    "symbol": symbol,
                    "score": float(values["score"][position]),
                    "atr_pct": float(values["atr_pct"][position]),
                    "price": float(values["price"][position]),
                }
            )
        return candidates

    def fresh_buy_at(self, symbol: str, position: int) -> bool:
        values = self.symbol_arrays.get(symbol)
        return bool(values is not None and values["fresh"][position])

    def exit_down_at(self, symbol: str, position: int) -> bool:
        values = self.exit_down_states.get(symbol)
        return bool(values is not None and values[position])

    def value_at(self, symbol: str, position: int, field: str) -> float | None:
        values = self.symbol_arrays.get(symbol)
        if values is None or field not in values:
            return None
        return _finite_float(values[field][position])

    def has_data_at(self, symbol: str, position: int) -> bool:
        values = self.symbol_arrays.get(symbol)
        return bool(values is not None and values["valid"][position])

    def _rankings_for(
        self,
        policy: ExperimentalLeaderPolicy,
    ) -> list[tuple[str, ...]]:
        key = (policy.late_chase_mode, policy.max_extension_atr)
        cached = self._rankings.get(key)
        if cached is not None:
            return cached

        rankings: list[tuple[str, ...]] = []
        all_symbols = set(self.symbol_arrays)
        for position in range(self.length):
            scores: dict[str, float] = {}
            # Rank every valid symbol here.  Membership filtering is applied
            # at plan time so canonical behavior can keep evaluating a held
            # symbol after it leaves the point-in-time entry universe.
            for symbol in all_symbols:
                values = self.symbol_arrays.get(symbol)
                if values is None or not self._entry_ok(values, position, policy):
                    continue
                scores[symbol] = float(values["score"][position])
            rankings.append(tuple(self.delegate.strategy.scorer.rank(scores)))
        self._rankings[key] = rankings
        return rankings

    @staticmethod
    def _entry_ok(
        values: dict[str, np.ndarray],
        position: int,
        policy: ExperimentalLeaderPolicy,
    ) -> bool:
        if not bool(values["base_ok"][position]):
            return False
        fresh = bool(values["fresh"][position])
        if policy.late_chase_mode == "fresh_only":
            return fresh
        if policy.late_chase_mode == "unlimited" or fresh:
            return True
        close = float(values["price"][position])
        kijun = float(values["kijun"][position])
        atr = float(values["atr"][position])
        if not all(math.isfinite(value) for value in (close, kijun, atr)):
            return False
        if atr <= 0.0 or policy.max_extension_atr is None:
            return False
        return max(0.0, close - kijun) / atr <= policy.max_extension_atr

    def _prepare_active_universe(self) -> list[set[str] | None]:
        schedule = self.delegate.universe_schedule
        if not schedule:
            return [None] * self.length
        return [
            active_universe_symbols(schedule, timestamp)
            for timestamp in self.full_index
        ]

    def _prepare_market_filter_states(self) -> dict[str, np.ndarray]:
        states: dict[str, np.ndarray] = {}
        for symbol, trend in self.delegate.market_filter_trends.items():
            positions = trend.index.searchsorted(self.full_index, side="right") - 1
            valid = positions >= 0
            output = np.zeros(self.length, dtype=bool)
            values = pd.to_numeric(trend, errors="coerce").to_numpy(dtype=float)
            output[valid] = values[positions[valid]] == 1.0
            states[symbol] = output
        return states

    def _prepare_symbol_arrays(self) -> dict[str, dict[str, np.ndarray]]:
        config = self.delegate.strategy.config
        use_ichimoku = (
            enabled_component(config, "filters", "ichimoku_cloud") is not None
        )
        use_ema = enabled_component(config, "filters", "ema_trend") is not None
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for symbol, frame in self.delegate.prepared.items():
            positions = np.asarray(self.row_positions[symbol], dtype=np.int64)
            valid = (positions >= 0) & (positions < len(frame))
            valid_positions = positions[valid]
            values = {
                "valid": valid,
                "score": self._float_values(frame, valid, valid_positions, "Score"),
                "atr_pct": self._float_values(
                    frame, valid, valid_positions, "ATR_pct"
                ),
                "price": self._float_values(frame, valid, valid_positions, "Close"),
                "atr": self._float_values(frame, valid, valid_positions, "ATR"),
                "kijun": self._float_values(
                    frame, valid, valid_positions, "Ichimoku_Kijun"
                ),
                "trend": self._float_values(frame, valid, valid_positions, "Trend"),
                "fresh": self._bool_values(
                    frame, valid, valid_positions, "BuySignal"
                ),
            }
            base_ok = valid.copy()
            base_ok &= values["trend"] == 1.0
            if config.market_trend_filter.enabled:
                base_ok &= self.market_filter_states.get(
                    symbol, np.zeros(self.length, dtype=bool)
                )
            if use_ichimoku:
                base_ok &= self._bool_values(
                    frame, valid, valid_positions, "Ichimoku_LongOk"
                )
            if use_ema:
                base_ok &= self._bool_values(
                    frame, valid, valid_positions, "EMA_LongOk"
                )
            base_ok &= (
                np.isfinite(values["score"])
                & np.isfinite(values["atr_pct"])
                & np.isfinite(values["price"])
            )
            values["base_ok"] = base_ok
            arrays[symbol] = values
        return arrays

    def _prepare_exit_down_states(self) -> dict[str, np.ndarray]:
        config = self.delegate.strategy.config
        triple_exit = enabled_component(config, "exits", "triple_supertrend_flip")
        if triple_exit is not None:
            raise ValueError(
                "Fast experimental cache currently supports single SuperTrend exits."
            )
        confirm_bars = max(1, int(config.exit.sell_confirm_bars))
        states: dict[str, np.ndarray] = {}
        for symbol, frame in self.delegate.prepared.items():
            positions = np.asarray(self.row_positions[symbol], dtype=np.int64)
            valid = positions >= 0
            raw_down = pd.to_numeric(
                frame.get("Trend"), errors="coerce"
            ).eq(-1)
            if confirm_bars > 1:
                confirmed = (
                    raw_down.rolling(
                        confirm_bars,
                        min_periods=confirm_bars,
                    )
                    .sum()
                    .eq(confirm_bars)
                )
            else:
                confirmed = raw_down
            output = np.zeros(self.length, dtype=bool)
            source = confirmed.fillna(False).to_numpy(dtype=bool)
            output[valid] = source[positions[valid]]
            states[symbol] = output
        return states

    def _float_values(
        self,
        frame: pd.DataFrame,
        valid: np.ndarray,
        valid_positions: np.ndarray,
        column: str,
    ) -> np.ndarray:
        output = np.full(self.length, np.nan, dtype=float)
        if column in frame:
            source = pd.to_numeric(
                frame[column], errors="coerce"
            ).to_numpy(dtype=float)
            output[valid] = source[valid_positions]
        return output

    def _bool_values(
        self,
        frame: pd.DataFrame,
        valid: np.ndarray,
        valid_positions: np.ndarray,
        column: str,
    ) -> np.ndarray:
        output = np.zeros(self.length, dtype=bool)
        if column in frame:
            source = frame[column].fillna(False).astype(bool).to_numpy()
            output[valid] = source[valid_positions]
        return output


class FastExperimentalPreparedLeaderBacktest:
    """Cached signal planner with canonical runner fills and accounting."""

    def __init__(
        self,
        delegate: PreparedLeaderBacktest,
        policy: ExperimentalLeaderPolicy,
        signal_cache: ExperimentalSignalCache,
    ) -> None:
        self.delegate = delegate
        self.policy = policy
        self.signal_cache = signal_cache
        self.blocked_symbols: set[str] = set()

    @property
    def strategy(self):
        """Expose the canonical strategy for playground portfolio overlays."""
        return self.delegate.strategy

    @property
    def prepared(self):
        """Expose prepared indicator frames for playground ATR sizing."""
        return self.delegate.prepared

    def report_frames(self, symbols: set[str]) -> dict[str, pd.DataFrame]:
        return self.delegate.report_frames(symbols)

    def build_order_plan(
        self,
        signal_ts,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        position = self.signal_cache.position_at(signal_ts)
        released = {
            symbol
            for symbol in self.blocked_symbols
            if self.signal_cache.fresh_buy_at(symbol, position)
        }
        self.blocked_symbols.difference_update(released)

        stopped = self._stopped_symbols(account)
        config = self.delegate.strategy.config
        if stopped:
            self.blocked_symbols.update(stopped)
            return OrderPlan(
                strategy_name=config.strategy.name,
                mode=mode,
                orders=tuple(
                    sell_all(
                        account.positions[symbol],
                        f"Fixed stop {float(self.policy.stop_loss_pct):.1%}",
                    )
                    for symbol in sorted(stopped)
                ),
                notes=("Experimental fixed stop; replacement deferred one signal bar.",),
            )

        held_positions = {
            symbol: held
            for symbol, held in account.positions.items()
            if held.quantity > 0
        }
        candidates = self.signal_cache.candidates_at(
            position,
            self.policy,
            self.blocked_symbols,
            set(held_positions),
        )
        orders: list[OrderIntent] = []
        max_positions = max(1, int(config.risk.max_position_count))
        target_candidates = candidates[:max_positions]
        target_symbols = {
            str(candidate["symbol"]) for candidate in target_candidates
        }
        sell_symbols: set[str] = set()

        for symbol, held in held_positions.items():
            if not self.signal_cache.has_data_at(symbol, position):
                orders.append(sell_all(held, "Held symbol missing from strategy data"))
                sell_symbols.add(symbol)
                continue

            sell_reason = None
            economics = account.position_economics.get(symbol)
            net_return_pct = (
                _finite_float(economics.net_return_pct)
                if economics is not None
                else None
            )
            if self.signal_cache.exit_down_at(symbol, position):
                sell_reason = "Supertrend down"
            elif symbol not in target_symbols:
                replacement = _first_replacement_candidate(
                    target_candidates,
                    held_symbols=set(held_positions),
                    sell_symbols=sell_symbols,
                )
                if replacement is not None:
                    current_score = self.signal_cache.value_at(
                        symbol, position, "score"
                    )
                    hurdle = (
                        float(replacement["atr_pct"])
                        * config.leader_rotation.hurdle_atr_mult
                    )
                    gate_passes = (
                        self.policy.rotation_profit_gate == "off"
                        or (
                            net_return_pct is not None
                            and net_return_pct >= 0.0
                        )
                    )
                    if (
                        current_score is not None
                        and float(replacement["score"]) - current_score > hurdle
                        and gate_passes
                    ):
                        sell_reason = "Leader rotation"
            if sell_reason:
                orders.append(sell_all(held, sell_reason))
                sell_symbols.add(symbol)

        kept_symbols = set(held_positions) - sell_symbols
        open_slots = max(0, max_positions - len(kept_symbols))
        buy_candidates = [
            candidate
            for candidate in candidates
            if candidate["symbol"] not in kept_symbols
            and candidate["symbol"] not in sell_symbols
        ]

        if sell_symbols and open_slots > 0:
            selected = buy_candidates[:open_slots]
            if selected:
                cash_allocation_pct = (
                    config.execution.allocation_pct / len(selected)
                )
                required_sells = tuple(sorted(sell_symbols))
                for candidate in selected:
                    orders.append(
                        OrderIntent(
                            symbol=str(candidate["symbol"]),
                            side="buy",
                            quantity=None,
                            order_type=config.execution.order_type,
                            reason="Post-sell leader entry",
                            cash_allocation_pct=cash_allocation_pct,
                            required_sell_symbols=required_sells,
                        )
                    )
            return OrderPlan(
                strategy_name=config.strategy.name,
                mode=mode,
                orders=tuple(orders),
            )

        estimated_cash = float(account.cash)
        remaining_buy_budget = (
            estimated_cash * config.execution.allocation_pct
        )
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
            estimated_cost = _estimated_buy_cost(
                qty,
                float(candidate["price"]),
                config,
            )
            estimated_cash = max(0.0, estimated_cash - estimated_cost)
            remaining_buy_budget = max(
                0.0, remaining_buy_budget - estimated_cost
            )
            open_slots -= 1

        return OrderPlan(
            strategy_name=config.strategy.name,
            mode=mode,
            orders=tuple(orders),
        )

    def _stopped_symbols(self, account: AccountSnapshot) -> set[str]:
        threshold = self.policy.stop_loss_pct
        if threshold is None:
            return set()
        return {
            symbol
            for symbol, held in account.positions.items()
            if held.quantity > 0
            and (economics := account.position_economics.get(symbol)) is not None
            and (net_return := _finite_float(economics.net_return_pct)) is not None
            and net_return <= -float(threshold)
        }


def _fresh_buy_signal_at(frame: pd.DataFrame | None, signal_ts) -> bool:
    if frame is None or frame.empty or "BuySignal" not in frame:
        return False
    available = frame.loc[:signal_ts]
    if available.empty:
        return False
    return bool(available.iloc[-1].get("BuySignal", False))


def _trend_is_up_at(trend: pd.Series, signal_ts) -> bool:
    if trend is None or trend.empty:
        return False
    available = trend.loc[:signal_ts]
    return bool(not available.empty and int(available.iloc[-1]) == 1)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _estimated_buy_cost(
    quantity: float,
    price: float,
    config,
) -> float:
    fill = price * (1.0 + config.costs.slippage_rate)
    return quantity * max(0.0, fill) * (1.0 + config.costs.fee_rate)
