from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from supertrend_quant.portfolio import AccountSnapshot, OrderIntent, OrderPlan


RegimeState = Literal["bull", "neutral", "bear"]
RiskAction = Literal["hold", "full_exit", "rebalance_50"]


@dataclass(frozen=True)
class RegimePolicy:
    name: str
    mode: Literal["canonical", "st_immediate", "st_confirmed", "three_state", "three_state_confirmed"]
    risk_action: RiskAction
    bear_confirm_bars: int = 1
    bull_confirm_bars: int = 1


def default_policies() -> list[RegimePolicy]:
    return [
        RegimePolicy("M0_CURRENT_A", "canonical", "hold"),
        RegimePolicy("M1_ST_DOWN_FULL_EXIT", "st_immediate", "full_exit"),
        RegimePolicy(
            "M2_ST_DOWN3_EXIT_UP2_REENTRY",
            "st_confirmed",
            "full_exit",
            bear_confirm_bars=3,
            bull_confirm_bars=2,
        ),
        RegimePolicy("M3_ST_DOWN_TRIM50", "st_immediate", "rebalance_50"),
        RegimePolicy("M4_ST_EMA200_THREE_STATE", "three_state", "full_exit"),
        RegimePolicy(
            "M5_THREE_STATE_BEAR2_BULL3",
            "three_state_confirmed",
            "full_exit",
            bear_confirm_bars=2,
            bull_confirm_bars=3,
        ),
    ]


def build_regime_states(
    market_trend: pd.Series,
    benchmark: pd.DataFrame,
    policy: RegimePolicy,
) -> pd.Series:
    if market_trend.empty:
        return pd.Series(dtype="object")
    if "Close" not in benchmark:
        raise ValueError("Market regime requires benchmark Close.")

    trend = pd.to_numeric(market_trend, errors="coerce").reindex(market_trend.index)
    close = pd.to_numeric(benchmark["Close"], errors="coerce").reindex(
        trend.index,
        method="ffill",
    )
    ema200 = pd.to_numeric(benchmark["Close"], errors="coerce").ewm(
        span=200,
        adjust=False,
    ).mean().reindex(trend.index, method="ffill")
    st_bull = trend.eq(1)
    ema_bull = close.gt(ema200)

    if policy.mode in {"canonical", "st_immediate"}:
        return pd.Series(
            st_bull.map({True: "bull", False: "bear"}),
            index=trend.index,
            dtype="object",
            name=policy.name,
        )

    if policy.mode == "st_confirmed":
        bull = _condition_streak(st_bull).ge(policy.bull_confirm_bars)
        bear_condition = ~st_bull
        bear = _condition_streak(bear_condition).ge(policy.bear_confirm_bars)
    else:
        bull_condition = st_bull & ema_bull
        bear_condition = (~st_bull) & (~ema_bull)
        if policy.mode == "three_state":
            bull = bull_condition
            bear = bear_condition
        elif policy.mode == "three_state_confirmed":
            bull = _condition_streak(bull_condition).ge(policy.bull_confirm_bars)
            bear = _condition_streak(bear_condition).ge(policy.bear_confirm_bars)
        else:
            raise ValueError(f"Unsupported regime mode: {policy.mode}")

    states = pd.Series("neutral", index=trend.index, dtype="object", name=policy.name)
    states.loc[bull] = "bull"
    states.loc[bear] = "bear"
    return states


@dataclass(frozen=True)
class RegimeManagedPreparedBacktest:
    delegate: object
    policy: RegimePolicy
    states: pd.Series

    def build_order_plan(
        self,
        signal_ts,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        from supertrend_quant.strategies import leader_rotation as leader_module

        state = _value_at(self.states, signal_ts)
        previous_state = _previous_value(self.states, signal_ts)
        canonical_trend_gate = leader_module._trend_is_up_at

        def regime_entry_gate(_trend, _signal_ts):
            return state == "bull"

        leader_module._trend_is_up_at = regime_entry_gate
        try:
            canonical_plan = self.delegate.build_order_plan(signal_ts, account, mode=mode)
        finally:
            leader_module._trend_is_up_at = canonical_trend_gate

        if self.policy.risk_action == "full_exit" and state == "bear":
            return _full_exit_plan(account, self.policy.name, mode)

        if self.policy.risk_action == "rebalance_50" and state != previous_state:
            if state == "bear" and account.positions:
                return _rebalance_plan(account, self.policy.name, mode, target_allocation=0.5)
            if state == "bull" and account.positions:
                if any(order.side.lower() == "sell" for order in canonical_plan.orders):
                    return canonical_plan
                return _rebalance_plan(account, self.policy.name, mode, target_allocation=1.0)

        return canonical_plan

    def report_frames(self, symbols: set[str]) -> dict[str, pd.DataFrame]:
        return self.delegate.report_frames(symbols)


def _condition_streak(condition: pd.Series) -> pd.Series:
    condition = condition.fillna(False).astype(bool)
    blocks = condition.ne(condition.shift(fill_value=False)).cumsum()
    streak = condition.groupby(blocks, sort=False).cumcount() + 1
    return streak.where(condition, 0).astype(int)


def _value_at(series: pd.Series, timestamp) -> RegimeState:
    available = series.loc[:timestamp]
    return "neutral" if available.empty else str(available.iloc[-1])  # type: ignore[return-value]


def _previous_value(series: pd.Series, timestamp) -> RegimeState:
    position = int(series.index.searchsorted(pd.Timestamp(timestamp), side="right")) - 1
    if position <= 0:
        return "neutral"
    return str(series.iloc[position - 1])  # type: ignore[return-value]


def _full_exit_plan(
    account: AccountSnapshot,
    policy_name: str,
    mode: str,
) -> OrderPlan:
    orders = tuple(
        OrderIntent(
            symbol=symbol,
            side="sell",
            quantity=position.quantity,
            reason=f"Market regime full exit: {policy_name}",
        )
        for symbol, position in sorted(account.positions.items())
        if position.quantity > 0
    )
    return OrderPlan(policy_name, mode, orders)


def _rebalance_plan(
    account: AccountSnapshot,
    policy_name: str,
    mode: str,
    *,
    target_allocation: float,
) -> OrderPlan:
    orders: list[OrderIntent] = []
    held_symbols = tuple(
        symbol
        for symbol, position in sorted(account.positions.items())
        if position.quantity > 0
    )
    for symbol in held_symbols:
        position = account.positions[symbol]
        orders.append(
            OrderIntent(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                reason=f"Market regime rebalance exit: {policy_name}",
            )
        )
    if held_symbols:
        allocation_each = target_allocation / len(held_symbols)
        for symbol in held_symbols:
            orders.append(
                OrderIntent(
                    symbol=symbol,
                    side="buy",
                    quantity=None,
                    cash_allocation_pct=allocation_each,
                    required_sell_symbols=held_symbols,
                    reason=(
                        f"Market regime rebalance to {target_allocation:.0%}: {policy_name}"
                    ),
                )
            )
    return OrderPlan(policy_name, mode, tuple(orders))

