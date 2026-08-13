from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from supertrend_quant.portfolio import (
    AccountSnapshot,
    OrderIntent,
    OrderPlan,
    estimate_quantity,
)


@dataclass(frozen=True)
class PortfolioRiskPolicy:
    name: str
    max_positions: int
    allocation_pct: float = 1.0
    target_portfolio_atr_pct: float | None = None
    drawdown_stop_pct: float | None = None
    cooldown_sessions: int = 0


def default_policies() -> list[PortfolioRiskPolicy]:
    return [
        PortfolioRiskPolicy("D0_CURRENT_A", 1),
        PortfolioRiskPolicy("D1_ALLOCATION_75PCT", 1, allocation_pct=0.75),
        PortfolioRiskPolicy("D2_MAX_POSITIONS_2_EQUAL", 2),
        PortfolioRiskPolicy("D3_MAX_POSITIONS_3_EQUAL", 3),
        PortfolioRiskPolicy("D4A_MAX1_ATR_RISK_2P0", 1, target_portfolio_atr_pct=0.020),
        PortfolioRiskPolicy("D4_MAX1_ATR_RISK_2P5", 1, target_portfolio_atr_pct=0.025),
        PortfolioRiskPolicy("D4B_MAX1_ATR_RISK_3P0", 1, target_portfolio_atr_pct=0.030),
        PortfolioRiskPolicy("D5_MAX2_ATR_RISK_2P5", 2, target_portfolio_atr_pct=0.025),
        PortfolioRiskPolicy(
            "D6_MAX2_ATR_DD15_COOLDOWN20",
            2,
            target_portfolio_atr_pct=0.025,
            drawdown_stop_pct=0.15,
            cooldown_sessions=20,
        ),
    ]


def atr_target_weight(
    atr_pct: float,
    *,
    max_positions: int,
    target_portfolio_atr_pct: float,
) -> float:
    if not math.isfinite(atr_pct) or atr_pct <= 0:
        return 0.0
    slot_cap = 1.0 / max(1, int(max_positions))
    slot_risk_budget = target_portfolio_atr_pct / max(1, int(max_positions))
    return max(0.0, min(slot_cap, slot_risk_budget / atr_pct))


class PortfolioRiskPreparedBacktest:
    def __init__(self, delegate, policy: PortfolioRiskPolicy):
        self.delegate = delegate
        self.policy = policy
        self._peak_equity = 0.0
        self._signal_number = -1
        self._cooldown_until = -1
        self._drawdown_stop_count = 0

    @property
    def drawdown_stop_count(self) -> int:
        return self._drawdown_stop_count

    def build_order_plan(
        self,
        signal_ts,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        self._signal_number += 1
        equity = _account_liquidation_value(account)

        if self._signal_number <= self._cooldown_until:
            return _liquidate_or_wait(account, self.policy.name, mode, "Drawdown cooldown")

        if self._cooldown_until >= 0 and self._signal_number > self._cooldown_until:
            self._peak_equity = equity
            self._cooldown_until = -1

        self._peak_equity = max(self._peak_equity, equity)
        drawdown = equity / self._peak_equity - 1.0 if self._peak_equity > 0 else 0.0
        if (
            self.policy.drawdown_stop_pct is not None
            and drawdown <= -float(self.policy.drawdown_stop_pct)
        ):
            self._drawdown_stop_count += 1
            self._cooldown_until = self._signal_number + max(1, self.policy.cooldown_sessions)
            return _liquidate_or_wait(account, self.policy.name, mode, "Drawdown stop")

        plan = self.delegate.build_order_plan(signal_ts, account, mode=mode)
        if self.policy.target_portfolio_atr_pct is None:
            return plan
        return self._resize_buys(plan, signal_ts, account, equity)

    def report_frames(self, symbols: set[str]) -> dict[str, pd.DataFrame]:
        return self.delegate.report_frames(symbols)

    def _resize_buys(
        self,
        plan: OrderPlan,
        signal_ts,
        account: AccountSnapshot,
        equity: float,
    ) -> OrderPlan:
        resized: list[OrderIntent] = []
        config = self.delegate.strategy.config
        for order in plan.orders:
            if order.side.lower() != "buy":
                resized.append(order)
                continue
            row = _prepared_row_at(self.delegate.prepared.get(order.symbol), signal_ts)
            if row is None:
                resized.append(order)
                continue
            atr_pct = _finite_float(row.get("ATR_pct"))
            price = _finite_float(row.get("Close"))
            if atr_pct is None or price is None:
                resized.append(order)
                continue
            weight = atr_target_weight(
                atr_pct,
                max_positions=self.policy.max_positions,
                target_portfolio_atr_pct=float(self.policy.target_portfolio_atr_pct),
            )
            quantity = estimate_quantity(
                equity,
                price,
                weight,
                fee_rate=config.costs.fee_rate,
                slippage_rate=config.costs.slippage_rate,
            )
            if quantity <= 0:
                continue
            resized.append(
                OrderIntent(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=quantity,
                    order_type=order.order_type,
                    price=order.price,
                    reason=(
                        f"{order.reason}; ATR risk weight {weight:.2%}"
                        if order.reason
                        else f"ATR risk weight {weight:.2%}"
                    ),
                    client_order_id=order.client_order_id,
                    cash_allocation_pct=None,
                    required_sell_symbols=order.required_sell_symbols,
                )
            )
        return OrderPlan(plan.strategy_name, plan.mode, tuple(resized), plan.notes)


def _account_liquidation_value(account: AccountSnapshot) -> float:
    value = float(account.cash)
    for symbol, position in account.positions.items():
        economics = account.position_economics.get(symbol)
        if economics is not None and economics.estimated_exit_proceeds is not None:
            value += float(economics.estimated_exit_proceeds) + float(economics.distributions)
        elif economics is not None and economics.raw_mark is not None:
            value += float(position.quantity) * float(economics.raw_mark)
        else:
            value += float(position.quantity) * float(position.avg_price)
    return value


def _liquidate_or_wait(
    account: AccountSnapshot,
    policy_name: str,
    mode: str,
    reason: str,
) -> OrderPlan:
    orders = tuple(
        OrderIntent(
            symbol=symbol,
            side="sell",
            quantity=position.quantity,
            reason=f"{reason}: {policy_name}",
        )
        for symbol, position in sorted(account.positions.items())
        if position.quantity > 0
    )
    return OrderPlan(policy_name, mode, orders)


def _prepared_row_at(frame: pd.DataFrame | None, signal_ts) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    available = frame.loc[:signal_ts]
    return None if available.empty else available.iloc[-1]


def _finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
