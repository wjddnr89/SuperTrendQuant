from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    avg_price: float


@dataclass(frozen=True)
class PositionEconomics:
    """Execution-price economics for one open position.

    ``entry_cost`` is the actual all-in cash paid.  ``distributions`` contains
    net economic distributions accrued while the position was held.  Marked
    fields are populated only when a raw executable quote is available.
    """

    entry_cost: float
    distributions: float = 0.0
    raw_mark: float | None = None
    estimated_exit_proceeds: float | None = None
    net_return_pct: float | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    total_asset_value: float | None = None
    position_economics: dict[str, PositionEconomics] = field(default_factory=dict)

    @property
    def total_position_count(self) -> int:
        return len([position for position in self.positions.values() if position.quantity > 0])


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    quantity: float | None
    order_type: str = "market"
    price: float | None = None
    reason: str = ""
    client_order_id: str | None = None
    cash_allocation_pct: float | None = None
    required_sell_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrderPlan:
    strategy_name: str
    mode: str
    orders: tuple[OrderIntent, ...]
    notes: tuple[str, ...] = ()

    @property
    def has_orders(self) -> bool:
        return bool(self.orders)


def estimate_quantity(
    cash: float,
    price: float,
    allocation_pct: float,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> int:
    """Estimate an integer buy quantity whose all-in cost fits the allocation.

    The default cost arguments preserve the original public API and result.
    Positive slippage is applied to the quote before the fee, matching the
    paper and backtest fill models.
    """
    if not math.isfinite(fee_rate) or not math.isfinite(slippage_rate):
        raise ValueError("fee_rate and slippage_rate must be finite.")
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("fee_rate and slippage_rate must be non-negative.")
    if not math.isfinite(cash) or not math.isfinite(price) or not math.isfinite(allocation_pct):
        return 0
    if cash <= 0 or price <= 0 or allocation_pct <= 0:
        return 0
    unit_cost = price * (1.0 + slippage_rate) * (1.0 + fee_rate)
    return int((cash * allocation_pct) // unit_cost)


def mark_position_economics(
    account: AccountSnapshot,
    raw_prices: dict[str, float],
    *,
    fee_rate: float,
    slippage_rate: float,
) -> AccountSnapshot:
    """Attach projected raw-price liquidation returns to known entry state."""

    marked: dict[str, PositionEconomics] = {}
    for symbol, position in account.positions.items():
        economics = account.position_economics.get(symbol)
        raw_mark = raw_prices.get(symbol)
        if economics is None or economics.entry_cost <= 0 or raw_mark is None:
            continue
        try:
            raw_mark = float(raw_mark)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(raw_mark) or raw_mark <= 0:
            continue
        exit_fill = raw_mark * (1.0 - slippage_rate)
        exit_proceeds = position.quantity * exit_fill * (1.0 - fee_rate)
        economic_recovery = exit_proceeds + economics.distributions
        marked[symbol] = PositionEconomics(
            entry_cost=economics.entry_cost,
            distributions=economics.distributions,
            raw_mark=raw_mark,
            estimated_exit_proceeds=exit_proceeds,
            net_return_pct=economic_recovery / economics.entry_cost - 1.0,
        )
    return AccountSnapshot(
        cash=account.cash,
        positions=dict(account.positions),
        total_asset_value=account.total_asset_value,
        position_economics=marked,
    )
