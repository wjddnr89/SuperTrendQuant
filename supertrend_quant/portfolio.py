from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    avg_price: float


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    total_asset_value: float | None = None

    @property
    def total_position_count(self) -> int:
        return len([position for position in self.positions.values() if position.quantity > 0])


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    price: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class OrderPlan:
    strategy_name: str
    mode: str
    orders: tuple[OrderIntent, ...]
    notes: tuple[str, ...] = ()

    @property
    def has_orders(self) -> bool:
        return bool(self.orders)


def estimate_quantity(cash: float, price: float, allocation_pct: float) -> int:
    if cash <= 0 or price <= 0:
        return 0
    return int((cash * allocation_pct) // price)
