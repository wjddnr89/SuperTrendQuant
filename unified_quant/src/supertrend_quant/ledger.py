from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

import pandas as pd

from .portfolio import AccountSnapshot, Position


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    action_type: str
    symbol: str
    message: str
    cash_delta: float = 0.0
    accrual_delta: float = 0.0


@dataclass(frozen=True)
class CashReceivable:
    event_id: str
    action_type: str
    symbol: str
    amount: float
    payment_date: str

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "action_type": self.action_type,
            "symbol": self.symbol,
            "amount": self.amount,
            "payment_date": self.payment_date,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CashReceivable":
        return cls(
            event_id=str(value["event_id"]),
            action_type=str(value.get("action_type") or "cash_dividend"),
            symbol=str(value.get("symbol") or ""),
            amount=float(value["amount"]),
            payment_date=str(value["payment_date"]),
        )


@dataclass
class PortfolioLedger:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    processed_event_ids: set[str] = field(default_factory=set)
    entitled_event_ids: set[str] = field(default_factory=set)
    cash_receivables: dict[str, CashReceivable] = field(default_factory=dict)
    unresolved_event_ids: set[str] = field(default_factory=set)
    dividend_tax_rate: float = 0.0

    @classmethod
    def from_account(
        cls,
        account: AccountSnapshot,
        *,
        processed_event_ids: Iterable[str] = (),
        entitled_event_ids: Iterable[str] = (),
        cash_receivables: Mapping[str, Mapping[str, Any] | CashReceivable] | None = None,
        unresolved_event_ids: Iterable[str] = (),
        dividend_tax_rate: float = 0.0,
    ) -> "PortfolioLedger":
        return cls(
            cash=float(account.cash),
            positions=dict(account.positions),
            processed_event_ids=set(processed_event_ids),
            entitled_event_ids=set(entitled_event_ids),
            cash_receivables={
                str(event_id): (
                    value
                    if isinstance(value, CashReceivable)
                    else CashReceivable.from_dict(value)
                )
                for event_id, value in (cash_receivables or {}).items()
            },
            unresolved_event_ids=set(unresolved_event_ids),
            dividend_tax_rate=dividend_tax_rate,
        )

    def snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(cash=self.cash, positions=dict(self.positions))

    @property
    def receivable_value(self) -> float:
        return sum(item.amount for item in self.cash_receivables.values())

    def buy(self, symbol: str, quantity: float, price: float, total_cost: float) -> None:
        if quantity <= 0 or total_cost < 0 or total_cost > self.cash:
            raise ValueError("Invalid ledger buy.")
        existing = self.positions.get(symbol)
        if existing is None:
            self.positions[symbol] = Position(symbol, quantity, price)
        else:
            total_quantity = existing.quantity + quantity
            average = (existing.quantity * existing.avg_price + quantity * price) / total_quantity
            self.positions[symbol] = Position(symbol, total_quantity, average)
        self.cash -= total_cost

    def sell(self, symbol: str, quantity: float, net_proceeds: float) -> Position:
        existing = self.positions.get(symbol)
        if existing is None or quantity <= 0 or quantity > existing.quantity:
            raise ValueError("Invalid ledger sell.")
        remaining = existing.quantity - quantity
        if remaining > 1e-12:
            self.positions[symbol] = Position(symbol, remaining, existing.avg_price)
        else:
            self.positions.pop(symbol, None)
        self.cash += net_proceeds
        return existing

    def apply_actions(
        self,
        actions: Iterable[Mapping[str, Any]],
        *,
        through: Any,
    ) -> tuple[LedgerEvent, ...]:
        cutoff = pd.Timestamp(through).normalize()
        events = self._settle_receivables(cutoff)
        eligible = []
        for action in actions:
            event_id = str(action.get("event_id") or "")
            if (
                not event_id
                or event_id in self.processed_event_ids
                or event_id in self.entitled_event_ids
            ):
                continue
            raw_effective = (
                action.get("ex_date")
                if str(action.get("ex_date") or "").strip()
                else action.get("effective_date")
            )
            effective = pd.to_datetime(
                raw_effective,
                errors="coerce",
            )
            if pd.isna(effective) or effective.normalize() > cutoff:
                continue
            eligible.append((effective.normalize(), event_id, action))
        for _, event_id, action in sorted(eligible, key=lambda item: (item[0], item[1])):
            events.append(self._apply_one(action, cutoff))
            if (
                event_id not in self.cash_receivables
                and event_id not in self.unresolved_event_ids
            ):
                self.processed_event_ids.add(event_id)
        return tuple(events)

    def _settle_receivables(self, cutoff: pd.Timestamp) -> list[LedgerEvent]:
        events: list[LedgerEvent] = []
        due = sorted(
            (
                receivable
                for receivable in self.cash_receivables.values()
                if pd.Timestamp(receivable.payment_date).normalize() <= cutoff
            ),
            key=lambda item: (item.payment_date, item.event_id),
        )
        for receivable in due:
            self.cash += receivable.amount
            self.cash_receivables.pop(receivable.event_id, None)
            self.entitled_event_ids.discard(receivable.event_id)
            self.processed_event_ids.add(receivable.event_id)
            events.append(
                LedgerEvent(
                    receivable.event_id,
                    receivable.action_type,
                    receivable.symbol,
                    "Cash distribution receivable paid.",
                    receivable.amount,
                    0.0,
                )
            )
        return events

    def _apply_one(self, action: Mapping[str, Any], cutoff: pd.Timestamp) -> LedgerEvent:
        event_id = str(action["event_id"])
        self.unresolved_event_ids.discard(event_id)
        action_type = str(action.get("action_type") or "").lower()
        symbol = str(action.get("symbol") or action.get("old_symbol") or "")
        position = self.positions.get(symbol)
        if position is None:
            return LedgerEvent(event_id, action_type, symbol, "No held position; marked processed.")
        cash_amount = _optional_float(action.get("cash_amount"))
        ratio = _optional_float(action.get("ratio"))

        if action_type in {"cash_dividend", "special_dividend"}:
            if cash_amount is None:
                self.unresolved_event_ids.add(event_id)
                return LedgerEvent(event_id, action_type, symbol, "Missing cash amount; event left unapplied.")
            cash_delta = position.quantity * cash_amount * (1.0 - self.dividend_tax_rate)
            raw_payment_date = action.get("payment_date") or action.get("pay_date")
            if raw_payment_date:
                payment_date = pd.Timestamp(raw_payment_date).normalize()
                if payment_date > cutoff:
                    self.cash_receivables[event_id] = CashReceivable(
                        event_id,
                        action_type,
                        symbol,
                        cash_delta,
                        payment_date.date().isoformat(),
                    )
                    self.entitled_event_ids.add(event_id)
                    return LedgerEvent(
                        event_id,
                        action_type,
                        symbol,
                        f"Cash distribution receivable recorded for {payment_date.date()}.",
                        0.0,
                        cash_delta,
                    )
            self.cash += cash_delta
            return LedgerEvent(
                event_id,
                action_type,
                symbol,
                "Cash distribution posted.",
                cash_delta,
                cash_delta,
            )

        if action_type in {"split", "stock_dividend", "capital_reduction"}:
            if ratio is None or ratio <= 0:
                self.unresolved_event_ids.add(event_id)
                return LedgerEvent(event_id, action_type, symbol, "Missing or invalid ratio; event left unapplied.")
            new_quantity = position.quantity * ratio
            cash_in_lieu, new_quantity, unresolved = _cash_in_lieu(action, new_quantity)
            if unresolved:
                self.unresolved_event_ids.add(event_id)
                return LedgerEvent(event_id, action_type, symbol, unresolved)
            if new_quantity > 1e-12:
                self.positions[symbol] = Position(symbol, new_quantity, position.avg_price / ratio)
            else:
                self.positions.pop(symbol, None)
            cash_delta = 0.0
            if action_type == "capital_reduction" and cash_amount:
                cash_delta = position.quantity * cash_amount
            cash_delta += cash_in_lieu
            self.cash += cash_delta
            return LedgerEvent(event_id, action_type, symbol, "Share quantity and cost basis adjusted.", cash_delta)

        if action_type == "spinoff":
            new_symbol = str(action.get("new_symbol") or "")
            if not new_symbol or ratio is None or ratio <= 0:
                self.unresolved_event_ids.add(event_id)
                return LedgerEvent(event_id, action_type, symbol, "Missing spin-off symbol or ratio; event left unapplied.")
            cost_fraction = float(_metadata(action).get("cost_basis_fraction", 0.0))
            cost_fraction = min(max(cost_fraction, 0.0), 1.0)
            new_quantity = position.quantity * ratio
            cash_in_lieu, new_quantity, unresolved = _cash_in_lieu(action, new_quantity)
            if unresolved:
                self.unresolved_event_ids.add(event_id)
                return LedgerEvent(event_id, action_type, symbol, unresolved)
            new_average = (
                (position.quantity * position.avg_price * cost_fraction) / new_quantity
                if new_quantity > 1e-12
                else 0.0
            )
            self.positions[symbol] = Position(
                symbol,
                position.quantity,
                position.avg_price * (1.0 - cost_fraction),
            )
            if new_quantity > 1e-12:
                self._merge_position(new_symbol, new_quantity, new_average)
            self.cash += cash_in_lieu
            return LedgerEvent(event_id, action_type, symbol, f"Spin-off position created: {new_symbol}.", cash_in_lieu)

        if action_type == "cash_merger":
            if cash_amount is None:
                self.unresolved_event_ids.add(event_id)
                return LedgerEvent(event_id, action_type, symbol, "Missing merger cash amount; event left unapplied.")
            cash_delta = position.quantity * cash_amount
            self.cash += cash_delta
            self.positions.pop(symbol, None)
            return LedgerEvent(event_id, action_type, symbol, "Position converted to cash.", cash_delta)

        if action_type == "stock_merger":
            new_symbol = str(action.get("new_symbol") or "")
            if not new_symbol or ratio is None or ratio <= 0:
                self.unresolved_event_ids.add(event_id)
                return LedgerEvent(event_id, action_type, symbol, "Missing merger symbol or ratio; event left unapplied.")
            new_quantity = position.quantity * ratio
            cash_in_lieu, new_quantity, unresolved = _cash_in_lieu(action, new_quantity)
            if unresolved:
                self.unresolved_event_ids.add(event_id)
                return LedgerEvent(event_id, action_type, symbol, unresolved)
            new_average = (
                position.quantity * position.avg_price / new_quantity
                if new_quantity > 1e-12
                else 0.0
            )
            self.positions.pop(symbol, None)
            if new_quantity > 1e-12:
                self._merge_position(new_symbol, new_quantity, new_average)
            self.cash += cash_in_lieu
            return LedgerEvent(event_id, action_type, symbol, f"Position converted to {new_symbol}.", cash_in_lieu)

        if action_type == "ticker_change":
            new_symbol = str(action.get("new_symbol") or "")
            if not new_symbol:
                self.unresolved_event_ids.add(event_id)
                return LedgerEvent(event_id, action_type, symbol, "Missing new ticker; event left unapplied.")
            self.positions.pop(symbol, None)
            self._merge_position(new_symbol, position.quantity, position.avg_price)
            return LedgerEvent(event_id, action_type, symbol, f"Ticker changed to {new_symbol}.")

        if action_type == "delisting":
            cash_delta = position.quantity * (cash_amount or 0.0)
            self.cash += cash_delta
            self.positions.pop(symbol, None)
            return LedgerEvent(event_id, action_type, symbol, "Delisted position removed.", cash_delta)

        self.unresolved_event_ids.add(event_id)
        return LedgerEvent(event_id, action_type, symbol, "Unsupported action; event left unapplied.")

    def _merge_position(self, symbol: str, quantity: float, avg_price: float) -> None:
        existing = self.positions.get(symbol)
        if existing is None:
            self.positions[symbol] = Position(symbol, quantity, avg_price)
            return
        total_quantity = existing.quantity + quantity
        average = (existing.quantity * existing.avg_price + quantity * avg_price) / total_quantity
        self.positions[symbol] = Position(symbol, total_quantity, average)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "" or pd.isna(value):
        return None
    return float(value)


def _metadata(action: Mapping[str, Any]) -> dict[str, Any]:
    value = action.get("metadata", {})
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        import json

        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _cash_in_lieu(
    action: Mapping[str, Any],
    quantity: float,
) -> tuple[float, float, str]:
    metadata = _metadata(action)
    if bool(metadata.get("allow_fractional", True)):
        return 0.0, quantity, ""
    whole_quantity = float(math.floor(quantity + 1e-12))
    fractional = quantity - whole_quantity
    if fractional <= 1e-12:
        return 0.0, whole_quantity, ""
    price = _optional_float(metadata.get("cash_in_lieu_price"))
    if price is None or price < 0:
        return 0.0, quantity, "Fractional settlement terms are missing; event left unapplied."
    return fractional * price, whole_quantity, ""
