from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .env import load_env
from .ledger import PortfolioLedger
from .portfolio import (
    AccountSnapshot,
    OrderIntent,
    OrderPlan,
    Position,
    PositionEconomics,
    estimate_quantity,
)


@dataclass(frozen=True)
class BrokerOrderResult:
    accepted: bool
    broker_order_id: str = ""
    status: str = ""
    detail: str = ""


class PaperBroker:
    def __init__(self, state_path: str | Path, initial_cash: float):
        self.state_path = Path(state_path)
        self.initial_cash = initial_cash

    def get_account(self) -> AccountSnapshot:
        state = self._load_state()
        positions = {
            symbol: Position(symbol=symbol, quantity=float(raw["quantity"]), avg_price=float(raw["avg_price"]))
            for symbol, raw in state.get("positions", {}).items()
            if float(raw.get("quantity", 0)) > 0
        }
        raw_economics = state.get("position_economics", {})
        position_economics = {
            symbol: PositionEconomics(
                entry_cost=float(raw["entry_cost"]),
                distributions=float(raw.get("distributions", 0.0)),
            )
            for symbol, raw in raw_economics.items()
            if symbol in positions
            and isinstance(raw, dict)
            and float(raw.get("entry_cost", 0.0)) > 0
        }
        return AccountSnapshot(
            cash=float(state.get("cash", self.initial_cash)),
            positions=positions,
            position_economics=position_economics,
        )

    def get_metadata(self, key: str, default=None):
        return self._load_state().get("metadata", {}).get(key, default)

    def set_metadata(self, key: str, value) -> None:
        state = self._load_state()
        state.setdefault("metadata", {})[key] = value
        self._save_state(state)

    def apply_corporate_actions(
        self,
        actions,
        *,
        through,
        dividend_tax_rate: float = 0.0,
    ) -> tuple[str, ...]:
        state = self._load_state()
        account = self.get_account()
        metadata = state.setdefault("metadata", {})
        ledger = PortfolioLedger.from_account(
            account,
            processed_event_ids=metadata.get("processed_corporate_action_ids", ()),
            entitled_event_ids=metadata.get("entitled_corporate_action_ids", ()),
            cash_receivables=metadata.get("corporate_action_receivables", {}),
            unresolved_event_ids=metadata.get("unresolved_corporate_action_ids", ()),
            dividend_tax_rate=dividend_tax_rate,
        )
        actions = tuple(actions)
        actions_by_event_id = {
            str(action.get("event_id") or ""): action
            for action in actions
            if str(action.get("event_id") or "")
        }
        events = ledger.apply_actions(actions, through=through)
        if not events:
            return ()
        state["cash"] = ledger.cash
        state["positions"] = {
            symbol: {"quantity": position.quantity, "avg_price": position.avg_price}
            for symbol, position in ledger.positions.items()
        }
        metadata["processed_corporate_action_ids"] = sorted(ledger.processed_event_ids)
        metadata["entitled_corporate_action_ids"] = sorted(ledger.entitled_event_ids)
        metadata["unresolved_corporate_action_ids"] = sorted(ledger.unresolved_event_ids)
        metadata["corporate_action_receivables"] = {
            event_id: receivable.to_dict()
            for event_id, receivable in ledger.cash_receivables.items()
        }
        economics = {
            symbol: {
                "entry_cost": item.entry_cost,
                "distributions": item.distributions,
            }
            for symbol, item in account.position_economics.items()
        }
        for event in events:
            action = actions_by_event_id.get(event.event_id, {})
            source = economics.get(event.symbol)
            if source is not None:
                economic_delta = (
                    event.accrual_delta
                    if event.action_type in {"cash_dividend", "special_dividend"}
                    else event.accrual_delta or event.cash_delta
                )
                source["distributions"] += economic_delta
            if event.event_id not in ledger.processed_event_ids:
                continue
            new_symbol = str(action.get("new_symbol") or "").strip()
            if not new_symbol or new_symbol == event.symbol or source is None:
                continue
            if event.action_type == "spinoff":
                metadata_value = action.get("metadata", {})
                if isinstance(metadata_value, str):
                    try:
                        metadata_value = json.loads(metadata_value)
                    except json.JSONDecodeError:
                        metadata_value = {}
                try:
                    fraction = float(metadata_value.get("cost_basis_fraction", 0.0))
                except (AttributeError, TypeError, ValueError):
                    fraction = 0.0
                fraction = min(max(fraction, 0.0), 1.0)
                child_cost = source["entry_cost"] * fraction
                source["entry_cost"] -= child_cost
                child = economics.setdefault(
                    new_symbol,
                    {"entry_cost": 0.0, "distributions": 0.0},
                )
                child["entry_cost"] += child_cost
            elif event.action_type in {"ticker_change", "stock_merger"}:
                moved = economics.pop(event.symbol)
                target = economics.setdefault(
                    new_symbol,
                    {"entry_cost": 0.0, "distributions": 0.0},
                )
                target["entry_cost"] += moved["entry_cost"]
                target["distributions"] += moved["distributions"]
        economics = {
            symbol: value
            for symbol, value in economics.items()
            if symbol in ledger.positions and value["entry_cost"] > 0
        }
        state["position_economics"] = economics
        state.setdefault("corporate_action_events", []).extend(
            {
                "event_id": event.event_id,
                "action_type": event.action_type,
                "symbol": event.symbol,
                "message": event.message,
                "cash_delta": event.cash_delta,
                "accrual_delta": event.accrual_delta,
            }
            for event in events
        )
        self._save_state(state)
        return tuple(event.message for event in events)

    def execute_plan(
        self,
        plan: OrderPlan,
        prices: dict[str, float],
        fee_rate: float,
        slippage_rate: float,
        metadata_updates: dict[str, object] | None = None,
    ) -> list[str]:
        fee_rate = float(fee_rate)
        slippage_rate = float(slippage_rate)
        if not math.isfinite(fee_rate) or not math.isfinite(slippage_rate):
            raise ValueError("fee_rate and slippage_rate must be finite.")
        if fee_rate < 0 or slippage_rate < 0:
            raise ValueError("fee_rate and slippage_rate must be non-negative.")

        validated_orders: list[tuple[OrderIntent, str, float | None]] = []
        for order in plan.orders:
            side = str(order.side).strip().lower()
            if side not in {"buy", "sell"}:
                raise ValueError(f"Unsupported paper order side: {order.side}")
            quantity = float(order.quantity) if order.quantity is not None else None
            is_cash_allocated_buy = (
                side == "buy"
                and order.cash_allocation_pct is not None
                and 0.0 < order.cash_allocation_pct <= 1.0
            )
            if not is_cash_allocated_buy and (
                quantity is None or not math.isfinite(quantity) or quantity <= 0
            ):
                raise ValueError(f"Paper order quantity must be positive: {order.quantity}")
            validated_orders.append((order, side, quantity))

        state = self._load_state()
        cash = float(state.get("cash", self.initial_cash))
        positions = state.setdefault("positions", {})
        economics = state.setdefault("position_economics", {})
        fills: list[str] = []
        filled_sell_symbols: set[str] = set()
        cash_allocation_base: float | None = None

        for order, side, quantity in validated_orders:
            raw_price = prices.get(order.symbol)
            if raw_price is None:
                fills.append(f"SKIP {order.symbol}: no price")
                continue
            price = float(raw_price)
            if not math.isfinite(price) or price <= 0:
                fills.append(f"SKIP {order.symbol}: no price")
                continue

            if side == "buy":
                if order.required_sell_symbols and not set(
                    order.required_sell_symbols
                ).issubset(filled_sell_symbols):
                    fills.append(f"SKIP BUY {order.symbol}: prerequisite sell not filled")
                    continue
                if order.cash_allocation_pct is not None:
                    if cash_allocation_base is None:
                        cash_allocation_base = cash
                    target_quantity = estimate_quantity(
                        cash_allocation_base,
                        price,
                        order.cash_allocation_pct,
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                    )
                    affordable_quantity = estimate_quantity(
                        cash,
                        price,
                        1.0,
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                    )
                    quantity = min(target_quantity, affordable_quantity)
                if quantity is None or quantity <= 0:
                    fills.append(f"SKIP BUY {order.symbol}: insufficient cash")
                    continue
                fill_price = price * (1.0 + slippage_rate)
                cost = quantity * fill_price * (1.0 + fee_rate)
                if cost > cash:
                    fills.append(f"SKIP BUY {order.symbol}: insufficient cash")
                    continue
                cash -= cost
                existing = positions.get(order.symbol, {"quantity": 0.0, "avg_price": 0.0})
                old_qty = float(existing["quantity"])
                new_qty = old_qty + quantity
                avg_price = ((old_qty * float(existing["avg_price"])) + (quantity * fill_price)) / new_qty
                positions[order.symbol] = {"quantity": new_qty, "avg_price": avg_price}
                economic = economics.setdefault(
                    order.symbol,
                    {"entry_cost": 0.0, "distributions": 0.0},
                )
                economic["entry_cost"] = float(economic.get("entry_cost", 0.0)) + cost
                fills.append(f"BUY {order.symbol} {quantity:g} @ {fill_price:.4f}")
            else:
                existing = positions.get(order.symbol)
                if not existing:
                    fills.append(f"SKIP SELL {order.symbol}: no position")
                    continue
                sell_qty = min(quantity, float(existing["quantity"]))
                fill_price = price * (1.0 - slippage_rate)
                cash += sell_qty * fill_price * (1.0 - fee_rate)
                remaining = float(existing["quantity"]) - sell_qty
                if remaining > 0:
                    existing["quantity"] = remaining
                    economic = economics.get(order.symbol)
                    if economic is not None:
                        retained_fraction = remaining / (remaining + sell_qty)
                        economic["entry_cost"] *= retained_fraction
                        economic["distributions"] *= retained_fraction
                else:
                    positions.pop(order.symbol, None)
                    economics.pop(order.symbol, None)
                filled_sell_symbols.add(order.symbol)
                fills.append(f"SELL {order.symbol} {sell_qty:g} @ {fill_price:.4f}")

        state["cash"] = cash
        state.setdefault("fills", []).extend(fills)
        if metadata_updates:
            state.setdefault("metadata", {}).update(metadata_updates)
        self._save_state(state)
        return fills

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"cash": self.initial_cash, "positions": {}, "fills": []}
        with self.state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.state_path)


class TossBroker:
    base_url = "https://openapi.tossinvest.com"

    def __init__(self):
        load_env()
        self.client_id = os.getenv("TOSS_CLIENT_ID")
        self.client_secret = os.getenv("TOSS_CLIENT_SECRET")
        self.account_seq = os.getenv("TOSS_ACCOUNT_SEQ", "1")
        self.token: str | None = None
        self.token_expiry = 0.0

    def get_account(self, market: str) -> AccountSnapshot:
        token = self._token()
        headers = self._headers(token)
        currency = "KRW" if market == "KR" else "USD"
        cash_res = requests.get(
            f"{self.base_url}/api/v1/buying-power",
            headers=headers,
            params={"currency": currency},
            timeout=10,
        )
        cash_res.raise_for_status()
        cash = float(cash_res.json().get("result", {}).get("cashBuyingPower", 0))

        holdings_res = requests.get(f"{self.base_url}/api/v1/holdings", headers=headers, timeout=10)
        holdings_res.raise_for_status()
        positions: dict[str, Position] = {}
        position_economics: dict[str, PositionEconomics] = {}
        total_position_value = 0.0
        for item in holdings_res.json().get("result", {}).get("items", []):
            currency = item.get("currency", "KRW")
            market_country = str(item.get("marketCountry") or "").upper()
            if market == "KR" and (market_country and market_country != "KR"):
                continue
            if market == "US" and (market_country and market_country != "US"):
                continue
            if not market_country and market == "KR" and currency != "KRW":
                continue
            if not market_country and market == "US" and currency == "KRW":
                continue
            symbol = item.get("symbol")
            qty = float(item.get("quantity", 0) or 0)
            if symbol and qty > 0:
                avg_price = float(item.get("averagePurchasePrice", item.get("purchasePrice", 0)) or 0)
                positions[symbol] = Position(
                    symbol=symbol,
                    quantity=qty,
                    avg_price=avg_price,
                )
                market_value = item.get("marketValue", {})
                if isinstance(market_value, dict):
                    marked_value = float(
                        market_value.get("amount", qty * avg_price) or 0
                    )
                    after_cost = market_value.get("amountAfterCost")
                else:
                    marked_value = qty * float(
                        item.get("lastPrice", avg_price) or avg_price
                    )
                    after_cost = None
                total_position_value += marked_value
                profit_loss = item.get("profitLoss", {})
                raw_return = (
                    profit_loss.get("rateAfterCost", profit_loss.get("rate"))
                    if isinstance(profit_loss, dict)
                    else None
                )
                try:
                    net_return = (
                        float(raw_return)
                        if raw_return is not None
                        else marked_value / (qty * avg_price) - 1.0
                        if qty > 0 and avg_price > 0
                        else None
                    )
                except (TypeError, ValueError, ZeroDivisionError):
                    net_return = None
                try:
                    estimated_exit = (
                        float(after_cost) if after_cost is not None else marked_value
                    )
                except (TypeError, ValueError):
                    estimated_exit = marked_value
                position_economics[symbol] = PositionEconomics(
                    entry_cost=qty * avg_price,
                    raw_mark=float(item.get("lastPrice") or avg_price),
                    estimated_exit_proceeds=estimated_exit,
                    net_return_pct=net_return,
                )
        return AccountSnapshot(
            cash=cash,
            positions=positions,
            total_asset_value=cash + total_position_value,
            position_economics=position_economics,
        )

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        token = self._token()
        prices = {}
        unique = tuple(dict.fromkeys(str(symbol) for symbol in symbols if str(symbol)))
        for start in range(0, len(unique), 200):
            batch = unique[start : start + 200]
            res = requests.get(
                f"{self.base_url}/api/v1/prices",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                params={"symbols": ",".join(batch)},
                timeout=10,
            )
            res.raise_for_status()
            for item in res.json().get("result", []):
                symbol = item.get("symbol")
                last_price = item.get("lastPrice")
                if symbol and last_price:
                    prices[symbol] = float(last_price)
        return prices

    def list_open_orders(self) -> list[dict]:
        token = self._token()
        res = requests.get(
            f"{self.base_url}/api/v1/orders",
            headers=self._headers(token),
            params={"status": "OPEN"},
            timeout=10,
        )
        res.raise_for_status()
        result = res.json().get("result", {})
        return result.get("orders", result.get("items", []))

    def get_order(self, order_id: str) -> dict:
        token = self._token()
        res = requests.get(
            f"{self.base_url}/api/v1/orders/{order_id}",
            headers=self._headers(token),
            timeout=10,
        )
        res.raise_for_status()
        result = res.json().get("result", {})
        if not isinstance(result, dict):
            raise RuntimeError("Toss order detail response is not an object.")
        return result

    def cancel_order(self, order_id: str) -> bool:
        token = self._token()
        res = requests.post(
            f"{self.base_url}/api/v1/orders/{order_id}/cancel",
            headers=self._headers(token),
            json={},
            timeout=10,
        )
        return res.status_code in {200, 204}

    def place_order(self, order: OrderIntent) -> bool:
        return self.place_order_detailed(order).accepted

    def place_order_detailed(self, order: OrderIntent) -> BrokerOrderResult:
        if order.quantity is None or order.quantity <= 0:
            raise ValueError("Live orders require a resolved positive quantity.")
        token = self._token()
        payload = {
            "symbol": str(order.symbol),
            "side": "BUY" if order.side.lower() == "buy" else "SELL",
            "orderType": "MARKET" if order.order_type.lower() == "market" else "LIMIT",
            "quantity": str(int(order.quantity)),
        }
        if order.client_order_id:
            payload["clientOrderId"] = order.client_order_id
        if order.order_type.lower() == "limit" and order.price is not None:
            payload["price"] = str(order.price)
        res = requests.post(
            f"{self.base_url}/api/v1/orders",
            headers=self._headers(token),
            json=payload,
            timeout=10,
        )
        accepted = res.status_code in {200, 201}
        try:
            response_payload = res.json()
        except ValueError:
            response_payload = {}
        result = (
            response_payload.get("result", {})
            if isinstance(response_payload, dict)
            else {}
        )
        if not isinstance(result, dict):
            result = {}
        broker_order_id = str(
            result.get("orderId")
            or result.get("id")
            or response_payload.get("orderId", "")
            if isinstance(response_payload, dict)
            else ""
        )
        status = str(result.get("status") or ("accepted" if accepted else "rejected"))
        error = (
            response_payload.get("error", {})
            if isinstance(response_payload, dict)
            else {}
        )
        if not isinstance(error, dict):
            error = {}
        detail = "" if accepted else ": ".join(
            value
            for value in (
                f"HTTP {res.status_code}",
                str(error.get("code") or ""),
                str(error.get("message") or ""),
            )
            if value
        )
        return BrokerOrderResult(accepted, broker_order_id, status, detail)

    def _token(self) -> str:
        if not self.client_id or not self.client_secret:
            raise RuntimeError("TOSS_CLIENT_ID and TOSS_CLIENT_SECRET are required for live trading.")
        if self.token and time.time() < self.token_expiry - 60:
            return self.token
        res = requests.post(
            f"{self.base_url}/oauth2/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=10,
        )
        res.raise_for_status()
        data = res.json()
        self.token = data["access_token"]
        self.token_expiry = time.time() + int(data.get("expires_in", 3600))
        return self.token

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "X-Tossinvest-Account": str(self.account_seq),
            "Content-Type": "application/json",
        }
