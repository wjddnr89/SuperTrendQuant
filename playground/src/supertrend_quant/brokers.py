from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import requests

from .env import load_env
from .portfolio import AccountSnapshot, OrderIntent, OrderPlan, Position


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
        return AccountSnapshot(cash=float(state.get("cash", self.initial_cash)), positions=positions)

    def get_metadata(self, key: str, default=None):
        return self._load_state().get("metadata", {}).get(key, default)

    def set_metadata(self, key: str, value) -> None:
        state = self._load_state()
        state.setdefault("metadata", {})[key] = value
        self._save_state(state)

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

        validated_orders: list[tuple[OrderIntent, str, float]] = []
        for order in plan.orders:
            side = str(order.side).strip().lower()
            if side not in {"buy", "sell"}:
                raise ValueError(f"Unsupported paper order side: {order.side}")
            quantity = float(order.quantity)
            if not math.isfinite(quantity) or quantity <= 0:
                raise ValueError(f"Paper order quantity must be positive: {order.quantity}")
            validated_orders.append((order, side, quantity))

        state = self._load_state()
        cash = float(state.get("cash", self.initial_cash))
        positions = state.setdefault("positions", {})
        fills: list[str] = []

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
                else:
                    positions.pop(order.symbol, None)
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
        total_position_value = 0.0
        for item in holdings_res.json().get("result", {}).get("items", []):
            currency = item.get("currency", "KRW")
            if market == "KR" and currency != "KRW":
                continue
            if market == "US" and currency == "KRW":
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
                    total_position_value += float(market_value.get("amount", qty * avg_price) or 0)
                else:
                    total_position_value += qty * float(item.get("lastPrice", avg_price) or avg_price)
        return AccountSnapshot(cash=cash, positions=positions, total_asset_value=cash + total_position_value)

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        token = self._token()
        res = requests.get(
            f"{self.base_url}/api/v1/prices",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            params={"symbols": ",".join(symbols)},
            timeout=10,
        )
        res.raise_for_status()
        prices = {}
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
        return res.status_code in {200, 201}

    def _token(self) -> str:
        if not self.client_id or not self.client_secret:
            raise RuntimeError("TOSS_CLIENT_ID and TOSS_CLIENT_SECRET are required for live trading.")
        if self.token and time.time() < self.token_expiry - 60:
            return self.token
        res = requests.post(
            f"{self.base_url}/oauth2/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
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
