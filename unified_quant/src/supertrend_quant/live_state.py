from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import AppConfig, RiskConfig
from .market_store.manifest import write_atomic
from .portfolio import AccountSnapshot, OrderIntent


LEDGER_BLOCKING_STATUSES = frozenset(
    {
        "submitting",
        "accepted",
        "open",
        "partially_filled",
        "filled",
        "inferred_filled",
        "unknown",
        "canceled",
        "rejected",
        "replaced",
        "cancel_rejected",
        "replace_rejected",
    }
)


def utc_now_text() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def strategy_config_hash(config: AppConfig) -> str:
    """Hash every input that can change signals, sizing, or eligibility."""

    payload = {
        "strategy": asdict(config.strategy),
        "scoring": asdict(config.scoring),
        "market": config.market,
        "universe": asdict(config.universe),
        "timeframe": config.timeframe,
        "data_price_mode": config.data_store.price_mode,
        "capital": asdict(config.capital),
        "costs": asdict(config.costs),
        "supertrend": asdict(config.supertrend),
        "market_trend_filter": asdict(config.market_trend_filter),
        "leader_rotation": asdict(config.leader_rotation),
        "exit": asdict(config.exit),
        "execution": asdict(config.execution),
        "risk": asdict(config.risk),
        "components": [asdict(value) for value in config.components],
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    return sha256(encoded).hexdigest()


def order_to_dict(order: OrderIntent) -> dict[str, Any]:
    return {
        "symbol": order.symbol,
        "side": order.side,
        "quantity": order.quantity,
        "order_type": order.order_type,
        "price": order.price,
        "reason": order.reason,
        "client_order_id": order.client_order_id,
        "cash_allocation_pct": order.cash_allocation_pct,
        "required_sell_symbols": list(order.required_sell_symbols),
    }


def order_from_dict(raw: dict[str, Any]) -> OrderIntent:
    return OrderIntent(
        symbol=str(raw["symbol"]),
        side=str(raw["side"]),
        quantity=(
            None if raw.get("quantity") is None else float(raw["quantity"])
        ),
        order_type=str(raw.get("order_type") or "market"),
        price=None if raw.get("price") is None else float(raw["price"]),
        reason=str(raw.get("reason") or ""),
        client_order_id=str(raw.get("client_order_id") or "") or None,
        cash_allocation_pct=(
            None
            if raw.get("cash_allocation_pct") is None
            else float(raw["cash_allocation_pct"])
        ),
        required_sell_symbols=tuple(
            str(value) for value in raw.get("required_sell_symbols", ())
        ),
    )


class SignalPlanStore:
    """Current pointer plus immutable copies of every executable signal plan."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise RuntimeError("Live signal plan has an unsupported schema.")
        return value

    def ensure(self, plan: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return self._ensure_locked(plan)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _ensure_locked(self, plan: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        existing = self.load()
        same_execution = bool(
            existing
            and existing.get("market") == plan.get("market")
            and existing.get("execution_session") == plan.get("execution_session")
        )
        if same_execution:
            invariant_keys = (
                "signal_session",
                "data_version",
                "strategy_hash",
                "plan_hash",
            )
            mismatched = [
                key
                for key in invariant_keys
                if existing.get(key) != plan.get(key)
            ]
            if mismatched:
                raise RuntimeError(
                    "A different durable live plan already exists for this execution "
                    "session: " + ", ".join(mismatched)
                )
            return existing, False

        content = _json_bytes(plan)
        immutable = (
            self.path.parent
            / "plans"
            / (
                f"{plan['market']}-{plan['execution_session']}-"
                f"{plan['plan_hash']}.json"
            )
        )
        if immutable.exists() and immutable.read_bytes() != content:
            raise RuntimeError(f"Immutable signal plan collision: {immutable}")
        if not immutable.exists():
            write_atomic(immutable, content)
        write_atomic(self.path, content)
        return plan, True


class LiveOrderLedger:
    """Append-only order and reconciliation events keyed by client order ID."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, event: dict[str, Any]) -> None:
        import fcntl

        record = {
            "schema_version": 1,
            "recorded_at": utc_now_text(),
            **event,
        }
        encoded = _json_bytes(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def events(self) -> tuple[dict[str, Any], ...]:
        if not self.path.is_file():
            return ()
        output: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Order ledger is corrupt at line {line_number}."
                ) from exc
            if not isinstance(value, dict) or value.get("schema_version") != 1:
                raise RuntimeError(
                    f"Order ledger has an unsupported record at line {line_number}."
                )
            output.append(value)
        return tuple(output)

    def latest_by_client_id(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self.events():
            client_order_id = str(event.get("client_order_id") or "")
            if client_order_id:
                latest[client_order_id] = event
        return latest

    def already_submitted(self, client_order_id: str) -> bool:
        event = self.latest_by_client_id().get(client_order_id)
        return bool(
            event
            and str(event.get("status") or "").lower() in LEDGER_BLOCKING_STATUSES
        )


class DailyRiskStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def update_and_check(
        self,
        *,
        session: str,
        account: AccountSnapshot,
        risk: RiskConfig,
    ) -> tuple[str | None, dict[str, Any]]:
        equity = _account_equity(account)
        previous = self._load()
        if previous.get("session") == session:
            opening_equity = float(previous.get("opening_equity", equity))
            baseline_source = str(previous.get("baseline_source") or "same_session")
        else:
            prior_equity = previous.get("last_equity")
            opening_equity = (
                float(prior_equity) if prior_equity is not None else equity
            )
            baseline_source = (
                "prior_observed_equity"
                if prior_equity is not None
                else "first_observation"
            )
        loss = max(0.0, opening_equity - equity)
        loss_pct = loss / opening_equity if opening_equity > 0 else 0.0
        absolute_breach = risk.max_daily_loss > 0 and loss >= risk.max_daily_loss
        percent_breach = (
            risk.max_daily_loss_pct > 0 and loss_pct >= risk.max_daily_loss_pct
        )
        state = {
            "schema_version": 1,
            "session": session,
            "opening_equity": opening_equity,
            "last_equity": equity,
            "loss": loss,
            "loss_pct": loss_pct,
            "baseline_source": baseline_source,
            "updated_at": utc_now_text(),
        }
        write_atomic(self.path, _json_bytes(state))
        if absolute_breach or percent_breach:
            return (
                "Daily loss limit reached: "
                f"loss={loss:,.2f}, loss_pct={loss_pct:.2%}",
                state,
            )
        return None, state

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("Live risk state must be a JSON object.")
        return value


def build_signal_plan(
    *,
    config: AppConfig,
    market: str,
    signal_session: str,
    execution_session: str,
    expires_at: str,
    data_version: str,
    orders: tuple[OrderIntent, ...],
    account: AccountSnapshot,
) -> dict[str, Any]:
    strategy_hash = strategy_config_hash(config)
    order_values = [order_to_dict(order) for order in orders]
    plan_identity = {
        "market": market,
        "signal_session": signal_session,
        "execution_session": execution_session,
        "data_version": data_version,
        "strategy_hash": strategy_hash,
        "orders": order_values,
    }
    plan_hash = sha256(_json_bytes(plan_identity)).hexdigest()
    return {
        "schema_version": 1,
        "plan_hash": plan_hash,
        "created_at": utc_now_text(),
        "market": market,
        "signal_session": signal_session,
        "execution_session": execution_session,
        "expires_at": expires_at,
        "data_version": data_version,
        "strategy_hash": strategy_hash,
        "starting_positions": {
            symbol: float(position.quantity)
            for symbol, position in sorted(account.positions.items())
        },
        "orders": order_values,
    }


def _account_equity(account: AccountSnapshot) -> float:
    if account.total_asset_value is not None:
        return float(account.total_asset_value)
    return float(account.cash) + sum(
        float(position.quantity) * float(position.avg_price)
        for position in account.positions.values()
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


__all__ = [
    "DailyRiskStateStore",
    "LEDGER_BLOCKING_STATUSES",
    "LiveOrderLedger",
    "SignalPlanStore",
    "build_signal_plan",
    "order_from_dict",
    "order_to_dict",
    "strategy_config_hash",
]
