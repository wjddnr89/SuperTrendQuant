from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .metrics import calculate_metrics
from .portfolio import AccountSnapshot, OrderIntent, OrderPlan, Position


def make_run_id(strategy_name: str, mode: str, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    safe_strategy = re.sub(r"[^A-Za-z0-9_.-]+", "_", strategy_name).strip("_") or "strategy"
    return f"{timestamp}_{mode}_{safe_strategy}"


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return asdict(config) if is_dataclass(config) else dict(config)


def account_to_dict(account: AccountSnapshot, prices: dict[str, float] | None = None) -> dict[str, Any]:
    prices = prices or {}
    positions = {
        symbol: position_to_dict(position, prices.get(symbol))
        for symbol, position in account.positions.items()
    }
    positions_value = sum(raw["market_value"] for raw in positions.values())
    total_equity = account.cash + positions_value
    return {
        "cash": account.cash,
        "positions": positions,
        "positions_value": positions_value,
        "equity": total_equity,
        "total_asset_value": account.total_asset_value,
    }


def position_to_dict(position: Position, price: float | None = None) -> dict[str, Any]:
    mark_price = price if price is not None else position.avg_price
    return {
        "symbol": position.symbol,
        "quantity": position.quantity,
        "avg_price": position.avg_price,
        "mark_price": mark_price,
        "market_value": position.quantity * mark_price,
    }


def order_to_dict(order: OrderIntent) -> dict[str, Any]:
    return {
        "symbol": order.symbol,
        "side": order.side,
        "quantity": order.quantity,
        "order_type": order.order_type,
        "price": order.price,
        "reason": order.reason,
    }


class PaperRunRecorder:
    def __init__(self, root_dir: str | Path, strategy_name: str, run_id: str | None = None):
        self.root_dir = Path(root_dir)
        self.run_id = run_id or make_run_id(strategy_name, "paper")
        self.run_dir = self.root_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(self, config: AppConfig) -> None:
        _write_json(
            self.run_dir / "metadata.json",
            {
                "run_id": self.run_id,
                "mode": "paper",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config_to_dict(config),
            },
        )

    def record_cycle(
        self,
        *,
        timestamp: datetime,
        market: str,
        candle_base: datetime | None,
        plan: OrderPlan,
        fills: list[str],
        account_before: AccountSnapshot,
        account_after: AccountSnapshot,
        prices: dict[str, float],
        notes: tuple[str, ...] = (),
    ) -> None:
        before = account_to_dict(account_before, prices)
        after = account_to_dict(account_after, prices)
        event = {
            "timestamp": timestamp.isoformat(),
            "market": market,
            "candle_base": candle_base.isoformat() if candle_base is not None else None,
            "strategy": plan.strategy_name,
            "orders": [order_to_dict(order) for order in plan.orders],
            "fills": fills,
            "notes": list(plan.notes + notes),
            "account_before": before,
            "account_after": after,
            "prices": prices,
        }
        _append_jsonl(self.run_dir / "cycles.jsonl", event)
        _append_csv(
            self.run_dir / "equity.csv",
            {
                "timestamp": event["timestamp"],
                "market": market,
                "candle_base": event["candle_base"],
                "equity": after["equity"],
                "cash": after["cash"],
                "positions_value": after["positions_value"],
                "position_count": len(account_after.positions),
                "order_count": len(plan.orders),
                "fill_count": len(fills),
            },
        )


def save_backtest_result(result, config: AppConfig, root_dir: str | Path, run_id: str | None = None) -> Path:
    run_id = run_id or make_run_id(config.strategy.name, "backtest")
    run_dir = Path(root_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "mode": "backtest",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "strategy": config.strategy.name,
            "market": config.market,
            "timeframe": config.timeframe,
            "period": config.period,
            "metrics": result.metrics,
            "trade_returns": result.trades,
            "skipped": list(result.skipped),
            "config": config_to_dict(config),
        },
    )
    result.equity.rename("equity").to_csv(run_dir / "equity.csv", header=True)
    return run_dir


def compare_paper_to_backtest(paper_dir: str | Path, backtest_dir: str | Path, interval: str) -> dict[str, Any]:
    paper_dir = Path(paper_dir)
    backtest_dir = Path(backtest_dir)
    paper_equity_path = paper_dir / "equity.csv"
    backtest_summary_path = backtest_dir / "summary.json"
    if not paper_equity_path.exists():
        raise FileNotFoundError(f"Paper equity file not found: {paper_equity_path}")
    if not backtest_summary_path.exists():
        raise FileNotFoundError(f"Backtest summary file not found: {backtest_summary_path}")

    paper_df = pd.read_csv(paper_equity_path)
    if paper_df.empty:
        raise ValueError(f"Paper equity file is empty: {paper_equity_path}")
    paper_equity = pd.Series(paper_df["equity"].astype(float).to_numpy(), index=pd.to_datetime(paper_df["timestamp"]))
    paper_metrics = calculate_metrics(paper_equity, [], interval)
    backtest_summary = _read_json(backtest_summary_path)
    backtest_metrics = backtest_summary.get("metrics", {})
    return {
        "paper_dir": str(paper_dir),
        "backtest_dir": str(backtest_dir),
        "paper_points": int(len(paper_df)),
        "paper_metrics": paper_metrics,
        "backtest_metrics": backtest_metrics,
        "diff": {
            key: paper_metrics.get(key, 0) - float(backtest_metrics.get(key, 0))
            for key in ("total_return", "mdd", "sharpe", "win_rate", "payoff_ratio")
            if key in paper_metrics
        },
    }


def latest_run_dir(root_dir: str | Path) -> Path:
    root = Path(root_dir)
    candidates = sorted([path for path in root.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directories under {root}")
    return candidates[-1]


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(item, handle, indent=2, ensure_ascii=False, default=str)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
