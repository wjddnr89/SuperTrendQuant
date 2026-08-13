from __future__ import annotations

import html
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DAILY_COLUMNS = (
    "execution_date",
    "signal_date",
    "account_id",
    "account_name",
    "hypothesis",
    "status",
    "equity",
    "cash",
    "positions_value",
    "daily_return",
    "cumulative_return",
    "drawdown",
    "position_symbol",
    "position_quantity",
    "position_avg_price",
    "mark_price",
    "order_count",
    "fill_count",
    "orders",
    "fills",
    "notes",
)


class PerformanceReporter:
    def __init__(self, root: str | Path, initial_cash: float) -> None:
        self.root = Path(root)
        self.initial_cash = float(initial_cash)
        self.daily_history_path = self.root / "daily_history.csv"
        self.weekly_history_path = self.root / "weekly_history.csv"

    def record_daily(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return self.load_daily()
        existing = self.load_daily()
        incoming = pd.DataFrame(rows)
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.drop_duplicates(
            ["execution_date", "account_id"], keep="last"
        )
        combined["execution_date"] = pd.to_datetime(
            combined["execution_date"]
        ).dt.date.astype(str)
        combined = combined.sort_values(["execution_date", "account_id"])
        combined = self._calculate_returns(combined)
        combined = combined.reindex(columns=DAILY_COLUMNS)
        _atomic_csv(self.daily_history_path, combined)

        execution_date = str(incoming.iloc[-1]["execution_date"])
        daily = combined.loc[combined["execution_date"] == execution_date]
        _atomic_csv(self.root / "daily" / f"{execution_date}.csv", daily)
        _write_json(
            self.root / "latest_daily.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "execution_date": execution_date,
                "accounts": daily.to_dict(orient="records"),
            },
        )
        self.generate_weekly(combined)
        self._write_dashboard(combined)
        return combined

    def load_daily(self) -> pd.DataFrame:
        if not self.daily_history_path.exists():
            return pd.DataFrame(columns=DAILY_COLUMNS)
        return pd.read_csv(self.daily_history_path)

    def generate_weekly(
        self,
        daily: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        daily = self.load_daily() if daily is None else daily.copy()
        if daily.empty:
            weekly = pd.DataFrame()
            _atomic_csv(self.weekly_history_path, weekly)
            return weekly

        daily["date"] = pd.to_datetime(daily["execution_date"])
        iso = daily["date"].dt.isocalendar()
        daily["iso_year"] = iso.year.astype(int)
        daily["iso_week"] = iso.week.astype(int)
        records: list[dict[str, Any]] = []
        for (year, week, account_id), group in daily.groupby(
            ["iso_year", "iso_week", "account_id"],
            sort=True,
        ):
            group = group.sort_values("date")
            last_equity = _number(group.iloc[-1].get("equity"))
            prior = daily.loc[
                (daily["account_id"] == account_id)
                & (daily["date"] < group.iloc[0]["date"])
            ].sort_values("date")
            base_equity = (
                _number(prior.iloc[-1].get("equity"))
                if not prior.empty
                else self.initial_cash
            )
            records.append(
                {
                    "iso_year": int(year),
                    "iso_week": int(week),
                    "week": f"{int(year)}-W{int(week):02d}",
                    "account_id": account_id,
                    "account_name": group.iloc[-1]["account_name"],
                    "hypothesis": group.iloc[-1]["hypothesis"],
                    "start_date": group.iloc[0]["execution_date"],
                    "end_date": group.iloc[-1]["execution_date"],
                    "trading_days": int(len(group)),
                    "start_equity": base_equity,
                    "end_equity": last_equity,
                    "weekly_return": (
                        last_equity / base_equity - 1.0
                        if base_equity
                        else 0.0
                    ),
                    "cumulative_return": (
                        last_equity / self.initial_cash - 1.0
                        if self.initial_cash
                        else 0.0
                    ),
                    "worst_drawdown": pd.to_numeric(
                        group["drawdown"], errors="coerce"
                    ).min(),
                    "orders": int(
                        pd.to_numeric(
                            group["order_count"], errors="coerce"
                        ).fillna(0).sum()
                    ),
                    "fills": int(
                        pd.to_numeric(
                            group["fill_count"], errors="coerce"
                        ).fillna(0).sum()
                    ),
                    "latest_position": group.iloc[-1]["position_symbol"],
                    "latest_status": group.iloc[-1]["status"],
                }
            )
        weekly = pd.DataFrame(records).sort_values(["iso_year", "iso_week", "account_id"])
        _atomic_csv(self.weekly_history_path, weekly)
        latest_week = str(weekly.iloc[-1]["week"])
        latest = weekly.loc[weekly["week"] == latest_week]
        _atomic_csv(self.root / "weekly" / f"{latest_week}.csv", latest)
        _write_json(
            self.root / "latest_weekly.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "week": latest_week,
                "accounts": latest.to_dict(orient="records"),
            },
        )
        return weekly

    def _calculate_returns(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["equity"] = pd.to_numeric(output["equity"], errors="coerce")
        for account_id, index in output.groupby("account_id").groups.items():
            account = output.loc[index].sort_values("execution_date")
            equities = account["equity"].astype(float)
            previous = equities.shift(1).fillna(self.initial_cash)
            output.loc[account.index, "daily_return"] = (
                equities / previous - 1.0
            ).to_numpy()
            output.loc[account.index, "cumulative_return"] = (
                equities / self.initial_cash - 1.0
            ).to_numpy()
            peaks = equities.cummax().clip(lower=self.initial_cash)
            output.loc[account.index, "drawdown"] = (
                equities / peaks - 1.0
            ).to_numpy()
        return output

    def _write_dashboard(self, daily: pd.DataFrame) -> None:
        if daily.empty:
            return
        latest_date = str(daily["execution_date"].max())
        latest = daily.loc[daily["execution_date"] == latest_date].copy()
        account_count = int(latest["account_id"].nunique())
        cards = []
        for row in latest.itertuples(index=False):
            cards.append(
                "<article>"
                f"<h2>{html.escape(str(row.account_id))} · "
                f"{html.escape(str(row.account_name))}</h2>"
                f"<p>{html.escape(str(row.hypothesis))}</p>"
                f"<strong>${float(row.equity):,.2f}</strong>"
                f"<dl><dt>일일</dt><dd>{float(row.daily_return):+.2%}</dd>"
                f"<dt>누적</dt><dd>{float(row.cumulative_return):+.2%}</dd>"
                f"<dt>낙폭</dt><dd>{float(row.drawdown):.2%}</dd>"
                f"<dt>보유</dt><dd>{html.escape(str(row.position_symbol or '-'))}</dd>"
                f"<dt>상태</dt><dd>{html.escape(str(row.status))}</dd></dl>"
                "</article>"
            )
        table = latest[
            [
                "account_id",
                "equity",
                "daily_return",
                "cumulative_return",
                "drawdown",
                "position_symbol",
                "orders",
                "fills",
            ]
        ].to_html(index=False, classes="summary", border=0)
        body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nasdaq {account_count}-account Paper</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#0d1117;color:#e6edf3}}
main{{max-width:1100px;margin:auto;padding:28px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
article{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px}}
strong{{font-size:30px}} dl{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
dt{{color:#8b949e}} dd{{margin:0;text-align:right}}
table{{width:100%;border-collapse:collapse;margin-top:24px;background:#161b22}}
th,td{{padding:10px;border-bottom:1px solid #30363d;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
</style></head><body><main>
<h1>Nasdaq-100 {account_count}계좌 가상실험</h1>
<p>기준일 {html.escape(latest_date)} · 전일 완성 일봉 신호 / 당일 첫 1분봉 시가 체결 / 당일 종가 평가</p>
<section class="cards">{''.join(cards)}</section>{table}
</main></body></html>"""
        path = self.root / "dashboard.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(body, encoding="utf-8")
        os.replace(temporary, path)


def daily_row(
    *,
    execution_date: date,
    signal_date: date,
    account_id: str,
    account_name: str,
    hypothesis: str,
    status: str,
    account,
    mark_prices: dict[str, float],
    plan,
    fills: list[str],
    notes: list[str],
) -> dict[str, Any]:
    positions_value = 0.0
    position_symbol = ""
    position_quantity = 0.0
    position_avg_price = 0.0
    mark_price = 0.0
    for symbol, position in sorted(account.positions.items()):
        price = float(mark_prices.get(symbol, position.avg_price))
        positions_value += position.quantity * price
        if not position_symbol:
            position_symbol = symbol
            position_quantity = position.quantity
            position_avg_price = position.avg_price
            mark_price = price
    equity = float(account.cash) + positions_value
    executed = [
        fill
        for fill in fills
        if fill.startswith("BUY ") or fill.startswith("SELL ")
    ]
    return {
        "execution_date": execution_date.isoformat(),
        "signal_date": signal_date.isoformat(),
        "account_id": account_id,
        "account_name": account_name,
        "hypothesis": hypothesis,
        "status": status,
        "equity": equity,
        "cash": float(account.cash),
        "positions_value": positions_value,
        "daily_return": 0.0,
        "cumulative_return": 0.0,
        "drawdown": 0.0,
        "position_symbol": position_symbol,
        "position_quantity": position_quantity,
        "position_avg_price": position_avg_price,
        "mark_price": mark_price,
        "order_count": len(plan.orders),
        "fill_count": len(executed),
        "orders": json.dumps(
            [
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "reason": order.reason,
                }
                for order in plan.orders
            ],
            ensure_ascii=False,
        ),
        "fills": json.dumps(fills, ensure_ascii=False),
        "notes": json.dumps(notes, ensure_ascii=False),
    }


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0
