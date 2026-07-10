from __future__ import annotations

import json
from pathlib import Path

from .portfolio import AccountSnapshot


class HoldingsStore:
    def __init__(self, path: str | Path = "holding.json"):
        self.path = Path(path)

    def load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {"KR": {}, "US": {}}
        try:
            content = self.path.read_text(encoding="utf-8").strip()
            if not content:
                return {"KR": {}, "US": {}}
            data = json.loads(content)
            if not isinstance(data, dict):
                return {"KR": {}, "US": {}}
            return {
                "KR": data.get("KR") if isinstance(data.get("KR"), dict) else {},
                "US": data.get("US") if isinstance(data.get("US"), dict) else {},
            }
        except Exception as exc:
            print(f"Holdings read failed: {exc}")
            return {"KR": {}, "US": {}}

    def save(self, data: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def sync_market(self, market: str, account: AccountSnapshot, universe_symbols: list[str]) -> dict[str, dict]:
        data = self.load()
        current = data.get(market, {})
        synced: dict[str, dict] = {}
        universe = set(universe_symbols)

        for symbol, position in account.positions.items():
            if symbol not in universe or position.quantity <= 0:
                continue
            existing_price = float(current.get(symbol, {}).get("buy_price", 0) or 0)
            api_price = float(position.avg_price or 0)
            buy_price = existing_price if api_price <= 0 and existing_price > 0 else api_price
            synced[symbol] = {"qty": int(position.quantity), "buy_price": buy_price}

        data[market] = synced
        self.save(data)
        return synced
