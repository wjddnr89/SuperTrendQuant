from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, TYPE_CHECKING

from .portfolio import AccountSnapshot

if TYPE_CHECKING:
    from .universe import UniverseMember


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

    def sync_market(
        self,
        market: str,
        account: AccountSnapshot,
        universe_symbols: list[str],
        members: Mapping[str, UniverseMember] | None = None,
    ) -> dict[str, dict]:
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
            item = {"qty": int(position.quantity), "buy_price": buy_price}
            member = members.get(symbol) if members else None
            if member is not None:
                item.update(
                    {
                        "market": member.market,
                        "exchange": member.exchange,
                        "name": member.name,
                        "security_type": member.security_type,
                        "yfinance_symbol": member.yfinance_symbol,
                        "benchmark": member.benchmark,
                        "profiles": list(member.profiles),
                    }
                )
            synced[symbol] = item

        data[market] = synced
        self.save(data)
        return synced

    def member_map(self, market: str) -> dict[str, UniverseMember]:
        from .universe import UniverseMember

        current = self.load().get(market, {})
        members: dict[str, UniverseMember] = {}
        for symbol, raw in current.items():
            if not isinstance(raw, dict):
                continue
            exchange = str(raw.get("exchange") or ("KOSPI" if market == "KR" else "US"))
            yfinance_symbol = str(raw.get("yfinance_symbol") or "")
            if not yfinance_symbol:
                yfinance_symbol = (
                    f"{symbol}.KQ"
                    if exchange == "KOSDAQ"
                    else f"{symbol}.KS"
                    if market == "KR"
                    else str(symbol).replace(".", "-")
                )
            benchmark = str(raw.get("benchmark") or ("^KQ11" if exchange == "KOSDAQ" else "^KS11" if market == "KR" else "QQQ"))
            members[str(symbol)] = UniverseMember(
                symbol=str(symbol),
                market=market,
                exchange=exchange,
                name=str(raw.get("name") or ""),
                security_type=str(raw.get("security_type") or "STOCK"),
                yfinance_symbol=yfinance_symbol,
                benchmark=benchmark,
                profiles=tuple(str(value) for value in raw.get("profiles", ()) or ()),
            )
        return members
