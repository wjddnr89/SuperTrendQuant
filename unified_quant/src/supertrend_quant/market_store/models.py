from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DataQuality(StrEnum):
    VALID = "valid"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class CorporateActionType(StrEnum):
    CASH_DIVIDEND = "cash_dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    SPLIT = "split"
    CAPITAL_REDUCTION = "capital_reduction"
    STOCK_DIVIDEND = "stock_dividend"
    SPINOFF = "spinoff"
    CASH_MERGER = "cash_merger"
    STOCK_MERGER = "stock_merger"
    TICKER_CHANGE = "ticker_change"
    DELISTING = "delisting"


@dataclass(frozen=True)
class SourceMetadata:
    source: str
    retrieved_at: str
    source_hash: str
    source_url: str = ""
    source_kind: str = "provider"
    official: bool = False


@dataclass(frozen=True)
class CorporateAction:
    event_id: str
    security_id: str
    action_type: CorporateActionType | str
    effective_date: str
    ex_date: str = ""
    announcement_date: str = ""
    record_date: str = ""
    payment_date: str = ""
    cash_amount: float | None = None
    ratio: float | None = None
    currency: str = "USD"
    new_security_id: str = ""
    new_symbol: str = ""
    source: SourceMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        source = self.source
        return {
            "event_id": self.event_id,
            "security_id": self.security_id,
            "action_type": str(self.action_type),
            "effective_date": self.effective_date,
            "ex_date": self.ex_date or self.effective_date,
            "announcement_date": self.announcement_date,
            "record_date": self.record_date,
            "payment_date": self.payment_date,
            "cash_amount": self.cash_amount,
            "ratio": self.ratio,
            "currency": self.currency,
            "new_security_id": self.new_security_id,
            "new_symbol": self.new_symbol,
            "source": source.source if source else "",
            "source_url": source.source_url if source else "",
            "source_kind": source.source_kind if source else "",
            "retrieved_at": source.retrieved_at if source else "",
            "source_hash": source.source_hash if source else "",
            "official": source.official if source else False,
            "metadata": dict(self.metadata),
        }
