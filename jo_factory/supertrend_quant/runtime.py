from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MarketSession:
    state: str
    market: str | None
    is_close_briefing: bool
    timezone: ZoneInfo | None


def check_market_schedule(now_kr: datetime | None = None, now_us: datetime | None = None) -> MarketSession:
    kr_tz = ZoneInfo("Asia/Seoul")
    us_tz = ZoneInfo("America/New_York")
    kr_now = now_kr or datetime.now(kr_tz)
    us_now = now_us or datetime.now(us_tz)

    if kr_now.tzinfo is None:
        kr_now = kr_now.replace(tzinfo=kr_tz)
    if us_now.tzinfo is None:
        us_now = us_now.replace(tzinfo=us_tz)

    kr_time = kr_now.strftime("%H:%M:%S")
    us_time = us_now.strftime("%H:%M:%S")

    if kr_now.weekday() <= 4 and kr_time[:5] == "15:30":
        return MarketSession("KR_CLOSE", "KR", True, kr_tz)
    if us_now.weekday() <= 4 and us_time[:5] == "16:00":
        return MarketSession("US_CLOSE", "US", True, us_tz)
    if kr_now.weekday() <= 4 and "09:00:00" <= kr_time < "15:30:00":
        return MarketSession("KR", "KR", False, kr_tz)
    if us_now.weekday() <= 4 and "09:30:00" <= us_time < "16:00:00":
        return MarketSession("US", "US", False, us_tz)
    return MarketSession("SLEEP", None, False, None)


def current_30m_candle_base(now: datetime) -> datetime:
    return now.replace(minute=(now.minute // 30) * 30, second=0, microsecond=0)
