from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
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


def current_candle_base(now: datetime, timeframe: str = "30m") -> datetime:
    """Return the active candle boundary for supported runtime timeframes."""
    if timeframe == "1d":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes = {"30m": 30, "60m": 60, "1h": 60, "2h": 120, "4h": 240}.get(timeframe)
    if minutes is None:
        raise ValueError(f"Unsupported runtime timeframe: {timeframe}")
    day_minutes = now.hour * 60 + now.minute
    floored = (day_minutes // minutes) * minutes
    return now.replace(hour=floored // 60, minute=floored % 60, second=0, microsecond=0)


def current_30m_candle_base(now: datetime) -> datetime:
    """Backward-compatible alias for callers that explicitly need 30-minute bars."""
    return current_candle_base(now, "30m")


def last_completed_bar_end(now: datetime, market: str, timeframe: str) -> datetime:
    """Return the latest point at which a configured candle is fully closed.

    Intraday boundaries are anchored to the regular-session open instead of
    midnight.  For daily strategies the current date is an availability
    boundary; callers must keep only candles strictly older than that date.
    """
    if timeframe == "1d":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes = {"30m": 30, "60m": 60, "1h": 60, "2h": 120, "4h": 240}.get(timeframe)
    if minutes is None:
        raise ValueError(f"Unsupported runtime timeframe: {timeframe}")
    open_hour, open_minute = (9, 0) if market.upper() == "KR" else (9, 30)
    session_open = now.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)
    if now <= session_open:
        return session_open
    elapsed_minutes = int((now - session_open).total_seconds() // 60)
    completed = elapsed_minutes // minutes
    return session_open + timedelta(minutes=completed * minutes)
