from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from .market_store.markets import exchange_calendar, market_spec


@dataclass(frozen=True)
class MarketSession:
    state: str
    market: str | None
    is_close_briefing: bool
    timezone: ZoneInfo | None


@dataclass(frozen=True)
class DailyExecutionWindow:
    market: str
    execution_session: str
    signal_session: str
    opens_at: datetime
    expires_at: datetime
    allowed: bool


def check_market_schedule(now_kr: datetime | None = None, now_us: datetime | None = None) -> MarketSession:
    kr_tz = ZoneInfo("Asia/Seoul")
    us_tz = ZoneInfo("America/New_York")
    kr_now = now_kr or datetime.now(kr_tz)
    us_now = now_us or datetime.now(us_tz)

    if kr_now.tzinfo is None:
        kr_now = kr_now.replace(tzinfo=kr_tz)
    if us_now.tzinfo is None:
        us_now = us_now.replace(tzinfo=us_tz)

    kr_bounds = _session_bounds("KR", kr_now)
    us_bounds = _session_bounds("US", us_now)

    if kr_bounds and kr_bounds[1] <= kr_now < kr_bounds[1] + timedelta(minutes=1):
        return MarketSession("KR_CLOSE", "KR", True, kr_tz)
    if us_bounds and us_bounds[1] <= us_now < us_bounds[1] + timedelta(minutes=1):
        return MarketSession("US_CLOSE", "US", True, us_tz)
    if kr_bounds and kr_bounds[0] <= kr_now < kr_bounds[1]:
        return MarketSession("KR", "KR", False, kr_tz)
    if us_bounds and us_bounds[0] <= us_now < us_bounds[1]:
        return MarketSession("US", "US", False, us_tz)
    return MarketSession("SLEEP", None, False, None)


def daily_execution_window(
    market: str,
    now: datetime,
    *,
    minutes: int = 15,
) -> DailyExecutionWindow | None:
    """Resolve the actual exchange session and its D+1 opening window."""

    normalized_market = str(market).upper()
    spec = market_spec(normalized_market)
    timezone = ZoneInfo(spec.timezone)
    local_now = now.replace(tzinfo=timezone) if now.tzinfo is None else now.astimezone(timezone)
    bounds = _session_bounds(normalized_market, local_now)
    if bounds is None:
        return None
    opens_at, _ = bounds
    expires_at = opens_at + timedelta(minutes=max(1, int(minutes)))
    session = pd.Timestamp(local_now.date())
    signal_session = exchange_calendar(normalized_market).previous_session(session)
    return DailyExecutionWindow(
        market=normalized_market,
        execution_session=session.date().isoformat(),
        signal_session=signal_session.date().isoformat(),
        opens_at=opens_at,
        expires_at=expires_at,
        allowed=opens_at <= local_now < expires_at,
    )


def _session_bounds(
    market: str,
    now: datetime,
) -> tuple[datetime, datetime] | None:
    spec = market_spec(market)
    timezone = ZoneInfo(spec.timezone)
    local_now = now.replace(tzinfo=timezone) if now.tzinfo is None else now.astimezone(timezone)
    session = pd.Timestamp(local_now.date())
    calendar = exchange_calendar(market)
    if not calendar.is_session(session):
        return None
    opens_at = calendar.session_open(session).to_pydatetime().astimezone(timezone)
    closes_at = calendar.session_close(session).to_pydatetime().astimezone(timezone)
    return opens_at, closes_at


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
