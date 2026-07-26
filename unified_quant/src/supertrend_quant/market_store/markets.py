from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd


@dataclass(frozen=True)
class MarketSpec:
    market: str
    calendar: str
    timezone: str
    currency: str
    publication_delay: timedelta
    open_time: str
    close_time: str


MARKET_SPECS: dict[str, MarketSpec] = {
    "US": MarketSpec(
        market="US",
        calendar="XNYS",
        timezone="America/New_York",
        currency="USD",
        publication_delay=timedelta(minutes=90),
        open_time="09:30",
        close_time="16:00",
    ),
    "KR": MarketSpec(
        market="KR",
        calendar="XKRX",
        timezone="Asia/Seoul",
        currency="KRW",
        publication_delay=timedelta(minutes=90),
        open_time="09:00",
        close_time="15:30",
    ),
}

# ``exchange_calendars`` is necessarily released before some newly enacted or
# ad-hoc exchange holidays are known.  Keep audited, narrowly scoped overrides
# here instead of silently treating an empty price response as a trading halt.
# 2026-07-17: Constitution Day became a public holiday again in 2026; KRX
# markets are closed on public holidays.
MARKET_CLOSURE_OVERRIDES: dict[str, frozenset[str]] = {
    "KR": frozenset({"2026-06-03", "2026-07-17"}),
}
MARKET_CLOSURE_OVERRIDE_SOURCES: dict[str, dict[str, tuple[str, ...]]] = {
    "KR": {
        "2026-06-03": (
            "https://www.nec.go.kr/site/nec/ex/bbs/"
            "View.do?bcIdx=289351&cbIdx=1104",
            "https://global.krx.co.kr/contents/GLB/06/0602/0602010201/"
            "GLB0602010201T1.jsp",
        ),
        "2026-07-17": (
            "https://www.mois.go.kr/video/bbs/type019/"
            "commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000255&nttId=123641",
            "https://global.krx.co.kr/contents/GLB/06/0602/0602010201/"
            "GLB0602010201T1.jsp",
        ),
    },
}


def market_spec(market: str) -> MarketSpec:
    normalized = str(market).strip().upper()
    try:
        return MARKET_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported market: {market}") from exc


def exchange_calendar(market: str):
    try:
        import exchange_calendars as xcals
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "exchange-calendars is required for market session calculations."
        ) from exc
    return xcals.get_calendar(market_spec(market).calendar)


def _remove_known_market_closures(
    market: str,
    sessions: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    closures = MARKET_CLOSURE_OVERRIDES.get(str(market).strip().upper(), frozenset())
    if not closures or len(sessions) == 0:
        return sessions
    return sessions[
        ~sessions.strftime("%Y-%m-%d").isin(closures)
    ]


def expected_completed_session(
    market: str,
    now: datetime | None = None,
    *,
    publication_delay: timedelta | None = None,
) -> str:
    """Return the newest exchange session whose provider delay has elapsed."""

    spec = market_spec(market)
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    current_utc = current.astimezone(UTC)
    calendar = exchange_calendar(spec.market)
    today = pd.Timestamp(current_utc.date())
    sessions = _remove_known_market_closures(
        spec.market,
        calendar.sessions_in_range(today - pd.Timedelta(days=21), today),
    )
    if len(sessions) == 0:
        raise RuntimeError(f"Could not resolve a recent {spec.calendar} session.")
    delay = spec.publication_delay if publication_delay is None else publication_delay
    last = sessions[-1]
    if current_utc >= calendar.session_close(last).to_pydatetime() + delay:
        return last.date().isoformat()
    if len(sessions) < 2:
        raise RuntimeError(f"Could not resolve the preceding {spec.calendar} session.")
    return sessions[-2].date().isoformat()


def recent_sessions(
    market: str,
    count: int,
    *,
    end: str | None = None,
) -> tuple[str, ...]:
    if count <= 0:
        raise ValueError("count must be positive.")
    calendar = exchange_calendar(market)
    end_session = pd.Timestamp(end or expected_completed_session(market))
    start = end_session - pd.Timedelta(days=max(32, count * 2 + 14))
    sessions = _remove_known_market_closures(
        market,
        calendar.sessions_in_range(start, end_session),
    )
    if len(sessions) < count:
        start = end_session - pd.Timedelta(days=count * 3 + 31)
        sessions = _remove_known_market_closures(
            market,
            calendar.sessions_in_range(start, end_session),
        )
    if len(sessions) < count:
        raise RuntimeError(
            f"Could not resolve {count} {market_spec(market).calendar} sessions."
        )
    return tuple(value.date().isoformat() for value in sessions[-count:])


def sessions_between(
    market: str,
    start: str,
    end: str,
) -> tuple[str, ...]:
    sessions = _remove_known_market_closures(
        market,
        exchange_calendar(market).sessions_in_range(
            pd.Timestamp(start),
            pd.Timestamp(end),
        ),
    )
    if len(sessions) == 0:
        raise RuntimeError(
            f"No {market_spec(market).calendar} sessions between {start} and {end}."
        )
    return tuple(value.date().isoformat() for value in sessions)


def release_metadata(market: str, **extra) -> dict[str, object]:
    spec = market_spec(market)
    closure_overrides = {
        session: list(sources)
        for session, sources in MARKET_CLOSURE_OVERRIDE_SOURCES.get(
            spec.market, {}
        ).items()
    }
    return {
        "market": spec.market,
        "calendar": spec.calendar,
        "timezone": spec.timezone,
        "currency": spec.currency,
        "calendar_closure_overrides": closure_overrides,
        **extra,
    }
