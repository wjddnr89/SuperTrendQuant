from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from bootstrap import configure_imports
from toss_data import normalize_us_intraday, regular_session_bars


configure_imports()

from supertrend_quant.indicators import calculate_supertrend  # noqa: E402


@dataclass(frozen=True)
class IntradayExitSignal:
    symbol: str
    timeframe: str
    signal_at: str
    trend: int
    bar_count: int

    @property
    def sell(self) -> bool:
        return self.trend == -1

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal_at": self.signal_at,
            "trend": self.trend,
            "sell": self.sell,
            "bar_count": self.bar_count,
        }


@dataclass(frozen=True)
class IntradayFence:
    symbol: str
    timeframe: str
    bar_at: str
    trend: int
    lower_fence: float
    bar_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bar_at": self.bar_at,
            "trend": self.trend,
            "lower_fence": self.lower_fence,
            "bar_count": self.bar_count,
        }


@dataclass(frozen=True)
class IntradayReplayExit:
    symbol: str
    session_date: str
    signal_at: str
    signal_close: float
    fence_bar_at: str
    lower_fence: float
    trend: int
    fill_at: str | None
    raw_fill_open: float | None

    @property
    def pending_next_session(self) -> bool:
        return self.fill_at is None

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "session_date": self.session_date,
            "signal_at": self.signal_at,
            "signal_close": self.signal_close,
            "fence_bar_at": self.fence_bar_at,
            "lower_fence": self.lower_fence,
            "trend": self.trend,
            "fill_at": self.fill_at,
            "raw_fill_open": self.raw_fill_open,
            "pending_next_session": self.pending_next_session,
        }


def build_intraday_exit_signal(
    symbol: str,
    minute_bars: pd.DataFrame,
    *,
    seed_bars: pd.DataFrame | None = None,
    through: date,
    period: int,
    multiplier: float,
    timeframe_minutes: int = 120,
    timezone: str = "America/New_York",
    regular_open: str = "09:30",
    regular_close: str = "16:00",
) -> IntradayExitSignal:
    bars = _combined_bars(
        minute_bars,
        seed_bars=seed_bars,
        through=through,
        timeframe_minutes=timeframe_minutes,
        timezone=timezone,
        regular_open=regular_open,
        regular_close=regular_close,
    )
    if bars.empty:
        raise RuntimeError(
            f"No completed {timeframe_minutes}-minute bars for {symbol} "
            f"through {through}."
        )
    last_session = pd.Timestamp(bars.index[-1]).date()
    if last_session != through:
        raise RuntimeError(
            f"Latest {timeframe_minutes}-minute bar for {symbol} is "
            f"{last_session}, expected {through}."
        )
    featured = calculate_supertrend(
        bars,
        period=int(period),
        multiplier=float(multiplier),
        atr_method="wilder",
    )
    trend = int(featured.iloc[-1].get("Trend", 0))
    if trend not in {-1, 1}:
        raise RuntimeError(
            f"{symbol} {timeframe_minutes}-minute SuperTrend is neutral; "
            f"history may be insufficient ({len(featured)} bars)."
        )
    return IntradayExitSignal(
        symbol=symbol,
        timeframe=f"{timeframe_minutes // 60}h",
        signal_at=pd.Timestamp(featured.index[-1]).isoformat(),
        trend=trend,
        bar_count=int(len(featured)),
    )


def build_active_intraday_fence(
    symbol: str,
    minute_bars: pd.DataFrame,
    *,
    seed_bars: pd.DataFrame | None,
    as_of: datetime,
    period: int,
    multiplier: float,
    timeframe_minutes: int = 120,
    timezone: str = "America/New_York",
    regular_open: str = "09:30",
    regular_close: str = "16:00",
) -> IntradayFence:
    bars = _combined_bars(
        minute_bars,
        seed_bars=seed_bars,
        through=as_of.date(),
        timeframe_minutes=timeframe_minutes,
        timezone=timezone,
        regular_open=regular_open,
        regular_close=regular_close,
    )
    if bars.empty:
        raise RuntimeError(f"No 2h history is available for {symbol}.")
    localized_now = pd.Timestamp(as_of)
    if localized_now.tzinfo is None:
        localized_now = localized_now.tz_localize(timezone)
    else:
        localized_now = localized_now.tz_convert(timezone)
    close_hour, close_minute = (
        int(part) for part in regular_close.split(":", 1)
    )
    completed: list[bool] = []
    for start in pd.DatetimeIndex(bars.index):
        session_close = start.normalize() + pd.Timedelta(
            hours=close_hour,
            minutes=close_minute,
        )
        bar_end = min(
            start + pd.Timedelta(minutes=timeframe_minutes),
            session_close,
        )
        completed.append(bar_end <= localized_now)
    bars = bars.loc[completed]
    if bars.empty:
        raise RuntimeError(f"No completed 2h bar is available for {symbol}.")
    featured = calculate_supertrend(
        bars,
        period=int(period),
        multiplier=float(multiplier),
        atr_method="wilder",
    )
    row = featured.iloc[-1]
    trend = int(row.get("Trend", 0))
    lower_fence = float(row.get("Supertrend_Up", float("nan")))
    if trend not in {-1, 1} or not pd.notna(lower_fence):
        raise RuntimeError(
            f"{symbol} active 2h fence is not ready ({len(featured)} bars)."
        )
    return IntradayFence(
        symbol=symbol,
        timeframe=f"{timeframe_minutes // 60}h",
        bar_at=pd.Timestamp(featured.index[-1]).isoformat(),
        trend=trend,
        lower_fence=lower_fence,
        bar_count=int(len(featured)),
    )


def replay_intraday_fence_exit(
    symbol: str,
    adjusted_minutes: pd.DataFrame,
    raw_minutes: pd.DataFrame,
    *,
    seed_bars: pd.DataFrame | None,
    session_date: date,
    period: int,
    multiplier: float,
    confirm_minutes: int = 1,
    timeframe_minutes: int = 120,
    timezone: str = "America/New_York",
    regular_open: str = "09:30",
    regular_close: str = "16:00",
) -> IntradayReplayExit | None:
    """Replay one completed session without requiring a live watcher."""

    adjusted = _session_minutes(
        adjusted_minutes,
        session_date=session_date,
        timezone=timezone,
        regular_open=regular_open,
        regular_close=regular_close,
    )
    raw = _session_minutes(
        raw_minutes,
        session_date=session_date,
        timezone=timezone,
        regular_open=regular_open,
        regular_close=regular_close,
    )
    _require_completed_session(symbol, adjusted, session_date, regular_open, regular_close)
    _require_completed_session(symbol, raw, session_date, regular_open, regular_close)

    bars = _combined_bars(
        adjusted_minutes,
        seed_bars=seed_bars,
        through=session_date,
        timeframe_minutes=timeframe_minutes,
        timezone=timezone,
        regular_open=regular_open,
        regular_close=regular_close,
    )
    featured = calculate_supertrend(
        bars,
        period=int(period),
        multiplier=float(multiplier),
        atr_method="wilder",
    )
    close_hour, close_minute = (int(part) for part in regular_close.split(":", 1))
    bar_end_values = []
    for start in pd.DatetimeIndex(featured.index):
        session_close = start.normalize() + pd.Timedelta(
            hours=close_hour,
            minutes=close_minute,
        )
        bar_end_values.append(
            min(
                start + pd.Timedelta(minutes=timeframe_minutes),
                session_close,
            )
        )
    bar_ends = pd.Series(bar_end_values, index=featured.index)

    required_confirmations = max(1, int(confirm_minutes))
    consecutive_breaches = 0
    for minute_at, minute in adjusted.iterrows():
        completed = featured.loc[bar_ends <= minute_at + pd.Timedelta(minutes=1)]
        if completed.empty:
            continue
        fence = completed.iloc[-1]
        trend = int(fence.get("Trend", 0))
        lower_fence = float(fence.get("Supertrend_Up", float("nan")))
        if trend not in {-1, 1} or not pd.notna(lower_fence):
            continue
        minute_close = float(minute["Close"])
        breached = trend == -1 or minute_close < lower_fence
        consecutive_breaches = consecutive_breaches + 1 if breached else 0
        if consecutive_breaches < required_confirmations:
            continue
        next_rows = raw.loc[raw.index > minute_at]
        if next_rows.empty:
            fill_at = None
            raw_fill_open = None
        else:
            fill_at = pd.Timestamp(next_rows.index[0]).isoformat()
            raw_fill_open = float(next_rows.iloc[0]["Open"])
        return IntradayReplayExit(
            symbol=symbol,
            session_date=session_date.isoformat(),
            signal_at=pd.Timestamp(minute_at).isoformat(),
            signal_close=minute_close,
            fence_bar_at=pd.Timestamp(completed.index[-1]).isoformat(),
            lower_fence=lower_fence,
            trend=trend,
            fill_at=fill_at,
            raw_fill_open=raw_fill_open,
        )
    session_featured = featured.loc[
        [timestamp.date() == session_date for timestamp in featured.index]
    ]
    if not session_featured.empty:
        final_bar = session_featured.iloc[-1]
        final_trend = int(final_bar.get("Trend", 0))
        final_fence = float(final_bar.get("Supertrend_Up", float("nan")))
        if final_trend == -1 and pd.notna(final_fence):
            # A late-session break may not accumulate the requested number of
            # minute confirmations before 16:00.  A completed bearish 2h close
            # is stronger evidence, so carry its exit to the next session open.
            final_minute_at = pd.Timestamp(adjusted.index[-1])
            return IntradayReplayExit(
                symbol=symbol,
                session_date=session_date.isoformat(),
                signal_at=final_minute_at.isoformat(),
                signal_close=float(adjusted.iloc[-1]["Close"]),
                fence_bar_at=pd.Timestamp(session_featured.index[-1]).isoformat(),
                lower_fence=final_fence,
                trend=final_trend,
                fill_at=None,
                raw_fill_open=None,
            )
    return None


def override_held_exit_trends(
    prepared_frames: dict[str, pd.DataFrame],
    signals: dict[str, IntradayExitSignal],
) -> dict[str, pd.DataFrame]:
    """Replace only the held-symbol exit state on the latest daily row."""

    output = dict(prepared_frames)
    for symbol, signal in signals.items():
        frame = output.get(symbol)
        if frame is None or frame.empty:
            raise RuntimeError(
                f"Held symbol {symbol} lacks prepared daily strategy data."
            )
        updated = frame.copy()
        updated.iloc[-1, updated.columns.get_loc("Trend")] = signal.trend
        output[symbol] = updated
    return output


def _combined_bars(
    minute_bars: pd.DataFrame,
    *,
    seed_bars: pd.DataFrame | None,
    through: date,
    timeframe_minutes: int,
    timezone: str,
    regular_open: str,
    regular_close: str,
) -> pd.DataFrame:
    recent = regular_session_bars(
        minute_bars,
        bar_minutes=timeframe_minutes,
        timezone=timezone,
        regular_open=regular_open,
        regular_close=regular_close,
        through=through,
    )
    seed = regular_session_bars(
        seed_bars if seed_bars is not None else pd.DataFrame(),
        bar_minutes=timeframe_minutes,
        timezone=timezone,
        regular_open=regular_open,
        regular_close=regular_close,
        through=through,
    )
    bars = pd.concat([seed, recent]).sort_index()
    return bars.loc[~bars.index.duplicated(keep="last")]


def _session_minutes(
    frame: pd.DataFrame,
    *,
    session_date: date,
    timezone: str,
    regular_open: str,
    regular_close: str,
) -> pd.DataFrame:
    localized = normalize_us_intraday(frame, timezone=timezone)
    open_clock = _clock_minutes(regular_open)
    close_clock = _clock_minutes(regular_close)
    selected = localized.loc[
        [
            timestamp.date() == session_date
            and open_clock <= timestamp.hour * 60 + timestamp.minute < close_clock
            for timestamp in localized.index
        ]
    ]
    return selected.sort_index()


def _require_completed_session(
    symbol: str,
    frame: pd.DataFrame,
    session_date: date,
    regular_open: str,
    regular_close: str,
) -> None:
    if frame.empty:
        raise RuntimeError(f"No intraday data for {symbol} on {session_date}.")
    expected_first = _clock_minutes(regular_open)
    expected_last = _clock_minutes(regular_close) - 1
    first = frame.index[0].hour * 60 + frame.index[0].minute
    last = frame.index[-1].hour * 60 + frame.index[-1].minute
    if first > expected_first or last < expected_last:
        raise RuntimeError(
            f"Incomplete intraday session for {symbol} on {session_date}: "
            f"{frame.index[0]} through {frame.index[-1]}."
        )


def _clock_minutes(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":", 1))
    return hour * 60 + minute
