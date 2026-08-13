from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, time as wall_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from bootstrap import configure_imports


configure_imports()

from supertrend_quant.env import load_env  # noqa: E402


CANDLE_COLUMNS = {
    "openPrice": "Open",
    "highPrice": "High",
    "lowPrice": "Low",
    "closePrice": "Close",
    "volume": "Volume",
}


@dataclass(frozen=True)
class CandlePage:
    frame: pd.DataFrame
    next_before: str | None


class TossMarketDataClient:
    """Read-only Toss market-data client.

    This class never sends an account header and exposes no order method.
    """

    base_url = "https://openapi.tossinvest.com"

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        request_interval_seconds: float = 0.22,
        timeout_seconds: float = 15.0,
    ) -> None:
        load_env()
        self.client_id = os.getenv("TOSS_CLIENT_ID", "").strip()
        self.client_secret = os.getenv("TOSS_CLIENT_SECRET", "").strip()
        self.session = session or requests.Session()
        self.request_interval_seconds = max(0.0, float(request_interval_seconds))
        self.timeout_seconds = float(timeout_seconds)
        self.token: str | None = None
        self.token_expiry = 0.0
        self._last_request_at = 0.0

    def credentials_available(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def candle_page(
        self,
        symbol: str,
        *,
        interval: str,
        count: int = 200,
        before: str | None = None,
        adjusted: bool = True,
    ) -> CandlePage:
        if interval not in {"1m", "1d"}:
            raise ValueError("interval must be 1m or 1d.")
        count = int(count)
        if count < 1 or count > 200:
            raise ValueError("count must be between 1 and 200.")
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "count": count,
            "adjusted": str(bool(adjusted)).lower(),
        }
        if before:
            params["before"] = before
        payload = self._get("/api/v1/candles", params=params)
        result = payload.get("result", {})
        candles = result.get("candles", [])
        return CandlePage(
            frame=_candles_to_frame(candles),
            next_before=result.get("nextBefore"),
        )

    def fetch_candles(
        self,
        symbol: str,
        *,
        interval: str,
        minimum_bars: int,
        adjusted: bool,
        before: str | None = None,
        max_pages: int = 30,
    ) -> pd.DataFrame:
        """Fetch at least ``minimum_bars`` using the documented cursor."""

        minimum_bars = max(1, int(minimum_bars))
        frames: list[pd.DataFrame] = []
        cursor = before
        seen_cursors: set[str] = set()
        unique_count = 0
        for _ in range(max(1, int(max_pages))):
            page = self.candle_page(
                symbol,
                interval=interval,
                count=min(200, max(1, minimum_bars - unique_count)),
                before=cursor,
                adjusted=adjusted,
            )
            if not page.frame.empty:
                frames.append(page.frame)
                unique_count = len(
                    pd.concat(frames).loc[
                        lambda value: ~value.index.duplicated(keep="last")
                    ]
                )
            if unique_count >= minimum_bars or not page.next_before:
                break
            next_cursor = str(page.next_before)
            if next_cursor in seen_cursors:
                raise RuntimeError(
                    f"Toss candle pagination did not advance for {symbol} {interval}: "
                    f"{next_cursor}"
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        if not frames:
            return _empty_candle_frame()
        output = pd.concat(frames).sort_index()
        output = output.loc[~output.index.duplicated(keep="last")]
        return output

    def first_regular_minute(
        self,
        symbol: str,
        session_date: date,
        *,
        timezone: str = "America/New_York",
        regular_open: str = "09:30",
        regular_close: str = "16:00",
        max_pages: int = 30,
    ) -> pd.Series | None:
        """Return the first regular-session minute, paging past the 200-row cap."""

        market_tz = ZoneInfo(timezone)
        open_time = _parse_clock(regular_open)
        close_time = _parse_clock(regular_close)
        cursor: str | None = None
        seen_cursors: set[str] = set()
        matches: list[pd.DataFrame] = []

        for _ in range(max(1, int(max_pages))):
            page = self.candle_page(
                symbol,
                interval="1m",
                count=200,
                before=cursor,
                adjusted=False,
            )
            if page.frame.empty:
                break
            localized = page.frame.copy()
            index = pd.DatetimeIndex(localized.index)
            if index.tz is None:
                index = index.tz_localize(market_tz)
            else:
                index = index.tz_convert(market_tz)
            localized.index = index
            mask = [
                timestamp.date() == session_date
                and open_time <= timestamp.time().replace(tzinfo=None) < close_time
                for timestamp in localized.index
            ]
            if any(mask):
                matches.append(localized.loc[mask])

            earliest = localized.index.min().date()
            if earliest < session_date or not page.next_before:
                break
            next_cursor = str(page.next_before)
            if next_cursor in seen_cursors:
                raise RuntimeError(
                    f"Toss minute pagination did not advance for {symbol}: {next_cursor}"
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        if not matches:
            return None
        session_frame = pd.concat(matches).sort_index()
        session_frame = session_frame.loc[
            ~session_frame.index.duplicated(keep="last")
        ]
        return session_frame.iloc[0]

    def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", path, params=params)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        token = self._token()
        headers = {"Authorization": f"Bearer {token}"}
        last_error: Exception | None = None
        for attempt in range(5):
            self._throttle()
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                params=params,
                timeout=self.timeout_seconds,
            )
            self._last_request_at = time.monotonic()
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()
            retry_after = _finite_float(response.headers.get("Retry-After")) or (
                2**attempt
            )
            last_error = RuntimeError(
                f"Toss rate limit for {path}; retry after {retry_after}s"
            )
            time.sleep(min(30.0, max(0.1, retry_after)))
        raise last_error or RuntimeError(f"Toss request failed: {path}")

    def _token(self) -> str:
        if not self.credentials_available():
            raise RuntimeError(
                "TOSS_CLIENT_ID and TOSS_CLIENT_SECRET are required in the "
                "repository-root .env file."
            )
        if self.token and time.time() < self.token_expiry - 60:
            return self.token
        self._throttle()
        response = self.session.post(
            f"{self.base_url}/oauth2/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=self.timeout_seconds,
        )
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        payload = response.json()
        self.token = str(payload["access_token"])
        self.token_expiry = time.time() + int(payload.get("expires_in", 86400))
        return self.token

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)


class TossDailyCache:
    def __init__(
        self,
        client: TossMarketDataClient,
        root: str | Path,
        *,
        history_bars: int,
        refresh_bars: int,
    ) -> None:
        self.client = client
        self.root = Path(root)
        self.history_bars = int(history_bars)
        self.refresh_bars = int(refresh_bars)

    def load_symbol(self, symbol: str, *, adjusted: bool) -> pd.DataFrame:
        mode = "adjusted" if adjusted else "raw"
        path = self.root / "daily" / mode / f"{_safe_symbol(symbol)}.csv"
        existing = normalize_us_daily(_read_frame(path))
        minimum = (
            self.refresh_bars
            if len(existing) >= self.history_bars
            else self.history_bars
        )
        fetched = normalize_us_daily(
            self.client.fetch_candles(
            symbol,
            interval="1d",
            minimum_bars=minimum,
            adjusted=adjusted,
            max_pages=max(3, math.ceil(self.history_bars / 200) + 1),
            )
        )
        merged = pd.concat([existing, fetched]).sort_index()
        merged = merged.loc[~merged.index.duplicated(keep="last")]
        _write_frame(path, merged)
        return merged

    def load_universe(
        self,
        symbols: list[str],
        *,
        raw_symbols: set[str] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        adjusted: dict[str, pd.DataFrame] = {}
        raw: dict[str, pd.DataFrame] = {}
        raw_symbols = set(symbols) if raw_symbols is None else set(raw_symbols)
        for index, symbol in enumerate(symbols, start=1):
            print(
                f"[daily-data] {index}/{len(symbols)} {symbol}",
                flush=True,
            )
            adjusted[symbol] = self.load_symbol(symbol, adjusted=True)
            if symbol in raw_symbols:
                raw[symbol] = self.load_symbol(symbol, adjusted=False)
        return adjusted, raw


class TossIntradayCache:
    """Persist a rolling adjusted 1-minute history for held-symbol exits."""

    def __init__(
        self,
        client: TossMarketDataClient,
        root: str | Path,
        *,
        history_minutes: int,
        refresh_minutes: int,
        max_pages: int,
        timezone: str,
    ) -> None:
        self.client = client
        self.root = Path(root)
        self.history_minutes = int(history_minutes)
        self.refresh_minutes = int(refresh_minutes)
        self.max_pages = int(max_pages)
        self.timezone = str(timezone)

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        path = self.root / "intraday" / "adjusted" / f"{_safe_symbol(symbol)}.csv"
        existing = normalize_us_intraday(
            _read_intraday_frame(path),
            timezone=self.timezone,
        )
        minimum = (
            self.refresh_minutes
            if len(existing) >= self.history_minutes
            else self.history_minutes
        )
        fetched = normalize_us_intraday(
            self.client.fetch_candles(
                symbol,
                interval="1m",
                minimum_bars=minimum,
                adjusted=True,
                max_pages=self.max_pages,
            ),
            timezone=self.timezone,
        )
        merged = pd.concat([existing, fetched]).sort_index()
        merged = merged.loc[~merged.index.duplicated(keep="last")]
        _write_frame(path, merged)
        return merged

    def refresh_symbol(
        self,
        symbol: str,
        *,
        latest_minutes: int = 10,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        mode = "adjusted" if adjusted else "raw"
        path = self.root / "intraday" / mode / f"{_safe_symbol(symbol)}.csv"
        existing = normalize_us_intraday(
            _read_intraday_frame(path),
            timezone=self.timezone,
        )
        fetched = normalize_us_intraday(
            self.client.fetch_candles(
                symbol,
                interval="1m",
                minimum_bars=max(2, int(latest_minutes)),
                adjusted=adjusted,
                max_pages=max(
                    2,
                    min(
                        self.max_pages,
                        math.ceil(max(2, int(latest_minutes)) / 200) + 1,
                    ),
                ),
            ),
            timezone=self.timezone,
        )
        merged = pd.concat([existing, fetched]).sort_index()
        merged = merged.loc[~merged.index.duplicated(keep="last")]
        _write_frame(path, merged)
        return merged


class YahooHourlySeedCache:
    """One-time adjusted hourly seed used before Toss minutes accumulate."""

    def __init__(
        self,
        root: str | Path,
        *,
        period: str,
        timezone: str,
    ) -> None:
        self.root = Path(root)
        self.period = str(period)
        self.timezone = str(timezone)

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        path = (
            self.root
            / "intraday"
            / "seed_yahoo_60m"
            / f"{_safe_symbol(symbol)}.csv"
        )
        existing = normalize_us_intraday(
            _read_intraday_frame(path),
            timezone=self.timezone,
        )
        if not existing.empty:
            return existing
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "yfinance is required once to seed the 2h exit history."
            ) from exc
        downloaded = yf.download(
            symbol,
            period=self.period,
            interval="60m",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if isinstance(downloaded.columns, pd.MultiIndex):
            downloaded.columns = downloaded.columns.get_level_values(0)
        missing = {
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        } - set(downloaded.columns)
        if downloaded.empty or missing:
            raise RuntimeError(
                f"Yahoo hourly seed is unavailable for {symbol}; missing={missing}."
            )
        seed = normalize_us_intraday(
            downloaded[list(CANDLE_COLUMNS.values())],
            timezone=self.timezone,
        )
        _write_frame(path, seed)
        return seed


class SessionOpenCache:
    def __init__(
        self,
        client: TossMarketDataClient,
        path: str | Path,
        *,
        timezone: str,
        regular_open: str,
        regular_close: str,
        max_pages: int,
    ) -> None:
        self.client = client
        self.path = Path(path)
        self.timezone = timezone
        self.regular_open = regular_open
        self.regular_close = regular_close
        self.max_pages = int(max_pages)

    def price(self, symbol: str, session_date: date) -> float | None:
        cached = self._read()
        key = (session_date.isoformat(), symbol)
        if key in cached:
            return cached[key]
        row = self.client.first_regular_minute(
            symbol,
            session_date,
            timezone=self.timezone,
            regular_open=self.regular_open,
            regular_close=self.regular_close,
            max_pages=self.max_pages,
        )
        if row is None:
            return None
        price = _finite_float(row.get("Open"))
        if price is None or price <= 0:
            return None
        self._append(session_date, symbol, price)
        return price

    def _read(self) -> dict[tuple[str, str], float]:
        if not self.path.exists():
            return {}
        frame = pd.read_csv(self.path)
        return {
            (str(row.session_date), str(row.symbol)): float(row.open_price)
            for row in frame.itertuples(index=False)
        }

    def _append(self, session_date: date, symbol: str, price: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame(
            [
                {
                    "session_date": session_date.isoformat(),
                    "symbol": symbol,
                    "open_price": price,
                    "captured_at": datetime.now().astimezone().isoformat(),
                }
            ]
        )
        if self.path.exists():
            existing = pd.read_csv(self.path)
            combined = pd.concat([existing, row], ignore_index=True)
            combined = combined.drop_duplicates(
                ["session_date", "symbol"], keep="last"
            )
        else:
            combined = row
        temporary = self.path.with_suffix(".tmp")
        combined.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, self.path)


def latest_completed_signal_date(
    benchmark: pd.DataFrame,
    execution_date: date,
) -> date:
    candidates = [
        timestamp.date()
        for timestamp in pd.DatetimeIndex(benchmark.index)
        if timestamp.date() < execution_date
    ]
    if not candidates:
        raise RuntimeError(
            f"No completed benchmark candle before {execution_date.isoformat()}."
        )
    return max(candidates)


def normalize_us_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Map provider timestamps to timezone-neutral US session dates."""

    if frame.empty:
        return frame.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert("America/New_York")
    normalized = frame.copy()
    normalized.index = pd.DatetimeIndex(
        [pd.Timestamp(timestamp.date()) for timestamp in index],
        name="timestamp",
    )
    normalized = normalized.loc[~normalized.index.duplicated(keep="last")]
    return normalized.sort_index()


def normalize_us_intraday(
    frame: pd.DataFrame,
    *,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize(timezone)
    else:
        index = index.tz_convert(timezone)
    normalized = frame.copy()
    normalized.index = index
    normalized.index.name = "timestamp"
    return normalized.sort_index()


def regular_session_bars(
    minute_bars: pd.DataFrame,
    *,
    bar_minutes: int,
    timezone: str = "America/New_York",
    regular_open: str = "09:30",
    regular_close: str = "16:00",
    through: date | None = None,
) -> pd.DataFrame:
    """Aggregate regular-session minutes into bars anchored at the open."""

    if minute_bars.empty:
        return _empty_candle_frame()
    bar_minutes = int(bar_minutes)
    if bar_minutes < 1:
        raise ValueError("bar_minutes must be positive.")
    localized = normalize_us_intraday(minute_bars, timezone=timezone)
    open_time = _parse_clock(regular_open)
    close_time = _parse_clock(regular_close)
    open_minutes = open_time.hour * 60 + open_time.minute
    keys: list[tuple[date, int] | None] = []
    for timestamp in localized.index:
        clock = timestamp.time().replace(tzinfo=None)
        session = timestamp.date()
        if (
            clock < open_time
            or clock >= close_time
            or (through is not None and session > through)
        ):
            keys.append(None)
            continue
        minute = timestamp.hour * 60 + timestamp.minute - open_minutes
        keys.append((session, minute // bar_minutes))

    records: list[dict[str, float]] = []
    indices: list[pd.Timestamp] = []
    keyed = localized.copy()
    keyed["_bucket"] = keys
    keyed = keyed.loc[keyed["_bucket"].notna()]
    for (session, bucket), piece in keyed.groupby("_bucket", sort=True):
        start = pd.Timestamp(session, tz=timezone) + pd.Timedelta(
            minutes=open_minutes + int(bucket) * bar_minutes
        )
        records.append(
            {
                "Open": float(piece.iloc[0]["Open"]),
                "High": float(piece["High"].max()),
                "Low": float(piece["Low"].min()),
                "Close": float(piece.iloc[-1]["Close"]),
                "Volume": float(piece["Volume"].sum()),
            }
        )
        indices.append(start)
    return pd.DataFrame(
        records,
        index=pd.DatetimeIndex(indices, name="timestamp"),
        columns=list(CANDLE_COLUMNS.values()),
    ).sort_index()


def truncate_daily(frame: pd.DataFrame, through: date) -> pd.DataFrame:
    mask = [timestamp.date() <= through for timestamp in pd.DatetimeIndex(frame.index)]
    return frame.loc[mask].copy()


def close_on(frame: pd.DataFrame, session_date: date) -> float | None:
    matches = frame.loc[
        [timestamp.date() == session_date for timestamp in pd.DatetimeIndex(frame.index)]
    ]
    if matches.empty:
        return None
    return _finite_float(matches.iloc[-1].get("Close"))


def _candles_to_frame(candles: list[dict[str, Any]]) -> pd.DataFrame:
    if not candles:
        return _empty_candle_frame()
    frame = pd.DataFrame(candles)
    if "timestamp" not in frame:
        return _empty_candle_frame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp")
    frame = frame.rename(columns=CANDLE_COLUMNS)
    for column in CANDLE_COLUMNS.values():
        if column not in frame:
            frame[column] = 0.0 if column == "Volume" else float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[list(CANDLE_COLUMNS.values())].sort_index()


def _empty_candle_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CANDLE_COLUMNS.values()))


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_candle_frame()
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    return frame.sort_index()


def _read_intraday_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_candle_frame()
    frame = pd.read_csv(path, index_col="timestamp")
    frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    frame = frame.loc[~frame.index.isna()]
    return frame.sort_index()


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output.index.name = "timestamp"
    temporary = path.with_suffix(".tmp")
    output.to_csv(temporary, encoding="utf-8-sig")
    os.replace(temporary, path)


def _parse_clock(value: str) -> wall_time:
    hour, minute = (int(part) for part in value.split(":", 1))
    return wall_time(hour=hour, minute=minute)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
