"""Strict Yahoo chart JSON client with immutable exact-byte response caching.

The client is intentionally small and personal-use only.  It performs at most
one HTTP request for each missing bounded symbol/date request, never retries,
never uses a crumb or token, and preserves the exact response bytes inside a
hash-checked gzip envelope.  Parsing is deliberately separate so HTTP/API
failures can be kept as evidence without ever being accepted as price history.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from .manifest import sha256_bytes, utc_now_iso, write_atomic


DEFAULT_ENDPOINT_TEMPLATE = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
)
_BOUNDED_QUERY = (
    ("events", "history"),
    ("includeAdjustedClose", "true"),
    ("interval", "1d"),
)
CACHE_SCHEMA = "yahoo_chart_raw_response/v2"
ALLOWED_US_EXCHANGE_NAMES = frozenset(
    {"ASE", "BTS", "NCM", "NGM", "NMS", "NYQ", "OQB", "OQX", "PCX", "PNK"}
)
US_EXCHANGE_TIMEZONE = "America/New_York"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def normalize_yahoo_symbol(symbol: str) -> str:
    """Normalize the US-equity forms used by the lifecycle datasets."""

    value = str(symbol).strip().upper()
    if value.endswith(".US"):
        value = value[:-3]
    value = value.replace(".", "-").replace("/", "-")
    if not value or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for character in value
    ):
        raise ValueError(f"Invalid Yahoo chart symbol: {symbol!r}")
    return value


def _validate_endpoint_template(endpoint_template: str) -> None:
    parsed = urlparse(endpoint_template.replace("{symbol}", "SYMBOL"))
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "query1.finance.yahoo.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v8/finance/chart/SYMBOL"
        or endpoint_template.count("{symbol}") != 1
    ):
        raise ValueError("Yahoo chart endpoint template is invalid.")


@dataclass(frozen=True)
class YahooChartCachedResponse:
    symbol: str
    source_url: str
    retrieved_at: str
    content: bytes
    content_type: str
    http_status: int
    wrapper_hash: str = ""
    request_period1: int = 0
    request_period2: int = 0

    @property
    def source_hash(self) -> str:
        return sha256_bytes(self.content)


@dataclass(frozen=True)
class YahooChartData:
    symbol: str
    currency: str
    bars: pd.DataFrame
    adjustment_basis: str = "raw_quote_ohlcv"


@dataclass(frozen=True)
class YahooChartNoDataEvidence:
    symbol: str
    kind: str
    http_status: int
    error_code: str = ""
    error_description: str = ""


def validate_yahoo_equity_metadata(
    meta: Any,
    expected_symbol: str,
) -> tuple[str, str]:
    """Require one USD US-equity identity, not merely a matching ticker label."""

    if not isinstance(meta, dict):
        raise ValueError("Yahoo chart result lacks metadata.")
    symbol = normalize_yahoo_symbol(str(meta.get("symbol", "")))
    expected = normalize_yahoo_symbol(expected_symbol)
    if symbol != expected:
        raise ValueError(
            f"Yahoo chart metadata symbol mismatch: expected {expected}, got {symbol}."
        )
    currency = str(meta.get("currency", "")).strip().upper()
    if currency != "USD":
        raise ValueError(f"Yahoo chart currency must be USD, got {currency or 'missing'}.")
    instrument_type = str(meta.get("instrumentType", "")).strip().upper()
    if instrument_type != "EQUITY":
        raise ValueError(
            "Yahoo chart instrument type must be EQUITY, got "
            f"{instrument_type or 'missing'}."
        )
    exchange_name = str(meta.get("exchangeName", "")).strip().upper()
    if exchange_name not in ALLOWED_US_EXCHANGE_NAMES:
        raise ValueError(
            "Yahoo chart exchange is not an allowed US exchange: "
            f"{exchange_name or 'missing'}."
        )
    timezone = str(meta.get("exchangeTimezoneName", "")).strip()
    if timezone != US_EXCHANGE_TIMEZONE:
        raise ValueError(
            "Yahoo chart exchange timezone must be America/New_York, got "
            f"{timezone or 'missing'}."
        )
    return symbol, currency


class YahooChartCache:
    """One-attempt Yahoo chart fetcher with an immutable raw gzip cache."""

    def __init__(
        self,
        root: Path,
        *,
        endpoint_template: str = DEFAULT_ENDPOINT_TEMPLATE,
        max_http_attempts: int = 400,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 50 * 1024 * 1024,
    ):
        _validate_endpoint_template(endpoint_template)
        if max_http_attempts <= 0:
            raise ValueError("Yahoo chart HTTP attempt cap must be positive.")
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("Yahoo chart timeout/response cap must be positive.")
        self.root = Path(root)
        self.endpoint_template = endpoint_template
        self.max_http_attempts = int(max_http_attempts)
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.http_attempts = 0
        self._attempted_urls: set[str] = set()

    @staticmethod
    def _request_bounds(period1: int, period2: int) -> tuple[int, int]:
        if isinstance(period1, bool) or isinstance(period2, bool):
            raise ValueError("Yahoo chart request bounds must be integer epochs.")
        try:
            start, end = int(period1), int(period2)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Yahoo chart request bounds must be integer epochs."
            ) from exc
        if start <= 0 or end <= start or end >= 4_102_444_800:
            raise ValueError("Yahoo chart request bounds are invalid.")
        return start, end

    def url(self, symbol: str, *, period1: int, period2: int) -> str:
        normalized = normalize_yahoo_symbol(symbol)
        start, end = self._request_bounds(period1, period2)
        endpoint = self.endpoint_template.format(symbol=quote(normalized, safe="-"))
        query = (
            ("period1", str(start)),
            ("period2", str(end)),
            *_BOUNDED_QUERY,
        )
        return endpoint + "?" + urlencode(query)

    def path(self, symbol: str, *, period1: int, period2: int) -> Path:
        source_url = self.url(symbol, period1=period1, period2=period2)
        return self.root / f"{sha256_bytes(source_url.encode())}.json.gz"

    def _decode(
        self,
        symbol: str,
        encoded: bytes,
        *,
        period1: int,
        period2: int,
    ) -> YahooChartCachedResponse:
        normalized = normalize_yahoo_symbol(symbol)
        start, end = self._request_bounds(period1, period2)
        source_url = self.url(normalized, period1=start, period2=end)
        path = self.path(normalized, period1=start, period2=end)
        try:
            decoded = gzip.decompress(encoded)
            envelope = json.loads(decoded)
        except Exception as exc:
            raise RuntimeError(f"Invalid Yahoo chart cache envelope: {path}") from exc
        if not isinstance(envelope, dict):
            raise RuntimeError("Yahoo chart cache envelope must be an object.")
        wrapper_hash = str(envelope.get("wrapper_sha256", ""))
        unhashed = dict(envelope)
        unhashed.pop("wrapper_sha256", None)
        if wrapper_hash != sha256_bytes(_canonical_json_bytes(unhashed)):
            raise RuntimeError("Yahoo chart cache wrapper hash mismatch.")
        if decoded != _canonical_json_bytes(envelope):
            raise RuntimeError("Yahoo chart cache wrapper is not canonical.")
        try:
            content = base64.b64decode(envelope["content_base64"], validate=True)
        except Exception as exc:
            raise RuntimeError("Yahoo chart cache content encoding is invalid.") from exc
        if envelope.get("schema") != CACHE_SCHEMA:
            raise RuntimeError("Wrong Yahoo chart cache schema.")
        if envelope.get("symbol") != normalized:
            raise RuntimeError("Yahoo chart cache symbol mismatch.")
        if envelope.get("request_period1") != start:
            raise RuntimeError("Yahoo chart cache period1 mismatch.")
        if envelope.get("request_period2") != end:
            raise RuntimeError("Yahoo chart cache period2 mismatch.")
        if envelope.get("source_url") != source_url:
            raise RuntimeError("Yahoo chart cache URL mismatch.")
        if envelope.get("content_sha256") != sha256_bytes(content):
            raise RuntimeError("Yahoo chart cache content hash mismatch.")
        try:
            http_status = int(envelope["http_status"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Yahoo chart cache HTTP status is invalid.") from exc
        return YahooChartCachedResponse(
            symbol=normalized,
            source_url=source_url,
            retrieved_at=str(envelope["retrieved_at"]),
            content=content,
            content_type=str(envelope.get("content_type") or "application/json"),
            http_status=http_status,
            wrapper_hash=wrapper_hash,
            request_period1=start,
            request_period2=end,
        )

    def get(
        self,
        symbol: str,
        *,
        period1: int,
        period2: int,
    ) -> YahooChartCachedResponse | None:
        path = self.path(symbol, period1=period1, period2=period2)
        return (
            self._decode(
                symbol,
                path.read_bytes(),
                period1=period1,
                period2=period2,
            )
            if path.is_file()
            else None
        )

    def provenance_payload(
        self,
        symbol: str,
        *,
        period1: int,
        period2: int,
    ) -> bytes | None:
        """Return the hash-bound request envelope without its hash field.

        The returned canonical bytes hash to ``wrapper_sha256`` and retain the
        exact requested URL, symbol, bounds, HTTP metadata, and raw response
        bytes.  Archiving these bytes makes request provenance independently
        reproducible after the local cache is removed.
        """

        response = self.get(symbol, period1=period1, period2=period2)
        if response is None:
            return None
        path = self.path(symbol, period1=period1, period2=period2)
        try:
            envelope = json.loads(gzip.decompress(path.read_bytes()))
        except Exception as exc:
            raise RuntimeError(
                f"Invalid Yahoo chart cache provenance envelope: {path}"
            ) from exc
        unhashed = dict(envelope)
        wrapper_hash = str(unhashed.pop("wrapper_sha256", ""))
        payload = _canonical_json_bytes(unhashed)
        if wrapper_hash != response.wrapper_hash or sha256_bytes(payload) != wrapper_hash:
            raise RuntimeError("Yahoo chart cache provenance hash mismatch.")
        return payload

    def fetch(
        self,
        symbol: str,
        *,
        period1: int,
        period2: int,
    ) -> YahooChartCachedResponse:
        normalized = normalize_yahoo_symbol(symbol)
        start, end = self._request_bounds(period1, period2)
        source_url = self.url(normalized, period1=start, period2=end)
        cached = self.get(normalized, period1=start, period2=end)
        if cached is not None:
            return cached
        if self.http_attempts >= self.max_http_attempts:
            raise RuntimeError("Yahoo chart HTTP attempt cap reached.")
        if source_url in self._attempted_urls:
            raise RuntimeError(
                "Yahoo chart request URL was already attempted in this run."
            )
        self._attempted_urls.add(source_url)
        self.http_attempts += 1
        request = Request(
            source_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "SuperTrendQuant personal cross-validation/2.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                content_type = str(
                    response.headers.get("Content-Type") or "application/json"
                )
                content = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            status = int(exc.code)
            headers = exc.headers or {}
            content_type = str(headers.get("Content-Type") or "application/json")
            content = exc.read(self.max_response_bytes + 1)
        except URLError as exc:
            raise RuntimeError(
                f"Yahoo chart single HTTP attempt failed for {normalized}: {exc.reason}"
            ) from None
        if len(content) > self.max_response_bytes:
            raise RuntimeError("Yahoo chart response exceeds configured byte cap.")
        unhashed = {
            "schema": CACHE_SCHEMA,
            "symbol": normalized,
            "request_period1": start,
            "request_period2": end,
            "source_url": source_url,
            "retrieved_at": utc_now_iso(),
            "http_status": status,
            "content_type": content_type,
            "content_sha256": sha256_bytes(content),
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
        envelope = {
            **unhashed,
            "wrapper_sha256": sha256_bytes(_canonical_json_bytes(unhashed)),
        }
        encoded = gzip.compress(_canonical_json_bytes(envelope), mtime=0)
        destination = self.path(normalized, period1=start, period2=end)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = self._decode(
                normalized,
                destination.read_bytes(),
                period1=start,
                period2=end,
            )
            if existing.content != content:
                raise RuntimeError("Yahoo chart cache changed for one request URL.")
            return existing
        write_atomic(destination, encoded)
        return self._decode(
            normalized,
            destination.read_bytes(),
            period1=start,
            period2=end,
        )

    def fill_missing(
        self, requests: Iterable[tuple[str, int, int]]
    ) -> dict[tuple[str, int, int], YahooChartCachedResponse]:
        ordered = tuple(
            dict.fromkeys(
                (
                    normalize_yahoo_symbol(symbol),
                    *self._request_bounds(period1, period2),
                )
                for symbol, period1, period2 in requests
            )
        )
        cached = {
            request: self.get(
                request[0], period1=request[1], period2=request[2]
            )
            for request in ordered
        }
        missing = [request for request, value in cached.items() if value is None]
        remaining = self.max_http_attempts - self.http_attempts
        if len(missing) > remaining:
            raise RuntimeError(
                "Yahoo chart request set exceeds run cap before any HTTP call: "
                f"{len(missing)} > {remaining}."
            )
        output: dict[tuple[str, int, int], YahooChartCachedResponse] = {}
        for request in ordered:
            output[request] = cached[request] or self.fetch(
                request[0], period1=request[1], period2=request[2]
            )
        return output


def _finite_number(value: Any, *, field: str, index: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Yahoo chart {field}[{index}] is not numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Yahoo chart {field}[{index}] is not numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Yahoo chart {field}[{index}] is not finite.")
    return parsed


def parse_yahoo_chart_json(content: bytes, expected_symbol: str) -> YahooChartData:
    """Parse only raw ``indicators.quote`` OHLCV from a Yahoo chart response."""

    if content.lstrip().startswith((b"<", b"<!")):
        raise ValueError("Yahoo chart response is HTML or a verification challenge.")
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Yahoo chart response is not valid JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("chart"), dict):
        raise ValueError("Yahoo chart response lacks the chart object.")
    chart = payload["chart"]
    if chart.get("error") is not None:
        raise ValueError("Yahoo chart API returned chart.error.")
    result = chart.get("result")
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise ValueError("Yahoo chart response must contain exactly one result.")
    item = result[0]
    symbol, currency = validate_yahoo_equity_metadata(
        item.get("meta"), expected_symbol
    )
    if str(item["meta"].get("dataGranularity", "")).strip() != "1d":
        raise ValueError("Yahoo chart dataGranularity must be exactly 1d.")

    timestamps = item.get("timestamp")
    indicators = item.get("indicators")
    if not isinstance(indicators, dict):
        raise ValueError("Yahoo chart result lacks indicators.")
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or len(quotes) != 1 or not isinstance(quotes[0], dict):
        raise ValueError("Yahoo chart result lacks one raw quote series.")
    quote_values = quotes[0]
    required = ("open", "high", "low", "close", "volume")
    if not all(isinstance(quote_values.get(field), list) for field in required):
        raise ValueError("Yahoo chart raw quote series lacks OHLCV arrays.")
    if timestamps in (None, []):
        if any(quote_values[field] for field in required):
            raise ValueError("Yahoo chart empty timestamps are misaligned with OHLCV.")
        return YahooChartData(
            symbol=symbol,
            currency=currency,
            bars=pd.DataFrame(columns=("session", *required)),
        )
    if not isinstance(timestamps, list):
        raise ValueError("Yahoo chart timestamps must be a list.")
    length = len(timestamps)
    if length == 0 or any(len(quote_values[field]) != length for field in required):
        raise ValueError("Yahoo chart timestamps and raw OHLCV arrays are misaligned.")

    epoch_values: list[int] = []
    for index, value in enumerate(timestamps):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Yahoo chart timestamp[{index}] is not an integer.")
        if value <= 0 or value >= 4_102_444_800:
            raise ValueError(f"Yahoo chart timestamp[{index}] is out of range.")
        epoch_values.append(value)
    if any(right <= left for left, right in zip(epoch_values, epoch_values[1:])):
        raise ValueError("Yahoo chart timestamps are not strictly increasing.")
    converted = pd.to_datetime(epoch_values, unit="s", utc=True, errors="coerce")
    if converted.isna().any():
        raise ValueError("Yahoo chart timestamps are invalid.")
    sessions = converted.tz_convert("America/New_York").normalize().tz_localize(None)
    if sessions.duplicated().any() or not sessions.is_monotonic_increasing:
        raise ValueError("Yahoo chart timestamps do not map to unique ordered sessions.")

    parsed: dict[str, list[float]] = {field: [] for field in required}
    for field in required:
        for index, value in enumerate(quote_values[field]):
            parsed[field].append(_finite_number(value, field=field, index=index))
    frame = pd.DataFrame({"session": sessions, **parsed})
    positive = frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    coherent = (
        frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "close"]].min(axis=1))
        & frame["high"].ge(frame["low"])
    )
    nonnegative_volume = frame["volume"].ge(0)
    if not bool((positive & coherent & nonnegative_volume).all()):
        raise ValueError("Yahoo chart raw quote series has invalid OHLCV bars.")
    return YahooChartData(symbol=symbol, currency=currency, bars=frame)


def _not_found_error(value: Any) -> tuple[bool, str, str]:
    if not isinstance(value, dict):
        return False, "", ""
    code = str(value.get("code", "")).strip()
    description = str(value.get("description", "")).strip()
    text = f"{code} {description}".lower()
    accepted = any(
        marker in text
        for marker in ("not found", "no data", "may be delisted", "delisted")
    )
    return accepted, code, description


def parse_yahoo_chart_no_data_evidence(
    content: bytes,
    expected_symbol: str,
    *,
    http_status: int,
    request_period1: int | None = None,
    request_period2: int | None = None,
) -> YahooChartNoDataEvidence:
    """Validate immutable Yahoo no-data bytes without pretending they are prices.

    Yahoo commonly returns a JSON ``chart.error`` with HTTP 404 for delisted
    tickers.  Two other exact no-price shapes are recognized: an HTTP-200 YHD
    retired-symbol placeholder with no timestamp or quote values, and an
    HTTP-400 bounded-history error whose echoed epochs exactly match the
    request.  The caller must still bind this evidence to the exact cached
    request URL and an independently verified official terminal event.
    """

    expected = normalize_yahoo_symbol(expected_symbol)
    if content.lstrip().startswith((b"<", b"<!")):
        raise ValueError("Yahoo chart no-data evidence is HTML or a challenge.")
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Yahoo chart no-data evidence is not valid JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("chart"), dict):
        raise ValueError("Yahoo chart no-data evidence lacks the chart object.")
    chart = payload["chart"]
    status = int(http_status)
    if status == 200 and chart.get("error") is None:
        result = chart.get("result")
        item = (
            result[0]
            if isinstance(result, list)
            and len(result) == 1
            and isinstance(result[0], dict)
            else None
        )
        meta = item.get("meta") if isinstance(item, dict) else None
        indicators = item.get("indicators") if isinstance(item, dict) else None
        retired_placeholder = (
            set(chart) == {"result", "error"}
            and isinstance(meta, dict)
            and normalize_yahoo_symbol(str(meta.get("symbol", ""))) == expected
            and "currency" in meta
            and meta.get("currency") is None
            and str(meta.get("instrumentType", "")).strip().upper()
            == "MUTUALFUND"
            and str(meta.get("exchangeName", "")).strip().upper() == "YHD"
            and str(meta.get("fullExchangeName", "")).strip().upper() == "YHD"
            and str(meta.get("exchangeTimezoneName", "")).strip()
            == US_EXCHANGE_TIMEZONE
            and str(meta.get("dataGranularity", "")).strip() == "1d"
            and str(meta.get("range", "")) == ""
            and set(item).issubset({"meta", "timestamp", "indicators"})
            and item.get("timestamp") in (None, [])
            and isinstance(indicators, dict)
            and indicators.get("quote") == [{}]
            and indicators.get("adjclose") == [{}]
            and set(indicators) == {"quote", "adjclose"}
        )
        if retired_placeholder:
            return YahooChartNoDataEvidence(
                symbol=expected,
                kind="http_200_empty_retired_yhd_placeholder",
                http_status=status,
            )
        parsed = parse_yahoo_chart_json(content, expected)
        if not parsed.bars.empty:
            raise ValueError("Yahoo chart no-data evidence contains price bars.")
        return YahooChartNoDataEvidence(
            symbol=expected,
            kind="http_200_empty_equity_chart",
            http_status=status,
        )
    if status == 400:
        if (
            isinstance(request_period1, bool)
            or isinstance(request_period2, bool)
            or not isinstance(request_period1, int)
            or not isinstance(request_period2, int)
            or request_period1 <= 0
            or request_period2 <= request_period1
            or request_period2 >= 4_102_444_800
        ):
            raise ValueError(
                "Yahoo bounded-history no-data evidence lacks request epochs."
            )
        period1 = request_period1
        period2 = request_period2
        expected_description = (
            f"Data doesn't exist for startDate = {period1}, endDate = {period2}"
        )
        error = chart.get("error")
        if not (
            set(chart) == {"result", "error"}
            and chart.get("result") is None
            and isinstance(error, dict)
            and set(error) == {"code", "description"}
            and error.get("code") == "Bad Request"
            and error.get("description") == expected_description
        ):
            raise ValueError(
                "Yahoo HTTP 400 response is not exact bounded-history no-data evidence."
            )
        return YahooChartNoDataEvidence(
            symbol=expected,
            kind="http_400_bounded_history_not_found",
            http_status=status,
            error_code="Bad Request",
            error_description=expected_description,
        )
    accepted, code, description = _not_found_error(chart.get("error"))
    if status not in {200, 404, 410} or not accepted:
        raise ValueError("Yahoo chart response is not recognized no-data evidence.")
    if chart.get("result") not in (None, []):
        raise ValueError("Yahoo chart not-found evidence unexpectedly contains a result.")
    return YahooChartNoDataEvidence(
        symbol=expected,
        kind="chart_not_found",
        http_status=status,
        error_code=code,
        error_description=description,
    )
