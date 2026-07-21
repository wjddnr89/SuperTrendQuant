"""Small reusable Stooq historical-CSV client with immutable raw caching.

There are deliberately no retries.  Callers derive the complete symbol set,
then ``fill_missing`` proves it fits under one run-wide cap before the first
request.  The same interface is shared by lifecycle cross-validation and
identity-repair tools.
"""

from __future__ import annotations

import base64
import csv
import gzip
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .manifest import sha256_bytes, utc_now_iso, write_atomic


DEFAULT_ENDPOINT_TEMPLATE = "https://stooq.com/q/d/l/?s={symbol}&i=d"


def normalize_us_symbol(symbol: str) -> str:
    value = str(symbol).strip().lower().replace(".", "-").replace("/", "-")
    if value.endswith("-us"):
        value = value[:-3]
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in value):
        raise ValueError(f"Invalid Stooq US symbol: {symbol!r}")
    return f"{value}.us"


@dataclass(frozen=True)
class StooqCachedResponse:
    symbol: str
    source_url: str
    retrieved_at: str
    content: bytes
    content_type: str
    http_status: int

    @property
    def source_hash(self) -> str:
        return sha256_bytes(self.content)


def _canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


class StooqHistoricalCache:
    """One-attempt keyless fetcher whose cache preserves exact response bytes."""

    def __init__(
        self,
        root: Path,
        *,
        endpoint_template: str = DEFAULT_ENDPOINT_TEMPLATE,
        max_http_attempts: int = 400,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 50 * 1024 * 1024,
    ):
        if not endpoint_template.startswith("https://stooq.com/") or "{symbol}" not in endpoint_template:
            raise ValueError("Stooq endpoint template is invalid.")
        if max_http_attempts <= 0:
            raise ValueError("Stooq HTTP attempt cap must be positive.")
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("Stooq timeout/response cap must be positive.")
        self.root = Path(root)
        self.endpoint_template = endpoint_template
        self.max_http_attempts = int(max_http_attempts)
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.http_attempts = 0

    def url(self, symbol: str) -> str:
        return self.endpoint_template.format(symbol=normalize_us_symbol(symbol))

    def path(self, symbol: str) -> Path:
        return self.root / f"{sha256_bytes(self.url(symbol).encode())}.json.gz"

    def _decode(self, symbol: str, encoded: bytes) -> StooqCachedResponse:
        normalized = normalize_us_symbol(symbol)
        try:
            envelope = json.loads(gzip.decompress(encoded))
            content = base64.b64decode(envelope["content_base64"], validate=True)
        except Exception as exc:
            raise RuntimeError(f"Invalid Stooq cache envelope: {self.path(symbol)}") from exc
        if envelope.get("schema") != "stooq_raw_response/v1":
            raise RuntimeError("Wrong Stooq cache schema.")
        if envelope.get("symbol") != normalized:
            raise RuntimeError("Stooq cache symbol mismatch.")
        if envelope.get("source_url") != self.url(symbol):
            raise RuntimeError("Stooq cache URL mismatch.")
        if envelope.get("source_hash") != sha256_bytes(content):
            raise RuntimeError("Stooq cache content hash mismatch.")
        return StooqCachedResponse(
            symbol=normalized,
            source_url=self.url(symbol),
            retrieved_at=str(envelope["retrieved_at"]),
            content=content,
            content_type=str(envelope.get("content_type") or "text/csv"),
            http_status=int(envelope["http_status"]),
        )

    def get(self, symbol: str) -> StooqCachedResponse | None:
        path = self.path(symbol)
        return self._decode(symbol, path.read_bytes()) if path.is_file() else None

    def fetch(self, symbol: str) -> StooqCachedResponse:
        if self.http_attempts >= self.max_http_attempts:
            raise RuntimeError("Stooq HTTP attempt cap reached.")
        self.http_attempts += 1
        normalized = normalize_us_symbol(symbol)
        request = Request(
            self.url(normalized),
            headers={"User-Agent": "SuperTrendQuant cross-validation/1.0"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                content_type = str(response.headers.get("Content-Type") or "text/csv")
                content = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            status = int(exc.code)
            content_type = str(exc.headers.get("Content-Type") or "text/plain")
            content = exc.read(self.max_response_bytes + 1)
        except URLError as exc:
            raise RuntimeError(
                f"Stooq single HTTP attempt failed for {normalized}: {exc.reason}"
            ) from None
        if len(content) > self.max_response_bytes:
            raise RuntimeError("Stooq response exceeds configured byte cap.")
        value = StooqCachedResponse(
            symbol=normalized,
            source_url=self.url(normalized),
            retrieved_at=utc_now_iso(),
            content=content,
            content_type=content_type,
            http_status=status,
        )
        envelope = {
            "schema": "stooq_raw_response/v1",
            "symbol": value.symbol,
            "source_url": value.source_url,
            "retrieved_at": value.retrieved_at,
            "http_status": value.http_status,
            "content_type": value.content_type,
            "source_hash": value.source_hash,
            "content_base64": base64.b64encode(value.content).decode("ascii"),
        }
        destination = self.path(normalized)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = self._decode(normalized, destination.read_bytes())
            if existing.content != content:
                raise RuntimeError("Stooq cache changed for one request URL.")
            return existing
        write_atomic(destination, gzip.compress(_canonical_json_bytes(envelope), mtime=0))
        return self._decode(normalized, destination.read_bytes())

    def fill_missing(self, symbols: Iterable[str]) -> dict[str, StooqCachedResponse]:
        ordered = tuple(dict.fromkeys(normalize_us_symbol(item) for item in symbols))
        missing = [symbol for symbol in ordered if self.get(symbol) is None]
        if len(missing) > self.max_http_attempts:
            raise RuntimeError(
                "Stooq request set exceeds run cap before any HTTP call: "
                f"{len(missing)} > {self.max_http_attempts}."
            )
        return {
            symbol: self.get(symbol) or self.fetch(symbol)
            for symbol in ordered
        }


def parse_stooq_daily_csv(content: bytes) -> pd.DataFrame | None:
    """Strict parser; HTML/challenge pages are never treated as no-data."""

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Stooq response is not UTF-8 CSV.") from exc
    stripped = text.strip()
    if not stripped or stripped.lower() in {"no data", "no data."}:
        return None
    reader = csv.DictReader(io.StringIO(text))
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("Stooq response lacks required OHLCV columns.")
    rows = list(reader)
    if not rows:
        return None
    frame = pd.DataFrame(rows).rename(
        columns={
            "Date": "session",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    frame["session"] = pd.to_datetime(frame["session"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any():
        raise ValueError("Stooq CSV contains invalid values.")
    if frame["session"].duplicated().any():
        raise ValueError("Stooq CSV has duplicate sessions.")
    positive = frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    coherent = frame["high"].ge(frame[["open", "close"]].max(axis=1)) & frame[
        "low"
    ].le(frame[["open", "close"]].min(axis=1))
    if not bool((positive & coherent).all()):
        raise ValueError("Stooq CSV has invalid OHLC bars.")
    return frame.sort_values("session").reset_index(drop=True)
