from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class RealtimeQuote:
    symbol: str
    price: float
    observed_at: datetime
    source: str


class QuoteProvider(Protocol):
    def quotes(self, symbols: list[str]) -> dict[str, RealtimeQuote]:
        ...


class RealtimeQuoteProvider(QuoteProvider, Protocol):
    pass


class BarSource(Protocol):
    """Reserved V2 seam; V1 implementations must not persist per-candle R2 data."""

    def load(self, symbols: list[str], timeframe: str, period: str) -> dict[str, pd.DataFrame]:
        ...


class IntradayHistoryProvider(BarSource, Protocol):
    pass


class TossRealtimeQuoteProvider:
    def __init__(self, broker):
        self.broker = broker

    def quotes(self, symbols: list[str]) -> dict[str, RealtimeQuote]:
        observed_at = datetime.now(UTC)
        return {
            symbol: RealtimeQuote(symbol, float(price), observed_at, "toss")
            for symbol, price in self.broker.get_prices(symbols).items()
            if float(price) > 0
        }


class FrameQuoteProvider:
    """Paper quote seam backed by the latest completed raw frame."""

    def __init__(self, frames: dict[str, pd.DataFrame], *, source: str = "completed_raw_bar"):
        self.frames = frames
        self.source = source

    def quotes(self, symbols: list[str]) -> dict[str, RealtimeQuote]:
        observed_at = datetime.now(UTC)
        output: dict[str, RealtimeQuote] = {}
        for symbol in symbols:
            frame = self.frames.get(symbol)
            if frame is None or frame.empty or "Close" not in frame:
                continue
            price = float(frame["Close"].iloc[-1])
            if price > 0:
                output[symbol] = RealtimeQuote(symbol, price, observed_at, self.source)
        return output


class RealtimeBarOverlay:
    """Ephemeral current-candle overlay; intentionally has no disk/R2 methods."""

    def __init__(self):
        self._frames: dict[str, pd.DataFrame] = {}

    def replace(self, symbol: str, frame: pd.DataFrame) -> None:
        self._frames[str(symbol)] = frame.copy()

    def merge(self, historical: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        output = {symbol: frame.copy() for symbol, frame in historical.items()}
        for symbol, overlay in self._frames.items():
            if symbol in output:
                output[symbol] = pd.concat([output[symbol], overlay]).sort_index()
                output[symbol] = output[symbol].loc[~output[symbol].index.duplicated(keep="last")]
            else:
                output[symbol] = overlay.copy()
        return output

    def clear(self) -> None:
        self._frames.clear()


# Backward-compatible name used by the V1 preview implementation.
InMemoryCandleOverlay = RealtimeBarOverlay
