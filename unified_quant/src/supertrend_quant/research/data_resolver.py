from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeAlias

from ..config import AppConfig, load_universe
from ..data import MarketData, download_market_data


MarketDataLoader: TypeAlias = Callable[[AppConfig], MarketData]
MarketDataSource: TypeAlias = MarketData | MarketDataLoader


class MarketDataMismatchError(ValueError):
    """Raised when a fixed data bundle is reused for a different data request."""


def data_request_key(config: AppConfig) -> tuple[object, ...]:
    """Return the fields that determine canonical historical data downloads."""

    return (
        config.market,
        config.timeframe,
        config.period,
        config.universe_file,
        tuple(config.symbols),
        bool(config.market_trend_filter.enabled),
        config.market_trend_filter.timeframe,
    )


def download_for_config(config: AppConfig) -> MarketData:
    return download_market_data(config, load_universe(config))


def fixed_data_supports(fixed_config: AppConfig, candidate: AppConfig) -> bool:
    fixed_base = data_request_key(fixed_config)[:5]
    candidate_base = data_request_key(candidate)[:5]
    if fixed_base != candidate_base:
        return False
    if not candidate.market_trend_filter.enabled:
        return True
    if candidate.market_trend_filter.timeframe == candidate.timeframe:
        return True
    return bool(
        fixed_config.market_trend_filter.enabled
        and fixed_config.market_trend_filter.timeframe
        == candidate.market_trend_filter.timeframe
    )


@dataclass
class MarketDataCache:
    """Lazy config-to-MarketData resolver for multi-timeframe research."""

    loader: MarketDataLoader = download_for_config
    _cache: dict[tuple[object, ...], MarketData] = field(default_factory=dict, init=False)

    def __call__(self, config: AppConfig) -> MarketData:
        key = data_request_key(config)
        if key not in self._cache:
            self._cache[key] = self.loader(config)
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    @property
    def cached_requests(self) -> tuple[tuple[object, ...], ...]:
        return tuple(self._cache)


def resolve_market_data(
    source: MarketDataSource,
    config: AppConfig,
    *,
    fixed_config: AppConfig | None = None,
) -> MarketData:
    """Resolve data and prevent a fixed bundle from being mislabeled."""

    if isinstance(source, MarketData):
        if fixed_config is not None and not fixed_data_supports(fixed_config, config):
            raise MarketDataMismatchError(
                "This search received one fixed MarketData bundle, but an overlay changes "
                "market/timeframe/period/universe/filter data requirements. Pass "
                "MarketDataCache (or another config -> MarketData callable) for such a grid."
            )
        return source
    if not callable(source):
        raise TypeError("market_data must be MarketData or a config -> MarketData callable.")
    resolved = source(config)
    if not isinstance(resolved, MarketData):
        raise TypeError("MarketData resolver must return supertrend_quant.data.MarketData.")
    return resolved


__all__ = [
    "MarketDataCache",
    "MarketDataLoader",
    "MarketDataMismatchError",
    "MarketDataSource",
    "data_request_key",
    "download_for_config",
    "fixed_data_supports",
    "resolve_market_data",
]
