from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeAlias

from ..config import AppConfig
from ..data import MarketData
from ..market_store.provider import ensure_configured_data_ready, load_configured_market_data
from ..universe import resolve_universe, universe_request_key


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
        universe_request_key(config),
        tuple(config.symbols),
        bool(config.market_trend_filter.enabled),
        config.market_trend_filter.timeframe,
        config.data_store.provider,
        config.data_store.price_mode,
        config.data_store.local_cache_dir,
        config.data_store.index_source_mode,
    )


def download_for_config(config: AppConfig) -> MarketData:
    ensure_configured_data_ready(config)
    resolved = resolve_universe(config, mode="research")
    return load_configured_market_data(
        config,
        list(resolved.eligible_symbols),
        resolved_universe=resolved,
    )


def fixed_data_supports(fixed_config: AppConfig, candidate: AppConfig) -> bool:
    fixed_key = data_request_key(fixed_config)
    candidate_key = data_request_key(candidate)
    fixed_base = (*fixed_key[:5], *fixed_key[7:])
    candidate_base = (*candidate_key[:5], *candidate_key[7:])
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
