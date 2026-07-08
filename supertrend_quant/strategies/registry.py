from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Strategy

if TYPE_CHECKING:
    from ..config import AppConfig


_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(strategy_cls: type[Strategy]) -> type[Strategy]:
    strategy_type = str(getattr(strategy_cls, "strategy_type", "")).strip()
    if not strategy_type:
        raise ValueError("Strategy classes must define a non-empty strategy_type.")
    if strategy_type in _REGISTRY:
        raise ValueError(f"Strategy type already registered: {strategy_type}")
    _REGISTRY[strategy_type] = strategy_cls
    return strategy_cls


def available_strategies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_strategy_class(strategy_type: str) -> type[Strategy]:
    try:
        return _REGISTRY[strategy_type]
    except KeyError as exc:
        available = ", ".join(available_strategies()) or "<none>"
        raise ValueError(f"Unsupported strategy type: {strategy_type}. Available strategies: {available}") from exc


def create_strategy(config: AppConfig) -> Strategy:
    strategy_cls = get_strategy_class(config.strategy.type)
    strategy_cls.validate_config(config)
    return strategy_cls(config)
