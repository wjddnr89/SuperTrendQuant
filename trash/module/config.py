# -*- coding: utf-8 -*-
"""
Configuration objects for the modular SuperTrend research backtest.

The module keeps its fast research-oriented StrategyConfig, but the names and
serialization helpers follow the supertrend_quant runtime shape:
signals.entries, signals.filters, signals.exits, and rotation.  That makes a
good module result easy to move into SuperTrendQuant/configs/strategies later.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from typing import Optional, Tuple

import yaml


DEFAULT_FEE_RATE = 0.00225
DEFAULT_SLIPPAGE_RATE = 0.0005

BASE_INTRADAY_INTERVAL = "30m"
DAILY_INTERVAL = "1d"

TIMEFRAMES = {
    "30m": {"source": "intraday", "rule": None, "interval": "30m", "scale": 1},
    "1h": {"source": "intraday", "rule": "1h", "interval": "1h", "scale": 2},
    "2h": {"source": "intraday", "rule": "2h", "interval": "2h", "scale": 4},
    "4h": {"source": "intraday", "rule": "4h", "interval": "4h", "scale": 8},
    "1d": {"source": "daily", "rule": None, "interval": "1d", "scale": 13},
}

VALID_SIGNALS = {"supertrend", "single_supertrend", "triple_supertrend"}
VALID_SELECTORS = {"leader_top1", "none"}
VALID_ASSET_FILTERS = {"none", "ichimoku_cloud", "ema_trend"}


@dataclass(frozen=True)
class StrategyConfig:
    market: str = "us"
    timeframe: str = "30m"
    period: str = "60d"
    initial_cash: float = 10_000.0

    signal: str = "supertrend"
    selector: str = "leader_top1"
    asset_filter: str = "none"
    market_filter: str = "none"

    rs_base_bars: int = 100
    rs_period: Optional[int] = None
    sell_confirm_bars: int = 1
    allow_late_chase: bool = True
    rotation_enabled: bool = True
    hurdle_atr_mult: float = 1.25

    st_period: int = 10
    st_multiplier: float = 3.0
    atr_method: str = "wilder"

    triple_settings: Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float]] = (
        (10, 1.0),
        (11, 2.0),
        (12, 3.0),
    )
    triple_exit_down_count: int = 2

    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_span_b: int = 52
    ichimoku_shift: int = 26
    ema_period: int = 200

    fee_rate: float = DEFAULT_FEE_RATE
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE
    min_coverage: float = 0.8
    allocation_pct: float = 1.0
    min_rotation_profit_pct: float = 0.0

    train_ratio: float = 0.6
    validation_ratio: float = 0.2

    def with_updates(self, **kwargs):
        return replace(self, **kwargs)


def validate_config(config: StrategyConfig) -> None:
    if config.market not in {"us", "kr"}:
        raise ValueError("market must be 'us' or 'kr'.")
    if config.timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {config.timeframe}")
    if normalize_signal(config.signal) not in VALID_SIGNALS:
        raise ValueError(f"Unsupported signal module: {config.signal}")
    if config.selector not in VALID_SELECTORS:
        raise ValueError(f"Unsupported selector: {config.selector}")
    for asset_filter in asset_filter_list(config.asset_filter):
        if asset_filter not in VALID_ASSET_FILTERS:
            raise ValueError(f"Unsupported asset filter: {asset_filter}")
    if config.sell_confirm_bars < 1:
        raise ValueError("sell_confirm_bars must be at least 1.")
    if config.rs_base_bars < 1:
        raise ValueError("rs_base_bars must be at least 1.")
    if config.rs_period is not None and config.rs_period < 1:
        raise ValueError("rs_period must be at least 1 when provided.")
    if config.train_ratio <= 0 or config.validation_ratio < 0:
        raise ValueError("train_ratio must be positive and validation_ratio cannot be negative.")
    if config.train_ratio + config.validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1.0.")


def resolve_rs_period(config: StrategyConfig) -> int:
    if config.rs_period is not None:
        return config.rs_period
    scale = TIMEFRAMES[config.timeframe]["scale"]
    return max(2, -(-config.rs_base_bars // scale))


def normalize_signal(signal: str) -> str:
    if signal == "single_supertrend":
        return "supertrend"
    return signal


def asset_filter_list(asset_filter: str) -> Tuple[str, ...]:
    if asset_filter in {None, "", "none"}:
        return ("none",)
    filters = tuple(item.strip() for item in str(asset_filter).split("+") if item.strip())
    return filters or ("none",)


def market_filter_timeframe(market_filter: str) -> Optional[str]:
    if market_filter in {"none", "", None}:
        return None

    if market_filter.startswith("qqq_"):
        tf = market_filter.replace("qqq_", "", 1)
    elif market_filter.startswith("market_"):
        tf = market_filter.replace("market_", "", 1)
    elif market_filter in TIMEFRAMES:
        tf = market_filter
    else:
        raise ValueError(
            "market_filter must be none, market_<timeframe>, qqq_<timeframe>, or a timeframe."
        )

    if tf not in TIMEFRAMES:
        raise ValueError(f"Unsupported market filter timeframe: {tf}")
    return tf


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return loaded


def config_from_strategy_runtime(
    strategy_path: str | Path,
    runtime_path: str | Path,
) -> StrategyConfig:
    strategy_raw = load_yaml(strategy_path)
    runtime_raw = load_yaml(runtime_path)
    return config_from_supertrend_quant_dicts(strategy_raw, runtime_raw)


def config_from_supertrend_quant_dicts(
    strategy_raw: dict[str, Any],
    runtime_raw: dict[str, Any],
) -> StrategyConfig:
    signals = _mapping(strategy_raw, "signals")
    entries = _component_group(signals, "entries")
    filters = _component_group(signals, "filters")
    exits = _component_group(signals, "exits")
    portfolio = _mapping(strategy_raw, "portfolio")
    rotation = _mapping(strategy_raw, "rotation")
    data = _mapping(runtime_raw, "data")
    capital = _mapping(runtime_raw, "capital")
    costs = _mapping(runtime_raw, "costs")

    entry = _first_enabled(entries, {"supertrend", "triple_supertrend"}) or {"type": "supertrend"}
    signal = "triple_supertrend" if entry.get("type") == "triple_supertrend" else "supertrend"

    st_period = int(entry.get("period", 10))
    st_multiplier = float(entry.get("multiplier", 3.0))
    atr_method = str(entry.get("atr_method", "wilder"))
    triple_settings = StrategyConfig.triple_settings
    if signal == "triple_supertrend":
        parsed = []
        settings = entry.get("settings", ())
        if isinstance(settings, list):
            for item in settings:
                if isinstance(item, dict):
                    parsed.append((int(item.get("period", 10)), float(item.get("multiplier", 3.0))))
        if len(parsed) == 3:
            triple_settings = tuple(parsed)  # type: ignore[assignment]

    benchmark_filter = _first_enabled(filters, {"benchmark_trend", "market_trend"})
    market_filter = (
        f"market_{benchmark_filter.get('timeframe', '1d')}"
        if benchmark_filter and benchmark_filter.get("enabled", True)
        else "none"
    )

    asset_filters = []
    if _first_enabled(filters, {"ichimoku_cloud"}):
        asset_filters.append("ichimoku_cloud")
    ema_filter = _first_enabled(filters, {"ema_trend"})
    if ema_filter:
        asset_filters.append("ema_trend")
    asset_filter = "+".join(asset_filters) if asset_filters else "none"

    relative_strength = _first_enabled(filters, {"relative_strength"})
    rs_period = None
    if relative_strength:
        lookback = relative_strength.get("lookback_bars", 100)
        if isinstance(lookback, dict):
            rs_period = int(lookback.get("default", 100))
        else:
            rs_period = int(lookback)

    exit_component = _first_enabled(exits, {"supertrend_flip", "triple_supertrend_flip"}) or {}
    sell_confirm_bars = int(exit_component.get("confirm_bars", 1))
    triple_exit_down_count = int(exit_component.get("down_count", 2))

    hurdle = rotation.get("hurdle", {})
    if not isinstance(hurdle, dict):
        hurdle = {}

    market = str(runtime_raw.get("market", "US")).lower()
    if market == "auto":
        market = "us"

    return StrategyConfig(
        market=market,
        timeframe=str(data.get("timeframe", runtime_raw.get("timeframe", "30m"))),
        period=str(data.get("period", runtime_raw.get("period", "60d"))),
        initial_cash=float(capital.get("initial_cash", runtime_raw.get("initial_cash", 10_000.0))),
        signal=signal,
        selector="leader_top1" if strategy_raw.get("type", "leader_rotation") == "leader_rotation" else "none",
        asset_filter=asset_filter,
        market_filter=market_filter,
        rs_period=rs_period,
        sell_confirm_bars=sell_confirm_bars,
        allow_late_chase=bool(rotation.get("allow_late_chase", True)),
        hurdle_atr_mult=float(hurdle.get("multiplier", 1.25)),
        st_period=st_period,
        st_multiplier=st_multiplier,
        atr_method=atr_method,
        triple_settings=triple_settings,
        triple_exit_down_count=triple_exit_down_count,
        ema_period=int(ema_filter.get("period", 200)) if ema_filter else 200,
        fee_rate=float(costs.get("fee_rate", DEFAULT_FEE_RATE)),
        slippage_rate=float(costs.get("slippage_rate", DEFAULT_SLIPPAGE_RATE)),
        allocation_pct=float(portfolio.get("allocation_pct", 1.0)),
        min_rotation_profit_pct=float(rotation.get("min_rotation_profit_pct", 0.0)),
    )


def strategy_dict_from_config(config: StrategyConfig, name: str = None) -> dict[str, Any]:
    signal = normalize_signal(config.signal)
    entry: dict[str, Any] = {
        "type": signal,
        "enabled": True,
        "atr_method": config.atr_method,
    }
    if signal == "triple_supertrend":
        entry["settings"] = [
            {"period": period, "multiplier": multiplier}
            for period, multiplier in config.triple_settings
        ]
    else:
        entry["period"] = config.st_period
        entry["multiplier"] = config.st_multiplier

    filters: list[dict[str, Any]] = []
    tf = market_filter_timeframe(config.market_filter)
    filters.append(
        {
            "type": "benchmark_trend",
            "enabled": tf is not None,
            "timeframe": tf or "1d",
        }
    )
    filters.append({"type": "relative_strength", "lookback_bars": resolve_rs_period(config)})
    for asset_filter in asset_filter_list(config.asset_filter):
        if asset_filter == "ichimoku_cloud":
            filters.append(
                {
                    "type": "ichimoku_cloud",
                    "enabled": True,
                    "tenkan": config.ichimoku_tenkan,
                    "kijun": config.ichimoku_kijun,
                    "span_b": config.ichimoku_span_b,
                    "shift": config.ichimoku_shift,
                }
            )
        elif asset_filter == "ema_trend":
            filters.append(
                {
                    "type": "ema_trend",
                    "enabled": True,
                    "period": config.ema_period,
                }
            )

    exit_type = "triple_supertrend_flip" if signal == "triple_supertrend" else "supertrend_flip"
    exit_component: dict[str, Any] = {
        "type": exit_type,
        "confirm_bars": config.sell_confirm_bars,
    }
    if exit_type == "triple_supertrend_flip":
        exit_component["down_count"] = config.triple_exit_down_count

    return {
        "name": name or f"module_{config.market}_{config.timeframe}_{signal}",
        "type": "leader_rotation" if config.selector == "leader_top1" else "simple_supertrend",
        "portfolio": {
            "max_positions": 1 if config.selector == "leader_top1" else 3,
            "allocation_pct": config.allocation_pct,
        },
        "signals": {
            "entries": [entry],
            "filters": filters,
            "exits": [exit_component],
        },
        "rotation": {
            "hurdle": {"multiplier": config.hurdle_atr_mult},
            "allow_late_chase": config.allow_late_chase,
            "min_rotation_profit_pct": config.min_rotation_profit_pct,
        },
    }


def runtime_dict_from_config(config: StrategyConfig, universe_file: str = "universe.json") -> dict[str, Any]:
    return {
        "name": f"module_research_{config.market}",
        "market": config.market.upper(),
        "universe_file": universe_file,
        "symbols": [],
        "data": {
            "timeframe": config.timeframe,
            "period": config.period,
        },
        "capital": {
            "initial_cash": config.initial_cash,
        },
        "costs": {
            "fee_rate": config.fee_rate,
            "slippage_rate": config.slippage_rate,
        },
        "execution": {
            "order_type": "market",
            "broker": "paper",
            "live_confirm_required": True,
        },
    }


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    return value if isinstance(value, dict) else {}


def _component_group(signals: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = signals.get(key, [])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _first_enabled(group: list[dict[str, Any]], types: set[str]) -> Optional[dict[str, Any]]:
    for item in group:
        if item.get("type") in types and bool(item.get("enabled", True)):
            return item
    return None
