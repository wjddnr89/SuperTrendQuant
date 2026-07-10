from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT = _PACKAGE_PROJECT_ROOT.parent


@dataclass(frozen=True)
class SupertrendConfig:
    enabled: bool = True
    period: int = 10
    multiplier: float = 3.0
    atr_method: str = "wilder"
    symbol_multipliers: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketTrendFilterConfig:
    enabled: bool = False
    timeframe: str = "1d"


@dataclass(frozen=True)
class LeaderRotationConfig:
    rs_period: int = 100
    rs_period_by_market: dict[str, int] = field(default_factory=dict)
    max_slots: int = 1
    hurdle_atr_mult: float = 1.25
    allow_late_chase: bool = True
    min_rotation_profit_pct: float = 0.01


@dataclass(frozen=True)
class ExitConfig:
    sell_confirm_bars: int = 1


@dataclass(frozen=True)
class CostsConfig:
    fee_rate: float = 0.00225
    slippage_rate: float = 0.0005


@dataclass(frozen=True)
class CapitalConfig:
    initial_cash: float = 10_000.0


@dataclass(frozen=True)
class ExecutionConfig:
    order_type: str = "market"
    allocation_pct: float = 0.9
    broker: str = "paper"
    live_confirm_required: bool = True


@dataclass(frozen=True)
class RiskConfig:
    max_position_count: int = 1


@dataclass(frozen=True)
class LiveConfig:
    holdings_file: str = "holding.json"
    loop_interval_seconds: int = 60


@dataclass(frozen=True)
class PaperConfig:
    state_file: str = "state/paper.json"
    results_dir: str = "results/paper"
    loop_interval_seconds: int = 60
    run_once_per_candle: bool = True


@dataclass(frozen=True)
class BacktestConfig:
    results_dir: str = "results/backtests"


@dataclass(frozen=True)
class ComponentConfig:
    type: str
    enabled: bool = True
    group: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyIdentity:
    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AppConfig:
    strategy: StrategyIdentity
    market: str = "US"
    universe_file: str = "universe.json"
    symbols: tuple[str, ...] = ()
    timeframe: str = "30m"
    period: str = "60d"
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    supertrend: SupertrendConfig = field(default_factory=SupertrendConfig)
    market_trend_filter: MarketTrendFilterConfig = field(default_factory=MarketTrendFilterConfig)
    leader_rotation: LeaderRotationConfig = field(default_factory=LeaderRotationConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    components: tuple[ComponentConfig, ...] = ()


def load_config(path: str | Path) -> AppConfig:
    path = _resolve_existing_path(path)
    raw = _load_mapping(path)
    return parse_config(raw)


def load_split_config(
    strategy_path: str | Path,
    runtime_path: str | Path,
) -> AppConfig:
    strategy_raw = _load_mapping(_resolve_existing_path(strategy_path))
    runtime_raw = _load_mapping(_resolve_existing_path(runtime_path))
    merged = compose_split_config(strategy_raw, runtime_raw)
    return parse_config(merged)


def compose_split_config(
    strategy_raw: dict[str, Any],
    runtime_raw: dict[str, Any],
) -> dict[str, Any]:
    _validate_strategy_schema(strategy_raw)
    _validate_runtime_schema(runtime_raw)
    data_raw = _optional_mapping(runtime_raw, "data")
    portfolio_raw = _optional_mapping(strategy_raw, "portfolio")
    capital_raw = _optional_mapping(runtime_raw, "capital")
    costs_raw = _optional_mapping(runtime_raw, "costs")
    execution_raw = _optional_mapping(runtime_raw, "execution")
    rotation_raw = _optional_mapping(strategy_raw, "rotation")
    live_raw = _optional_mapping(runtime_raw, "live")
    paper_raw = _optional_mapping(runtime_raw, "paper")
    backtest_raw = _optional_mapping(runtime_raw, "backtest")
    hurdle_raw = rotation_raw.get("hurdle", {})
    if not isinstance(hurdle_raw, dict):
        hurdle_raw = {}
    rs_period, rs_period_by_market = _relative_strength_periods(strategy_raw)

    strategy_type = str(strategy_raw.get("type") or portfolio_raw.get("mode") or "").strip()
    if strategy_type == "leader":
        strategy_type = "leader_rotation"
    portfolio_mode = portfolio_raw.get("mode")
    if portfolio_mode and portfolio_mode != strategy_type:
        raise ValueError("portfolio.mode must match strategy type.")
    max_positions = int(portfolio_raw.get("max_positions", 1))
    composed = {
        "strategy": {
            "name": str(strategy_raw.get("name") or strategy_type),
            "type": strategy_type,
            "params": _optional_mapping(strategy_raw, "params"),
        },
        "market": str(runtime_raw.get("market", "US")).upper(),
        "universe": {
            "file": str(runtime_raw.get("universe_file") or "universe.json"),
            "symbols": runtime_raw.get("symbols", ()) or (),
        },
        "timeframe": str(data_raw.get("timeframe") or strategy_raw.get("timeframe") or "30m"),
        "period": str(data_raw.get("period") or strategy_raw.get("period") or "60d"),
        "capital": {
            "initial_cash": runtime_raw.get("initial_cash", capital_raw.get("initial_cash", 10_000.0)),
        },
        "costs": {
            "fee_rate": costs_raw.get("fee_rate", 0.00225),
            "slippage_rate": costs_raw.get("slippage_rate", 0.0005),
        },
        "indicators": _compose_indicators(strategy_raw),
        "filters": _compose_filters(strategy_raw),
        "components": _compose_components(strategy_raw),
        "leader_rotation": {
            "rs_period": rs_period,
            "rs_period_by_market": rs_period_by_market,
            "max_slots": max_positions,
            "hurdle_atr_mult": hurdle_raw.get("multiplier", 1.25),
            "allow_late_chase": rotation_raw.get("allow_late_chase", True),
            "min_rotation_profit_pct": rotation_raw.get("min_rotation_profit_pct", 0.01),
        },
        "exit": {
            "sell_confirm_bars": _find_exit_component_value(strategy_raw, "confirm_bars", default=1),
        },
        "execution": {
            "order_type": execution_raw.get("order_type", "market"),
            "allocation_pct": portfolio_raw.get("allocation_pct", 0.9),
            "broker": execution_raw.get("broker", "paper"),
            "live_confirm_required": execution_raw.get("live_confirm_required", True),
        },
        "risk": {
            "max_position_count": max_positions,
        },
        "live": {
            "holdings_file": live_raw.get("holdings_file", "holding.json"),
            "loop_interval_seconds": live_raw.get("loop_interval_seconds", 60),
        },
        "paper": {
            "state_file": paper_raw.get("state_file", "state/paper.json"),
            "results_dir": paper_raw.get("results_dir", "results/paper"),
            "loop_interval_seconds": paper_raw.get("loop_interval_seconds", 60),
            "run_once_per_candle": paper_raw.get("run_once_per_candle", True),
        },
        "backtest": {
            "results_dir": backtest_raw.get("results_dir", "results/backtests"),
        },
    }
    return composed


def parse_config(raw: dict[str, Any]) -> AppConfig:
    strategy_raw = _required_mapping(raw, "strategy")
    strategy = StrategyIdentity(
        name=str(strategy_raw.get("name") or strategy_raw.get("type") or "").strip(),
        type=str(strategy_raw.get("type") or "").strip(),
        params=dict(_optional_mapping(strategy_raw, "params")),
    )
    if not strategy.name or not strategy.type:
        raise ValueError("strategy.name and strategy.type are required.")

    indicators = _optional_mapping(raw, "indicators")
    filters = _optional_mapping(raw, "filters")
    universe = raw.get("universe", {})
    if isinstance(universe, str):
        universe_file = universe
        symbols: tuple[str, ...] = tuple(str(symbol) for symbol in raw.get("symbols", ()) or ())
    elif isinstance(universe, dict):
        universe_file = str(universe.get("file") or raw.get("universe_file") or "universe.json")
        symbols = tuple(str(symbol) for symbol in universe.get("symbols", raw.get("symbols", ())) or ())
    else:
        raise ValueError("universe must be a string or mapping.")

    return AppConfig(
        strategy=strategy,
        market=str(raw.get("market", "US")).upper(),
        universe_file=universe_file,
        symbols=symbols,
        timeframe=str(raw.get("timeframe", "30m")),
        period=str(raw.get("period", "60d")),
        capital=CapitalConfig(**_known(_optional_mapping(raw, "capital"), {"initial_cash"})),
        costs=CostsConfig(**_known(_optional_mapping(raw, "costs"), {"fee_rate", "slippage_rate"})),
        supertrend=SupertrendConfig(
            **_known(
                _optional_mapping(indicators, "supertrend"),
                {"enabled", "period", "multiplier", "atr_method", "symbol_multipliers"},
            )
        ),
        market_trend_filter=MarketTrendFilterConfig(
            **_known(
                _optional_mapping(filters, "market_trend"),
                {"enabled", "timeframe"},
            )
        ),
        leader_rotation=LeaderRotationConfig(
            **_known(
                _optional_mapping(raw, "leader_rotation"),
                {
                    "rs_period",
                    "rs_period_by_market",
                    "max_slots",
                    "hurdle_atr_mult",
                    "allow_late_chase",
                    "min_rotation_profit_pct",
                },
            )
        ),
        exit=ExitConfig(**_known(_optional_mapping(raw, "exit"), {"sell_confirm_bars"})),
        execution=ExecutionConfig(
            **_known(
                _optional_mapping(raw, "execution"),
                {"order_type", "allocation_pct", "broker", "live_confirm_required"},
            )
        ),
        risk=RiskConfig(**_known(_optional_mapping(raw, "risk"), {"max_position_count"})),
        live=LiveConfig(
            **_known(
                _optional_mapping(raw, "live"),
                {"holdings_file", "loop_interval_seconds"},
            )
        ),
        paper=PaperConfig(
            **_known(
                _optional_mapping(raw, "paper"),
                {"state_file", "results_dir", "loop_interval_seconds", "run_once_per_candle"},
            )
        ),
        backtest=BacktestConfig(**_known(_optional_mapping(raw, "backtest"), {"results_dir"})),
        components=_parse_components(raw.get("components", ())),
    )


def _compose_indicators(strategy_raw: dict[str, Any]) -> dict[str, Any]:
    # Triple SuperTrend still uses the single-ST defaults for ATR percentage
    # and benchmark-trend filtering.  Its own settings remain available in
    # ``components`` and are composed by the strategy at runtime.
    supertrend = (
        _find_component(strategy_raw, "supertrend")
        or _find_component(strategy_raw, "triple_supertrend")
        or {}
    )
    return {
        "supertrend": {
            "enabled": supertrend.get("enabled", True),
            "period": supertrend.get("period", 10),
            "multiplier": supertrend.get("multiplier", 3.0),
            "atr_method": supertrend.get("atr_method", "wilder"),
            "symbol_multipliers": supertrend.get("symbol_multipliers", {}),
        }
    }


def _compose_filters(strategy_raw: dict[str, Any]) -> dict[str, Any]:
    market_filter = _find_component(strategy_raw, "benchmark_trend")
    if not market_filter:
        market_filter = _find_component(strategy_raw, "market_trend")
    return {
        "market_trend": {
            "enabled": bool(market_filter.get("enabled", True)) if market_filter else False,
            "timeframe": market_filter.get("timeframe", "1d") if market_filter else "1d",
        }
    }


def _find_component(strategy_raw: dict[str, Any], component_type: str) -> dict[str, Any] | None:
    signals = _optional_mapping(strategy_raw, "signals")
    groups = [
        signals.get("entries", []),
        signals.get("filters", []),
        signals.get("exits", []),
    ]
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, dict) and item.get("type") == component_type:
                return item
    if strategy_raw.get("type") == component_type:
        return strategy_raw
    return None


def _find_component_value(
    strategy_raw: dict[str, Any],
    component_type: str,
    key: str,
    default: Any,
) -> Any:
    component = _find_component(strategy_raw, component_type)
    if not component:
        return default
    return component.get(key, default)


def _find_exit_component_value(
    strategy_raw: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    for component_type in ("supertrend_flip", "triple_supertrend_flip"):
        component = _find_component(strategy_raw, component_type)
        if component and bool(component.get("enabled", True)):
            return component.get(key, default)
    return default


def _relative_strength_periods(strategy_raw: dict[str, Any]) -> tuple[int, dict[str, int]]:
    relative_strength = _find_component(strategy_raw, "relative_strength")
    if not relative_strength:
        return 100, {}
    lookback = relative_strength.get("lookback_bars", 100)
    if isinstance(lookback, dict):
        by_market = {str(key).upper(): int(value) for key, value in lookback.items() if key != "default"}
        default = int(lookback.get("default", by_market.get("US", 100)))
        return default, by_market
    return int(lookback), {}


def _compose_components(strategy_raw: dict[str, Any]) -> list[dict[str, Any]]:
    signals = _optional_mapping(strategy_raw, "signals")
    components: list[dict[str, Any]] = []
    for group_name in ("entries", "filters", "exits"):
        group = signals.get(group_name, [])
        if not isinstance(group, list):
            raise ValueError(f"signals.{group_name} must be a list.")
        for item in group:
            if not isinstance(item, dict):
                raise ValueError(f"signals.{group_name} items must be mappings.")
            component_type = str(item.get("type") or "").strip()
            if not component_type:
                raise ValueError(f"signals.{group_name} items require type.")
            _validate_component_keys(group_name, component_type, item)
            components.append(
                {
                    "type": component_type,
                    "enabled": bool(item.get("enabled", True)),
                    "group": group_name,
                    "params": {
                        key: value
                        for key, value in item.items()
                        if key not in {"type", "enabled"}
                    },
                }
            )
    return components


def _validate_strategy_schema(raw: dict[str, Any]) -> None:
    _reject_unknown_keys(raw, {"name", "type", "params", "portfolio", "signals", "rotation", "timeframe", "period"}, "strategy")
    _optional_mapping(raw, "params")
    _reject_unknown_keys(_optional_mapping(raw, "portfolio"), {"max_positions", "allocation_pct"}, "strategy.portfolio")
    rotation = _optional_mapping(raw, "rotation")
    _reject_unknown_keys(rotation, {"hurdle", "allow_late_chase", "min_rotation_profit_pct"}, "strategy.rotation")
    hurdle = rotation.get("hurdle", {})
    if hurdle is not None:
        if not isinstance(hurdle, dict):
            raise ValueError("strategy.rotation.hurdle must be a mapping.")
        _reject_unknown_keys(hurdle, {"multiplier"}, "strategy.rotation.hurdle")


def _validate_runtime_schema(raw: dict[str, Any]) -> None:
    _reject_unknown_keys(
        raw,
        {
            "name",
            "market",
            "universe_file",
            "symbols",
            "data",
            "capital",
            "costs",
            "execution",
            "live",
            "paper",
            "backtest",
        },
        "runtime",
    )
    _reject_unknown_keys(_optional_mapping(raw, "data"), {"timeframe", "period"}, "runtime.data")
    _reject_unknown_keys(_optional_mapping(raw, "capital"), {"initial_cash"}, "runtime.capital")
    _reject_unknown_keys(_optional_mapping(raw, "costs"), {"fee_rate", "slippage_rate"}, "runtime.costs")
    _reject_unknown_keys(_optional_mapping(raw, "execution"), {"order_type", "broker", "live_confirm_required"}, "runtime.execution")
    _reject_unknown_keys(_optional_mapping(raw, "live"), {"holdings_file", "loop_interval_seconds"}, "runtime.live")
    _reject_unknown_keys(
        _optional_mapping(raw, "paper"),
        {"state_file", "results_dir", "loop_interval_seconds", "run_once_per_candle"},
        "runtime.paper",
    )
    _reject_unknown_keys(_optional_mapping(raw, "backtest"), {"results_dir"}, "runtime.backtest")


def _reject_unknown_keys(raw: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unsupported keys for {label}: {', '.join(sorted(unknown))}")


def _validate_component_keys(group_name: str, component_type: str, raw: dict[str, Any]) -> None:
    allowed = {
        ("entries", "supertrend"): {"type", "enabled", "period", "multiplier", "atr_method", "symbol_multipliers"},
        ("entries", "triple_supertrend"): {"type", "enabled", "atr_method", "settings"},
        ("filters", "benchmark_trend"): {"type", "enabled", "timeframe"},
        ("filters", "market_trend"): {"type", "enabled", "timeframe"},
        ("filters", "relative_strength"): {"type", "enabled", "lookback_bars"},
        ("filters", "ichimoku_cloud"): {"type", "enabled", "tenkan", "kijun", "span_b", "shift"},
        ("filters", "ema_trend"): {"type", "enabled", "period"},
        ("exits", "supertrend_flip"): {"type", "enabled", "confirm_bars"},
        ("exits", "triple_supertrend_flip"): {"type", "enabled", "down_count", "confirm_bars"},
    }.get((group_name, component_type))
    if allowed is None:
        raise ValueError(f"Unsupported component: signals.{group_name} type={component_type}")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"Unsupported keys for signals.{group_name} type={component_type}: {', '.join(sorted(unknown))}"
        )
    if (group_name, component_type) == ("entries", "triple_supertrend"):
        _validate_triple_supertrend_settings(raw.get("settings"))


def _validate_triple_supertrend_settings(raw: Any) -> None:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("signals.entries type=triple_supertrend settings must contain exactly three mappings.")
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"signals.entries type=triple_supertrend settings[{index}] must be a mapping."
            )
        _reject_unknown_keys(
            item,
            {"period", "multiplier"},
            f"signals.entries type=triple_supertrend settings[{index}]",
        )
        if "period" not in item or "multiplier" not in item:
            raise ValueError(
                f"signals.entries type=triple_supertrend settings[{index}] requires period and multiplier."
            )


def _parse_components(raw: Any) -> tuple[ComponentConfig, ...]:
    if raw in (None, ()):
        return ()
    if not isinstance(raw, list):
        raise ValueError("components must be a list.")
    components: list[ComponentConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("components items must be mappings.")
        component_type = str(item.get("type") or "").strip()
        if not component_type:
            raise ValueError("components items require type.")
        params = item.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("components.params must be a mapping.")
        components.append(
            ComponentConfig(
                type=component_type,
                enabled=bool(item.get("enabled", True)),
                group=str(item.get("group", "")),
                params=params,
            )
        )
    return tuple(components)


def load_universe(config: AppConfig) -> list[str]:
    if config.symbols:
        return list(config.symbols)

    path = _resolve_existing_path(config.universe_file)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if config.market == "KR":
        return list(data.get("KR_UNIVERSE_MAP", {}).keys())
    if config.market == "US":
        return list(data.get("US_UNIVERSE_LIST", []))
    if config.market == "AUTO":
        return list(data.get("US_UNIVERSE_LIST", []))
    raise ValueError("market must be US or KR.")


def load_universe_for_market(universe_file: str, market: str) -> list[str]:
    path = _resolve_existing_path(universe_file)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if market == "KR":
        return list(data.get("KR_UNIVERSE_MAP", {}).keys())
    if market == "US":
        return list(data.get("US_UNIVERSE_LIST", []))
    raise ValueError("market must be US or KR.")


def to_yfinance_symbol(symbol: str, market: str, universe_file: str = "universe.json") -> str:
    if market != "KR":
        return symbol

    path = _resolve_existing_path(universe_file)
    if not path.exists():
        return symbol
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    kr_market = data.get("KR_UNIVERSE_MAP", {}).get(symbol)
    if kr_market == "KOSPI":
        return f"{symbol}.KS"
    if kr_market == "KOSDAQ":
        return f"{symbol}.KQ"
    return symbol


def benchmark_for_symbol(symbol: str, market: str, universe_file: str = "universe.json") -> str:
    if market != "KR":
        return "QQQ"
    path = _resolve_existing_path(universe_file)
    if not path.exists():
        return "^KS11"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    kr_market = data.get("KR_UNIVERSE_MAP", {}).get(symbol)
    if kr_market == "KOSDAQ":
        return "^KQ11"
    return "^KS11"


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(handle) or {}
        elif suffix == ".toml":
            loaded = tomllib.load(handle)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .toml.")
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping.")
    return loaded


def _resolve_existing_path(path: str | Path) -> Path:
    """Resolve project-owned relative files without depending on process cwd.

    User-provided paths in the current working directory remain highest
    priority.  The unified package root and repository root are fallbacks for
    bundled configs and the shared universe file.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    for root in (_PACKAGE_PROJECT_ROOT, _REPOSITORY_ROOT):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return candidate


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _optional_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _known(raw: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: raw[key] for key in keys if key in raw}
