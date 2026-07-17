from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from ..config import AppConfig, ComponentConfig


ENTRY_COMPONENTS = frozenset({"supertrend", "single_supertrend", "triple_supertrend"})
EXIT_COMPONENTS = frozenset({"supertrend_flip", "triple_supertrend_flip"})
MARKET_FILTER_COMPONENTS = frozenset({"benchmark_trend", "market_trend"})
ASSET_FILTER_COMPONENTS = frozenset({"ichimoku_cloud", "ema_trend"})
RS_METHOD_ALIASES = {
    "relative_strength": "relative_strength",
    "excess": "relative_strength",
    "excess_rs": "relative_strength",
    "vol_adjusted": "vol_adjusted_relative_strength",
    "vol_adjusted_rs": "vol_adjusted_relative_strength",
    "vol_adjusted_relative_strength": "vol_adjusted_relative_strength",
    "composite": "composite_relative_strength",
    "composite_rs": "composite_relative_strength",
    "composite_relative_strength": "composite_relative_strength",
    "skip_recent": "skip_recent_relative_strength",
    "skip_recent_rs": "skip_recent_relative_strength",
    "skip_1m": "skip_recent_relative_strength",
    "skip_1m_rs": "skip_recent_relative_strength",
    "skip_recent_relative_strength": "skip_recent_relative_strength",
    "beta_adjusted": "beta_adjusted_alpha",
    "beta_adjusted_alpha": "beta_adjusted_alpha",
    "dual_momentum": "dual_momentum",
}


def replace_path(instance: Any, path: str, value: Any) -> Any:
    """Return a dataclass copy with one dotted path replaced.

    This is intentionally strict: misspelled paths fail instead of silently
    creating research parameters that the production runtime never consumes.
    """

    head, separator, tail = path.partition(".")
    if not is_dataclass(instance):
        raise TypeError(f"Cannot replace {path!r}: {type(instance).__name__} is not a dataclass.")
    known = {item.name for item in fields(instance)}
    if head not in known:
        raise KeyError(f"Unknown config path: {path}")
    if not separator:
        return replace(instance, **{head: value})
    child = getattr(instance, head)
    return replace(instance, **{head: replace_path(child, tail, value)})


def upsert_component(
    config: AppConfig,
    *,
    group: str,
    component_type: str,
    enabled: bool = True,
    params: Mapping[str, Any] | None = None,
    exclusive_types: Iterable[str] = (),
) -> AppConfig:
    """Immutably add or replace a canonical strategy component."""

    excluded = set(exclusive_types)
    excluded.add(component_type)
    kept = tuple(
        component
        for component in config.components
        if not (component.group == group and component.type in excluded)
    )
    component = ComponentConfig(
        type=component_type,
        enabled=bool(enabled),
        group=group,
        params=dict(params or {}),
    )
    return replace(config, components=kept + (component,))


def remove_components(
    config: AppConfig,
    *,
    group: str | None = None,
    component_types: Iterable[str] = (),
) -> AppConfig:
    """Return a config without the selected component types."""

    selected = set(component_types)
    kept = tuple(
        component
        for component in config.components
        if not (
            (group is None or component.group == group)
            and (not selected or component.type in selected)
        )
    )
    return replace(config, components=kept)


def component(config: AppConfig, group: str, component_type: str) -> ComponentConfig | None:
    return next(
        (
            item
            for item in config.components
            if item.group == group and item.type == component_type
        ),
        None,
    )


def active_entry(config: AppConfig) -> ComponentConfig | None:
    enabled = [
        item
        for item in config.components
        if item.group == "entries" and item.enabled and item.type in ENTRY_COMPONENTS
    ]
    # Exported triple configs may also carry a base SuperTrend component for
    # ATR/hurdle parameters; strategy semantics still prioritize Triple ST.
    return next((item for item in enabled if item.type == "triple_supertrend"), enabled[0] if enabled else None)


def with_timeframe(config: AppConfig, timeframe: str) -> AppConfig:
    timeframe = str(timeframe).strip()
    if not timeframe:
        raise ValueError("timeframe must not be empty.")
    return replace(config, timeframe=timeframe)


def with_single_entry(
    config: AppConfig,
    *,
    period: int | None = None,
    multiplier: float | None = None,
    atr_method: str | None = None,
    symbol_multipliers: Mapping[str, float] | None = None,
    enabled: bool = True,
) -> AppConfig:
    period = config.supertrend.period if period is None else int(period)
    multiplier = config.supertrend.multiplier if multiplier is None else float(multiplier)
    atr_method = config.supertrend.atr_method if atr_method is None else str(atr_method)
    multipliers = (
        dict(config.supertrend.symbol_multipliers)
        if symbol_multipliers is None
        else {str(symbol): float(value) for symbol, value in symbol_multipliers.items()}
    )
    if period < 1 or multiplier <= 0:
        raise ValueError("SuperTrend period and multiplier must be positive.")

    updated = replace(
        config,
        supertrend=replace(
            config.supertrend,
            enabled=bool(enabled),
            period=period,
            multiplier=multiplier,
            atr_method=atr_method,
            symbol_multipliers=multipliers,
        ),
    )
    updated = upsert_component(
        updated,
        group="entries",
        component_type="supertrend",
        enabled=enabled,
        params={
            "period": period,
            "multiplier": multiplier,
            "atr_method": atr_method,
            "symbol_multipliers": multipliers,
        },
        exclusive_types=ENTRY_COMPONENTS,
    )
    current_exit = next(
        (item for item in updated.components if item.group == "exits" and item.type in EXIT_COMPONENTS),
        None,
    )
    exit_params = dict(current_exit.params) if current_exit is not None else {}
    exit_params.pop("down_count", None)
    exit_params["confirm_bars"] = int(
        exit_params.get("confirm_bars", updated.exit.sell_confirm_bars)
    )
    return upsert_component(
        updated,
        group="exits",
        component_type="supertrend_flip",
        params=exit_params,
        exclusive_types=EXIT_COMPONENTS,
    )


def with_triple_entry(
    config: AppConfig,
    settings: Sequence[tuple[int, float] | Mapping[str, Any]] = (
        (10, 1.0),
        (11, 2.0),
        (12, 3.0),
    ),
    *,
    atr_method: str | None = None,
    enabled: bool = True,
) -> AppConfig:
    parsed: list[dict[str, int | float]] = []
    for setting in settings:
        if isinstance(setting, Mapping):
            period = int(setting["period"])
            multiplier = float(setting["multiplier"])
        else:
            period = int(setting[0])
            multiplier = float(setting[1])
        if period < 1 or multiplier <= 0:
            raise ValueError("Triple SuperTrend periods and multipliers must be positive.")
        parsed.append({"period": period, "multiplier": multiplier})
    if len(parsed) != 3:
        raise ValueError("triple_supertrend requires exactly three settings.")

    method = config.supertrend.atr_method if atr_method is None else str(atr_method)
    updated = upsert_component(
        config,
        group="entries",
        component_type="triple_supertrend",
        enabled=enabled,
        params={"settings": parsed, "atr_method": method},
        exclusive_types=ENTRY_COMPONENTS,
    )
    current_exit = next(
        (item for item in updated.components if item.group == "exits" and item.type in EXIT_COMPONENTS),
        None,
    )
    exit_params = dict(current_exit.params) if current_exit is not None else {}
    exit_params["confirm_bars"] = int(
        exit_params.get("confirm_bars", updated.exit.sell_confirm_bars)
    )
    exit_params["down_count"] = int(exit_params.get("down_count", 2))
    return upsert_component(
        updated,
        group="exits",
        component_type="triple_supertrend_flip",
        params=exit_params,
        exclusive_types=EXIT_COMPONENTS,
    )


def with_entry(config: AppConfig, entry: str | ComponentConfig | Mapping[str, Any]) -> AppConfig:
    """Apply a single/triple entry description to a config."""

    if isinstance(entry, ComponentConfig):
        raw = {"type": entry.type, "enabled": entry.enabled, **entry.params}
    elif isinstance(entry, str):
        raw = {"type": entry}
    else:
        raw = dict(entry)

    entry_type = str(raw.pop("type", "supertrend"))
    enabled = bool(raw.pop("enabled", True))
    if entry_type in {"single", "single_supertrend", "supertrend"}:
        return with_single_entry(config, enabled=enabled, **raw)
    if entry_type in {"triple", "triple_supertrend"}:
        return with_triple_entry(config, enabled=enabled, **raw)
    raise ValueError(f"Unsupported research entry type: {entry_type}")


def with_filter(
    config: AppConfig,
    filter_type: str,
    *,
    enabled: bool = True,
    **params: Any,
) -> AppConfig:
    updated = upsert_component(
        config,
        group="filters",
        component_type=filter_type,
        enabled=enabled,
        params=params,
    )
    if filter_type in MARKET_FILTER_COMPONENTS:
        timeframe = str(params.get("timeframe", config.market_trend_filter.timeframe))
        updated = replace(
            updated,
            market_trend_filter=replace(
                updated.market_trend_filter,
                enabled=bool(enabled),
                timeframe=timeframe,
            ),
        )
    return updated


def with_market_filter(config: AppConfig, value: str | None) -> AppConfig:
    if value is None or str(value).lower() in {"", "none", "off", "false"}:
        updated = remove_components(
            config,
            group="filters",
            component_types=MARKET_FILTER_COMPONENTS,
        )
        return replace(
            updated,
            market_trend_filter=replace(updated.market_trend_filter, enabled=False),
        )

    timeframe = str(value)
    for prefix in ("market_", "benchmark_", "qqq_"):
        if timeframe.startswith(prefix):
            timeframe = timeframe[len(prefix) :]
            break
    updated = remove_components(
        config,
        group="filters",
        component_types=MARKET_FILTER_COMPONENTS,
    )
    return with_filter(updated, "benchmark_trend", enabled=True, timeframe=timeframe)


def with_asset_filters(config: AppConfig, value: str | Iterable[str]) -> AppConfig:
    if isinstance(value, str):
        names = tuple(item.strip() for item in value.split("+") if item.strip())
    else:
        names = tuple(str(item).strip() for item in value if str(item).strip())
    names = () if names in {("none",), ("off",)} else names
    unknown = set(names) - ASSET_FILTER_COMPONENTS
    if unknown:
        raise ValueError(f"Unsupported asset filters: {', '.join(sorted(unknown))}")

    updated = remove_components(
        config,
        group="filters",
        component_types=ASSET_FILTER_COMPONENTS,
    )
    for name in names:
        defaults: dict[str, Any]
        if name == "ichimoku_cloud":
            defaults = {"tenkan": 9, "kijun": 26, "span_b": 52, "shift": 26}
        else:
            defaults = {"period": 200}
        updated = with_filter(updated, name, **defaults)
    return updated


def with_filters(
    config: AppConfig,
    filters: Iterable[str | ComponentConfig | Mapping[str, Any]],
    *,
    replace_known: bool = True,
) -> AppConfig:
    updated = config
    if replace_known:
        updated = remove_components(
            updated,
            group="filters",
            component_types=MARKET_FILTER_COMPONENTS | ASSET_FILTER_COMPONENTS,
        )
        updated = replace(
            updated,
            market_trend_filter=replace(updated.market_trend_filter, enabled=False),
        )
    for item in filters:
        if isinstance(item, ComponentConfig):
            raw = {"type": item.type, "enabled": item.enabled, **item.params}
        elif isinstance(item, str):
            raw = {"type": item}
        else:
            raw = dict(item)
        filter_type = str(raw.pop("type"))
        if filter_type in {"", "none"}:
            continue
        enabled = bool(raw.pop("enabled", True))
        updated = with_filter(updated, filter_type, enabled=enabled, **raw)
    return updated


def with_sell_confirmation(
    config: AppConfig,
    confirm_bars: int,
    *,
    down_count: int | None = None,
) -> AppConfig:
    confirm_bars = int(confirm_bars)
    if confirm_bars < 1:
        raise ValueError("sell confirmation must be at least one bar.")
    updated = replace(
        config,
        exit=replace(config.exit, sell_confirm_bars=confirm_bars),
    )

    existing = next(
        (
            item
            for item in updated.components
            if item.group == "exits" and item.type in EXIT_COMPONENTS
        ),
        None,
    )
    entry = active_entry(updated)
    exit_type = (
        existing.type
        if existing is not None
        else "triple_supertrend_flip"
        if entry is not None and entry.type == "triple_supertrend"
        else "supertrend_flip"
    )
    params = dict(existing.params) if existing is not None else {}
    params["confirm_bars"] = confirm_bars
    if exit_type == "triple_supertrend_flip":
        params["down_count"] = int(down_count if down_count is not None else params.get("down_count", 2))
    return upsert_component(
        updated,
        group="exits",
        component_type=exit_type,
        params=params,
        exclusive_types=EXIT_COMPONENTS,
    )


def with_relative_strength(
    config: AppConfig,
    period: int | Mapping[str, int],
) -> AppConfig:
    return with_rs_scoring(config, method="relative_strength", period=period)


def with_rs_scoring(
    config: AppConfig,
    *,
    method: str | None = None,
    period: int | Mapping[str, int] | None = None,
) -> AppConfig:
    scoring_type = config.scoring.type
    if method is not None:
        raw_method = str(method).strip().lower()
        try:
            scoring_type = RS_METHOD_ALIASES[raw_method]
        except KeyError as exc:
            raise ValueError(f"Unsupported rs_method: {raw_method}") from exc

    if period is None:
        period = config.scoring.params.get("lookback_bars", 100)
    if isinstance(period, Mapping):
        values = {str(key).upper(): int(value) for key, value in period.items()}
        if any(value < 1 for value in values.values()):
            raise ValueError("RS periods must be positive.")
        current = config.scoring.params.get("lookback_bars", 100)
        current_default = (
            int(current.get("default", current.get("DEFAULT", current.get("US", 100))))
            if isinstance(current, Mapping)
            else int(current)
        )
        default = int(values.pop("DEFAULT", values.get("US", current_default)))
        lookback: int | dict[str, int] = {"default": default, **values}
    else:
        default = int(period)
        if default < 1:
            raise ValueError("RS period must be positive.")
        lookback = default
    return replace(
        config,
        scoring=replace(
            config.scoring,
            type=scoring_type,
            params={"lookback_bars": lookback},
        ),
    )


def with_rotation_hurdle(config: AppConfig, multiplier: float) -> AppConfig:
    value = float(multiplier)
    if value < 0:
        raise ValueError("rotation hurdle multiplier must be non-negative.")
    return replace(
        config,
        leader_rotation=replace(config.leader_rotation, hurdle_atr_mult=value),
    )


def with_max_positions(config: AppConfig, count: int) -> AppConfig:
    count = int(count)
    if count < 1:
        raise ValueError("max_positions must be at least one.")
    return replace(
        config,
        leader_rotation=replace(config.leader_rotation, max_slots=count),
        risk=replace(config.risk, max_position_count=count),
    )


def with_costs(
    config: AppConfig,
    *,
    fee_rate: float | None = None,
    slippage_rate: float | None = None,
) -> AppConfig:
    fee = config.costs.fee_rate if fee_rate is None else float(fee_rate)
    slippage = config.costs.slippage_rate if slippage_rate is None else float(slippage_rate)
    if fee < 0 or slippage < 0:
        raise ValueError("fee and slippage rates cannot be negative.")
    return replace(
        config,
        costs=replace(config.costs, fee_rate=fee, slippage_rate=slippage),
    )


def apply_config_overlay(config: AppConfig, overlay: Mapping[str, Any]) -> AppConfig:
    """Apply a research parameter mapping without mutating the base config.

    Supported keys intentionally map to production concepts. Dotted keys offer
    a strict escape hatch for additional immutable AppConfig fields.
    """

    values = dict(overlay)
    updated = config

    if "timeframe" in values:
        updated = with_timeframe(updated, values.pop("timeframe"))
    entry_value = values.pop("entry", values.pop("signal", None))
    if entry_value is not None:
        updated = with_entry(updated, entry_value)

    entry = active_entry(updated)
    single_updates = {
        key: values.pop(key)
        for key in ("period", "multiplier", "atr_method")
        if key in values
    }
    for alias, target in (("st_period", "period"), ("st_multiplier", "multiplier")):
        if alias in values:
            single_updates[target] = values.pop(alias)
    if single_updates:
        if entry is not None and entry.type == "triple_supertrend":
            raise ValueError("Use triple_settings to change a triple_supertrend entry.")
        updated = with_single_entry(updated, **single_updates)
    if "triple_settings" in values:
        updated = with_triple_entry(updated, values.pop("triple_settings"))

    if "filters" in values:
        updated = with_filters(updated, values.pop("filters"))
    if "market_filter" in values:
        updated = with_market_filter(updated, values.pop("market_filter"))
    if "asset_filter" in values:
        updated = with_asset_filters(updated, values.pop("asset_filter"))
    if "asset_filters" in values:
        updated = with_asset_filters(updated, values.pop("asset_filters"))

    confirm = values.pop("sell_confirm_bars", values.pop("sell_confirmation", None))
    down_count = values.pop("triple_exit_down_count", None)
    if confirm is not None or down_count is not None:
        updated = with_sell_confirmation(
            updated,
            updated.exit.sell_confirm_bars if confirm is None else confirm,
            down_count=down_count,
        )

    rs_method = values.pop("rs_method", values.pop("rs_score", values.pop("scoring_type", None)))
    rs = values.pop("rs_period", values.pop("relative_strength", None))
    if rs_method is not None or rs is not None:
        updated = with_rs_scoring(updated, method=rs_method, period=rs)

    hurdle = values.pop("hurdle", values.pop("rotation_hurdle", values.pop("hurdle_atr_mult", None)))
    if hurdle is not None:
        updated = with_rotation_hurdle(updated, hurdle)

    max_positions = values.pop(
        "max_positions",
        values.pop("leader_count", values.pop("leader_positions", None)),
    )
    if max_positions is not None:
        updated = with_max_positions(updated, max_positions)

    costs = values.pop("costs", None)
    fee = values.pop("fee_rate", None)
    slippage = values.pop("slippage_rate", None)
    if costs is not None:
        if not isinstance(costs, Mapping):
            raise TypeError("costs overlay must be a mapping.")
        fee = costs.get("fee_rate", fee)
        slippage = costs.get("slippage_rate", slippage)
    if fee is not None or slippage is not None:
        updated = with_costs(updated, fee_rate=fee, slippage_rate=slippage)

    for path, value in tuple(values.items()):
        if "." not in path:
            raise KeyError(f"Unsupported research overlay key: {path}")
        updated = replace_path(updated, path, value)
    return updated


__all__ = [
    "ASSET_FILTER_COMPONENTS",
    "ENTRY_COMPONENTS",
    "EXIT_COMPONENTS",
    "MARKET_FILTER_COMPONENTS",
    "active_entry",
    "apply_config_overlay",
    "component",
    "remove_components",
    "replace_path",
    "upsert_component",
    "with_asset_filters",
    "with_costs",
    "with_entry",
    "with_filter",
    "with_filters",
    "with_max_positions",
    "with_market_filter",
    "with_relative_strength",
    "with_rotation_hurdle",
    "with_rs_scoring",
    "with_sell_confirmation",
    "with_single_entry",
    "with_timeframe",
    "with_triple_entry",
]
