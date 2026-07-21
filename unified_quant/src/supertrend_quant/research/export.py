from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ..config import AppConfig, ComponentConfig, compose_split_config, parse_config


def _serialized_component(component: ComponentConfig) -> dict[str, Any]:
    return {
        "type": component.type,
        "enabled": component.enabled,
        **component.params,
    }


def _base_supertrend_entry(config: AppConfig) -> dict[str, Any]:
    """Encode the base ST used for ATR/hurdles alongside a triple entry.

    Triple strategy components intentionally retain the canonical base
    SuperTrend for ATR percentage and benchmark filters. The strict triple
    schema has no fields for those base parameters, so an additional
    SuperTrend component is the lossless split-YAML representation. Runtime
    entry selection explicitly prioritizes an enabled triple component.
    """

    return {
        "type": "supertrend",
        "enabled": config.supertrend.enabled,
        "period": config.supertrend.period,
        "multiplier": config.supertrend.multiplier,
        "atr_method": config.supertrend.atr_method,
        "symbol_multipliers": dict(config.supertrend.symbol_multipliers),
    }


def config_to_split_dicts(
    config: AppConfig,
    *,
    strategy_name: str | None = None,
    runtime_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Serialize a researched canonical config back to strict split YAML shapes."""
    grouped: dict[str, list[dict[str, Any]]] = {
        "entries": [],
        "filters": [],
        "exits": [],
    }
    has_triple_entry = any(
        component.group == "entries"
        and component.type == "triple_supertrend"
        and component.enabled
        for component in config.components
    )
    has_base_entry = any(
        component.group == "entries" and component.type == "supertrend"
        for component in config.components
    )
    if has_triple_entry and not has_base_entry:
        grouped["entries"].append(_base_supertrend_entry(config))

    for component in config.components:
        if component.group not in grouped:
            continue
        grouped[component.group].append(_serialized_component(component))

    strategy: dict[str, Any] = {
        "name": strategy_name or config.strategy.name,
        "type": config.strategy.type,
        "portfolio": {
            "max_positions": config.risk.max_position_count,
            "allocation_pct": config.execution.allocation_pct,
        },
        "scoring": {
            "type": config.scoring.type,
            "params": dict(config.scoring.params),
        },
        "signals": grouped,
    }
    if config.strategy.type == "leader_rotation":
        strategy["rotation"] = {
            "hurdle": {"multiplier": config.leader_rotation.hurdle_atr_mult},
            "allow_late_chase": config.leader_rotation.allow_late_chase,
            "min_rotation_profit_pct": config.leader_rotation.min_rotation_profit_pct,
        }
    if config.strategy.params:
        strategy["params"] = dict(config.strategy.params)

    universe = {
        "source": config.universe.source,
        "profiles": {
            market: list(profiles)
            for market, profiles in config.universe.profiles.items()
        },
        "file": config.universe.file,
        "history_file": config.universe.history_file,
        "symbols": list(config.universe.symbols),
        "refresh": config.universe.refresh,
        "snapshot_dir": config.universe.snapshot_dir,
        "filters": asdict(config.universe.filters),
    }
    if config.symbols:
        universe.update({"source": "symbols", "symbols": list(config.symbols)})
    elif config.universe.source == "file" and config.universe_file != config.universe.file:
        universe["file"] = config.universe_file

    runtime: dict[str, Any] = {
        "name": runtime_name or f"{config.strategy.name}_runtime",
        "market": config.market,
        "universe": universe,
        "data": {"timeframe": config.timeframe, "period": config.period},
        "capital": {"initial_cash": config.capital.initial_cash},
        "costs": {
            "fee_rate": config.costs.fee_rate,
            "slippage_rate": config.costs.slippage_rate,
        },
        "execution": {
            "order_type": config.execution.order_type,
            "broker": config.execution.broker,
            "live_confirm_required": config.execution.live_confirm_required,
        },
        "live": {
            "holdings_file": config.live.holdings_file,
            "loop_interval_seconds": config.live.loop_interval_seconds,
        },
        "paper": {
            "state_file": config.paper.state_file,
            "results_dir": config.paper.results_dir,
            "loop_interval_seconds": config.paper.loop_interval_seconds,
            "run_once_per_candle": config.paper.run_once_per_candle,
        },
        "backtest": {"results_dir": config.backtest.results_dir},
    }
    return strategy, runtime


def config_to_data_dict(config: AppConfig) -> dict[str, Any]:
    """Serialize shared market-data settings independently of runtime."""

    return {"data_store": asdict(config.data_store)}


def strict_split_roundtrip(config: AppConfig) -> AppConfig:
    """Serialize and immediately reload through the public strict parser."""

    strategy, runtime = config_to_split_dicts(config)
    return parse_config(compose_split_config(strategy, runtime, config_to_data_dict(config)))


def split_yaml_text(config: AppConfig) -> tuple[str, str, str]:
    strategy, runtime = config_to_split_dicts(config)
    data = config_to_data_dict(config)
    # Keep stdout/save paths on the same strict schema gate as normal runtime
    # loading. This catches unsupported research components before promotion.
    parse_config(compose_split_config(strategy, runtime, data))
    return (
        yaml.safe_dump(strategy, sort_keys=False, allow_unicode=True).strip() + "\n",
        yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True).strip() + "\n",
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip() + "\n",
    )


def save_split_yaml(config: AppConfig, directory: str | Path) -> tuple[Path, Path, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    strategy_path = target / "strategy.yaml"
    runtime_path = target / "runtime.yaml"
    data_path = target / "data.yaml"
    strategy_text, runtime_text, data_text = split_yaml_text(config)
    strategy_path.write_text(strategy_text, encoding="utf-8")
    runtime_path.write_text(runtime_text, encoding="utf-8")
    data_path.write_text(data_text, encoding="utf-8")
    return strategy_path, runtime_path, data_path


__all__ = [
    "config_to_data_dict",
    "config_to_split_dicts",
    "save_split_yaml",
    "split_yaml_text",
    "strict_split_roundtrip",
]
