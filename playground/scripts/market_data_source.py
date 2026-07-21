from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from supertrend_quant.config import AppConfig
from supertrend_quant.data import MarketData
from supertrend_quant.research.data_resolver import download_for_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_PACKAGE_ROOT = REPOSITORY_ROOT / "unified_quant" / "src" / "supertrend_quant"
UNIFIED_PACKAGE_NAME = "_supertrend_quant_local_store"


@lru_cache(maxsize=1)
def _unified_modules() -> tuple[ModuleType, ModuleType]:
    spec = importlib.util.spec_from_file_location(
        UNIFIED_PACKAGE_NAME,
        UNIFIED_PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(UNIFIED_PACKAGE_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the canonical unified_quant package.")
    package = importlib.util.module_from_spec(spec)
    sys.modules[UNIFIED_PACKAGE_NAME] = package
    spec.loader.exec_module(package)
    config_module = importlib.import_module(f"{UNIFIED_PACKAGE_NAME}.config")
    resolver_module = importlib.import_module(
        f"{UNIFIED_PACKAGE_NAME}.research.data_resolver"
    )
    return config_module, resolver_module


def _local_config(
    config: AppConfig,
    *,
    strategy_path: str,
    runtime_path: str,
):
    config_module, _ = _unified_modules()
    local = config_module.load_split_config(strategy_path, runtime_path)
    local = replace(
        local,
        period=config.period,
        timeframe=config.timeframe,
        data_store=replace(local.data_store, provider="parquet"),
    )

    if local.universe.source == "history_file":
        history_name = Path(local.universe.history_file).name.lower()
        if "nasdaq100" not in history_name:
            raise ValueError(
                "Local history_file experiments require an index_events runtime; "
                "automatic conversion is supported only for Nasdaq-100 scripts."
            )
        universe = replace(
            local.universe,
            source="index_events",
            profiles={"US": ("nasdaq100",)},
            history_file="",
            file="",
            symbols=(),
            filters=replace(local.universe.filters, enabled=False),
        )
        local = replace(local, universe=universe, universe_file="", symbols=())
    return local


def load_experiment_market_data(
    config: AppConfig,
    *,
    data_source: str,
    strategy_path: str,
    runtime_path: str,
) -> MarketData:
    if data_source == "yahoo":
        return download_for_config(config)
    if data_source != "local":
        raise ValueError(f"Unsupported data source: {data_source}")

    _, resolver_module = _unified_modules()
    local_config = _local_config(
        config,
        strategy_path=strategy_path,
        runtime_path=runtime_path,
    )
    loaded = resolver_module.download_for_config(local_config, allow_stale=True)
    return MarketData(
        bars=loaded.bars,
        benchmark=loaded.benchmark,
        filter_benchmark=loaded.filter_benchmark,
        skipped=loaded.skipped,
        universe_snapshot=loaded.universe_snapshot,
        universe_schedule=loaded.universe_schedule,
    )


__all__ = ["load_experiment_market_data"]
