from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from supertrend_quant.config import AppConfig, ScoringConfig, StrategyIdentity
from supertrend_quant.data import MarketData
from supertrend_quant.market_store import provider as provider_module
from supertrend_quant.research import data_resolver


def _minimal_config() -> AppConfig:
    return AppConfig(
        strategy=StrategyIdentity(name="test", type="leader_rotation"),
        scoring=ScoringConfig(type="relative_strength", params={"lookback_bars": 1}),
        symbols=("AAA",),
    )


class ResearchStaleCacheBoundaryTest(unittest.TestCase):
    def test_research_resolver_propagates_stale_choice_without_changing_default(self):
        config = _minimal_config()
        resolved = SimpleNamespace(eligible_symbols=("AAA",), schedule=())
        loaded = MarketData(bars={})

        with (
            patch.object(data_resolver, "ensure_configured_data_ready") as preflight,
            patch.object(data_resolver, "resolve_universe", return_value=resolved),
            patch.object(
                data_resolver,
                "load_configured_market_data",
                return_value=loaded,
            ) as loader,
        ):
            self.assertIs(data_resolver.download_for_config(config, allow_stale=True), loaded)
            preflight.assert_not_called()
            loader.assert_called_once_with(
                config,
                ["AAA"],
                resolved_universe=resolved,
                allow_stale=True,
            )

        with (
            patch.object(data_resolver, "ensure_configured_data_ready") as preflight,
            patch.object(data_resolver, "resolve_universe", return_value=resolved),
            patch.object(data_resolver, "load_configured_market_data", return_value=loaded),
        ):
            data_resolver.download_for_config(config)
            preflight.assert_called_once_with(config)

    def test_configured_provider_skips_only_freshness_when_explicitly_allowed(self):
        config = _minimal_config()
        loaded = MarketData(bars={"AAA": pd.DataFrame({"Close": [1.0]})})

        with (
            patch.object(provider_module, "ensure_configured_data_ready") as preflight,
            patch.object(
                provider_module.ParquetMarketDataProvider,
                "load",
                return_value=loaded,
            ),
        ):
            self.assertIs(
                provider_module.load_configured_market_data(
                    config,
                    ["AAA"],
                    allow_stale=True,
                ),
                loaded,
            )
            preflight.assert_not_called()

        with (
            patch.object(provider_module, "ensure_configured_data_ready") as preflight,
            patch.object(
                provider_module.ParquetMarketDataProvider,
                "load",
                return_value=loaded,
            ),
        ):
            provider_module.load_configured_market_data(config, ["AAA"])
            preflight.assert_called_once_with(config, force_sync=False)


if __name__ == "__main__":
    unittest.main()
