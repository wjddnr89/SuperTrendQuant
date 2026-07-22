from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .ranking import validate_scoring_config


_PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT = _PACKAGE_PROJECT_ROOT.parent
DEFAULT_DATA_CONFIG_PATH = _PACKAGE_PROJECT_ROOT / "configs" / "data.yaml"

UNIVERSE_PROFILE_MARKETS: dict[str, str] = {
    "nasdaq100": "US",
    "sp500": "US",
    "russell3000": "US",
    "dow30": "US",
    "kospi200": "KR",
    "kosdaq150": "KR",
}


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
    max_slots: int = 1
    hurdle_atr_mult: float = 1.25
    allow_late_chase: bool = True
    min_rotation_profit_pct: float = 0.0


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
class ScoringConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UniverseFilterConfig:
    enabled: bool = False
    exclude_managed: bool = True
    exclude_suspended: bool = True
    exclude_delisting: bool = True
    exclude_etf_etn: bool = True
    exclude_spac: bool = True
    exclude_preferred: bool = True
    min_price: dict[str, float] = field(
        default_factory=lambda: {"US": 5.0, "KR": 1_000.0}
    )
    avg_turnover_window: int = 20
    min_avg_turnover: dict[str, float] = field(
        default_factory=lambda: {"US": 10_000_000.0, "KR": 1_000_000_000.0}
    )
    min_history_daily_bars: int = 120


@dataclass(frozen=True)
class UniverseConfig:
    source: str = "file"
    profiles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    file: str = "universe.json"
    history_file: str = ""
    symbols: tuple[str, ...] = ()
    refresh: str = "daily"
    snapshot_dir: str = "state/universes"
    filters: UniverseFilterConfig = field(default_factory=UniverseFilterConfig)


@dataclass(frozen=True)
class R2Config:
    enabled: bool = False
    endpoint_env: str = "R2_ENDPOINT_URL"
    bucket: str = ""
    prefix: str = "supertrend-quant"
    region: str = "auto"
    access_key_env: str = "R2_ACCESS_KEY_ID"
    secret_key_env: str = "R2_SECRET_ACCESS_KEY"
    account_id_env: str = "CLOUDFLARE_ACCOUNT_ID"
    api_token_env: str = "CLOUDFLARE_API_TOKEN"
    privacy_attestation_path_env: str = "R2_PRIVACY_ATTESTATION_PATH"
    privacy_attestation_sha256_env: str = "R2_PRIVACY_ATTESTATION_SHA256"
    privacy_attestation_max_age_seconds: int = 900
    jurisdiction: str = "default"


@dataclass(frozen=True)
class DataStoreConfig:
    provider: str = "parquet"
    ingest_source: str = "eodhd"
    auto_sync: bool = False
    price_mode: str = "total_return_adjusted"
    dividend_tax_rate: float = 0.0
    incomplete_action_policy: str = "warn"
    local_cache_dir: str = "data/cache"
    index_source_mode: str = "best_effort"
    publish_enabled: bool = False
    r2: R2Config = field(default_factory=R2Config)

    @property
    def signal_price_mode(self) -> str:
        return self.price_mode

    @property
    def dividend_withholding_rate(self) -> float:
        return self.dividend_tax_rate


@dataclass(frozen=True)
class AppConfig:
    strategy: StrategyIdentity
    scoring: ScoringConfig
    market: str = "US"
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    # Compatibility fields retained for callers that still use replace(...,
    # symbols=...) or the old CLI overrides. New serialization emits only the
    # nested ``universe`` mapping.
    universe_file: str = "universe.json"
    symbols: tuple[str, ...] = ()
    timeframe: str = "1d"
    period: str = "max"
    data_store: DataStoreConfig = field(default_factory=DataStoreConfig)
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
    data_path: str | Path | None = None,
) -> AppConfig:
    strategy_raw = _load_mapping(_resolve_existing_path(strategy_path))
    runtime_raw = _load_mapping(_resolve_existing_path(runtime_path))
    data_raw = _load_mapping(_resolve_existing_path(data_path or DEFAULT_DATA_CONFIG_PATH))
    merged = compose_split_config(strategy_raw, runtime_raw, data_raw)
    return parse_config(merged)


def load_data_store_config(
    path: str | Path | None = None,
    *,
    market: str = "US",
) -> DataStoreConfig:
    """Load the shared market-data configuration without a strategy/runtime."""

    raw = _load_mapping(_resolve_existing_path(path or DEFAULT_DATA_CONFIG_PATH))
    _validate_data_config_schema(raw)
    return _parse_data_store_config(
        {"data_store": _data_store_mapping_for_market(raw, market)}
    )


def compose_split_config(
    strategy_raw: dict[str, Any],
    runtime_raw: dict[str, Any],
    shared_data_raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_strategy_schema(strategy_raw)
    _validate_runtime_schema(runtime_raw)
    if shared_data_raw:
        _validate_data_config_schema(shared_data_raw)
    runtime_data_raw = _optional_mapping(runtime_raw, "data")
    market = str(runtime_raw.get("market", "US")).upper()
    data_store_raw = (
        _data_store_mapping_for_market(shared_data_raw, market)
        if shared_data_raw
        else {}
    )
    portfolio_raw = _optional_mapping(strategy_raw, "portfolio")
    capital_raw = _optional_mapping(runtime_raw, "capital")
    costs_raw = _optional_mapping(runtime_raw, "costs")
    execution_raw = _optional_mapping(runtime_raw, "execution")
    rotation_raw = _optional_mapping(strategy_raw, "rotation")
    scoring_raw = _required_mapping(strategy_raw, "scoring")
    runtime_universe = runtime_raw.get("universe")
    live_raw = _optional_mapping(runtime_raw, "live")
    paper_raw = _optional_mapping(runtime_raw, "paper")
    backtest_raw = _optional_mapping(runtime_raw, "backtest")
    hurdle_raw = rotation_raw.get("hurdle", {})
    if not isinstance(hurdle_raw, dict):
        hurdle_raw = {}
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
        "scoring": {
            "type": str(scoring_raw.get("type") or "").strip(),
            "params": _optional_mapping(scoring_raw, "params"),
        },
        "market": market,
        "universe": (
            dict(runtime_universe)
            if isinstance(runtime_universe, dict)
            else {
                "source": "symbols" if runtime_raw.get("symbols") else "file",
                "file": str(runtime_raw.get("universe_file") or "universe.json"),
                "symbols": runtime_raw.get("symbols", ()) or (),
            }
        ),
        "timeframe": str(runtime_data_raw.get("timeframe") or strategy_raw.get("timeframe") or "1d"),
        "period": str(runtime_data_raw.get("period") or strategy_raw.get("period") or "max"),
        "data_store": dict(data_store_raw),
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
            "max_slots": max_positions,
            "hurdle_atr_mult": hurdle_raw.get("multiplier", 1.25),
            "allow_late_chase": rotation_raw.get("allow_late_chase", True),
            "min_rotation_profit_pct": rotation_raw.get("min_rotation_profit_pct", 0.0),
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

    scoring_raw = _required_mapping(raw, "scoring")
    scoring = ScoringConfig(
        type=str(scoring_raw.get("type") or "").strip(),
        params=dict(_optional_mapping(scoring_raw, "params")),
    )
    market = str(raw.get("market", "US")).upper()
    validate_scoring_config(scoring, market)

    indicators = _optional_mapping(raw, "indicators")
    filters = _optional_mapping(raw, "filters")
    universe_config = _parse_universe_config(raw)
    _validate_universe_market_selection(universe_config, market)
    universe_file = universe_config.file
    symbols = universe_config.symbols

    return AppConfig(
        strategy=strategy,
        scoring=scoring,
        market=market,
        universe=universe_config,
        universe_file=universe_file,
        symbols=symbols,
        timeframe=str(raw.get("timeframe", "1d")),
        period=str(raw.get("period", "max")),
        data_store=_parse_data_store_config(raw),
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


def _parse_universe_config(raw: dict[str, Any]) -> UniverseConfig:
    value = raw.get("universe", {})
    legacy_file = str(raw.get("universe_file") or "universe.json")
    legacy_symbols = tuple(str(symbol) for symbol in raw.get("symbols", ()) or ())
    if isinstance(value, str):
        value = {"source": "symbols" if legacy_symbols else "file", "file": value, "symbols": legacy_symbols}
    elif not isinstance(value, dict):
        raise ValueError("universe must be a string or mapping.")

    _validate_universe_mapping(value)
    symbols = tuple(str(symbol).strip() for symbol in value.get("symbols", legacy_symbols) or () if str(symbol).strip())
    source = str(value.get("source") or ("symbols" if symbols else "file")).strip().lower()
    profiles_raw = value.get("profiles", {}) or {}
    profiles = {
        str(market).upper(): tuple(str(profile).strip().lower() for profile in selected)
        for market, selected in profiles_raw.items()
    }
    filters_raw = value.get("filters", {}) or {}
    filters_enabled = bool(filters_raw.get("enabled", source == "profiles"))
    filters = UniverseFilterConfig(
        enabled=filters_enabled,
        exclude_managed=bool(filters_raw.get("exclude_managed", True)),
        exclude_suspended=bool(filters_raw.get("exclude_suspended", True)),
        exclude_delisting=bool(filters_raw.get("exclude_delisting", True)),
        exclude_etf_etn=bool(filters_raw.get("exclude_etf_etn", True)),
        exclude_spac=bool(filters_raw.get("exclude_spac", True)),
        exclude_preferred=bool(filters_raw.get("exclude_preferred", True)),
        min_price=_market_number_map(
            filters_raw.get("min_price", {"US": 5.0, "KR": 1_000.0}),
            "universe.filters.min_price",
        ),
        avg_turnover_window=_positive_int(
            filters_raw.get("avg_turnover_window", 20),
            "universe.filters.avg_turnover_window",
        ),
        min_avg_turnover=_market_number_map(
            filters_raw.get(
                "min_avg_turnover",
                {"US": 10_000_000.0, "KR": 1_000_000_000.0},
            ),
            "universe.filters.min_avg_turnover",
        ),
        min_history_daily_bars=_positive_int(
            filters_raw.get("min_history_daily_bars", 120),
            "universe.filters.min_history_daily_bars",
        ),
    )
    config = UniverseConfig(
        source=source,
        profiles=profiles,
        file=str(value.get("file") or legacy_file),
        history_file=str(value.get("history_file") or ""),
        symbols=symbols,
        refresh=str(value.get("refresh") or "daily").strip().lower(),
        snapshot_dir=str(value.get("snapshot_dir") or "state/universes"),
        filters=filters,
    )
    _validate_universe_config(config)
    return config


def _parse_data_store_config(raw: dict[str, Any]) -> DataStoreConfig:
    value = raw.get("data_store", {}) or {}
    if not isinstance(value, dict):
        raise ValueError("data_store must be a mapping.")
    _validate_data_store_mapping(value)
    r2_raw = value.get("r2", {}) or {}
    _reject_conflicting_aliases(value, "signal_price_mode", "price_mode")
    _reject_conflicting_aliases(
        value,
        "dividend_withholding_rate",
        "dividend_tax_rate",
    )
    config = DataStoreConfig(
        provider=str(value.get("provider", "parquet")).strip().lower(),
        ingest_source=str(value.get("ingest_source", "eodhd")).strip().lower(),
        auto_sync=bool(value.get("auto_sync", False)),
        price_mode=str(
            value.get("signal_price_mode", value.get("price_mode", "total_return_adjusted"))
        ).strip().lower(),
        dividend_tax_rate=float(
            value.get("dividend_withholding_rate", value.get("dividend_tax_rate", 0.0))
        ),
        incomplete_action_policy=str(value.get("incomplete_action_policy", "warn")).strip().lower(),
        local_cache_dir=str(value.get("local_cache_dir", "data/cache")),
        index_source_mode=str(value.get("index_source_mode", "best_effort")).strip().lower(),
        publish_enabled=bool(value.get("publish_enabled", False)),
        r2=R2Config(
            enabled=bool(r2_raw.get("enabled", False)),
            endpoint_env=str(r2_raw.get("endpoint_env", "R2_ENDPOINT_URL")),
            bucket=str(r2_raw.get("bucket", "")),
            prefix=str(r2_raw.get("prefix", "supertrend-quant")).strip("/"),
            region=str(r2_raw.get("region", "auto")),
            access_key_env=str(r2_raw.get("access_key_env", "R2_ACCESS_KEY_ID")),
            secret_key_env=str(r2_raw.get("secret_key_env", "R2_SECRET_ACCESS_KEY")),
            account_id_env=str(
                r2_raw.get("account_id_env", "CLOUDFLARE_ACCOUNT_ID")
            ),
            api_token_env=str(
                r2_raw.get("api_token_env", "CLOUDFLARE_API_TOKEN")
            ),
            privacy_attestation_path_env=str(
                r2_raw.get(
                    "privacy_attestation_path_env",
                    "R2_PRIVACY_ATTESTATION_PATH",
                )
            ),
            privacy_attestation_sha256_env=str(
                r2_raw.get(
                    "privacy_attestation_sha256_env",
                    "R2_PRIVACY_ATTESTATION_SHA256",
                )
            ),
            privacy_attestation_max_age_seconds=int(
                r2_raw.get("privacy_attestation_max_age_seconds", 900)
            ),
            jurisdiction=str(r2_raw.get("jurisdiction", "default"))
            .strip()
            .lower(),
        ),
    )
    if config.provider not in {"parquet", "yahoo"}:
        raise ValueError("data_store.provider must be parquet or yahoo.")
    if config.ingest_source not in {"eodhd", "yahoo"}:
        raise ValueError("data_store.ingest_source must be eodhd or yahoo.")
    if config.price_mode not in {"total_return_adjusted", "split_adjusted", "raw"}:
        raise ValueError(
            "data_store.price_mode must be total_return_adjusted, split_adjusted, or raw."
        )
    if not 0.0 <= config.dividend_tax_rate <= 1.0:
        raise ValueError("data_store.dividend_tax_rate must be between 0 and 1.")
    if config.incomplete_action_policy not in {"warn", "block"}:
        raise ValueError("data_store.incomplete_action_policy must be warn or block.")
    if config.index_source_mode not in {"best_effort", "official_only"}:
        raise ValueError("data_store.index_source_mode must be best_effort or official_only.")
    if config.r2.enabled and (not config.r2.endpoint_env or not config.r2.bucket):
        raise ValueError("data_store.r2 endpoint_env and bucket are required when enabled.")
    if config.r2.jurisdiction not in {"default", "eu", "fedramp"}:
        raise ValueError("data_store.r2.jurisdiction must be default, eu, or fedramp.")
    if not 60 <= config.r2.privacy_attestation_max_age_seconds <= 3600:
        raise ValueError(
            "data_store.r2.privacy_attestation_max_age_seconds must be between "
            "60 and 3600."
        )
    r2_env_names = (
        config.r2.access_key_env,
        config.r2.secret_key_env,
        config.r2.account_id_env,
        config.r2.api_token_env,
        config.r2.privacy_attestation_path_env,
        config.r2.privacy_attestation_sha256_env,
    )
    if config.r2.enabled and any(not value.strip() for value in r2_env_names):
        raise ValueError("data_store.r2 environment variable names must not be empty.")
    return config


def _validate_data_store_mapping(raw: dict[str, Any]) -> None:
    _reject_unknown_keys(
        raw,
        {
            "provider",
            "ingest_source",
            "auto_sync",
            "signal_price_mode",
            "dividend_withholding_rate",
            "price_mode",
            "dividend_tax_rate",
            "incomplete_action_policy",
            "local_cache_dir",
            "index_source_mode",
            "publish_enabled",
            "r2",
        },
        "data_store",
    )
    r2 = raw.get("r2", {}) or {}
    if not isinstance(r2, dict):
        raise ValueError("data_store.r2 must be a mapping.")
    _reject_unknown_keys(
        r2,
        {
            "enabled",
            "endpoint_env",
            "bucket",
            "prefix",
            "region",
            "access_key_env",
            "secret_key_env",
            "account_id_env",
            "api_token_env",
            "privacy_attestation_path_env",
            "privacy_attestation_sha256_env",
            "privacy_attestation_max_age_seconds",
            "jurisdiction",
        },
        "data_store.r2",
    )


def _reject_conflicting_aliases(raw: dict[str, Any], canonical: str, legacy: str) -> None:
    if canonical in raw and legacy in raw and raw[canonical] != raw[legacy]:
        raise ValueError(
            f"data_store.{canonical} conflicts with legacy data_store.{legacy}."
        )


def _validate_universe_mapping(raw: dict[str, Any]) -> None:
    _reject_unknown_keys(
        raw,
        {
            "source",
            "profiles",
            "file",
            "history_file",
            "symbols",
            "refresh",
            "snapshot_dir",
            "filters",
        },
        "universe",
    )
    profiles = raw.get("profiles", {}) or {}
    if not isinstance(profiles, dict):
        raise ValueError("universe.profiles must be a mapping.")
    for market, selected in profiles.items():
        if str(market).upper() not in {"US", "KR"}:
            raise ValueError(f"Unsupported universe profile market: {market}")
        if not isinstance(selected, (list, tuple)):
            raise ValueError(f"universe.profiles.{market} must be a list.")
    symbols = raw.get("symbols", ()) or ()
    if not isinstance(symbols, (list, tuple)):
        raise ValueError("universe.symbols must be a list.")
    filters = raw.get("filters", {}) or {}
    if not isinstance(filters, dict):
        raise ValueError("universe.filters must be a mapping.")
    _reject_unknown_keys(
        filters,
        {
            "enabled",
            "exclude_managed",
            "exclude_suspended",
            "exclude_delisting",
            "exclude_etf_etn",
            "exclude_spac",
            "exclude_preferred",
            "min_price",
            "avg_turnover_window",
            "min_avg_turnover",
            "min_history_daily_bars",
        },
        "universe.filters",
    )


def _validate_universe_config(config: UniverseConfig) -> None:
    if config.source not in {"profiles", "file", "symbols", "history_file", "index_events"}:
        raise ValueError(
            "universe.source must be profiles, file, symbols, history_file, or index_events."
        )
    if config.refresh != "daily":
        raise ValueError("universe.refresh must be daily.")
    if config.source == "profiles" and not any(config.profiles.values()):
        raise ValueError("universe.source=profiles requires at least one profile.")
    if config.source == "symbols" and not config.symbols:
        raise ValueError("universe.source=symbols requires universe.symbols.")
    if config.source == "file" and not config.file.strip():
        raise ValueError("universe.source=file requires universe.file.")
    if config.source == "history_file" and not config.history_file.strip():
        raise ValueError("universe.source=history_file requires universe.history_file.")
    if config.source == "history_file" and config.filters.enabled:
        raise ValueError("universe.filters are not supported with universe.source=history_file.")
    if config.source == "index_events" and not any(config.profiles.values()):
        raise ValueError("universe.source=index_events requires at least one profile.")
    if config.source == "index_events" and config.filters.enabled:
        raise ValueError("universe.filters are not supported with universe.source=index_events.")
    for market, selected in config.profiles.items():
        seen: set[str] = set()
        for profile in selected:
            expected_market = UNIVERSE_PROFILE_MARKETS.get(profile)
            if expected_market is None:
                available = ", ".join(sorted(UNIVERSE_PROFILE_MARKETS))
                raise ValueError(f"Unsupported universe profile: {profile}. Available profiles: {available}")
            if expected_market != market:
                raise ValueError(f"Universe profile {profile} belongs to {expected_market}, not {market}.")
            if profile in seen:
                raise ValueError(f"Duplicate universe profile for {market}: {profile}")
            seen.add(profile)


def _validate_universe_market_selection(config: UniverseConfig, market: str) -> None:
    if config.source not in {"profiles", "index_events"} or market == "AUTO":
        return
    if market not in {"US", "KR"}:
        raise ValueError("market must be US, KR, or AUTO.")
    configured_markets = {key for key, profiles in config.profiles.items() if profiles}
    if configured_markets != {market}:
        raise ValueError(
            f"market={market} requires universe.profiles to contain only {market}."
        )


def _market_number_map(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    unknown = {str(key).upper() for key in value} - {"US", "KR"}
    if unknown:
        raise ValueError(f"Unsupported market keys for {label}: {', '.join(sorted(unknown))}")
    parsed = {"US": 0.0, "KR": 0.0}
    for key, raw_number in value.items():
        try:
            number = float(raw_number)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.{key} must be a non-negative number.") from exc
        if number < 0:
            raise ValueError(f"{label}.{key} must be a non-negative number.")
        parsed[str(key).upper()] = number
    return parsed


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer.") from exc
    if parsed < 1 or parsed != value:
        raise ValueError(f"{label} must be a positive integer.")
    return parsed


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
    _reject_unknown_keys(
        raw,
        {"name", "type", "params", "portfolio", "scoring", "signals", "rotation", "timeframe", "period"},
        "strategy",
    )
    _optional_mapping(raw, "params")
    _reject_unknown_keys(_optional_mapping(raw, "portfolio"), {"max_positions", "allocation_pct"}, "strategy.portfolio")
    scoring = _required_mapping(raw, "scoring")
    _reject_unknown_keys(scoring, {"type", "params"}, "strategy.scoring")
    if not str(scoring.get("type") or "").strip():
        raise ValueError("strategy.scoring.type is required.")
    _optional_mapping(scoring, "params")
    rotation = _optional_mapping(raw, "rotation")
    _reject_unknown_keys(rotation, {"hurdle", "allow_late_chase", "min_rotation_profit_pct"}, "strategy.rotation")
    hurdle = rotation.get("hurdle", {})
    if hurdle is not None:
        if not isinstance(hurdle, dict):
            raise ValueError("strategy.rotation.hurdle must be a mapping.")
        _reject_unknown_keys(hurdle, {"multiplier"}, "strategy.rotation.hurdle")


def _validate_data_config_schema(raw: dict[str, Any]) -> None:
    _reject_unknown_keys(raw, {"data_store", "market_overrides"}, "data config")
    data_store = _required_mapping(raw, "data_store")
    _validate_data_store_mapping(data_store)
    overrides = _optional_mapping(raw, "market_overrides")
    for market, override in overrides.items():
        normalized = str(market).upper()
        if normalized not in {"US", "KR", "AUTO"}:
            raise ValueError(f"Unsupported data config market override: {market}")
        if not isinstance(override, dict):
            raise ValueError(f"data config market_overrides.{market} must be a mapping.")
        _validate_data_store_mapping(override)


def _data_store_mapping_for_market(raw: dict[str, Any], market: str) -> dict[str, Any]:
    base = _required_mapping(raw, "data_store")
    overrides = _optional_mapping(raw, "market_overrides")
    selected = overrides.get(str(market).upper())
    if selected is None:
        return dict(base)
    if not isinstance(selected, dict):
        raise ValueError(f"data config market_overrides.{market} must be a mapping.")
    return dict(selected)


def _validate_runtime_schema(raw: dict[str, Any]) -> None:
    _reject_unknown_keys(
        raw,
        {
            "name",
            "market",
            "universe",
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
    universe = raw.get("universe")
    if universe is not None:
        if not isinstance(universe, dict):
            raise ValueError("runtime.universe must be a mapping.")
        _validate_universe_mapping(universe)
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
    from .universe import resolve_universe

    return list(resolve_universe(config, mode="backtest").eligible_symbols)


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
