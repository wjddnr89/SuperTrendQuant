from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import AppConfig
from ..data import MarketData
from .adjustments import apply_adjustment_factors
from .models import DataQuality
from .preflight import DailyPreflight
from .repository import LocalDatasetRepository
from .schemas import DATASET_SPECS
from .storage import DatasetCache, ObjectNotFound, R2ObjectStore, publish_repository


INDEX_BENCHMARK_ETFS = {
    "sp500": "SPY",
    "nasdaq100": "QQQ",
    "russell3000": "IWV",
}

REQUIRED_SYNC_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
)


class ParquetMarketDataProvider:
    """Query immutable Parquet versions with DuckDB and convert once to pandas."""

    def __init__(self, root: str | Path, *, capture_query_plans: bool = False):
        self.root = Path(root)
        self.repository = LocalDatasetRepository(self.root)
        self._release_versions: dict[str, str] = {}
        self._has_release = False
        self.capture_query_plans = capture_query_plans
        self.query_counts: dict[str, int] = {}
        self.query_files: dict[str, tuple[str, ...]] = {}
        self.query_plans: dict[str, str] = {}

    def load(self, config: AppConfig, symbols: list[str]) -> MarketData:
        if config.timeframe != "1d":
            raise ValueError("ParquetMarketDataProvider V1 supports timeframe=1d only.")
        release, _ = self.repository.current_release()
        self._has_release = release is not None
        self._release_versions = dict(release.dataset_versions) if release is not None else {}
        requested = tuple(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))
        benchmark_symbol = _benchmark_symbol(config)
        all_symbols = tuple(dict.fromkeys((*requested, benchmark_symbol)))
        symbol_map = self._resolve_security_ids(all_symbols)
        missing_symbols = tuple(symbol for symbol in requested if symbol not in symbol_map)
        security_ids = tuple(dict.fromkeys(symbol_map.values()))
        price_manifest = self._manifest("daily_price_raw")
        completed_boundary = (
            release.completed_session
            if release is not None
            else (price_manifest.completed_session if price_manifest is not None else "")
        )
        period_start, period_end = _period_bounds(config.period, completed_boundary)
        raw = self._query_dataset(
            "daily_price_raw",
            where_column="security_id",
            values=security_ids,
            columns=("security_id", "session", "open", "high", "low", "close", "volume"),
            date_column="session",
            min_session=period_start,
            max_session=period_end,
        )
        if raw.empty:
            raise ValueError("daily_price_raw has no rows for the requested securities.")
        factors = self._query_dataset(
            "adjustment_factors",
            where_column="security_id",
            values=security_ids,
            columns=("security_id", "session", "split_factor", "total_return_factor"),
            required=False,
            date_column="session",
            min_session=period_start,
            max_session=period_end,
        )
        actions = self._query_dataset(
            "corporate_actions",
            where_column="security_id",
            values=security_ids,
            required=False,
            date_column="effective_date",
            min_session=period_start,
            max_session=period_end,
        )
        raw = _filter_period(raw, config.period)
        factors = _filter_period(factors, config.period) if not factors.empty else factors
        adjusted, quality, warnings = self._adjust(config, raw, factors)
        if not actions.empty:
            symbol_by_security = {security_id: symbol for symbol, security_id in symbol_map.items()}
            actions["symbol"] = actions["security_id"].astype(str).map(symbol_by_security).fillna("")
        execution_bars = _frames_by_requested_symbol(raw, requested, symbol_map)
        signal_bars = _frames_by_requested_symbol(adjusted, requested, symbol_map)
        missing_symbols = tuple(
            sorted(set(missing_symbols) | (set(requested) - set(signal_bars)))
        )
        if missing_symbols:
            quality = DataQuality.BLOCKED
            warnings.append(
                "Selected universe members have no usable daily prices: "
                + ", ".join(missing_symbols)
            )

        benchmark: dict[str, pd.DataFrame] | None = None
        if benchmark_symbol in symbol_map:
            benchmark_frame = _frame_for_security(adjusted, symbol_map[benchmark_symbol])
            if not benchmark_frame.empty:
                benchmark = {symbol: benchmark_frame.copy() for symbol in requested}
        else:
            warnings.append(f"Benchmark ETF is missing from symbol history: {benchmark_symbol}")
            quality = DataQuality.DEGRADED

        versions: dict[str, str] = {}
        completed_sessions: list[str] = []
        if release is not None:
            versions["release"] = release.version
            if release.quality == DataQuality.BLOCKED:
                quality = DataQuality.BLOCKED
            elif release.quality != DataQuality.VALID and quality != DataQuality.BLOCKED:
                quality = DataQuality.DEGRADED
            warnings.extend(release.warnings)
        for dataset in ("daily_price_raw", "adjustment_factors", "corporate_actions", "symbol_history"):
            version = self._release_versions.get(dataset)
            manifest = (
                self.repository.manifest_for_version(dataset, version)
                if version
                else (None if self._has_release else self.repository.current_manifest(dataset))
            )
            if manifest is not None:
                versions[dataset] = manifest.version
                if manifest.completed_session:
                    completed_sessions.append(manifest.completed_session)
                if manifest.quality != DataQuality.VALID:
                    if quality != DataQuality.BLOCKED:
                        quality = DataQuality.DEGRADED
                    warnings.extend(manifest.warnings)
        completed_session = (
            release.completed_session
            if release is not None
            else (min(completed_sessions) if completed_sessions else "")
        )
        if completed_session:
            stale = tuple(
                symbol
                for symbol, frame in signal_bars.items()
                if frame.empty
                or pd.Timestamp(frame.index[-1]).date()
                < pd.Timestamp(completed_session).date()
            )
            if stale:
                quality = DataQuality.BLOCKED
                warnings.append(
                    f"Selected universe is incomplete through {completed_session}: "
                    + ", ".join(stale)
                )
        return MarketData(
            bars=signal_bars,
            execution_bars=execution_bars,
            benchmark=benchmark,
            filter_benchmark=benchmark,
            benchmark_symbol=benchmark_symbol,
            corporate_actions=tuple(actions.to_dict("records")) if not actions.empty else (),
            data_version=";".join(f"{name}={version}" for name, version in sorted(versions.items())),
            completed_session=completed_session,
            data_quality=str(quality),
            warnings=tuple(dict.fromkeys(warnings)),
            skipped=missing_symbols,
        )

    def _resolve_security_ids(self, symbols: tuple[str, ...]) -> dict[str, str]:
        history = self._query_dataset(
            "symbol_history",
            where_column="symbol",
            values=symbols,
            columns=("security_id", "symbol", "effective_from", "effective_to"),
        )
        if history.empty:
            return {}
        history["_start"] = pd.to_datetime(history["effective_from"], errors="coerce")
        history["_end"] = pd.to_datetime(history["effective_to"], errors="coerce")
        history = history.sort_values(["symbol", "_end", "_start"], na_position="last")
        latest = history.groupby("symbol", sort=False).tail(1)
        return dict(zip(latest["symbol"].astype(str), latest["security_id"].astype(str)))

    def _query_dataset(
        self,
        dataset: str,
        *,
        where_column: str,
        values: tuple[str, ...],
        columns: tuple[str, ...] = (),
        required: bool = True,
        date_column: str = "",
        min_session: str = "",
        max_session: str = "",
    ) -> pd.DataFrame:
        if not values:
            return pd.DataFrame()
        version = self._release_versions.get(dataset)
        manifest = (
            self.repository.manifest_for_version(dataset, version)
            if version
            else (None if self._has_release else self.repository.current_manifest(dataset))
        )
        if manifest is None:
            if required:
                raise ValueError(f"Required Parquet dataset is missing: {dataset}")
            return pd.DataFrame()
        try:
            import duckdb
        except ModuleNotFoundError as exc:
            raise RuntimeError("duckdb is required for Parquet market-data queries.") from exc
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", where_column):
            raise ValueError(f"Unsafe column name: {where_column}")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column) for column in columns):
            raise ValueError("Unsafe projected column name.")
        if date_column and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", date_column):
            raise ValueError(f"Unsafe date column name: {date_column}")
        paths = [
            str(path)
            for path in self.repository.parquet_paths(
                dataset,
                manifest.version,
                min_session=min_session,
                max_session=max_session,
            )
        ]
        if not paths:
            return pd.DataFrame()
        path_sql = ", ".join(_sql_string(path) for path in paths)
        placeholders = ", ".join("?" for _ in values)
        projection = ", ".join(columns) if columns else "*"
        predicates = [f"{where_column} IN ({placeholders})"]
        parameters: list[str] = list(values)
        if date_column and min_session:
            predicates.append(f"CAST({date_column} AS DATE) >= CAST(? AS DATE)")
            parameters.append(min_session)
        if date_column and max_session:
            predicates.append(f"CAST({date_column} AS DATE) <= CAST(? AS DATE)")
            parameters.append(max_session)
        query = (
            f"SELECT {projection} FROM read_parquet([{path_sql}], hive_partitioning=false) "
            f"WHERE {' AND '.join(predicates)}"
        )
        self.query_counts[dataset] = self.query_counts.get(dataset, 0) + 1
        self.query_files[dataset] = tuple(paths)
        connection = duckdb.connect(database=":memory:")
        try:
            if self.capture_query_plans:
                explained = connection.execute(f"EXPLAIN {query}", parameters).fetchall()
                self.query_plans[dataset] = "\n".join(str(row[-1]) for row in explained)
            return connection.execute(query, parameters).fetch_df()
        finally:
            connection.close()

    def _manifest(self, dataset: str):
        version = self._release_versions.get(dataset)
        if version:
            return self.repository.manifest_for_version(dataset, version)
        if self._has_release:
            return None
        return self.repository.current_manifest(dataset)

    @staticmethod
    def _adjust(
        config: AppConfig,
        raw: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> tuple[pd.DataFrame, DataQuality, list[str]]:
        if config.data_store.price_mode == "raw":
            return raw.copy(), DataQuality.VALID, []
        if factors.empty:
            warning = "Adjustment factors are missing; signal data cannot be constructed."
            if config.data_store.incomplete_action_policy == "block":
                raise ValueError(warning)
            return raw.copy(), DataQuality.DEGRADED, [warning]
        return (
            apply_adjustment_factors(raw, factors, mode=config.data_store.price_mode),
            DataQuality.VALID,
            [],
        )


def _benchmark_symbol(config: AppConfig) -> str:
    profiles = config.universe.profiles.get("US", ())
    for profile in profiles:
        if profile in INDEX_BENCHMARK_ETFS:
            return INDEX_BENCHMARK_ETFS[profile]
    return "QQQ"


def _filter_period(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    if frame.empty or period == "max" or "session" not in frame:
        return frame.copy()
    match = re.fullmatch(r"(\d+)(d|mo|y)", period.strip().lower())
    if not match:
        raise ValueError(f"Unsupported period for Parquet data: {period}")
    amount = int(match.group(1))
    unit = match.group(2)
    sessions = pd.to_datetime(frame["session"], errors="coerce")
    end = sessions.max()
    if unit == "d":
        start = end - pd.Timedelta(days=amount)
    elif unit == "mo":
        start = end - pd.DateOffset(months=amount)
    else:
        start = end - pd.DateOffset(years=amount)
    return frame.loc[sessions >= start].copy()


def _period_bounds(period: str, completed_session: str) -> tuple[str, str]:
    """Translate a public period into an ISO range usable for file and SQL pruning."""

    if not completed_session:
        return "", ""
    end = pd.Timestamp(completed_session)
    if period == "max":
        return "", end.date().isoformat()
    match = re.fullmatch(r"(\d+)(d|mo|y)", period.strip().lower())
    if not match:
        raise ValueError(f"Unsupported period for Parquet data: {period}")
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "d":
        start = end - pd.Timedelta(days=amount)
    elif unit == "mo":
        start = end - pd.DateOffset(months=amount)
    else:
        start = end - pd.DateOffset(years=amount)
    return start.date().isoformat(), end.date().isoformat()


def _frames_by_requested_symbol(
    frame: pd.DataFrame,
    symbols: tuple[str, ...],
    symbol_map: dict[str, str],
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        security_id = symbol_map.get(symbol)
        if security_id is None:
            continue
        value = _frame_for_security(frame, security_id)
        if not value.empty:
            output[symbol] = value
    return output


def _frame_for_security(frame: pd.DataFrame, security_id: str) -> pd.DataFrame:
    value = frame.loc[frame["security_id"].astype(str) == security_id].copy()
    if value.empty:
        return pd.DataFrame()
    value["session"] = pd.to_datetime(value["session"])
    value = value.sort_values("session").set_index("session")
    columns = [column for column in ("open", "high", "low", "close", "volume") if column in value]
    value = value[columns].rename(columns={column: column.title() for column in columns})
    value.index.name = "Date"
    return value


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def ensure_configured_data_ready(
    config: AppConfig,
    *,
    force_sync: bool = False,
) -> None:
    """Run the shared preflight before universe resolution or market-data loading."""
    if config.data_store.provider == "yahoo" or config.market == "KR":
        return
    repository = LocalDatasetRepository(config.data_store.local_cache_dir)
    release, _ = repository.current_release()
    price_manifest = repository.current_manifest("daily_price_raw")
    completed = (
        release.completed_session
        if release is not None
        else (price_manifest.completed_session if price_manifest is not None else "")
    )

    def sync_callback(expected: str) -> str:
        remote_store = None
        if config.data_store.r2.enabled:
            remote_store = R2ObjectStore(config.data_store.r2)
            cache = DatasetCache(
                config.data_store.local_cache_dir,
                remote_store,
            )
            try:
                cache.sync_release()
            except ObjectNotFound:
                datasets = list(REQUIRED_SYNC_DATASETS)
                if config.universe.source == "index_events":
                    datasets.extend(("index_constituent_anchors", "index_membership_events"))
                for dataset in datasets:
                    try:
                        cache.sync(dataset)
                    except ObjectNotFound:
                        continue
            updated_release, _ = repository.current_release()
            updated = repository.current_manifest("daily_price_raw")
            remote_completed = (
                updated_release.completed_session
                if updated_release is not None
                else (updated.completed_session if updated is not None else "")
            )
            if remote_completed >= expected:
                return remote_completed

        # R2 may itself be stale (or disabled). The same preflight then performs
        # one provider sync for this expected session and validates a new local
        # cross-dataset release before consumers are allowed to proceed.
        from .ingest import DailyDataSynchronizer

        synced = DailyDataSynchronizer(repository).sync(
            expected,
            refresh_security_master=True,
        )
        if remote_store is not None and config.data_store.publish_enabled:
            published = publish_repository(repository, remote_store, tuple(DATASET_SPECS))
            conflicts = [item for item in published if item.conflict]
            if conflicts:
                details = "; ".join(item.detail for item in conflicts)
                raise RuntimeError(f"R2 publication conflict: {details}")
        return synced.completed_session

    preflight = DailyPreflight(Path(config.data_store.local_cache_dir) / "state" / "preflight.json")
    result = preflight.run(
        completed,
        auto_sync=config.data_store.auto_sync,
        sync=sync_callback,
        force=force_sync,
    )
    if not result.ready:
        raise RuntimeError(result.warning + " Run quant-data sync or provide --force for a manual sync.")


def load_configured_market_data(
    config: AppConfig,
    symbols: list[str],
    *,
    resolved_universe=None,
    force_sync: bool = False,
) -> MarketData:
    """Load the configured source and enforce daily Parquet freshness."""
    if config.data_store.provider == "yahoo" or config.market == "KR":
        from ..data import download_market_data

        return download_market_data(config, symbols, resolved_universe=resolved_universe)
    ensure_configured_data_ready(config, force_sync=force_sync)
    market_data = ParquetMarketDataProvider(config.data_store.local_cache_dir).load(config, symbols)
    if resolved_universe is None:
        return market_data
    return replace(
        market_data,
        universe_snapshot=resolved_universe.snapshot.to_dict(),
        universe_schedule=resolved_universe.schedule_as_dicts(),
    )
