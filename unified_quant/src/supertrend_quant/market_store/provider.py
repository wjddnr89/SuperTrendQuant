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

# These actions can create a position whose security was never an index member
# (most notably a spin-off child).  Market-data loading follows the exact
# successor IDs recursively, while entry eligibility remains the originally
# requested universe.
POSITION_CREATING_ACTIONS = frozenset({"spinoff", "stock_merger", "ticker_change"})
OLD_SYMBOL_ACTIONS = frozenset(
    {
        "cash_dividend",
        "special_dividend",
        "split",
        "stock_dividend",
        "capital_reduction",
        "spinoff",
        "stock_merger",
        "ticker_change",
        "cash_merger",
        "delisting",
    }
)
# Corporate-action dates can fall on the next trading session even though the
# old symbol's final price/history date is the preceding Friday (or the session
# before a longer US-market closure).  Keep the same narrow calendar bound used
# by ``_actions_in_symbol_intervals`` so the terminal event is not discarded by
# the earlier requested-interval filter.
OLD_SYMBOL_ACTION_MAX_CALENDAR_GAP = pd.Timedelta(days=10)

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
        self._resolved_symbol_history = pd.DataFrame()

    def load(
        self,
        config: AppConfig,
        symbols: list[str],
        *,
        universe_schedule: tuple[dict[str, Any], ...] = (),
    ) -> MarketData:
        if config.timeframe != "1d":
            raise ValueError("ParquetMarketDataProvider V1 supports timeframe=1d only.")
        release, _ = self.repository.current_release()
        self._has_release = release is not None
        self._release_versions = dict(release.dataset_versions) if release is not None else {}
        requested = tuple(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))
        benchmark_symbol = _benchmark_symbol(config)
        all_symbols = tuple(dict.fromkeys((*requested, benchmark_symbol)))
        symbol_map = self._resolve_security_ids(all_symbols)
        requested_intervals = _requested_identity_intervals(
            requested,
            symbol_map,
            universe_schedule=universe_schedule,
            index_event_universe=config.universe.source == "index_events",
        )
        missing_symbols = tuple(
            symbol for symbol in requested if symbol not in requested_intervals
        )
        requested_security_ids = tuple(
            dict.fromkeys(
                interval[0]
                for intervals in requested_intervals.values()
                for interval in intervals
            )
        )
        benchmark_security_ids = symbol_map.get(benchmark_symbol, ())
        if len(benchmark_security_ids) > 1:
            raise ValueError(
                "Benchmark ticker resolves to multiple issuer identities: "
                f"{benchmark_symbol} -> {', '.join(benchmark_security_ids)}"
            )
        # The initial lookup is symbol-scoped, so a current ticker such as
        # FBIN does not reveal its immediately preceding FBHS interval.  Load
        # the complete identities before assigning actions to the symbol that
        # actually held the entitlement.
        self._append_symbol_history_for_ids(requested_security_ids)
        closure_intervals = (
            self._identity_history_intervals(
                {
                    symbol: tuple(
                        dict.fromkeys(interval[0] for interval in intervals)
                    )
                    for symbol, intervals in requested_intervals.items()
                }
            )
            if config.universe.source == "index_events"
            else requested_intervals
        )
        price_manifest = self._manifest("daily_price_raw")
        completed_boundary = (
            release.completed_session
            if release is not None
            else (price_manifest.completed_session if price_manifest is not None else "")
        )
        period_start, period_end = _period_bounds(config.period, completed_boundary)
        (
            actions,
            linked_symbols,
            managed_security_ids,
            linked_symbol_ids,
        ) = self._action_linked_closure(
            requested,
            requested_security_ids,
            period_start=period_start,
            period_end=period_end,
            requested_intervals=closure_intervals,
        )
        successor_actions = actions
        if config.universe.source != "index_events" and not actions.empty:
            actions = _collapse_static_action_aliases(actions, requested_intervals)
        symbol_map = self._symbol_map_from_resolved_history()
        loaded_symbols = tuple(dict.fromkeys((*requested, *linked_symbols)))
        requested_ids_by_symbol = {
            symbol: tuple(dict.fromkeys(interval[0] for interval in intervals))
            for symbol, intervals in requested_intervals.items()
        }
        if config.universe.source == "index_events":
            frame_intervals = {}
            for symbol, security_ids_for_symbol in requested_ids_by_symbol.items():
                if len(security_ids_for_symbol) == 1:
                    # One issuer changing aliases should retain causal feature
                    # warm-up from its earlier ticker history.
                    frame_intervals.update(
                        self._identity_feature_intervals(
                            {symbol: security_ids_for_symbol}
                        )
                    )
                else:
                    # A reused display ticker must remain schedule-bounded;
                    # expanding both issuers can overlap and splice histories.
                    frame_intervals[symbol] = requested_intervals[symbol]
        else:
            frame_intervals = dict(requested_intervals)
        # A lifecycle successor can also be an independently requested index
        # member (for example, a spin-off child added the following session).
        # Keep its exact action-linked identity interval in addition to any
        # requested intervals instead of letting one role overwrite the other.
        for symbol, intervals in self._action_successor_intervals(
            successor_actions
        ).items():
            if symbol in requested and len(requested_ids_by_symbol.get(symbol, ())) == 1:
                # The single-ID requested frame already includes prior aliases
                # and the successor's complete symbol lifetime.
                continue
            frame_intervals[symbol] = tuple(
                dict.fromkeys((*frame_intervals.get(symbol, ()), *intervals))
            )
        security_ids = tuple(
            dict.fromkeys((*managed_security_ids, *benchmark_security_ids))
        )
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
        raw = _filter_period(raw, config.period)
        factors = _filter_period(factors, config.period) if not factors.empty else factors
        # Validate every exact action successor, including successors which
        # also appear in the requested universe.  Otherwise a missing factor
        # on the action date can silently default to 1.0 for a requested child.
        linked_security_ids = {
            security_id
            for security_ids in linked_symbol_ids.values()
            for security_id in security_ids
        }
        if config.data_store.price_mode != "raw" and linked_security_ids:
            factor_ids = set(factors["security_id"].astype(str)) if not factors.empty else set()
            missing_linked_factors = sorted(linked_security_ids - factor_ids)
            if missing_linked_factors:
                raise ValueError(
                    "Action-linked securities have no adjustment factors: "
                    + ", ".join(missing_linked_factors)
                )
            linked_raw = raw.loc[
                raw["security_id"].astype(str).isin(linked_security_ids),
                ["security_id", "session"],
            ].copy()
            linked_factors = factors.loc[
                factors["security_id"].astype(str).isin(linked_security_ids),
                ["security_id", "session"],
            ].copy()
            linked_raw["session"] = pd.to_datetime(
                linked_raw["session"], errors="coerce"
            ).dt.normalize()
            linked_factors["session"] = pd.to_datetime(
                linked_factors["session"], errors="coerce"
            ).dt.normalize()
            factor_keys = set(
                linked_factors.dropna(subset=["session"])
                .astype({"security_id": str})
                .itertuples(index=False, name=None)
            )
            missing_factor_sessions = [
                (str(row.security_id), row.session)
                for row in linked_raw.dropna(subset=["session"]).itertuples(index=False)
                if (str(row.security_id), row.session) not in factor_keys
            ]
            if missing_factor_sessions:
                sample = ", ".join(
                    f"{security_id}/{pd.Timestamp(session).date()}"
                    for security_id, session in missing_factor_sessions[:5]
                )
                raise ValueError(
                    "Action-linked securities are missing adjustment-factor sessions: "
                    + sample
                )
        adjusted, quality, warnings = self._adjust(config, raw, factors)
        execution_bars = _frames_by_identity_intervals(
            raw,
            loaded_symbols,
            frame_intervals,
        )
        signal_bars = _frames_by_identity_intervals(
            adjusted,
            loaded_symbols,
            frame_intervals,
        )
        missing_action_linked = tuple(
            symbol
            for symbol in linked_symbols
            if symbol not in execution_bars or symbol not in signal_bars
        )
        if missing_action_linked:
            raise ValueError(
                "Action-linked securities have no usable daily prices: "
                + ", ".join(missing_action_linked)
            )
        missing_symbols = tuple(
            sorted(set(missing_symbols) | (set(requested) - set(signal_bars)))
        )
        if missing_symbols:
            quality = (
                DataQuality.DEGRADED
                if config.universe.source == "index_events"
                else DataQuality.BLOCKED
            )
            warnings.append(
                "Selected universe members have no usable daily prices: "
                + ", ".join(missing_symbols)
            )

        benchmark: dict[str, pd.DataFrame] | None = None
        if benchmark_symbol in symbol_map:
            benchmark_frame = _frame_for_securities(adjusted, symbol_map[benchmark_symbol])
            if not benchmark_frame.empty:
                benchmark = {symbol: benchmark_frame.copy() for symbol in loaded_symbols}
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
            # Action-linked successors/children are loaded so an existing
            # holding can be valued and settled, but a static universe does
            # not make those exit-only securities current entry candidates.
            expected_current = set(requested)
            if config.universe.source == "index_events":
                scheduled = _scheduled_symbols_on(
                    universe_schedule,
                    completed_session,
                )
                if scheduled is not None:
                    expected_current = set(scheduled) & set(requested)
                else:
                    master_version = self._release_versions.get("security_master")
                    master = self.repository.read_frame("security_master", master_version)
                    master_ends = pd.to_datetime(master["active_to"], errors="coerce")
                    active_ids = set(
                        master.loc[
                            master_ends.isna()
                            | (master_ends >= pd.Timestamp(completed_session)),
                            "security_id",
                        ].astype(str)
                    )
                    expected_current = {
                        symbol
                        for symbol, security_ids_for_symbol in symbol_map.items()
                        if any(security_id in active_ids for security_id in security_ids_for_symbol)
                    }
            stale = tuple(
                symbol
                for symbol in sorted(expected_current)
                if (
                    symbol not in signal_bars
                    or signal_bars[symbol].empty
                    or pd.Timestamp(signal_bars[symbol].index[-1]).date()
                    < pd.Timestamp(completed_session).date()
                )
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
            entry_symbols=requested,
        )

    def _resolve_security_ids(self, symbols: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
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
        history = history.sort_values(["symbol", "_start", "_end"], na_position="last")
        self._resolved_symbol_history = history.copy()
        return {
            str(symbol): tuple(dict.fromkeys(group["security_id"].astype(str)))
            for symbol, group in history.groupby("symbol", sort=False)
        }

    def _symbol_map_from_resolved_history(self) -> dict[str, tuple[str, ...]]:
        history = self._resolved_symbol_history
        if history.empty:
            return {}
        return {
            str(symbol): tuple(dict.fromkeys(group["security_id"].astype(str)))
            for symbol, group in history.groupby("symbol", sort=False)
        }

    def _append_symbol_history_for_ids(self, security_ids: tuple[str, ...]) -> None:
        if not security_ids:
            return
        history = self._query_dataset(
            "symbol_history",
            where_column="security_id",
            values=security_ids,
            columns=("security_id", "symbol", "effective_from", "effective_to"),
        )
        if history.empty:
            raise ValueError(
                "Action-linked securities are missing symbol history: "
                + ", ".join(security_ids)
            )
        history["_start"] = pd.to_datetime(history["effective_from"], errors="coerce")
        history["_end"] = pd.to_datetime(history["effective_to"], errors="coerce")
        combined = pd.concat(
            [self._resolved_symbol_history, history],
            ignore_index=True,
            sort=False,
        ).drop_duplicates(
            ["security_id", "symbol", "effective_from"], keep="last"
        )
        self._resolved_symbol_history = combined.sort_values(
            ["symbol", "_start", "_end"], na_position="last"
        ).reset_index(drop=True)

    def _action_linked_closure(
        self,
        requested: tuple[str, ...],
        requested_security_ids: tuple[str, ...],
        *,
        period_start: str,
        period_end: str,
        requested_intervals: dict[
            str, tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...]
        ],
    ) -> tuple[
        pd.DataFrame,
        tuple[str, ...],
        tuple[str, ...],
        dict[str, tuple[str, ...]],
    ]:
        """Load the exact successor graph needed to manage held positions.

        Only actions owned by a currently reachable symbol are followed.  This
        prevents a reused security ID's unrelated ticker interval from pulling
        arbitrary securities into a backtest.
        """

        reachable_symbols = list(requested)
        reachable_set = set(reachable_symbols)
        action_owner_set = set(requested)
        managed_ids = list(dict.fromkeys(requested_security_ids))
        managed_set = set(managed_ids)
        queried_ids: set[str] = set()
        action_frames: list[pd.DataFrame] = []
        linked_symbol_ids: dict[str, list[str]] = {}

        while True:
            pending_ids = tuple(
                security_id
                for security_id in managed_ids
                if security_id not in queried_ids
            )
            if pending_ids:
                frame = self._query_dataset(
                    "corporate_actions",
                    where_column="security_id",
                    values=pending_ids,
                    required=False,
                    date_column="effective_date",
                    min_session=period_start,
                    max_session=period_end,
                )
                queried_ids.update(pending_ids)
                if not frame.empty:
                    action_frames.append(frame)

            if not action_frames:
                return pd.DataFrame(), (), tuple(managed_ids), {}

            raw_actions = pd.concat(action_frames, ignore_index=True, sort=False)
            if "event_id" in raw_actions:
                raw_actions = raw_actions.drop_duplicates("event_id", keep="last")
            raw_actions = _filter_actions_to_requested_intervals(
                raw_actions,
                requested_intervals,
                requested_security_ids,
            )
            mapped = self._actions_in_symbol_intervals(raw_actions)
            mapped_symbols = mapped["symbol"].astype(str)
            action_types = mapped["action_type"].astype(str).str.lower()
            new_symbols = mapped["new_symbol"].fillna("").astype(str)
            # Walk ticker aliases backwards only through an exact transition
            # into a reachable symbol.  Merely sharing a security-history row
            # is not enough, which keeps unrelated reused-ticker eras out of
            # the successor graph and out of entry eligibility.
            predecessor_ticker = action_types.eq("ticker_change") & new_symbols.isin(
                action_owner_set
            )
            selected = mapped.loc[
                mapped_symbols.isin(action_owner_set) | predecessor_ticker
            ].copy()

            history_expansion: list[str] = []
            changed = False
            for old_symbol in selected.loc[
                selected["action_type"].astype(str).str.lower().eq("ticker_change"),
                "symbol",
            ].astype(str):
                if old_symbol and old_symbol not in action_owner_set:
                    action_owner_set.add(old_symbol)
                    changed = True
            for row in selected.itertuples(index=False):
                if str(getattr(row, "action_type", "")).lower() not in POSITION_CREATING_ACTIONS:
                    continue
                new_symbol = str(getattr(row, "new_symbol", "") or "").strip()
                new_security_id = str(
                    getattr(row, "new_security_id", "") or ""
                ).strip()
                if not new_symbol or not new_security_id:
                    raise ValueError(
                        "Position-creating corporate action lacks an exact successor: "
                        + str(getattr(row, "event_id", ""))
                    )
                if new_symbol not in reachable_set:
                    reachable_set.add(new_symbol)
                    reachable_symbols.append(new_symbol)
                    changed = True
                if new_symbol not in action_owner_set:
                    action_owner_set.add(new_symbol)
                    changed = True
                if new_security_id not in managed_set:
                    managed_set.add(new_security_id)
                    managed_ids.append(new_security_id)
                    history_expansion.append(new_security_id)
                    changed = True
                ids_for_symbol = linked_symbol_ids.setdefault(new_symbol, [])
                if new_security_id not in ids_for_symbol:
                    ids_for_symbol.append(new_security_id)
                history = self._resolved_symbol_history
                pair_exists = bool(
                    not history.empty
                    and (
                        history["security_id"].astype(str).eq(new_security_id)
                        & history["symbol"].astype(str).eq(new_symbol)
                    ).any()
                )
                if not pair_exists and new_security_id not in history_expansion:
                    history_expansion.append(new_security_id)
                    changed = True

            if history_expansion:
                self._append_symbol_history_for_ids(tuple(history_expansion))
                for row in selected.itertuples(index=False):
                    if str(getattr(row, "action_type", "")).lower() not in POSITION_CREATING_ACTIONS:
                        continue
                    new_symbol = str(getattr(row, "new_symbol", "") or "").strip()
                    new_security_id = str(
                        getattr(row, "new_security_id", "") or ""
                    ).strip()
                    history = self._resolved_symbol_history
                    if not bool(
                        (
                            history["security_id"].astype(str).eq(new_security_id)
                            & history["symbol"].astype(str).eq(new_symbol)
                        ).any()
                    ):
                        raise ValueError(
                            "Corporate-action successor disagrees with symbol history: "
                            f"{new_security_id}/{new_symbol}"
                        )

            if not changed:
                linked = tuple(
                    symbol for symbol in reachable_symbols if symbol not in set(requested)
                )
                return (
                    selected.reset_index(drop=True),
                    linked,
                    tuple(managed_ids),
                    {
                        symbol: tuple(security_ids)
                        for symbol, security_ids in linked_symbol_ids.items()
                    },
                )

    def _linked_identity_intervals(
        self,
        linked_symbol_ids: dict[str, tuple[str, ...]],
    ) -> dict[
        str, tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...]
    ]:
        return self._identity_history_intervals(linked_symbol_ids)

    def _identity_feature_intervals(
        self,
        symbol_ids: dict[str, tuple[str, ...]],
    ) -> dict[
        str, tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...]
    ]:
        """Include prior aliases of one issuer as causal feature warm-up.

        Entry eligibility still comes from the point-in-time universe schedule.
        The expanded start only lets a new ticker alias inherit the same
        security ID's historical prices.  Each alias remains bounded at its own
        final history date, and different security IDs remain separate segments.
        """

        history = self._resolved_symbol_history
        output: dict[
            str, list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]
        ] = {}
        for symbol, security_ids in symbol_ids.items():
            for security_id in security_ids:
                identity = history.loc[
                    history["security_id"].astype(str).eq(security_id)
                ]
                matches = identity.loc[
                    identity["symbol"].astype(str).eq(symbol)
                ]
                if matches.empty:
                    raise ValueError(
                        "Selected security identity is missing its symbol interval: "
                        f"{security_id}/{symbol}"
                    )
                starts = identity["_start"].dropna()
                start = pd.Timestamp(starts.min()) if not starts.empty else None
                ends = matches["_end"]
                end = (
                    None
                    if ends.isna().any()
                    else pd.Timestamp(ends.max()) + pd.Timedelta(days=1)
                )
                output.setdefault(symbol, []).append(
                    (security_id, start, end)
                )
        return {
            symbol: tuple(dict.fromkeys(intervals))
            for symbol, intervals in output.items()
        }

    def _action_successor_intervals(
        self,
        actions: pd.DataFrame,
    ) -> dict[
        str, tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...]
    ]:
        """Bound exact successor prices from the action date through its alias."""

        if actions.empty:
            return {}
        history = self._resolved_symbol_history
        action_dates = _corporate_action_dates(actions)
        output: dict[
            str, list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]
        ] = {}
        for position, row in enumerate(actions.itertuples(index=False)):
            if (
                str(getattr(row, "action_type", "")).lower()
                not in POSITION_CREATING_ACTIONS
            ):
                continue
            symbol = str(getattr(row, "new_symbol", "") or "").strip()
            security_id = str(
                getattr(row, "new_security_id", "") or ""
            ).strip()
            source_symbol = str(getattr(row, "symbol", "") or "").strip()
            source_security_id = str(
                getattr(row, "security_id", "") or ""
            ).strip()
            if symbol == source_symbol and security_id == source_security_id:
                # Static aliases collapse a same-identity rename into a no-op.
                # Its requested frame already spans the complete identity.
                continue
            start = action_dates.iloc[position]
            if not symbol or not security_id or pd.isna(start):
                continue
            start = pd.Timestamp(start).normalize()
            matches = history.loc[
                history["security_id"].astype(str).eq(security_id)
                & history["symbol"].astype(str).eq(symbol)
                & (history["_start"].isna() | history["_start"].le(start))
                & (history["_end"].isna() | history["_end"].ge(start))
            ]
            if matches.empty:
                raise ValueError(
                    "Corporate-action successor date is outside symbol history: "
                    f"{security_id}/{symbol}/{start.date()}"
                )
            ends = matches["_end"]
            end = (
                None
                if ends.isna().any()
                else pd.Timestamp(ends.max()) + pd.Timedelta(days=1)
            )
            output.setdefault(symbol, []).append((security_id, start, end))
        return {
            symbol: tuple(dict.fromkeys(intervals))
            for symbol, intervals in output.items()
        }

    def _identity_history_intervals(
        self,
        symbol_ids: dict[str, tuple[str, ...]],
    ) -> dict[
        str, tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...]
    ]:
        history = self._resolved_symbol_history
        output: dict[
            str, list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]
        ] = {}
        for symbol, security_ids in symbol_ids.items():
            for security_id in security_ids:
                matches = history.loc[
                    history["security_id"].astype(str).eq(security_id)
                    & history["symbol"].astype(str).eq(symbol)
                ]
                for _, row in matches.iterrows():
                    start = None if pd.isna(row["_start"]) else pd.Timestamp(row["_start"])
                    end = (
                        None
                        if pd.isna(row["_end"])
                        else pd.Timestamp(row["_end"]) + pd.Timedelta(days=1)
                    )
                    output.setdefault(symbol, []).append(
                        (security_id, start, end)
                    )
                if matches.empty:
                    raise ValueError(
                        "Selected security identity is missing its symbol interval: "
                        f"{security_id}/{symbol}"
                    )
        return {
            symbol: tuple(intervals)
            for symbol, intervals in output.items()
        }

    def _actions_in_symbol_intervals(self, actions: pd.DataFrame) -> pd.DataFrame:
        history = self._resolved_symbol_history
        if actions.empty or history.empty:
            return actions.iloc[0:0].assign(symbol=pd.Series(dtype=str))
        working = actions.copy()
        working["_action_row"] = range(len(working))
        working_dates = _corporate_action_dates(working)
        same_day_inbound_identities = {
            (
                str(row.new_security_id).strip(),
                pd.Timestamp(working_dates.iloc[position]).normalize(),
            )
            for position, row in enumerate(working.itertuples(index=False))
            if (
                str(getattr(row, "action_type", "")).lower()
                in POSITION_CREATING_ACTIONS
                and str(getattr(row, "new_security_id", "") or "").strip()
                and not pd.isna(working_dates.iloc[position])
            )
        }
        same_day_ticker_successors = {
            (
                str(row.new_security_id).strip(),
                pd.Timestamp(working_dates.iloc[position]).normalize(),
            )
            for position, row in enumerate(working.itertuples(index=False))
            if (
                str(getattr(row, "action_type", "")).lower()
                == "ticker_change"
                and str(getattr(row, "new_security_id", "") or "").strip()
                and not pd.isna(working_dates.iloc[position])
            )
        }
        joined = working.merge(
            history[["security_id", "symbol", "_start", "_end"]],
            on="security_id",
            how="left",
        )
        action_dates = _corporate_action_dates(joined)
        consumes_old_symbol = joined["action_type"].astype(str).str.lower().isin(
            OLD_SYMBOL_ACTIONS
        )
        active_on_date = (
            joined["symbol"].notna()
            & (joined["_start"].isna() | (action_dates >= joined["_start"]))
            & (joined["_end"].isna() | (action_dates <= joined["_end"]))
        )
        # Symbol-history bounds are inclusive dates.  Position-consuming
        # actions at a ticker boundary must address the symbol held through the
        # preceding trading session, not the new symbol that starts on the
        # action date.  A ten-calendar-day bound safely spans weekends and US
        # holiday closures while still rejecting unrelated reused-ticker eras.
        recently_held_before = (
            joined["symbol"].notna()
            & joined["_start"].notna()
            & joined["_start"].lt(action_dates)
            & (
                joined["_end"].isna()
                | joined["_end"].ge(action_dates - pd.Timedelta(days=10))
            )
        )
        # Prefer the predecessor alias whenever one existed before a ticker or
        # terminal transition.  A same-session action-created child can,
        # however, begin and be liquidated on that very date (BMYRT's reviewed
        # first-session exit mark).  In that narrow case there is no prior
        # alias to select, so fall back to the identity active on the action
        # date instead of silently dropping the settlement event.
        has_recent_predecessor = recently_held_before.groupby(
            joined["_action_row"]
        ).transform("any")
        same_day_created_terminal = pd.Series(
            [
                bool(
                    str(row.action_type).lower()
                    in {"cash_merger", "delisting"}
                    and not pd.isna(action_dates.iloc[position])
                    and (
                        str(row.security_id).strip(),
                        pd.Timestamp(action_dates.iloc[position]).normalize(),
                    )
                    in same_day_inbound_identities
                )
                for position, row in enumerate(joined.itertuples(index=False))
            ],
            index=joined.index,
        )
        ticker_successor_entitlement_types = {
            "cash_dividend",
            "special_dividend",
            "split",
            "stock_dividend",
            "capital_reduction",
        }
        same_day_ticker_successor_entitlement = pd.Series(
            [
                bool(
                    str(row.action_type).lower()
                    in ticker_successor_entitlement_types
                    and not pd.isna(action_dates.iloc[position])
                    and (
                        str(row.security_id).strip(),
                        pd.Timestamp(action_dates.iloc[position]).normalize(),
                    )
                    in same_day_ticker_successors
                )
                for position, row in enumerate(joined.itertuples(index=False))
            ],
            index=joined.index,
        )
        old_symbol_valid = recently_held_before | (
            ~has_recent_predecessor
            & (
                same_day_created_terminal
                | same_day_ticker_successor_entitlement
            )
            & active_on_date
        )
        valid = active_on_date.where(~consumes_old_symbol, old_symbol_valid)
        selected = (
            joined.loc[valid]
            .sort_values(["_action_row", "_start"], na_position="first")
            .drop_duplicates("_action_row", keep="last")
        )
        # Some reviewed ticker transitions move to a distinct security ID and
        # carry a same-open split or distribution on the successor (LB ->
        # BBWI).  Those entitlements belong to the prior-close predecessor.
        # Keep the global entitlement-before-transition ordering by assigning
        # only exact same-day ticker-successor actions to that predecessor;
        # never generalize this to stock mergers or spin-offs, whose new shares
        # may not own the successor's prior-close entitlement.
        selected_dates = _corporate_action_dates(selected)
        ticker_predecessors: dict[tuple[str, pd.Timestamp], str] = {}
        for position, row in enumerate(selected.itertuples(index=False)):
            if str(row.action_type).lower() != "ticker_change":
                continue
            successor_id = str(getattr(row, "new_security_id", "") or "").strip()
            predecessor = str(getattr(row, "symbol", "") or "").strip()
            action_date = selected_dates.iloc[position]
            if not successor_id or not predecessor or pd.isna(action_date):
                continue
            key = (successor_id, pd.Timestamp(action_date).normalize())
            prior = ticker_predecessors.get(key)
            if prior is not None and prior != predecessor:
                raise ValueError(
                    "Ambiguous same-day ticker predecessor for successor action: "
                    f"{successor_id}/{pd.Timestamp(action_date).date()}"
                )
            ticker_predecessors[key] = predecessor
        entitlement_types = ticker_successor_entitlement_types
        for position, index in enumerate(selected.index):
            row = selected.loc[index]
            if str(row.get("action_type") or "").lower() not in entitlement_types:
                continue
            action_date = selected_dates.iloc[position]
            if pd.isna(action_date):
                continue
            predecessor = ticker_predecessors.get(
                (
                    str(row.get("security_id") or "").strip(),
                    pd.Timestamp(action_date).normalize(),
                )
            )
            if predecessor:
                selected.at[index, "symbol"] = predecessor
        return selected.drop(columns=["_action_row", "_start", "_end"])

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
            f"SELECT {projection} FROM read_parquet("
            f"[{path_sql}], hive_partitioning=false, union_by_name=true) "
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


def _requested_identity_intervals(
    requested: tuple[str, ...],
    symbol_map: dict[str, tuple[str, ...]],
    *,
    universe_schedule: tuple[dict[str, Any], ...],
    index_event_universe: bool,
) -> dict[
    str, tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...]
]:
    scheduled: dict[
        str, list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]
    ] = {}
    if index_event_universe and universe_schedule:
        entries = sorted(
            universe_schedule,
            key=lambda item: str(item.get("effective_date", "")),
        )
        for position, entry in enumerate(entries):
            start = pd.to_datetime(entry.get("effective_date"), errors="coerce")
            if pd.isna(start):
                continue
            end = (
                pd.to_datetime(
                    entries[position + 1].get("effective_date"), errors="coerce"
                )
                if position + 1 < len(entries)
                else None
            )
            if end is not None and pd.isna(end):
                end = None
            seen_in_entry: dict[str, str] = {}
            for member in entry.get("members", ()) or ():
                symbol = str(member.get("symbol") or "")
                security_id = str(member.get("security_id") or "")
                if symbol not in requested or not security_id:
                    continue
                prior = seen_in_entry.get(symbol)
                if prior is not None and prior != security_id:
                    raise ValueError(
                        "Index schedule maps one ticker to multiple issuers on the "
                        f"same date: {symbol}/{start.date()}"
                    )
                seen_in_entry[symbol] = security_id
                scheduled.setdefault(symbol, []).append(
                    (security_id, pd.Timestamp(start).normalize(), None if end is None else pd.Timestamp(end).normalize())
                )

    output: dict[
        str, tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...]
    ] = {}
    for symbol in requested:
        intervals = scheduled.get(symbol, [])
        if intervals:
            output[symbol] = tuple(intervals)
            continue
        security_ids = symbol_map.get(symbol, ())
        if len(security_ids) > 1:
            mode = "index schedule" if index_event_universe else "static universe"
            raise ValueError(
                f"Ambiguous ticker identity in {mode}: {symbol} -> "
                + ", ".join(security_ids)
            )
        if security_ids:
            output[symbol] = ((security_ids[0], None, None),)
    return output


def _filter_actions_to_requested_intervals(
    actions: pd.DataFrame,
    requested_intervals: dict[
        str, tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...]
    ],
    requested_security_ids: tuple[str, ...],
) -> pd.DataFrame:
    if actions.empty:
        return actions
    by_security: dict[
        str, list[tuple[pd.Timestamp | None, pd.Timestamp | None]]
    ] = {}
    for intervals in requested_intervals.values():
        for security_id, start, end in intervals:
            by_security.setdefault(security_id, []).append((start, end))
    requested_ids = set(requested_security_ids)
    dates = _corporate_action_dates(actions)
    action_types = actions["action_type"].fillna("").astype(str).str.lower()
    keep = []
    for position, security_id in enumerate(actions["security_id"].astype(str)):
        if security_id not in requested_ids:
            keep.append(True)
            continue
        action_date = dates.iloc[position]
        action_type = action_types.iloc[position]

        def inside_interval(start, end) -> bool:
            if start is not None and action_date < start:
                return False
            if end is None or action_date <= end:
                return True
            if action_type not in OLD_SYMBOL_ACTIONS:
                return False
            # Identity-history intervals use the day after ``effective_to`` as
            # their exclusive end.  Compare against the actual final-held date
            # to mirror ``_actions_in_symbol_intervals`` exactly.
            final_held_date = pd.Timestamp(end) - pd.Timedelta(days=1)
            return bool(
                action_date
                <= final_held_date + OLD_SYMBOL_ACTION_MAX_CALENDAR_GAP
            )

        keep.append(
            bool(
                not pd.isna(action_date)
                and any(
                    inside_interval(start, end)
                    for start, end in by_security.get(security_id, ())
                )
            )
        )
    return actions.loc[keep].copy()


def _collapse_static_action_aliases(
    actions: pd.DataFrame,
    requested_intervals: dict[
        str, tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...]
    ],
) -> pd.DataFrame:
    alias_by_security: dict[str, str] = {}
    for symbol, intervals in requested_intervals.items():
        for security_id, _, _ in intervals:
            prior = alias_by_security.get(security_id)
            if prior is not None and prior != symbol:
                raise ValueError(
                    "Static universe contains multiple aliases for one identity: "
                    f"{security_id} -> {prior}, {symbol}"
                )
            alias_by_security[security_id] = symbol
    output = actions.copy()
    security_ids = output["security_id"].astype(str)
    collapsed = security_ids.map(alias_by_security)
    output["symbol"] = collapsed.where(collapsed.notna(), output["symbol"])
    same_identity_ticker = (
        output["action_type"].astype(str).str.lower().eq("ticker_change")
        & security_ids.eq(output["new_security_id"].fillna("").astype(str))
        & collapsed.notna()
    )
    output.loc[same_identity_ticker, "new_symbol"] = collapsed.loc[
        same_identity_ticker
    ]
    return output


def _corporate_action_dates(actions: pd.DataFrame) -> pd.Series:
    if "ex_date" not in actions:
        return pd.to_datetime(actions["effective_date"], errors="coerce").dt.normalize()
    ex_dates = actions["ex_date"]
    has_ex_date = ex_dates.notna() & ex_dates.astype(str).str.strip().ne("")
    return pd.to_datetime(
        ex_dates.where(has_ex_date, actions["effective_date"]),
        errors="coerce",
    ).dt.normalize()


def _frames_by_identity_intervals(
    frame: pd.DataFrame,
    symbols: tuple[str, ...],
    identity_intervals: dict[
        str, tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...]
    ],
) -> dict[str, pd.DataFrame]:
    groups = {
        str(security_id): group
        for security_id, group in frame.groupby(
            frame["security_id"].astype(str),
            sort=False,
        )
    }
    output: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        intervals = identity_intervals.get(symbol)
        if intervals is None:
            continue
        distinct_ids = {interval[0] for interval in intervals}
        pieces: list[pd.DataFrame] = []
        session_owners: dict[pd.Timestamp, str] = {}
        for security_id, start, end in intervals:
            source = groups.get(security_id)
            if source is None:
                continue
            piece = source.copy()
            sessions = pd.to_datetime(piece["session"], errors="coerce")
            mask = pd.Series(True, index=piece.index)
            if start is not None:
                mask &= sessions >= start
            if end is not None:
                mask &= sessions < end
            piece = piece.loc[mask].copy()
            if piece.empty:
                continue
            if len(distinct_ids) > 1:
                for session in pd.to_datetime(
                    piece["session"], errors="coerce"
                ).dropna():
                    normalized = pd.Timestamp(session).normalize()
                    owner = session_owners.get(normalized)
                    if owner is not None and owner != security_id:
                        raise ValueError(
                            "Reused ticker issuer price histories overlap on one "
                            f"session: {symbol}/{normalized.date()} -> "
                            f"{owner}, {security_id}"
                        )
                    session_owners[normalized] = security_id
                piece["IdentitySegment"] = security_id
            pieces.append(piece)
        value = _frame_from_pieces(pieces)
        if not value.empty:
            output[symbol] = value
    return output


def _scheduled_symbols_on(
    schedule: tuple[dict[str, Any], ...],
    completed_session: str,
) -> tuple[str, ...] | None:
    if not schedule:
        return None
    cutoff = pd.Timestamp(completed_session)
    eligible = [
        entry
        for entry in schedule
        if pd.Timestamp(str(entry.get("effective_date", ""))) <= cutoff
    ]
    if not eligible:
        return ()
    selected = max(eligible, key=lambda entry: str(entry.get("effective_date", "")))
    symbols = selected.get("symbols")
    if symbols is not None:
        return tuple(str(symbol) for symbol in symbols)
    return tuple(
        str(member.get("symbol", ""))
        for member in selected.get("members", ())
        if str(member.get("symbol", ""))
    )


def _frame_for_securities(frame: pd.DataFrame, security_ids: tuple[str, ...]) -> pd.DataFrame:
    pieces = [
        frame.loc[frame["security_id"].astype(str) == security_id].copy()
        for security_id in security_ids
    ]
    pieces = [piece for piece in pieces if not piece.empty]
    return _frame_from_pieces(pieces)


def _frame_from_pieces(pieces: list[pd.DataFrame]) -> pd.DataFrame:
    if not pieces:
        return pd.DataFrame()
    value = pd.concat(pieces, ignore_index=True)
    value["session"] = pd.to_datetime(value["session"])
    value = value.sort_values(["session", "security_id"]).drop_duplicates("session", keep="last")
    value = value.set_index("session")
    columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "IdentitySegment",
        )
        if column in value
    ]
    value = value[columns].rename(
        columns={
            column: column.title()
            for column in columns
            if column != "IdentitySegment"
        }
    )
    value.index.name = "Date"
    return value


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
        from .ingest import configured_daily_synchronizer

        synced = configured_daily_synchronizer(
            repository, config.data_store.ingest_source
        ).sync(
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
    allow_stale: bool = False,
) -> MarketData:
    """Load the configured source, normally enforcing daily Parquet freshness."""
    if config.data_store.provider == "yahoo" or config.market == "KR":
        from ..data import download_market_data

        return download_market_data(config, symbols, resolved_universe=resolved_universe)
    if not allow_stale:
        ensure_configured_data_ready(config, force_sync=force_sync)
    schedule = resolved_universe.schedule_as_dicts() if resolved_universe is not None else ()
    market_data = ParquetMarketDataProvider(config.data_store.local_cache_dir).load(
        config,
        symbols,
        universe_schedule=schedule,
    )
    if resolved_universe is None:
        return market_data
    return replace(
        market_data,
        universe_snapshot=resolved_universe.snapshot.to_dict(),
        universe_schedule=resolved_universe.schedule_as_dicts(),
    )
