from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd

from .config import UNIVERSE_PROFILE_MARKETS, UniverseFilterConfig, _resolve_existing_path

if TYPE_CHECKING:
    from .config import AppConfig


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    market: str
    exchange: str
    security_id: str = ""
    name: str = ""
    security_type: str = "STOCK"
    yfinance_symbol: str = ""
    benchmark: str = ""
    profiles: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> UniverseMember:
        return cls(
            symbol=str(raw["symbol"]),
            market=str(raw["market"]),
            exchange=str(raw.get("exchange") or raw["market"]),
            security_id=str(raw.get("security_id") or ""),
            name=str(raw.get("name") or ""),
            security_type=str(raw.get("security_type") or "STOCK"),
            yfinance_symbol=str(raw.get("yfinance_symbol") or ""),
            benchmark=str(raw.get("benchmark") or ""),
            profiles=tuple(str(value) for value in raw.get("profiles", ()) or ()),
        )


@dataclass(frozen=True)
class UniverseSnapshot:
    schema_version: int
    as_of: str
    created_at: str
    market: str
    source: str
    profiles: tuple[str, ...]
    selection_hash: str
    raw_members: tuple[UniverseMember, ...]
    eligible_members: tuple[UniverseMember, ...]
    rejected: tuple[dict[str, Any], ...]
    filters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of": self.as_of,
            "created_at": self.created_at,
            "market": self.market,
            "source": self.source,
            "profiles": list(self.profiles),
            "selection_hash": self.selection_hash,
            "raw_members": [member.to_dict() for member in self.raw_members],
            "eligible_members": [member.to_dict() for member in self.eligible_members],
            "rejected": list(self.rejected),
            "filters": self.filters,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> UniverseSnapshot:
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            as_of=str(raw["as_of"]),
            created_at=str(raw.get("created_at") or ""),
            market=str(raw["market"]),
            source=str(raw["source"]),
            profiles=tuple(str(value) for value in raw.get("profiles", ()) or ()),
            selection_hash=str(raw["selection_hash"]),
            raw_members=tuple(UniverseMember.from_dict(item) for item in raw.get("raw_members", ())),
            eligible_members=tuple(UniverseMember.from_dict(item) for item in raw.get("eligible_members", ())),
            rejected=tuple(dict(item) for item in raw.get("rejected", ())),
            filters=dict(raw.get("filters", {})),
        )


@dataclass(frozen=True)
class UniverseScheduleEntry:
    effective_date: str
    members: tuple[UniverseMember, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(member.symbol for member in self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_date": self.effective_date,
            "symbols": list(self.symbols),
            "members": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True)
class ResolvedUniverse:
    eligible_members: tuple[UniverseMember, ...]
    exit_only_members: tuple[UniverseMember, ...]
    raw_members: tuple[UniverseMember, ...]
    snapshot: UniverseSnapshot
    entries_allowed: bool = True
    refresh_error: str | None = None
    schedule: tuple[UniverseScheduleEntry, ...] = ()

    @property
    def members(self) -> tuple[UniverseMember, ...]:
        return self.eligible_members

    @property
    def eligible_symbols(self) -> tuple[str, ...]:
        return tuple(member.symbol for member in self.eligible_members)

    @property
    def exit_only_symbols(self) -> tuple[str, ...]:
        return tuple(member.symbol for member in self.exit_only_members)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.eligible_symbols + self.exit_only_symbols))

    @property
    def member_map(self) -> dict[str, UniverseMember]:
        return {
            member.symbol: member
            for member in self.raw_members + self.eligible_members + self.exit_only_members
        }

    def member_for(self, symbol: str) -> UniverseMember | None:
        return self.member_map.get(symbol)

    def yfinance_symbol_for(self, symbol: str) -> str:
        member = self.member_for(symbol)
        return member.yfinance_symbol if member and member.yfinance_symbol else symbol

    def benchmark_for(self, symbol: str) -> str:
        member = self.member_for(symbol)
        if member and member.benchmark:
            return member.benchmark
        return "^KS11" if self.snapshot.market == "KR" else "QQQ"

    def schedule_as_dicts(self) -> tuple[dict[str, Any], ...]:
        return tuple(entry.to_dict() for entry in self.schedule)


class UniverseProvider(Protocol):
    def __call__(self, as_of: date) -> Iterable[UniverseMember]:
        ...


class PriceHistoryLoader(Protocol):
    def __call__(self, members: tuple[UniverseMember, ...], required_bars: int) -> Mapping[str, pd.DataFrame]:
        ...


class StatusLoader(Protocol):
    def __call__(self, as_of: date) -> tuple[set[str], set[str]]:
        ...


_PROVIDERS: dict[str, UniverseProvider] = {}


def register_universe_provider(
    profile: str,
    provider: UniverseProvider | None = None,
) -> UniverseProvider | Callable[[UniverseProvider], UniverseProvider]:
    normalized = str(profile).strip().lower()
    if normalized not in UNIVERSE_PROFILE_MARKETS:
        raise ValueError(f"Unknown universe profile: {profile}")

    def register(candidate: UniverseProvider) -> UniverseProvider:
        if normalized in _PROVIDERS:
            raise ValueError(f"Universe provider already registered: {normalized}")
        _PROVIDERS[normalized] = candidate
        return candidate

    return register(provider) if provider is not None else register


def available_universe_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


def resolve_universe(
    config: AppConfig,
    *,
    market: str | None = None,
    held_symbols: Iterable[str] = (),
    previously_managed: Mapping[str, UniverseMember] | Iterable[str] = (),
    mode: str = "backtest",
    as_of: date | None = None,
    force_refresh: bool = False,
    price_loader: PriceHistoryLoader | None = None,
    status_loader: StatusLoader | None = None,
) -> ResolvedUniverse:
    selected_market = str(market or config.market).upper()
    if selected_market == "AUTO":
        selected_market = "US"
    if selected_market not in {"US", "KR"}:
        raise ValueError("market must be US, KR, or AUTO.")
    resolved_date = as_of or _market_date(selected_market)
    held = {str(symbol) for symbol in held_symbols}
    managed_map = _managed_member_map(previously_managed, selected_market)

    explicit_symbols = tuple(str(symbol) for symbol in config.symbols if str(symbol))
    universe_config = config.universe
    source = "symbols" if explicit_symbols else universe_config.source
    profiles = (
        tuple(universe_config.profiles.get(selected_market, ()))
        if source in {"profiles", "index_events"}
        else ()
    )
    if source in {"profiles", "index_events"} and not profiles:
        raise ValueError(f"No universe profiles configured for market={selected_market}.")
    selection_hash = _selection_hash(config, selected_market, source, profiles, explicit_symbols)
    snapshot_path = _snapshot_path(universe_config.snapshot_dir, selected_market, selection_hash, resolved_date)

    if source == "history_file":
        schedule = _load_history_file_schedule(
            universe_config.history_file,
            selected_market,
            config.universe_file,
        )
        raw_members = _unique_members(
            member
            for entry in schedule
            for member in entry.members
        )
        snapshot = _build_snapshot(
            resolved_date,
            selected_market,
            source,
            _schedule_profiles(schedule),
            selection_hash,
            raw_members,
            raw_members,
            (),
            universe_config.filters,
        )
        return _with_exit_only(snapshot, held, managed_map, True, None, schedule=schedule)

    if source == "index_events":
        schedule = _load_index_event_schedule(config, profiles, resolved_date)
        effective = max(
            (entry for entry in schedule if entry.effective_date <= resolved_date.isoformat()),
            key=lambda entry: entry.effective_date,
        )
        raw_members = effective.members
        snapshot = _build_snapshot(
            resolved_date,
            selected_market,
            source,
            profiles,
            selection_hash,
            raw_members,
            raw_members,
            (),
            universe_config.filters,
        )
        return _with_exit_only(snapshot, held, managed_map, True, None, schedule=schedule)

    if source == "symbols":
        symbols = explicit_symbols or universe_config.symbols
        raw_members = tuple(
            _member_for_symbol(symbol, selected_market, (), config.universe_file)
            for symbol in sorted(set(symbols))
        )
        snapshot = _build_snapshot(
            resolved_date,
            selected_market,
            source,
            (),
            selection_hash,
            raw_members,
            raw_members,
            (),
            universe_config.filters,
        )
        return _with_exit_only(snapshot, held, managed_map, True, None)

    should_snapshot = source == "profiles" or universe_config.filters.enabled
    if should_snapshot and snapshot_path.exists() and not force_refresh:
        snapshot = _read_snapshot(snapshot_path)
        return _with_exit_only(snapshot, held, managed_map, True, None)

    try:
        if source == "profiles":
            raw_members = _load_profile_members(profiles, resolved_date, selected_market)
        else:
            raw_members = _load_file_members(universe_config.file, selected_market)
        if not raw_members:
            raise RuntimeError(f"Universe source returned no members for market={selected_market}.")
        eligible, rejected = _apply_filters(
            raw_members,
            universe_config.filters,
            resolved_date,
            selected_market,
            price_loader=price_loader,
            status_loader=status_loader,
        )
        snapshot = _build_snapshot(
            resolved_date,
            selected_market,
            source,
            profiles,
            selection_hash,
            raw_members,
            eligible,
            rejected,
            universe_config.filters,
        )
        if should_snapshot:
            _write_snapshot(snapshot_path, snapshot)
        return _with_exit_only(snapshot, held, managed_map, True, None)
    except Exception as exc:
        if mode not in {"live", "paper"}:
            raise RuntimeError(f"Universe refresh failed for {selected_market}: {exc}") from exc
        fallback = _latest_snapshot(universe_config.snapshot_dir, selected_market, selection_hash)
        if fallback is None:
            raise RuntimeError(
                f"Universe refresh failed for {selected_market} and no prior snapshot is available: {exc}"
            ) from exc
        return _with_exit_only(fallback, held, managed_map, False, str(exc))


def snapshot_summary(resolved: ResolvedUniverse) -> dict[str, Any]:
    snapshot = resolved.snapshot
    return {
        "source": snapshot.source,
        "profiles": list(snapshot.profiles),
        "as_of": snapshot.as_of,
        "selection_hash": snapshot.selection_hash,
        "raw_count": len(snapshot.raw_members),
        "eligible_count": len(snapshot.eligible_members),
        "rejected_count": len(snapshot.rejected),
        "entries_allowed": resolved.entries_allowed,
        "refresh_error": resolved.refresh_error,
        "rolling_schedule_count": len(resolved.schedule),
        "survivorship_bias_warning": (
            "Current constituent snapshot is applied to the full historical period."
            if snapshot.source == "profiles"
            else None
        ),
    }


def universe_request_key(
    config: AppConfig,
    *,
    market: str | None = None,
    as_of: date | None = None,
) -> tuple[str, str]:
    selected_market = str(market or config.market).upper()
    if selected_market == "AUTO":
        selected_market = "US"
    resolved_date = as_of or _market_date(selected_market)
    explicit_symbols = tuple(str(symbol) for symbol in config.symbols if str(symbol))
    source = "symbols" if explicit_symbols else config.universe.source
    profiles = (
        tuple(config.universe.profiles.get(selected_market, ()))
        if source in {"profiles", "index_events"}
        else ()
    )
    return (
        resolved_date.isoformat(),
        _selection_hash(config, selected_market, source, profiles, explicit_symbols),
    )


def _with_exit_only(
    snapshot: UniverseSnapshot,
    held: set[str],
    managed_map: Mapping[str, UniverseMember],
    entries_allowed: bool,
    refresh_error: str | None,
    schedule: tuple[UniverseScheduleEntry, ...] = (),
) -> ResolvedUniverse:
    raw_map = {member.symbol: member for member in snapshot.raw_members}
    eligible_map = {member.symbol: member for member in snapshot.eligible_members}
    exit_only = []
    for symbol in sorted(held - set(eligible_map)):
        member = raw_map.get(symbol) or managed_map.get(symbol)
        if member is not None:
            exit_only.append(member)
    return ResolvedUniverse(
        eligible_members=snapshot.eligible_members,
        exit_only_members=tuple(exit_only),
        raw_members=snapshot.raw_members,
        snapshot=snapshot,
        entries_allowed=entries_allowed,
        refresh_error=refresh_error,
        schedule=schedule,
    )


def _managed_member_map(
    value: Mapping[str, UniverseMember] | Iterable[str],
    market: str,
) -> dict[str, UniverseMember]:
    if isinstance(value, Mapping):
        return {
            str(symbol): member
            for symbol, member in value.items()
            if isinstance(member, UniverseMember)
        }
    return {
        str(symbol): _member_for_symbol(str(symbol), market, (), "universe.json")
        for symbol in value
    }


def _load_profile_members(profiles: tuple[str, ...], as_of: date, market: str) -> tuple[UniverseMember, ...]:
    combined: dict[str, UniverseMember] = {}
    benchmark = _profile_benchmark(profiles, market)
    for profile in profiles:
        provider = _PROVIDERS.get(profile)
        if provider is None:
            raise ValueError(f"No provider registered for universe profile={profile}")
        for raw_member in provider(as_of):
            symbol = str(raw_member.symbol).strip()
            if not symbol:
                continue
            existing = combined.get(symbol)
            member_profiles = tuple(sorted(set((existing.profiles if existing else ()) + (profile,))))
            member = UniverseMember(
                symbol=symbol,
                market=market,
                exchange=raw_member.exchange,
                security_id=raw_member.security_id,
                name=raw_member.name,
                security_type=raw_member.security_type,
                yfinance_symbol=raw_member.yfinance_symbol or _default_yfinance_symbol(symbol, raw_member.exchange),
                benchmark=(
                    _benchmark_for_exchange(raw_member.exchange)
                    if market == "KR"
                    else benchmark
                ),
                profiles=member_profiles,
            )
            combined[symbol] = member
    return tuple(combined[symbol] for symbol in sorted(combined))


def _load_index_event_schedule(
    config: AppConfig,
    profiles: tuple[str, ...],
    as_of: date,
) -> tuple[UniverseScheduleEntry, ...]:
    if config.market not in {"US", "AUTO"}:
        raise ValueError("index_events universe currently supports market=US only.")
    from .market_store.index_membership import IndexEventReplayer
    from .market_store.repository import LocalDatasetRepository

    repository = LocalDatasetRepository(config.data_store.local_cache_dir)
    release, _ = repository.current_release()

    class _ReleaseRepositoryView:
        """Expose only the versions pinned by the selected immutable release."""

        def current_manifest(self, dataset: str):
            version = release.dataset_versions.get(dataset) if release is not None else None
            return (
                repository.manifest_for_version(dataset, version)
                if version is not None
                else None
            )

        def read_frame(
            self,
            dataset: str,
            _version: str | None = None,
        ) -> pd.DataFrame:
            version = release.dataset_versions.get(dataset) if release is not None else None
            if version is None:
                return pd.DataFrame()
            return repository.read_frame(dataset, version)

    def release_frame(dataset: str) -> pd.DataFrame:
        version = release.dataset_versions.get(dataset) if release is not None else None
        if release is not None and version is None:
            raise FileNotFoundError(f"Dataset is not part of release {release.version}: {dataset}")
        return repository.read_frame(dataset, version)

    anchors = release_frame("index_constituent_anchors")
    events = release_frame("index_membership_events")
    try:
        overlays = release_frame("custom_universe_overlays")
    except FileNotFoundError:
        overlays = pd.DataFrame()
    symbol_history = release_frame("symbol_history")
    try:
        security_master = release_frame("security_master")
    except FileNotFoundError:
        security_master = pd.DataFrame()
    from .market_store.operational_validation import (
        reviewed_operational_index_identity_gap_fingerprints,
    )

    reviewed_identity_gaps = frozenset(
        reviewed_operational_index_identity_gap_fingerprints(
            _ReleaseRepositoryView()
        )
    )
    replayer = IndexEventReplayer(anchors, events, overlays)
    anchor_dates = pd.to_datetime(
        anchors.loc[anchors["index_id"].astype(str).isin(profiles), "anchor_date"],
        errors="coerce",
    ).dropna()
    if anchor_dates.empty:
        raise ValueError(f"No index anchors found for profiles: {', '.join(profiles)}")
    # All selected indices must have an available anchor at the schedule start.
    first_by_profile = [
        pd.to_datetime(
            anchors.loc[anchors["index_id"].astype(str) == profile, "anchor_date"],
            errors="coerce",
        ).min()
        for profile in profiles
    ]
    if any(pd.isna(value) for value in first_by_profile):
        raise ValueError(f"No index anchor found for one of: {', '.join(profiles)}")
    start = max(first_by_profile).normalize()
    end = pd.Timestamp(as_of).normalize()
    change_dates = {start, end}
    relevant_anchors = anchors.loc[
        anchors["index_id"].astype(str).isin(profiles)
    ]
    relevant_events = events.loc[
        events["index_id"].astype(str).isin(profiles)
    ]
    relevant_security_ids = set(relevant_anchors["security_id"].astype(str)) | set(
        relevant_events["security_id"].astype(str)
    )
    for value in pd.to_datetime(
        relevant_events["effective_date"],
        errors="coerce",
    ).dropna():
        normalized = value.normalize()
        if start <= normalized <= end:
            change_dates.add(normalized)
    if not overlays.empty:
        relevant = overlays.loc[overlays["index_id"].astype(str).isin(profiles)]
        if "security_id" in relevant:
            relevant_security_ids.update(relevant["security_id"].astype(str))
        for column in ("effective_from", "effective_to"):
            for value in pd.to_datetime(relevant[column], errors="coerce").dropna():
                normalized = value.normalize()
                if start <= normalized <= end:
                    change_dates.add(normalized)

    # Membership is keyed by stable security ID, but the strategy schedule is
    # keyed by ticker.  A constituent can change ticker without an index event;
    # include those symbol-history boundaries so an existing position and its
    # corporate actions move to the new symbol on the correct session.
    relevant_history = symbol_history.loc[
        symbol_history["security_id"].astype(str).isin(relevant_security_ids)
    ]
    for value in pd.to_datetime(
        relevant_history["effective_from"], errors="coerce"
    ).dropna():
        normalized = value.normalize()
        if start <= normalized <= end:
            change_dates.add(normalized)

    entries: list[UniverseScheduleEntry] = []
    prior_identity_membership: tuple[tuple[str, str], ...] | None = None
    for effective_date in sorted(change_dates):
        memberships = [
            replayer.members_on(
                profile,
                effective_date,
                source_mode=config.data_store.index_source_mode,
            )
            for profile in profiles
        ]
        profile_by_security: dict[str, set[str]] = {}
        for profile, membership in zip(profiles, memberships):
            for security_id in membership.security_ids:
                profile_by_security.setdefault(security_id, set()).add(profile)
        members = _unique_members(
            _index_security_member(
                security_id,
                effective_date,
                tuple(sorted(member_profiles)),
                symbol_history,
                security_master,
                events,
                reviewed_identity_gaps,
            )
            for security_id, member_profiles in profile_by_security.items()
        )
        identity_membership = tuple(
            (member.symbol, member.security_id)
            for member in members
        )
        if identity_membership != prior_identity_membership:
            entries.append(UniverseScheduleEntry(effective_date.date().isoformat(), members))
            prior_identity_membership = identity_membership
    if not entries:
        raise ValueError("Index event replay produced no universe schedule.")
    return tuple(entries)


def _index_security_member(
    security_id: str,
    effective_date: pd.Timestamp,
    profiles: tuple[str, ...],
    symbol_history: pd.DataFrame,
    security_master: pd.DataFrame,
    index_events: pd.DataFrame | None = None,
    reviewed_identity_gap_fingerprints: Iterable[str] = (),
) -> UniverseMember:
    history = symbol_history.loc[symbol_history["security_id"].astype(str) == security_id].copy()
    starts = pd.to_datetime(history["effective_from"], errors="coerce")
    ends = pd.to_datetime(history["effective_to"], errors="coerce")
    active = history.loc[(starts <= effective_date) & (ends.isna() | (ends >= effective_date))]
    if active.empty:
        if not _reviewed_inactive_index_member_allowed(
            security_id=security_id,
            effective_date=effective_date,
            profiles=profiles,
            index_events=index_events,
            reviewed_identity_gap_fingerprints=reviewed_identity_gap_fingerprints,
        ):
            raise ValueError(
                f"No active symbol for security_id={security_id} on {effective_date.date()}"
            )
        # The exact reviewed operational exception represents an index source
        # whose REMOVE date lags a legal terminal transition.  Keep the final
        # historical alias only as the schedule's display identity; the
        # corporate-action ledger retires it and the runner's stale-schedule
        # guard prevents any post-transition re-entry.
        historical = history.loc[starts <= effective_date]
        if historical.empty:
            raise ValueError(
                f"No historical symbol for reviewed security_id={security_id} "
                f"on {effective_date.date()}"
            )
        symbol_row = historical.sort_values("effective_from").iloc[-1]
    else:
        symbol_row = active.sort_values("effective_from").iloc[-1]
    master = security_master.loc[
        security_master["security_id"].astype(str) == security_id
    ] if not security_master.empty else pd.DataFrame()
    master_row = master.iloc[-1] if not master.empty else {}
    symbol = str(symbol_row["symbol"])
    exchange = str(symbol_row.get("exchange") or master_row.get("exchange") or "US")
    return UniverseMember(
        symbol=symbol,
        market="US",
        exchange=exchange,
        security_id=security_id,
        name=str(master_row.get("name") or ""),
        security_type=str(master_row.get("asset_type") or "STOCK"),
        yfinance_symbol=_default_yfinance_symbol(symbol, exchange),
        benchmark=_profile_benchmark(profiles, "US"),
        profiles=profiles,
    )


def _reviewed_inactive_index_member_allowed(
    *,
    security_id: str,
    effective_date: pd.Timestamp,
    profiles: tuple[str, ...],
    index_events: pd.DataFrame | None,
    reviewed_identity_gap_fingerprints: Iterable[str],
) -> bool:
    """Allow only an exact, release-bound reviewed index identity gap."""

    allowed = {
        str(value).strip().lower()
        for value in reviewed_identity_gap_fingerprints
        if str(value).strip()
    }
    required_columns = {
        "event_id",
        "index_id",
        "effective_date",
        "operation",
        "security_id",
        "source",
        "source_hash",
    }
    if (
        not allowed
        or not profiles
        or index_events is None
        or index_events.empty
        or not required_columns.issubset(index_events.columns)
    ):
        return False

    from .market_store.validation import index_member_identity_gap_fingerprint

    replay_date = effective_date.date().isoformat()
    prepared = index_events.copy()
    prepared["_effective_date"] = pd.to_datetime(
        prepared["effective_date"], errors="coerce"
    ).dt.normalize()
    prepared = prepared.loc[
        prepared["security_id"].astype(str).eq(security_id)
        & prepared["operation"].astype(str).str.upper().eq("REMOVE")
        & prepared["_effective_date"].notna()
        & (prepared["_effective_date"] > effective_date.normalize())
    ]

    for index_id in profiles:
        removals = prepared.loc[
            prepared["index_id"].astype(str).eq(index_id)
        ].sort_values(["_effective_date", "event_id"], kind="stable")
        if removals.empty:
            return False
        next_remove = removals.iloc[0]
        fingerprint = index_member_identity_gap_fingerprint(
            index_id=index_id,
            replay_date=replay_date,
            security_id=security_id,
            next_remove_event_id=str(next_remove.get("event_id", "") or ""),
            next_remove_effective_date=pd.Timestamp(
                next_remove["_effective_date"]
            ).date().isoformat(),
            next_remove_source=str(next_remove.get("source", "") or ""),
            next_remove_source_hash=str(
                next_remove.get("source_hash", "") or ""
            ),
        )
        if fingerprint not in allowed:
            return False
    return True


def _load_file_members(path: str, market: str) -> tuple[UniverseMember, ...]:
    resolved = _resolve_existing_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if market == "US":
        return tuple(
            _member_for_symbol(symbol, market, (), str(resolved))
            for symbol in sorted(set(raw.get("US_UNIVERSE_LIST", ())))
        )
    market_map = raw.get("KR_UNIVERSE_MAP", {})
    return tuple(
        UniverseMember(
            symbol=str(symbol),
            market="KR",
            exchange=str(exchange),
            yfinance_symbol=_default_yfinance_symbol(str(symbol), str(exchange)),
            benchmark=_benchmark_for_exchange(str(exchange)),
        )
        for symbol, exchange in sorted(market_map.items())
    )


def _load_history_file_schedule(
    path: str,
    market: str,
    universe_file: str,
) -> tuple[UniverseScheduleEntry, ...]:
    resolved = _resolve_existing_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        root: Mapping[str, Any] = {"snapshots": raw}
    elif isinstance(raw, Mapping):
        root = raw
    else:
        raise ValueError("universe.history_file must contain a JSON mapping or list.")

    configured_market = str(root.get("market") or market).upper()
    if configured_market != market:
        raise ValueError(f"history_file market={configured_market} does not match market={market}.")
    default_profiles = _history_profiles(root)
    snapshots = root.get("snapshots") or root.get("history") or ()
    if not isinstance(snapshots, list):
        raise ValueError("history_file snapshots must be a list.")

    entries: list[UniverseScheduleEntry] = []
    seen_dates: set[str] = set()
    for raw_snapshot in snapshots:
        if not isinstance(raw_snapshot, Mapping):
            raise ValueError("Each history_file snapshot must be a mapping.")
        effective_date = _history_effective_date(raw_snapshot)
        if effective_date in seen_dates:
            raise ValueError(f"Duplicate history_file effective_date: {effective_date}")
        seen_dates.add(effective_date)
        profiles = _history_profiles(raw_snapshot) or default_profiles
        raw_members = raw_snapshot.get("members", raw_snapshot.get("symbols", ()))
        if not isinstance(raw_members, list):
            raise ValueError(f"history_file snapshot {effective_date} members/symbols must be a list.")
        members = _unique_members(
            _history_member(item, market, profiles, universe_file)
            for item in raw_members
        )
        if not members:
            raise ValueError(f"history_file snapshot {effective_date} has no members.")
        entries.append(UniverseScheduleEntry(effective_date, members))

    if not entries:
        raise ValueError("history_file must contain at least one snapshot.")
    return tuple(sorted(entries, key=lambda item: item.effective_date))


def _history_effective_date(raw: Mapping[str, Any]) -> str:
    value = raw.get("effective_date") or raw.get("date") or raw.get("as_of")
    if not value:
        raise ValueError("Each history_file snapshot requires effective_date.")
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception as exc:
        raise ValueError(f"Invalid history_file effective_date: {value}") from exc


def _history_profiles(raw: Mapping[str, Any]) -> tuple[str, ...]:
    value = raw.get("profiles", raw.get("profile", ()))
    if isinstance(value, str):
        return (value.strip().lower(),) if value.strip() else ()
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("history_file profiles/profile must be a string or list.")
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _history_member(
    item: Any,
    market: str,
    profiles: tuple[str, ...],
    universe_file: str,
) -> UniverseMember:
    if isinstance(item, str):
        return _member_for_symbol(item.strip(), market, profiles, universe_file)
    if not isinstance(item, Mapping):
        raise ValueError("history_file members must be ticker strings or mappings.")
    symbol = str(item.get("symbol") or "").strip()
    if not symbol:
        raise ValueError("history_file member mapping requires symbol.")
    member_market = str(item.get("market") or market).upper()
    exchange = str(item.get("exchange") or ("KOSPI" if member_market == "KR" else member_market))
    member_profiles = _history_profiles(item) or profiles
    return UniverseMember(
        symbol=symbol,
        market=member_market,
        exchange=exchange,
        name=str(item.get("name") or ""),
        security_type=str(item.get("security_type") or "STOCK"),
        yfinance_symbol=str(item.get("yfinance_symbol") or _default_yfinance_symbol(symbol, exchange)),
        benchmark=str(
            item.get("benchmark")
            or (_benchmark_for_exchange(exchange) if member_market == "KR" else _profile_benchmark(member_profiles, member_market))
        ),
        profiles=member_profiles,
    )


def _unique_members(members: Iterable[UniverseMember]) -> tuple[UniverseMember, ...]:
    by_symbol: dict[str, UniverseMember] = {}
    for member in members:
        if member.symbol:
            existing = by_symbol.get(member.symbol)
            if (
                existing is not None
                and existing.security_id
                and member.security_id
                and existing.security_id != member.security_id
            ):
                raise ValueError(
                    "One ticker maps to multiple security identities in the same "
                    f"universe snapshot: {member.symbol} -> "
                    f"{existing.security_id}, {member.security_id}"
                )
            by_symbol[member.symbol] = member
    return tuple(by_symbol[symbol] for symbol in sorted(by_symbol))


def _schedule_profiles(schedule: tuple[UniverseScheduleEntry, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                profile
                for entry in schedule
                for member in entry.members
                for profile in member.profiles
            }
        )
    )


def _member_for_symbol(symbol: str, market: str, profiles: tuple[str, ...], universe_file: str) -> UniverseMember:
    exchange = market
    if market == "KR":
        exchange = "KOSPI"
        try:
            members = _load_file_members(universe_file, "KR")
            known = {member.symbol: member for member in members}.get(symbol)
            if known is not None:
                return known
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return UniverseMember(
        symbol=symbol,
        market=market,
        exchange=exchange,
        yfinance_symbol=_default_yfinance_symbol(symbol, exchange),
        benchmark=_profile_benchmark(profiles, market),
        profiles=profiles,
    )


def _apply_filters(
    raw_members: tuple[UniverseMember, ...],
    filters: UniverseFilterConfig,
    as_of: date,
    market: str,
    *,
    price_loader: PriceHistoryLoader | None,
    status_loader: StatusLoader | None,
) -> tuple[tuple[UniverseMember, ...], tuple[dict[str, Any], ...]]:
    if not filters.enabled:
        return raw_members, ()
    managed: set[str] = set()
    delisted: set[str] = set()
    if market == "KR" and (filters.exclude_managed or filters.exclude_delisting):
        managed, delisted = (status_loader or _load_kr_status)(as_of)
    history = (price_loader or _download_daily_history)(
        raw_members,
        max(filters.min_history_daily_bars, filters.avg_turnover_window),
    )
    latest_dates = [
        pd.Timestamp(frame.dropna(subset=["Close"]).index[-1]).date()
        for frame in history.values()
        if not frame.empty and "Close" in frame and not frame.dropna(subset=["Close"]).empty
    ]
    reference_latest = max(latest_dates) if latest_dates else None
    eligible: list[UniverseMember] = []
    rejected: list[dict[str, Any]] = []
    for member in raw_members:
        reasons: list[str] = []
        name_upper = member.name.upper()
        type_upper = member.security_type.upper()
        if filters.exclude_managed and member.symbol in managed:
            reasons.append("managed")
        if filters.exclude_delisting and member.symbol in delisted:
            reasons.append("delisting")
        if filters.exclude_etf_etn and (
            type_upper in {"ETF", "ETN"}
            or re.search(r"(?:^|\s)(?:ETF|ETN)(?:\s|$)", name_upper)
        ):
            reasons.append("etf_etn")
        if filters.exclude_spac and (
            "SPAC" in name_upper or "ACQUISITION CORP" in name_upper
        ):
            reasons.append("spac")
        if filters.exclude_preferred and _is_preferred(member):
            reasons.append("preferred")

        frame = history.get(member.symbol)
        if frame is None or frame.empty or not {"Close", "Volume"}.issubset(frame.columns):
            reasons.append("missing_history")
        else:
            valid = frame[["Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
            valid = valid.replace([math.inf, -math.inf], float("nan")).dropna()
            if len(valid) < filters.min_history_daily_bars:
                reasons.append("insufficient_history")
            if valid.empty:
                reasons.append("invalid_history")
            else:
                latest_date = pd.Timestamp(valid.index[-1]).date()
                if filters.exclude_suspended and reference_latest and latest_date < reference_latest:
                    reasons.append("suspended_or_stale")
                latest_close = float(valid["Close"].iloc[-1])
                if latest_close < filters.min_price.get(market, 0.0):
                    reasons.append("min_price")
                if len(valid) < filters.avg_turnover_window:
                    reasons.append("turnover_window")
                else:
                    turnover = (valid["Close"] * valid["Volume"]).tail(filters.avg_turnover_window).mean()
                    if not math.isfinite(float(turnover)):
                        reasons.append("invalid_turnover")
                    elif float(turnover) < filters.min_avg_turnover.get(market, 0.0):
                        reasons.append("min_avg_turnover")
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            rejected.append({"symbol": member.symbol, "reasons": reasons})
        else:
            eligible.append(member)
    return tuple(eligible), tuple(rejected)


def _is_preferred(member: UniverseMember) -> bool:
    name = member.name.strip().upper()
    security_type = member.security_type.upper()
    if "PREFERRED" in security_type or "PREFERRED" in name or "PREF " in name:
        return True
    if member.market == "KR":
        return bool(re.search(r"우(?:B|C)?$", member.name.strip(), flags=re.IGNORECASE))
    return False


def _download_daily_history(
    members: tuple[UniverseMember, ...],
    required_bars: int,
) -> Mapping[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError("yfinance is required for universe filters.") from exc
    by_yf = {member.yfinance_symbol or member.symbol: member.symbol for member in members}
    out: dict[str, pd.DataFrame] = {}
    period = "2y" if required_bars > 180 else "1y"
    for batch in _batched(tuple(by_yf), 100):
        raw = yf.download(
            tickers=list(batch),
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        for yf_symbol in batch:
            frame = _extract_close_volume(raw, yf_symbol)
            if not frame.empty:
                out[by_yf[yf_symbol]] = frame
    return out


def _extract_close_volume(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol in raw.columns.get_level_values(0):
            frame = raw[symbol].copy()
        elif symbol in raw.columns.get_level_values(1):
            frame = raw.xs(symbol, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = raw.copy()
    if not {"Close", "Volume"}.issubset(frame.columns):
        return pd.DataFrame()
    return frame[["Close", "Volume"]].copy()


def _load_kr_status(as_of: date) -> tuple[set[str], set[str]]:
    try:
        import FinanceDataReader as fdr
    except ModuleNotFoundError as exc:
        raise RuntimeError("finance-datareader is required for KRX status filters.") from exc
    managed = _symbols_from_listing(fdr.StockListing("KRX-ADMIN"))
    delisted_frame = fdr.StockListing("KRX-DELISTING", str(as_of.year - 2), str(as_of))
    delisted = _symbols_from_listing(delisted_frame)
    return managed, delisted


def _symbols_from_listing(frame: pd.DataFrame) -> set[str]:
    for column in ("Code", "Symbol", "종목코드", "단축코드"):
        if column in frame.columns:
            return {str(value).strip().zfill(6) for value in frame[column].dropna()}
    return set()


def _build_snapshot(
    as_of: date,
    market: str,
    source: str,
    profiles: tuple[str, ...],
    selection_hash: str,
    raw_members: tuple[UniverseMember, ...],
    eligible_members: tuple[UniverseMember, ...],
    rejected: tuple[dict[str, Any], ...],
    filters: UniverseFilterConfig,
) -> UniverseSnapshot:
    return UniverseSnapshot(
        schema_version=1,
        as_of=as_of.isoformat(),
        created_at=datetime.now(timezone.utc).isoformat(),
        market=market,
        source=source,
        profiles=profiles,
        selection_hash=selection_hash,
        raw_members=raw_members,
        eligible_members=eligible_members,
        rejected=rejected,
        filters=asdict(filters),
    )


def _selection_hash(
    config: AppConfig,
    market: str,
    source: str,
    profiles: tuple[str, ...],
    explicit_symbols: tuple[str, ...],
) -> str:
    payload = {
        "market": market,
        "source": source,
        "profiles": profiles,
        "file": config.universe.file,
        "history_file": config.universe.history_file,
        "symbols": explicit_symbols or config.universe.symbols,
        "filters": asdict(config.universe.filters),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _snapshot_path(root: str, market: str, selection_hash: str, as_of: date) -> Path:
    return Path(root).expanduser() / market / selection_hash / f"{as_of.isoformat()}.json"


def _write_snapshot(path: Path, snapshot: UniverseSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_snapshot(path: Path) -> UniverseSnapshot:
    return UniverseSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _latest_snapshot(root: str, market: str, selection_hash: str) -> UniverseSnapshot | None:
    directory = Path(root).expanduser() / market / selection_hash
    paths = sorted(directory.glob("*.json")) if directory.exists() else []
    return _read_snapshot(paths[-1]) if paths else None


def _profile_benchmark(profiles: tuple[str, ...], market: str) -> str:
    if market == "KR":
        return "^KS11"
    if len(profiles) != 1:
        return "SPY" if profiles else "QQQ"
    return {
        "nasdaq100": "QQQ",
        "sp500": "SPY",
        "russell3000": "IWV",
        "dow30": "DIA",
    }.get(profiles[0], "QQQ")


def _benchmark_for_exchange(exchange: str) -> str:
    return "^KQ11" if str(exchange).upper() == "KOSDAQ" else "^KS11"


def _default_yfinance_symbol(symbol: str, exchange: str) -> str:
    normalized_exchange = str(exchange).upper()
    if normalized_exchange == "KOSPI":
        return f"{symbol}.KS"
    if normalized_exchange == "KOSDAQ":
        return f"{symbol}.KQ"
    return symbol.replace(".", "-")


def _market_date(market: str) -> date:
    zone = ZoneInfo("Asia/Seoul") if market == "KR" else ZoneInfo("America/New_York")
    return datetime.now(zone).date()


def _batched(values: tuple[str, ...], size: int) -> Iterable[tuple[str, ...]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _frame_members(frame: pd.DataFrame, market: str, exchange: str, profile: str) -> tuple[UniverseMember, ...]:
    symbol_column = _find_column(frame, ("Symbol", "Ticker", "종목코드", "단축코드", "Code"))
    if symbol_column is None:
        raise ValueError(f"No symbol column found for universe profile={profile}")
    name_column = _find_column(frame, ("Name", "Security", "Company", "종목명", "한글종목명"))
    type_column = _find_column(frame, ("Type", "Security Type", "종목구분"))
    members = []
    for _, row in frame.iterrows():
        symbol = str(row[symbol_column]).strip()
        if not symbol or symbol.lower() == "nan":
            continue
        if market == "KR":
            symbol = symbol.split(".")[0].zfill(6)
        name = str(row[name_column]).strip() if name_column is not None and pd.notna(row[name_column]) else ""
        security_type = (
            str(row[type_column]).strip()
            if type_column is not None and pd.notna(row[type_column])
            else "STOCK"
        )
        members.append(
            UniverseMember(
                symbol=symbol,
                market=market,
                exchange=exchange,
                name=name,
                security_type=security_type,
                yfinance_symbol=_default_yfinance_symbol(symbol, exchange),
                profiles=(profile,),
            )
        )
    return tuple(members)


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> Any | None:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def _read_public_tables(urls: tuple[str, ...]) -> list[pd.DataFrame]:
    import requests

    errors = []
    for url in urls:
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            if tables:
                return tables
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Unable to load public constituent tables: " + "; ".join(errors))


def _public_members(
    urls: tuple[str, ...],
    *,
    market: str,
    exchange: str,
    profile: str,
    minimum_count: int,
    maximum_count: int,
) -> tuple[UniverseMember, ...]:
    errors: list[str] = []
    for url in urls:
        try:
            for frame in _read_public_tables((url,)):
                copy = frame.copy()
                if isinstance(copy.columns, pd.MultiIndex):
                    copy.columns = [str(column[-1]) for column in copy.columns]
                if _find_column(copy, ("Symbol", "Ticker")) is None:
                    continue
                members = _frame_members(copy, market, exchange, profile)
                if minimum_count <= len(members) <= maximum_count:
                    return members
            errors.append(f"{url}: no table in expected count range")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Unable to load {profile} constituents: " + "; ".join(errors))


@register_universe_provider("nasdaq100")
def _nasdaq100_provider(as_of: date) -> Iterable[UniverseMember]:
    try:
        return _nasdaq100_api_members()
    except Exception:
        pass
    return _public_members(
        (
            "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies",
            "https://en.wikipedia.org/wiki/Nasdaq-100",
        ),
        market="US",
        exchange="NASDAQ",
        profile="nasdaq100",
        minimum_count=90,
        maximum_count=110,
    )


def _nasdaq100_api_members() -> tuple[UniverseMember, ...]:
    import requests

    response = requests.get(
        "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
        timeout=30,
        headers={
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "Mozilla/5.0",
        },
    )
    response.raise_for_status()
    payload = response.json()
    rows = (((payload.get("data") or {}).get("data") or {}).get("rows")) or ()
    members = []
    for row in rows:
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        members.append(
            UniverseMember(
                symbol=symbol.replace("/", "-"),
                market="US",
                exchange="NASDAQ",
                name=str(row.get("companyName") or ""),
                yfinance_symbol=symbol.replace("/", "-"),
                benchmark="QQQ",
                profiles=("nasdaq100",),
            )
        )
    if not 90 <= len(members) <= 110:
        raise RuntimeError(f"unexpected Nasdaq-100 member count: {len(members)}")
    return tuple(members)


@register_universe_provider("sp500")
def _sp500_provider(as_of: date) -> Iterable[UniverseMember]:
    try:
        import FinanceDataReader as fdr

        frame = fdr.StockListing("S&P500")
        members = _frame_members(frame, "US", "US", "sp500")
        if not 450 <= len(members) <= 550:
            raise ValueError(f"unexpected S&P 500 member count: {len(members)}")
        return members
    except Exception:
        return _public_members(
            ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",),
            market="US",
            exchange="US",
            profile="sp500",
            minimum_count=450,
            maximum_count=550,
        )


@register_universe_provider("dow30")
def _dow30_provider(as_of: date) -> Iterable[UniverseMember]:
    return _public_members(
        (
            "https://www.spglobal.com/spdji/en/indices/equity/dow-jones-industrial-average/",
            "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        ),
        market="US",
        exchange="US",
        profile="dow30",
        minimum_count=30,
        maximum_count=30,
    )


@register_universe_provider("russell3000")
def _russell3000_provider(as_of: date) -> Iterable[UniverseMember]:
    raise RuntimeError(
        "Russell 3000 requires universe.source=index_events with stored point-in-time data."
    )


def _kr_index_provider(profile: str, expected_name: str, market: str, as_of: date) -> Iterable[UniverseMember]:
    try:
        from pykrx import stock
    except ModuleNotFoundError as exc:
        raise RuntimeError("pykrx is required for KRX index universe profiles.") from exc
    date_text = as_of.strftime("%Y%m%d")
    index_ticker = next(
        (
            ticker
            for ticker in stock.get_index_ticker_list(date_text, market=market)
            if re.sub(r"\s+", "", stock.get_index_ticker_name(ticker)).lower()
            == re.sub(r"\s+", "", expected_name).lower()
        ),
        None,
    )
    if index_ticker is None:
        raise RuntimeError(f"KRX index not found: {expected_name}")
    symbols = stock.get_index_portfolio_deposit_file(index_ticker, date_text)
    expected_range = (180, 220) if profile == "kospi200" else (130, 170)
    if not expected_range[0] <= len(symbols) <= expected_range[1]:
        raise RuntimeError(f"Unexpected {profile} member count: {len(symbols)}")
    return tuple(
        UniverseMember(
            symbol=str(symbol).zfill(6),
            market="KR",
            exchange=market,
            name=str(stock.get_market_ticker_name(str(symbol))) or "",
            yfinance_symbol=_default_yfinance_symbol(str(symbol).zfill(6), market),
            profiles=(profile,),
        )
        for symbol in symbols
    )


@register_universe_provider("kospi200")
def _kospi200_provider(as_of: date) -> Iterable[UniverseMember]:
    return _kr_index_provider("kospi200", "코스피 200", "KOSPI", as_of)


@register_universe_provider("kosdaq150")
def _kosdaq150_provider(as_of: date) -> Iterable[UniverseMember]:
    return _kr_index_provider("kosdaq150", "코스닥 150", "KOSDAQ", as_of)


__all__ = [
    "PriceHistoryLoader",
    "ResolvedUniverse",
    "StatusLoader",
    "UniverseMember",
    "UniverseProvider",
    "UniverseScheduleEntry",
    "UniverseSnapshot",
    "available_universe_profiles",
    "register_universe_provider",
    "resolve_universe",
    "snapshot_summary",
    "universe_request_key",
]
