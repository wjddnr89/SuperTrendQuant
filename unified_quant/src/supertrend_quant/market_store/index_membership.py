from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class IndexMembership:
    index_id: str
    as_of: str
    anchor_date: str
    security_ids: tuple[str, ...]
    applied_event_ids: tuple[str, ...]
    applied_overlay_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class IndexEventReplayer:
    def __init__(
        self,
        anchors: pd.DataFrame,
        events: pd.DataFrame,
        overlays: pd.DataFrame | None = None,
    ):
        self.anchors = anchors.copy()
        self.events = events.copy()
        self.overlays = overlays.copy() if overlays is not None else pd.DataFrame()

    def members_on(
        self,
        index_id: str,
        as_of: str | date | pd.Timestamp,
        *,
        source_mode: str = "best_effort",
    ) -> IndexMembership:
        if source_mode not in {"best_effort", "official_only"}:
            raise ValueError("source_mode must be best_effort or official_only.")
        cutoff = pd.Timestamp(as_of).normalize()
        anchors = self.anchors.loc[self.anchors["index_id"].astype(str) == index_id].copy()
        anchors["_date"] = pd.to_datetime(anchors["anchor_date"], errors="coerce").dt.normalize()
        eligible = anchors.loc[anchors["_date"] <= cutoff]
        if eligible.empty:
            raise ValueError(f"No index anchor for {index_id} on or before {cutoff.date()}")
        anchor_date = eligible["_date"].max()
        anchor_rows = eligible.loc[eligible["_date"] == anchor_date]
        if source_mode == "official_only" and not anchor_rows["official"].fillna(False).astype(bool).all():
            raise ValueError(
                f"official_only coverage is incomplete: {index_id} anchor {anchor_date.date()} "
                "contains non-official rows"
            )
        members = set(anchor_rows["security_id"].astype(str))

        events = self.events.loc[self.events["index_id"].astype(str) == index_id].copy()
        events["_date"] = pd.to_datetime(events["effective_date"], errors="coerce").dt.normalize()
        events = events.loc[(events["_date"] > anchor_date) & (events["_date"] <= cutoff)]
        events, conflict_warnings = _resolve_event_conflicts(events)
        if source_mode == "official_only":
            unofficial = events.loc[~events["official"].fillna(False).astype(bool)]
            if not unofficial.empty:
                first = unofficial.sort_values(["_date", "event_id"]).iloc[0]
                raise ValueError(
                    "official_only coverage is incomplete: "
                    f"{index_id} has non-official-only event {first['event_id']} "
                    f"on {pd.Timestamp(first['_date']).date()}"
                )
            events = events.loc[events["official"].fillna(False).astype(bool)]
        events = events.sort_values(["_date", "event_id"], kind="stable")
        warnings: list[str] = list(conflict_warnings)
        applied: list[str] = []
        for row in events.itertuples(index=False):
            operation = str(row.operation).upper()
            security_id = str(row.security_id)
            if operation == "ADD":
                if security_id in members:
                    warnings.append(f"Duplicate ADD ignored: {row.event_id}")
                members.add(security_id)
            elif operation == "REMOVE":
                if security_id not in members:
                    warnings.append(f"Missing REMOVE ignored: {row.event_id}")
                members.discard(security_id)
            else:
                raise ValueError(f"Invalid index operation: {operation}")
            applied.append(str(row.event_id))

        overlay_ids: list[str] = []
        if not self.overlays.empty:
            overlays = self.overlays.loc[self.overlays["index_id"].astype(str) == index_id].copy()
            starts = pd.to_datetime(overlays["effective_from"], errors="coerce").dt.normalize()
            ends = pd.to_datetime(overlays["effective_to"], errors="coerce").dt.normalize()
            overlays = overlays.loc[(starts <= cutoff) & (ends.isna() | (ends >= cutoff))]
            overlays = overlays.sort_values(["effective_from", "overlay_id"], kind="stable")
            for row in overlays.itertuples(index=False):
                if str(row.operation).upper() == "ADD":
                    members.add(str(row.security_id))
                elif str(row.operation).upper() == "REMOVE":
                    members.discard(str(row.security_id))
                else:
                    raise ValueError(f"Invalid overlay operation: {row.operation}")
                overlay_ids.append(str(row.overlay_id))

        return IndexMembership(
            index_id=index_id,
            as_of=cutoff.date().isoformat(),
            anchor_date=anchor_date.date().isoformat(),
            security_ids=tuple(sorted(members)),
            applied_event_ids=tuple(applied),
            applied_overlay_ids=tuple(overlay_ids),
            warnings=tuple(warnings),
        )

    def schedule(
        self,
        index_id: str,
        start: str | date | pd.Timestamp,
        end: str | date | pd.Timestamp,
        *,
        source_mode: str = "best_effort",
    ) -> tuple[IndexMembership, ...]:
        start_at = pd.Timestamp(start).normalize()
        end_at = pd.Timestamp(end).normalize()
        if end_at < start_at:
            raise ValueError("end must be on or after start.")
        dates = {start_at, end_at}
        events = self.events.loc[self.events["index_id"].astype(str) == index_id]
        for value in pd.to_datetime(events["effective_date"], errors="coerce").dropna():
            normalized = value.normalize()
            if start_at <= normalized <= end_at:
                dates.add(normalized)
        if not self.overlays.empty:
            overlays = self.overlays.loc[self.overlays["index_id"].astype(str) == index_id]
            for column in ("effective_from", "effective_to"):
                for value in pd.to_datetime(overlays[column], errors="coerce").dropna():
                    normalized = value.normalize()
                    if start_at <= normalized <= end_at:
                        dates.add(normalized)
        memberships: list[IndexMembership] = []
        prior: tuple[str, ...] | None = None
        for as_of in sorted(dates):
            current = self.members_on(index_id, as_of, source_mode=source_mode)
            if current.security_ids != prior:
                memberships.append(current)
                prior = current.security_ids
        return tuple(memberships)


def _resolve_event_conflicts(events: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Deduplicate semantic events and let official evidence beat lower-grade rows."""
    if events.empty:
        return events, ()
    selected: list[pd.DataFrame] = []
    warnings: list[str] = []
    for (effective_date, security_id), group in events.groupby(
        ["_date", "security_id"], sort=True, dropna=False
    ):
        operations = set(group["operation"].astype(str).str.upper())
        official = group.loc[group["official"].fillna(False).astype(bool)]
        official_operations = set(official["operation"].astype(str).str.upper())
        if len(operations) > 1:
            if len(official_operations) != 1:
                event_ids = ", ".join(sorted(group["event_id"].astype(str)))
                raise ValueError(
                    "Unresolved same-grade index event conflict for "
                    f"{security_id} on {pd.Timestamp(effective_date).date()}: {event_ids}"
                )
            chosen_operation = next(iter(official_operations))
            group = official.loc[
                official["operation"].astype(str).str.upper() == chosen_operation
            ]
            warnings.append(
                "Official index event overrode a conflicting non-official event: "
                f"{security_id} on {pd.Timestamp(effective_date).date()}"
            )
        elif not official.empty:
            group = official
        selected.append(group.sort_values("event_id", kind="stable").tail(1))
    return pd.concat(selected, ignore_index=True), tuple(warnings)
