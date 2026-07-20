from __future__ import annotations

from dataclasses import dataclass


SOURCE_COLUMNS = ("source", "retrieved_at", "source_hash")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    primary_key: tuple[str, ...]
    required_columns: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    partition_columns: tuple[str, ...] = ()


def _spec(
    name: str,
    primary_key: tuple[str, ...],
    columns: tuple[str, ...],
    *,
    dates: tuple[str, ...] = (),
    partitions: tuple[str, ...] = (),
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        primary_key=primary_key,
        required_columns=tuple(dict.fromkeys((*columns, *SOURCE_COLUMNS))),
        date_columns=dates,
        partition_columns=partitions,
    )


DATASET_SPECS: dict[str, DatasetSpec] = {
    "security_master": _spec(
        "security_master",
        ("security_id",),
        (
            "security_id",
            "primary_symbol",
            "name",
            "exchange",
            "asset_type",
            "currency",
            "country",
            "active_from",
            "active_to",
        ),
        dates=("active_from", "active_to"),
    ),
    "symbol_history": _spec(
        "symbol_history",
        ("security_id", "symbol", "effective_from"),
        (
            "security_id",
            "symbol",
            "exchange",
            "effective_from",
            "effective_to",
        ),
        dates=("effective_from", "effective_to"),
    ),
    "daily_price_raw": _spec(
        "daily_price_raw",
        ("security_id", "session"),
        (
            "security_id",
            "session",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "currency",
        ),
        dates=("session",),
        partitions=("year", "month"),
    ),
    "corporate_actions": _spec(
        "corporate_actions",
        ("event_id",),
        (
            "event_id",
            "security_id",
            "action_type",
            "effective_date",
            "ex_date",
            "announcement_date",
            "record_date",
            "payment_date",
            "cash_amount",
            "ratio",
            "currency",
            "new_security_id",
            "new_symbol",
            "official",
            "source_url",
            "source_kind",
        ),
        dates=("effective_date", "ex_date", "announcement_date", "record_date", "payment_date"),
        partitions=("year",),
    ),
    "lifecycle_resolutions": _spec(
        "lifecycle_resolutions",
        ("candidate_id",),
        (
            "candidate_id",
            "security_id",
            "symbol",
            "last_price_date",
            "resolution",
            "event_id",
            "exception_code",
            "exception_reason",
            "reviewed_by",
            "reviewed_at",
            "recheck_after",
            "successor_security_id",
            "successor_symbol",
            "source_url",
        ),
        dates=("last_price_date", "reviewed_at", "recheck_after"),
    ),
    "cross_validation_reports": _spec(
        "cross_validation_reports",
        ("report_id",),
        (
            "report_id",
            "base_release_version",
            "validated_at",
            "status",
            "provider",
            "policy_sha256",
            "lifecycle_evidence_report_sha256",
            "validated_versions_json",
            "event_count",
            "event_mismatch_count",
            "nonterminal_event_count",
            "reviewed_nonterminal_event_count",
            "permanent_exception_count",
            "permanent_exception_mismatch_count",
            "price_target_count",
            "price_pass_count",
            "price_exception_count",
            "price_unresolved_count",
            "price_mismatch_count",
            "overlap_session_count",
            "report_archive_id",
        ),
        dates=("validated_at",),
    ),
    "adjustment_factors": _spec(
        "adjustment_factors",
        ("security_id", "session"),
        (
            "security_id",
            "session",
            "split_factor",
            "total_return_factor",
            "source_version",
            "calculated_at",
        ),
        dates=("session",),
        partitions=("year",),
    ),
    "index_constituent_anchors": _spec(
        "index_constituent_anchors",
        ("index_id", "anchor_date", "security_id"),
        (
            "index_id",
            "anchor_date",
            "security_id",
            "official",
            "source_url",
            "source_kind",
        ),
        dates=("anchor_date",),
    ),
    "index_membership_events": _spec(
        "index_membership_events",
        ("event_id",),
        (
            "event_id",
            "index_id",
            "announcement_date",
            "effective_date",
            "operation",
            "security_id",
            "official",
            "source_url",
            "source_kind",
        ),
        dates=("effective_date", "announcement_date"),
        partitions=("year",),
    ),
    "custom_universe_overlays": _spec(
        "custom_universe_overlays",
        ("overlay_id",),
        (
            "overlay_id",
            "index_id",
            "effective_from",
            "effective_to",
            "operation",
            "security_id",
            "reason",
            "source_url",
            "source_kind",
        ),
        dates=("effective_from", "effective_to"),
    ),
    "source_archive": _spec(
        "source_archive",
        ("archive_id",),
        (
            "archive_id",
            "dataset",
            "object_path",
            "content_type",
            "effective_date",
        ),
        dates=("effective_date",),
    ),
}


def dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset {name!r}. Available: {available}") from exc
