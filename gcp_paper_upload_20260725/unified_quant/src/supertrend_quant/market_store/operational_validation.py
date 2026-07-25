"""Reviewed operational snapshot policy layered over strict validation.

``validate_repository_snapshot`` intentionally remains strict by default.  The
CLI and local mutation workflows use the wrapper in this module so the one
reviewed NBL/community-index replay gap is allowed only while every relevant
row still equals the repaired terminal state.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

import pandas as pd

from .validation import (
    ValidationReport,
    index_member_identity_gap_fingerprint,
    validate_repository_snapshot,
)


TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE: dict[str, Any] = {
    "security_id": "US:EODHD:3dd6d6ce-e7a1-5078-b258-df5b18404c9d",
    "symbol": "NBL",
    "last_real_session": "2020-10-02",
    "first_price_session": "2015-01-02",
    "retained_price_rows": 1_449,
    "retained_price_session_inventory_sha256": (
        "094be43b01ad4663abaf4f39e6aceeebc7c5ace8708d8692363764d43bbf563a"
    ),
    "market_transition_session": "2020-10-05",
    "event_id": "dd616ed974225aa84c3b537b9d21ce2a27e13a1f5af8c7e2488b765a007cac31",
    "candidate_id": "ac20b7b59563fe0b7f96a244092df17a3c5de3188d75144ff1c46cfa4ccb7955",
    "successor_security_id": "US:EODHD:fe5efe55-952b-5929-9eb0-f61c469314e8",
    "successor_symbol": "CVX",
    "ratio": "0.1191",
    "official_source_url": (
        "https://www.sec.gov/Archives/edgar/data/72207/"
        "000119312520263378/0001193125-20-263378.txt"
    ),
    "official_source_hash": (
        "fe5554317c372cb8fe924762d304049d0605b2f335ac4a3641bf2c22945ddffc"
    ),
    "official_retrieved_at": "2026-07-18T10:30:44.927075Z",
    "repair_reviewed_at": "2026-07-18T14:00:00Z",
    "raw_source_hash": (
        "f468258431303cd0278595d4457bac8af8e741cb2737cea142f28f1e25d5c5da"
    ),
    "raw_retrieved_at": "2026-07-16T15:56:47.970768Z",
    "terminal_ohlcv": (8.14, 8.51, 8.12, 8.46, 13_126_428.0),
    "master_active_from": "2015-01-02",
    "history_effective_from": "2015-01-01",
    "identity_exchange": "NASDAQ",
    "identity_source": "official_terminal_boundary_repair",
    "resolution_source": "terminal_boundary_repair",
    "resolution_reviewer": "terminal_boundary_repair_v1",
    "action_source": "sec_edgar+stored_price_crosscheck",
    "action_source_kind": "official_crosscheck",
    "index_id": "sp500",
    "replay_date": "2020-10-07",
    "next_remove_event_id": (
        "59d17bfad7dceb1c4903d45cc083841209982df638ec0f18006f2d3a7987d12d"
    ),
    "next_remove_effective_date": "2020-10-12",
    "next_remove_source": "community_sp500_history",
    "next_remove_source_url": (
        "https://raw.githubusercontent.com/fja05680/sp500/master/"
        "S%26P%20500%20Historical%20Components%20%26%20Changes%20"
        "%28Updated%29.csv"
    ),
    "next_remove_source_hash": (
        "39a9202c9ef69a74c0ff07e2113ad41fb6da7c8c5b6cd9541f0185fb4391e717"
    ),
    "fingerprint": (
        "989c5d44ef1b8cf8a682d807b63a62ebe3c3f38eb6f57e6314b3fe381d5c7d04"
    ),
}

_REQUIRED_DATASETS = (
    "corporate_actions",
    "lifecycle_resolutions",
    "daily_price_raw",
    "security_master",
    "symbol_history",
    "index_membership_events",
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else pd.Timestamp(parsed).date().isoformat()


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _required_frames(repository) -> dict[str, pd.DataFrame] | None:
    frames: dict[str, pd.DataFrame] = {}
    for dataset in _REQUIRED_DATASETS:
        if repository.current_manifest(dataset) is None:
            return None
        frame = repository.read_frame(dataset)
        if frame.empty:
            return None
        frames[dataset] = frame
    return frames


def _one_row(
    frame: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
    mask: pd.Series,
) -> Mapping[str, Any] | None:
    if not set(required_columns).issubset(frame.columns):
        return None
    rows = frame.loc[mask]
    return rows.iloc[0] if len(rows) == 1 else None


def _action_is_exact(frame: pd.DataFrame, expected: Mapping[str, Any]) -> bool:
    required = (
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
        "source",
        "retrieved_at",
        "source_hash",
    )
    if not {"security_id", "action_type"}.issubset(frame.columns):
        return False
    row = _one_row(
        frame,
        required_columns=required,
        mask=(
            frame["security_id"].astype(str).eq(expected["security_id"])
            & frame["action_type"].astype(str).eq("stock_merger")
        ),
    )
    if row is None:
        return False
    exact_text = {
        "event_id": expected["event_id"],
        "security_id": expected["security_id"],
        "action_type": "stock_merger",
        "currency": "USD",
        "new_security_id": expected["successor_security_id"],
        "new_symbol": expected["successor_symbol"],
        "source_url": expected["official_source_url"],
        "source_kind": expected["action_source_kind"],
        "source": expected["action_source"],
        "retrieved_at": expected["official_retrieved_at"],
        "source_hash": expected["official_source_hash"],
    }
    return bool(
        all(_text(row.get(field)) == value for field, value in exact_text.items())
        and _date(row.get("effective_date"))
        == expected["market_transition_session"]
        and _date(row.get("ex_date")) == expected["market_transition_session"]
        and _date(row.get("announcement_date"))
        == expected["market_transition_session"]
        and _date(row.get("record_date")) == ""
        and _date(row.get("payment_date")) == ""
        and _decimal(row.get("cash_amount")) is None
        and _decimal(row.get("ratio")) == Decimal(expected["ratio"])
        and _text(row.get("official")).lower() == "true"
        and _text(row.get("metadata")) == ""
    )


def _resolution_is_exact(frame: pd.DataFrame, expected: Mapping[str, Any]) -> bool:
    required = (
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
        "source",
        "retrieved_at",
        "source_hash",
    )
    if "security_id" not in frame.columns:
        return False
    row = _one_row(
        frame,
        required_columns=required,
        mask=frame["security_id"].astype(str).eq(expected["security_id"]),
    )
    if row is None:
        return False
    exact = {
        "candidate_id": expected["candidate_id"],
        "security_id": expected["security_id"],
        "symbol": expected["symbol"],
        "resolution": "applied",
        "event_id": expected["event_id"],
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": expected["resolution_reviewer"],
        "reviewed_at": expected["repair_reviewed_at"],
        "recheck_after": "",
        "successor_security_id": expected["successor_security_id"],
        "successor_symbol": expected["successor_symbol"],
        "source_url": expected["official_source_url"],
        "source": expected["resolution_source"],
        "retrieved_at": expected["repair_reviewed_at"],
        "source_hash": expected["official_source_hash"],
    }
    return bool(
        all(_text(row.get(field)) == value for field, value in exact.items())
        and _date(row.get("last_price_date")) == expected["last_real_session"]
    )


def _identity_is_exact(
    frame: pd.DataFrame,
    expected: Mapping[str, Any],
    *,
    history: bool,
) -> bool:
    symbol_field = "symbol" if history else "primary_symbol"
    start_field = "effective_from" if history else "active_from"
    end_field = "effective_to" if history else "active_to"
    expected_start = (
        expected["history_effective_from"]
        if history
        else expected["master_active_from"]
    )
    required = (
        "security_id",
        symbol_field,
        "exchange",
        start_field,
        end_field,
        "source",
        "source_url",
        "retrieved_at",
        "source_hash",
    )
    if "security_id" not in frame.columns:
        return False
    row = _one_row(
        frame,
        required_columns=required,
        mask=frame["security_id"].astype(str).eq(expected["security_id"]),
    )
    if row is None:
        return False
    return bool(
        _text(row.get(symbol_field)).upper() == expected["symbol"]
        and _text(row.get("exchange")).upper() == expected["identity_exchange"]
        and _date(row.get(start_field)) == expected_start
        and _date(row.get(end_field)) == expected["last_real_session"]
        and _text(row.get("source")) == expected["identity_source"]
        and _text(row.get("source_url")) == expected["official_source_url"]
        and _text(row.get("retrieved_at")) == expected["repair_reviewed_at"]
        and _text(row.get("source_hash")) == expected["official_source_hash"]
    )


def _prices_are_exact(frame: pd.DataFrame, expected: Mapping[str, Any]) -> bool:
    required = {
        "security_id",
        "session",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "currency",
        "source",
        "retrieved_at",
        "source_hash",
    }
    if not required.issubset(frame.columns):
        return False
    rows = frame.loc[
        frame["security_id"].astype(str).eq(expected["security_id"])
    ].copy()
    if rows.empty:
        return False
    sessions = pd.to_datetime(rows["session"], errors="coerce").dt.normalize()
    if bool(sessions.isna().any()) or bool(sessions.duplicated().any()):
        return False
    session_inventory = sorted(
        pd.Timestamp(value).date().isoformat() for value in sessions
    )
    inventory_sha256 = hashlib.sha256(
        ("\n".join(session_inventory) + "\n").encode()
    ).hexdigest()
    if (
        len(rows) != expected["retained_price_rows"]
        or inventory_sha256
        != expected["retained_price_session_inventory_sha256"]
        or sessions.min().date().isoformat() != expected["first_price_session"]
        or sessions.max().date().isoformat() != expected["last_real_session"]
    ):
        return False
    if not (
        rows["currency"].map(_text).eq("USD").all()
        and rows["source"].map(_text).eq("eodhd_eod").all()
        and rows["retrieved_at"].map(_text).eq(expected["raw_retrieved_at"]).all()
        and rows["source_hash"].map(_text).eq(expected["raw_source_hash"]).all()
    ):
        return False
    terminal = rows.loc[
        sessions.eq(pd.Timestamp(expected["last_real_session"]))
    ]
    if len(terminal) != 1:
        return False
    actual = tuple(_decimal(terminal.iloc[0].get(field)) for field in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ))
    wanted = tuple(Decimal(str(value)) for value in expected["terminal_ohlcv"])
    return actual == wanted


def _remove_lineage_is_exact(
    frame: pd.DataFrame,
    expected: Mapping[str, Any],
) -> bool:
    required = (
        "event_id",
        "index_id",
        "effective_date",
        "operation",
        "security_id",
        "official",
        "source",
        "source_url",
        "source_kind",
        "source_hash",
    )
    if "event_id" not in frame.columns:
        return False
    row = _one_row(
        frame,
        required_columns=required,
        mask=frame["event_id"].astype(str).eq(expected["next_remove_event_id"]),
    )
    if row is None:
        return False
    return bool(
        _text(row.get("event_id")) == expected["next_remove_event_id"]
        and _text(row.get("index_id")) == expected["index_id"]
        and _date(row.get("effective_date"))
        == expected["next_remove_effective_date"]
        and _text(row.get("operation")).upper() == "REMOVE"
        and _text(row.get("security_id")) == expected["security_id"]
        and _text(row.get("official")).lower() == "false"
        and _text(row.get("source")) == expected["next_remove_source"]
        and _text(row.get("source_url")) == expected["next_remove_source_url"]
        and _text(row.get("source_kind")) == "community"
        and _text(row.get("source_hash"))
        == expected["next_remove_source_hash"]
    )


def reviewed_operational_index_identity_gap_fingerprints(
    repository,
) -> tuple[str, ...]:
    """Return the NBL fingerprint only for the complete exact repaired state."""

    frames = _required_frames(repository)
    if frames is None:
        return ()
    expected = TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE
    exact = (
        _action_is_exact(frames["corporate_actions"], expected)
        and _resolution_is_exact(frames["lifecycle_resolutions"], expected)
        and _prices_are_exact(frames["daily_price_raw"], expected)
        and _identity_is_exact(frames["security_master"], expected, history=False)
        and _identity_is_exact(frames["symbol_history"], expected, history=True)
        and _remove_lineage_is_exact(
            frames["index_membership_events"], expected
        )
    )
    if not exact:
        return ()
    fingerprint = index_member_identity_gap_fingerprint(
        index_id=expected["index_id"],
        replay_date=expected["replay_date"],
        security_id=expected["security_id"],
        next_remove_event_id=expected["next_remove_event_id"],
        next_remove_effective_date=expected["next_remove_effective_date"],
        next_remove_source=expected["next_remove_source"],
        next_remove_source_hash=expected["next_remove_source_hash"],
    )
    return (fingerprint,) if fingerprint == expected["fingerprint"] else ()


def validate_operational_repository_snapshot(repository) -> ValidationReport:
    """Run strict cross-dataset checks with only exact reviewed local policy."""

    reviewed = reviewed_operational_index_identity_gap_fingerprints(repository)
    return validate_repository_snapshot(
        repository,
        allowed_index_identity_gap_fingerprints=reviewed,
    )
