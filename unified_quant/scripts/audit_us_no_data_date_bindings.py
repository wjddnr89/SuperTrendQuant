#!/usr/bin/env python3
"""Offline, plan-only audit of the 18 reviewed Yahoo no-data date bindings.

The inventory is deliberately finite and release pinned.  This command reads
the current local Parquet release, one already-generated cross-validation
report, and immutable local source-archive objects.  It never performs HTTP,
EODHD, R2, dataset writes, or lifecycle-action application.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import exchange_calendars as xcals
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supertrend_quant.market_store.cross_validation import (  # noqa: E402
    REVIEWED_NO_DATA_UNSUPPORTED_PATH_BASIS,
    REVIEWED_PERMANENT_EXCEPTION_NO_DATA_BASIS,
    canonical_json_bytes,
)
from supertrend_quant.market_store.manifest import sha256_bytes, write_atomic  # noqa: E402
from supertrend_quant.market_store.repository import LocalDatasetRepository  # noqa: E402


PINNED_RELEASE_VERSION = "20260715-20260718T230255094849Z"
AUDIT_SCHEMA = "us_no_data_date_binding_audit/v1"


@dataclass(frozen=True)
class CaseSpec:
    symbol: str
    target_id: str
    disposition: str
    priority: str
    repair_scope: str


CASES = (
    CaseSpec("SIVB", "0f3aa404673cefefa3bebf423ad557fa8892a5a30cbf174ab76cb92c9f3903e2", "blocked_independent_successor_price", "P0", "policy_only_blocked"),
    CaseSpec("BMYRT", "abfebadbc76b0c17e2e76e4190c3a45a75b71a02dcf47d7c8a39d42d5f0f465d", "accepted_exact_reviewed_exception", "P3", "none"),
    CaseSpec("WIN", "e9b8a0a88e0797ed15f72220da20a2f6b98c787cab0e1ac9347668f5ddeaacdc", "dataset_repair_required", "P2", "bankruptcy_otc_or_market_exit_gap"),
    CaseSpec("FLT", "f1d33b3660eab9c87a6d293a44cd07f559d36fbb7f5250adde9590a94e70e0ee", "dataset_repair_required", "P1", "identity_interval_and_old_sid_tail"),
    CaseSpec("CDAY", "0e8ae718d558c843f3add73eccc7549680728dde024cb664c157d0d3f112c6c0", "dataset_repair_required", "P1", "identity_interval_and_old_sid_tail"),
    CaseSpec("CHK", "2cb7ba1eb9074566f2e39214a50bbeca6df9aa08e3ee991d38d8d3ce468b907a", "dataset_repair_required", "P2", "bankruptcy_otc_or_market_exit_gap"),
    CaseSpec("FTR", "22ba248880440714112a86d9709aad5cdeb437ae0cf48f9a390fc189513eb364", "dataset_repair_required", "P2", "bankruptcy_otc_or_market_exit_gap"),
    CaseSpec("XEC", "5a7d992833e6db3ede1850047f5dedf2a806e2055304ff54490e177a60a07bd8", "dataset_repair_required", "P2", "identity_interval_and_old_sid_tail"),
    CaseSpec("TFCFA", "a1a391246a313a058341bfae9a17b5cabb6c94d2d930672e8a06414c869f3ec5", "accepted_exact_reviewed_exception", "P0", "none"),
    CaseSpec("TFCF", "28ecd1a7f4224e6d0bbe1db6b77030131d16edd952c182f2afd36cf32ddcb2af", "accepted_exact_reviewed_exception", "P0", "none"),
    CaseSpec("HCP", "db3ded074c2702bbea5ba8cd96b43422af814773ede30fa2516bc6f1e38f3d25", "dataset_repair_required", "P1", "identity_interval_and_old_sid_tail"),
    CaseSpec("UTX", "b35f720518c5b28fc0aff4b1bdd9a13231da9d9ed57ca4444371dec3aa56dbed", "dataset_repair_required", "P1", "identity_interval_and_old_sid_tail"),
    CaseSpec("VAL", "932963b415c64528e31076c5b39807c1799c3835cc9b8d000999fce4df7b8f67", "accepted_exact_reviewed_exception", "P3", "none"),
    CaseSpec("COG", "aed2031265aeba347424145a0518f26e021ec0608fb262842f5be5661ca1add6", "dataset_repair_required", "P1", "identity_interval_and_old_sid_tail"),
    CaseSpec("CTRP", "0043cc09f8d447c3b32e26cebb651aba0118eeca87e1cea5ec0535a8f42cfee9", "dataset_repair_required", "P1", "identity_interval_and_old_sid_tail"),
    CaseSpec("SYMC", "f9545eaea4f28e17630c659c0a5e0237065817901aea79cb701b362f46d77b59", "dataset_repair_required", "P1", "identity_interval_and_old_sid_tail"),
    CaseSpec("ENDP", "7b535b9546abd42e77bc1c094b69f0532405bd8e2c62fdc0a45084032bbe6eb6", "dataset_repair_required", "P2", "bankruptcy_otc_or_market_exit_gap"),
    CaseSpec("ARNC", "ef89f10e83177128247e7b62c97338dba1c62fdce831e5840631075781afa79d", "dataset_repair_required", "P0", "missing_terminal_action_only"),
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
    text = _text(value)
    if not text:
        return ""
    parsed = pd.Timestamp(text)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.date().isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_metadata(value: Any) -> dict[str, Any]:
    text = _text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _xnys_relation(last_session: str, event_date: str) -> tuple[str, int]:
    if not event_date:
        return "no_terminal_event", 0
    if event_date == last_session:
        return "event_on_terminal_session", 0
    calendar = xcals.get_calendar("XNYS")
    terminal = pd.Timestamp(last_session)
    event = pd.Timestamp(event_date)
    if event > terminal:
        sessions = calendar.sessions_in_range(terminal + pd.Timedelta(days=1), event)
        count = len(sessions)
        return (
            "event_on_next_xnys_session"
            if count == 1 and _date(sessions[0]) == event_date
            else "event_after_multi_session_gap",
            count,
        )
    sessions = calendar.sessions_in_range(event, terminal)
    return "event_precedes_terminal_price", -len(sessions)


def _index_scope(
    security_id: str,
    terminal_session: str,
    anchors: pd.DataFrame,
    membership: pd.DataFrame,
) -> list[dict[str, Any]]:
    index_ids = sorted(
        set(
            anchors.loc[
                anchors["security_id"].map(_text).eq(security_id), "index_id"
            ].map(_text)
        )
        | set(
            membership.loc[
                membership["security_id"].map(_text).eq(security_id), "index_id"
            ].map(_text)
        )
    )
    output: list[dict[str, Any]] = []
    for index_id in index_ids:
        anchor_rows = anchors.loc[
            anchors["security_id"].map(_text).eq(security_id)
            & anchors["index_id"].map(_text).eq(index_id)
            & anchors["anchor_date"].map(_date).le(terminal_session)
        ]
        event_rows = membership.loc[
            membership["security_id"].map(_text).eq(security_id)
            & membership["index_id"].map(_text).eq(index_id)
        ].copy()
        event_rows["_date"] = event_rows["effective_date"].map(_date)
        before = event_rows.loc[event_rows["_date"].le(terminal_session)].sort_values(
            "_date"
        )
        member = not anchor_rows.empty
        if not before.empty:
            member = _text(before.iloc[-1].get("operation")).upper() == "ADD"
        future_remove = event_rows.loc[
            event_rows["_date"].gt(terminal_session)
            & event_rows["operation"].map(_text).str.upper().eq("REMOVE")
        ].sort_values("_date")
        output.append(
            {
                "index_id": index_id,
                "member_on_terminal_session": member,
                "next_remove_date": (
                    _text(future_remove.iloc[0].get("_date"))
                    if not future_remove.empty
                    else ""
                ),
            }
        )
    return output


def _archive_binding(
    root: Path,
    archive: pd.DataFrame,
    source_hash: str,
    source_url: str,
) -> dict[str, Any]:
    if not source_hash:
        return {
            "source_hash": "",
            "source_url": source_url,
            "archive_row_count": 0,
            "payload_sha256_verified": False,
        }
    rows = archive.loc[
        archive["source_hash"].map(_text).str.lower().eq(source_hash.lower())
        & archive["source_url"].map(_text).eq(source_url)
    ]
    _require(len(rows) == 1, "Official source archive pair is not unique: " + source_hash)
    row = rows.iloc[0]
    object_path = _text(row.get("object_path"))
    payload = gzip.decompress((root / object_path).read_bytes())
    _require(sha256_bytes(payload) == source_hash.lower(), "Archive bytes changed: " + source_hash)
    return {
        "source_hash": source_hash.lower(),
        "source_url": source_url,
        "archive_row_count": 1,
        "archive_dataset": _text(row.get("dataset")),
        "object_path": object_path,
        "content_type": _text(row.get("content_type")),
        "payload_bytes": len(payload),
        "payload_sha256_verified": True,
    }


def _successor_check(
    event: Mapping[str, Any] | None,
    report_prices: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if event is None:
        return {
            "security_id": "",
            "symbol": "",
            "target_id": "",
            "status": "not_modeled",
        }
    security_id = _text(event.get("new_security_id"))
    symbol = _text(event.get("new_symbol")).upper()
    if not security_id:
        return {
            "security_id": "",
            "symbol": "",
            "target_id": "",
            "status": "not_required",
        }
    candidates = [
        item
        for item in report_prices
        if _text(item.get("security_id")) == security_id
        and _text(item.get("provider_symbol") or item.get("symbol")).upper() == symbol
    ]
    _require(len(candidates) == 1, "Successor target is not unique: " + symbol)
    item = candidates[0]
    return {
        "security_id": security_id,
        "symbol": symbol,
        "target_id": _text(item.get("target_id")),
        "status": _text(item.get("status")),
    }


def build_audit(
    repository: LocalDatasetRepository,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    release, _ = repository.current_release()
    _require(release is not None, "Current local release is missing.")
    _require(
        release.version == PINNED_RELEASE_VERSION
        and _text(report.get("base_release_version")) == PINNED_RELEASE_VERSION,
        "Audit release changed; review the finite inventory again.",
    )
    report_prices = [item for item in report.get("prices", ()) if isinstance(item, Mapping)]
    by_target = {_text(item.get("target_id")): item for item in report_prices}
    _require(len(by_target) == len(report_prices), "Cross-validation target IDs are not unique.")

    frames = {
        name: repository.read_frame(name, release.dataset_versions[name])
        for name in (
            "security_master",
            "symbol_history",
            "corporate_actions",
            "daily_price_raw",
            "lifecycle_resolutions",
            "source_archive",
            "index_constituent_anchors",
            "index_membership_events",
        )
    }
    actions = frames["corporate_actions"]
    resolutions = frames["lifecycle_resolutions"]
    history = frames["symbol_history"]
    master = frames["security_master"]
    prices = frames["daily_price_raw"]
    archive = frames["source_archive"]
    calendar = xcals.get_calendar("XNYS")

    rows: list[dict[str, Any]] = []
    for spec in CASES:
        item = by_target.get(spec.target_id)
        _require(item is not None and _text(item.get("symbol")).upper() == spec.symbol, "Finite date-binding target changed: " + spec.symbol)
        exception = item.get("exception")
        _require(isinstance(exception, Mapping), "No-data exception diagnostic is missing: " + spec.symbol)
        terminal_session = _date(exception.get("terminal_calendar", {}).get("terminal_session"))
        _require(bool(terminal_session), "Terminal session is missing: " + spec.symbol)
        security_id = _text(item.get("security_id"))

        master_rows = master.loc[master["security_id"].map(_text).eq(security_id)]
        _require(len(master_rows) == 1, "security_master row is not unique: " + spec.symbol)
        master_row = master_rows.iloc[0]
        history_rows = history.loc[
            history["security_id"].map(_text).eq(security_id)
            & history["symbol"].map(_text).str.upper().eq(spec.symbol)
            & history["effective_from"].map(_date).eq(_date(item.get("identity_active_from")))
            & history["effective_to"].map(_date).eq(_date(item.get("identity_active_to")))
        ]
        _require(len(history_rows) == 1, "symbol_history interval is not unique: " + spec.symbol)
        history_row = history_rows.iloc[0]

        price_rows = prices.loc[prices["security_id"].map(_text).eq(security_id)].copy()
        price_rows["_session"] = price_rows["session"].map(_date)
        active_from = _date(item.get("identity_active_from"))
        active_to = _date(item.get("identity_active_to"))
        price_rows = price_rows.loc[price_rows["_session"].ge(active_from)]
        if active_to:
            price_rows = price_rows.loc[price_rows["_session"].le(active_to)]
        _require(not price_rows.empty and price_rows["_session"].max() == terminal_session, "Stored terminal price changed: " + spec.symbol)
        next_session = _date(calendar.next_session(pd.Timestamp(terminal_session)))

        event_id = _text(item.get("terminal_event_id"))
        event_rows = actions.loc[actions["event_id"].map(_text).eq(event_id)] if event_id else actions.iloc[0:0]
        _require(len(event_rows) <= 1, "Terminal event is not unique: " + spec.symbol)
        event = event_rows.iloc[0].to_dict() if len(event_rows) else None
        metadata = _json_metadata(event.get("metadata")) if event else {}
        event_date = _date(event.get("effective_date")) if event else ""
        legal_date = ""
        if event:
            legal_date = next(
                (
                    _date(metadata.get(key))
                    for key in (
                        "legal_completion_date",
                        "legal_cancellation_date",
                        "official_completion_date",
                        "nasdaq_suspension_date",
                        "last_trading_session",
                    )
                    if _date(metadata.get(key))
                ),
                event_date,
            )
        relation, session_delta = _xnys_relation(terminal_session, event_date)

        resolution_rows = resolutions.loc[
            resolutions["security_id"].map(_text).eq(security_id)
            & resolutions["symbol"].map(_text).str.upper().eq(spec.symbol)
            & resolutions["last_price_date"].map(_date).eq(terminal_session)
        ]
        _require(len(resolution_rows) <= 1, "Lifecycle resolution is not unique: " + spec.symbol)
        resolution = resolution_rows.iloc[0].to_dict() if len(resolution_rows) else None

        if event:
            evidence_kind = "official_action_event"
            source_hash = _text(event.get("source_hash")).lower()
            source_url = _text(event.get("source_url"))
        elif resolution:
            evidence_kind = "official_lifecycle_exception"
            source_hash = _text(resolution.get("source_hash")).lower()
            source_url = _text(resolution.get("source_url"))
        else:
            evidence_kind = "identity_boundary_only_not_terminal_event"
            source_hash = _text(history_row.get("source_hash")).lower()
            source_url = _text(history_row.get("source_url"))
        archive_binding = _archive_binding(repository.root, archive, source_hash, source_url)

        tail_rows = pd.DataFrame()
        if event_date and event_date <= terminal_session:
            tail_rows = price_rows.loc[price_rows["_session"].ge(event_date)]
        successor = _successor_check(event, report_prices)
        expected_status = "explicit_exception" if spec.disposition == "accepted_exact_reviewed_exception" else "mismatch"
        _require(_text(item.get("status")) == expected_status, "Reviewed disposition did not reach its exact expected status: " + spec.symbol)
        if spec.symbol == "BMYRT":
            _require(_text(item.get("validation_basis")) == REVIEWED_NO_DATA_UNSUPPORTED_PATH_BASIS, "BMYRT exact unsupported-path binding is missing.")
        if spec.symbol in {"TFCFA", "TFCF", "VAL"}:
            _require(_text(item.get("validation_basis")) == REVIEWED_PERMANENT_EXCEPTION_NO_DATA_BASIS, "Permanent exception no-data binding is missing: " + spec.symbol)

        rows.append(
            {
                "symbol": spec.symbol,
                "security_id": security_id,
                "target_id": spec.target_id,
                "cross_validation_status": _text(item.get("status")),
                "validation_basis": _text(item.get("validation_basis")),
                "disposition": spec.disposition,
                "repair_scope": spec.repair_scope,
                "priority": spec.priority,
                "stored_security_master_active_to": _date(master_row.get("active_to")),
                "stored_symbol_effective_from": _date(history_row.get("effective_from")),
                "stored_symbol_effective_to": _date(history_row.get("effective_to")),
                "last_price_session": terminal_session,
                "next_xnys_session": next_session,
                "terminal_price_rows": len(price_rows),
                "official_event_id": event_id,
                "official_action_type": _text(event.get("action_type")) if event else "",
                "official_event_effective_date": event_date,
                "official_legal_date": legal_date,
                "market_halt_date": _date(metadata.get("nasdaq_halt_date")),
                "date_relation": relation,
                "xnys_session_delta": session_delta,
                "official_evidence_kind": evidence_kind,
                "official_source": archive_binding,
                "lifecycle_resolution": (
                    {
                        "candidate_id": _text(resolution.get("candidate_id")),
                        "resolution": _text(resolution.get("resolution")),
                        "event_id": _text(resolution.get("event_id")),
                        "exception_code": _text(resolution.get("exception_code")),
                    }
                    if resolution
                    else None
                ),
                "old_sid_rows_on_or_after_event": len(tail_rows),
                "old_sid_tail_start": _text(tail_rows["_session"].min()) if len(tail_rows) else "",
                "old_sid_tail_end": _text(tail_rows["_session"].max()) if len(tail_rows) else "",
                "successor": successor,
                "backtest_index_scope": _index_scope(
                    security_id,
                    terminal_session,
                    frames["index_constituent_anchors"],
                    frames["index_membership_events"],
                ),
            }
        )

    disposition_counts = pd.Series([row["disposition"] for row in rows]).value_counts()
    repair_scope_counts = pd.Series([row["repair_scope"] for row in rows]).value_counts()
    direct_terminal_membership = [
        row
        for row in rows
        if any(
            scope["member_on_terminal_session"]
            for scope in row["backtest_index_scope"]
        )
    ]
    direct_terminal_members_by_index: dict[str, list[str]] = {}
    for row in direct_terminal_membership:
        for scope in row["backtest_index_scope"]:
            if scope["member_on_terminal_session"]:
                direct_terminal_members_by_index.setdefault(
                    scope["index_id"], []
                ).append(row["symbol"])
    direct_terminal_members_by_index = {
        index_id: sorted(symbols)
        for index_id, symbols in sorted(direct_terminal_members_by_index.items())
    }
    direct_terminal_symbols = {
        row["symbol"] for row in direct_terminal_membership
    }
    repair_required_direct = sorted(
        row["symbol"]
        for row in direct_terminal_membership
        if row["disposition"] == "dataset_repair_required"
    )
    blocked_direct = sorted(
        row["symbol"]
        for row in direct_terminal_membership
        if row["disposition"] == "blocked_independent_successor_price"
    )
    accepted_limitation_direct = sorted(
        row["symbol"]
        for row in direct_terminal_membership
        if row["disposition"] == "accepted_exact_reviewed_exception"
    )
    return {
        "schema": AUDIT_SCHEMA,
        "release_version": release.version,
        "cross_validation_report_sha256": sha256_bytes(canonical_json_bytes(report)),
        "network_accessed": False,
        "http_attempts": 0,
        "eodhd_calls": 0,
        "r2_accessed": False,
        "dataset_writes_performed": False,
        "raw_action_identity_apply_performed": False,
        "generic_date_tolerance_added": False,
        "summary": {
            "audited_case_count": len(rows),
            "accepted_exact_reviewed_exception_count": int(
                disposition_counts.get("accepted_exact_reviewed_exception", 0)
            ),
            "dataset_repair_required_count": int(
                disposition_counts.get("dataset_repair_required", 0)
            ),
            "blocked_independent_successor_price_count": int(
                disposition_counts.get("blocked_independent_successor_price", 0)
            ),
            "identity_interval_and_old_sid_tail_count": int(
                repair_scope_counts.get("identity_interval_and_old_sid_tail", 0)
            ),
            "bankruptcy_otc_or_market_exit_gap_count": int(
                repair_scope_counts.get("bankruptcy_otc_or_market_exit_gap", 0)
            ),
            "missing_terminal_action_only_count": int(
                repair_scope_counts.get("missing_terminal_action_only", 0)
            ),
            "old_sid_rows_on_or_after_event_total": sum(
                int(row["old_sid_rows_on_or_after_event"])
                for row in rows
                if row["repair_scope"] == "identity_interval_and_old_sid_tail"
            ),
        },
        "backtest_impact": {
            "direct_terminal_membership_case_count": len(
                direct_terminal_membership
            ),
            "direct_terminal_membership_symbols": sorted(
                row["symbol"] for row in direct_terminal_membership
            ),
            "direct_terminal_members_by_index": direct_terminal_members_by_index,
            "repair_required_direct_symbols": repair_required_direct,
            "blocked_direct_symbols": blocked_direct,
            "accepted_limitation_direct_symbols": accepted_limitation_direct,
            "repair_required_outside_terminal_membership_count": sum(
                row["disposition"] == "dataset_repair_required"
                and row["symbol"] not in direct_terminal_symbols
                for row in rows
            ),
            "interpretation": (
                "Direct means the security is a modeled index member on its "
                "stored terminal price session. It does not claim that cases "
                "outside terminal membership have zero downstream identity or "
                "signal risk."
            ),
        },
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/cache")
    parser.add_argument(
        "--cross-validation-report",
        default="/tmp/us-no-data-date-binding-current-report-v2.json",
    )
    parser.add_argument(
        "--output", default="/tmp/us-no-data-date-binding-audit.json"
    )
    args = parser.parse_args()
    repository = LocalDatasetRepository(Path(args.data_root))
    report = json.loads(Path(args.cross_validation_report).read_text())
    audit = build_audit(repository, report)
    payload = canonical_json_bytes(audit)
    write_atomic(Path(args.output), payload)
    print(
        json.dumps(
            {
                "status": "plan_only_complete",
                "output": str(Path(args.output)),
                "audit_sha256": sha256_bytes(payload),
                "summary": audit["summary"],
                "network_accessed": False,
                "dataset_writes_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
