"""Exact reviewed policies for terminal events the generic parser cannot model.

These exceptions do not turn an estimate into an official economic value.
They bind one action, one resolution, one archived lifecycle report and one
explicit backtest policy.  Any action, report, metadata, warning or policy
drift fails closed.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

import exchange_calendars as xcals
import pandas as pd

from .lifecycle import canonical_lifecycle_event_id
from .lifecycle_coverage import lifecycle_candidate_id
from .manifest import sha256_bytes


ACTION_FIELDS = (
    "event_id",
    "security_id",
    "action_type",
    "effective_date",
    "new_security_id",
    "new_symbol",
    "ratio",
    "cash_amount",
    "currency",
    "source_kind",
    "source_url",
    "source_hash",
)

REVIEWED_TERMINAL_POLICY_EXCEPTION_FIELDS = (
    *ACTION_FIELDS,
    "candidate_id",
    "symbol",
    "last_price_date",
    "ex_date",
    "announcement_date",
    "payment_date",
    "action_metadata_sha256",
    "report_effective_date",
    "report_action_type",
    "report_new_symbol",
    "report_ratio",
    "report_cash_amount",
    "report_source_url",
    "report_source_hash",
    "report_candidate_active_to",
    "report_manual_review_reason",
    "report_crosscheck_passed",
    "report_crosscheck_date_passed",
    "report_crosscheck_economic_terms_passed",
    "allowed_report_mismatches",
    "policy_code",
    "required_release_warning",
    "lifecycle_evidence_report_sha256",
    "filing_accession_number",
    "filing_date",
)

TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTION_EVENT_IDS = frozenset(
    {
        "350bb85a7395ef9272e5f2867afdd4e523c99c258752120977ec1f35e36a2c8a",
        "cb355a88e767bd5f557350ddf9c13f1b324da6e8f96c622a2e1f8eeea01fa36a",
        "f553d393e8bda37561276fec20d5b9bce5f722609e466e96bc9e199c624891c1",
    }
)

# Updated only after the complete normalized YAML registry is reviewed.
TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTIONS_SHA256 = (
    "5e461b25741175c1930ff3e937fa238dd21f0899fb6ae51ca3eb4db9bf7dbe6c"
)

_POLICY_RULES = {
    "abmd_nontradeable_cvr_lower_bound/v1": {
        "action_type": "cash_merger",
        "source_kind": "official_lower_bound_policy",
        "allowed_report_mismatches": (
            "cash_amount",
            "source_hash",
            "source_url",
        ),
        "requires_warning": True,
    },
    "celg_next_session_cvr_delivery/v1": {
        "action_type": "stock_merger",
        "source_kind": "official_crosscheck",
        "allowed_report_mismatches": ("effective_date",),
        "requires_warning": True,
    },
    "para_no_election_default_stock/v1": {
        "action_type": "stock_merger",
        "source_kind": "sec_filing_default_stock_policy",
        "allowed_report_mismatches": ("cash_amount", "effective_date"),
        "requires_warning": False,
    },
}

_REPORT_COMPARISON_FIELDS = (
    "action_type",
    "effective_date",
    "new_symbol",
    "ratio",
    "cash_amount",
    "source_url",
    "source_hash",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def _exact_number(value: Any, field: str) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeError(f"Reviewed terminal policy {field} is not numeric.") from exc
    _require(parsed.is_finite(), f"Reviewed terminal policy {field} is not finite.")
    if parsed == 0:
        return "0"
    return format(parsed.normalize(), "f")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(value))


def action_metadata_sha256(value: Any) -> str:
    """Hash one action metadata object after canonical JSON normalization."""

    if isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raw = _text(value)
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("Terminal policy action metadata is invalid JSON.") from exc
    _require(isinstance(parsed, dict), "Terminal policy metadata must be an object.")
    return _canonical_json_sha256(parsed)


def _digest(value: Any, field: str) -> str:
    output = _text(value).lower()
    _require(
        len(output) == 64
        and all(character in "0123456789abcdef" for character in output),
        f"Reviewed terminal policy {field} must be lowercase SHA-256.",
    )
    return output


def _exact_iso_date(value: Any, field: str, *, empty_allowed: bool = False) -> str:
    output = _date(value)
    raw = _text(value)
    _require(
        (empty_allowed and not raw) or (bool(output) and raw == output),
        f"Reviewed terminal policy {field} must be an exact ISO date.",
    )
    return output


def canonical_reviewed_terminal_policy_exception(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one complete reviewed terminal policy."""

    _require(isinstance(value, Mapping), "Reviewed terminal policy must be an object.")
    _require(
        set(value) == set(REVIEWED_TERMINAL_POLICY_EXCEPTION_FIELDS),
        "Reviewed terminal policy fields are not exact.",
    )
    output = {
        "event_id": _digest(value.get("event_id"), "event_id"),
        "security_id": _text(value.get("security_id")),
        "action_type": _text(value.get("action_type")).lower(),
        "effective_date": _exact_iso_date(value.get("effective_date"), "effective_date"),
        "new_security_id": _text(value.get("new_security_id")),
        "new_symbol": _text(value.get("new_symbol")).upper(),
        "ratio": _exact_number(value.get("ratio"), "ratio"),
        "cash_amount": _exact_number(value.get("cash_amount"), "cash_amount"),
        "currency": _text(value.get("currency")).upper(),
        "source_kind": _text(value.get("source_kind")),
        "source_url": _text(value.get("source_url")),
        "source_hash": _digest(value.get("source_hash"), "source_hash"),
        "candidate_id": _digest(value.get("candidate_id"), "candidate_id"),
        "symbol": _text(value.get("symbol")).upper(),
        "last_price_date": _exact_iso_date(
            value.get("last_price_date"), "last_price_date"
        ),
        "ex_date": _exact_iso_date(value.get("ex_date"), "ex_date"),
        "announcement_date": _exact_iso_date(
            value.get("announcement_date"), "announcement_date"
        ),
        "payment_date": _exact_iso_date(
            value.get("payment_date"), "payment_date", empty_allowed=True
        ),
        "action_metadata_sha256": _digest(
            value.get("action_metadata_sha256"), "action_metadata_sha256"
        ),
        "report_effective_date": _exact_iso_date(
            value.get("report_effective_date"), "report_effective_date"
        ),
        "report_action_type": _text(value.get("report_action_type")).lower(),
        "report_new_symbol": _text(value.get("report_new_symbol")).upper(),
        "report_ratio": _exact_number(value.get("report_ratio"), "report_ratio"),
        "report_cash_amount": _exact_number(
            value.get("report_cash_amount"), "report_cash_amount"
        ),
        "report_source_url": _text(value.get("report_source_url")),
        "report_source_hash": _digest(
            value.get("report_source_hash"), "report_source_hash"
        ),
        "report_candidate_active_to": _exact_iso_date(
            value.get("report_candidate_active_to"), "report_candidate_active_to"
        ),
        "report_manual_review_reason": _text(
            value.get("report_manual_review_reason")
        ),
        "report_crosscheck_passed": value.get("report_crosscheck_passed"),
        "report_crosscheck_date_passed": value.get(
            "report_crosscheck_date_passed"
        ),
        "report_crosscheck_economic_terms_passed": value.get(
            "report_crosscheck_economic_terms_passed"
        ),
        "allowed_report_mismatches": list(
            value.get("allowed_report_mismatches") or ()
        ),
        "policy_code": _text(value.get("policy_code")),
        "required_release_warning": _text(value.get("required_release_warning")),
        "lifecycle_evidence_report_sha256": _digest(
            value.get("lifecycle_evidence_report_sha256"),
            "lifecycle_evidence_report_sha256",
        ),
        "filing_accession_number": _text(value.get("filing_accession_number")),
        "filing_date": _exact_iso_date(value.get("filing_date"), "filing_date"),
    }
    _require(
        bool(output["security_id"] and output["symbol"]),
        "Reviewed terminal policy identity is incomplete.",
    )
    _require(output["currency"] == "USD", "Reviewed terminal policy must use USD.")
    for field in (
        "report_crosscheck_passed",
        "report_crosscheck_date_passed",
        "report_crosscheck_economic_terms_passed",
    ):
        _require(type(output[field]) is bool, f"Reviewed terminal policy {field} is invalid.")
    _require(
        bool(output["source_url"] and output["report_source_url"]),
        "Reviewed terminal policy source URLs are incomplete.",
    )
    _require(
        bool(output["report_manual_review_reason"] and output["filing_accession_number"]),
        "Reviewed terminal policy report review provenance is incomplete.",
    )

    policy_code = str(output["policy_code"])
    rule = _POLICY_RULES.get(policy_code)
    _require(rule is not None, "Reviewed terminal policy_code is not approved.")
    _require(
        output["action_type"] == rule["action_type"]
        and output["report_action_type"] == rule["action_type"]
        and output["source_kind"] == rule["source_kind"],
        "Reviewed terminal policy action type/source kind is invalid.",
    )
    _require(
        tuple(output["allowed_report_mismatches"])
        == rule["allowed_report_mismatches"],
        "Reviewed terminal policy mismatch scope is not exact.",
    )
    _require(
        bool(output["required_release_warning"]) is bool(rule["requires_warning"]),
        "Reviewed terminal policy release-warning requirement is invalid.",
    )

    if output["action_type"] == "cash_merger":
        _require(
            not output["new_security_id"]
            and not output["new_symbol"]
            and output["ratio"] is None
            and output["cash_amount"] is not None
            and Decimal(str(output["cash_amount"])) > 0,
            "Reviewed terminal cash-merger policy terms are invalid.",
        )
    else:
        _require(
            bool(output["new_security_id"] and output["new_symbol"])
            and output["ratio"] is not None
            and Decimal(str(output["ratio"])) > 0
            and (
                output["cash_amount"] is None
                or Decimal(str(output["cash_amount"])) >= 0
            ),
            "Reviewed terminal stock-merger policy terms are invalid.",
        )

    _require(
        output["event_id"]
        == canonical_lifecycle_event_id(
            str(output["security_id"]),
            str(output["action_type"]),
            str(output["effective_date"]),
        )
        and output["candidate_id"]
        == lifecycle_candidate_id(
            str(output["security_id"]), str(output["last_price_date"])
        ),
        "Reviewed terminal policy canonical IDs are invalid.",
    )
    terminal = pd.Timestamp(str(output["last_price_date"]))
    effective = pd.Timestamp(str(output["effective_date"]))
    sessions = xcals.get_calendar("XNYS").sessions_in_range(
        terminal + pd.Timedelta(days=1), effective
    )
    normalized = [pd.Timestamp(item).tz_localize(None).normalize() for item in sessions]
    _require(
        normalized == [effective.normalize()],
        "Reviewed terminal policy effective date is not the next XNYS session.",
    )

    action_report = {
        "action_type": output["action_type"],
        "effective_date": output["effective_date"],
        "new_symbol": output["new_symbol"],
        "ratio": output["ratio"],
        "cash_amount": output["cash_amount"],
        "source_url": output["source_url"],
        "source_hash": output["source_hash"],
    }
    report = {
        "action_type": output["report_action_type"],
        "effective_date": output["report_effective_date"],
        "new_symbol": output["report_new_symbol"],
        "ratio": output["report_ratio"],
        "cash_amount": output["report_cash_amount"],
        "source_url": output["report_source_url"],
        "source_hash": output["report_source_hash"],
    }
    observed = {
        field for field in _REPORT_COMPARISON_FIELDS if action_report[field] != report[field]
    }
    _require(
        observed == set(output["allowed_report_mismatches"]),
        "Reviewed terminal policy does not exactly describe report drift.",
    )
    return output


def reviewed_terminal_policy_exceptions(
    events_policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = events_policy.get("reviewed_terminal_policy_exceptions")
    _require(isinstance(raw, list), "Reviewed terminal policy registry must be a list.")
    output: dict[str, dict[str, Any]] = {}
    for value in raw:
        normalized = canonical_reviewed_terminal_policy_exception(value)
        event_id = str(normalized["event_id"])
        _require(event_id not in output, f"Duplicate terminal policy event_id: {event_id}")
        output[event_id] = normalized
    return output


def reviewed_terminal_policy_exception_sha256(value: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(canonical_reviewed_terminal_policy_exception(value))


def reviewed_terminal_policy_exception_inventory_sha256(
    events_policy: Mapping[str, Any],
) -> str:
    return _canonical_json_sha256(reviewed_terminal_policy_exceptions(events_policy))


def reviewed_terminal_policy_action_mismatches(
    action: Mapping[str, Any], exception: Mapping[str, Any]
) -> tuple[str, ...]:
    expected = canonical_reviewed_terminal_policy_exception(exception)
    actual = {
        "event_id": _text(action.get("event_id")).lower(),
        "security_id": _text(action.get("security_id")),
        "action_type": _text(action.get("action_type")).lower(),
        "effective_date": _date(action.get("effective_date")),
        "new_security_id": _text(action.get("new_security_id")),
        "new_symbol": _text(action.get("new_symbol")).upper(),
        "ratio": _exact_number(action.get("ratio"), "ratio"),
        "cash_amount": _exact_number(action.get("cash_amount"), "cash_amount"),
        "currency": _text(action.get("currency")).upper(),
        "source_kind": _text(action.get("source_kind")),
        "source_url": _text(action.get("source_url")),
        "source_hash": _text(action.get("source_hash")).lower(),
        "ex_date": _date(action.get("ex_date")),
        "announcement_date": _date(action.get("announcement_date")),
        "payment_date": _date(action.get("payment_date")),
        "action_metadata_sha256": action_metadata_sha256(action.get("metadata")),
    }
    mismatches = [field for field, observed in actual.items() if observed != expected[field]]
    if _text(action.get("official")).lower() != "true":
        mismatches.append("official")
    return tuple(mismatches)


def reviewed_terminal_policy_report_mismatches(
    action: Mapping[str, Any],
    resolution: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    exception: Mapping[str, Any],
    lifecycle_report_sha256: str,
) -> tuple[str, ...]:
    expected = canonical_reviewed_terminal_policy_exception(exception)
    if not isinstance(record, Mapping):
        return ("lifecycle_report_record",)
    value = record.get("verified_event")
    event = value if isinstance(value, Mapping) else record.get("parsed")
    if not isinstance(event, Mapping):
        return ("lifecycle_report_event",)
    mismatches: list[str] = []

    resolution_pairs = {
        "candidate_id": (_text(resolution.get("candidate_id")).lower(), expected["candidate_id"]),
        "resolution_event_id": (_text(resolution.get("event_id")).lower(), expected["event_id"]),
        "resolution_security_id": (_text(resolution.get("security_id")), expected["security_id"]),
        "resolution_symbol": (_text(resolution.get("symbol")).upper(), expected["symbol"]),
        "last_price_date": (_date(resolution.get("last_price_date")), expected["last_price_date"]),
        "successor_security_id": (
            _text(resolution.get("successor_security_id")),
            expected["new_security_id"],
        ),
        "resolution_source_url": (_text(resolution.get("source_url")), expected["source_url"]),
        "resolution_source_hash": (
            _text(resolution.get("source_hash")).lower(),
            expected["source_hash"],
        ),
    }
    mismatches.extend(
        field for field, (observed, wanted) in resolution_pairs.items() if observed != wanted
    )

    candidate = record.get("candidate")
    if not isinstance(candidate, Mapping):
        mismatches.append("lifecycle_report_candidate")
    else:
        candidate_pairs = {
            "candidate_security_id": (_text(candidate.get("security_id")), expected["security_id"]),
            "candidate_symbol": (_text(candidate.get("symbol")).upper(), expected["symbol"]),
            "candidate_last_price_date": (
                _date(candidate.get("last_price_date")),
                expected["last_price_date"],
            ),
            "candidate_active_to": (
                _date(candidate.get("active_to")),
                expected["report_candidate_active_to"],
            ),
        }
        mismatches.extend(
            field for field, (observed, wanted) in candidate_pairs.items() if observed != wanted
        )

    report_source_url = _text(event.get("source_url") or record.get("source_url"))
    report_source_hash = _text(
        event.get("source_hash") or record.get("source_hash")
    ).lower()
    report_pairs = {
        "report_action_type": (_text(event.get("action_type")).lower(), expected["report_action_type"]),
        "report_effective_date": (_date(event.get("effective_date")), expected["report_effective_date"]),
        "report_new_symbol": (_text(event.get("new_symbol")).upper(), expected["report_new_symbol"]),
        "report_ratio": (_exact_number(event.get("ratio"), "report_ratio"), expected["report_ratio"]),
        "report_cash_amount": (
            _exact_number(event.get("cash_amount"), "report_cash_amount"),
            expected["report_cash_amount"],
        ),
        "report_source_url": (report_source_url, expected["report_source_url"]),
        "report_source_hash": (report_source_hash, expected["report_source_hash"]),
        "lifecycle_evidence_report_sha256": (
            _text(lifecycle_report_sha256).lower(),
            expected["lifecycle_evidence_report_sha256"],
        ),
        "manual_review_reason": (
            _text(record.get("manual_review_reason")),
            expected["report_manual_review_reason"],
        ),
        "report_successor_security_id": (
            _text(record.get("successor_security_id")),
            expected["new_security_id"],
        ),
    }
    mismatches.extend(
        field for field, (observed, wanted) in report_pairs.items() if observed != wanted
    )
    if record.get("eligible_for_apply") is not False:
        mismatches.append("eligible_for_apply")
    crosscheck = record.get("crosscheck")
    if not isinstance(crosscheck, Mapping):
        mismatches.append("crosscheck")
    else:
        crosscheck_pairs = {
            "crosscheck_passed": (
                crosscheck.get("passed"),
                expected["report_crosscheck_passed"],
            ),
            "crosscheck_date_passed": (
                crosscheck.get("date_passed"),
                expected["report_crosscheck_date_passed"],
            ),
            "crosscheck_economic_terms_passed": (
                crosscheck.get("economic_terms_passed"),
                expected["report_crosscheck_economic_terms_passed"],
            ),
        }
        mismatches.extend(
            field
            for field, (observed, wanted) in crosscheck_pairs.items()
            if observed is not wanted
        )
    filing = record.get("filing")
    if not isinstance(filing, Mapping):
        mismatches.append("filing")
    else:
        if _text(filing.get("accession_number")) != expected["filing_accession_number"]:
            mismatches.append("filing_accession_number")
        if _date(filing.get("filing_date")) != expected["filing_date"]:
            mismatches.append("filing_date")
    return tuple(dict.fromkeys(mismatches))


def reviewed_terminal_policy_release_warning_mismatches(
    warnings: tuple[str, ...] | list[str], exception: Mapping[str, Any]
) -> tuple[str, ...]:
    expected = canonical_reviewed_terminal_policy_exception(exception)
    warning = str(expected["required_release_warning"])
    if warning and tuple(warnings).count(warning) != 1:
        return ("required_release_warning",)
    return ()
