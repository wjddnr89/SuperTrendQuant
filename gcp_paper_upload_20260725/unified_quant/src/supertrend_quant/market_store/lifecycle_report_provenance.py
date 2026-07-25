from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from supertrend_quant.market_store.lifecycle_coverage import (
    DEFAULT_SELECTION_RULE,
    lifecycle_candidate_set_sha256,
)


LIFECYCLE_REPORT_SCHEMA = "us_lifecycle_sec_collection/v1"
LIFECYCLE_COLLECTOR_VERSION = "us_lifecycle_collector_v1"
LIFECYCLE_COLLECTOR_CONFIG_VERSION = "us_terminal_sec_evidence/v1"
SEC_FETCH_POLICY_CACHE_ONLY = "cache_only"
SEC_FETCH_POLICY_FETCH_MISSING = "fetch_missing_opt_in"
SEC_FETCH_POLICIES = frozenset(
    {SEC_FETCH_POLICY_CACHE_ONLY, SEC_FETCH_POLICY_FETCH_MISSING}
)
DEFAULT_SEC_MAX_HTTP_ATTEMPTS = 200
DEFAULT_SEC_MAX_HTTP_ATTEMPTS_PER_CANDIDATE = 24
SEC_MAX_HTTP_ATTEMPTS_PER_REQUEST = 4

REPORT_BINDING_FIELDS = (
    "report_schema",
    "collector_version",
    "collector_config_version",
    "sec_fetch_policy",
    "sec_max_http_attempts",
    "sec_max_http_attempts_per_candidate",
    "sec_max_http_attempts_per_request",
    "sec_http_attempts",
    "sec_http_attempts_by_candidate",
    "release_version",
    "completed_session",
    "candidate_selection_rule",
    "candidate_count",
    "candidate_set_sha256",
    "hints_sha256",
    "input_dataset_versions",
    "input_dataset_versions_sha256",
    "collector_config_sha256",
    "collection_context_sha256",
)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_value(candidate: Any, field: str) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(field)
    return getattr(candidate, field)


def build_lifecycle_report_binding(
    *,
    release_version: str,
    completed_session: str,
    dataset_versions: Mapping[str, Any],
    candidates: Iterable[Any],
    hints_path: Path,
    sec_fetch_policy: str = SEC_FETCH_POLICY_CACHE_ONLY,
    sec_max_http_attempts: int = DEFAULT_SEC_MAX_HTTP_ATTEMPTS,
    sec_max_http_attempts_per_candidate: int = (
        DEFAULT_SEC_MAX_HTTP_ATTEMPTS_PER_CANDIDATE
    ),
    sec_max_http_attempts_per_request: int = SEC_MAX_HTTP_ATTEMPTS_PER_REQUEST,
    sec_http_attempts: int = 0,
    sec_http_attempts_by_candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the audited context that makes a collector report reusable.

    The raw hints bytes are hashed deliberately: parsing and re-serializing YAML
    would make distinct reviewed inputs look identical.  The candidate hash uses
    the same canonical contract as the lifecycle coverage gate.
    """

    fetch_policy = str(sec_fetch_policy).strip()
    if fetch_policy not in SEC_FETCH_POLICIES:
        raise ValueError(f"Unsupported SEC fetch policy: {sec_fetch_policy!r}")
    global_cap = _positive_int(sec_max_http_attempts, "sec_max_http_attempts")
    candidate_cap = _positive_int(
        sec_max_http_attempts_per_candidate,
        "sec_max_http_attempts_per_candidate",
    )
    request_cap = _positive_int(
        sec_max_http_attempts_per_request,
        "sec_max_http_attempts_per_request",
    )
    actual_attempts = _nonnegative_int(sec_http_attempts, "sec_http_attempts")
    if actual_attempts > global_cap:
        raise ValueError(
            "sec_http_attempts cannot exceed sec_max_http_attempts: "
            f"{actual_attempts}>{global_cap}"
        )
    attempts_by_candidate = {
        str(key): _nonnegative_int(value, f"sec_http_attempts_by_candidate[{key!r}]")
        for key, value in sorted((sec_http_attempts_by_candidate or {}).items())
    }
    if any(value > candidate_cap for value in attempts_by_candidate.values()):
        raise ValueError(
            "A candidate SEC HTTP attempt count exceeds the per-candidate cap."
        )
    if sum(attempts_by_candidate.values()) > actual_attempts:
        raise ValueError(
            "Per-candidate SEC HTTP attempts cannot exceed the global attempt count."
        )

    candidate_rows = [
        {
            "security_id": str(_candidate_value(candidate, "security_id")),
            "last_price_date": _candidate_value(candidate, "last_price_date"),
        }
        for candidate in candidates
    ]
    candidate_frame = pd.DataFrame(
        candidate_rows,
        columns=("security_id", "last_price_date"),
    )
    versions = {
        str(dataset): str(version)
        for dataset, version in sorted(dataset_versions.items())
    }
    hints_sha256 = hashlib.sha256(Path(hints_path).read_bytes()).hexdigest()
    versions_sha256 = _json_sha256(versions)
    config = {
        "report_schema": LIFECYCLE_REPORT_SCHEMA,
        "collector_version": LIFECYCLE_COLLECTOR_VERSION,
        "collector_config_version": LIFECYCLE_COLLECTOR_CONFIG_VERSION,
        "candidate_selection_rule": DEFAULT_SELECTION_RULE,
        "hints_sha256": hints_sha256,
        "sec_fetch_policy": fetch_policy,
        "sec_max_http_attempts": global_cap,
        "sec_max_http_attempts_per_candidate": candidate_cap,
        "sec_max_http_attempts_per_request": request_cap,
    }
    binding: dict[str, Any] = {
        **config,
        "release_version": str(release_version),
        "completed_session": str(completed_session),
        "candidate_count": len(candidate_rows),
        "candidate_set_sha256": lifecycle_candidate_set_sha256(
            candidate_frame,
            selection_rule=DEFAULT_SELECTION_RULE,
        ),
        "input_dataset_versions": versions,
        "input_dataset_versions_sha256": versions_sha256,
        "collector_config_sha256": _json_sha256(config),
        "sec_http_attempts": actual_attempts,
        "sec_http_attempts_by_candidate": attempts_by_candidate,
    }
    binding["collection_context_sha256"] = _json_sha256(binding)
    return binding


def update_lifecycle_report_http_attempts(
    binding: Mapping[str, Any],
    *,
    sec_http_attempts: int,
    sec_http_attempts_by_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Refresh only the audited runtime counters and their context digest."""

    output = dict(binding)
    actual_attempts = _nonnegative_int(sec_http_attempts, "sec_http_attempts")
    global_cap = _positive_int(
        output.get("sec_max_http_attempts"),
        "sec_max_http_attempts",
    )
    candidate_cap = _positive_int(
        output.get("sec_max_http_attempts_per_candidate"),
        "sec_max_http_attempts_per_candidate",
    )
    attempts_by_candidate = {
        str(key): _nonnegative_int(value, f"sec_http_attempts_by_candidate[{key!r}]")
        for key, value in sorted(sec_http_attempts_by_candidate.items())
    }
    if actual_attempts > global_cap:
        raise ValueError("SEC HTTP attempt count exceeds its report cap.")
    if any(value > candidate_cap for value in attempts_by_candidate.values()):
        raise ValueError("SEC candidate HTTP attempt count exceeds its report cap.")
    if sum(attempts_by_candidate.values()) > actual_attempts:
        raise ValueError(
            "Per-candidate SEC HTTP attempts cannot exceed the global attempt count."
        )
    output["sec_http_attempts"] = actual_attempts
    output["sec_http_attempts_by_candidate"] = attempts_by_candidate
    output.pop("collection_context_sha256", None)
    output["collection_context_sha256"] = _json_sha256(output)
    return output


def validate_lifecycle_report_binding(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    purpose: str,
) -> None:
    """Reject a report collected under any different audited context."""

    for field in REPORT_BINDING_FIELDS:
        wanted = expected.get(field)
        observed = report.get(field)
        if observed != wanted:
            raise RuntimeError(
                f"Lifecycle report provenance mismatch during {purpose}: "
                f"field={field}, expected={wanted!r}, found={observed!r}. "
                "The existing records cannot be reused; rebuild the report from "
                "the current release and exact hints file."
            )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer.") from exc
    if parsed < 1 or str(parsed) != str(value).strip():
        raise ValueError(f"{field} must be a positive canonical integer.")
    return parsed


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer.") from exc
    if parsed < 0 or str(parsed) != str(value).strip():
        raise ValueError(f"{field} must be a non-negative canonical integer.")
    return parsed
