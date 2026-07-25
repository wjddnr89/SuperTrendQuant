"""Publication-only reviewed exceptions for delayed terminal settlements.

The generic terminal-transition audit remains strict.  This module may filter
a delayed terminal action only when the exact release proves either that the
security left every target index long before its terminal price and later had
an official zero-distribution cancellation, or that an official depositary
notice mandates a positive cash settlement on the later engine session.  In
both cases the action, resolution, terminal price, and archived source are
exactly bound and code-pinned.

The current index-removal rows are intentionally described as non-primary
history provenance.  They are not promoted to official index-provider facts;
using any exception therefore degrades release quality and emits a warning.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yaml

from .manifest import sha256_bytes


DEFAULT_REVIEWED_TERMINAL_READINESS_EXCEPTIONS = (
    Path(__file__).resolve().parents[3]
    / "configs/drafts/us_terminal_readiness_reviewed_exceptions.yaml"
)

TARGET_INDEX_IDS = frozenset({"sp500", "nasdaq100"})
POLICY_CODE = "removed_from_target_indices_before_terminal_price/v1"
INDEX_PROVENANCE_CLASS = "reviewed_non_primary_index_history"
CASH_SETTLEMENT_POLICY_CODE = "official_delayed_cash_settlement/v1"
CASH_SETTLEMENT_INDEX_PROVENANCE_CLASS = "no_target_index_membership"

TRUSTED_REVIEWED_TERMINAL_READINESS_EXCEPTION_EVENT_IDS = frozenset(
    {
        "1377981293c7eebce5cb6da722da3c4058a29077c96847b258d21df5af601902",
        "1849ca428dda73f0322d36334d81ce5a00f0185702e3ea2b7e5e4fea6fdb7704",
        "4294b6bfa674fab682ba9c299b4fae27ae54b60081c58140f6846b933a47e1ef",
        "6b6b3440b4c3c0466e5b8d2ee6a8339cd230998a837c7cb573dae75fff565b98",
        "7d150e99cfe15587e4e9994dfaebde08942117f970a7d11ce94fd05b84bc85f5",
        "d1eeadf8f4779212ff4b3162af1731bf7bfc804c10756cc4bb8074c307e1f746",
    }
)

# Updated only after the complete normalized draft registry is reviewed.
TRUSTED_REVIEWED_TERMINAL_READINESS_EXCEPTIONS_SHA256 = (
    "5fdddec490aa6366595f22c8ac0b00ffe35df1bcc26e9ac3083875bef41fc3d6"
)

# ``action_row_sha256`` in the reviewed registry pins the complete row as it
# existed at review time.  Later whole-dataset rewrites legitimately refresh
# transport metadata such as ``retrieved_at`` and may add extension columns,
# so publication uses this narrower, separately code-pinned projection.  The
# projection still covers every identity, economic, date, and source-binding
# field; only non-economic retrieval/extension metadata is excluded.
_REVIEWED_ACTION_PROJECTION_FIELDS = (
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
    "source_hash",
)

TRUSTED_REVIEWED_TERMINAL_ACTION_PROJECTION_SHA256: Mapping[str, str] = {
    "1377981293c7eebce5cb6da722da3c4058a29077c96847b258d21df5af601902": (
        "8c5ce268e00fadf7b90aeeb032c46b08ab52023856ae2c531490761a9d5ece3f"
    ),
    "1849ca428dda73f0322d36334d81ce5a00f0185702e3ea2b7e5e4fea6fdb7704": (
        "e01a5a74003791ecf9bddbbfe04e835b88136f0ca8dd7deb1a65ec9cdd757b88"
    ),
    "4294b6bfa674fab682ba9c299b4fae27ae54b60081c58140f6846b933a47e1ef": (
        "45efd6c37c17217733fd5ccb29ba2b798ded8a8876628fee6a5a87f2af53b349"
    ),
    "6b6b3440b4c3c0466e5b8d2ee6a8339cd230998a837c7cb573dae75fff565b98": (
        "2f2d5bbf3e7b9f6f47fc4bddb4f88e397a610477df6e41dcbacb8359c992ac69"
    ),
    "7d150e99cfe15587e4e9994dfaebde08942117f970a7d11ce94fd05b84bc85f5": (
        "5e669752436a0d0e54ac44bbbfedfdc127f7266479d20973cb0ebe7a53fa0829"
    ),
    "d1eeadf8f4779212ff4b3162af1731bf7bfc804c10756cc4bb8074c307e1f746": (
        "927379c392f9fc8c4f0434f9c71d2ffef7b53ce80eb0117423eec8b7534ee48e"
    ),
}

_REVIEWED_RESOLUTION_PROJECTION_FIELDS = (
    "candidate_id",
    "security_id",
    "symbol",
    "last_price_date",
    "resolution",
    "event_id",
    "exception_code",
    "exception_reason",
    "recheck_after",
    "successor_security_id",
    "successor_symbol",
    "source_url",
    "source",
    "source_hash",
)

TRUSTED_REVIEWED_TERMINAL_RESOLUTION_PROJECTION_SHA256: Mapping[str, str] = {
    "1377981293c7eebce5cb6da722da3c4058a29077c96847b258d21df5af601902": (
        "22a31168d69799f1c14a94e702a4f52ce1eb98b9d7f7e6470640c7208eec833e"
    ),
    "1849ca428dda73f0322d36334d81ce5a00f0185702e3ea2b7e5e4fea6fdb7704": (
        "70cdbc6015bf99e88da0eabd60fd4a33b10370a63e6eeeeb9b436458ba19c89b"
    ),
    "4294b6bfa674fab682ba9c299b4fae27ae54b60081c58140f6846b933a47e1ef": (
        "22624ac7b06b7f661f5eb56cffde70638c2f68265ca0497f291cc4f1b73dfa98"
    ),
    "6b6b3440b4c3c0466e5b8d2ee6a8339cd230998a837c7cb573dae75fff565b98": (
        "a54019f72d7b8adf38220040eb33fd044c408ae0a6487759df29c41e291ad0e4"
    ),
    "7d150e99cfe15587e4e9994dfaebde08942117f970a7d11ce94fd05b84bc85f5": (
        "9d34a6bcf5c5ea25771cecc9f6dac8af28e5edb0244747f7df6924653ec25bf6"
    ),
    "d1eeadf8f4779212ff4b3162af1731bf7bfc804c10756cc4bb8074c307e1f746": (
        "15dd6b7e3ac9e0973d38ef9bfa880c3796943696b247e6167b94b2e56f56fc4a"
    ),
}

_REVIEWED_REMOVAL_PROJECTION_FIELDS = (
    "event_id",
    "index_id",
    "announcement_date",
    "effective_date",
    "operation",
    "security_id",
    "official",
    "source",
    "source_url",
    "source_kind",
    "source_hash",
)

TRUSTED_REVIEWED_TERMINAL_REMOVAL_PROJECTION_SHA256: Mapping[str, str] = {
    "0f8cf68e134784aa4f5ea03139c48dae7da24fc70cca1b6307407f2948252ecd": (
        "b6347fb8be58edc7ad3c3541378a9e51d2ccc0507f666ab0b59ddff5ab5b100a"
    ),
    "276e90238b31de7ee65e99f2f9c94b7162a3df92f75fad995bfa13a58b501442": (
        "6a000f34fdc677cb62d6af37f2235431604a8efb774c6e39267254b6ee4d41c4"
    ),
    "578f85614508819dba917a8a06bb91f8e5fb0dcc5273895b9cc250773df7e952": (
        "6825af2ff02282c4091c25a8d2b7ec58e42e7dbd47cbee2e7dd5e619c81c6385"
    ),
    "74ac8771963c4c4f7a946e3afd6be226171e1171adc977318ede95d76eaf02a3": (
        "c18ad34cc8dfde1fb962c60bbd69478899b8036397558368df69b15706721fcd"
    ),
    "8986348b93af95d969f49e81b2a020047a4f559e33adf057cc7fc7b66013732f": (
        "53863f201543d7d2651882ffe8846ab0c76d883283f24ab33358784cc9af0017"
    ),
    "abced592a7a42801b82f2c9b307f05c9298b4c6fc1193651d6e1a7b6d40c7715": (
        "f5ddb2e5ab9a79a451bfc00458dbbc4dc73c263e36e001bbdea5636829539f58"
    ),
    "e1c383b41ee8132f0ee92a61936f6639939233dbd12a58bf819e58ed8328b2f2": (
        "6f5edc44d8be8fc538cf52c842242e71b2513cfc03c28f385c530cb58c0b4065"
    ),
}

_SPEC_FIELDS = (
    "event_id",
    "security_id",
    "symbol",
    "issue_code",
    "action_type",
    "last_price_session",
    "expected_transition_session",
    "engine_session",
    "action_date_field",
    "action_date",
    "action_official",
    "action_cash_amount",
    "action_ratio",
    "action_new_security_id",
    "action_new_symbol",
    "action_source_kind",
    "action_source",
    "action_source_url",
    "action_source_hash",
    "action_row_sha256",
    "resolution_candidate_id",
    "resolution_source_url",
    "resolution_source_hash",
    "resolution_row_sha256",
    "target_index_ids",
    "index_removals",
    "index_provenance_class",
    "policy_code",
    "required_release_warning",
)

_REMOVAL_FIELDS = (
    "event_id",
    "index_id",
    "effective_date",
    "operation",
    "official",
    "source",
    "source_kind",
    "source_url",
    "source_hash",
    "row_sha256",
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


def _digest(value: Any, field: str) -> str:
    output = _text(value).lower()
    _require(
        len(output) == 64
        and all(character in "0123456789abcdef" for character in output),
        f"Reviewed terminal readiness {field} must be lowercase SHA-256.",
    )
    return output


def _iso_date(value: Any, field: str) -> str:
    raw = _text(value)
    parsed = pd.to_datetime(raw, errors="coerce")
    output = "" if pd.isna(parsed) else pd.Timestamp(parsed).date().isoformat()
    _require(
        bool(output) and raw == output,
        f"Reviewed terminal readiness {field} must be an exact ISO date.",
    )
    return output


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(value))


def _row_scalar(value: Any) -> Any:
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def release_row_sha256(row: Mapping[str, Any]) -> str:
    """Hash every field in one stored release row with stable null handling."""

    return _canonical_json_sha256(
        {str(key): _row_scalar(value) for key, value in row.items()}
    )


def reviewed_terminal_action_projection_sha256(row: Mapping[str, Any]) -> str:
    """Hash the fixed reviewed economic/identity/source action projection."""

    return _canonical_json_sha256(
        {
            field: _row_scalar(row.get(field))
            for field in _REVIEWED_ACTION_PROJECTION_FIELDS
        }
    )


def reviewed_terminal_resolution_projection_sha256(row: Mapping[str, Any]) -> str:
    """Hash the fixed reviewed resolution identity/source projection."""

    return _canonical_json_sha256(
        {
            field: _row_scalar(row.get(field))
            for field in _REVIEWED_RESOLUTION_PROJECTION_FIELDS
        }
    )


def reviewed_terminal_removal_projection_sha256(row: Mapping[str, Any]) -> str:
    """Hash the fixed reviewed index-removal identity/source projection."""

    return _canonical_json_sha256(
        {
            field: _row_scalar(row.get(field))
            for field in _REVIEWED_REMOVAL_PROJECTION_FIELDS
        }
    )


def _canonical_removal(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "Reviewed index removal must be an object.")
    _require(
        set(value) == set(_REMOVAL_FIELDS),
        "Reviewed index-removal fields are not exact.",
    )
    official = value.get("official")
    _require(type(official) is bool, "Reviewed index-removal official flag is invalid.")
    output = {
        "event_id": _digest(value.get("event_id"), "index_removal.event_id"),
        "index_id": _text(value.get("index_id")).lower(),
        "effective_date": _iso_date(
            value.get("effective_date"), "index_removal.effective_date"
        ),
        "operation": _text(value.get("operation")).upper(),
        "official": official,
        "source": _text(value.get("source")),
        "source_kind": _text(value.get("source_kind")),
        "source_url": _text(value.get("source_url")),
        "source_hash": _digest(
            value.get("source_hash"), "index_removal.source_hash"
        ),
        "row_sha256": _digest(
            value.get("row_sha256"), "index_removal.row_sha256"
        ),
    }
    _require(
        output["index_id"] in TARGET_INDEX_IDS,
        "Reviewed index removal is outside the target index inventory.",
    )
    _require(
        output["operation"] == "REMOVE",
        "Reviewed index-removal operation must be REMOVE.",
    )
    _require(
        output["official"] is False
        and output["source_kind"] in {"community", "derived_identity"},
        "Reviewed non-primary index provenance is not described exactly.",
    )
    _require(
        bool(output["source"] and output["source_url"]),
        "Reviewed index-removal provenance is incomplete.",
    )
    return output


def canonical_reviewed_terminal_readiness_exception(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one exact delayed-terminal exception."""

    _require(
        isinstance(value, Mapping),
        "Reviewed terminal readiness exception must be an object.",
    )
    _require(
        set(value) == set(_SPEC_FIELDS),
        "Reviewed terminal readiness exception fields are not exact.",
    )
    action_official = value.get("action_official")
    _require(
        type(action_official) is bool,
        "Reviewed terminal readiness action_official flag is invalid.",
    )
    action_cash = value.get("action_cash_amount")
    _require(
        isinstance(action_cash, (int, float))
        and not isinstance(action_cash, bool)
        and np.isfinite(float(action_cash))
        and float(action_cash) >= 0.0,
        "Reviewed terminal readiness action cash amount is invalid.",
    )
    action_ratio = value.get("action_ratio")
    _require(
        action_ratio is None,
        "Reviewed terminal readiness action ratio must be null.",
    )
    removals = [_canonical_removal(item) for item in value.get("index_removals") or ()]
    removals.sort(key=lambda item: (item["index_id"], item["effective_date"]))
    target_index_ids = sorted(
        {_text(item).lower() for item in value.get("target_index_ids") or ()}
    )
    output = {
        "event_id": _digest(value.get("event_id"), "event_id"),
        "security_id": _text(value.get("security_id")),
        "symbol": _text(value.get("symbol")).upper(),
        "issue_code": _text(value.get("issue_code")),
        "action_type": _text(value.get("action_type")).lower(),
        "last_price_session": _iso_date(
            value.get("last_price_session"), "last_price_session"
        ),
        "expected_transition_session": _iso_date(
            value.get("expected_transition_session"),
            "expected_transition_session",
        ),
        "engine_session": _iso_date(value.get("engine_session"), "engine_session"),
        "action_date_field": _text(value.get("action_date_field")),
        "action_date": _iso_date(value.get("action_date"), "action_date"),
        "action_official": action_official,
        "action_cash_amount": float(action_cash),
        "action_ratio": None,
        "action_new_security_id": _text(value.get("action_new_security_id")),
        "action_new_symbol": _text(value.get("action_new_symbol")).upper(),
        "action_source_kind": _text(value.get("action_source_kind")),
        "action_source": _text(value.get("action_source")),
        "action_source_url": _text(value.get("action_source_url")),
        "action_source_hash": _digest(
            value.get("action_source_hash"), "action_source_hash"
        ),
        "action_row_sha256": _digest(
            value.get("action_row_sha256"), "action_row_sha256"
        ),
        "resolution_candidate_id": _digest(
            value.get("resolution_candidate_id"), "resolution_candidate_id"
        ),
        "resolution_source_url": _text(value.get("resolution_source_url")),
        "resolution_source_hash": _digest(
            value.get("resolution_source_hash"), "resolution_source_hash"
        ),
        "resolution_row_sha256": _digest(
            value.get("resolution_row_sha256"), "resolution_row_sha256"
        ),
        "target_index_ids": target_index_ids,
        "index_removals": removals,
        "index_provenance_class": _text(value.get("index_provenance_class")),
        "policy_code": _text(value.get("policy_code")),
        "required_release_warning": _text(value.get("required_release_warning")),
    }
    _require(
        bool(output["security_id"] and output["symbol"]),
        "Reviewed terminal readiness identity is incomplete.",
    )
    _require(
        output["issue_code"] == "terminal_action_after_expected_session"
        and output["action_type"] == "delisting"
        and output["action_date_field"] == "ex_date",
        "Reviewed terminal readiness issue/action scope is invalid.",
    )
    _require(
        output["action_official"] is True
        and not output["action_new_security_id"]
        and not output["action_new_symbol"],
        "Reviewed terminal readiness official terminal action is invalid.",
    )
    _require(
        output["action_date"] == output["engine_session"]
        and output["engine_session"] > output["expected_transition_session"],
        "Reviewed terminal readiness delayed engine boundary is invalid.",
    )
    calendar = xcals.get_calendar("XNYS")
    terminal = pd.Timestamp(output["last_price_session"])
    _require(
        calendar.is_session(terminal),
        "Reviewed terminal readiness last price is not an XNYS session.",
    )
    expected = pd.Timestamp(calendar.next_session(terminal)).tz_localize(None)
    _require(
        expected.date().isoformat() == output["expected_transition_session"],
        "Reviewed terminal readiness expected transition is not the next XNYS session.",
    )
    if output["policy_code"] == POLICY_CODE:
        _require(
            output["action_cash_amount"] == 0.0
            and output["action_source_kind"] == "official_crosscheck"
            and output["action_source"] == "sec_edgar+stored_price_crosscheck",
            "Reviewed terminal readiness zero-cancellation action is invalid.",
        )
        _require(
            target_index_ids
            and set(target_index_ids) == {item["index_id"] for item in removals}
            and len(removals) == len(target_index_ids),
            "Reviewed terminal readiness target-index removal inventory is not exact.",
        )
        last_price = pd.Timestamp(output["last_price_session"])
        _require(
            all(
                pd.Timestamp(item["effective_date"]) + pd.Timedelta(days=365)
                < last_price
                for item in removals
            ),
            "Reviewed index removal must precede the terminal price by more than one year.",
        )
        _require(
            output["index_provenance_class"] == INDEX_PROVENANCE_CLASS,
            "Reviewed terminal readiness index provenance is invalid.",
        )
    elif output["policy_code"] == CASH_SETTLEMENT_POLICY_CODE:
        _require(
            output["action_cash_amount"] > 0.0
            and output["action_source_kind"]
            == "depositary_corporate_action_notice"
            and output["action_source"] == "official_ntcoy_cash_termination"
            and not target_index_ids
            and not removals
            and output["index_provenance_class"]
            == CASH_SETTLEMENT_INDEX_PROVENANCE_CLASS,
            "Reviewed terminal readiness cash-settlement scope is invalid.",
        )
    else:
        raise RuntimeError("Reviewed terminal readiness policy code is invalid.")
    _require(
        bool(output["required_release_warning"]),
        "Reviewed terminal readiness policy warning is missing.",
    )
    return output


def reviewed_terminal_readiness_exceptions(
    policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _require(
        isinstance(policy, Mapping)
        and set(policy) == {"schema", "reviewed_terminal_readiness_exceptions"}
        and _text(policy.get("schema"))
        == "us_terminal_readiness_reviewed_exceptions/v1",
        "Reviewed terminal readiness policy envelope is invalid.",
    )
    raw = policy.get("reviewed_terminal_readiness_exceptions")
    _require(
        isinstance(raw, list),
        "Reviewed terminal readiness exception registry must be a list.",
    )
    output: dict[str, dict[str, Any]] = {}
    for value in raw:
        normalized = canonical_reviewed_terminal_readiness_exception(value)
        event_id = str(normalized["event_id"])
        _require(event_id not in output, f"Duplicate reviewed event_id: {event_id}")
        output[event_id] = normalized
    _require(
        set(output) == set(TRUSTED_REVIEWED_TERMINAL_READINESS_EXCEPTION_EVENT_IDS),
        "Reviewed terminal readiness event inventory changed.",
    )
    return output


def reviewed_terminal_readiness_exception_inventory_sha256(
    policy: Mapping[str, Any],
) -> str:
    return _canonical_json_sha256(reviewed_terminal_readiness_exceptions(policy))


def load_code_pinned_reviewed_terminal_readiness_exceptions(
    path: str | Path = DEFAULT_REVIEWED_TERMINAL_READINESS_EXCEPTIONS,
) -> dict[str, dict[str, Any]]:
    policy = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    registry = reviewed_terminal_readiness_exceptions(policy)
    actual = _canonical_json_sha256(registry)
    _require(
        actual == TRUSTED_REVIEWED_TERMINAL_READINESS_EXCEPTIONS_SHA256,
        "Reviewed terminal readiness exception registry is not code-pinned.",
    )
    return registry


def _unique_row(frame: pd.DataFrame, field: str, value: str, label: str) -> dict[str, Any]:
    rows = frame.loc[frame[field].astype(str).eq(value)]
    _require(len(rows) == 1, f"Reviewed terminal readiness {label} is not unique.")
    return rows.iloc[0].to_dict()


def _archive_binding_exists(
    archives: pd.DataFrame, source_hash: str, source_url: str
) -> bool:
    rows = archives.loc[
        archives["archive_id"].astype(str).eq(source_hash)
        & archives["source_hash"].astype(str).eq(source_hash)
        & archives["source_url"].astype(str).eq(source_url)
    ]
    return len(rows) == 1


def _validate_issue(issue: Any, spec: Mapping[str, Any]) -> None:
    actual = issue.to_dict()
    expected = {
        "code": spec["issue_code"],
        "security_id": spec["security_id"],
        "symbol": spec["symbol"],
        "event_id": spec["event_id"],
        "action_type": spec["action_type"],
        "last_price_session": spec["last_price_session"],
        "expected_transition_session": spec["expected_transition_session"],
        "engine_session": spec["engine_session"],
        "action_date_field": spec["action_date_field"],
        "action_date": spec["action_date"],
        "first_reentry_session": "",
        "affected_index_ids": [],
        "successor_security_id": "",
        "successor_symbol": "",
        "successor_blockers": [],
    }
    for field, value in expected.items():
        _require(
            actual.get(field) == value,
            f"Reviewed terminal readiness issue drifted: {spec['symbol']}/{field}.",
        )


def _validate_action_and_resolution(
    actions: pd.DataFrame,
    resolutions: pd.DataFrame,
    archives: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    event_id = str(spec["event_id"])
    action = _unique_row(actions, "event_id", event_id, "action")
    _require(
        set(TRUSTED_REVIEWED_TERMINAL_ACTION_PROJECTION_SHA256)
        == set(TRUSTED_REVIEWED_TERMINAL_READINESS_EXCEPTION_EVENT_IDS),
        "Reviewed terminal readiness action projection inventory is not exact.",
    )
    _require(
        reviewed_terminal_action_projection_sha256(action)
        == TRUSTED_REVIEWED_TERMINAL_ACTION_PROJECTION_SHA256[event_id],
        f"Reviewed terminal readiness action row drifted: {spec['symbol']}.",
    )
    action_expected = {
        "event_id": event_id,
        "security_id": spec["security_id"],
        "action_type": spec["action_type"],
        "effective_date": spec["action_date"],
        "ex_date": spec["action_date"],
        "cash_amount": spec["action_cash_amount"],
        "ratio": None,
        "new_security_id": spec["action_new_security_id"],
        "new_symbol": spec["action_new_symbol"],
        "official": True,
        "source_kind": spec["action_source_kind"],
        "source": spec["action_source"],
        "source_url": spec["action_source_url"],
        "source_hash": spec["action_source_hash"],
    }
    if spec["policy_code"] == CASH_SETTLEMENT_POLICY_CODE:
        action_expected["payment_date"] = spec["action_date"]
    for field, value in action_expected.items():
        actual = _row_scalar(action.get(field))
        _require(
            actual == value,
            f"Reviewed terminal readiness action drifted: {spec['symbol']}/{field}.",
        )
    _require(
        _archive_binding_exists(
            archives,
            str(spec["action_source_hash"]),
            str(spec["action_source_url"]),
        ),
        f"Reviewed terminal readiness action archive is missing: {spec['symbol']}.",
    )

    resolution = _unique_row(resolutions, "event_id", event_id, "resolution")
    _require(
        reviewed_terminal_resolution_projection_sha256(resolution)
        == TRUSTED_REVIEWED_TERMINAL_RESOLUTION_PROJECTION_SHA256[event_id],
        f"Reviewed terminal readiness resolution row drifted: {spec['symbol']}.",
    )
    resolution_expected = {
        "candidate_id": spec["resolution_candidate_id"],
        "security_id": spec["security_id"],
        "symbol": spec["symbol"],
        "last_price_date": spec["last_price_session"],
        "resolution": "applied",
        "event_id": event_id,
        "exception_code": "",
        "exception_reason": "",
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        "source_url": spec["resolution_source_url"],
        "source": (
            "official_ntcoy_cash_termination"
            if spec["policy_code"] == CASH_SETTLEMENT_POLICY_CODE
            else "lifecycle_finalizer"
        ),
        "source_hash": spec["resolution_source_hash"],
    }
    for field, value in resolution_expected.items():
        _require(
            _row_scalar(resolution.get(field)) == value,
            f"Reviewed terminal readiness resolution drifted: {spec['symbol']}/{field}.",
        )
    _require(
        spec["resolution_source_url"] == spec["action_source_url"]
        and spec["resolution_source_hash"] == spec["action_source_hash"],
        "Reviewed terminal readiness action/resolution evidence is not identical.",
    )


def _validate_index_exit_and_no_reentry(
    events: pd.DataFrame,
    anchors: pd.DataFrame,
    prices: pd.DataFrame,
    archives: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    security_id = str(spec["security_id"])
    security_events = events.loc[events["security_id"].astype(str).eq(security_id)].copy()
    security_anchors = anchors.loc[
        anchors["security_id"].astype(str).eq(security_id)
    ].copy()
    known_indices = {
        _text(value).lower()
        for value in pd.concat(
            [
                security_events.get("index_id", pd.Series(dtype="object")),
                security_anchors.get("index_id", pd.Series(dtype="object")),
            ],
            ignore_index=True,
        )
        if _text(value).lower() in TARGET_INDEX_IDS
    }
    _require(
        known_indices == set(spec["target_index_ids"]),
        f"Reviewed terminal readiness historical index inventory drifted: {spec['symbol']}.",
    )

    for removal in spec["index_removals"]:
        row = _unique_row(
            security_events,
            "event_id",
            str(removal["event_id"]),
            "index removal",
        )
        _require(
            reviewed_terminal_removal_projection_sha256(row)
            == TRUSTED_REVIEWED_TERMINAL_REMOVAL_PROJECTION_SHA256[
                str(removal["event_id"])
            ],
            f"Reviewed terminal readiness index-removal row drifted: "
            f"{spec['symbol']}/{removal['index_id']}.",
        )
        expected = {
            "event_id": removal["event_id"],
            "index_id": removal["index_id"],
            "effective_date": removal["effective_date"],
            "operation": removal["operation"],
            "security_id": security_id,
            "official": removal["official"],
            "source": removal["source"],
            "source_kind": removal["source_kind"],
            "source_url": removal["source_url"],
            "source_hash": removal["source_hash"],
        }
        for field, value in expected.items():
            _require(
                _row_scalar(row.get(field)) == value,
                f"Reviewed terminal readiness index removal drifted: "
                f"{spec['symbol']}/{removal['index_id']}/{field}.",
            )
        _require(
            _archive_binding_exists(
                archives,
                str(removal["source_hash"]),
                str(removal["source_url"]),
            ),
            f"Reviewed terminal readiness index archive is missing: "
            f"{spec['symbol']}/{removal['index_id']}.",
        )

        removal_date = str(removal["effective_date"])
        index_events = security_events.loc[
            security_events["index_id"].astype(str).str.lower().eq(
                str(removal["index_id"])
            )
        ]
        later_events = index_events.loc[
            pd.to_datetime(index_events["effective_date"], errors="coerce")
            > pd.Timestamp(removal_date)
        ]
        _require(
            later_events.empty,
            f"Reviewed terminal readiness index event re-entry exists: "
            f"{spec['symbol']}/{removal['index_id']}.",
        )
        later_anchors = security_anchors.loc[
            security_anchors["index_id"].astype(str).str.lower().eq(
                str(removal["index_id"])
            )
            & (
                pd.to_datetime(security_anchors["anchor_date"], errors="coerce")
                > pd.Timestamp(removal_date)
            )
        ]
        _require(
            later_anchors.empty,
            f"Reviewed terminal readiness anchor re-entry exists: "
            f"{spec['symbol']}/{removal['index_id']}.",
        )

    security_prices = prices.loc[
        prices["security_id"].astype(str).eq(security_id)
    ].copy()
    sessions = pd.to_datetime(security_prices["session"], errors="coerce")
    _require(
        not security_prices.empty
        and sessions.notna().all()
        and sessions.max().date().isoformat() == spec["last_price_session"],
        f"Reviewed terminal readiness terminal price boundary drifted: {spec['symbol']}.",
    )
    terminal_rows = security_prices.loc[
        sessions.dt.date.astype(str).eq(str(spec["last_price_session"]))
    ]
    terminal_close = pd.to_numeric(terminal_rows["close"], errors="coerce")
    _require(
        len(terminal_rows) == 1
        and terminal_close.notna().all()
        and np.isfinite(terminal_close).all()
        and bool((terminal_close > 0).all()),
        f"Reviewed terminal readiness final close is invalid: {spec['symbol']}.",
    )


def validate_publication_terminal_readiness_exceptions(
    repository,
    release,
    report,
    *,
    policy_path: str | Path = DEFAULT_REVIEWED_TERMINAL_READINESS_EXCEPTIONS,
) -> dict[str, Any]:
    """Apply exact reviewed exceptions to one strict terminal audit report.

    This is intentionally a publication helper, not a replacement for
    ``TerminalTransitionReport.raise_for_errors``.
    """

    registry = load_code_pinned_reviewed_terminal_readiness_exceptions(policy_path)
    removal_event_ids = {
        str(removal["event_id"])
        for spec in registry.values()
        for removal in spec["index_removals"]
    }
    _require(
        set(TRUSTED_REVIEWED_TERMINAL_ACTION_PROJECTION_SHA256)
        == set(TRUSTED_REVIEWED_TERMINAL_RESOLUTION_PROJECTION_SHA256)
        == set(TRUSTED_REVIEWED_TERMINAL_READINESS_EXCEPTION_EVENT_IDS)
        and removal_event_ids.issubset(
            TRUSTED_REVIEWED_TERMINAL_REMOVAL_PROJECTION_SHA256
        ),
        "Reviewed terminal readiness projection pin inventory is not exact.",
    )
    issue_by_event: dict[str, list[Any]] = {}
    for issue in report.issues:
        issue_by_event.setdefault(_text(issue.event_id), []).append(issue)

    versions = release.dataset_versions
    required = (
        "corporate_actions",
        "lifecycle_resolutions",
        "index_membership_events",
        "index_constituent_anchors",
        "daily_price_raw",
        "source_archive",
    )
    missing = [dataset for dataset in required if not versions.get(dataset)]
    _require(
        not missing,
        "Reviewed terminal readiness release datasets are missing: "
        + ", ".join(missing),
    )
    frames = {
        dataset: repository.read_frame(dataset, versions[dataset])
        for dataset in required
    }

    reviewed_event_ids: set[str] = set()
    reviewed_details: list[dict[str, Any]] = []
    for event_id, spec in sorted(registry.items()):
        issues = issue_by_event.get(event_id, [])
        # A later exact repair may make the strict audit issue disappear.  An
        # unused reviewed policy must never turn that improvement into a new
        # blocker; every issue that is present is still validated fail-closed.
        if not issues:
            continue
        _require(
            len(issues) == 1,
            f"Reviewed terminal readiness issue is duplicated: "
            f"{spec['symbol']}.",
        )
        issue = issues[0]
        _validate_issue(issue, spec)
        _validate_action_and_resolution(
            frames["corporate_actions"],
            frames["lifecycle_resolutions"],
            frames["source_archive"],
            spec,
        )
        _validate_index_exit_and_no_reentry(
            frames["index_membership_events"],
            frames["index_constituent_anchors"],
            frames["daily_price_raw"],
            frames["source_archive"],
            spec,
        )
        reviewed_event_ids.add(event_id)
        reviewed_details.append(
            {
                "event_id": event_id,
                "security_id": spec["security_id"],
                "symbol": spec["symbol"],
                "issue_code": spec["issue_code"],
                "last_price_session": spec["last_price_session"],
                "expected_transition_session": spec[
                    "expected_transition_session"
                ],
                "engine_session": spec["engine_session"],
                "action_source_url": spec["action_source_url"],
                "action_source_hash": spec["action_source_hash"],
                "index_removals": list(spec["index_removals"]),
                "index_provenance_class": spec["index_provenance_class"],
                "policy_code": spec["policy_code"],
                "warning": spec["required_release_warning"],
            }
        )

    remaining = tuple(
        issue
        for issue in report.issues
        if not (
            issue.code == "terminal_action_after_expected_session"
            and issue.event_id in reviewed_event_ids
        )
    )
    base = report.to_dict()
    raw_issue_count = len(report.issues)
    raw_issue_counts = dict(base["issue_counts"])
    raw_risk_symbols = list(base["risk_symbols"])
    raw_risk_security_ids = list(base["risk_security_ids"])
    remaining_counts = dict(sorted(Counter(item.code for item in remaining).items()))
    warnings = [str(item["warning"]) for item in reviewed_details]
    return {
        **base,
        "raw_ready": report.ready,
        "raw_issue_count": raw_issue_count,
        "raw_issue_counts": raw_issue_counts,
        "raw_risk_symbols": raw_risk_symbols,
        "raw_risk_security_ids": raw_risk_security_ids,
        "ready": not remaining,
        "issue_count": len(remaining),
        "issue_counts": remaining_counts,
        "risk_symbols": sorted({item.symbol for item in remaining if item.symbol}),
        "risk_security_ids": sorted(
            {item.security_id for item in remaining if item.security_id}
        ),
        "issues": [item.to_dict() for item in remaining],
        "reviewed_exception_count": len(reviewed_details),
        "reviewed_exception_inventory_sha256": (
            TRUSTED_REVIEWED_TERMINAL_READINESS_EXCEPTIONS_SHA256
        ),
        "reviewed_exceptions": reviewed_details,
        "quality": "degraded" if reviewed_details else "validated",
        "quality_degraded": bool(reviewed_details),
        "release_warnings": warnings,
    }
