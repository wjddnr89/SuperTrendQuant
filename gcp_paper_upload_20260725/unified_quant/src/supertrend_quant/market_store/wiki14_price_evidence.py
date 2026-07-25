"""Fail-closed price-only validation for the frozen 14-symbol WIKI archive.

The reviewed archive is useful only as an identity-bound raw OHLCV arbiter.
Its formal Kaggle license is ``Unknown`` and its corporate-action fields are
not accepted as authoritative.  This module therefore replays the exact local
price relation and Triple Supertrend sensitivity while keeping actions and
adjustment factors explicitly unvalidated.

The YAML registry is not authority by itself: both its complete normalized
inventory and the immutable provenance payload are pinned in code.
"""

from __future__ import annotations

import gzip
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..indicators import add_triple_supertrend
from .adjustments import apply_adjustment_factors
from .manifest import sha256_bytes


REVIEWED_WIKI14_PRICE_ONLY_BASIS = "frozen_wiki14_identity_bound_price_only/v1"
REVIEWED_WIKI14_PRICE_ONLY_POLICY_KEY = "reviewed_wiki14_price_only_evidence"
WIKI14_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "marketneutral/quandl-wiki-prices-us-equites/WIKI_PRICES.csv"
)
WIKI14_LICENSE_WARNING = (
    "Kaggle Quandl WIKI licenseName=Unknown; private/internal-only; "
    "redistribution/public publication blocked."
)
WIKI14_LIMITATION = (
    "Frozen identity-bound Quandl WIKI raw OHLCV is accepted for price-only "
    "arbitration. Its Unknown license restricts use to private/internal-only, "
    "and it never attests corporate actions or adjustment factors."
)
WIKI14_PROVENANCE_SHA256 = (
    "16691eab9edc01f626d00551ba17e922d3f869d928c13478aa0443fbc329209e"
)
WIKI14_PROVENANCE_RETRIEVED_AT = "2026-07-19T12:00:00Z"
WIKI14_EXTRACT_RETRIEVED_AT = "2026-07-18T03:58:26.808706Z"
WIKI14_ARCHIVE_EFFECTIVE_DATE = "2026-07-15"
WIKI14_EXTRACT_INVENTORY_SHA256 = (
    "173635ad3c82264826d118bcfd963cc884e50365539ee623c4b24d682060b0f5"
)
WIKI14_IDENTITY_SCHEMA_INVENTORY_SHA256 = (
    "1c82f703ed1e88790dd1b066614c25389bb4a442c3e0369655fad5583b14880b"
)
WIKI14_ARCHIVE_ARTIFACT_INVENTORY_SHA256 = (
    "134d0d92fa4e31e6c4deb0ab7fa0a57ccf865e0e7d01f712c08d29e87b493ab2"
)

WIKI_COLUMNS = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ex-dividend",
    "split_ratio",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
)
SIGNAL_COLUMNS = (
    "TripleST1_Trend",
    "TripleST2_Trend",
    "TripleST3_Trend",
    "TripleAllUp",
    "TripleDownCount",
    "TripleBuySignal",
    "TripleSellSignal",
)
SPEC_FIELDS = (
    "target_id",
    "security_id",
    "symbol",
    "target_provider_symbol",
    "identity_active_from",
    "identity_active_to",
    "terminal_event_id",
    "identity_bound_provider_symbol",
    "extract_sha256",
)

TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS = frozenset(
    {
        "06b8e2e6e06e71b682f4792e0bc3cff9ace1c6bea0a4d07ee4e7aa3f99afa2d7",
        "1bbd0db44c10b63aa9fb4a2ee1399e83b699a97dd343d0e541682b6066af659a",
        "296e2d263b549d741aea400cfe59b164b8f06e7420426e95d251f7ee926e26b0",
        "599705d408171c2606e2a09958f8fccca2571bd1df4ca7dbadcd4074bbdeebba",
        "5dd6c8006be4eeb5f7947aac282769adc1239c83ec9d14169fba5eea8e564f79",
        "7516f04c6e27d612002fc1d3468720f45ccc1b141e7e8a5fe78564c261d87729",
        "7f1cdc57371b12e2913ee5d00f5e262c91a800648786c55d0e78bf97d08270b8",
        "9aff0d4d65ce790b98d6f050f6323044ce01a2fbd378c93b5c0f668f9fffab44",
        "a17b2c14178839f2206e64fef7f77c68d90a2849d3e81b782401348c0b6e5c71",
        "addfc00ac2573c1efd61fff6686797169ffdc2b1c6da1552521d936d80b19a03",
        "b5f68dcfc88bb04c75a326e0667864ba403d443ffb195c6dcab2c94d36ca576a",
        "d04a3aa3adfa1f2290f64d1fa9747fb1a1d59d5e106a0e4e4b2841f024ff4c08",
        "da410eb4d833807a0cb86317cdd3c14011b49d06bf34528abbe7e75c6e32eafa",
        "f3e261c422fd805f271ad9104511d7a1a76e7228adbee3494a964399a2ceb235",
    }
)

# Filled after canonicalizing the complete YAML registry.  A target cannot be
# added, removed, or rebound by configuration without changing this code pin.
TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256 = (
    "afe2291011675e936be3f5bee022e246523dad2c015348001123ffbcc46d6099"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(value))


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else pd.Timestamp(parsed).date().isoformat()


def _digest(value: Any, field: str, *, allow_empty: bool = False) -> str:
    digest = _text(value).lower()
    if allow_empty and not digest:
        return ""
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RuntimeError(f"Frozen WIKI14 price-only {field} must be SHA-256.")
    return digest


def canonical_wiki14_price_only_spec(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(SPEC_FIELDS):
        raise RuntimeError("Frozen WIKI14 price-only policy fields are not exact.")
    result = {
        "target_id": _digest(value.get("target_id"), "target_id"),
        "security_id": _text(value.get("security_id")),
        "symbol": _text(value.get("symbol")).upper(),
        "target_provider_symbol": _text(value.get("target_provider_symbol")).upper(),
        "identity_active_from": _date(value.get("identity_active_from")),
        "identity_active_to": _date(value.get("identity_active_to")),
        "terminal_event_id": _digest(
            value.get("terminal_event_id"), "terminal_event_id", allow_empty=True
        ),
        "identity_bound_provider_symbol": _text(
            value.get("identity_bound_provider_symbol")
        ),
        "extract_sha256": _digest(value.get("extract_sha256"), "extract_sha256"),
    }
    if (
        not result["security_id"]
        or not result["symbol"]
        or result["target_provider_symbol"] != result["symbol"]
        or not result["identity_active_from"]
        or not result["identity_bound_provider_symbol"]
    ):
        raise RuntimeError("Frozen WIKI14 price-only policy entry is incomplete.")
    return result


def wiki14_price_only_registry(
    prices_policy: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    values = prices_policy.get(REVIEWED_WIKI14_PRICE_ONLY_POLICY_KEY)
    if not isinstance(values, list):
        raise RuntimeError("Frozen WIKI14 price-only policy registry must be a list.")
    output: dict[str, dict[str, str]] = {}
    for raw in values:
        spec = canonical_wiki14_price_only_spec(raw)
        if spec["target_id"] in output:
            raise RuntimeError(
                "Frozen WIKI14 price-only target is duplicated: " + spec["target_id"]
            )
        output[spec["target_id"]] = spec
    return output


def wiki14_price_only_inventory_sha256(prices_policy: Mapping[str, Any]) -> str:
    registry = wiki14_price_only_registry(prices_policy)
    return _canonical_json_sha256([registry[key] for key in sorted(registry)])


def wiki14_price_only_spec_sha256(spec: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(canonical_wiki14_price_only_spec(spec))


def _safe_archive_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    path = (base / object_path).resolve()
    if path == base or base not in path.parents:
        raise RuntimeError("Frozen WIKI14 source_archive object path is unsafe.")
    return path


def _archive_payload(
    repository: Any,
    archive: pd.DataFrame,
    *,
    archive_id: str,
    expected: Mapping[str, str],
) -> bytes:
    rows = archive.loc[archive["archive_id"].map(_text).eq(archive_id)]
    if len(rows) != 1:
        raise RuntimeError(
            "Frozen WIKI14 source_archive evidence is absent or duplicated: "
            + archive_id
        )
    row = rows.iloc[0]
    common = {
        "archive_id": archive_id,
        "source_hash": archive_id,
        "source_url": WIKI14_DOWNLOAD_URL,
        "effective_date": WIKI14_ARCHIVE_EFFECTIVE_DATE,
        **dict(expected),
    }
    changed = [key for key, expected_value in common.items() if _text(row.get(key)) != expected_value]
    if changed:
        raise RuntimeError(
            "Frozen WIKI14 source_archive metadata changed: " + ", ".join(changed)
        )
    path = _safe_archive_path(Path(repository.root), _text(row.get("object_path")))
    if _text(row.get("object_path")) != common["object_path"] or not path.is_file():
        raise RuntimeError("Frozen WIKI14 archived payload path changed or is missing.")
    encoded = path.read_bytes()
    try:
        payload = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    except (OSError, EOFError) as exc:
        raise RuntimeError("Frozen WIKI14 archived payload is not valid gzip.") from exc
    if sha256_bytes(payload) != archive_id:
        raise RuntimeError("Frozen WIKI14 archived payload hash changed: " + archive_id)
    return payload


def _one(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(f"Frozen WIKI14 {label} inventory changed.")
    return rows.iloc[0]


def _raw_economics(frame: pd.DataFrame) -> list[list[str]]:
    return [
        [
            str(row.date),
            *[
                format(float(getattr(row, column)), ".17g")
                for column in ("open", "high", "low", "close", "volume")
            ],
        ]
        for row in frame.sort_values("date").itertuples(index=False)
    ]


def _economic_actions(frame: pd.DataFrame) -> list[list[Any]]:
    columns = (
        "event_id", "security_id", "action_type", "effective_date", "ex_date",
        "ratio", "cash_amount", "currency", "new_security_id", "new_symbol",
        "official", "source", "source_hash", "source_url",
    )
    output: list[list[Any]] = []
    for row in frame.sort_values(["effective_date", "event_id"]).itertuples(index=False):
        item: list[Any] = []
        for column in columns:
            value = getattr(row, column)
            if pd.isna(value):
                value = None
            elif isinstance(value, (bool, np.bool_)):
                value = bool(value)
            elif isinstance(value, (float, np.floating)):
                value = format(float(value), ".17g")
            else:
                value = str(value)
            item.append(value)
        output.append(item)
    return output


def _economic_factors(frame: pd.DataFrame) -> list[list[str]]:
    return [
        [
            _date(row.session),
            format(float(row.split_factor), ".17g"),
            format(float(row.total_return_factor), ".17g"),
        ]
        for row in frame.sort_values("session").itertuples(index=False)
    ]


def _relation_sha256(joined: pd.DataFrame) -> str:
    records: list[list[str]] = []
    for row in joined.itertuples(index=False):
        records.append(
            [str(row.date)]
            + [
                format(float(getattr(row, f"{column}_eod")), ".17g")
                for column in ("open", "high", "low", "close", "volume")
            ]
            + [
                format(float(getattr(row, f"{column}_wiki")), ".17g")
                for column in ("open", "high", "low", "close", "volume")
            ]
        )
    return _canonical_json_sha256(records)


def _action_coverage(
    wiki: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    wiki_amounts = pd.to_numeric(wiki["ex-dividend"], errors="raise")
    wiki_div = sorted(
        [str(row.date), format(float(row.amount), ".17g")]
        for row in pd.DataFrame(
            {"date": wiki["date"].astype(str), "amount": wiki_amounts}
        ).loc[wiki_amounts.gt(0)].itertuples(index=False)
    )
    current_dividend_frame = actions.loc[
        actions["action_type"].astype(str).eq("cash_dividend")
        & actions["effective_date"].astype(str).between(start, end)
    ]
    current_div = sorted(
        [str(row.effective_date), format(float(row.cash_amount), ".17g")]
        for row in current_dividend_frame.itertuples(index=False)
    )
    split_ratios = pd.to_numeric(wiki["split_ratio"], errors="raise")
    wiki_split = sorted(
        [str(row.date), format(float(row.ratio), ".17g")]
        for row in pd.DataFrame(
            {"date": wiki["date"].astype(str), "ratio": split_ratios}
        ).loc[~split_ratios.eq(1.0)].itertuples(index=False)
    )
    split_actions = actions.loc[
        actions["action_type"].astype(str).isin(
            ["split", "stock_dividend", "capital_reduction"]
        )
        & actions["effective_date"].astype(str).between(start, end)
    ]
    current_split = sorted(
        [
            str(row.effective_date),
            str(row.action_type),
            None if pd.isna(row.ratio) else format(float(row.ratio), ".17g"),
        ]
        for row in split_actions.itertuples(index=False)
    )
    return {
        "wiki_dividends": wiki_div,
        "current_dividends": current_div,
        "wiki_dividends_missing_from_current": sorted(
            value for value in wiki_div if value not in current_div
        ),
        "current_dividends_missing_from_wiki": sorted(
            value for value in current_div if value not in wiki_div
        ),
        "wiki_splits": wiki_split,
        "current_split_like_actions": current_split,
    }


def _triple_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame["session"], errors="raise"),
            "Open": pd.to_numeric(frame["open"], errors="raise"),
            "High": pd.to_numeric(frame["high"], errors="raise"),
            "Low": pd.to_numeric(frame["low"], errors="raise"),
            "Close": pd.to_numeric(frame["close"], errors="raise"),
            "Volume": pd.to_numeric(frame["volume"], errors="raise"),
        }
    )
    return add_triple_supertrend(
        prepared,
        settings=((10, 1.0), (11, 2.0), (12, 3.0)),
        atr_method="wilder",
        exit_down_count=2,
    )


def _signal_sha256(frame: pd.DataFrame) -> str:
    records: list[list[Any]] = []
    for session, values in zip(
        frame["Date"].dt.date.astype(str),
        frame[list(SIGNAL_COLUMNS)].itertuples(index=False, name=None),
        strict=True,
    ):
        records.append(
            [
                session,
                *[
                    bool(value) if isinstance(value, (bool, np.bool_)) else int(value)
                    for value in values
                ],
            ]
        )
    return _canonical_json_sha256(records)


def _recompute_audit(
    spec: Mapping[str, str],
    wiki_full: pd.DataFrame,
    archived_audit: Mapping[str, Any],
    *,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    master: pd.DataFrame,
    history: pd.DataFrame,
    actions: pd.DataFrame,
) -> dict[str, Any]:
    security_id = spec["security_id"]
    symbol = spec["symbol"]
    master_row = _one(
        master,
        master["security_id"].map(_text).eq(security_id),
        symbol + " security_master",
    )
    history_row = _one(
        history,
        history["security_id"].map(_text).eq(security_id)
        & history["symbol"].map(_text).str.upper().eq(symbol),
        symbol + " symbol_history",
    )
    identity = {
        "security_id": security_id,
        "symbol": symbol,
        "provider_symbol": _text(master_row.get("provider_symbol")),
        "active_from": _date(master_row.get("active_from")),
        "active_to": _date(master_row.get("active_to")),
        "master_source": _text(master_row.get("source")),
        "master_source_hash": _text(master_row.get("source_hash")),
        "history_effective_from": _date(history_row.get("effective_from")),
        "history_effective_to": _date(history_row.get("effective_to")),
        "history_source": _text(history_row.get("source")),
        "history_source_hash": _text(history_row.get("source_hash")),
    }
    expected_identity = archived_audit.get("identity")
    if (
        not isinstance(expected_identity, Mapping)
        or identity != dict(expected_identity)
        or identity["provider_symbol"] != spec["identity_bound_provider_symbol"]
        or identity["history_effective_from"] != spec["identity_active_from"]
        or identity["history_effective_to"] != spec["identity_active_to"]
        or _canonical_json_sha256(identity) != _text(archived_audit.get("identity_sha256"))
    ):
        raise RuntimeError("Frozen WIKI14 exact identity interval changed: " + symbol)

    expected_schema = archived_audit.get("identity_schema")
    identity_schema = {
        "master_primary_symbol": _text(master_row.get("primary_symbol")).upper(),
        "master_exchange": _text(master_row.get("exchange")).upper(),
        "master_asset_type": _text(master_row.get("asset_type")).upper(),
        "master_currency": _text(master_row.get("currency")).upper(),
        "master_country": _text(master_row.get("country")).upper(),
        "history_symbol": _text(history_row.get("symbol")).upper(),
        "history_exchange": _text(history_row.get("exchange")).upper(),
        "raw_price_currency": "USD",
    }
    if (
        not isinstance(expected_schema, Mapping)
        or identity_schema != dict(expected_schema)
        or archived_audit.get("identity_schema_inventory_sha256")
        != WIKI14_IDENTITY_SCHEMA_INVENTORY_SHA256
    ):
        raise RuntimeError("Frozen WIKI14 identity schema changed: " + symbol)

    target_prices = prices.loc[prices["security_id"].map(_text).eq(security_id)].copy()
    target_prices["date"] = pd.to_datetime(
        target_prices["session"], errors="raise"
    ).dt.date.astype(str)
    if target_prices.empty or target_prices["date"].duplicated().any():
        raise RuntimeError("Frozen WIKI14 raw price inventory changed: " + symbol)
    currencies = {_text(value).upper() for value in target_prices["currency"]}
    if currencies != {"USD"}:
        raise RuntimeError("Frozen WIKI14 raw price currency changed: " + symbol)
    if (
        tuple(sorted(target_prices["source_hash"].map(_text).unique()))
        != tuple(archived_audit.get("raw_price_source_sha256s") or ())
        or _canonical_json_sha256(_raw_economics(target_prices))
        != _text(archived_audit.get("raw_economics_sha256"))
    ):
        raise RuntimeError("Frozen WIKI14 raw price economics changed: " + symbol)

    relation = archived_audit.get("reviewed_relation")
    if not isinstance(relation, Mapping):
        raise RuntimeError("Frozen WIKI14 reviewed relation is missing: " + symbol)
    start = _date(relation.get("start"))
    end = _date(relation.get("end"))
    wiki = wiki_full.loc[wiki_full["date"].astype(str).between(start, end)].copy()
    overlap = target_prices.loc[target_prices["date"].between(start, end)].copy()
    joined = overlap.merge(
        wiki,
        on="date",
        suffixes=("_eod", "_wiki"),
        validate="one_to_one",
    ).sort_values("date", ignore_index=True)
    if (
        len(wiki) != int(relation.get("session_count", -1))
        or len(joined) != len(wiki)
        or _relation_sha256(joined) != _text(relation.get("relation_sha256"))
    ):
        raise RuntimeError("Frozen WIKI14 exact price relation changed: " + symbol)

    target_actions = actions.loc[actions["security_id"].map(_text).eq(security_id)].copy()
    terminal = target_actions.loc[
        target_actions["action_type"].map(_text).isin(
            ["cash_merger", "stock_merger", "ticker_change", "delisting"]
        )
    ]
    terminal_event_id = spec["terminal_event_id"]
    if terminal_event_id:
        terminal_row = _one(
            terminal,
            terminal["event_id"].map(_text).eq(terminal_event_id),
            symbol + " terminal action",
        )
        if _text(terminal_row.get("source_hash")) != _text(
            archived_audit.get("terminal_source_sha256")
        ):
            raise RuntimeError("Frozen WIKI14 terminal source changed: " + symbol)
    elif len(terminal):
        raise RuntimeError("Frozen WIKI14 identity gained a terminal action: " + symbol)
    coverage = _action_coverage(wiki, target_actions, start=start, end=end)
    expected_coverage = archived_audit.get("action_coverage")
    coverage_sha256 = _canonical_json_sha256(coverage)
    if (
        not isinstance(expected_coverage, Mapping)
        or coverage_sha256 != _text(expected_coverage.get("coverage_sha256"))
        or _canonical_json_sha256(_economic_actions(target_actions))
        != _text(expected_coverage.get("all_actions_sha256"))
    ):
        raise RuntimeError("Frozen WIKI14 action-gap fingerprint changed: " + symbol)
    action_coverage = {
        **coverage,
        "coverage_sha256": coverage_sha256,
        "all_actions_sha256": _canonical_json_sha256(
            _economic_actions(target_actions)
        ),
        "actions_rewritten": False,
        "price_only_pass_must_not_imply_action_pass": True,
    }

    target_factors = factors.loc[factors["security_id"].map(_text).eq(security_id)].copy()
    factor_sessions = set(pd.to_datetime(target_factors["session"]).dt.date.astype(str))
    expected_factors = archived_audit.get("factor_coverage")
    factor_hash = _canonical_json_sha256(_economic_factors(target_factors))
    if (
        not isinstance(expected_factors, Mapping)
        or len(target_factors) != int(expected_factors.get("row_count", -1))
        or factor_sessions != set(target_prices["date"])
        or factor_hash != _text(expected_factors.get("economics_sha256"))
    ):
        raise RuntimeError("Frozen WIKI14 factor economics changed: " + symbol)

    candidate = target_prices.drop(columns="date").copy()
    candidate["_date"] = pd.to_datetime(
        candidate["session"], errors="raise"
    ).dt.date.astype(str)
    by_date = wiki.set_index("date")
    replace = candidate["_date"].isin(by_date.index)
    for column in ("open", "high", "low", "close", "volume"):
        candidate.loc[replace, column] = candidate.loc[replace, "_date"].map(
            pd.to_numeric(by_date[column], errors="raise")
        )
    candidate = candidate.drop(columns="_date")
    current_adjusted = apply_adjustment_factors(
        target_prices.drop(columns="date"),
        target_factors,
        mode="total_return_adjusted",
    ).sort_values("session", ignore_index=True)
    substituted_adjusted = apply_adjustment_factors(
        candidate,
        target_factors,
        mode="total_return_adjusted",
    ).sort_values("session", ignore_index=True)
    current_signal = _triple_signal_frame(current_adjusted)
    substituted_signal = _triple_signal_frame(substituted_adjusted)
    differences = {
        column: int((~current_signal[column].eq(substituted_signal[column])).sum())
        for column in SIGNAL_COLUMNS
    }
    triple_supertrend = {
        "current_signal_sha256": _signal_sha256(current_signal),
        "substituted_signal_sha256": _signal_sha256(substituted_signal),
        "field_differences": differences,
    }
    if any(differences.values()) or triple_supertrend != archived_audit.get(
        "triple_supertrend"
    ):
        raise RuntimeError(
            "Frozen WIKI14 substitution changed Triple Supertrend: " + symbol
        )

    computed = {
        "status": "passed_price_only_arbitration",
        "target_id": spec["target_id"],
        "symbol": symbol,
        "security_id": security_id,
        "identity": identity,
        "identity_sha256": _text(archived_audit.get("identity_sha256")),
        "identity_schema": identity_schema,
        "identity_schema_inventory_sha256": WIKI14_IDENTITY_SCHEMA_INVENTORY_SHA256,
        "terminal_event_id": terminal_event_id,
        "terminal_source_sha256": _text(archived_audit.get("terminal_source_sha256")),
        "raw_price_source_sha256s": list(
            archived_audit.get("raw_price_source_sha256s") or ()
        ),
        "raw_economics_sha256": _text(archived_audit.get("raw_economics_sha256")),
        "reviewed_relation": dict(relation),
        "action_coverage": action_coverage,
        "factor_coverage": {
            "status": "current_economics_pinned_not_independent_action_factor_pass",
            "row_count": len(target_factors),
            "economics_sha256": factor_hash,
            "factors_rewritten": False,
            "price_only_pass_must_not_imply_factor_pass": True,
        },
        "triple_supertrend": triple_supertrend,
        "raw_price_rewritten": False,
        "corporate_actions_rewritten": False,
        "adjustment_factors_rewritten": False,
        "identity_rewritten": False,
    }
    if computed != dict(archived_audit):
        raise RuntimeError("Frozen WIKI14 reviewed audit changed: " + symbol)
    return computed


def verify_wiki14_price_only_evidence(
    repository: Any,
    archive: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    master: pd.DataFrame,
    history: pd.DataFrame,
    actions: pd.DataFrame,
    targets: Mapping[str, Mapping[str, Any]],
    prices_policy: Mapping[str, Any],
    release_warnings: tuple[str, ...] | list[str] = (),
) -> dict[str, dict[str, Any]]:
    """Recompute the exact 14 reviewed price-only cases from local inputs."""

    registry = wiki14_price_only_registry(prices_policy)
    inventory_hash = wiki14_price_only_inventory_sha256(prices_policy)
    if (
        set(registry) != set(TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS)
        or inventory_hash != TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256
        or set(targets) != set(TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS)
    ):
        raise RuntimeError("Frozen WIKI14 price-only inventory is not code-pinned.")
    if WIKI14_LICENSE_WARNING not in set(release_warnings):
        raise RuntimeError("Frozen WIKI14 private/internal-only warning is missing.")

    provenance_payload = _archive_payload(
        repository,
        archive,
        archive_id=WIKI14_PROVENANCE_SHA256,
        expected={
            "dataset": "reviewed_us_wiki14_price_only_arbitration",
            "source": "reviewed_us_wiki14_price_only_arbitration",
            "content_type": "application/json",
            "retrieved_at": WIKI14_PROVENANCE_RETRIEVED_AT,
            "object_path": (
                f"archives/{WIKI14_ARCHIVE_EFFECTIVE_DATE}/"
                f"{WIKI14_PROVENANCE_SHA256}.json.gz"
            ),
        },
    )
    try:
        provenance = json.loads(provenance_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Frozen WIKI14 provenance is not valid JSON.") from exc
    if provenance_payload != _canonical_json_bytes(provenance):
        raise RuntimeError("Frozen WIKI14 provenance is not canonical JSON.")

    scope = provenance.get("scope") if isinstance(provenance, Mapping) else None
    license_policy = (
        provenance.get("license_policy") if isinstance(provenance, Mapping) else None
    )
    frozen = (
        provenance.get("frozen_evidence") if isinstance(provenance, Mapping) else None
    )
    audits = (
        provenance.get("price_arbitrations")
        if isinstance(provenance, Mapping)
        else None
    )
    ordered_security_ids = [registry[key]["security_id"] for key in registry]
    if (
        provenance.get("schema") != "us_wiki14_price_only_arbitration/v1"
        or not isinstance(scope, Mapping)
        or scope.get("passed_price_only_security_ids") != ordered_security_ids
        or scope.get("write_dataset") != "source_archive"
        or scope.get("non_write_datasets")
        != [
            "daily_price_raw",
            "corporate_actions",
            "adjustment_factors",
            "security_master",
            "symbol_history",
            "index_constituent_anchors",
            "index_membership_events",
            "lifecycle_resolutions",
        ]
        or scope.get("generic_symbol_or_ticker_exception_allowed") is not False
        or scope.get("identity_schema_inventory_sha256")
        != WIKI14_IDENTITY_SCHEMA_INVENTORY_SHA256
        or scope.get("archive_effective_date") != WIKI14_ARCHIVE_EFFECTIVE_DATE
        or not isinstance(license_policy, Mapping)
        or license_policy.get("formal_license_name") != "Unknown"
        or license_policy.get("allowed_scope") != "private_internal_only"
        or license_policy.get("redistribution_allowed") is not False
        or license_policy.get("public_publication_allowed") is not False
        or license_policy.get("local_apply_ack_required") is not True
        or license_policy.get("private_r2_publisher_ack_required_separately") is not True
        or license_policy.get("fail_closed") is not True
        or not isinstance(frozen, Mapping)
        or frozen.get("extract_inventory_sha256")
        != WIKI14_EXTRACT_INVENTORY_SHA256
        or not isinstance(audits, list)
        or len(audits) != 14
    ):
        raise RuntimeError("Frozen WIKI14 provenance scope/license changed.")

    extracts = frozen.get("extracts")
    if not isinstance(extracts, list) or len(extracts) != 14:
        raise RuntimeError("Frozen WIKI14 extract inventory changed.")
    extract_by_symbol = {
        _text(item.get("symbol")).upper(): item
        for item in extracts
        if isinstance(item, Mapping)
    }
    audit_by_target = {
        _text(item.get("target_id")): item
        for item in audits
        if isinstance(item, Mapping)
    }
    if (
        len(extract_by_symbol) != 14
        or set(audit_by_target) != set(TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS)
    ):
        raise RuntimeError("Frozen WIKI14 reviewed target inventory changed.")

    artifact_hashes: list[str] = []
    output: dict[str, dict[str, Any]] = {}
    for target_id, spec in registry.items():
        target = targets[target_id]
        expected_target = {
            "target_id": target_id,
            "security_id": spec["security_id"],
            "symbol": spec["symbol"],
            "provider_symbol": spec["target_provider_symbol"],
            "active_from": spec["identity_active_from"],
            "active_to": spec["identity_active_to"],
            "terminal_event_id": spec["terminal_event_id"],
        }
        if any(_text(target.get(key)) != value for key, value in expected_target.items()):
            raise RuntimeError(
                "Frozen WIKI14 target identity/provider interval changed: " + target_id
            )
        archived_extract = extract_by_symbol.get(spec["symbol"])
        archived_audit = audit_by_target[target_id]
        if (
            not isinstance(archived_extract, Mapping)
            or _text(archived_extract.get("security_id")) != spec["security_id"]
            or _text(archived_extract.get("extract_sha256"))
            != spec["extract_sha256"]
            or _text(archived_audit.get("security_id")) != spec["security_id"]
            or _text(archived_audit.get("symbol")).upper() != spec["symbol"]
            or _text(archived_audit.get("terminal_event_id"))
            != spec["terminal_event_id"]
        ):
            raise RuntimeError("Frozen WIKI14 policy/provenance binding changed: " + target_id)

        symbol_lower = spec["symbol"].lower()
        extract_hash = spec["extract_sha256"]
        extract_payload = _archive_payload(
            repository,
            archive,
            archive_id=extract_hash,
            expected={
                "dataset": f"kaggle_quandl_wiki_{symbol_lower}_full_price_extract",
                "source": f"kaggle_quandl_wiki_{symbol_lower}_full_price_extract",
                "content_type": "text/csv",
                "retrieved_at": WIKI14_EXTRACT_RETRIEVED_AT,
                "object_path": (
                    f"archives/{WIKI14_ARCHIVE_EFFECTIVE_DATE}/{extract_hash}.csv.gz"
                ),
            },
        )
        if len(extract_payload) != int(archived_extract.get("extract_size", -1)):
            raise RuntimeError("Frozen WIKI14 extract size changed: " + spec["symbol"])
        wiki = pd.read_csv(io.BytesIO(extract_payload))
        if (
            tuple(wiki.columns) != WIKI_COLUMNS
            or len(wiki) != int(archived_extract.get("full_rows", -1))
            or set(wiki["ticker"].map(_text)) != {spec["symbol"]}
            or wiki["date"].map(_date).duplicated().any()
        ):
            raise RuntimeError("Frozen WIKI14 extract content changed: " + spec["symbol"])
        wiki["date"] = wiki["date"].map(_date)
        for column in ("open", "high", "low", "close", "volume"):
            values = pd.to_numeric(wiki[column], errors="raise")
            if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
                raise RuntimeError("Frozen WIKI14 price is not finite: " + spec["symbol"])

        audit = _recompute_audit(
            spec,
            wiki,
            archived_audit,
            prices=prices,
            factors=factors,
            master=master,
            history=history,
            actions=actions,
        )
        diagnostic = {
            "validation_basis": REVIEWED_WIKI14_PRICE_ONLY_BASIS,
            "policy_spec_sha256": wiki14_price_only_spec_sha256(spec),
            "policy_registry_sha256": TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256,
            "provenance_sha256": WIKI14_PROVENANCE_SHA256,
            "extract_sha256": extract_hash,
            "target_id": target_id,
            "security_id": spec["security_id"],
            "symbol": spec["symbol"],
            "target_provider_symbol": spec["target_provider_symbol"],
            "identity_bound_provider_symbol": spec["identity_bound_provider_symbol"],
            "identity_active_from": spec["identity_active_from"],
            "identity_active_to": spec["identity_active_to"],
            "terminal_event_id": spec["terminal_event_id"],
            "archived_audit_sha256": _canonical_json_sha256(audit),
            "overlap_session_count": int(audit["reviewed_relation"]["session_count"]),
            "overlap_start": _date(audit["reviewed_relation"]["start"]),
            "overlap_end": _date(audit["reviewed_relation"]["end"]),
            "relation_sha256": _text(audit["reviewed_relation"]["relation_sha256"]),
            "triple_supertrend": dict(audit["triple_supertrend"]),
            "action_coverage_sha256": _text(audit["action_coverage"]["coverage_sha256"]),
            "factor_economics_sha256": _text(audit["factor_coverage"]["economics_sha256"]),
            "action_factor_status": "incomplete_not_rewritten",
            "raw_price_rewritten": False,
            "corporate_actions_rewritten": False,
            "adjustment_factors_rewritten": False,
            "generic_ticker_reuse_allowed": False,
            "yahoo_symbol_only_identity_reuse_allowed": False,
            "price_only_pass_must_not_imply_action_factor_pass": True,
            "private_internal_only": True,
            "redistribution_allowed": False,
            "public_publication_allowed": False,
            "limitation": WIKI14_LIMITATION,
        }
        output[target_id] = {
            **diagnostic,
            "projection_sha256": _canonical_json_sha256(diagnostic),
        }
        artifact_hashes.append(extract_hash)

    artifact_hashes.append(WIKI14_PROVENANCE_SHA256)
    if _canonical_json_sha256(artifact_hashes) != WIKI14_ARCHIVE_ARTIFACT_INVENTORY_SHA256:
        raise RuntimeError("Frozen WIKI14 archive artifact inventory changed.")
    return output
