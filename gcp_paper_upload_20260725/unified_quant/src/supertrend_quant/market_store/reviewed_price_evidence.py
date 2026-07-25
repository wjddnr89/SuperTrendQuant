"""Exact, code-pinned review path for narrow Yahoo price quirks.

This module is deliberately not a second permissive Yahoo parser.  A payload
may enter this path only after its target, immutable response bytes, cache
wrapper, observed metadata and complete local comparison projection have been
reviewed and pinned.  The same functions are used by the offline collector and
the publication gate so a re-written report cannot bless changed bytes.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any, Iterable, Mapping

import pandas as pd

from ..indicators import add_triple_supertrend
from .manifest import sha256_bytes
from .yahoo_chart import normalize_yahoo_symbol


REVIEWED_PRICE_EVIDENCE_BASIS = "reviewed_exact_price_evidence/v1"
REVIEWED_PRICE_CASE_CODES = frozenset(
    {
        "tiny_ohlc_disagreement",
        "retired_yhd_metadata_limitation",
        "single_all_null_quote_row",
        "otcid_exchange_code",
    }
)
# The YAML is not authority by itself.  Publication requires this complete
# reviewed target inventory and its normalized hash to match code as well.
TRUSTED_REVIEWED_PRICE_EVIDENCE_TARGET_IDS = frozenset(
    {
        "084ea79dbd2d403f570e8aecd780367e8e714caa408935712d7b01fda7c2b485",
        "1c35c20c1e50ebd119618a823b4cc49ef55aa701ec24d2a4564995022138bb3f",
        "2b025b50241eff6b6d3fa48f9072b04fd8297605710302562157d08f7d36a4b8",
        "3893c9be614824bdd7d6dea415ed25981ff28946969dfe55d94edc54970e1ea3",
        "3922fe53e0433bb1ce8cdeccef6de837c963fee64e92c26c4244832c6ff9a761",
        "417f2715f9774e20fa90f071b452f7c40d2fd3aaa5e5b0c3a0debef8dc0b212e",
        "4f99ccb72c52d48c5cef660ca6a088b0acc8275962ad53e5f1bfb9ea38301ef9",
        "63f956662bcd5fe8caac688ffd77f0ebb0af0ed349321836816967ba0d7988ee",
        "69a9990fab1dc3027ca79bd5a584768b6e2bd3d8cff83b298311ed2cdab62113",
        "7017c9bf4a030b9f3d08b71e7591c3f27abc5f79cfa083e3bf0469c0695f8082",
        "8da911e511165e8a8731e1a9878eaa873c931b5102f0917fd603d521f36f5eea",
        "b25aa525bb3fcf5b791083b1d8a694fde94e4a72ea5a12e0f4e5008571bb58f6",
        "b4ca2124443b0b05a5ce3fbacfcc5fc1b45dab595e5812273bd5af2ad09a1478",
        "d5a785bc99e55a74b2595bc8c3ca1109c321b0a9c2fc2d760a725509a25e8650",
        "d76b90408144482b2a2156b184b6a7af16926b3a0b837bc5685e5702f033642b",
        "e3c6bde376922ea4c3ffdf3d7f753e21ca1c025ab8e1e4d17ee3d03aea65ab25",
        "fc949d30859345b8e883d280a2b7a33180abae54b53baf072c9ade86d1436fac",
    }
)
TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256 = (
    "fc449298c9b7b798e7b56126264788b6aaaa4e837c9b3006f5e32a043fce74b1"
)
REVIEWED_PRICE_EVIDENCE_FIELDS = (
    "target_id",
    "security_id",
    "symbol",
    "identity_active_from",
    "identity_active_to",
    "case_code",
    "source_sha256",
    "cache_wrapper_sha256",
    "observed_currency",
    "observed_instrument_type",
    "observed_exchange_name",
    "observed_exchange_timezone",
    "observed_data_granularity",
    "official_event_id",
    "official_evidence_sha256",
    "official_effective_date",
    "expected_raw_row_count",
    "expected_all_null_row_count",
    "expected_all_null_sessions_sha256",
    "allowed_invalid_rows",
    "expected_mismatch_rows",
    "expected_projection_sha256",
    "limitation",
)
TRIPLE_SUPERTREND_SETTINGS = ((10, 1.0), (11, 2.0), (12, 3.0))
TRIPLE_SUPERTREND_ATR_METHOD = "wilder"
TRIPLE_SUPERTREND_EXIT_DOWN_COUNT = 2


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def canonical_json_sha256(value: Any) -> str:
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
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).date().isoformat()


def _digest(value: Any, field: str, *, allow_empty: bool = False) -> str:
    digest = _text(value).lower()
    if allow_empty and not digest:
        return ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"Reviewed price evidence {field} must be SHA-256.")
    return digest


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"Reviewed price evidence {field} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Reviewed price evidence {field} must be an integer."
        ) from exc
    if parsed != value:
        raise RuntimeError(f"Reviewed price evidence {field} must be exact.")
    return parsed


def _number_text(value: Any) -> str:
    if isinstance(value, bool):
        raise RuntimeError("Boolean is not a reviewed price number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Reviewed price number is invalid.") from exc
    if not math.isfinite(number):
        raise RuntimeError("Reviewed price number is not finite.")
    return format(number, ".17g")


def _canonical_invalid_row(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "session",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }:
        raise RuntimeError("Reviewed invalid Yahoo row fields are not exact.")
    session = _date(value.get("session"))
    if not session or _text(value.get("session")) != session:
        raise RuntimeError("Reviewed invalid Yahoo row session is invalid.")
    return {
        "session": session,
        **{
            field: _number_text(value.get(field))
            for field in ("open", "high", "low", "close", "volume")
        },
    }


def _canonical_mismatch_row(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "session",
        "field",
        "internal",
        "provider",
        "normalized_provider",
        "median_scale",
    }:
        raise RuntimeError("Reviewed price mismatch row fields are not exact.")
    session = _date(value.get("session"))
    field = _text(value.get("field")).lower()
    if not session or _text(value.get("session")) != session or field not in {
        "open",
        "high",
        "low",
        "close",
    }:
        raise RuntimeError("Reviewed price mismatch identity is invalid.")
    return {
        "session": session,
        "field": field,
        **{
            key: _number_text(value.get(key))
            for key in (
                "internal",
                "provider",
                "normalized_provider",
                "median_scale",
            )
        },
    }


def canonical_reviewed_price_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one exact review record and reject schema drift."""

    if not isinstance(value, Mapping) or set(value) != set(
        REVIEWED_PRICE_EVIDENCE_FIELDS
    ):
        raise RuntimeError("Reviewed price evidence fields are not exact.")
    case_code = _text(value.get("case_code"))
    if case_code not in REVIEWED_PRICE_CASE_CODES:
        raise RuntimeError("Reviewed price evidence case code is not allowed.")
    symbol = normalize_yahoo_symbol(_text(value.get("symbol")))
    active_from = _date(value.get("identity_active_from"))
    active_to = _date(value.get("identity_active_to"))
    if not active_from or _text(value.get("identity_active_from")) != active_from:
        raise RuntimeError("Reviewed price evidence active_from is not exact.")
    if active_to and _text(value.get("identity_active_to")) != active_to:
        raise RuntimeError("Reviewed price evidence active_to is not exact.")
    official_effective_date = _date(value.get("official_effective_date"))
    if official_effective_date and _text(value.get("official_effective_date")) != official_effective_date:
        raise RuntimeError("Reviewed price evidence official date is not exact.")
    invalid_rows = [
        _canonical_invalid_row(item) for item in value.get("allowed_invalid_rows", ())
    ]
    mismatch_rows = [
        _canonical_mismatch_row(item) for item in value.get("expected_mismatch_rows", ())
    ]
    if len(invalid_rows) != len({item["session"] for item in invalid_rows}):
        raise RuntimeError("Reviewed invalid Yahoo rows are duplicated.")
    if len(mismatch_rows) != len(
        {(item["session"], item["field"]) for item in mismatch_rows}
    ):
        raise RuntimeError("Reviewed price mismatches are duplicated.")
    output = {
        "target_id": _digest(value.get("target_id"), "target_id"),
        "security_id": _text(value.get("security_id")),
        "symbol": symbol,
        "identity_active_from": active_from,
        "identity_active_to": active_to,
        "case_code": case_code,
        "source_sha256": _digest(value.get("source_sha256"), "source_sha256"),
        "cache_wrapper_sha256": _digest(
            value.get("cache_wrapper_sha256"), "cache_wrapper_sha256"
        ),
        "observed_currency": _text(value.get("observed_currency")).upper(),
        "observed_instrument_type": _text(
            value.get("observed_instrument_type")
        ).upper(),
        "observed_exchange_name": _text(value.get("observed_exchange_name")).upper(),
        "observed_exchange_timezone": _text(
            value.get("observed_exchange_timezone")
        ),
        "observed_data_granularity": _text(
            value.get("observed_data_granularity")
        ),
        "official_event_id": _digest(
            value.get("official_event_id"), "official_event_id", allow_empty=True
        ),
        "official_evidence_sha256": _digest(
            value.get("official_evidence_sha256"),
            "official_evidence_sha256",
            allow_empty=True,
        ),
        "official_effective_date": official_effective_date,
        "expected_raw_row_count": _integer(
            value.get("expected_raw_row_count"), "expected_raw_row_count"
        ),
        "expected_all_null_row_count": _integer(
            value.get("expected_all_null_row_count"),
            "expected_all_null_row_count",
        ),
        "expected_all_null_sessions_sha256": _digest(
            value.get("expected_all_null_sessions_sha256"),
            "expected_all_null_sessions_sha256",
        ),
        "allowed_invalid_rows": invalid_rows,
        "expected_mismatch_rows": mismatch_rows,
        "expected_projection_sha256": _digest(
            value.get("expected_projection_sha256"),
            "expected_projection_sha256",
        ),
        "limitation": _text(value.get("limitation")),
    }
    if not output["security_id"] or not output["limitation"]:
        raise RuntimeError("Reviewed price evidence identity/limitation is missing.")
    event_values = (
        output["official_event_id"],
        output["official_evidence_sha256"],
        output["official_effective_date"],
    )
    if any(event_values) != all(event_values):
        raise RuntimeError("Reviewed price official event binding is incomplete.")
    if case_code == "tiny_ohlc_disagreement":
        if (
            output["observed_currency"] != "USD"
            or output["observed_instrument_type"] != "EQUITY"
            or not mismatch_rows
            or invalid_rows
            or output["expected_all_null_row_count"] != 0
            or any(event_values)
        ):
            raise RuntimeError("Tiny OHLC review contract is invalid.")
    elif case_code == "retired_yhd_metadata_limitation":
        if (
            output["observed_currency"]
            or output["observed_instrument_type"] != "MUTUALFUND"
            or output["observed_exchange_name"] != "YHD"
            or (not all(event_values) and not output["identity_active_to"])
        ):
            raise RuntimeError("Retired YHD review contract is invalid.")
    elif case_code == "single_all_null_quote_row":
        if (
            output["observed_currency"] != "USD"
            or output["observed_instrument_type"] != "EQUITY"
            or output["expected_all_null_row_count"] != 1
            or mismatch_rows
            or invalid_rows
            or any(event_values)
        ):
            raise RuntimeError("Single-null-row review contract is invalid.")
    elif case_code == "otcid_exchange_code":
        if (
            output["observed_currency"] != "USD"
            or output["observed_instrument_type"] != "EQUITY"
            or output["observed_exchange_name"] != "OID"
            or output["expected_all_null_row_count"] != 0
            or mismatch_rows
            or invalid_rows
            or not all(event_values)
        ):
            raise RuntimeError("OTCID exchange-code review contract is invalid.")
    return output


def reviewed_price_evidence_registry(
    prices_policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = prices_policy.get("reviewed_price_evidence")
    if not isinstance(raw, list):
        raise RuntimeError("Policy reviewed_price_evidence must be a list.")
    output: dict[str, dict[str, Any]] = {}
    for item in raw:
        normalized = canonical_reviewed_price_evidence(item)
        target_id = normalized["target_id"]
        if target_id in output:
            raise RuntimeError("Reviewed price target_id is duplicated.")
        output[target_id] = normalized
    return output


def reviewed_price_evidence_inventory_sha256(
    prices_policy: Mapping[str, Any],
) -> str:
    raw = prices_policy.get("reviewed_price_evidence")
    if not isinstance(raw, list):
        raise RuntimeError("Policy reviewed_price_evidence must be a list.")
    return canonical_json_sha256(
        [canonical_reviewed_price_evidence(item) for item in raw]
    )


def reviewed_price_evidence_sha256(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256(canonical_reviewed_price_evidence(value))


def _parse_payload(
    content: bytes,
    expected_symbol: str,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Parse one exact payload while retaining only explicitly reviewed quirks."""

    expected = canonical_reviewed_price_evidence(spec)
    if sha256_bytes(content) != expected["source_sha256"]:
        raise RuntimeError("Reviewed Yahoo response bytes changed.")
    if content.lstrip().startswith((b"<", b"<!")):
        raise RuntimeError("Reviewed Yahoo response is HTML.")
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Reviewed Yahoo response is invalid JSON.") from exc
    chart = payload.get("chart") if isinstance(payload, dict) else None
    result = chart.get("result") if isinstance(chart, dict) else None
    if (
        not isinstance(chart, dict)
        or chart.get("error") is not None
        or not isinstance(result, list)
        or len(result) != 1
        or not isinstance(result[0], dict)
    ):
        raise RuntimeError("Reviewed Yahoo response shape changed.")
    item = result[0]
    meta = item.get("meta")
    if not isinstance(meta, dict):
        raise RuntimeError("Reviewed Yahoo metadata is missing.")
    observed = {
        "symbol": normalize_yahoo_symbol(_text(meta.get("symbol"))),
        "currency": _text(meta.get("currency")).upper(),
        "instrument_type": _text(meta.get("instrumentType")).upper(),
        "exchange_name": _text(meta.get("exchangeName")).upper(),
        "exchange_timezone": _text(meta.get("exchangeTimezoneName")),
        "data_granularity": _text(meta.get("dataGranularity")),
    }
    wanted = {
        "symbol": normalize_yahoo_symbol(expected_symbol),
        "currency": expected["observed_currency"],
        "instrument_type": expected["observed_instrument_type"],
        "exchange_name": expected["observed_exchange_name"],
        "exchange_timezone": expected["observed_exchange_timezone"],
        "data_granularity": expected["observed_data_granularity"],
    }
    if observed != wanted:
        raise RuntimeError("Reviewed Yahoo metadata changed.")
    timestamps = item.get("timestamp")
    indicators = item.get("indicators")
    quotes = indicators.get("quote") if isinstance(indicators, dict) else None
    if (
        not isinstance(timestamps, list)
        or not isinstance(quotes, list)
        or len(quotes) != 1
        or not isinstance(quotes[0], dict)
    ):
        raise RuntimeError("Reviewed Yahoo quote arrays are missing.")
    quote = quotes[0]
    fields = ("open", "high", "low", "close", "volume")
    if len(timestamps) != expected["expected_raw_row_count"] or any(
        not isinstance(quote.get(field), list)
        or len(quote[field]) != len(timestamps)
        for field in fields
    ):
        raise RuntimeError("Reviewed Yahoo quote inventory changed.")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in timestamps
    ):
        raise RuntimeError("Reviewed Yahoo timestamps changed type.")
    sessions = pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce")
    if sessions.isna().any():
        raise RuntimeError("Reviewed Yahoo timestamps are invalid.")
    sessions = sessions.tz_convert("America/New_York").normalize().tz_localize(None)
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise RuntimeError("Reviewed Yahoo sessions are not unique and ordered.")

    allowed_invalid = {
        item["session"]: item for item in expected["allowed_invalid_rows"]
    }
    seen_invalid: set[str] = set()
    all_null_sessions: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, session in enumerate(sessions):
        session_text = session.date().isoformat()
        raw_values = {field: quote[field][index] for field in fields}
        if all(value is None for value in raw_values.values()):
            all_null_sessions.append(session_text)
            continue
        try:
            values = {field: float(raw_values[field]) for field in fields}
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Reviewed Yahoo {session_text} has a partial/non-numeric quote."
            ) from exc
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError("Reviewed Yahoo quote is not finite.")
        coherent = bool(
            values["open"] > 0
            and values["high"] > 0
            and values["low"] > 0
            and values["close"] > 0
            and values["high"] >= max(values["open"], values["close"])
            and values["low"] <= min(values["open"], values["close"])
            and values["high"] >= values["low"]
            and values["volume"] >= 0
        )
        actual_invalid = {
            "session": session_text,
            **{field: _number_text(values[field]) for field in fields},
        }
        if not coherent:
            if allowed_invalid.get(session_text) != actual_invalid:
                raise RuntimeError("Unreviewed invalid Yahoo OHLCV row encountered.")
            seen_invalid.add(session_text)
        elif session_text in allowed_invalid:
            raise RuntimeError("Reviewed invalid Yahoo row is no longer invalid.")
        rows.append({"session": session, **values})
    if set(allowed_invalid) != seen_invalid:
        raise RuntimeError("Reviewed invalid Yahoo row inventory changed.")
    if (
        len(all_null_sessions) != expected["expected_all_null_row_count"]
        or canonical_json_sha256(all_null_sessions)
        != expected["expected_all_null_sessions_sha256"]
    ):
        raise RuntimeError("Reviewed all-null Yahoo session inventory changed.")
    return pd.DataFrame(rows, columns=("session", *fields)), observed, all_null_sessions


def _frame_projection(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in frame.loc[:, list(columns)].to_dict(orient="records"):
        projected: dict[str, str] = {}
        for key, value in row.items():
            projected[key] = _date(value) if key == "session" else _number_text(value)
        output.append(projected)
    return output


def build_reviewed_price_projection(
    *,
    content: bytes,
    spec: Mapping[str, Any],
    target: Mapping[str, Any],
    internal_prices: pd.DataFrame,
    split_dates: Iterable[str],
    policy_prices: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute the complete exact numeric and signal review projection."""

    expected = canonical_reviewed_price_evidence(spec)
    target_projection = {
        "target_id": _text(target.get("target_id")).lower(),
        "security_id": _text(target.get("security_id")),
        "symbol": normalize_yahoo_symbol(_text(target.get("symbol"))),
        "identity_active_from": _date(target.get("active_from")),
        "identity_active_to": _date(target.get("active_to")),
    }
    if target_projection != {
        key: expected[key]
        for key in (
            "target_id",
            "security_id",
            "symbol",
            "identity_active_from",
            "identity_active_to",
        )
    }:
        raise RuntimeError("Reviewed price target identity changed.")
    provider, metadata, all_null_sessions = _parse_payload(
        content, expected["symbol"], expected
    )
    internal = internal_prices.copy()
    internal["session"] = pd.to_datetime(
        internal["session"], errors="coerce"
    ).dt.normalize()
    if internal["session"].isna().any():
        raise RuntimeError("Reviewed internal sessions are invalid.")
    for field in ("open", "high", "low", "close", "volume"):
        internal[field] = pd.to_numeric(internal[field], errors="coerce")
    if internal[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise RuntimeError("Reviewed internal OHLCV contains nulls.")
    active_from = expected["identity_active_from"]
    active_to = expected["identity_active_to"]
    internal = internal.loc[
        internal["session"].ge(pd.Timestamp(active_from))
    ].copy()
    provider = provider.loc[provider["session"].ge(pd.Timestamp(active_from))].copy()
    if active_to:
        internal = internal.loc[
            internal["session"].le(pd.Timestamp(active_to))
        ].copy()
        provider = provider.loc[
            provider["session"].le(pd.Timestamp(active_to))
        ].copy()
    internal = internal.sort_values("session", kind="stable")
    provider = provider.sort_values("session", kind="stable")
    if internal["session"].duplicated().any() or provider["session"].duplicated().any():
        raise RuntimeError("Reviewed price sessions are duplicated.")
    internal_sessions = set(internal["session"])
    provider_sessions = set(provider["session"])
    internal_only = sorted(internal_sessions - provider_sessions)
    provider_only = sorted(provider_sessions - internal_sessions)
    joined = internal.merge(
        provider,
        on="session",
        suffixes=("_internal", "_provider"),
        validate="one_to_one",
    ).sort_values("session", kind="stable")
    if joined.empty:
        raise RuntimeError("Reviewed price evidence has no overlap.")
    currencies = sorted(
        {
            _text(value).upper()
            for value in internal.get("currency", pd.Series(dtype="object"))
            if _text(value)
        }
    )
    if currencies != ["USD"]:
        raise RuntimeError("Reviewed internal price currency is not exact USD.")

    boundaries = sorted(
        pd.Timestamp(value).normalize()
        for value in split_dates
        if _date(value)
        and joined["session"].min() <= pd.Timestamp(value).normalize()
        <= joined["session"].max()
    )
    joined["regime"] = joined["session"].map(
        lambda value: sum(boundary <= value for boundary in boundaries)
    )
    mismatch_rows: list[dict[str, str]] = []
    regimes: list[dict[str, Any]] = []
    scale_tolerance = float(policy_prices["scale_stability_relative_tolerance"])
    close_tolerance = float(policy_prices["close_relative_tolerance"])
    ohl_tolerance = float(policy_prices["ohl_relative_tolerance"])
    absolute_tolerance = float(policy_prices["absolute_price_tolerance_usd"])
    configured_min_regime = int(policy_prices["minimum_split_regime_sessions"])
    minimum_regime = 1 if len(internal) < configured_min_regime else configured_min_regime
    substituted = internal.set_index("session")[["open", "high", "low", "close"]].copy()
    for regime_id, group in joined.groupby("regime", sort=True):
        ratio = group["close_provider"] / group["close_internal"]
        median_scale = float(ratio.median())
        max_deviation = float(((ratio / median_scale) - 1.0).abs().max())
        stable = bool(
            len(group) >= minimum_regime
            and math.isfinite(median_scale)
            and median_scale > 0
            and max_deviation <= scale_tolerance
        )
        regimes.append(
            {
                "regime": int(regime_id),
                "start": group["session"].min().date().isoformat(),
                "end": group["session"].max().date().isoformat(),
                "session_count": len(group),
                "median_scale": _number_text(median_scale),
                "maximum_scale_deviation": _number_text(max_deviation),
                "stable": stable,
            }
        )
        if not stable:
            continue
        for field in ("open", "high", "low", "close"):
            normalized = group[f"{field}_provider"] / median_scale
            internal_values = group[f"{field}_internal"]
            tolerance = close_tolerance if field == "close" else ohl_tolerance
            passed = (normalized - internal_values).abs().le(
                pd.Series(
                    [
                        max(absolute_tolerance, abs(value) * tolerance)
                        for value in internal_values
                    ],
                    index=group.index,
                )
            )
            for row_index in group.index[~passed]:
                session = group.loc[row_index, "session"]
                item = {
                    "session": session.date().isoformat(),
                    "field": field,
                    "internal": _number_text(group.loc[row_index, f"{field}_internal"]),
                    "provider": _number_text(group.loc[row_index, f"{field}_provider"]),
                    "normalized_provider": _number_text(normalized.loc[row_index]),
                    "median_scale": _number_text(median_scale),
                }
                mismatch_rows.append(item)
                substituted.loc[session, field] = float(normalized.loc[row_index])
    mismatch_rows = sorted(mismatch_rows, key=lambda item: (item["session"], item["field"]))
    expected_mismatches = sorted(
        expected["expected_mismatch_rows"],
        key=lambda item: (item["session"], item["field"]),
    )
    if mismatch_rows != expected_mismatches:
        raise RuntimeError("Reviewed price mismatch row inventory changed.")

    signal = {
        "settings": [
            {"period": period, "multiplier": _number_text(multiplier)}
            for period, multiplier in TRIPLE_SUPERTREND_SETTINGS
        ],
        "atr_method": TRIPLE_SUPERTREND_ATR_METHOD,
        "exit_down_count": TRIPLE_SUPERTREND_EXIT_DOWN_COUNT,
        "trend_change_count": 0,
        "buy_change_count": 0,
        "sell_change_count": 0,
    }
    if expected["case_code"] == "tiny_ohlc_disagreement":
        base_input = internal.set_index("session")[["high", "low", "close"]].rename(
            columns={"high": "High", "low": "Low", "close": "Close"}
        )
        alternate_input = substituted[["high", "low", "close"]].rename(
            columns={"high": "High", "low": "Low", "close": "Close"}
        )
        base_signal = add_triple_supertrend(
            base_input,
            settings=TRIPLE_SUPERTREND_SETTINGS,
            atr_method=TRIPLE_SUPERTREND_ATR_METHOD,
            exit_down_count=TRIPLE_SUPERTREND_EXIT_DOWN_COUNT,
        )
        alternate_signal = add_triple_supertrend(
            alternate_input,
            settings=TRIPLE_SUPERTREND_SETTINGS,
            atr_method=TRIPLE_SUPERTREND_ATR_METHOD,
            exit_down_count=TRIPLE_SUPERTREND_EXIT_DOWN_COUNT,
        )
        trend_columns = [f"TripleST{index}_Trend" for index in range(1, 4)]
        signal.update(
            {
                "trend_change_count": int(
                    base_signal[trend_columns].ne(
                        alternate_signal[trend_columns]
                    ).any(axis=1).sum()
                ),
                "buy_change_count": int(
                    base_signal["TripleBuySignal"]
                    .ne(alternate_signal["TripleBuySignal"])
                    .sum()
                ),
                "sell_change_count": int(
                    base_signal["TripleSellSignal"]
                    .ne(alternate_signal["TripleSellSignal"])
                    .sum()
                ),
            }
        )
        if any(
            signal[key] != 0
            for key in (
                "trend_change_count",
                "buy_change_count",
                "sell_change_count",
            )
        ):
            raise RuntimeError("Reviewed tiny OHLC disagreement changes Triple Supertrend.")

    overlap_projection = []
    for row in joined.to_dict(orient="records"):
        overlap_projection.append(
            {
                "session": _date(row["session"]),
                **{
                    f"{field}_{side}": _number_text(row[f"{field}_{side}"])
                    for field in ("open", "high", "low", "close", "volume")
                    for side in ("internal", "provider")
                },
            }
        )
    projection = {
        "schema": REVIEWED_PRICE_EVIDENCE_BASIS,
        **target_projection,
        "case_code": expected["case_code"],
        "source_sha256": expected["source_sha256"],
        "cache_wrapper_sha256": expected["cache_wrapper_sha256"],
        "metadata": metadata,
        "raw_row_count": expected["expected_raw_row_count"],
        "all_null_row_count": len(all_null_sessions),
        "all_null_sessions_sha256": canonical_json_sha256(all_null_sessions),
        "accepted_provider_row_count": len(provider),
        "internal_row_count": len(internal),
        "overlap_row_count": len(joined),
        "coverage_ratio": _number_text(len(joined) / len(internal)),
        "internal_only_sessions": [_date(value) for value in internal_only],
        "provider_only_sessions": [_date(value) for value in provider_only],
        "internal_ohlcv_sha256": canonical_json_sha256(
            _frame_projection(
                internal, ("session", "open", "high", "low", "close", "volume")
            )
        ),
        "provider_ohlcv_sha256": canonical_json_sha256(
            _frame_projection(
                provider, ("session", "open", "high", "low", "close", "volume")
            )
        ),
        "overlap_ohlcv_sha256": canonical_json_sha256(overlap_projection),
        "split_boundaries": [_date(value) for value in boundaries],
        "regimes": regimes,
        "mismatch_rows": mismatch_rows,
        "mismatch_rows_sha256": canonical_json_sha256(mismatch_rows),
        "signal": signal,
        "limitation": expected["limitation"],
    }
    minimum_coverage = float(policy_prices["minimum_session_coverage_ratio"])
    if (
        len(joined) < min(int(policy_prices["minimum_overlap_sessions"]), len(internal))
        or len(joined) / len(internal) < minimum_coverage
        or not all(item["stable"] for item in regimes)
    ):
        raise RuntimeError("Reviewed price evidence fails numeric/session policy.")
    return provider, projection


def verify_reviewed_price_projection(
    projection: Mapping[str, Any], spec: Mapping[str, Any]
) -> str:
    expected = canonical_reviewed_price_evidence(spec)
    digest = canonical_json_sha256(projection)
    if digest != expected["expected_projection_sha256"]:
        raise RuntimeError("Reviewed price projection hash changed.")
    return digest
