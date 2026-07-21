"""Code-pinned, price-only validation of the frozen BBBY/BBT WIKI extract.

The archived Quandl WIKI bytes are a narrow replacement for two unsafe Yahoo
symbol-only comparisons.  They are not an action or adjustment-factor source.
Both the offline collector and the publication gate call this module against
the immutable archive and the current release inputs; neither trusts values
copied into a cross-validation report.
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


REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS = (
    "frozen_wiki_identity_bound_price_only/v1"
)
REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_POLICY_KEY = (
    "reviewed_source_archive_price_only_evidence"
)
WIKI_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "marketneutral/quandl-wiki-prices-us-equites/WIKI_PRICES.csv"
)
WIKI_LICENSE_WARNING = (
    "Kaggle Quandl WIKI licenseName=Unknown; private/internal-only; "
    "redistribution/public publication blocked."
)
WIKI_EXTRACT_SHA256 = (
    "a6a6f651265825ed9ed95a1dfb9889f70586a728aa53eeae8585b8c00e4af52f"
)
WIKI_PROVENANCE_SHA256 = (
    "d73bf90641034b56b4ce42d9cef2fd4dff23a6db8c101cc7ed9b49af4c7140c8"
)
WIKI_EXTRACT_SIZE = 186_580
WIKI_EXTRACT_LINE_COUNT = 1_464
WIKI_EXTRACT_RETRIEVED_AT = "2026-07-18T03:58:26.808706Z"
WIKI_PROVENANCE_RETRIEVED_AT = "2026-07-19T04:30:00Z"
BBBY_SECURITY_ID = "US:EODHD:dbd287da-b35f-5aaa-873c-76f941b4d93b"
BBT_SECURITY_ID = "US:EODHD:aadcce22-62c7-522f-bbeb-861933af1d99"
LEGACY_DD_SECURITY_ID = "US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1"
TFC_SECURITY_ID = "US:EODHD:e9a02afb-49bb-545e-8b5a-824d630a1332"
BBT_TICKER_EVENT_ID = (
    "bba9b3139a40f93f1b90790cdbded3fe0db106526ca9890f1c611c67ee267131"
)
BBT_TICKER_SOURCE_SHA256 = (
    "094d8ee3bb3b2cdf33ff4492d6cc93a3738098e4d8bfbeb0c7962d8d9bc3208d"
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
    "provenance_sha256",
    "raw_price_source_sha256",
    "master_source",
    "identity_source_sha256",
    "relation_sha256",
    "overlap_session_count",
    "overlap_start",
    "overlap_end",
    "triple_supertrend_signal_sha256",
    "wiki_dividends_missing_from_current",
    "current_dividends_missing_from_wiki",
    "limitation",
)

TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS = frozenset(
    {
        "849f9b435475b46a2c328f943f76697c3fe5bfaa74b3c2b3f2d4db58fcc9e40a",
        "ed969b35974af909d34adab11ace79a964e9fc06d70e543d52ef573576cfd994",
    }
)
# Filled from the strict normalized policy inventory below.  The YAML alone is
# never sufficient authority for adding another ticker or changing a pin.
TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256 = (
    "fc172be411a43504ac64d9016931e73f6d64813709c36387908cd4c4feebe665"
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


def _digest(value: Any, field: str) -> str:
    digest = _text(value).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RuntimeError(f"Frozen WIKI price-only {field} must be SHA-256.")
    return digest


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"Frozen WIKI price-only {field} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Frozen WIKI price-only {field} must be an integer."
        ) from exc
    if result != value:
        raise RuntimeError(f"Frozen WIKI price-only {field} must be exact.")
    return result


def _dividend_events(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RuntimeError(f"Frozen WIKI price-only {field} must be a list.")
    output: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"date", "amount"}:
            raise RuntimeError(f"Frozen WIKI price-only {field} fields are not exact.")
        date = _date(item.get("date"))
        try:
            amount = float(item.get("amount"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Frozen WIKI price-only {field} amount is invalid."
            ) from exc
        if _text(item.get("date")) != date or not math.isfinite(amount) or amount <= 0:
            raise RuntimeError(f"Frozen WIKI price-only {field} value is invalid.")
        output.append({"date": date, "amount": amount})
    if output != sorted(output, key=lambda item: (item["date"], item["amount"])):
        raise RuntimeError(f"Frozen WIKI price-only {field} must be sorted.")
    if len(output) != len({(item["date"], item["amount"]) for item in output}):
        raise RuntimeError(f"Frozen WIKI price-only {field} is duplicated.")
    return output


def canonical_source_archive_price_only_spec(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(SPEC_FIELDS):
        raise RuntimeError("Frozen WIKI price-only policy fields are not exact.")
    output = {
        "target_id": _digest(value.get("target_id"), "target_id"),
        "security_id": _text(value.get("security_id")),
        "symbol": _text(value.get("symbol")).upper(),
        "target_provider_symbol": _text(value.get("target_provider_symbol")).upper(),
        "identity_active_from": _date(value.get("identity_active_from")),
        "identity_active_to": _date(value.get("identity_active_to")),
        "terminal_event_id": _digest(value.get("terminal_event_id"), "terminal_event_id"),
        "identity_bound_provider_symbol": _text(
            value.get("identity_bound_provider_symbol")
        ),
        "extract_sha256": _digest(value.get("extract_sha256"), "extract_sha256"),
        "provenance_sha256": _digest(
            value.get("provenance_sha256"), "provenance_sha256"
        ),
        "raw_price_source_sha256": _digest(
            value.get("raw_price_source_sha256"), "raw_price_source_sha256"
        ),
        "master_source": _text(value.get("master_source")),
        "identity_source_sha256": _digest(
            value.get("identity_source_sha256"), "identity_source_sha256"
        ),
        "relation_sha256": _digest(
            value.get("relation_sha256"), "relation_sha256"
        ),
        "overlap_session_count": _integer(
            value.get("overlap_session_count"), "overlap_session_count"
        ),
        "overlap_start": _date(value.get("overlap_start")),
        "overlap_end": _date(value.get("overlap_end")),
        "triple_supertrend_signal_sha256": _digest(
            value.get("triple_supertrend_signal_sha256"),
            "triple_supertrend_signal_sha256",
        ),
        "wiki_dividends_missing_from_current": _dividend_events(
            value.get("wiki_dividends_missing_from_current"),
            "wiki_dividends_missing_from_current",
        ),
        "current_dividends_missing_from_wiki": _dividend_events(
            value.get("current_dividends_missing_from_wiki"),
            "current_dividends_missing_from_wiki",
        ),
        "limitation": _text(value.get("limitation")),
    }
    if (
        not output["security_id"]
        or not output["symbol"]
        or output["target_provider_symbol"] != output["symbol"]
        or not output["identity_active_from"]
        or not output["identity_bound_provider_symbol"]
        or output["extract_sha256"] != WIKI_EXTRACT_SHA256
        or output["provenance_sha256"] != WIKI_PROVENANCE_SHA256
        or output["overlap_session_count"] <= 0
        or not output["overlap_start"]
        or not output["overlap_end"]
        or not output["limitation"]
    ):
        raise RuntimeError("Frozen WIKI price-only policy entry is incomplete.")
    return output


def source_archive_price_only_registry(
    prices_policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    values = prices_policy.get(REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_POLICY_KEY)
    if not isinstance(values, list):
        raise RuntimeError("Frozen WIKI price-only policy registry must be a list.")
    output: dict[str, dict[str, Any]] = {}
    for raw in values:
        spec = canonical_source_archive_price_only_spec(raw)
        target_id = spec["target_id"]
        if target_id in output:
            raise RuntimeError("Frozen WIKI price-only target is duplicated: " + target_id)
        output[target_id] = spec
    return output


def source_archive_price_only_inventory_sha256(
    prices_policy: Mapping[str, Any],
) -> str:
    registry = source_archive_price_only_registry(prices_policy)
    return _canonical_json_sha256([registry[key] for key in sorted(registry)])


def source_archive_price_only_spec_sha256(spec: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(canonical_source_archive_price_only_spec(spec))


def _safe_archive_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    target = (base / object_path).resolve()
    if target == base or base not in target.parents:
        raise RuntimeError("Frozen WIKI source_archive object path is unsafe.")
    return target


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
            "Frozen WIKI source_archive evidence is absent or duplicated: "
            + archive_id
        )
    row = rows.iloc[0]
    common = {
        "archive_id": archive_id,
        "source_hash": archive_id,
        "source_url": WIKI_DOWNLOAD_URL,
        "effective_date": "2026-07-15",
        **dict(expected),
    }
    changed = [key for key, value in common.items() if _text(row.get(key)) != value]
    if changed:
        raise RuntimeError(
            "Frozen WIKI source_archive metadata changed: " + ", ".join(changed)
        )
    path = _safe_archive_path(Path(repository.root), _text(row.get("object_path")))
    if _text(row.get("object_path")) != common["object_path"] or not path.is_file():
        raise RuntimeError("Frozen WIKI archived payload path changed or is missing.")
    encoded = path.read_bytes()
    try:
        payload = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    except (OSError, EOFError) as exc:
        raise RuntimeError("Frozen WIKI archived payload is not valid gzip.") from exc
    if sha256_bytes(payload) != archive_id:
        raise RuntimeError("Frozen WIKI archived payload hash changed: " + archive_id)
    return payload


def _one(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(f"Frozen WIKI {label} inventory changed.")
    return rows.iloc[0]


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
                    bool(value)
                    if isinstance(value, (bool, np.bool_))
                    else int(value)
                    for value in values
                ],
            ]
        )
    return _canonical_json_sha256(records)


def _event_pairs(values: list[dict[str, Any]]) -> list[list[Any]]:
    return [[item["date"], item["amount"]] for item in values]


def _target_diagnostic(
    spec: Mapping[str, Any],
    wiki: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    master: pd.DataFrame,
    history: pd.DataFrame,
    actions: pd.DataFrame,
    target: Mapping[str, Any],
    archived_audit: Mapping[str, Any],
) -> dict[str, Any]:
    target_expected = {
        "target_id": spec["target_id"],
        "security_id": spec["security_id"],
        "symbol": spec["symbol"],
        "provider_symbol": spec["target_provider_symbol"],
        "active_from": spec["identity_active_from"],
        "active_to": spec["identity_active_to"],
        "terminal_event_id": spec["terminal_event_id"],
    }
    if any(_text(target.get(key)) != value for key, value in target_expected.items()):
        raise RuntimeError(
            "Frozen WIKI target identity/provider interval changed: "
            + spec["target_id"]
        )

    master_row = _one(
        master,
        master["security_id"].map(_text).eq(spec["security_id"]),
        spec["symbol"] + " security_master",
    )
    history_row = _one(
        history,
        history["security_id"].map(_text).eq(spec["security_id"])
        & history["symbol"].map(_text).str.upper().eq(spec["symbol"]),
        spec["symbol"] + " symbol_history",
    )
    master_expected = {
        "primary_symbol": spec["symbol"],
        "provider_symbol": spec["identity_bound_provider_symbol"],
        "active_from": "2015-01-02",
        "active_to": "2023-05-02" if spec["symbol"] == "BBBY" else "2019-12-06",
        "source": spec["master_source"],
        "source_hash": spec["identity_source_sha256"],
    }
    history_expected = {
        "symbol": spec["symbol"],
        "effective_from": spec["identity_active_from"],
        "effective_to": spec["identity_active_to"],
        "source": spec["master_source"],
        "source_hash": spec["identity_source_sha256"],
    }
    if any(_text(master_row.get(key)) != value for key, value in master_expected.items()):
        raise RuntimeError("Frozen WIKI identity-bound master pin changed: " + spec["symbol"])
    if any(_text(history_row.get(key)) != value for key, value in history_expected.items()):
        raise RuntimeError("Frozen WIKI identity-bound history pin changed: " + spec["symbol"])

    if spec["symbol"] == "BBT":
        ticker = _one(
            actions,
            actions["event_id"].map(_text).eq(BBT_TICKER_EVENT_ID),
            "BBT official ticker-change action",
        )
        ticker_expected = {
            "security_id": BBT_SECURITY_ID,
            "action_type": "ticker_change",
            "effective_date": "2019-12-06",
            "new_security_id": TFC_SECURITY_ID,
            "new_symbol": "TFC",
            "source_hash": BBT_TICKER_SOURCE_SHA256,
        }
        if (
            any(_text(ticker.get(key)) != value for key, value in ticker_expected.items())
            or not bool(ticker.get("official"))
        ):
            raise RuntimeError("Frozen WIKI BBT -> TFC identity boundary changed.")

    target_prices = prices.loc[
        prices["security_id"].map(_text).eq(spec["security_id"])
    ].copy()
    if target_prices.empty or target_prices["session"].map(_date).duplicated().any():
        raise RuntimeError("Frozen WIKI internal raw price inventory changed: " + spec["symbol"])
    numeric_columns = ("open", "high", "low", "close", "volume")
    for column in numeric_columns:
        values = pd.to_numeric(target_prices[column], errors="raise")
        if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
            raise RuntimeError("Frozen WIKI internal raw price is not finite: " + spec["symbol"])
    if (
        set(target_prices["source"].map(_text)) != {"eodhd_eod"}
        or set(target_prices["source_hash"].map(_text))
        != {spec["raw_price_source_sha256"]}
    ):
        raise RuntimeError("Frozen WIKI internal raw price provenance changed: " + spec["symbol"])

    wiki = wiki.copy()
    wiki["date"] = wiki["date"].map(_date)
    if wiki["date"].duplicated().any():
        raise RuntimeError("Frozen WIKI extract sessions are duplicated: " + spec["symbol"])
    for column in numeric_columns:
        values = pd.to_numeric(wiki[column], errors="raise")
        if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
            raise RuntimeError("Frozen WIKI archived price is not finite: " + spec["symbol"])
    target_prices["date"] = target_prices["session"].map(_date)
    overlap = target_prices.loc[
        target_prices["date"].between(spec["overlap_start"], spec["overlap_end"])
    ].copy()
    if not set(wiki["date"]).issubset(set(overlap["date"])):
        raise RuntimeError("Frozen WIKI/current session relation changed: " + spec["symbol"])
    joined = overlap.merge(
        wiki,
        on="date",
        suffixes=("_eod", "_wiki"),
        validate="one_to_one",
    ).sort_values("date", ignore_index=True)
    relation_sha256 = _relation_sha256(joined)
    if (
        len(joined) != spec["overlap_session_count"]
        or _date(joined["date"].min()) != spec["overlap_start"]
        or _date(joined["date"].max()) != spec["overlap_end"]
        or relation_sha256 != spec["relation_sha256"]
    ):
        raise RuntimeError("Frozen WIKI exact price relation changed: " + spec["symbol"])

    target_factors = factors.loc[
        factors["security_id"].map(_text).eq(spec["security_id"])
    ].copy()
    if target_factors.empty:
        raise RuntimeError("Frozen WIKI current factors are missing: " + spec["symbol"])
    for column in ("split_factor", "total_return_factor"):
        values = pd.to_numeric(target_factors[column], errors="raise")
        if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
            raise RuntimeError("Frozen WIKI current factors are not finite: " + spec["symbol"])
    if set(pd.to_numeric(target_factors["split_factor"], errors="raise")) != {1.0}:
        raise RuntimeError("Frozen WIKI split-factor economics changed: " + spec["symbol"])

    candidate = target_prices.drop(columns="date").copy()
    candidate["_date"] = candidate["session"].map(_date)
    by_date = wiki.set_index("date")
    replace = candidate["_date"].isin(by_date.index)
    for column in numeric_columns:
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
    current_signal_sha256 = _signal_sha256(current_signal)
    substituted_signal_sha256 = _signal_sha256(substituted_signal)
    if (
        any(differences.values())
        or current_signal_sha256 != substituted_signal_sha256
        or current_signal_sha256 != spec["triple_supertrend_signal_sha256"]
    ):
        raise RuntimeError(
            "Frozen WIKI substitution changed Triple Supertrend: " + spec["symbol"]
        )

    wiki_amounts = pd.to_numeric(wiki["ex-dividend"], errors="raise")
    if not bool(np.isfinite(wiki_amounts.to_numpy(dtype=float)).all()):
        raise RuntimeError("Frozen WIKI dividends are not finite: " + spec["symbol"])
    wiki_events = sorted(
        (str(row.date), float(row.amount))
        for row in pd.DataFrame(
            {"date": wiki["date"], "amount": wiki_amounts}
        ).loc[wiki_amounts.gt(0)].itertuples(index=False)
    )
    current = actions.loc[
        actions["security_id"].map(_text).eq(spec["security_id"])
        & actions["action_type"].map(_text).eq("cash_dividend")
        & actions["effective_date"].map(_date).between(
            spec["overlap_start"], spec["overlap_end"]
        )
    ]
    current_events = sorted(
        (_date(row.effective_date), float(row.cash_amount))
        for row in current.itertuples(index=False)
    )
    missing = sorted(set(wiki_events) - set(current_events))
    extra = sorted(set(current_events) - set(wiki_events))
    expected_missing = [
        (item["date"], item["amount"])
        for item in spec["wiki_dividends_missing_from_current"]
    ]
    expected_extra = [
        (item["date"], item["amount"])
        for item in spec["current_dividends_missing_from_wiki"]
    ]
    if missing != expected_missing or extra != expected_extra:
        raise RuntimeError("Frozen WIKI action-coverage gap changed: " + spec["symbol"])

    relation = archived_audit.get("raw_price_relation")
    signal = archived_audit.get("wiki_raw_substitution_sensitivity")
    coverage = archived_audit.get("action_factor_coverage")
    identity = archived_audit.get("identity")
    if not all(isinstance(value, Mapping) for value in (relation, signal, coverage, identity)):
        raise RuntimeError("Frozen WIKI reviewed provenance audit is incomplete.")
    if (
        _text(archived_audit.get("status")) != "passed_price_only_arbitration"
        or _text(archived_audit.get("security_id")) != spec["security_id"]
        or _text(archived_audit.get("symbol")) != spec["symbol"]
        or _text(relation.get("relation_sha256")) != relation_sha256
        or _integer(relation.get("row_count"), "archived overlap row_count")
        != len(joined)
        or _text(signal.get("current_signal_sha256")) != current_signal_sha256
        or _text(signal.get("substituted_signal_sha256"))
        != substituted_signal_sha256
        or signal.get("triple_supertrend_field_differences") != differences
        or _text(coverage.get("status")) != "incomplete_not_rewritten"
        or coverage.get("wiki_events_missing_from_current")
        != _event_pairs(spec["wiki_dividends_missing_from_current"])
        or coverage.get("current_events_missing_from_wiki")
        != _event_pairs(spec["current_dividends_missing_from_wiki"])
        or coverage.get("price_only_pass_must_not_imply_action_factor_pass") is not True
        or coverage.get("actions_rewritten") is not False
        or coverage.get("factors_rewritten") is not False
        or _text(identity.get("security_id")) != spec["security_id"]
        or _text(identity.get("provider_symbol"))
        != spec["identity_bound_provider_symbol"]
        or identity.get("yahoo_symbol_only_identity_reuse_allowed") is not False
        or archived_audit.get("raw_price_rewritten") is not False
        or archived_audit.get("corporate_actions_rewritten") is not False
        or archived_audit.get("adjustment_factors_rewritten") is not False
    ):
        raise RuntimeError("Frozen WIKI reviewed provenance/result binding changed.")

    diagnostic = {
        "validation_basis": REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS,
        "policy_spec_sha256": source_archive_price_only_spec_sha256(spec),
        "policy_registry_sha256": TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256,
        "extract_sha256": WIKI_EXTRACT_SHA256,
        "provenance_sha256": WIKI_PROVENANCE_SHA256,
        "target_id": spec["target_id"],
        "security_id": spec["security_id"],
        "symbol": spec["symbol"],
        "target_provider_symbol": spec["target_provider_symbol"],
        "identity_bound_provider_symbol": spec["identity_bound_provider_symbol"],
        "identity_active_from": spec["identity_active_from"],
        "identity_active_to": spec["identity_active_to"],
        "terminal_event_id": spec["terminal_event_id"],
        "raw_price_source_sha256": spec["raw_price_source_sha256"],
        "identity_source_sha256": spec["identity_source_sha256"],
        "overlap_session_count": len(joined),
        "overlap_start": spec["overlap_start"],
        "overlap_end": spec["overlap_end"],
        "relation_sha256": relation_sha256,
        "triple_supertrend_field_differences": differences,
        "current_signal_sha256": current_signal_sha256,
        "substituted_signal_sha256": substituted_signal_sha256,
        "action_factor_status": "incomplete_not_rewritten",
        "wiki_dividends_missing_from_current": spec[
            "wiki_dividends_missing_from_current"
        ],
        "current_dividends_missing_from_wiki": spec[
            "current_dividends_missing_from_wiki"
        ],
        "raw_price_rewritten": False,
        "corporate_actions_rewritten": False,
        "adjustment_factors_rewritten": False,
        "generic_ticker_reuse_allowed": False,
        "yahoo_symbol_only_identity_reuse_allowed": False,
        "price_only_pass_must_not_imply_action_factor_pass": True,
        "limitation": spec["limitation"],
    }
    return {
        **diagnostic,
        "projection_sha256": _canonical_json_sha256(diagnostic),
    }


def verify_source_archive_price_only_evidence(
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
    """Recompute both reviewed price-only cases from immutable local inputs."""

    registry = source_archive_price_only_registry(prices_policy)
    inventory_hash = source_archive_price_only_inventory_sha256(prices_policy)
    if (
        set(registry) != set(TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS)
        or inventory_hash != TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256
    ):
        raise RuntimeError("Frozen WIKI price-only policy inventory is not code-pinned.")
    if set(targets) != set(TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS):
        raise RuntimeError("Frozen WIKI price-only targets are not the exact reviewed pair.")
    if WIKI_LICENSE_WARNING not in set(release_warnings):
        raise RuntimeError("Frozen WIKI private/internal-only release warning is missing.")

    extract_payload = _archive_payload(
        repository,
        archive,
        archive_id=WIKI_EXTRACT_SHA256,
        expected={
            "dataset": "kaggle_quandl_wiki_bbby_bbt_price_extract",
            "source": "kaggle_quandl_wiki_bbby_bbt_price_extract",
            "content_type": "text/csv",
            "retrieved_at": WIKI_EXTRACT_RETRIEVED_AT,
            "object_path": (
                "archives/2026-07-15/" + WIKI_EXTRACT_SHA256 + ".csv.gz"
            ),
        },
    )
    if (
        len(extract_payload) != WIKI_EXTRACT_SIZE
        or len(extract_payload.splitlines()) != WIKI_EXTRACT_LINE_COUNT
    ):
        raise RuntimeError("Frozen WIKI extract size/line inventory changed.")
    extract = pd.read_csv(io.BytesIO(extract_payload))
    if tuple(extract.columns) != WIKI_COLUMNS:
        raise RuntimeError("Frozen WIKI extract columns changed.")
    if set(extract["ticker"].map(_text)) != {"BBBY", "BBT"}:
        raise RuntimeError("Frozen WIKI extract ticker inventory changed.")

    provenance_payload = _archive_payload(
        repository,
        archive,
        archive_id=WIKI_PROVENANCE_SHA256,
        expected={
            "dataset": "reviewed_us_wiki_price_arbitration",
            "source": "reviewed_us_wiki_price_arbitration",
            "content_type": "application/json",
            "retrieved_at": WIKI_PROVENANCE_RETRIEVED_AT,
            "object_path": (
                "archives/2026-07-15/" + WIKI_PROVENANCE_SHA256 + ".json.gz"
            ),
        },
    )
    try:
        provenance = json.loads(provenance_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Frozen WIKI provenance is not valid JSON.") from exc
    if provenance_payload != _canonical_json_bytes(provenance):
        raise RuntimeError("Frozen WIKI provenance is not canonical JSON.")
    scope = provenance.get("scope") if isinstance(provenance, Mapping) else None
    license_policy = (
        provenance.get("license_policy") if isinstance(provenance, Mapping) else None
    )
    legacy_dd = provenance.get("legacy_dd") if isinstance(provenance, Mapping) else None
    frozen = provenance.get("frozen_evidence") if isinstance(provenance, Mapping) else None
    audits = provenance.get("price_arbitrations") if isinstance(provenance, Mapping) else None
    if (
        provenance.get("schema") != "us_wiki_price_arbitration/v1"
        or not isinstance(scope, Mapping)
        or scope.get("passed_price_only") != [BBBY_SECURITY_ID, BBT_SECURITY_ID]
        or scope.get("blocked") != [LEGACY_DD_SECURITY_ID]
        or scope.get("write_dataset") != "source_archive"
        or not isinstance(license_policy, Mapping)
        or license_policy.get("formal_license_name") != "Unknown"
        or license_policy.get("allowed_scope") != "private_internal_only"
        or license_policy.get("redistribution_allowed") is not False
        or license_policy.get("public_publication_allowed") is not False
        or license_policy.get("fail_closed") is not True
        or not isinstance(legacy_dd, Mapping)
        or legacy_dd.get("security_id") != LEGACY_DD_SECURITY_ID
        or legacy_dd.get("status") != "blocked_fail_closed"
        or legacy_dd.get("apply_allowed") is not False
        or legacy_dd.get("wiki_2015_07_01_is_cash_dividend") is not False
        or legacy_dd.get("raw_price_rewritten") is not False
        or legacy_dd.get("corporate_actions_rewritten") is not False
        or legacy_dd.get("adjustment_factors_rewritten") is not False
        or not isinstance(frozen, Mapping)
        or frozen.get("extract_sha256") != WIKI_EXTRACT_SHA256
        or frozen.get("extract_size") != WIKI_EXTRACT_SIZE
        or frozen.get("extract_line_count") != WIKI_EXTRACT_LINE_COUNT
        or frozen.get("metadata_license_name") != "Unknown"
        or not isinstance(audits, list)
        or len(audits) != 2
    ):
        raise RuntimeError("Frozen WIKI provenance scope/license/DD block changed.")
    archived_by_security = {
        _text(item.get("security_id")): item
        for item in audits
        if isinstance(item, Mapping)
    }
    if set(archived_by_security) != {BBBY_SECURITY_ID, BBT_SECURITY_ID}:
        raise RuntimeError("Frozen WIKI reviewed arbitration inventory changed.")

    output: dict[str, dict[str, Any]] = {}
    for target_id, spec in registry.items():
        symbol = spec["symbol"]
        wiki = extract.loc[extract["ticker"].map(_text).eq(symbol)].copy()
        output[target_id] = _target_diagnostic(
            spec,
            wiki,
            prices=prices,
            factors=factors,
            master=master,
            history=history,
            actions=actions,
            target=targets[target_id],
            archived_audit=archived_by_security[spec["security_id"]],
        )
    return output
