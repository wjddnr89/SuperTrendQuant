#!/usr/bin/env python3
"""Repair FRC->FRCB continuity and PARA->PSKY default-stock succession.

The two cases look terminal in the provider history but are not zero-value
delistings:

* OCC memo 52352 records that the same First Republic common shares (CUSIP
  33616C100) changed from FRC to FRCB and began OTC trading on 2023-05-03.
  The FDIC receivership remains unresolved, so this repair keeps the security
  alive under FRCB and never invents a final cash recovery.
* Paramount's 2025-08-07 Form 8-K says a Class B holder who made neither a
  cash nor stock election retained one New Paramount Class B share.  Cash
  proration applied only to cash electors.  A deterministic, no-election
  backtest therefore receives one PSKY share per PARA share.

Only FRCB requires provider collection.  It is opt-in and hard-capped at three
one-shot EODHD calls (EOD, dividends, splits); PARA/PSKY is repaired from the
already-stored PSKY history.  Offline plan/apply never constructs an HTTP
client.  Apply uses a repository-wide lock, current-release/pointer CAS,
rollback journal and an exact repaired-state idempotency check.  This script
never accesses R2.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import json
import math
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from supertrend_quant.env import load_env
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import EodhdCallBudget, EodhdClient, SourceArtifact
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


FRC_SECURITY_ID = "US:EODHD:3453b450-03d1-52ee-9100-cf80856b06ef"
PARA_SECURITY_ID = "US:EODHD:f60b749b-3d84-552a-9dc9-39e742f67537"
PSKY_SECURITY_ID = "US:EODHD:fe84848c-624b-5aba-b542-24af3959f97f"

FRC_OLD_SYMBOL = "FRC"
FRC_NEW_SYMBOL = "FRCB"
FRC_TRANSITION = "2023-05-03"
FRC_OLD_LAST = "2023-05-02"
FRC_INDEX_EXIT = "2023-05-04"
FRC_PROVIDER_SYMBOL = "FRCB.US"

PARA_SYMBOL = "PARA"
PSKY_SYMBOL = "PSKY"
PARA_LAST = "2025-08-06"
PARA_TRANSITION = "2025-08-07"

FETCH_START = FRC_TRANSITION
FETCH_END = "2026-07-15"
EODHD_ENDPOINTS = ("eod", "div", "splits")
EXPECTED_EODHD_CALLS = 3
REQUEST_PARAMS = {"from": FETCH_START, "to": FETCH_END}
REQUEST_URLS = {
    endpoint: (
        f"https://eodhd.com/api/{endpoint}/{FRC_PROVIDER_SYMBOL}"
        f"?from={FETCH_START}&to={FETCH_END}"
    )
    for endpoint in EODHD_ENDPOINTS
}

# Corrections are deliberately empty until a quarantined raw EOD response has
# been reviewed.  A future entry must bind the exact raw-response SHA-256 and
# session, then state the one high/low value that may be expanded to the
# smallest valid OHLC envelope.  No heuristic or symbol-wide repair exists.
FRCB_EOD_ENVELOPE_CORRECTION_ALLOWLIST: Mapping[
    str, Mapping[str, tuple[Mapping[str, Any], ...]]
] = {
    "3c96be232fb5f567e77a94fe315b67aa61e520d1611842620c04680fb5df6ab3": {
        "2024-12-30": (
            {
                "field": "low",
                "observed": 0.0,
                "corrected": 0.003,
                "observed_row": {
                    "open": 0.003,
                    "high": 0.006,
                    "low": 0.0,
                    "close": 0.004,
                    "volume": 629864,
                },
                "justification": (
                    "Exact-hash reviewed zero-low provider defect; replace only low "
                    "with the minimum positive observed OHLC boundary."
                ),
            },
        )
    }
}

# The first three calls were charged, but their response bodies were not
# preserved.  This note is intentionally stored separately from every raw
# acquisition envelope and cannot be used to replay or validate market data.
PRIOR_FAILED_ATTEMPT_NOTE = {
    "schema": "us_frcb_prior_failed_attempt_note/v1",
    "budget_period": "2026-07-18",
    "budget_used_before": 8832,
    "budget_used_after": 8835,
    "budget_delta": 3,
    "raw_responses_preserved": False,
    "raw_payload_binding": None,
    "evidence_scope": "operator-observed budget transition and persistent ledger only",
}

OCC_MEMO_URL = "https://infomemo.theocc.com/infomemos?number=52352"
OCC_MEMO_NUMBER = "52352"
FRC_CUSIP = "33616C100"
FDIC_URL = (
    "https://www.fdic.gov/resources/resolutions/bank-failures/"
    "failed-bank-list/first-republic.html"
)
FDIC_ARCHIVE_SHA256 = (
    "30c6bad80710f702144fa3ef61ca1ae14e81503fb0f436abdad7f91ebe8e51eb"
)
PARA_SEC_URL = (
    "https://www.sec.gov/Archives/edgar/data/813828/"
    "000119312525175027/0001193125-25-175027.txt"
)
PARA_SEC_SHA256 = (
    "61ea922a72a55f05b79c2cf00e9c4b0367434c35bf25a78fd4d815d3b20e68be"
)

FRC_EVENT_ID = canonical_lifecycle_event_id(
    FRC_SECURITY_ID, "ticker_change", FRC_TRANSITION
)
PARA_EVENT_ID = canonical_lifecycle_event_id(
    PARA_SECURITY_ID, "stock_merger", PARA_TRANSITION
)

WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "lifecycle_resolutions",
    "adjustment_factors",
    "source_archive",
)


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _event_id(source: str, action_type: str, effective_date: str) -> str:
    return hashlib.sha256(
        f"{source}|{FRC_SECURITY_ID}|{action_type}|{effective_date}".encode()
    ).hexdigest()


def _occ_reviewed_extraction() -> dict[str, Any]:
    """Deterministic reviewed extraction; it is not represented as raw PDF."""

    return {
        "schema": "occ_reviewed_memo_extraction/v1",
        "memo_number": OCC_MEMO_NUMBER,
        "source_url": OCC_MEMO_URL,
        "subject": "First Republic Bank - Symbol Change",
        "effective_date": FRC_TRANSITION,
        "old_symbol": FRC_OLD_SYMBOL,
        "new_symbol": FRC_NEW_SYMBOL,
        "market": "OTC",
        "cusip": FRC_CUSIP,
        "contract_multiplier": 1,
        "deliverable_per_contract": "100 First Republic Bank (FRCB) Common Shares",
        "reviewed_claim": (
            "FRC and FRCB are the same First Republic common-share identity; "
            "only the market and ticker changed on 2023-05-03."
        ),
    }


def _occ_artifact() -> SourceArtifact:
    return SourceArtifact(
        source="occ_reviewed_memo_extraction",
        source_url=OCC_MEMO_URL,
        retrieved_at="2026-07-18T00:00:00Z",
        content=_canonical_json_bytes(_occ_reviewed_extraction()),
        content_type="application/json",
    )


def _assert_reviewed_local_evidence(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
) -> None:
    expected = {
        FDIC_ARCHIVE_SHA256: FDIC_URL,
        PARA_SEC_SHA256: PARA_SEC_URL,
    }
    for digest, url in expected.items():
        rows = source_archive.loc[
            source_archive["source_hash"].astype(str).eq(digest)
            & source_archive["source_url"].astype(str).eq(url)
        ]
        if len(rows) != 1:
            raise ValueError(f"Reviewed lifecycle evidence is missing or ambiguous: {url}")
        path = repository.root / _text(rows.iloc[0]["object_path"])
        if not path.is_file():
            raise ValueError(f"Reviewed lifecycle archive is missing: {path}")
        try:
            content = gzip.decompress(path.read_bytes())
        except Exception as exc:
            raise ValueError(f"Reviewed lifecycle archive is unreadable: {path}") from exc
        if sha256_bytes(content) != digest:
            raise ValueError(f"Reviewed lifecycle archive hash changed: {path}")

    para_path = repository.root / _text(
        source_archive.loc[
            source_archive["source_hash"].astype(str).eq(PARA_SEC_SHA256)
        ].iloc[0]["object_path"]
    )
    normalized = gzip.decompress(para_path.read_bytes()).decode(
        "utf-8", errors="replace"
    )
    required = (
        "neither a cash or stock election was made",
        "remained issued and outstanding as one (1)",
        "proration mechanism",
        "285,889,212",
    )
    # EDGAR text contains numeric HTML entities between some words, but the
    # reviewed phrases above remain literal in the stored filing.
    if any(phrase not in normalized for phrase in required):
        raise ValueError("Stored PARA filing no longer proves the default-stock policy.")


@dataclass(frozen=True)
class ProviderBundle:
    prices: pd.DataFrame
    corporate_actions: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    original_http_attempts: int
    budget_used_before: int | None = None
    budget_used_after: int | None = None
    budget_receipt: Mapping[str, Any] | None = None
    envelope_corrections: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class RawQuarantine:
    quarantine_id: str
    path: Path
    artifacts: tuple[SourceArtifact, ...]
    budget_receipt: Mapping[str, Any]
    acquisition_signature: Mapping[str, Any]


@dataclass
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    archive_artifacts: tuple[SourceArtifact, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]


class OhlcvValidationError(ValueError):
    """Non-sensitive, row-level report for an invalid provider price envelope."""

    def __init__(
        self,
        diagnostics: Iterable[Mapping[str, Any]],
        *,
        quarantine_path: Path | None = None,
        diagnostic_path: Path | None = None,
    ):
        self.diagnostics = tuple(dict(item) for item in diagnostics)
        self.quarantine_path = quarantine_path
        self.diagnostic_path = diagnostic_path
        locations = ""
        if quarantine_path is not None:
            locations += f" quarantine={quarantine_path}"
        if diagnostic_path is not None:
            locations += f" diagnostic={diagnostic_path}"
        super().__init__(
            "FRCB bundle contains invalid OHLCV relationships: "
            f"{json.dumps(self.diagnostics, sort_keys=True, separators=(',', ':'))}"
            f"{locations}"
        )


class ExactThreeFrcbEodhdClient(EodhdClient):
    """One budget claim and one HTTP attempt per reviewed endpoint, no retry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempted_endpoints: list[str] = []

    def get_raw_artifact(
        self,
        endpoint: str,
        *,
        params: dict[str, object] | None = None,
        retrieved_at: str,
    ) -> SourceArtifact:
        normalized = endpoint.strip("/")
        position = len(self.attempted_endpoints)
        if position >= EXPECTED_EODHD_CALLS:
            raise RuntimeError("FRCB client refused a fourth EODHD request.")
        expected = f"{EODHD_ENDPOINTS[position]}/{FRC_PROVIDER_SYMBOL}"
        if normalized != expected or dict(params or {}) != REQUEST_PARAMS:
            raise RuntimeError(
                "FRCB client refused a non-reviewed request: "
                f"expected={expected}/{REQUEST_PARAMS}, observed={normalized}/{params}."
            )
        self.budget.claim()
        self.attempted_endpoints.append(normalized)
        response = self.session.get(
            f"{self.base_url}/{normalized}",
            params={**REQUEST_PARAMS, "api_token": self.token, "fmt": "json"},
            timeout=120,
        )
        response.raise_for_status()
        short_endpoint = EODHD_ENDPOINTS[position]
        return SourceArtifact(
            source=f"eodhd_{short_endpoint}",
            source_url=REQUEST_URLS[short_endpoint],
            retrieved_at=retrieved_at,
            content=bytes(response.content),
            content_type=str(
                getattr(response, "headers", {}).get(
                    "Content-Type", "application/json"
                )
            ),
        )

    def get_json(self, endpoint: str, *, params: dict[str, object] | None = None):
        artifact = self.get_raw_artifact(
            endpoint, params=params, retrieved_at=utc_now_iso()
        )
        value = json.loads(artifact.content)
        if not isinstance(value, list):
            raise RuntimeError(f"EODHD {endpoint.strip('/')} did not return a row list.")
        return value


def _budget_used(budget: EodhdCallBudget) -> int:
    used = int(budget.seed_used)
    if budget.state_path.is_file():
        try:
            value = json.loads(budget.state_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            value = {}
        if _text(value.get("period")) == budget.period:
            used = max(used, int(value.get("used", 0)))
    return used


def _source_artifact(endpoint: str, rows: list[dict[str, Any]], at: str) -> SourceArtifact:
    return SourceArtifact(
        source=f"eodhd_{endpoint}",
        source_url=REQUEST_URLS[endpoint],
        retrieved_at=at,
        content=_canonical_json_bytes(rows),
        content_type="application/json",
    )


def _price_frame(rows: Any, artifact: SourceArtifact) -> pd.DataFrame:
    if not isinstance(rows, list):
        raise ValueError("FRCB EOD payload is not a list.")
    records: list[dict[str, Any]] = []
    for row in rows:
        session = _date(row.get("date"))
        if not session:
            continue
        if session < FETCH_START or session > FETCH_END:
            raise ValueError("FRCB EOD row is outside the frozen request range.")
        records.append(
            {
                "security_id": FRC_SECURITY_ID,
                "session": session,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume", 0),
                "currency": "USD",
                "source": artifact.source,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
                "source_url": artifact.source_url,
            }
        )
    return pd.DataFrame(
        records,
        columns=tuple(
            dict.fromkeys((*dataset_spec("daily_price_raw").required_columns, "source_url"))
        ),
    )


def _split_ratio(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    for separator in ("/", ":"):
        if separator in text:
            left, right = text.split(separator, 1)
            try:
                ratio = float(left) / float(right)
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            return ratio if math.isfinite(ratio) and ratio > 0 else None
    try:
        ratio = float(text)
    except (TypeError, ValueError):
        return None
    return ratio if math.isfinite(ratio) and ratio > 0 else None


def _provider_action(
    artifact: SourceArtifact,
    *,
    action_type: str,
    effective_date: str,
    cash_amount: float | None = None,
    ratio: float | None = None,
    announcement_date: str = "",
    record_date: str = "",
    payment_date: str = "",
) -> dict[str, Any]:
    return {
        "event_id": _event_id(artifact.source, action_type, effective_date),
        "security_id": FRC_SECURITY_ID,
        "action_type": action_type,
        "effective_date": effective_date,
        "ex_date": effective_date,
        "announcement_date": announcement_date,
        "record_date": record_date,
        "payment_date": payment_date,
        "cash_amount": cash_amount,
        "ratio": ratio,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_url": artifact.source_url,
        "source_kind": "provider",
        "source": artifact.source,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
        "metadata": None,
    }


def _action_frame(endpoint: str, rows: Any, artifact: SourceArtifact) -> pd.DataFrame:
    if not isinstance(rows, list):
        raise ValueError(f"FRCB {endpoint} payload is not a list.")
    records: list[dict[str, Any]] = []
    for row in rows:
        effective = _date(row.get("date"))
        if not effective:
            continue
        if effective < FETCH_START or effective > FETCH_END:
            raise ValueError(f"FRCB {endpoint} row is outside the frozen range.")
        if endpoint == "div":
            try:
                amount = float(row.get("unadjustedValue", row.get("value")))
            except (TypeError, ValueError):
                raise ValueError(f"FRCB dividend amount is invalid on {effective}.")
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError(f"FRCB dividend amount is invalid on {effective}.")
            records.append(
                _provider_action(
                    artifact,
                    action_type="cash_dividend",
                    effective_date=effective,
                    cash_amount=amount,
                    announcement_date=_date(row.get("declarationDate")),
                    record_date=_date(row.get("recordDate")),
                    payment_date=_date(row.get("paymentDate")),
                )
            )
        else:
            ratio = _split_ratio(row.get("split"))
            if ratio is None:
                raise ValueError(f"FRCB split ratio is invalid on {effective}.")
            records.append(
                _provider_action(
                    artifact,
                    action_type="split",
                    effective_date=effective,
                    ratio=ratio,
                )
            )
    return pd.DataFrame(
        records,
        columns=tuple(
            dict.fromkeys((*dataset_spec("corporate_actions").required_columns, "metadata"))
        ),
    )


def _diagnostic_value(value: Any) -> float | int | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _text(value)
    if not math.isfinite(number):
        return _text(value)
    return int(number) if number.is_integer() else number


def invalid_ohlcv_diagnostics(prices: pd.DataFrame) -> list[dict[str, Any]]:
    """Return exact bad rows/fields without URLs, credentials, or raw bodies."""

    fields = ("open", "high", "low", "close", "volume")
    numeric = prices[list(fields)].apply(pd.to_numeric, errors="coerce")
    diagnostics: list[dict[str, Any]] = []
    for index, row in numeric.iterrows():
        violations: list[dict[str, Any]] = []

        def add(rule: str, names: tuple[str, ...]) -> None:
            violations.append(
                {
                    "rule": rule,
                    "fields": list(names),
                    "values": {
                        name: _diagnostic_value(prices.at[index, name]) for name in names
                    },
                }
            )

        for field in fields:
            if not math.isfinite(float(row[field])):
                add("finite", (field,))
        for field in ("open", "high", "low", "close"):
            if math.isfinite(float(row[field])) and float(row[field]) <= 0:
                add("positive", (field,))
        if math.isfinite(float(row["volume"])) and float(row["volume"]) < 0:
            add("non_negative", ("volume",))
        pairs = (
            ("open_below_low", "open", "low", lambda a, b: a < b),
            ("open_above_high", "open", "high", lambda a, b: a > b),
            ("close_below_low", "close", "low", lambda a, b: a < b),
            ("close_above_high", "close", "high", lambda a, b: a > b),
            ("high_below_low", "high", "low", lambda a, b: a < b),
        )
        for rule, left, right, predicate in pairs:
            if (
                math.isfinite(float(row[left]))
                and math.isfinite(float(row[right]))
                and predicate(float(row[left]), float(row[right]))
            ):
                add(rule, (left, right))
        if violations:
            diagnostics.append(
                {
                    "provider_symbol": FRC_PROVIDER_SYMBOL,
                    "session": _date(prices.at[index, "session"]),
                    "row_values": {
                        field: _diagnostic_value(prices.at[index, field])
                        for field in fields
                    },
                    "violations": violations,
                }
            )
    return sorted(diagnostics, key=lambda item: item["session"])


def _apply_allowlisted_envelope_corrections(
    prices: pd.DataFrame,
    *,
    raw_eod_sha256: str,
) -> tuple[pd.DataFrame, tuple[Mapping[str, Any], ...]]:
    """Expand only high/low for an exact reviewed raw hash and exact date."""

    diagnostics = invalid_ohlcv_diagnostics(prices)
    allowlist = FRCB_EOD_ENVELOPE_CORRECTION_ALLOWLIST.get(raw_eod_sha256, {})
    invalid_sessions = {item["session"] for item in diagnostics}
    if not diagnostics:
        if allowlist:
            raise ValueError("FRCB correction allowlist is stale for an already-valid raw hash.")
        return prices.copy(), ()
    if set(allowlist) != invalid_sessions:
        raise OhlcvValidationError(diagnostics)

    corrected = prices.copy()
    audit: list[Mapping[str, Any]] = []
    required_keys = {
        "field",
        "observed",
        "corrected",
        "observed_row",
        "justification",
    }
    for session in sorted(invalid_sessions):
        matching = corrected.index[corrected["session"].map(_date).eq(session)].tolist()
        if len(matching) != 1:
            raise ValueError(f"FRCB allowlisted session is missing or ambiguous: {session}")
        index = matching[0]
        original = {
            field: float(pd.to_numeric(corrected.at[index, field], errors="raise"))
            for field in ("open", "high", "low", "close")
        }
        original_with_volume = {
            **original,
            "volume": float(
                pd.to_numeric(corrected.at[index, "volume"], errors="raise")
            ),
        }
        instructions = tuple(allowlist[session])
        if not instructions:
            raise ValueError(f"FRCB allowlist has no correction for {session}.")
        seen_fields: set[str] = set()
        for instruction in instructions:
            if set(instruction) != required_keys:
                raise ValueError(f"FRCB allowlist keys changed for {session}.")
            field = _text(instruction["field"])
            if field not in {"high", "low"} or field in seen_fields:
                raise ValueError(f"FRCB allowlist field is not a unique high/low: {session}.")
            seen_fields.add(field)
            observed = float(instruction["observed"])
            replacement = float(instruction["corrected"])
            observed_row = {
                name: float(value)
                for name, value in dict(instruction["observed_row"]).items()
            }
            if (
                set(observed_row) != set(original_with_volume)
                or observed_row != original_with_volume
                or observed != original[field]
                or not _text(instruction["justification"])
            ):
                raise ValueError(f"FRCB allowlist observed value changed for {session}.")
            if observed <= 0:
                peers = [
                    value
                    for name, value in original.items()
                    if name != field and value > 0
                ]
                if not peers:
                    raise ValueError(
                        f"FRCB allowlist has no positive envelope peer for {session}."
                    )
                minimal = max(peers) if field == "high" else min(peers)
            else:
                minimal = (
                    max(original.values())
                    if field == "high"
                    else min(original.values())
                )
            if replacement != minimal or replacement == observed:
                raise ValueError(
                    f"FRCB allowlist is not a minimal envelope expansion for {session}."
                )
            corrected.at[index, field] = replacement
            audit.append(
                {
                    "raw_eod_sha256": raw_eod_sha256,
                    "session": session,
                    "field": field,
                    "observed": observed,
                    "corrected": replacement,
                    "justification": _text(instruction["justification"]),
                }
            )
    remaining = invalid_ohlcv_diagnostics(corrected)
    if remaining:
        raise OhlcvValidationError(remaining)
    return corrected, tuple(audit)


def bundle_from_artifacts(
    artifacts: Iterable[SourceArtifact],
    *,
    original_http_attempts: int,
    budget_used_before: int | None = None,
    budget_used_after: int | None = None,
    budget_receipt: Mapping[str, Any] | None = None,
) -> ProviderBundle:
    by_endpoint = {
        artifact.source.removeprefix("eodhd_"): artifact for artifact in artifacts
    }
    if set(by_endpoint) != set(EODHD_ENDPOINTS):
        raise ValueError("FRCB replay lacks one of the exact three endpoint artifacts.")
    parsed = {
        endpoint: json.loads(by_endpoint[endpoint].content)
        for endpoint in EODHD_ENDPOINTS
    }
    raw_prices = _price_frame(parsed["eod"], by_endpoint["eod"])
    prices, corrections = _apply_allowlisted_envelope_corrections(
        raw_prices, raw_eod_sha256=by_endpoint["eod"].source_hash
    )
    actions = pd.concat(
        [
            _action_frame("div", parsed["div"], by_endpoint["div"]),
            _action_frame("splits", parsed["splits"], by_endpoint["splits"]),
        ],
        ignore_index=True,
        sort=False,
    )
    bundle = ProviderBundle(
        prices=prices,
        corporate_actions=actions,
        artifacts=tuple(by_endpoint[item] for item in EODHD_ENDPOINTS),
        original_http_attempts=int(original_http_attempts),
        budget_used_before=budget_used_before,
        budget_used_after=budget_used_after,
        budget_receipt=dict(budget_receipt) if budget_receipt is not None else None,
        envelope_corrections=corrections,
    )
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: ProviderBundle) -> dict[str, Any]:
    if bundle.original_http_attempts != EXPECTED_EODHD_CALLS:
        raise ValueError("FRCB bundle must prove exactly three original HTTP attempts.")
    if (
        bundle.budget_used_before is not None
        and bundle.budget_used_after is not None
        and bundle.budget_used_after - bundle.budget_used_before
        != EXPECTED_EODHD_CALLS
    ):
        raise ValueError("FRCB EODHD budget delta is not exactly three.")
    if bundle.budget_receipt is not None:
        receipt = dict(bundle.budget_receipt)
        required = {
            "schema",
            "period",
            "used_before",
            "used_after",
            "delta",
            "daily_limit",
            "reserve",
            "safety_ceiling",
        }
        if set(receipt) != required or receipt.get("schema") != "eodhd_budget_receipt/v1":
            raise ValueError("FRCB budget receipt schema changed.")
        if (
            int(receipt["used_before"]) != bundle.budget_used_before
            or int(receipt["used_after"]) != bundle.budget_used_after
            or int(receipt["delta"]) != EXPECTED_EODHD_CALLS
        ):
            raise ValueError("FRCB budget receipt does not bind the actual three calls.")
    prices = bundle.prices.copy()
    if prices.empty or prices["security_id"].astype(str).ne(FRC_SECURITY_ID).any():
        raise ValueError("FRCB bundle has no exact FRC security lineage.")
    if prices["session"].astype(str).duplicated().any():
        raise ValueError("FRCB bundle contains duplicate sessions.")
    sessions = set(prices["session"].astype(str))
    if not {FRC_TRANSITION, FRC_INDEX_EXIT}.issubset(sessions):
        raise ValueError("FRCB bundle lacks the 2023-05-03/04 tradable exit bridge.")
    if max(sessions) < "2026-06-01":
        raise ValueError("FRCB bundle does not demonstrate continuing OTC trading.")
    numeric = prices[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not numeric.map(lambda value: math.isfinite(float(value))).all().all():
        raise ValueError("FRCB bundle contains non-finite OHLCV.")
    diagnostics = invalid_ohlcv_diagnostics(prices)
    if diagnostics:
        raise OhlcvValidationError(diagnostics)
    if bundle.corporate_actions["event_id"].astype(str).duplicated().any():
        raise ValueError("FRCB provider actions contain duplicate IDs.")
    for artifact, endpoint in zip(bundle.artifacts, EODHD_ENDPOINTS, strict=True):
        if artifact.source != f"eodhd_{endpoint}":
            raise ValueError("FRCB artifact order changed.")
        if artifact.source_url != REQUEST_URLS[endpoint]:
            raise ValueError("FRCB artifact request URL changed.")
        if sha256_bytes(artifact.content) != artifact.source_hash:
            raise ValueError("FRCB artifact hash is inconsistent.")
    raw_eod = bundle.artifacts[0]
    expected_prices, expected_corrections = _apply_allowlisted_envelope_corrections(
        _price_frame(json.loads(raw_eod.content), raw_eod),
        raw_eod_sha256=raw_eod.source_hash,
    )
    try:
        pd.testing.assert_frame_equal(
            prices.sort_values("session").reset_index(drop=True),
            expected_prices.sort_values("session").reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError("FRCB prices are not an exact raw/allowlist derivation.") from exc
    if tuple(bundle.envelope_corrections) != tuple(expected_corrections):
        raise ValueError("FRCB envelope-correction audit changed.")
    return {
        "frcb_price_rows": len(prices),
        "frcb_provider_action_rows": len(bundle.corporate_actions),
        "frcb_first_session": min(sessions),
        "frcb_last_session": max(sessions),
        "frcb_index_exit_session_present": FRC_INDEX_EXIT in sessions,
        "frcb_envelope_correction_rows": len(bundle.envelope_corrections),
    }


def _bundle_signature() -> dict[str, Any]:
    return {
        "schema": "us_frcb_eodhd_bundle/v2",
        "security_id": FRC_SECURITY_ID,
        "provider_symbol": FRC_PROVIDER_SYMBOL,
        "fetch_start": FETCH_START,
        "fetch_end": FETCH_END,
        "request_urls": REQUEST_URLS,
        "expected_http_attempts": EXPECTED_EODHD_CALLS,
        "correction_allowlist_sha256": sha256_bytes(
            _canonical_json_bytes(FRCB_EOD_ENVELOPE_CORRECTION_ALLOWLIST)
        ),
    }


def bundle_cache_path(cache_root: Path) -> Path:
    digest = sha256_bytes(_canonical_json_bytes(_bundle_signature()))
    return cache_root / "state/us-frc-para-lifecycle" / f"{digest}.json.gz"


def prior_failed_attempt_note_path(cache_root: Path) -> Path:
    return (
        cache_root
        / "state/us-frc-para-lifecycle"
        / "prior-failed-attempt-20260718-8832-8835.json"
    )


def _write_prior_failed_attempt_note(cache_root: Path) -> Path:
    path = prior_failed_attempt_note_path(cache_root)
    encoded = _canonical_json_bytes(PRIOR_FAILED_ATTEMPT_NOTE)
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Immutable prior-attempt note conflict: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, encoded)
    return path


def _budget_receipt(
    budget: EodhdCallBudget, *, used_before: int, used_after: int
) -> dict[str, Any]:
    return {
        "schema": "eodhd_budget_receipt/v1",
        "period": str(budget.period),
        "used_before": int(used_before),
        "used_after": int(used_after),
        "delta": int(used_after) - int(used_before),
        "daily_limit": int(budget.limit),
        "reserve": int(budget.reserve),
        "safety_ceiling": int(budget.ceiling),
    }


def _artifact_cache_rows(artifacts: Iterable[SourceArtifact]) -> list[dict[str, Any]]:
    return [
        {
            "source": item.source,
            "source_url": item.source_url,
            "retrieved_at": item.retrieved_at,
            "content_type": item.content_type,
            "content_sha256": item.source_hash,
            "content_base64": base64.b64encode(item.content).decode("ascii"),
        }
        for item in artifacts
    ]


def _raw_quarantine_envelope(
    artifacts: Iterable[SourceArtifact], budget_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    items = tuple(artifacts)
    if len(items) != EXPECTED_EODHD_CALLS:
        raise ValueError("FRCB quarantine requires all three raw responses.")
    if int(budget_receipt.get("delta", -1)) != EXPECTED_EODHD_CALLS:
        raise ValueError("FRCB quarantine receipt delta is not exactly three.")
    return {
        "schema": "us_frcb_raw_quarantine/v1",
        "signature": _bundle_signature(),
        "validation_status": "unvalidated",
        "budget_receipt": dict(budget_receipt),
        "artifacts": _artifact_cache_rows(items),
    }


def _write_raw_quarantine(
    cache_root: Path,
    artifacts: Iterable[SourceArtifact],
    budget_receipt: Mapping[str, Any],
) -> tuple[str, Path]:
    envelope = _raw_quarantine_envelope(artifacts, budget_receipt)
    content = _canonical_json_bytes(envelope)
    quarantine_id = sha256_bytes(content)
    path = (
        cache_root
        / "state/us-frc-para-lifecycle/quarantine"
        / f"{quarantine_id}.json.gz"
    )
    encoded = gzip.compress(content, mtime=0)
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Immutable FRCB quarantine hash collision: {path}")
        return quarantine_id, path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, encoded)
    return quarantine_id, path


def raw_quarantine_path(cache_root: Path, quarantine_id: str) -> Path:
    normalized = _text(quarantine_id).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("FRCB quarantine id must be an exact lowercase SHA-256.")
    return (
        cache_root
        / "state/us-frc-para-lifecycle/quarantine"
        / f"{normalized}.json.gz"
    )


def _validate_budget_receipt_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    required = {
        "schema",
        "period",
        "used_before",
        "used_after",
        "delta",
        "daily_limit",
        "reserve",
        "safety_ceiling",
    }
    if set(receipt) != required or receipt.get("schema") != "eodhd_budget_receipt/v1":
        raise ValueError("FRCB quarantine budget receipt schema changed.")
    before = int(receipt["used_before"])
    after = int(receipt["used_after"])
    if after - before != EXPECTED_EODHD_CALLS or int(receipt["delta"]) != EXPECTED_EODHD_CALLS:
        raise ValueError("FRCB quarantine budget receipt is not an exact three-call claim.")
    if int(receipt["safety_ceiling"]) != int(receipt["daily_limit"]) - int(
        receipt["reserve"]
    ):
        raise ValueError("FRCB quarantine budget safety ceiling changed.")
    return receipt


def _validate_acquisition_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(value)
    expected = _bundle_signature()
    if set(observed) != set(expected):
        raise ValueError("FRCB quarantine acquisition signature schema changed.")
    for key in set(expected) - {"correction_allowlist_sha256"}:
        if observed.get(key) != expected[key]:
            raise ValueError(f"FRCB quarantine acquisition signature changed: {key}.")
    accepted_policy_hashes = {
        sha256_bytes(_canonical_json_bytes({})),
        expected["correction_allowlist_sha256"],
    }
    if observed.get("correction_allowlist_sha256") not in accepted_policy_hashes:
        raise ValueError("FRCB quarantine has an unknown correction-policy hash.")
    return observed


def read_raw_quarantine(cache_root: Path, quarantine_id: str) -> RawQuarantine:
    path = raw_quarantine_path(cache_root, quarantine_id)
    if not path.is_file():
        raise FileNotFoundError(f"FRCB quarantine does not exist: {path}")
    try:
        content = gzip.decompress(path.read_bytes())
        envelope = json.loads(content)
    except Exception as exc:
        raise ValueError(f"FRCB quarantine is unreadable: {path}") from exc
    if content != _canonical_json_bytes(envelope):
        raise ValueError("FRCB quarantine wrapper is not canonical JSON.")
    if sha256_bytes(content) != _text(quarantine_id).lower():
        raise ValueError("FRCB quarantine content-address hash mismatch.")
    if set(envelope) != {
        "schema",
        "signature",
        "validation_status",
        "budget_receipt",
        "artifacts",
    }:
        raise ValueError("FRCB quarantine wrapper fields changed.")
    if (
        envelope.get("schema") != "us_frcb_raw_quarantine/v1"
        or envelope.get("validation_status") != "unvalidated"
    ):
        raise ValueError("FRCB quarantine wrapper schema/status changed.")
    signature = _validate_acquisition_signature(envelope["signature"])
    receipt = _validate_budget_receipt_dict(envelope["budget_receipt"])
    items = envelope["artifacts"]
    if not isinstance(items, list) or len(items) != EXPECTED_EODHD_CALLS:
        raise ValueError("FRCB quarantine does not contain exactly three raw responses.")
    artifacts: list[SourceArtifact] = []
    required_item_keys = {
        "source",
        "source_url",
        "retrieved_at",
        "content_type",
        "content_sha256",
        "content_base64",
    }
    for item, endpoint in zip(items, EODHD_ENDPOINTS, strict=True):
        if set(item) != required_item_keys:
            raise ValueError("FRCB quarantine artifact fields changed.")
        if (
            item.get("source") != f"eodhd_{endpoint}"
            or item.get("source_url") != REQUEST_URLS[endpoint]
        ):
            raise ValueError("FRCB quarantine endpoint identity/order changed.")
        try:
            raw = base64.b64decode(item["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError("FRCB quarantine raw body is not valid base64.") from exc
        if sha256_bytes(raw) != item.get("content_sha256"):
            raise ValueError("FRCB quarantine raw body hash mismatch.")
        artifacts.append(
            SourceArtifact(
                source=str(item["source"]),
                source_url=str(item["source_url"]),
                retrieved_at=str(item["retrieved_at"]),
                content=raw,
                content_type=str(item["content_type"]),
            )
        )
    return RawQuarantine(
        quarantine_id=_text(quarantine_id).lower(),
        path=path,
        artifacts=tuple(artifacts),
        budget_receipt=receipt,
        acquisition_signature=signature,
    )


def promote_raw_quarantine(
    cache_root: Path, quarantine_id: str
) -> tuple[ProviderBundle, RawQuarantine, Path]:
    quarantine = read_raw_quarantine(cache_root, quarantine_id)
    receipt = quarantine.budget_receipt
    bundle = bundle_from_artifacts(
        quarantine.artifacts,
        original_http_attempts=EXPECTED_EODHD_CALLS,
        budget_used_before=int(receipt["used_before"]),
        budget_used_after=int(receipt["used_after"]),
        budget_receipt=receipt,
    )
    path = bundle_cache_path(cache_root)
    _write_bundle_cache(path, bundle)
    return bundle, quarantine, path


def _write_validation_diagnostic(
    cache_root: Path,
    *,
    quarantine_id: str,
    artifacts: Iterable[SourceArtifact],
    error: Exception,
) -> Path:
    diagnostics = (
        list(error.diagnostics) if isinstance(error, OhlcvValidationError) else []
    )
    report = {
        "schema": "us_frcb_quarantine_diagnostic/v1",
        "quarantine_id": quarantine_id,
        "validation_status": "failed",
        "error_type": type(error).__name__,
        "invalid_ohlcv_rows": diagnostics,
        "raw_artifact_hashes": {
            item.source: item.source_hash for item in artifacts
        },
    }
    content = _canonical_json_bytes(report)
    report_hash = sha256_bytes(content)
    path = (
        cache_root
        / "state/us-frc-para-lifecycle/quarantine"
        / f"{quarantine_id}.{report_hash}.validation.json"
    )
    if path.is_file():
        if path.read_bytes() != content:
            raise RuntimeError(f"Immutable FRCB diagnostic hash collision: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, content)
    return path


def _write_bundle_cache(path: Path, bundle: ProviderBundle) -> None:
    validate_bundle(bundle)
    payload = {
        "signature": _bundle_signature(),
        "original_http_attempts": bundle.original_http_attempts,
        "budget_used_before": bundle.budget_used_before,
        "budget_used_after": bundle.budget_used_after,
        "budget_receipt": dict(bundle.budget_receipt or {}),
        "envelope_corrections": [dict(item) for item in bundle.envelope_corrections],
        "artifacts": _artifact_cache_rows(bundle.artifacts),
    }
    envelope = {
        "schema": "us_frcb_validated_bundle_cache/v1",
        "payload": payload,
        "payload_sha256": sha256_bytes(_canonical_json_bytes(payload)),
    }
    encoded = gzip.compress(_canonical_json_bytes(envelope), mtime=0)
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Immutable FRCB bundle cache conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, encoded)


def _read_bundle_cache(path: Path) -> ProviderBundle | None:
    if not path.is_file():
        return None
    try:
        content = gzip.decompress(path.read_bytes())
        envelope = json.loads(content)
    except Exception as exc:
        raise ValueError(f"FRCB bundle cache is unreadable: {path}") from exc
    if content != _canonical_json_bytes(envelope):
        raise ValueError("FRCB bundle cache is not canonical JSON.")
    if set(envelope) != {"schema", "payload", "payload_sha256"} or envelope.get(
        "schema"
    ) != "us_frcb_validated_bundle_cache/v1":
        raise ValueError("FRCB bundle cache wrapper schema changed.")
    value = envelope["payload"]
    if envelope.get("payload_sha256") != sha256_bytes(_canonical_json_bytes(value)):
        raise ValueError("FRCB bundle cache payload hash mismatch.")
    if value.get("signature") != _bundle_signature():
        raise ValueError("FRCB bundle signature changed.")
    artifacts: list[SourceArtifact] = []
    for item in value.get("artifacts", []):
        content = base64.b64decode(item["content_base64"], validate=True)
        if sha256_bytes(content) != item.get("content_sha256"):
            raise ValueError("FRCB cached artifact hash mismatch.")
        artifacts.append(
            SourceArtifact(
                source=str(item["source"]),
                source_url=str(item["source_url"]),
                retrieved_at=str(item["retrieved_at"]),
                content=content,
                content_type=str(item["content_type"]),
            )
        )
    bundle = bundle_from_artifacts(
        artifacts,
        original_http_attempts=int(value.get("original_http_attempts", 0)),
        budget_used_before=value.get("budget_used_before"),
        budget_used_after=value.get("budget_used_after"),
        budget_receipt=value.get("budget_receipt") or None,
    )
    if value.get("envelope_corrections", []) != [
        dict(item) for item in bundle.envelope_corrections
    ]:
        raise ValueError("FRCB cached correction metadata changed.")
    return bundle


def collect_bundle(
    cache_root: Path,
    *,
    client_factory: Callable[..., ExactThreeFrcbEodhdClient] = ExactThreeFrcbEodhdClient,
    budget_factory: Callable[[], EodhdCallBudget] = EodhdCallBudget,
) -> ProviderBundle:
    load_env()
    _write_prior_failed_attempt_note(cache_root)
    budget = budget_factory()
    before = _budget_used(budget)
    client = client_factory(budget=budget)
    artifacts: list[SourceArtifact] = []
    retrieved_at = utc_now_iso()
    for endpoint in EODHD_ENDPOINTS:
        artifacts.append(
            client.get_raw_artifact(
                f"{endpoint}/{FRC_PROVIDER_SYMBOL}",
                params=REQUEST_PARAMS,
                retrieved_at=retrieved_at,
            )
        )
    if len(client.attempted_endpoints) != EXPECTED_EODHD_CALLS:
        raise RuntimeError("FRCB collector did not make exactly three requests.")
    after = _budget_used(budget)
    receipt = _budget_receipt(budget, used_before=before, used_after=after)
    quarantine_id, quarantine_path = _write_raw_quarantine(
        cache_root, artifacts, receipt
    )
    try:
        bundle = bundle_from_artifacts(
            artifacts,
            original_http_attempts=len(client.attempted_endpoints),
            budget_used_before=before,
            budget_used_after=after,
            budget_receipt=receipt,
        )
    except Exception as exc:
        diagnostic_path = _write_validation_diagnostic(
            cache_root,
            quarantine_id=quarantine_id,
            artifacts=artifacts,
            error=exc,
        )
        if isinstance(exc, OhlcvValidationError):
            raise OhlcvValidationError(
                exc.diagnostics,
                quarantine_path=quarantine_path,
                diagnostic_path=diagnostic_path,
            ) from exc
        if hasattr(exc, "add_note"):
            exc.add_note(
                f"Raw FRCB responses preserved at {quarantine_path}; "
                f"diagnostic={diagnostic_path}."
            )
        raise
    _write_bundle_cache(bundle_cache_path(cache_root), bundle)
    return bundle


def _official_action_rows(occ: SourceArtifact) -> pd.DataFrame:
    rows = [
        {
            "event_id": FRC_EVENT_ID,
            "security_id": FRC_SECURITY_ID,
            "action_type": "ticker_change",
            "effective_date": FRC_TRANSITION,
            "ex_date": FRC_TRANSITION,
            "announcement_date": "2023-05-02",
            "record_date": "",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "new_security_id": FRC_SECURITY_ID,
            "new_symbol": FRC_NEW_SYMBOL,
            "official": True,
            "source_url": OCC_MEMO_URL,
            "source_kind": "clearing_notice_reviewed_extraction",
            "source": occ.source,
            "retrieved_at": occ.retrieved_at,
            "source_hash": occ.source_hash,
            "metadata": json.dumps(
                {"memo_number": OCC_MEMO_NUMBER, "cusip": FRC_CUSIP},
                sort_keys=True,
            ),
        },
        {
            "event_id": PARA_EVENT_ID,
            "security_id": PARA_SECURITY_ID,
            "action_type": "stock_merger",
            "effective_date": PARA_TRANSITION,
            "ex_date": PARA_TRANSITION,
            "announcement_date": "2025-08-07",
            "record_date": "",
            "payment_date": "",
            "cash_amount": None,
            "ratio": 1.0,
            "currency": "USD",
            "new_security_id": PSKY_SECURITY_ID,
            "new_symbol": PSKY_SYMBOL,
            "official": True,
            "source_url": PARA_SEC_URL,
            "source_kind": "sec_filing_default_stock_policy",
            "source": "sec_edgar_filing",
            "retrieved_at": "2026-07-18T08:13:18.114738Z",
            "source_hash": PARA_SEC_SHA256,
            "metadata": json.dumps(
                {
                    "backtest_policy": "no_election_default_stock",
                    "cash_elector_proration_excluded": True,
                },
                sort_keys=True,
            ),
        },
    ]
    return pd.DataFrame(
        rows,
        columns=tuple(
            dict.fromkeys((*dataset_spec("corporate_actions").required_columns, "metadata"))
        ),
    )


def _history_rows(occ: SourceArtifact) -> pd.DataFrame:
    columns = dataset_spec("symbol_history").required_columns
    rows = [
        {
            "security_id": FRC_SECURITY_ID,
            "symbol": FRC_OLD_SYMBOL,
            "exchange": "NYSE",
            "effective_from": "2015-01-01",
            "effective_to": FRC_OLD_LAST,
            "source": occ.source,
            "retrieved_at": occ.retrieved_at,
            "source_hash": occ.source_hash,
            "source_url": OCC_MEMO_URL,
        },
        {
            "security_id": FRC_SECURITY_ID,
            "symbol": FRC_NEW_SYMBOL,
            "exchange": "PINK",
            "effective_from": FRC_TRANSITION,
            "effective_to": "",
            "source": occ.source,
            "retrieved_at": occ.retrieved_at,
            "source_hash": occ.source_hash,
            "source_url": OCC_MEMO_URL,
        },
        {
            "security_id": PARA_SECURITY_ID,
            "symbol": PARA_SYMBOL,
            "exchange": "NASDAQ",
            "effective_from": "2022-02-17",
            # The legal identity remains valid on the merger effective date.
            # There is intentionally no fabricated PARA price for this date;
            # keeping the symbol resolvable merely closes the schedule boundary
            # until the 1:1 PSKY successor action is replayed.
            "effective_to": PARA_TRANSITION,
            "source": "sec_edgar_filing",
            "retrieved_at": "2026-07-18T08:13:18.114738Z",
            "source_hash": PARA_SEC_SHA256,
            "source_url": PARA_SEC_URL,
        },
        {
            "security_id": PSKY_SECURITY_ID,
            "symbol": PSKY_SYMBOL,
            "exchange": "NASDAQ",
            "effective_from": PARA_TRANSITION,
            "effective_to": "",
            "source": "sec_edgar_filing",
            "retrieved_at": "2026-07-18T08:13:18.114738Z",
            "source_hash": PARA_SEC_SHA256,
            "source_url": PARA_SEC_URL,
        },
    ]
    return pd.DataFrame(rows, columns=tuple(dict.fromkeys((*columns, "source_url"))))


def _archive_id(artifact: SourceArtifact) -> str:
    # Empty dividend and split payloads are both ``[]`` and therefore share a
    # content hash.  Bind the archive primary key to endpoint identity while
    # retaining the true content hash in ``source_hash`` and object_path.
    return sha256_bytes(f"{artifact.source}|{artifact.source_hash}".encode("utf-8"))


def _archive_row(artifact: SourceArtifact, completed_session: str) -> dict[str, Any]:
    suffix = "json" if artifact.content_type == "application/json" else "bin"
    return {
        "archive_id": _archive_id(artifact),
        "dataset": artifact.source,
        "object_path": (
            f"archives/{completed_session}/{artifact.source_hash}.{suffix}.gz"
        ),
        "content_type": artifact.content_type,
        "effective_date": completed_session,
        "source": artifact.source,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
        "source_url": artifact.source_url,
    }


def _correction_metadata_artifact(bundle: ProviderBundle) -> SourceArtifact | None:
    if not bundle.envelope_corrections:
        return None
    raw_eod = bundle.artifacts[0]
    return SourceArtifact(
        source="frcb_reviewed_ohlcv_envelope_correction",
        source_url=raw_eod.source_url,
        retrieved_at=raw_eod.retrieved_at,
        content=_canonical_json_bytes(
            {
                "schema": "frcb_reviewed_ohlcv_envelope_correction/v1",
                "provider_symbol": FRC_PROVIDER_SYMBOL,
                "raw_eod_sha256": raw_eod.source_hash,
                "correction_policy_sha256": _bundle_signature()[
                    "correction_allowlist_sha256"
                ],
                "corrections": [dict(item) for item in bundle.envelope_corrections],
                "unchanged_fields": ["open", "high", "close", "volume"],
                "review_scope": "exact raw hash, exact session, exact observed row",
            }
        ),
        content_type="application/json",
    )


def _correction_release_warnings(bundle: ProviderBundle) -> tuple[str, ...]:
    if not bundle.envelope_corrections:
        return ()
    raw_hash = bundle.artifacts[0].source_hash
    details = "; ".join(
        f"{item['session']} {item['field']}={item['observed']}->{item['corrected']}"
        for item in bundle.envelope_corrections
    )
    return (
        "FRCB EODHD raw OHLCV required an exact-hash reviewed envelope correction "
        f"({details}; raw_eod_sha256={raw_hash}); all other fields remain unchanged, "
        "so release quality remains degraded.",
    )


def _set_master_identity(master: pd.DataFrame, occ: SourceArtifact) -> pd.DataFrame:
    output = master.copy()
    for security_id in (FRC_SECURITY_ID, PARA_SECURITY_ID, PSKY_SECURITY_ID):
        if int(output["security_id"].astype(str).eq(security_id).sum()) != 1:
            raise ValueError(f"Lifecycle repair master identity is missing: {security_id}")
    frc = output["security_id"].astype(str).eq(FRC_SECURITY_ID)
    output.loc[frc, "primary_symbol"] = FRC_NEW_SYMBOL
    output.loc[frc, "provider_symbol"] = FRC_PROVIDER_SYMBOL
    if "action_provider_symbol" in output:
        output.loc[frc, "action_provider_symbol"] = FRC_PROVIDER_SYMBOL
    output.loc[frc, "exchange"] = "PINK"
    output.loc[frc, "active_to"] = ""
    output.loc[frc, "source"] = occ.source
    output.loc[frc, "retrieved_at"] = occ.retrieved_at
    output.loc[frc, "source_hash"] = occ.source_hash
    if "source_url" in output:
        output.loc[frc, "source_url"] = OCC_MEMO_URL

    para = output["security_id"].astype(str).eq(PARA_SECURITY_ID)
    output.loc[para, "active_to"] = PARA_TRANSITION
    psky = output["security_id"].astype(str).eq(PSKY_SECURITY_ID)
    output.loc[psky, "primary_symbol"] = PSKY_SYMBOL
    output.loc[psky, "provider_symbol"] = "PSKY.US"
    if "action_provider_symbol" in output:
        output.loc[psky, "action_provider_symbol"] = "PSKY.US"
    output.loc[psky, "exchange"] = "NASDAQ"
    output.loc[psky, "active_from"] = PARA_TRANSITION
    output.loc[psky, "active_to"] = ""
    output.loc[psky, "source"] = "sec_edgar_filing"
    output.loc[psky, "retrieved_at"] = "2026-07-18T08:13:18.114738Z"
    output.loc[psky, "source_hash"] = PARA_SEC_SHA256
    if "source_url" in output:
        output.loc[psky, "source_url"] = PARA_SEC_URL
    return output


def _update_resolution(
    resolutions: pd.DataFrame,
    *,
    security_id: str,
    last_price_date: str,
    event_id: str,
    successor_id: str,
    successor_symbol: str,
    source: str,
    source_url: str,
    source_hash: str,
    retrieved_at: str,
) -> pd.DataFrame:
    output = resolutions.copy()
    mask = (
        output["security_id"].astype(str).eq(security_id)
        & output["last_price_date"].astype(str).eq(last_price_date)
    )
    if int(mask.sum()) != 1:
        raise ValueError(f"Lifecycle resolution is missing or ambiguous: {security_id}")
    output.loc[mask, "resolution"] = "applied"
    output.loc[mask, "event_id"] = event_id
    output.loc[mask, "exception_code"] = ""
    output.loc[mask, "exception_reason"] = ""
    output.loc[mask, "reviewed_by"] = "us_frc_para_lifecycle_repair_v1"
    output.loc[mask, "reviewed_at"] = "2026-07-18T00:00:00Z"
    output.loc[mask, "recheck_after"] = ""
    output.loc[mask, "successor_security_id"] = successor_id
    output.loc[mask, "successor_symbol"] = successor_symbol
    output.loc[mask, "source_url"] = source_url
    output.loc[mask, "source"] = source
    output.loc[mask, "retrieved_at"] = retrieved_at
    output.loc[mask, "source_hash"] = source_hash
    return output


def prepare_frames(
    existing: Mapping[str, pd.DataFrame],
    bundle: ProviderBundle,
    *,
    completed_session: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], tuple[SourceArtifact, ...]]:
    validate_bundle(bundle)
    occ = _occ_artifact()
    master = _set_master_identity(existing["security_master"], occ)

    history = existing["symbol_history"].copy()
    replace_history_ids = {FRC_SECURITY_ID, PARA_SECURITY_ID, PSKY_SECURITY_ID}
    history = history.loc[
        ~history["security_id"].astype(str).isin(replace_history_ids)
    ]
    history = pd.concat([history, _history_rows(occ)], ignore_index=True, sort=False)

    prices = existing["daily_price_raw"].copy()
    remove_frc_tail = (
        prices["security_id"].astype(str).eq(FRC_SECURITY_ID)
        & prices["session"].astype(str).ge(FRC_TRANSITION)
    )
    remove_para_tail = (
        prices["security_id"].astype(str).eq(PARA_SECURITY_ID)
        & prices["session"].astype(str).gt(PARA_LAST)
    )
    remove_psky_predecessor = (
        prices["security_id"].astype(str).eq(PSKY_SECURITY_ID)
        & prices["session"].astype(str).lt(PARA_TRANSITION)
    )
    prices = prices.loc[
        ~(remove_frc_tail | remove_para_tail | remove_psky_predecessor)
    ]
    prices = pd.concat([prices, bundle.prices], ignore_index=True, sort=False)

    actions = existing["corporate_actions"].copy()
    action_dates = actions["effective_date"].astype(str)
    remove_frc_tail_actions = (
        actions["security_id"].astype(str).eq(FRC_SECURITY_ID)
        & action_dates.ge(FRC_TRANSITION)
    )
    remove_para_terminal = (
        actions["security_id"].astype(str).eq(PARA_SECURITY_ID)
        & action_dates.ge(PARA_TRANSITION)
        & actions["action_type"].astype(str).isin(
            {"stock_merger", "cash_merger", "ticker_change", "delisting"}
        )
    )
    remove_psky_predecessor_actions = (
        actions["security_id"].astype(str).eq(PSKY_SECURITY_ID)
        & action_dates.lt(PARA_TRANSITION)
    )
    actions = actions.loc[
        ~(
            remove_frc_tail_actions
            | remove_para_terminal
            | remove_psky_predecessor_actions
            | actions["event_id"].astype(str).isin({FRC_EVENT_ID, PARA_EVENT_ID})
        )
    ]
    actions = pd.concat(
        [actions, bundle.corporate_actions, _official_action_rows(occ)],
        ignore_index=True,
        sort=False,
    )

    resolutions = _update_resolution(
        existing["lifecycle_resolutions"],
        security_id=FRC_SECURITY_ID,
        last_price_date=FRC_OLD_LAST,
        event_id=FRC_EVENT_ID,
        successor_id=FRC_SECURITY_ID,
        successor_symbol=FRC_NEW_SYMBOL,
        source=occ.source,
        source_url=OCC_MEMO_URL,
        source_hash=occ.source_hash,
        retrieved_at=occ.retrieved_at,
    )
    resolutions = _update_resolution(
        resolutions,
        security_id=PARA_SECURITY_ID,
        last_price_date=PARA_LAST,
        event_id=PARA_EVENT_ID,
        successor_id=PSKY_SECURITY_ID,
        successor_symbol=PSKY_SYMBOL,
        source="sec_edgar_filing",
        source_url=PARA_SEC_URL,
        source_hash=PARA_SEC_SHA256,
        retrieved_at="2026-07-18T08:13:18.114738Z",
    )

    factors = existing["adjustment_factors"].copy()
    touched = {FRC_SECURITY_ID, PARA_SECURITY_ID, PSKY_SECURITY_ID}
    factors = factors.loc[~factors["security_id"].astype(str).isin(touched)]
    touched_prices = prices.loc[prices["security_id"].astype(str).isin(touched)]
    touched_actions = actions.loc[actions["security_id"].astype(str).isin(touched)]
    rebuilt = build_adjustment_factors(
        touched_prices,
        touched_actions,
        source_version=(
            "repair_us_frc_para_lifecycle/"
            f"{completed_session}/{sha256_bytes(_canonical_json_bytes(_bundle_signature()))}"
        ),
    )
    factors = pd.concat([factors, rebuilt], ignore_index=True, sort=False)

    correction_artifact = _correction_metadata_artifact(bundle)
    artifacts = (
        *bundle.artifacts,
        occ,
        *((correction_artifact,) if correction_artifact is not None else ()),
    )
    archive = existing["source_archive"].copy()
    archive_ids = {_archive_id(item) for item in artifacts}
    archive = archive.loc[~archive["archive_id"].astype(str).isin(archive_ids)]
    archive = pd.concat(
        [
            archive,
            pd.DataFrame(
                [_archive_row(item, completed_session) for item in artifacts],
                columns=tuple(
                    dict.fromkeys(
                        (*dataset_spec("source_archive").required_columns, "source_url")
                    )
                ),
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    frames = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "adjustment_factors": factors,
        "source_archive": archive,
    }
    for dataset, frame in frames.items():
        validate_dataset(
            dataset,
            frame,
            incomplete_action_policy="warn",
            completed_session=completed_session,
        ).raise_for_errors()
    if not identity_is_repaired(frames):
        raise RuntimeError("Prepared FRC/PARA snapshot failed its exact invariant.")

    summary = {
        "status": "validated_dry_run",
        "network_accessed_this_run": False,
        "eodhd_http_attempts_this_run": 0,
        "eodhd_original_bundle_attempts": bundle.original_http_attempts,
        "eodhd_expected_calls": EXPECTED_EODHD_CALLS,
        "eodhd_provider_symbols": [FRC_PROVIDER_SYMBOL],
        "psky_eodhd_calls": 0,
        **validate_bundle(bundle),
        "frc_policy": "same_security_ticker_change_then_exit_only",
        "frc_final_recovery_modeled": False,
        "para_policy": "no_election_default_stock_one_for_one",
        "para_cash_elector_proration_modeled": False,
        "official_action_event_ids": [FRC_EVENT_ID, PARA_EVENT_ID],
        "frcb_raw_eod_sha256": bundle.artifacts[0].source_hash,
        "frcb_envelope_corrections": [
            dict(item) for item in bundle.envelope_corrections
        ],
        "frcb_correction_metadata_sha256": (
            correction_artifact.source_hash if correction_artifact is not None else None
        ),
        "correction_warnings": list(_correction_release_warnings(bundle)),
        "psky_pre_transition_price_rows_removed": int(
            remove_psky_predecessor.sum()
        ),
        "psky_pre_transition_action_rows_removed": int(
            remove_psky_predecessor_actions.sum()
        ),
    }
    return frames, summary, tuple(artifacts)


def _one_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series | None:
    rows = frame.loc[frame[column].astype(str).eq(value)]
    return rows.iloc[0] if len(rows) == 1 else None


def identity_is_repaired(frames: Mapping[str, pd.DataFrame]) -> bool:
    try:
        master = frames["security_master"]
        history = frames["symbol_history"]
        prices = frames["daily_price_raw"]
        actions = frames["corporate_actions"]
        resolutions = frames["lifecycle_resolutions"]
        factors = frames["adjustment_factors"]
        frc = _one_row(master, "security_id", FRC_SECURITY_ID)
        para = _one_row(master, "security_id", PARA_SECURITY_ID)
        psky = _one_row(master, "security_id", PSKY_SECURITY_ID)
        if frc is None or para is None or psky is None:
            return False
        if not (
            _text(frc.get("primary_symbol")) == FRC_NEW_SYMBOL
            and _text(frc.get("provider_symbol")) == FRC_PROVIDER_SYMBOL
            and _text(frc.get("active_to")) == ""
            and _date(para.get("active_to")) == PARA_TRANSITION
            and _date(psky.get("active_from")) == PARA_TRANSITION
        ):
            return False
        exact_history = {
            (FRC_SECURITY_ID, FRC_OLD_SYMBOL, "2015-01-01", FRC_OLD_LAST),
            (FRC_SECURITY_ID, FRC_NEW_SYMBOL, FRC_TRANSITION, ""),
            (PARA_SECURITY_ID, PARA_SYMBOL, "2022-02-17", PARA_TRANSITION),
            (PSKY_SECURITY_ID, PSKY_SYMBOL, PARA_TRANSITION, ""),
        }
        observed_history = {
            (
                _text(row.security_id),
                _text(row.symbol),
                _date(row.effective_from),
                _date(row.effective_to),
            )
            for row in history.loc[
                history["security_id"].astype(str).isin(
                    {FRC_SECURITY_ID, PARA_SECURITY_ID, PSKY_SECURITY_ID}
                )
            ].itertuples(index=False)
        }
        if observed_history != exact_history:
            return False
        frc_sessions = set(
            prices.loc[
                prices["security_id"].astype(str).eq(FRC_SECURITY_ID), "session"
            ].astype(str)
        )
        if not {FRC_TRANSITION, FRC_INDEX_EXIT}.issubset(frc_sessions):
            return False
        if (
            prices["security_id"].astype(str).eq(PSKY_SECURITY_ID)
            & prices["session"].astype(str).lt(PARA_TRANSITION)
        ).any():
            return False
        for event_id, action_type, ratio, successor in (
            (FRC_EVENT_ID, "ticker_change", None, FRC_SECURITY_ID),
            (PARA_EVENT_ID, "stock_merger", 1.0, PSKY_SECURITY_ID),
        ):
            row = _one_row(actions, "event_id", event_id)
            if row is None or _text(row.get("action_type")) != action_type:
                return False
            if _text(row.get("new_security_id")) != successor or not bool(
                row.get("official")
            ):
                return False
            if ratio is not None and abs(float(row.get("ratio")) - ratio) > 1e-12:
                return False
        for security_id, event_id, successor in (
            (FRC_SECURITY_ID, FRC_EVENT_ID, FRC_SECURITY_ID),
            (PARA_SECURITY_ID, PARA_EVENT_ID, PSKY_SECURITY_ID),
        ):
            rows = resolutions.loc[
                resolutions["security_id"].astype(str).eq(security_id)
            ]
            applied = rows.loc[
                rows["resolution"].astype(str).eq("applied")
                & rows["event_id"].astype(str).eq(event_id)
                & rows["successor_security_id"].astype(str).eq(successor)
            ]
            if len(applied) != 1:
                return False
        price_keys = {
            (str(row.security_id), _date(row.session))
            for row in prices.loc[
                prices["security_id"].astype(str).isin(
                    {FRC_SECURITY_ID, PARA_SECURITY_ID, PSKY_SECURITY_ID}
                )
            ].itertuples(index=False)
        }
        factor_keys = {
            (str(row.security_id), _date(row.session))
            for row in factors.loc[
                factors["security_id"].astype(str).isin(
                    {FRC_SECURITY_ID, PARA_SECURITY_ID, PSKY_SECURITY_ID}
                )
            ].itertuples(index=False)
        }
        return price_keys == factor_keys
    except (KeyError, TypeError, ValueError):
        return False


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        overrides: Mapping[str, pd.DataFrame],
    ):
        self.base = base
        self.versions = dict(versions)
        self.overrides = dict(overrides)

    def current_manifest(self, dataset: str):
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        version = self.versions.get(dataset)
        if not version:
            return pd.DataFrame(columns=dataset_spec(dataset).required_columns)
        return self.base.read_frame(dataset, version)


def _capture_pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"FRC/PARA release/pointer mismatch: {dataset}")
        output[dataset] = etag
    return output


def prepare_run(
    repository: LocalDatasetRepository,
    *,
    allow_fetch: bool,
    collector: Callable[[Path], ProviderBundle] | None = None,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required for FRC/PARA repair.")
    missing = sorted(set(WRITE_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    existing = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in WRITE_DATASETS
    }
    pointer_etags = _capture_pointer_etags(repository, release)
    if identity_is_repaired(existing):
        validate_repository_snapshot(repository).raise_for_errors()
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            frames=existing,
            archive_artifacts=(),
            warnings=release.warnings,
            summary={
                "status": "already_repaired",
                "release_version": release.version,
                "network_accessed_this_run": False,
                "eodhd_http_attempts_this_run": 0,
            },
        )
    _assert_reviewed_local_evidence(repository, existing["source_archive"])
    cache_path = bundle_cache_path(repository.root)
    bundle = _read_bundle_cache(cache_path)
    fetched_now = False
    if bundle is None:
        if not allow_fetch:
            raise RuntimeError(
                "FRCB EODHD bundle is missing; requirements plan calls for exactly "
                "three requests to FRCB.US (eod/div/splits)."
            )
        bundle = (collector or (lambda root: collect_bundle(root)))(repository.root)
        fetched_now = True
    frames, summary, artifacts = prepare_frames(
        existing, bundle, completed_session=release.completed_session
    )
    warnings = tuple(
        dict.fromkeys((*release.warnings, *_correction_release_warnings(bundle)))
    )
    candidate = _CandidateRepository(repository, release.dataset_versions, frames)
    validate_repository_snapshot(candidate).raise_for_errors()
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        frames=frames,
        archive_artifacts=artifacts,
        warnings=warnings,
        summary={
            **summary,
            "release_version": release.version,
            "network_accessed_this_run": fetched_now,
            "eodhd_http_attempts_this_run": EXPECTED_EODHD_CALLS if fetched_now else 0,
            "bundle_cache_path": str(cache_path),
            "quality": DataQuality.DEGRADED.value if warnings else DataQuality.VALID.value,
            "warnings": list(warnings),
        },
    )


def _persist_artifacts(
    repository: LocalDatasetRepository,
    artifacts: Iterable[SourceArtifact],
    completed_session: str,
) -> None:
    for artifact in artifacts:
        suffix = "json" if artifact.content_type == "application/json" else "bin"
        path = (
            repository.root
            / f"archives/{completed_session}/{artifact.source_hash}.{suffix}.gz"
        )
        encoded = gzip.compress(artifact.content, mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            if gzip.decompress(path.read_bytes()) != artifact.content:
                raise RuntimeError(f"Immutable lifecycle archive changed: {path}")
        else:
            write_atomic(path, encoded)


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / "recovery"
        if recovery.exists() and tuple(recovery.rglob("*.json")):
            raise RuntimeError("A recovery marker blocks FRC/PARA writes.")
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    write_atomic(path, _canonical_json_bytes(dict(value)))


def _restore(
    repository: LocalDatasetRepository,
    *,
    old_release: bytes,
    old_pointers: Mapping[str, bytes],
    planned: Mapping[str, str],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release:
            repository.objects.put("releases/current.json", old_release, if_match=current.etag)
    except Exception as exc:
        errors.append(f"release: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            if current.data != old_pointers[dataset]:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned[dataset]:
                    raise RuntimeError(f"unexpected pointer {pointer.version}")
                repository.objects.put(key, old_pointers[dataset], if_match=current.etag)
        except Exception as exc:
            errors.append(f"{dataset}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_repaired":
        return dict(prepared.summary)
    if prepared.summary.get("network_accessed_this_run"):
        raise RuntimeError(
            "Fetch and apply must be separate invocations; replay the cached bundle offline."
        )
    with _exclusive_repository_lock(repository):
        current, current_etag = repository.current_release()
        if (
            current is None
            or current.version != prepared.release.version
            or current_etag != prepared.release_etag
        ):
            raise RuntimeError("Current release changed after FRC/PARA preflight.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"FRC/PARA pointer changed before apply: {dataset}")
            old_pointers[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        session = prepared.release.completed_session.replace("-", "")
        planned = {
            dataset: f"frc-para-lifecycle-{session}-{transaction_id}-{dataset}"
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/frc-para-lifecycle-repair"
            / f"{transaction_id}.json"
        )
        journal = {
            "schema": "frc_para_lifecycle_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value).decode("ascii")
                for dataset, value in old_pointers.items()
            },
            "planned_versions": planned,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        try:
            _persist_artifacts(
                repository,
                prepared.archive_artifacts,
                prepared.release.completed_session,
            )
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                # Direct lifecycle repairs must not sever the cryptographic
                # binding installed by the SEC finalizer.  Start from the
                # current manifest metadata, then add this repair's audit
                # fields.  In particular this preserves
                # ``evidence_report_sha256`` for the publication and
                # cross-validation gates.
                manifest_metadata = dict(
                    repository.manifest_for_version(
                        dataset, prepared.release.dataset_versions[dataset]
                    ).metadata
                )
                manifest_metadata.update(
                    {
                        "operation": "repair_us_frc_para_lifecycle",
                        "frc_policy": "same_security_ticker_change_exit_only",
                        "para_policy": "no_election_default_stock_one_for_one",
                        "eodhd_http_attempts_this_run": 0,
                        "r2_accessed": False,
                        "frcb_raw_eod_sha256": prepared.summary.get(
                            "frcb_raw_eod_sha256"
                        ),
                        "frcb_envelope_corrections": prepared.summary.get(
                            "frcb_envelope_corrections", []
                        ),
                    }
                )
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="warn",
                    metadata=manifest_metadata,
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(f"FRC/PARA write conflicted: {dataset}")
                versions[dataset] = result.manifest.version
            written = {
                dataset: repository.read_frame(dataset, planned[dataset])
                for dataset in WRITE_DATASETS
            }
            if not identity_is_repaired(written):
                raise RuntimeError("Written FRC/PARA snapshot failed its invariant.")
            candidate = _CandidateRepository(repository, versions, written)
            validate_repository_snapshot(candidate).raise_for_errors()
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=(DataQuality.DEGRADED if prepared.warnings else DataQuality.VALID),
                warnings=prepared.warnings,
                expected_etag=prepared.release_etag,
            )
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": committed.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            return {
                **prepared.summary,
                "status": "applied",
                "transaction_id": transaction_id,
                "new_release_version": committed.version,
                "quality": committed.quality,
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore(
                repository,
                old_release=old_release.data,
                old_pointers=old_pointers,
                planned=planned,
            )
            journal.update(
                {
                    "status": "rollback_failed" if rollback_errors else "rolled_back",
                    "original_error": f"{type(original).__name__}: {original}",
                    "rollback_errors": list(rollback_errors),
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            if rollback_errors:
                recovery = (
                    repository.root
                    / "recovery/frc-para-lifecycle-repair"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    f"FRC/PARA rollback failed; recovery marker created: {recovery}"
                ) from original
            raise


def requirements_plan() -> dict[str, Any]:
    return {
        "status": "requirements_plan",
        "network_accessed": False,
        "r2_accessed": False,
        "eodhd_total_calls": EXPECTED_EODHD_CALLS,
        "eodhd_requests": [
            {
                "endpoint": endpoint,
                "symbol": FRC_PROVIDER_SYMBOL,
                "params": REQUEST_PARAMS,
                "attempts": 1,
                "retries": 0,
            }
            for endpoint in EODHD_ENDPOINTS
        ],
        "psky_eodhd_calls": 0,
        "frc_policy": "FRC->FRCB same security; no terminal zero/recovery event",
        "para_policy": "ordinary no-election holder receives PSKY 1:1",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="data/cache")
    parser.add_argument("--requirements-plan", action="store_true")
    parser.add_argument("--fetch-missing-eodhd", action="store_true")
    parser.add_argument(
        "--promote-quarantine",
        metavar="SHA256",
        help="Validate one exact raw quarantine and promote it without network access.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--offline-plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    promote_id = _text(getattr(args, "promote_quarantine", ""))
    if args.requirements_plan:
        if args.fetch_missing_eodhd or args.apply or promote_id:
            raise ValueError("Requirements plan cannot fetch or apply.")
        return requirements_plan()
    if args.offline_plan and args.fetch_missing_eodhd:
        raise ValueError("Offline plan cannot fetch EODHD.")
    if args.apply and args.fetch_missing_eodhd:
        raise ValueError("Fetch and apply must be separate invocations.")
    if promote_id and args.fetch_missing_eodhd:
        raise ValueError("Quarantine promotion cannot fetch EODHD.")
    if promote_id and args.apply:
        raise ValueError("Quarantine promotion and release apply must be separate invocations.")
    repository = LocalDatasetRepository(args.cache_root)
    promotion: dict[str, Any] = {}
    if promote_id:
        promoted, quarantine, promoted_path = promote_raw_quarantine(
            repository.root, promote_id
        )
        promotion = {
            "quarantine_promoted": True,
            "quarantine_id": quarantine.quarantine_id,
            "quarantine_path": str(quarantine.path),
            "promoted_bundle_cache_path": str(promoted_path),
            "promotion_network_accessed": False,
            "promotion_eodhd_http_attempts": 0,
            "promoted_budget_receipt": dict(quarantine.budget_receipt),
            "promoted_raw_eod_sha256": promoted.artifacts[0].source_hash,
            "promoted_envelope_corrections": [
                dict(item) for item in promoted.envelope_corrections
            ],
        }
    prepared = prepare_run(repository, allow_fetch=bool(args.fetch_missing_eodhd))
    prepared.summary.update(promotion)
    if not args.apply:
        return prepared.summary
    return apply_repair(repository, prepared)


def main() -> int:
    summary = run(_parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
