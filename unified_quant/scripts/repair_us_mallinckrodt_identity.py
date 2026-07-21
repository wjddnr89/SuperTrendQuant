#!/usr/bin/env python3
"""Repair the two Mallinckrodt common-equity identities without linking them.

The frozen bootstrap mislabeled the legacy Mallinckrodt security as a
Muniholdings fund and started the 2022 reorganized security too late.  This
collector uses only two audited EODHD provider codes:

* ``MNKKQ.US`` -> legacy ordinary shares, ending 2022-06-15;
* ``MNKTQ.US`` -> 2022 reorganized ordinary shares, ending 2023-11-13.

For each code, eod/div/splits is one immutable, opt-in request (six maximum,
no retry).  Existing ``MNK_old`` and ``MNK`` bars are used only as strict
same-provider overlap proofs.  ``MNKPF.US`` is a preferred security and is
explicitly excluded.  Provider rows after each legal boundary are retained in
raw evidence but never in the repaired price datasets.

This script does not create bankruptcy cancellation actions.  It leaves two
exact terminal candidates so the separately pinned official-evidence workflow
can promote 2022-06-16 and 2023-11-14 zero-recovery delistings.  No successor
link is permitted between the predecessor and reorganized securities.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.index_membership import IndexEventReplayer
from supertrend_quant.market_store.ingest import EodhdClient, SourceArtifact
from supertrend_quant.market_store.lifecycle import build_lifecycle_candidates
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


DEFAULT_CACHE_ROOT = Path("data/cache")

LEGACY_SECURITY_ID = "US:EODHD:81d711c5-9688-5f2b-9f36-63c8fe3211bf"
REORGANIZED_SECURITY_ID = "US:EODHD:647b8b62-0015-5a56-8a63-4da7ba287025"
LEGACY_CODE = "MNKKQ"
REORGANIZED_CODE = "MNKTQ"
EXCLUDED_PREFERRED_CODE = "MNKPF"

LEGACY_START = "2015-01-02"
LEGACY_LAST = "2022-06-15"
LEGACY_CANCELLATION = "2022-06-16"
REORGANIZED_START = "2022-06-17"
REORGANIZED_LAST = "2023-11-13"
REORGANIZED_CANCELLATION = "2023-11-14"

PROVIDER_RANGES: dict[str, tuple[str, str]] = {
    LEGACY_CODE: ("2015-01-02", "2022-07-12"),
    REORGANIZED_CODE: ("2022-06-17", "2023-12-29"),
}
TARGET_RANGES: dict[str, tuple[str, str]] = {
    LEGACY_CODE: (LEGACY_START, LEGACY_LAST),
    REORGANIZED_CODE: (REORGANIZED_START, REORGANIZED_LAST),
}
# NYSE stopped reporting the legacy common after 2020-10-09 and the OTC
# MNKKQ series begins on 2020-10-13.  The old snapshot carried 2020-10-12 as
# a zero-volume flat bar; it is not an observed trade and must not survive the
# repair.  No other exchange session is allowed to disappear.
DOCUMENTED_NON_TRADING_SESSIONS: dict[str, frozenset[str]] = {
    LEGACY_CODE: frozenset({"2020-10-12"}),
    REORGANIZED_CODE: frozenset(),
}
# The refreshed provider series differs from the archived series only by up to
# half a cent of vendor rounding.  Keep the relative tolerance negligible so
# large-dollar bars cannot hide a material change behind a percentage band.
OVERLAP_PRICE_RTOL = 0.00001
OVERLAP_PRICE_ATOL = 0.0051
MINIMUM_OVERLAP_VOLUME_MATCH_RATIO = 0.98
CODE_TO_SECURITY_ID = {
    LEGACY_CODE: LEGACY_SECURITY_ID,
    REORGANIZED_CODE: REORGANIZED_SECURITY_ID,
}
EODHD_ENDPOINTS = ("eod", "div", "splits")
EODHD_REQUESTS = tuple(
    (code, endpoint)
    for code in (LEGACY_CODE, REORGANIZED_CODE)
    for endpoint in EODHD_ENDPOINTS
)
MAX_EODHD_HTTP_ATTEMPTS = len(EODHD_REQUESTS)
if MAX_EODHD_HTTP_ATTEMPTS != 6:
    raise RuntimeError("Mallinckrodt EODHD inventory changed without a cap audit.")

WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "index_constituent_anchors",
    "index_membership_events",
    "source_archive",
)
EXPECTED_INDEX_BOUNDARIES = {
    "sp500": {"anchor_date": "2015-01-07", "remove_date": "2017-07-26"}
}
OFFICIAL_BINDINGS = {
    "mallinckrodt_2022_cancellation": {
        "security_id": LEGACY_SECURITY_ID,
        "symbol": "MNK",
        "candidate_last_price_date": LEGACY_LAST,
        "effective_date": LEGACY_CANCELLATION,
        "cash_amount": 0.0,
    },
    "mallinckrodt_2023_cancellation": {
        "security_id": REORGANIZED_SECURITY_ID,
        "symbol": "MNK",
        "candidate_last_price_date": REORGANIZED_LAST,
        "effective_date": REORGANIZED_CANCELLATION,
        "cash_amount": 0.0,
    },
}


@dataclass(frozen=True)
class LocalPreflight:
    existing: dict[str, pd.DataFrame]
    pointer_etags: dict[str, str | None]
    contaminated_hashes: frozenset[str]
    already_repaired: bool = False


@dataclass(frozen=True)
class ProviderBundle:
    prices: pd.DataFrame
    actions: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    http_attempts: int


@dataclass
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    preflight: LocalPreflight
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    archive_artifacts: tuple[SourceArtifact, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair legacy and reorganized Mallinckrodt identities."
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument(
        "--fetch-eodhd-mnk",
        action="store_true",
        help=(
            "Allow one request for eod/div/splits on MNKKQ and MNKTQ "
            "(six attempts maximum, no retry)."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--offline-plan", action="store_true")
    return parser.parse_args(argv)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _expected_sessions(start: str, end: str) -> tuple[str, ...]:
    import exchange_calendars as xcals

    values = xcals.get_calendar("XNYS").sessions_in_range(start, end)
    return tuple(pd.Timestamp(value).date().isoformat() for value in values)


def _target_expected_sessions(code: str) -> tuple[str, ...]:
    start, end = TARGET_RANGES[code]
    excluded = DOCUMENTED_NON_TRADING_SESSIONS[code]
    expected = tuple(
        session for session in _expected_sessions(start, end) if session not in excluded
    )
    if excluded - set(_expected_sessions(start, end)):
        raise RuntimeError(f"Documented non-trading session is outside {code} range.")
    return expected


def _concat_unique(
    frames: Iterable[pd.DataFrame], *, keys: tuple[str, ...]
) -> pd.DataFrame:
    frame_values = tuple(frames)
    values = [
        frame for frame in frame_values if frame is not None and not frame.empty
    ]
    if not values:
        columns: list[str] = []
        for frame in frame_values:
            columns.extend(str(column) for column in frame.columns)
        return pd.DataFrame(columns=tuple(dict.fromkeys(columns)))
    output = pd.concat(values, ignore_index=True, sort=False)
    return output.drop_duplicates(list(keys), keep="last").reset_index(drop=True)


def _frame_columns(dataset: str, *frames: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *dataset_spec(dataset).required_columns,
                *(column for frame in frames for column in frame.columns),
            )
        )
    )


def _align(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = ""
    return output.loc[:, list(columns)]


def _read_release_frames(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, pd.DataFrame]:
    missing = sorted(set(WRITE_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError(f"Mallinckrodt repair inputs are missing: {missing}")
    return {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in WRITE_DATASETS
    }


def _capture_pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions[dataset]:
            raise RuntimeError(f"Current pointer does not match release: {dataset}")
        values[dataset] = etag
    return values


def _affected(frame: pd.DataFrame) -> pd.Series:
    return frame["security_id"].astype(str).isin(
        (LEGACY_SECURITY_ID, REORGANIZED_SECURITY_ID)
    )


def _snapshot_is_repaired(existing: Mapping[str, pd.DataFrame]) -> bool:
    master = existing["security_master"]
    rows = master.loc[_affected(master)].copy()
    rows["security_id"] = rows["security_id"].astype(str)
    rows = rows.set_index("security_id")
    if set(rows.index) != {LEGACY_SECURITY_ID, REORGANIZED_SECURITY_ID}:
        return False
    wanted = {
        LEGACY_SECURITY_ID: ("MNKKQ.US", LEGACY_START, LEGACY_LAST, "legacy"),
        REORGANIZED_SECURITY_ID: (
            "MNKTQ.US",
            REORGANIZED_START,
            REORGANIZED_LAST,
            "reorganized",
        ),
    }
    for security_id, (provider, start, end, token) in wanted.items():
        row = rows.loc[security_id]
        if (
            str(row.get("provider_symbol", "")) != provider
            or str(row.get("active_from", "")) != start
            or str(row.get("active_to", "")) != end
            or token not in str(row.get("name", "")).lower()
        ):
            return False
    prices = existing["daily_price_raw"]
    for code, security_id in CODE_TO_SECURITY_ID.items():
        subset = prices.loc[prices["security_id"].astype(str).eq(security_id)]
        expected = _target_expected_sessions(code)
        actual = tuple(sorted(subset["session"].astype(str)))
        if actual != expected:
            return False
    affected_actions = existing["corporate_actions"].loc[
        _affected(existing["corporate_actions"])
    ]
    if (
        affected_actions.get("new_security_id", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .any()
    ):
        return False
    return True


def build_local_preflight(
    repository: LocalDatasetRepository, release: DataRelease
) -> LocalPreflight:
    existing = _read_release_frames(repository, release)
    master = existing["security_master"]
    rows = master.loc[_affected(master)]
    if set(rows["security_id"].astype(str)) != {
        LEGACY_SECURITY_ID,
        REORGANIZED_SECURITY_ID,
    } or len(rows) != 2:
        raise ValueError("Expected exactly two audited Mallinckrodt security ids.")
    pointer_etags = _capture_pointer_etags(repository, release)
    if _snapshot_is_repaired(existing):
        repaired = LocalPreflight(existing, pointer_etags, frozenset(), True)
        validate_identity_and_history_gate(repaired, existing)
        validate_index_replay_gate(
            existing,
            existing,
            completed_session=release.completed_session,
        )
        validate_lifecycle_candidate_gate(existing, release)
        return repaired

    legacy_master = rows.loc[
        rows["security_id"].astype(str).eq(LEGACY_SECURITY_ID)
    ].iloc[0]
    reorganized_master = rows.loc[
        rows["security_id"].astype(str).eq(REORGANIZED_SECURITY_ID)
    ].iloc[0]
    if "muniholdings" not in str(legacy_master["name"]).lower():
        raise ValueError(
            "Legacy Mallinckrodt preflight no longer has the audited Muniholdings mislabel."
        )
    if "mallinckrodt" not in str(reorganized_master["name"]).lower():
        raise ValueError("Reorganized Mallinckrodt identity changed before repair.")
    prices = existing["daily_price_raw"]
    legacy = prices.loc[prices["security_id"].astype(str).eq(LEGACY_SECURITY_ID)]
    reorganized = prices.loc[
        prices["security_id"].astype(str).eq(REORGANIZED_SECURITY_ID)
    ]
    if legacy.empty or reorganized.empty:
        raise ValueError("Both audited Mallinckrodt overlap histories are required.")
    legacy_range = (
        legacy["session"].astype(str).min(),
        legacy["session"].astype(str).max(),
    )
    reorganized_range = (
        reorganized["session"].astype(str).min(),
        reorganized["session"].astype(str).max(),
    )
    if legacy_range != ("2015-01-02", "2020-10-12"):
        raise ValueError(f"Audited MNK_old overlap range changed: {legacy_range}")
    if reorganized_range != ("2022-10-27", "2023-08-28"):
        raise ValueError(f"Audited MNK overlap range changed: {reorganized_range}")
    hashes = frozenset(
        prices.loc[_affected(prices), "source_hash"].dropna().astype(str)
    )
    if not hashes:
        raise ValueError("Existing Mallinckrodt overlap rows lack provenance hashes.")
    return LocalPreflight(existing, pointer_etags, hashes, False)


class CappedSingleAttemptEodhdClient(EodhdClient):
    """One HTTP attempt per request and a six-attempt run-wide cap."""

    def __init__(
        self, *args: Any, max_attempts: int = MAX_EODHD_HTTP_ATTEMPTS, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.max_attempts = int(max_attempts)
        self._attempt_count = 0
        self._lock = threading.Lock()

    @property
    def attempt_count(self) -> int:
        with self._lock:
            return self._attempt_count

    def get_json(
        self, endpoint: str, *, params: dict[str, object] | None = None
    ) -> Any:
        safe_endpoint = "/" + endpoint.strip("/")
        query = {**(params or {}), "api_token": self.token, "fmt": "json"}
        with self._lock:
            if self._attempt_count >= self.max_attempts:
                raise RuntimeError("Mallinckrodt EODHD request cap reached before HTTP.")
            self.budget.claim()
            self._attempt_count += 1
        try:
            response = self.session.get(
                self.base_url + safe_endpoint,
                params=query,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f"HTTP {status}" if status else type(exc).__name__
            raise RuntimeError(
                "Mallinckrodt EODHD single attempt failed for "
                f"{safe_endpoint}: {detail}"
            ) from None


def _eodhd_params(code: str) -> dict[str, str]:
    start, end = PROVIDER_RANGES[code]
    return {"from": start, "to": end}


def _public_url(code: str, endpoint: str) -> str:
    return (
        f"https://eodhd.com/api/{endpoint}/{code}.US?"
        + urlencode(_eodhd_params(code))
    )


class EodhdMallinckrodtSource:
    """Six token-free immutable endpoint caches for MNKKQ and MNKTQ."""

    SCHEMA = "mallinckrodt_eodhd_raw/v1"

    def __init__(
        self,
        root: str | Path,
        *,
        allow_http: bool,
        client_factory: Callable[[], Any] = CappedSingleAttemptEodhdClient,
    ):
        self.root = Path(root)
        self.allow_http = bool(allow_http)
        self.client_factory = client_factory
        self.http_attempts = 0

    def path(self, code: str, endpoint: str) -> Path:
        return self.root / f"{sha256_bytes(_public_url(code, endpoint).encode())}.json.gz"

    def _decode(self, code: str, endpoint: str, payload: bytes) -> SourceArtifact:
        path = self.path(code, endpoint)
        try:
            value = json.loads(gzip.decompress(payload))
            content = base64.b64decode(value["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError(f"Unreadable Mallinckrodt endpoint cache: {path}") from exc
        if (
            value.get("schema") != self.SCHEMA
            or value.get("code") != code
            or value.get("endpoint") != endpoint
            or value.get("source_url") != _public_url(code, endpoint)
            or value.get("source_hash") != sha256_bytes(content)
        ):
            raise ValueError(f"Mallinckrodt endpoint cache identity mismatch: {path}")
        return SourceArtifact(
            source=f"eodhd_{endpoint}",
            source_url=_public_url(code, endpoint),
            retrieved_at=str(value["retrieved_at"]),
            content=content,
            content_type="application/json",
        )

    def get(self, code: str, endpoint: str) -> SourceArtifact | None:
        path = self.path(code, endpoint)
        return self._decode(code, endpoint, path.read_bytes()) if path.is_file() else None

    def _store(
        self, code: str, endpoint: str, artifact: SourceArtifact
    ) -> SourceArtifact:
        value = {
            "schema": self.SCHEMA,
            "code": code,
            "endpoint": endpoint,
            "source_url": artifact.source_url,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
        }
        path = self.path(code, endpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(_canonical_json_bytes(value), mtime=0)
        if path.is_file():
            existing = self._decode(code, endpoint, path.read_bytes())
            if existing.content != artifact.content:
                raise RuntimeError(f"Immutable Mallinckrodt cache changed: {path}")
            return existing
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(encoded)
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self._decode(code, endpoint, path.read_bytes())
                if existing.content != artifact.content:
                    raise RuntimeError(f"Immutable Mallinckrodt cache changed: {path}")
        finally:
            temporary.unlink(missing_ok=True)
        return self._decode(code, endpoint, path.read_bytes())

    def fetch(self) -> ProviderBundle:
        cached = {
            request: self.get(*request) for request in EODHD_REQUESTS
        }
        missing = [request for request, artifact in cached.items() if artifact is None]
        if missing and not self.allow_http:
            raise FileNotFoundError(
                "Mallinckrodt endpoint cache is incomplete; explicitly allow fetch: "
                + ", ".join(f"{code}/{endpoint}" for code, endpoint in missing)
            )
        if len(missing) > MAX_EODHD_HTTP_ATTEMPTS:
            raise RuntimeError("Missing Mallinckrodt requests exceed the frozen cap.")
        client = self.client_factory() if missing else None
        for code, endpoint in missing:
            rows = client.get_json(
                f"{endpoint}/{code}.US",
                params=_eodhd_params(code),
            )
            artifact = SourceArtifact(
                source=f"eodhd_{endpoint}",
                source_url=_public_url(code, endpoint),
                retrieved_at=utc_now_iso(),
                content=_canonical_json_bytes(rows),
                content_type="application/json",
            )
            cached[(code, endpoint)] = self._store(code, endpoint, artifact)
        self.http_attempts = (
            int(getattr(client, "attempt_count", len(missing))) if client else 0
        )
        if self.http_attempts > MAX_EODHD_HTTP_ATTEMPTS:
            raise RuntimeError("Mallinckrodt source exceeded its six-attempt cap.")
        artifacts = tuple(cached[request] for request in EODHD_REQUESTS)
        prices = _concat_unique(
            (
                _eodhd_price_frame(
                    code,
                    cached[(code, "eod")],
                )
                for code in (LEGACY_CODE, REORGANIZED_CODE)
            ),
            keys=dataset_spec("daily_price_raw").primary_key,
        )
        actions = _concat_unique(
            (
                _eodhd_action_frame(
                    code,
                    cached[(code, "div")],
                    cached[(code, "splits")],
                )
                for code in (LEGACY_CODE, REORGANIZED_CODE)
            ),
            keys=dataset_spec("corporate_actions").primary_key,
        )
        return ProviderBundle(prices, actions, artifacts, self.http_attempts)


def _artifact_json_rows(artifact: SourceArtifact) -> list[dict[str, Any]]:
    try:
        value = json.loads(artifact.content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Provider artifact is invalid JSON: {artifact.source_url}") from exc
    if value in (None, {}, []):
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Provider artifact has wrong shape: {artifact.source_url}")
    return value


def _eodhd_price_frame(code: str, artifact: SourceArtifact) -> pd.DataFrame:
    rows = []
    for item in _artifact_json_rows(artifact):
        if not item.get("date") or item.get("close") is None:
            continue
        rows.append(
            {
                "security_id": CODE_TO_SECURITY_ID[code],
                "session": str(item["date"]),
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "close": item.get("close"),
                "volume": item.get("volume", 0),
                "currency": "USD",
                "source": f"eodhd_{code.lower()}_eod",
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    columns = _frame_columns("daily_price_raw", pd.DataFrame(rows))
    return pd.DataFrame(rows, columns=columns)


def _parse_split(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    for separator in ("/", ":"):
        if separator in text:
            left, right = text.split(separator, 1)
            try:
                denominator = float(right)
                return float(left) / denominator if denominator else None
            except ValueError:
                return None
    try:
        return float(text)
    except ValueError:
        return None


def _action_id(*values: Any) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode()).hexdigest()


def _eodhd_action_frame(
    code: str,
    dividends: SourceArtifact,
    splits: SourceArtifact,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    security_id = CODE_TO_SECURITY_ID[code]
    for endpoint, artifact in (("div", dividends), ("splits", splits)):
        for item in _artifact_json_rows(artifact):
            effective = str(item.get("date") or "")
            if not effective:
                continue
            if endpoint == "div":
                action_type = "cash_dividend"
                cash = item.get("unadjustedValue", item.get("value"))
                ratio = None
            else:
                action_type = "split"
                cash = None
                ratio = _parse_split(item.get("split"))
                if ratio is None:
                    continue
            rows.append(
                {
                    "event_id": _action_id(
                        "mnk-provider",
                        code,
                        security_id,
                        action_type,
                        effective,
                        cash,
                        ratio,
                    ),
                    "security_id": security_id,
                    "action_type": action_type,
                    "effective_date": effective,
                    "ex_date": effective,
                    "announcement_date": str(item.get("declarationDate") or ""),
                    "record_date": str(item.get("recordDate") or ""),
                    "payment_date": str(item.get("paymentDate") or ""),
                    "cash_amount": cash,
                    "ratio": ratio,
                    "currency": str(item.get("currency") or "USD"),
                    "new_security_id": "",
                    "new_symbol": "",
                    "official": False,
                    "source_url": artifact.source_url,
                    "source_kind": "provider",
                    "source": f"eodhd_{code.lower()}_{endpoint}",
                    "retrieved_at": artifact.retrieved_at,
                    "source_hash": artifact.source_hash,
                }
            )
    return pd.DataFrame(
        rows,
        columns=_frame_columns("corporate_actions", pd.DataFrame(rows)),
    )


def _sessions_exact(
    frame: pd.DataFrame, expected: tuple[str, ...], label: str
) -> None:
    actual = tuple(sorted(frame["session"].astype(str)))
    if len(actual) != len(set(actual)):
        raise ValueError(f"{label} contains duplicate sessions.")
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"{label} session coverage mismatch: expected={len(expected)}, "
            f"actual={len(actual)}, missing={missing[:8]}, extra={extra[:8]}"
        )


def _validate_ohlcv(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label} price history is empty.")
    numeric = frame.loc[:, ["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any():
        raise ValueError(f"{label} contains non-numeric OHLCV values.")
    if (numeric.loc[:, ["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{label} contains non-positive OHLC values.")
    if (numeric["volume"] < 0).any():
        raise ValueError(f"{label} contains negative volume.")
    if (
        numeric["high"]
        < numeric.loc[:, ["open", "low", "close"]].max(axis=1) - 1e-10
    ).any() or (
        numeric["low"]
        > numeric.loc[:, ["open", "high", "close"]].min(axis=1) + 1e-10
    ).any():
        raise ValueError(f"{label} violates OHLC bounds.")


def _strict_overlap(
    existing: pd.DataFrame,
    provider: pd.DataFrame,
    *,
    label: str,
    allowed_missing_sessions: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    left = existing.copy()
    right = provider.copy()
    left["session"] = left["session"].astype(str)
    right["session"] = right["session"].astype(str)
    if left["session"].duplicated().any() or right["session"].duplicated().any():
        raise ValueError(f"{label} overlap contains duplicate sessions.")
    missing_sessions = set(left["session"]) - set(right["session"])
    if missing_sessions != set(allowed_missing_sessions):
        raise ValueError(
            f"{label} provider history has unexpected missing overlap sessions: "
            f"{sorted(missing_sessions)}"
        )
    if missing_sessions:
        missing_rows = left.loc[left["session"].isin(missing_sessions)]
        numeric = missing_rows.loc[:, ["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        )
        flat = numeric.loc[:, ["open", "high", "low", "close"]].nunique(axis=1).eq(1)
        if numeric.isna().any().any() or not bool(
            (flat & numeric["volume"].eq(0)).all()
        ):
            raise ValueError(
                f"{label} documented non-trading row is not a zero-volume flat carry."
            )
    comparable = left.loc[~left["session"].isin(missing_sessions)].copy()
    merged = comparable.loc[:, ["session", "open", "high", "low", "close", "volume"]].merge(
        right.loc[:, ["session", "open", "high", "low", "close", "volume"]],
        on="session",
        how="inner",
        suffixes=("_existing", "_provider"),
        validate="one_to_one",
    )
    if len(merged) != len(comparable) or set(merged["session"]) != set(
        comparable["session"]
    ):
        raise ValueError(
            f"{label} provider history does not cover every existing overlap row."
        )
    max_deltas: dict[str, float] = {}
    for column in ("open", "high", "low", "close"):
        old = pd.to_numeric(merged[f"{column}_existing"], errors="coerce")
        new = pd.to_numeric(merged[f"{column}_provider"], errors="coerce")
        if old.isna().any() or new.isna().any():
            raise ValueError(f"{label} overlap contains invalid {column} values.")
        delta = (old - new).abs()
        max_deltas[column] = float(delta.max()) if len(delta) else 0.0
        passed = np.isclose(
            old, new, rtol=OVERLAP_PRICE_RTOL, atol=OVERLAP_PRICE_ATOL
        )
        if not bool(passed.all()):
            bad = merged.loc[~passed, "session"]
            raise ValueError(
                f"{label} {column} overlap mismatch: {bad.head(5).tolist()}"
            )
    old_volume = pd.to_numeric(merged["volume_existing"], errors="coerce")
    new_volume = pd.to_numeric(merged["volume_provider"], errors="coerce")
    if old_volume.isna().any() or new_volume.isna().any():
        raise ValueError(f"{label} volume overlap contains invalid values.")
    volume_passed = np.isclose(old_volume, new_volume, rtol=0.02, atol=1.0)
    volume_match_ratio = float(volume_passed.mean())
    if volume_match_ratio < MINIMUM_OVERLAP_VOLUME_MATCH_RATIO:
        raise ValueError(
            f"{label} volume overlap match ratio is below the audited floor: "
            f"{volume_match_ratio:.6f} < {MINIMUM_OVERLAP_VOLUME_MATCH_RATIO:.6f}."
        )
    volume_denominator = np.maximum(
        np.maximum(old_volume.abs(), new_volume.abs()), 1.0
    )
    volume_relative_deviation = (old_volume - new_volume).abs() / volume_denominator
    return {
        "rows": len(merged),
        "first_session": merged["session"].min(),
        "last_session": merged["session"].max(),
        "ohlc_max_abs_delta": max_deltas,
        "documented_non_trading_sessions_removed": sorted(missing_sessions),
        "price_rtol": OVERLAP_PRICE_RTOL,
        "price_atol": OVERLAP_PRICE_ATOL,
        "volume_match_ratio": volume_match_ratio,
        "minimum_volume_match_ratio": MINIMUM_OVERLAP_VOLUME_MATCH_RATIO,
        "volume_mismatch_sessions": merged.loc[
            ~volume_passed, "session"
        ].astype(str).tolist(),
        "maximum_volume_relative_deviation": float(
            volume_relative_deviation.max()
        ),
        "source_relationship": "same_eodhd_series_identity_proof",
        "independent_cross_validation": False,
    }


def _assert_frame_matches_raw_artifacts(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    dataset: str,
) -> None:
    keys = list(dataset_spec(dataset).primary_key)
    columns = _frame_columns(dataset, actual, expected)
    left = _align(actual, columns).sort_values(keys).reset_index(drop=True)
    right = _align(expected, columns).sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(
            f"Mallinckrodt {dataset} rows do not match archived raw artifacts."
        ) from exc


def _validate_bundle_derivation(bundle: ProviderBundle) -> None:
    derived_prices = _concat_unique(
        (
            _eodhd_price_frame(code, _artifact_for_code(bundle, code, "eod"))
            for code in (LEGACY_CODE, REORGANIZED_CODE)
        ),
        keys=dataset_spec("daily_price_raw").primary_key,
    )
    derived_actions = _concat_unique(
        (
            _eodhd_action_frame(
                code,
                _artifact_for_code(bundle, code, "div"),
                _artifact_for_code(bundle, code, "splits"),
            )
            for code in (LEGACY_CODE, REORGANIZED_CODE)
        ),
        keys=dataset_spec("corporate_actions").primary_key,
    )
    _assert_frame_matches_raw_artifacts(
        bundle.prices,
        derived_prices,
        dataset="daily_price_raw",
    )
    _assert_frame_matches_raw_artifacts(
        bundle.actions,
        derived_actions,
        dataset="corporate_actions",
    )


def validate_provider_bundle(
    preflight: LocalPreflight,
    bundle: ProviderBundle,
) -> dict[str, Any]:
    if len(bundle.artifacts) != MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError("Mallinckrodt bundle must retain all six raw request artifacts.")
    urls = [artifact.source_url for artifact in bundle.artifacts]
    if len(set(urls)) != MAX_EODHD_HTTP_ATTEMPTS or set(urls) != {
        _public_url(code, endpoint) for code, endpoint in EODHD_REQUESTS
    }:
        raise ValueError("Mallinckrodt raw request inventory is incomplete or duplicated.")
    if not 0 <= bundle.http_attempts <= MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError("Mallinckrodt bundle exceeded the HTTP cap.")
    if any(EXCLUDED_PREFERRED_CODE in url for url in urls):
        raise ValueError("MNKPF preferred-security data entered the common-equity bundle.")

    existing_prices = preflight.existing["daily_price_raw"]
    overlap: dict[str, Any] = {}
    tails: dict[str, int] = {}
    for code, security_id in CODE_TO_SECURITY_ID.items():
        provider = bundle.prices.loc[
            bundle.prices["security_id"].astype(str).eq(security_id)
        ].copy()
        _validate_ohlcv(provider, code)
        provider_dates = provider["session"].astype(str)
        provider_range = (provider_dates.min(), provider_dates.max())
        if provider_range != PROVIDER_RANGES[code]:
            raise ValueError(
                f"{code} provider boundary changed: expected={PROVIDER_RANGES[code]}, "
                f"actual={provider_range}"
            )
        start, end = TARGET_RANGES[code]
        target = provider.loc[provider_dates.between(start, end)].copy()
        _sessions_exact(target, _target_expected_sessions(code), f"{code} target")
        tail = provider.loc[provider_dates.gt(end)]
        if tail.empty:
            raise ValueError(f"{code} has no audited provider tail to trim.")
        tails[code] = len(tail)
        existing = existing_prices.loc[
            existing_prices["security_id"].astype(str).eq(security_id)
        ].copy()
        overlap[code] = _strict_overlap(
            existing,
            provider,
            label=f"{code}/{security_id}",
            allowed_missing_sessions=DOCUMENTED_NON_TRADING_SESSIONS[code],
        )
    legacy = bundle.prices.loc[
        bundle.prices["security_id"].astype(str).eq(LEGACY_SECURITY_ID)
        & bundle.prices["session"].astype(str).between(
            REORGANIZED_START, PROVIDER_RANGES[LEGACY_CODE][1]
        )
    ]
    reorganized = bundle.prices.loc[
        bundle.prices["security_id"].astype(str).eq(REORGANIZED_SECURITY_ID)
        & bundle.prices["session"].astype(str).between(
            REORGANIZED_START, PROVIDER_RANGES[LEGACY_CODE][1]
        )
    ]
    common = set(legacy["session"].astype(str)) & set(
        reorganized["session"].astype(str)
    )
    if not common:
        raise ValueError("Provider evidence no longer exposes the audited two-identity overlap.")
    _validate_bundle_derivation(bundle)
    return {
        "overlap": overlap,
        "provider_tail_rows_trimmed": tails,
        "two_identity_overlap_sessions": len(common),
        "preferred_code_excluded": f"{EXCLUDED_PREFERRED_CODE}.US",
        "yahoo_claimed": False,
    }


def _artifact_for_code(
    bundle: ProviderBundle, code: str, endpoint: str = "eod"
) -> SourceArtifact:
    url = _public_url(code, endpoint)
    matches = [artifact for artifact in bundle.artifacts if artifact.source_url == url]
    if len(matches) != 1:
        raise ValueError(f"Provider artifact is not unique: {code}/{endpoint}")
    return matches[0]


def rewrite_security_identities(
    master: pd.DataFrame,
    history: pd.DataFrame,
    *,
    bundle: ProviderBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_master = master.copy()
    wanted = {
        LEGACY_SECURITY_ID: {
            "primary_symbol": "MNK",
            "provider_symbol": "MNKKQ.US",
            "action_provider_symbol": "MNKKQ.US",
            "name": "Mallinckrodt plc (legacy ordinary shares)",
            "exchange": "NYSE",
            "asset_type": "STOCK",
            "currency": "USD",
            "country": "US",
            "active_from": LEGACY_START,
            "active_to": LEGACY_LAST,
            "code": LEGACY_CODE,
        },
        REORGANIZED_SECURITY_ID: {
            "primary_symbol": "MNK",
            "provider_symbol": "MNKTQ.US",
            "action_provider_symbol": "MNKTQ.US",
            "name": "Mallinckrodt plc (2022 reorganized ordinary shares)",
            "exchange": "NYSE MKT",
            "asset_type": "STOCK",
            "currency": "USD",
            "country": "US",
            "active_from": REORGANIZED_START,
            "active_to": REORGANIZED_LAST,
            "code": REORGANIZED_CODE,
        },
    }
    for security_id, values in wanted.items():
        indices = output_master.index[
            output_master["security_id"].astype(str).eq(security_id)
        ]
        if len(indices) != 1:
            raise ValueError(f"Mallinckrodt master identity changed: {security_id}")
        artifact = _artifact_for_code(bundle, str(values["code"]))
        for column, value in values.items():
            if column == "code":
                continue
            if column not in output_master:
                output_master[column] = ""
            output_master.loc[indices, column] = value
        output_master.loc[indices, "source"] = "mallinckrodt_identity_repair"
        if "source_url" not in output_master:
            output_master["source_url"] = ""
        output_master.loc[indices, "source_url"] = artifact.source_url
        output_master.loc[indices, "retrieved_at"] = artifact.retrieved_at
        output_master.loc[indices, "source_hash"] = artifact.source_hash

    keep = ~history["security_id"].astype(str).isin(
        (LEGACY_SECURITY_ID, REORGANIZED_SECURITY_ID)
    )
    rows = []
    for code, security_id in CODE_TO_SECURITY_ID.items():
        start, end = TARGET_RANGES[code]
        artifact = _artifact_for_code(bundle, code)
        rows.append(
            {
                "security_id": security_id,
                "symbol": "MNK",
                "exchange": (
                    "NYSE" if security_id == LEGACY_SECURITY_ID else "NYSE MKT"
                ),
                "effective_from": start,
                "effective_to": end,
                "source": "mallinckrodt_identity_repair",
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    additions = pd.DataFrame(rows)
    columns = _frame_columns("symbol_history", history, additions)
    output_history = pd.concat(
        [
            _align(history.loc[keep], columns),
            _align(additions, columns),
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(
        list(dataset_spec("symbol_history").primary_key), keep="last"
    )
    return output_master.reset_index(drop=True), output_history.reset_index(drop=True)


def _trim_provider_frames(
    bundle: ProviderBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    for code, security_id in CODE_TO_SECURITY_ID.items():
        start, end = TARGET_RANGES[code]
        price_dates = bundle.prices["session"].astype(str)
        price_parts.append(
            bundle.prices.loc[
                bundle.prices["security_id"].astype(str).eq(security_id)
                & price_dates.between(start, end)
            ].copy()
        )
        action_dates = bundle.actions.get(
            "effective_date", pd.Series(index=bundle.actions.index, dtype=str)
        ).astype(str)
        action_parts.append(
            bundle.actions.loc[
                bundle.actions["security_id"].astype(str).eq(security_id)
                & action_dates.between(start, end)
            ].copy()
        )
    prices = _concat_unique(
        price_parts,
        keys=dataset_spec("daily_price_raw").primary_key,
    )
    actions = _concat_unique(
        action_parts,
        keys=dataset_spec("corporate_actions").primary_key,
    )
    if not actions.empty:
        successor = actions["new_security_id"].fillna("").astype(str).str.strip()
        successor_symbol = actions["new_symbol"].fillna("").astype(str).str.strip()
        if successor.ne("").any() or successor_symbol.ne("").any():
            raise ValueError("Mallinckrodt provider actions contain a successor link.")
    return prices, actions


def rewrite_prices_actions_factors(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    provider_prices: pd.DataFrame,
    provider_actions: pd.DataFrame,
    source_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_columns = _frame_columns("daily_price_raw", prices, provider_prices)
    output_prices = pd.concat(
        [
            _align(prices.loc[~_affected(prices)], price_columns),
            _align(provider_prices, price_columns),
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(
        list(dataset_spec("daily_price_raw").primary_key), keep="last"
    )
    action_columns = _frame_columns("corporate_actions", actions, provider_actions)
    output_actions = pd.concat(
        [
            _align(actions.loc[~_affected(actions)], action_columns),
            _align(provider_actions, action_columns),
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(
        list(dataset_spec("corporate_actions").primary_key), keep="last"
    )
    affected_prices = output_prices.loc[_affected(output_prices)].copy()
    affected_actions = output_actions.loc[_affected(output_actions)].copy()
    rebuilt = build_adjustment_factors(
        affected_prices,
        affected_actions,
        source_version=source_version,
    )
    factor_columns = _frame_columns("adjustment_factors", factors, rebuilt)
    output_factors = pd.concat(
        [
            _align(factors.loc[~_affected(factors)], factor_columns),
            _align(rebuilt, factor_columns),
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(
        list(dataset_spec("adjustment_factors").primary_key), keep="last"
    )
    return (
        output_prices.sort_values(["security_id", "session"]).reset_index(drop=True),
        output_actions.sort_values(["security_id", "effective_date"]).reset_index(
            drop=True
        ),
        output_factors.sort_values(["security_id", "session"]).reset_index(drop=True),
    )


def _remapped_event_id(row: Mapping[str, Any], security_id: str) -> str:
    return sha256_bytes(
        _canonical_json_bytes(
            {
                "operation": "mallinckrodt_index_identity_remap",
                "prior_event_id": str(row.get("event_id") or ""),
                "index_id": str(row.get("index_id") or ""),
                "effective_date": str(row.get("effective_date") or ""),
                "membership_operation": str(row.get("operation") or "").upper(),
                "security_id": security_id,
            }
        )
    )


def rewrite_index_references(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    *,
    legacy_artifact: SourceArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    output_anchors = anchors.copy()
    anchor_dates = pd.to_datetime(output_anchors["anchor_date"], errors="coerce")
    historical_anchors = (
        output_anchors["security_id"].astype(str).eq(REORGANIZED_SECURITY_ID)
        & anchor_dates.le(pd.Timestamp(LEGACY_CANCELLATION))
    )
    output_anchors.loc[historical_anchors, "security_id"] = LEGACY_SECURITY_ID
    for column, value in {
        "source": "mallinckrodt_identity_repair",
        "source_url": legacy_artifact.source_url,
        "source_kind": "provider_overlap_identity",
        "retrieved_at": legacy_artifact.retrieved_at,
        "source_hash": legacy_artifact.source_hash,
    }.items():
        if column not in output_anchors:
            output_anchors[column] = ""
        output_anchors.loc[historical_anchors, column] = value
    output_anchors = output_anchors.drop_duplicates(
        list(dataset_spec("index_constituent_anchors").primary_key), keep="last"
    )

    output_events = events.copy()
    event_dates = pd.to_datetime(output_events["effective_date"], errors="coerce")
    historical_events = (
        output_events["security_id"].astype(str).eq(REORGANIZED_SECURITY_ID)
        & event_dates.le(pd.Timestamp(LEGACY_CANCELLATION))
    )
    for index in output_events.index[historical_events]:
        row = output_events.loc[index].to_dict()
        output_events.loc[index, "security_id"] = LEGACY_SECURITY_ID
        output_events.loc[index, "event_id"] = _remapped_event_id(
            row, LEGACY_SECURITY_ID
        )
    for column, value in {
        "source": "mallinckrodt_identity_repair",
        "source_url": legacy_artifact.source_url,
        "source_kind": "provider_overlap_identity",
        "retrieved_at": legacy_artifact.retrieved_at,
        "source_hash": legacy_artifact.source_hash,
    }.items():
        if column not in output_events:
            output_events[column] = ""
        output_events.loc[historical_events, column] = value
    output_events = output_events.drop_duplicates(
        list(dataset_spec("index_membership_events").primary_key), keep="last"
    )
    return (
        output_anchors.reset_index(drop=True),
        output_events.reset_index(drop=True),
        {
            "anchors_remapped_to_legacy": int(historical_anchors.sum()),
            "events_remapped_to_legacy": int(historical_events.sum()),
        },
    )


def _artifact_extension(artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower()
    if "json" in content_type:
        return "json"
    if "csv" in content_type:
        return "csv"
    if "html" in content_type:
        return "html"
    if "pdf" in content_type:
        return "pdf"
    return "bin"


def _request_archive_artifact(artifact: SourceArtifact) -> SourceArtifact:
    content = _canonical_json_bytes(
        {
            "schema": "mallinckrodt_request_archive/v1",
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
            "content_sha256": artifact.source_hash,
            "content_type": artifact.content_type,
            "source": artifact.source,
            "source_url": artifact.source_url,
        }
    )
    return SourceArtifact(
        source=artifact.source,
        source_url=artifact.source_url,
        retrieved_at=artifact.retrieved_at,
        content=content,
        content_type="application/vnd.supertrendquant.source-envelope+json",
    )


def _validate_request_envelopes(
    raw_artifacts: Iterable[SourceArtifact],
    archived_artifacts: Iterable[SourceArtifact],
) -> None:
    archived_by_url = {artifact.source_url: artifact for artifact in archived_artifacts}
    for raw in raw_artifacts:
        archived = archived_by_url.get(raw.source_url)
        if archived is None:
            raise ValueError(f"Mallinckrodt request envelope is absent: {raw.source_url}")
        try:
            value = json.loads(archived.content)
            content = base64.b64decode(value["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError(
                f"Mallinckrodt request envelope is invalid: {raw.source_url}"
            ) from exc
        if (
            value.get("schema") != "mallinckrodt_request_archive/v1"
            or value.get("source_url") != raw.source_url
            or value.get("content_sha256") != raw.source_hash
            or content != raw.content
        ):
            raise ValueError(
                f"Mallinckrodt request envelope changed raw bytes: {raw.source_url}"
            )


def _repair_manifest_artifact(
    *,
    release: DataRelease,
    bundle: ProviderBundle,
    provider_validation: Mapping[str, Any],
    index_stats: Mapping[str, Any],
) -> SourceArtifact:
    content = _canonical_json_bytes(
        {
            "schema": "mallinckrodt_identity_repair_manifest/v1",
            "base_release_version": release.version,
            "completed_session": release.completed_session,
            "legacy_security_id": LEGACY_SECURITY_ID,
            "reorganized_security_id": REORGANIZED_SECURITY_ID,
            "provider_ranges": PROVIDER_RANGES,
            "target_ranges": TARGET_RANGES,
            "raw_artifacts": [
                {
                    "source_url": artifact.source_url,
                    "source_sha256": artifact.source_hash,
                }
                for artifact in bundle.artifacts
            ],
            "provider_validation": provider_validation,
            "index_rewrite": dict(index_stats),
            "preferred_code_excluded": f"{EXCLUDED_PREFERRED_CODE}.US",
            "no_successor_link": True,
            "official_evidence_bindings": OFFICIAL_BINDINGS,
            "independent_cross_validation": False,
            "cross_validation_scope": "same_provider_overlap_identity_only",
        }
    )
    return SourceArtifact(
        source="mallinckrodt_identity_repair_manifest",
        source_url="local://mallinckrodt-identity-repair/manifest-v1",
        retrieved_at=utc_now_iso(),
        content=content,
        content_type="application/json",
    )


def append_source_archive(
    frame: pd.DataFrame,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> pd.DataFrame:
    rows = [
        {
            "archive_id": artifact.source_hash,
            "dataset": artifact.source,
            "object_path": (
                f"archives/{completed_session}/{artifact.source_hash}."
                f"{_artifact_extension(artifact)}.gz"
            ),
            "content_type": artifact.content_type,
            "effective_date": completed_session,
            "source": artifact.source,
            "source_url": artifact.source_url,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
        }
        for artifact in artifacts
    ]
    additions = pd.DataFrame(rows)
    columns = _frame_columns("source_archive", frame, additions)
    return _concat_unique(
        (_align(frame, columns), _align(additions, columns)),
        keys=dataset_spec("source_archive").primary_key,
    )


class _FrameRepository:
    def __init__(self, frames: Mapping[str, pd.DataFrame]):
        self.frames = frames

    def current_manifest(self, dataset: str) -> object | None:
        return object() if dataset in self.frames else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        return self.frames[dataset].copy()


def _issue_counts(
    frames: Mapping[str, pd.DataFrame],
) -> dict[tuple[str, str], int]:
    report = validate_repository_snapshot(_FrameRepository(frames))
    output: dict[tuple[str, str], int] = {}
    for issue in report.issues:
        key = (issue.code, issue.severity)
        output[key] = output.get(key, 0) + int(issue.row_count or 1)
    return output


def _validate_nonregression(
    previous: Mapping[str, pd.DataFrame],
    candidate: Mapping[str, pd.DataFrame],
) -> None:
    before = _issue_counts(previous)
    after = _issue_counts(candidate)
    regressions = {
        key: (before.get(key, 0), count)
        for key, count in after.items()
        if count > before.get(key, 0)
    }
    if regressions:
        raise ValueError(
            "Mallinckrodt repair introduced repository validation issues: "
            f"{regressions}"
        )


def validate_index_replay_gate(
    previous: Mapping[str, pd.DataFrame],
    candidate: Mapping[str, pd.DataFrame],
    *,
    completed_session: str,
) -> dict[str, Any]:
    before = IndexEventReplayer(
        previous["index_constituent_anchors"],
        previous["index_membership_events"],
    )
    after = IndexEventReplayer(
        candidate["index_constituent_anchors"],
        candidate["index_membership_events"],
    )
    anchors = candidate["index_constituent_anchors"]
    events = candidate["index_membership_events"]
    checked = 0
    for index_id, boundary in EXPECTED_INDEX_BOUNDARIES.items():
        anchor = anchors.loc[
            anchors["index_id"].astype(str).eq(index_id)
            & anchors["security_id"].astype(str).eq(LEGACY_SECURITY_ID)
            & anchors["anchor_date"].astype(str).eq(boundary["anchor_date"])
        ]
        removal = events.loc[
            events["index_id"].astype(str).eq(index_id)
            & events["security_id"].astype(str).eq(LEGACY_SECURITY_ID)
            & events["operation"].astype(str).str.upper().eq("REMOVE")
            & events["effective_date"].astype(str).eq(boundary["remove_date"])
        ]
        if len(anchor) != 1 or len(removal) != 1:
            raise ValueError(
                f"Mallinckrodt {index_id} canonical anchor/removal is missing."
            )
        dates = (
            boundary["anchor_date"],
            (pd.Timestamp(boundary["remove_date"]) - pd.Timedelta(days=1))
            .date()
            .isoformat(),
            boundary["remove_date"],
            completed_session,
        )
        for value in dates:
            prior = before.members_on(index_id, value)
            current = after.members_on(index_id, value)
            if len(prior.security_ids) != len(current.security_ids):
                raise ValueError(
                    f"Mallinckrodt remap changed {index_id} cardinality on {value}."
                )
            members = set(current.security_ids)
            should_exist = pd.Timestamp(value) < pd.Timestamp(boundary["remove_date"])
            if (LEGACY_SECURITY_ID in members) != should_exist:
                raise ValueError(
                    f"Legacy Mallinckrodt membership boundary is wrong: {value}."
                )
            if REORGANIZED_SECURITY_ID in members:
                raise ValueError(
                    f"Reorganized Mallinckrodt leaked into historical {index_id}."
                )
            if any(
                LEGACY_SECURITY_ID in warning
                or REORGANIZED_SECURITY_ID in warning
                for warning in current.warnings
            ):
                raise ValueError(f"Mallinckrodt replay warning remains on {value}.")
            checked += 1
    return {
        "index_snapshots_checked": checked,
        "boundaries": EXPECTED_INDEX_BOUNDARIES,
    }


def validate_identity_and_history_gate(
    preflight: LocalPreflight,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    master = frames["security_master"]
    affected_master = master.loc[_affected(master)]
    if len(affected_master) != 2:
        raise ValueError("Mallinckrodt repair must leave exactly two common identities.")
    if affected_master["name"].astype(str).str.contains(
        "Muniholdings", case=False
    ).any():
        raise ValueError("The false Muniholdings identity survived repair.")
    expected_master = {
        LEGACY_SECURITY_ID: (
            "MNKKQ.US",
            LEGACY_START,
            LEGACY_LAST,
            "legacy ordinary shares",
        ),
        REORGANIZED_SECURITY_ID: (
            "MNKTQ.US",
            REORGANIZED_START,
            REORGANIZED_LAST,
            "reorganized ordinary shares",
        ),
    }
    for security_id, (provider, start, end, name_token) in expected_master.items():
        row = affected_master.loc[
            affected_master["security_id"].astype(str).eq(security_id)
        ]
        if len(row) != 1:
            raise ValueError(f"Mallinckrodt master row is not unique: {security_id}")
        value = row.iloc[0]
        if (
            str(value.get("primary_symbol", "")) != "MNK"
            or str(value.get("provider_symbol", "")) != provider
            or str(value.get("action_provider_symbol", "")) != provider
            or str(value.get("active_from", "")) != start
            or str(value.get("active_to", "")) != end
            or name_token not in str(value.get("name", "")).lower()
        ):
            raise ValueError(f"Mallinckrodt master boundary is wrong: {security_id}")

    history = frames["symbol_history"]
    affected_history = history.loc[_affected(history)].copy()
    if len(affected_history) != 2:
        raise ValueError("Mallinckrodt symbol history must have two disjoint rows.")
    intervals = {
        str(row.security_id): (
            str(row.symbol),
            str(row.effective_from),
            str(row.effective_to),
        )
        for row in affected_history.itertuples(index=False)
    }
    if intervals != {
        LEGACY_SECURITY_ID: ("MNK", LEGACY_START, LEGACY_LAST),
        REORGANIZED_SECURITY_ID: (
            "MNK",
            REORGANIZED_START,
            REORGANIZED_LAST,
        ),
    }:
        raise ValueError(f"Mallinckrodt symbol intervals are wrong: {intervals}")

    prices = frames["daily_price_raw"]
    factors = frames["adjustment_factors"]
    actions = frames["corporate_actions"]
    price_rows: dict[str, int] = {}
    for code, security_id in CODE_TO_SECURITY_ID.items():
        subset = prices.loc[prices["security_id"].astype(str).eq(security_id)].copy()
        expected = _target_expected_sessions(code)
        _sessions_exact(subset, expected, f"{code} repaired")
        _validate_ohlcv(subset, f"{code} repaired")
        price_rows[code] = len(subset)
        factor = factors.loc[factors["security_id"].astype(str).eq(security_id)]
        _sessions_exact(factor, expected, f"{code} factors")
        if set(factor["source_version"].astype(str)) == {""}:
            raise ValueError(f"{code} factor source_version is blank.")
        if "source_url" in subset:
            urls = set(subset["source_url"].astype(str))
            if urls != {_public_url(code, "eod")}:
                raise ValueError(f"{code} repaired prices have wrong source URLs: {urls}")
    if set(prices.loc[_affected(prices), "source_hash"].astype(str)) & set(
        preflight.contaminated_hashes
    ):
        raise ValueError("Contaminated MNK_old/MNK price hashes survived replacement.")
    affected_actions = actions.loc[_affected(actions)].copy()
    if not affected_actions.empty:
        successor = affected_actions["new_security_id"].fillna("").astype(str).str.strip()
        symbols = affected_actions["new_symbol"].fillna("").astype(str).str.strip()
        if successor.ne("").any() or symbols.ne("").any():
            raise ValueError("A Mallinckrodt successor link survived repair.")
        dates = pd.to_datetime(affected_actions["effective_date"], errors="coerce")
        ids = affected_actions["security_id"].astype(str)
        invalid = (
            ids.eq(LEGACY_SECURITY_ID)
            & ~dates.between(pd.Timestamp(LEGACY_START), pd.Timestamp(LEGACY_LAST))
        ) | (
            ids.eq(REORGANIZED_SECURITY_ID)
            & ~dates.between(
                pd.Timestamp(REORGANIZED_START), pd.Timestamp(REORGANIZED_LAST)
            )
        )
        if invalid.any():
            raise ValueError("Mallinckrodt provider actions cross legal boundaries.")
        if affected_actions["action_type"].astype(str).isin(
            {"delisting", "ticker_change", "stock_merger"}
        ).any():
            raise ValueError("Provider data invented a Mallinckrodt lifecycle action.")
    return {
        "legacy_security_id": LEGACY_SECURITY_ID,
        "reorganized_security_id": REORGANIZED_SECURITY_ID,
        "price_rows": price_rows,
        "false_muniholdings_rows_remaining": 0,
        "successor_links": 0,
    }


def validate_lifecycle_candidate_gate(
    frames: Mapping[str, pd.DataFrame],
    release: DataRelease,
) -> dict[str, Any]:
    """Bind both terminal histories without claiming both are index candidates.

    ``build_lifecycle_candidates`` intentionally considers only securities
    referenced by an index anchor or event.  The legacy identity is such a
    security, while the post-2022 reorganized shares are not.  Requiring both
    ids from that global candidate builder would therefore make this repair
    impossible on the audited snapshot.  The reorganized binding is instead
    proven directly from its unique master row and exact terminal price date.
    """
    candidates = build_lifecycle_candidates(_FrameRepository(frames), release=release)
    found = {
        candidate.security_id: candidate
        for candidate in candidates
        if candidate.security_id in {LEGACY_SECURITY_ID, REORGANIZED_SECURITY_ID}
    }
    expected = {
        LEGACY_SECURITY_ID: LEGACY_LAST,
        REORGANIZED_SECURITY_ID: REORGANIZED_LAST,
    }
    legacy_candidate = found.get(LEGACY_SECURITY_ID)
    if (
        legacy_candidate is None
        or legacy_candidate.symbol != "MNK"
        or legacy_candidate.last_price_date != LEGACY_LAST
    ):
        raise ValueError("Legacy Mallinckrodt must remain an exact indexed lifecycle candidate.")

    master = frames["security_master"]
    prices = frames["daily_price_raw"]
    binding_modes: dict[str, str] = {}
    for security_id, last in expected.items():
        master_row = master.loc[master["security_id"].astype(str).eq(security_id)]
        if len(master_row) != 1:
            raise ValueError(f"Mallinckrodt lifecycle master row is not unique: {security_id}")
        price_rows = prices.loc[prices["security_id"].astype(str).eq(security_id)]
        observed_last = (
            price_rows["session"].astype(str).max() if not price_rows.empty else ""
        )
        if (
            str(master_row.iloc[0].get("primary_symbol", "")) != "MNK"
            or str(master_row.iloc[0].get("active_to", "")) != last
            or observed_last != last
        ):
            raise ValueError(
                "Mallinckrodt terminal binding changed: "
                f"{security_id}/master={master_row.iloc[0].to_dict()}/last={observed_last}"
            )
        candidate = found.get(security_id)
        if candidate is not None and (
            candidate.symbol != "MNK" or candidate.last_price_date != last
        ):
            raise ValueError(
                f"Mallinckrodt indexed lifecycle binding changed: {security_id}/{candidate}"
            )
        binding_modes[security_id] = (
            "indexed_lifecycle_candidate"
            if candidate is not None
            else "exact_terminal_history"
        )

    for evidence_id, binding in OFFICIAL_BINDINGS.items():
        security_id = str(binding["security_id"])
        if (
            expected.get(security_id) != str(binding["candidate_last_price_date"])
            or str(binding["symbol"]) != "MNK"
            or float(binding["cash_amount"]) != 0.0
            or pd.Timestamp(binding["effective_date"])
            != pd.Timestamp(binding["candidate_last_price_date"]) + pd.Timedelta(days=1)
        ):
            raise ValueError(f"Official Mallinckrodt binding changed: {evidence_id}")
    return {
        evidence_id: {
            **binding,
            "binding_status_after_repair": "ready_for_hash_pin",
            "binding_proof": binding_modes[str(binding["security_id"])],
        }
        for evidence_id, binding in OFFICIAL_BINDINGS.items()
    }


def validate_archive_gate(
    frames: Mapping[str, pd.DataFrame],
    artifacts: Iterable[SourceArtifact],
) -> None:
    archive = frames["source_archive"]
    pairs = set(
        zip(archive["source_url"].astype(str), archive["source_hash"].astype(str))
    )
    values = tuple(artifacts)
    for artifact in values:
        if (artifact.source_url, artifact.source_hash) not in pairs:
            raise ValueError(
                "Mallinckrodt evidence is absent from source_archive: "
                f"{artifact.source_url}"
            )
    request_urls = {_public_url(code, endpoint) for code, endpoint in EODHD_REQUESTS}
    archived_request_urls = {
        artifact.source_url for artifact in values if artifact.source_url in request_urls
    }
    if archived_request_urls != request_urls:
        raise ValueError("Not every Mallinckrodt raw request is archived.")


def validate_candidate_frames(
    preflight: LocalPreflight,
    frames: dict[str, pd.DataFrame],
    artifacts: tuple[SourceArtifact, ...],
    *,
    release: DataRelease,
) -> dict[str, Any]:
    for dataset in WRITE_DATASETS:
        report = validate_dataset(
            dataset,
            frames[dataset],
            incomplete_action_policy="warn",
            completed_session=release.completed_session,
        )
        report.raise_for_errors()
    _validate_nonregression(preflight.existing, frames)
    identity = validate_identity_and_history_gate(preflight, frames)
    replay = validate_index_replay_gate(
        preflight.existing,
        frames,
        completed_session=release.completed_session,
    )
    lifecycle = validate_lifecycle_candidate_gate(frames, release)
    validate_archive_gate(frames, artifacts)
    return {
        "identity_and_history": identity,
        "index_replay": replay,
        "lifecycle_candidates": lifecycle,
        "archive": "passed",
    }


def prepare_repair(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
    preflight: LocalPreflight,
    *,
    bundle: ProviderBundle,
) -> PreparedRepair:
    if preflight.already_repaired:
        raise ValueError("An already repaired snapshot cannot be prepared again.")
    provider_validation = validate_provider_bundle(preflight, bundle)
    provider_prices, provider_actions = _trim_provider_frames(bundle)
    master, history = rewrite_security_identities(
        preflight.existing["security_master"],
        preflight.existing["symbol_history"],
        bundle=bundle,
    )
    source_version = "mallinckrodt-identity:" + sha256_bytes(
        _canonical_json_bytes(
            {
                "security_ids": [LEGACY_SECURITY_ID, REORGANIZED_SECURITY_ID],
                "target_ranges": TARGET_RANGES,
                "raw_hashes": sorted(
                    artifact.source_hash for artifact in bundle.artifacts
                ),
                "no_successor_link": True,
            }
        )
    )
    prices, actions, factors = rewrite_prices_actions_factors(
        preflight.existing["daily_price_raw"],
        preflight.existing["corporate_actions"],
        preflight.existing["adjustment_factors"],
        provider_prices=provider_prices,
        provider_actions=provider_actions,
        source_version=source_version,
    )
    legacy_artifact = _artifact_for_code(bundle, LEGACY_CODE)
    anchors, events, index_stats = rewrite_index_references(
        preflight.existing["index_constituent_anchors"],
        preflight.existing["index_membership_events"],
        legacy_artifact=legacy_artifact,
    )
    manifest = _repair_manifest_artifact(
        release=release,
        bundle=bundle,
        provider_validation=provider_validation,
        index_stats=index_stats,
    )
    request_artifacts = tuple(
        _request_archive_artifact(artifact) for artifact in bundle.artifacts
    )
    _validate_request_envelopes(bundle.artifacts, request_artifacts)
    archive_artifacts = (*request_artifacts, manifest)
    archive = append_source_archive(
        preflight.existing["source_archive"],
        archive_artifacts,
        completed_session=release.completed_session,
    )
    frames = {
        **preflight.existing,
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": archive,
    }
    gates = validate_candidate_frames(
        preflight,
        frames,
        tuple(archive_artifacts),
        release=release,
    )
    current, current_etag = repository.current_release()
    if (
        current is None
        or current.version != release.version
        or current_etag != release_etag
    ):
        raise RuntimeError("Current release changed during Mallinckrodt preparation.")
    warnings = tuple(release.warnings)
    summary = {
        "status": "validated_dry_run",
        "base_release_version": release.version,
        "completed_session": release.completed_session,
        "legacy_security_id": LEGACY_SECURITY_ID,
        "reorganized_security_id": REORGANIZED_SECURITY_ID,
        "provider_codes": [f"{LEGACY_CODE}.US", f"{REORGANIZED_CODE}.US"],
        "preferred_code_excluded": f"{EXCLUDED_PREFERRED_CODE}.US",
        "maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "eodhd_http_attempts_this_run": bundle.http_attempts,
        "provider_validation": provider_validation,
        "index_rewrite": index_stats,
        "no_successor_link": True,
        "independent_cross_validation": False,
        "cross_validation_scope": "same_provider_overlap_identity_only",
        "official_evidence_bindings": gates["lifecycle_candidates"],
        "gates": gates,
        "warnings": list(warnings),
        "write_datasets": list(WRITE_DATASETS),
    }
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        preflight=preflight,
        pointer_etags=preflight.pointer_etags,
        frames=frames,
        archive_artifacts=tuple(archive_artifacts),
        warnings=warnings,
        summary=summary,
    )


def _cache_inventory(root: Path) -> dict[str, bool]:
    source = EodhdMallinckrodtSource(
        root / "state/mallinckrodt-identity/eodhd",
        allow_http=False,
    )
    return {
        f"{code}.{endpoint}": source.path(code, endpoint).is_file()
        for code, endpoint in EODHD_REQUESTS
    }


def build_offline_plan(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, Any]:
    preflight = build_local_preflight(repository, release)
    inventory = _cache_inventory(repository.root)
    return {
        "status": "already_repaired" if preflight.already_repaired else "offline_plan",
        "release_version": release.version,
        "network_clients_constructed": 0,
        "http_attempts": 0,
        "maximum_eodhd_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "next_run_maximum_http_attempts": sum(not value for value in inventory.values()),
        "cache_inventory": inventory,
        "provider_codes": [f"{LEGACY_CODE}.US", f"{REORGANIZED_CODE}.US"],
        "preferred_code_excluded": f"{EXCLUDED_PREFERRED_CODE}.US",
        "no_successor_link": True,
        "official_evidence_bindings": OFFICIAL_BINDINGS,
        "would_write": False,
    }


def _persist_archive_payloads(
    repository: LocalDatasetRepository,
    artifacts: Iterable[SourceArtifact],
    completed_session: str,
) -> None:
    for artifact in artifacts:
        path = (
            repository.root
            / "archives"
            / completed_session
            / f"{artifact.source_hash}.{_artifact_extension(artifact)}.gz"
        )
        if path.is_file():
            try:
                existing = gzip.decompress(path.read_bytes())
            except Exception as exc:
                raise RuntimeError(
                    f"Unreadable Mallinckrodt archive payload: {path}"
                ) from exc
            if existing != artifact.content:
                raise RuntimeError(f"Conflicting Mallinckrodt archive payload: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, gzip.compress(artifact.content, mtime=0))
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"Mallinckrodt archive verification failed: {path}")


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
        pending = tuple(recovery.rglob("*.json")) if recovery.exists() else ()
        if pending:
            raise RuntimeError(
                "A market-store recovery marker blocks Mallinckrodt writes: "
                + ", ".join(str(item) for item in pending)
            )
        transactions = repository.root / "transactions"
        interrupted = []
        if transactions.exists():
            for item in transactions.rglob("*.json"):
                try:
                    status = str(json.loads(item.read_bytes()).get("status") or "")
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    interrupted.append(item)
        if interrupted:
            raise RuntimeError(
                "Interrupted market-data transaction requires recovery: "
                + ", ".join(str(item) for item in interrupted)
            )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    write_atomic(path, _canonical_json_bytes(dict(value)))


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            ours = observed.version == committed_release_version or all(
                observed.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not ours:
                raise RuntimeError(
                    f"Unexpected release during rollback: {observed.version}"
                )
            repository.objects.put(
                "releases/current.json",
                old_release_bytes,
                if_match=current.etag,
            )
        if repository.objects.get("releases/current.json").data != old_release_bytes:
            raise RuntimeError("Release rollback verification failed.")
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            if current.data != old_pointer_bytes[dataset]:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"Unexpected pointer during rollback: {pointer.version}"
                    )
                repository.objects.put(
                    key,
                    old_pointer_bytes[dataset],
                    if_match=current.etag,
                )
            if repository.objects.get(key).data != old_pointer_bytes[dataset]:
                raise RuntimeError(f"Pointer rollback verification failed: {dataset}")
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_release_unchanged(
    repository: LocalDatasetRepository,
    release: DataRelease,
    etag: str | None,
) -> None:
    current, current_etag = repository.current_release()
    if current is None or current.version != release.version or current_etag != etag:
        raise RuntimeError("Current release changed after Mallinckrodt preflight.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(
            repository,
            prepared.release,
            prepared.release_etag,
        )
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(
                    f"{dataset} pointer changed before Mallinckrodt apply."
                )
            old_pointers[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        session = prepared.release.completed_session.replace("-", "")
        planned = {
            dataset: f"mallinckrodt-identity-{session}-{transaction_id}-{dataset}"
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/mallinckrodt-identity-repair"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
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
        committed: DataRelease | None = None
        try:
            _persist_archive_payloads(
                repository,
                prepared.archive_artifacts,
                prepared.release.completed_session,
            )
            inject("after_archive")
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="warn",
                    metadata={
                        "operation": "repair_us_mallinckrodt_identity",
                        "legacy_security_id": LEGACY_SECURITY_ID,
                        "reorganized_security_id": REORGANIZED_SECURITY_ID,
                        "strict_mallinckrodt_gate": "passed",
                        "no_successor_link": True,
                    },
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version
                inject(f"after_write:{dataset}")

            written = {
                dataset: repository.read_frame(dataset, planned[dataset])
                for dataset in WRITE_DATASETS
            }
            validate_candidate_frames(
                prepared.preflight,
                written,
                prepared.archive_artifacts,
                release=prepared.release,
            )
            inject("before_release_commit")
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=(
                    DataQuality.DEGRADED
                    if prepared.warnings
                    else DataQuality.VALID
                ),
                warnings=prepared.warnings,
                expected_etag=prepared.release_etag,
            )
            inject("after_release_commit")
            current, _ = repository.current_release()
            if current is None or current.to_bytes() != committed.to_bytes():
                raise RuntimeError("Committed Mallinckrodt release is not current.")
            for dataset, version in committed.dataset_versions.items():
                pointer, _ = repository.current_pointer(dataset)
                if pointer is None or pointer.version != version:
                    raise RuntimeError(
                        f"Mallinckrodt committed pointer mismatch: {dataset}"
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
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=(committed.version if committed else ""),
            )
            journal.update(
                {
                    "status": (
                        "rollback_failed" if rollback_errors else "rolled_back"
                    ),
                    "original_error": f"{type(original).__name__}: {original}",
                    "rollback_errors": list(rollback_errors),
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            if rollback_errors:
                recovery = (
                    repository.root
                    / "recovery/mallinckrodt-identity-repair"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "Mallinckrodt rollback failed; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}"
                ) from original
            raise


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[
        [str | Path], LocalDatasetRepository
    ] = LocalDatasetRepository,
    source_factory: Callable[..., EodhdMallinckrodtSource] = (
        EodhdMallinckrodtSource
    ),
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current release is required for Mallinckrodt repair.")
    if args.offline_plan:
        return build_offline_plan(repository, release)
    preflight = build_local_preflight(repository, release)
    if preflight.already_repaired:
        return {
            "status": "already_repaired",
            "release_version": release.version,
            "http_attempts": 0,
            "official_evidence_bindings": OFFICIAL_BINDINGS,
        }
    source = source_factory(
        repository.root / "state/mallinckrodt-identity/eodhd",
        allow_http=bool(args.fetch_eodhd_mnk),
    )
    prepared = prepare_repair(
        repository,
        release,
        release_etag,
        preflight,
        bundle=source.fetch(),
    )
    return apply_repair(repository, prepared) if args.apply else prepared.summary


def main(argv: list[str] | None = None) -> int:
    result = run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
